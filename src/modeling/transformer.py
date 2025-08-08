from typing import Optional, Tuple, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

# from assignment1 implementations
from .vocab_emb import ParallelVocabEmbedding
from .pos_emb import NTKAwareRoPE
from .norm import GroupRMSNorm

# from assignment2 implementations
from .mlp import (
    MLPActivationType,
    DenseMLPWithLoRA,
    SparseMLPWithLoRA,
)

# from assignment3 implementations
from .attention import (
    AttnQKVPackFormat,
    AttnQKVLayout,
    OfflineSlidingWindowAttn,
    OnlineSlidingWindowAttn,
)

from .config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)


@config_dataclass
class TransformerConfig(BaseConfig):
    """Transformer Configurations Dataclass"""
    
    # common transformer configurations
    num_layers: int = make_required_field()
    hidden_size: int = make_required_field()
    ffh_size: int = make_required_field()
    max_seq_len: int = make_required_field()
    param_dtype: torch.dtype = torch.float32
    param_device: str = "cpu"
    init_base_seed: int = 42
    
    # fixed distributed configurations
    rank: int = make_fixed_field(0)
    world_size: int = make_fixed_field(1)
    process_group: Optional[ProcessGroup] = make_fixed_field(None)
    
    # vocab embedding configurations
    vocab_size: int = make_required_field()
    vocab_init_mean: float = 0.0
    vocab_init_std: float = 1.0
    
    # positional embedding configurations
    rope_base: int = 10000
    rope_ratio: int = 1
    rope_dynamic: bool = False
    
    # normalization configurations
    group_size: Optional[int] = None
    eps: float = 1e-5
    norm_init_range: tuple = (-1.0, 1.0)
    
    # projection configurations
    proj_init_seed: int = 42
    proj_init_mean: float = 0.0
    proj_init_std: float = 1.0
    lm_head_tied: bool = False
    
    # attention configurations
    online_attn_block_size: Optional[int] = None # NOTE: if None, then use offline mode, otherwise use online mode
    head_dim: int = make_required_field()
    num_q_head: int = make_required_field()
    num_kv_head: int = make_required_field()
    qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V
    qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD
    window_size: Optional[int] = None
    causal: bool = False
    softmax_dropout_rate: float = 0.0
    softmax_dropout_seed: int = 42
    softmax_scale: Optional[float] = None
    softmax_cap: Optional[float] = None
    softmax_temp: float = 1.0
    softmax_clip_range: Tuple[float, float] = (0., 1.)
    apply_qk_norm: bool = False
    qk_norm_group_size: Optional[int] = None # NOTE: the other configurations of qk norm are the same as the ones of normalization above
    
    # dense mlp configurations
    activation_type: MLPActivationType = MLPActivationType.SILU
    lora_rank: int = 0
    lora_alpha: Optional[float] = None
    lora_dropout_rate: float = 0.0
    lora_dropout_seed: int = 42
    lora_init_base_seed: int = 42
    
    # sparse mlp configurations (optional)
    num_experts: Optional[int] = None # NOTE: if None, then use dense mlp, otherwise use sparse mlp
    moe_topk: int = 1
    gate_init_mean: float = 0.0
    gate_init_std: float = 1.0


class TransformerDecoderKVCache(nn.Module):
    """Transformer KV cache module
    This is a simple module to manage cached past key-value pairs for each transformer decoder layer \
        tradeoff memory footprint for avoiding redundant computation during inference.
    """
    def __init__(
        self,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        num_layers: int = 1,
    ):
        """Initialize Transformer KV cache module
        
        Args:
            qkv_layout (AttnQKVLayout, optional): Layout of the q, k, v tensors. Defaults to AttnQKVLayout.BSHD.
            num_layers (int, optional): Number of transformer layers. Defaults to 1.
        """
        super().__init__()
        self.qkv_layout = qkv_layout
        self.num_layers = num_layers
        self.k_cache = [[]] * num_layers
        self.v_cache = [[]] * num_layers
        if qkv_layout == AttnQKVLayout.THD:
            self.cu_seqlens = [[]] * num_layers

    def has(self, layer_idx: int) -> bool:
        """Check if cached past key-value pairs exist for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            bool: True if cached past key-value pairs exist for the layer, False otherwise
        """
        if len(self.k_cache[layer_idx]) == 0 or len(self.v_cache[layer_idx]) == 0:
            return False
        return True

    def get(
        self, 
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: (k, v, optional cu_seqlens)
            
        Raises:
            KeyError: If cached past key-value pairs do not exist for the layer
        """
        if self.has(layer_idx) is False:
            raise ValueError(f"{layer_idx} layer doesn't exist in kv_cache.")
        if self.qkv_layout == AttnQKVLayout.THD:
            return (self.k_cache[layer_idx].clone(),self.v_cache[layer_idx].clone(),self.cu_seqlens[layer_idx].clone())
        else:
            return (self.k_cache[layer_idx].clone(),self.v_cache[layer_idx].clone(),None)

    def set(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Set cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to set
            v (torch.Tensor): Value tensor to set
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to set. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD
        """
        self.k_cache[layer_idx] = k.clone()
        self.v_cache[layer_idx] = v.clone()
        if self.qkv_layout == AttnQKVLayout.THD:
            self.cu_seqlens[layer_idx] = cu_seqlens
            

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Dynamically append current cached past key-value pairs with their optional cumulative sequence lengths to the existing ones for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to append
            v (torch.Tensor): Value tensor to append
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to append. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD, \
                and all of the pass-in arguments should be consistent with the existing ones.
        """
        if self.has(layer_idx) is False:
            self.set(layer_idx,k,v,cu_seqlens)
        else:
            if self.qkv_layout == AttnQKVLayout.BSHD:
                self.k_cache[layer_idx] = torch.cat((self.k_cache[layer_idx],k),dim=1)
                self.v_cache[layer_idx] = torch.cat((self.v_cache[layer_idx],v),dim=1)
            elif self.qkv_layout == AttnQKVLayout.SBHD:
                self.k_cache[layer_idx] = torch.cat((self.k_cache[layer_idx],k),dim=0)
                self.v_cache[layer_idx] = torch.cat((self.v_cache[layer_idx],v),dim=0)
            else:
                _, h, d = self.k_cache[layer_idx].shape
                k_states = torch.empty((0,h,d),dtype=k.dtype,device=k.device)
                v_states = torch.empty((0,h,d),dtype=v.dtype,device=v.device)
                new_cu_seqlens = torch.zeros_like(cu_seqlens)
                for i in range(len(cu_seqlens)-1):
                    k_states = torch.cat((k_states,self.k_cache[layer_idx][self.cu_seqlens[layer_idx][i]:self.cu_seqlens[layer_idx][i+1]],k[cu_seqlens[i]:cu_seqlens[i+1]]),dim=0)
                    v_states = torch.cat((v_states,self.v_cache[layer_idx][self.cu_seqlens[layer_idx][i]:self.cu_seqlens[layer_idx][i+1]],v[cu_seqlens[i]:cu_seqlens[i+1]]),dim=0)
                    new_cu_seqlens[i+1] = self.cu_seqlens[layer_idx][i+1] + cu_seqlens[i+1]
                self.k_cache[layer_idx] = k_states
                self.v_cache[layer_idx] = v_states
                self.cu_seqlens[layer_idx] = new_cu_seqlens
    
    def reset(self):
        """Clear the cache memory and reset to the initial state
        """
        self.k_cache = [[]] * self.num_layers
        self.v_cache = [[]] * self.num_layers
        if self.qkv_layout == AttnQKVLayout.THD:
            self.cu_seqlens = [[]] * self.num_layers
    

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer module
    This is a variant of transformer decoder layer, consisting of two sub-layers: \
            one offline / online self-attention layer, along with qkv projection, ntk-aware rope and out projection, \
            and one dense / sparse feed-forward mlp layer, supporting LoRA adaption intrinsically, \
        which are concatenated sequentially with residual connections and group rms normalization.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        """Initialize Transformer Decoder Layer module
        
        Args:
            config (TransformerConfig): transformer configuration
            layer_idx (int): layer index, in the range of [0, num_layers). Defaults to 0.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed+layer_idx+1,
            dtype=config.param_dtype,
            device=config.param_device
        )
        self.rope = NTKAwareRoPE(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            ratio=config.rope_ratio,
            dynamic=config.rope_dynamic,
            dtype=config.param_dtype,
            device=config.param_device
        )
        if self.config.online_attn_block_size is None:
            self.attn = OfflineSlidingWindowAttn(
                head_dim=config.head_dim,
                num_q_head=config.num_q_head,
                num_kv_head=config.num_kv_head,
                qkv_pack_format=config.qkv_pack_format,
                qkv_layout=config.qkv_layout,
                window_size=config.window_size,
                causal=config.causal,
                softmax_dropout_rate=config.softmax_dropout_rate,
                softmax_dropout_seed=config.softmax_dropout_seed+layer_idx,
                softmax_scale=config.softmax_scale,
                softmax_cap=config.softmax_cap,
                softmax_temp=config.softmax_temp,
                softmax_clip_range=config.softmax_clip_range,
                apply_qk_norm=config.apply_qk_norm,
                group_size=config.qk_norm_group_size,
                eps=config.eps,
                init_range=config.norm_init_range,
                init_seed=config.init_base_seed+layer_idx+2,
                dtype=config.param_dtype,
                device=config.param_device
            )
        else:
            self.attn = OnlineSlidingWindowAttn(
                seqlen_q=config.max_seq_len,
                seqlen_kv=config.max_seq_len,
                block_size_q=config.online_attn_block_size,
                block_size_kv=config.online_attn_block_size,
                head_dim=config.head_dim,
                num_q_head=config.num_q_head,
                num_kv_head=config.num_kv_head,
                window_size=config.window_size,
                causal=config.causal,
                softmax_scale=config.softmax_scale,
                softmax_cap=config.softmax_cap,
                softmax_temp=config.softmax_temp,
                apply_qk_norm=config.apply_qk_norm,
                group_size=config.qk_norm_group_size,
                eps=config.eps,
                init_range=config.norm_init_range,
                init_seed=config.init_base_seed+layer_idx+2,
                dtype=config.param_dtype,
                device=config.param_device
            )
        
        self.mlp_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed+layer_idx+3,
            dtype=config.param_dtype,
            device=config.param_device
        )
        if self.config.num_experts is None:
            self.mlp = DenseMLPWithLoRA(
                hidden_size=config.hidden_size,
                ffh_size=config.ffh_size,
                activation_type=config.activation_type,
                init_base_seed=config.init_base_seed+layer_idx+4,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout_rate=config.lora_dropout_rate,
                lora_dropout_seed=config.lora_dropout_seed+layer_idx,
                lora_init_base_seed=config.lora_init_base_seed+layer_idx,
                dtype=config.param_dtype,
                device=config.param_device
            )
        else:
            self.mlp = SparseMLPWithLoRA(
                hidden_size=config.hidden_size,
                ffh_size=config.ffh_size,
                activation_type=config.activation_type,
                num_experts=config.num_experts,
                moe_topk=config.moe_topk,
                rank=config.rank,
                world_size=config.world_size,
                process_group=config.process_group,
                init_mean=config.gate_init_mean,
                init_std=config.gate_init_std,
                init_base_seed=config.init_base_seed+layer_idx+4,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout_rate=config.lora_dropout_rate,
                lora_dropout_seed=config.lora_dropout_seed+layer_idx,
                lora_init_base_seed=config.lora_init_base_seed+layer_idx,
                dtype=config.param_dtype,
                device=config.param_device
            )
        self.qkv_proj = nn.Parameter(torch.empty((config.hidden_size,config.head_dim*config.num_q_head+2*config.head_dim*config.num_kv_head),
                                    dtype=config.param_dtype,device=config.param_device))
        self.o_proj = nn.Parameter(torch.empty((config.head_dim*config.num_q_head,config.hidden_size),dtype=config.param_dtype,device=config.param_device))
        self.reset_parameters()
    def forward(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Layer module
        
        Args:
            input(torch.Tensor): input hidden states tensor, with shape: [batch_size, seq_len, hidden_size]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input` is ensured to be `1` to remain the 3-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as input
        """
        b, s, h = input.shape
        input_dtype, input_device = input.dtype, input.device
        X = input.clone().to(dtype=self.config.param_dtype,device=self.config.param_device)
        if self.config.qkv_layout.value == "thd":
            cu_seqlens_q = cu_seqlens.clone()
            cu_seqlens_kv = cu_seqlens.clone()
        else:
            cu_seqlens_q = None
            cu_seqlens_kv = None
        R = X
        X = self.attn_norm(X)
        X = X@self.qkv_proj
        q = X[:,:,:self.config.head_dim*self.config.num_q_head].reshape(b,s,-1,self.config.head_dim)
        k = X[:,:,self.config.head_dim*self.config.num_q_head:self.config.head_dim*self.config.num_q_head+self.config.head_dim*self.config.num_kv_head].reshape(b,s,-1,self.config.head_dim)
        v = X[:,:,self.config.head_dim*self.config.num_q_head+self.config.head_dim*self.config.num_kv_head:].reshape(b,s,-1,self.config.head_dim)
        if self.config.qkv_layout.value == "sbhd":
            q, k, v = q.transpose(0,1), k.transpose(0,1), v.transpose(0,1)
        elif self.config.qkv_layout.value == "thd":
            q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
        # rope module
        offset = 0
        if self.config.qkv_layout.value == "bshd":
            if kv_cache is not None and kv_cache.has(self.layer_idx) is True:
                offset = kv_cache.get(self.layer_idx)[0].shape[1]
            q, k = self.rope(q,offset), self.rope(k,offset)
        elif self.config.qkv_layout.value == "sbhd":
            if kv_cache is not None and kv_cache.has(self.layer_idx) is True:
                offset = kv_cache.get(self.layer_idx)[0].shape[0]
            q, k = self.rope(q.transpose(0,1),offset).transpose(0,1), self.rope(k.transpose(0,1),offset).transpose(0,1)
        else:
            for i in range(len(cu_seqlens_q)-1):
                if kv_cache is not None and kv_cache.has(self.layer_idx) is True:
                    offset = kv_cache.get(self.layer_idx)[2][i+1] - kv_cache.get(self.layer_idx)[2][i]
                q[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = self.rope(q[cu_seqlens_q[i]:cu_seqlens_q[i+1]].unsqueeze(0),offset).squeeze(0)
                k[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]] = self.rope(k[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]].unsqueeze(0),offset).squeeze(0)
        # kv_cache module
        if kv_cache is not None:
            kv_cache.append(self.layer_idx,k,v,cu_seqlens_kv)
            k, v, cu_seqlens_kv = kv_cache.get(self.layer_idx)
        # attention module
        if self.config.qkv_pack_format == AttnQKVPackFormat.QKV:
            O = self.attn(torch.cat((q,k,v),dim=-2),cu_seqlens_q=cu_seqlens_q,cu_seqlens_k=cu_seqlens_kv)
        elif self.config.qkv_pack_format == AttnQKVPackFormat.Q_KV:
            O = self.attn(q,torch.cat((k,v),dim=-2),cu_seqlens_q=cu_seqlens_q,cu_seqlens_k=cu_seqlens_kv)
        else:
            if isinstance(self.attn,OnlineSlidingWindowAttn):
                O = torch.zeros(b, q.shape[1], q.shape[2], q.shape[3], device=self.config.param_device,dtype=self.config.param_dtype)
                global_lse = torch.full((b, q.shape[2], q.shape[1]),float("-inf"),device=self.config.param_device,dtype=self.config.param_dtype)
                block_q_num = (q.shape[1]+self.attn.block_size_q-1) // self.attn.block_size_q
                block_kv_num = (k.shape[1]+self.attn.block_size_kv-1) // self.attn.block_size_kv
                for i in range(block_q_num):
                    for j in range(block_kv_num):
                        self.attn(q[:,i*self.attn.block_size_q:(i+1)*self.attn.block_size_q],
                                  k[:,j*self.attn.block_size_kv:(j+1)*self.attn.block_size_kv],
                                  v[:,j*self.attn.block_size_kv:(j+1)*self.attn.block_size_kv],O,global_lse,i,j)
            else:
                O = self.attn(q,k,v,cu_seqlens_q=cu_seqlens_q,cu_seqlens_k=cu_seqlens_kv)
        if self.config.qkv_layout.value == "bshd":
            O = O.reshape(b,s,-1)
        elif self.config.qkv_layout.value == "sbhd":
            O = O.reshape(s,b,-1).transpose(0,1)
        else:
            O = O.unsqueeze(0).reshape(b,s,-1)
        X = O@self.o_proj + R
        R = X
        X = self.mlp_norm(X)
        O = self.mlp(X)
        O = O + R
        return O.to(dtype=input_dtype,device=input_device)
        
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Layer module"""
        torch.manual_seed(self.config.proj_init_seed+self.layer_idx+1)
        self.qkv_proj = torch.nn.init.normal_(self.qkv_proj, mean=self.config.proj_init_mean, std=self.config.proj_init_std)
        torch.manual_seed(self.config.proj_init_seed+self.layer_idx+2)
        self.o_proj = torch.nn.init.normal_(self.o_proj, mean=self.config.proj_init_mean, std=self.config.proj_init_std)
        
class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block module
    
    This is a standard decoder-only transformer block for language modeling, \
        which mainly consists of a sequence of transformer decoder layers, \
        transforming the hidden states of input token ids initialized from vocab embedding, \
        and finally returning the vocab logits with a lm head projection.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
    ):
        """Initialize Transformer Decoder Block module
        
        Args:
            config (TransformerConfig): transformer configuration
        """
        # raise ValueError(f"{config.online_attn_block_size}, {config.qkv_layout}, {config.qkv_pack_format}")
        # None, SBHD, QKV
        super().__init__()
        self.config = config
        self.vocab_emb = ParallelVocabEmbedding(
            vocab_size=config.vocab_size,
            emb_size=config.hidden_size,
            rank=config.rank,
            world_size=config.world_size,
            process_group=config.process_group,
            init_mean=config.vocab_init_mean,
            init_std=config.vocab_init_std,
            init_base_seed=config.init_base_seed,
            dtype=config.param_dtype,
            device=config.param_device
        )
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config=config,layer_idx=i) for i in range(config.num_layers)
        ])
        
        self.final_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed,
            dtype=config.param_dtype,
            device=config.param_device
        )
        
        self.lm_head = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            device=config.param_device,
            dtype=config.param_dtype
        )
        self.reset_parameters()
        self.kv_cache = TransformerDecoderKVCache(config.qkv_layout,config.num_layers)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Block module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths, with shape: [inner_batch_size + 1, ]
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output tensor as vocab logits, with shape: [batch_size, seq_len, vocab_size]
        """
        # raise ValueError(f"{self.config.lm_head_tied}, {self.config.num_experts}, {self.config.max_seq_len}, {self.config.rope_ratio}, {input_ids.shape}")
        # True, 2, 8192, 1, torch.Size([4, 2048])
        # raise ValueError(f"{self.config.param_dtype},{self.config.moe_topk},{self.config.num_layers},{self.config.hidden_size},{self.config.ffh_size},{self.config.vocab_size},{self.config.group_size},{self.config.head_dim},{self.config.num_kv_head},{self.config.num_q_head},{self.config.causal},{self.config.softmax_scale},{self.config.softmax_cap},{self.config.softmax_temp},{self.config.apply_qk_norm},{self.config.qk_norm_group_size},{self.config.lora_rank},{self.config.activation_type}")
        input_dtype, input_device = input_ids.dtype, input_ids.device
        X = input_ids.clone().to(self.config.param_device)
        X = self.vocab_emb(X)
        if self.training is True:
            kv_cache = None
        else:
            kv_cache = self.kv_cache
        for decoder_layer in self.decoder_layers:
            X = decoder_layer(X,cu_seqlens=cu_seqlens,kv_cache=kv_cache)
        X = self.final_norm(X)
        logits = self.lm_head(X)
        return logits.to(input_device)
            
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""
        return self.kv_cache
    
    def set_kv_cache(self, kv_cache: TransformerDecoderKVCache):
        """Set the TransformerDecoderKVCache object managing the kv cache memory"""
        self.kv_cache = kv_cache
    
    def reset_kv_cache(self):
        """Clear the cache memory and reset to the initial state"""
        self.kv_cache.reset()
       
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Block module"""
        if self.config.lm_head_tied:
            self.lm_head.weight = self.vocab_emb.Tr
        else:
            torch.manual_seed(self.config.proj_init_seed)
            self.lm_head.weight.data.normal_(mean=self.config.proj_init_mean, std=self.config.proj_init_std)
     
    def num_parameters(
        self,
        learnable_only: bool = False, 
        unit: Literal["1", "K", "M", "B"] = "1"
    ) -> float:
        """Compute the number of (learnable) parameters in the Llama Model module
        
        Args:
            learnable_only(bool, optional): whether to count only learnable parameters or not, default to False
            unit(str, optional): unit of the number of parameters, default to '1' for "1", \
                other options include 'K' for "1 thousand", 'M' for "1 million", 'B' for "1 billion"
        Returns:
            float: the number of (learnable) parameters in the Llama Model module in the specified unit
        """
        if unit == "1":
            u = 1
        elif unit == "K":
            u = 1 / pow(10,3)
        elif unit == "M":
            u = 1 / pow(10,6)
        elif unit == "B":
            u = 1 / pow(10,9)
        if learnable_only is True:
            return float(sum(p.numel() for p in self.parameters() if p.requires_grad) * u)
        else:
            return float(sum(p.numel() for p in self.parameters()) * u)
        
    def num_memory_footprint(
        self,
        unit: Literal["B", "KB", "MB", "GB"] = "B"
    ) -> float:
        """Compute the theoretical memory footprint of the Llama Model module's parameters
        
        Args:
            unit(str, optional): unit of the memory footprint, default to 'B' for "1 byte", \
                other options include 'KB' for "1 kilobyte", 'MB' for "1 megabyte", 'GB' for "1 gigabyte"
                
        Returns:
            float: the theoretical memory footprint of the Llama Model module's parameters in the specified unit
        """
        if self.config.param_dtype == torch.float32:
            u = 4
        elif self.config.param_dtype == torch.float16 or self.config.param_dtype == torch.bfloat16:
            u = 2
        if unit == "B":      
            return self.num_parameters() * u
        elif unit == "KB":
            return self.num_parameters() * u / pow(1024,1)
        elif unit == "MB":
            return self.num_parameters() * u / pow(1024,2)
        elif unit == "GB":
            return self.num_parameters() * u / pow(1024,3)
