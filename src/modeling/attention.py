from typing import Optional, Tuple
from enum import Enum

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from assignment1 implementations
from .norm import GroupRMSNorm


class AttnQKVPackFormat(Enum):
    QKV = "qkv_packed"
    Q_KV = "q_kv_packed"
    Q_K_V = "q_k_v_packed"


class AttnQKVLayout(Enum):
    BSHD = "bshd"
    SBHD = "sbhd"
    THD = "thd"


class OfflineSlidingWindowAttn(nn.Module):
    """Offline Sliding-Window Attention module
    This is a generalized variant of standard self-attention equipped with the sliding-window trick \
        to make use of spatial locality in language for computational efficiency, \
        with applying other methods to improve stability.
    """
    def __init__(
        self,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_dropout_rate: float = 0.0,
        softmax_dropout_seed: int = 42,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_clip_range: Tuple[float, float] = (0., 1.),
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Offline Sliding-Window Attention module
        
        Args:
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            qkv_pack_format(AttnQKVPackFormat, default = "q_k_v_packed"): qkv packed format
            qkv_layout(AttnQKVLayout, default = "bshd"): qkv shape layout
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_dropout_rate(float, default = 0.0): dropout probability for the softmax probs
            softmax_dropout_seed(int, default = 42): random seed for softmax drooput
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            softmax_clip_range(float, default = (0.0, 1.0): the range for softmax clipping to prevent the outliers from growing further
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_q_head = num_q_head
        self.num_kv_head = num_kv_head
        self.qkv_pack_format = qkv_pack_format
        self.qkv_layout = qkv_layout
        self.window_size = window_size
        self.causal = causal
        self.softmax_dropout = nn.Dropout(softmax_dropout_rate)
        self.softmax_dropout_seed = softmax_dropout_seed
        self.softmax_scale = 1.0/math.sqrt(head_dim) if softmax_scale is None else softmax_scale
        self.softmax_cap = softmax_cap
        self.softmax_temp = softmax_temp
        self.softmax_clip_range = softmax_clip_range
        self.apply_qk_norm = apply_qk_norm
        if self.apply_qk_norm:
            group_size = head_dim if group_size is None else group_size
            self.q_norm = GroupRMSNorm(head_dim * num_q_head, group_size, eps, init_range, init_seed, dtype, device)
            self.kv_norm = GroupRMSNorm(head_dim * num_kv_head, group_size, eps, init_range, init_seed, dtype, device)
        
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, or query-key-value packed tensor if the qkv_pack_format is "qkv_packed"
            k(Optional[torch.Tensor], default = None): key tensor, or key-value packed tensor if the qkv_pack_format is "q_kv_packed", or None if qkv_pack_format is "qkv_packed"
            v(Optional[torch.Tensor], default = None): value tensor if the qkv_pack_format is "q_k_v_packed", otherwise None
            cu_seqlens_q(Optional[torch.Tensor], default = None): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(Optional[torch.Tensor], default = None): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
        Returns:
            torch.Tensor: output tensor o, with the same shape as q
        """
        q_dtype = q.dtype
        q_device = q.device
        # prepare q, k, v
        if self.qkv_pack_format.value == "q_k_v_packed":
            q_states, k_states, v_states = q, k, v
        elif self.qkv_pack_format.value == "q_kv_packed":
            q_states, k_states, v_states = q, k[...,:self.num_kv_head,:], k[...,self.num_kv_head:,:]
        else:
            q_states, k_states, v_states = q[...,:self.num_q_head,:], q[...,self.num_q_head:self.num_q_head+self.num_kv_head,:], q[...,self.num_q_head+self.num_kv_head:,:]
        # apply qk norm
        if self.apply_qk_norm:
            if self.qkv_layout.value != "thd":
                q_states = self.q_norm(q_states.view(q_states.shape[:2]+(self.head_dim * self.num_q_head,))).view(q_states.shape)
                k_states = self.kv_norm(k_states.view(k_states.shape[:2]+(self.head_dim * self.num_kv_head,))).view(k_states.shape)
            else:
                q_states = self.q_norm(q_states.view(1,q_states.shape[0],self.head_dim * self.num_q_head)).view(q_states.shape)
                k_states = self.kv_norm(k_states.view(1,k_states.shape[0],self.head_dim * self.num_kv_head)).view(k_states.shape)
        # repeat kv
        if self.num_q_head != self.num_kv_head:
            k_states = torch.repeat_interleave(k_states, self.num_q_head//self.num_kv_head, dim=-2)
            v_states = torch.repeat_interleave(v_states, self.num_q_head//self.num_kv_head, dim=-2)
                
        # attention module
        if self.qkv_layout.value != "thd":
            if self.qkv_layout.value == "sbhd":
                q_states, k_states, v_states = q_states.transpose(0,1), k_states.transpose(0,1), v_states.transpose(0,1)
            q_states, k_states, v_states = q_states.transpose(1,2), k_states.transpose(1,2), v_states.transpose(1,2)
            attn_weights = (q_states @ (k_states.transpose(-2,-1))) * self.softmax_scale
            if self.softmax_cap is None:
                attn_weights = attn_weights / self.softmax_temp
            else:
                attn_weights = self.softmax_cap * torch.tanh(attn_weights / self.softmax_cap)
            offset = k_states.shape[-2]-q_states.shape[-2]
            if self.window_size is not None:    
                sw_mask = torch.tril(torch.ones((q_states.shape[-2],k_states.shape[-2]),dtype=q_dtype,device=q_device),diagonal=-1-self.window_size+offset) + \
                    torch.triu(torch.ones((q_states.shape[-2],k_states.shape[-2]),dtype=q_dtype,device=q_device),diagonal=1+self.window_size+offset)
                sw_mask = sw_mask.masked_fill(sw_mask == 1, float('-inf'))
                attn_weights = attn_weights + sw_mask
            if self.causal is True:
                casual_mask = torch.triu(torch.ones((q_states.shape[-2],k_states.shape[-2]),dtype=q_dtype,device=q_device),diagonal=1+offset)
                casual_mask = casual_mask.masked_fill(casual_mask == 1, float('-inf'))
                attn_weights = attn_weights + casual_mask
            attn_weights = F.softmax(attn_weights,dim=-1).nan_to_num(0.0)
            attn_weights = torch.clip(((self.softmax_clip_range[1]-self.softmax_clip_range[0])*attn_weights + self.softmax_clip_range[0]),min=0.0,max=1.0)
            torch.manual_seed(self.softmax_dropout_seed)
            attn_weights = self.softmax_dropout(attn_weights)
            attn_weights = attn_weights @ v_states
            attn_weights = attn_weights.transpose(1,2)
            if self.qkv_layout.value == "sbhd":
                attn_weights = attn_weights.transpose(0,1)
        else:
            batch = cu_seqlens_q.shape[0]-1
            attn_weights = torch.empty((q_states.shape[0],q_states.shape[1],v_states.shape[2]),dtype=q_dtype,device=q_device)
            for i in range(batch):
                q_sub_states = q_states[cu_seqlens_q[i]:cu_seqlens_q[i+1],...]
                k_sub_states = k_states[cu_seqlens_k[i]:cu_seqlens_k[i+1],...]
                v_sub_states = v_states[cu_seqlens_k[i]:cu_seqlens_k[i+1],...]
                q_sub_states, k_sub_states, v_sub_states = q_sub_states.transpose(0,1), k_sub_states.transpose(0,1), v_sub_states.transpose(0,1)
                attn_sub_weights = (q_sub_states @ (k_sub_states.transpose(-2,-1))) * self.softmax_scale
                if self.softmax_cap is None:
                    attn_sub_weights = attn_sub_weights / self.softmax_temp
                else:
                    attn_sub_weights = self.softmax_cap * torch.tanh(attn_sub_weights / self.softmax_cap)
                offset = k_sub_states.shape[-2]-q_sub_states.shape[-2]
                if self.window_size is not None:    
                    sw_mask = torch.tril(torch.ones((q_sub_states.shape[-2],k_sub_states.shape[-2]),dtype=q_dtype,device=q_device),diagonal=-1-self.window_size+offset) + \
                        torch.triu(torch.ones((q_sub_states.shape[-2],k_sub_states.shape[-2]),dtype=q_dtype,device=q_device),diagonal=1+self.window_size+offset)
                    sw_mask = sw_mask.masked_fill(sw_mask == 1, float('-inf'))
                    attn_sub_weights = attn_sub_weights + sw_mask
                if self.causal is True:
                    casual_mask = torch.triu(torch.ones((q_sub_states.shape[-2],k_sub_states.shape[-2]),dtype=q_dtype,device=q_device),diagonal=1+offset)
                    casual_mask = casual_mask.masked_fill(casual_mask == 1, float('-inf'))
                    attn_sub_weights = attn_sub_weights + casual_mask
                attn_sub_weights = F.softmax(attn_sub_weights,dim=-1,dtype=q_dtype).nan_to_num(0.0)
                attn_sub_weights = torch.clip(((self.softmax_clip_range[1]-self.softmax_clip_range[0])*attn_sub_weights + self.softmax_clip_range[0]),min=0.0,max=1.0)
                torch.manual_seed(self.softmax_dropout_seed)
                attn_sub_weights = self.softmax_dropout(attn_sub_weights)
                attn_sub_weights = attn_sub_weights @ v_sub_states
                attn_sub_weights = attn_sub_weights.transpose(0,1)
                attn_weights[cu_seqlens_q[i]:cu_seqlens_q[i+1],...] = attn_sub_weights
        return attn_weights.to(q_dtype).to(q.device)
        
        

    
class OnlineSlidingWindowAttn(OfflineSlidingWindowAttn):
    """Online Sliding-Window Attention module
    This is a online version of Offline Sliding-Window Attention module \
        which only apply attention on a block of q, k, v in "bshd" layout and "q_k_v_packed" format \
            and update the global o with the local block of o using lse
    """
    def __init__(
        self,
        seqlen_q: int,
        seqlen_kv: int,
        block_size_q: int,
        block_size_kv: int,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Online Sliding-Window Attention module
        
        Args:
            seqlen_q(int): the sequence length of q
            seqlen_kv(int): the sequence length of kv
            block_size_q(int): the block size of q
            block_size_kv(int): the block size of kv
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__(
            head_dim=head_dim,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            window_size=window_size,
            causal=causal,
            softmax_scale=softmax_scale,
            softmax_cap=softmax_cap,
            softmax_temp=softmax_temp,
            apply_qk_norm=apply_qk_norm,
            group_size=group_size,
            eps=eps,
            init_range=init_range,
            init_seed=init_seed,
            dtype=dtype,
            device=device,
        )
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        offset = seqlen_kv-seqlen_q
        if self.window_size is not None:    
            sw_mask = torch.tril(torch.ones((seqlen_q,seqlen_kv)),diagonal=-1-self.window_size+offset) + \
                torch.triu(torch.ones((seqlen_q,seqlen_kv)),diagonal=1+self.window_size+offset)
            self.sw_mask = sw_mask.masked_fill(sw_mask == 1, float('-inf'))
        if self.causal is True:
            casual_mask = torch.triu(torch.ones((seqlen_q,seqlen_kv),dtype=torch.float32),diagonal=1+offset)
            self.casual_mask = casual_mask.masked_fill(casual_mask == 1, float('-inf'))
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_o: torch.Tensor,
        global_lse: torch.Tensor,
        block_idx_q: int,
        block_idx_kv: int,
    ) -> None:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, block_size_q, num_q_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            global_o(torch.Tensor): global output tensor to be updated inplace, with shape: [batch_size, seqlen_q, num_q_head, head_dim]
            global_lse(torch.Tensor): global lse tensor to be updated inplace, with shape: [batch_size, num_q_head, seqlen_q]
            block_idx_q(int): the block index of q
            block_idx_kv(int): the block index of kv
        """
        q_dtype = q.dtype
        q_device = q.device
        q_states, k_states, v_states = q, k, v
        # apply qk norm
        if self.apply_qk_norm:
            q_states = self.q_norm(q_states.view(q_states.shape[:2]+(self.head_dim * self.num_q_head,))).view(q_states.shape)
            k_states = self.kv_norm(k_states.view(k_states.shape[:2]+(self.head_dim * self.num_kv_head,))).view(k_states.shape)
        if self.num_q_head != self.num_kv_head:
            k_states = torch.repeat_interleave(k_states, self.num_q_head//self.num_kv_head, dim=-2)
            v_states = torch.repeat_interleave(v_states, self.num_q_head//self.num_kv_head, dim=-2)
        q_states, k_states, v_states = q_states.transpose(1,2), k_states.transpose(1,2), v_states.transpose(1,2)
        global_o = global_o.transpose(1,2)
        if (block_idx_q+1) * self.block_size_q >= self.seqlen_q:
            q_states = q_states[...,:self.seqlen_q-block_idx_q * self.block_size_q,:]
        if (block_idx_kv+1) * self.block_size_kv >= self.seqlen_kv:
            k_states = k_states[...,:self.seqlen_kv-block_idx_kv * self.block_size_kv,:]
            v_states = v_states[...,:self.seqlen_kv-block_idx_kv * self.block_size_kv,:]
        attn_weights = (q_states @ (k_states.transpose(-2,-1))) * self.softmax_scale
        if self.softmax_cap is None:
            attn_weights = attn_weights / self.softmax_temp
        else:
            attn_weights = self.softmax_cap * torch.tanh(attn_weights / self.softmax_cap)
        if self.window_size is not None:    
            attn_weights = attn_weights + self.sw_mask[block_idx_q * self.block_size_q:(block_idx_q+1) * self.block_size_q, block_idx_kv * self.block_size_kv:(block_idx_kv+1) * self.block_size_kv].to(q_dtype).to(q_device)
        if self.causal is True:
            attn_weights = attn_weights + self.casual_mask[block_idx_q * self.block_size_q:(block_idx_q+1) * self.block_size_q, block_idx_kv * self.block_size_kv:(block_idx_kv+1) * self.block_size_kv].to(q_dtype).to(q_device)
        
        lse_2 = torch.logsumexp(attn_weights, dim=-1)
        lse_1 = global_lse[..., block_idx_q * self.block_size_q : (block_idx_q + 1) * self.block_size_q].clone().detach()
        lse_max = torch.maximum(lse_1,lse_2)
        lse_min = torch.minimum(lse_1,lse_2)
        # lse = torch.log(torch.exp(lse_1).nan_to_num(0)+torch.exp(lse_2).nan_to_num(0))
        # lse = lse_max + torch.log1p(torch.exp(lse_min-lse_max)).nan_to_num(0)
        lse = lse_max + F.softplus(lse_min-lse_max).nan_to_num(0)
        global_lse[..., block_idx_q * self.block_size_q : (block_idx_q + 1) * self.block_size_q] = lse
        attn_weights = F.softmax(attn_weights,dim=-1).nan_to_num(0.0)
        attn_weights = attn_weights @ v_states
        global_o[..., block_idx_q * self.block_size_q : (block_idx_q + 1) * self.block_size_q,:] = \
            torch.exp(lse_1-lse).nan_to_num(0).unsqueeze(-1) * global_o[..., block_idx_q * self.block_size_q : (block_idx_q + 1) * self.block_size_q,:] + \
            torch.exp(lse_2-lse).nan_to_num(0).unsqueeze(-1) * attn_weights
        global_o = global_o.transpose(1,2)
        
