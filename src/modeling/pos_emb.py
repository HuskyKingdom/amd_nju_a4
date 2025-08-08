import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import apply_rotary_pos_emb


class NTKAwareRoPE(nn.Module):
    """NTK-aware RoPE module
    This is a series variants of the RoPE modules based on NTK theory to enhance its extrapolation ability.
    """
    
    def __init__(self, 
        dim: int, 
        max_seq_len: int,
        base: int = 10000,
        ratio: int = 1,
        dynamic: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ) -> None:
        """Initialize NTK-aware RoPE Module
        
        Args:
            dim (int): The dimension of the RoPE
            max_seq_len (int): The maximum sequence length used in training
            base (int, optional): The base of the NTK. Defaults to 10000.
            ratio (int, optional): The ratio of the NTK. Defaults to 1.
            dynamic (bool, optional): Whether to use dynamic mode. Defaults to False.
            dtype (torch.dtype, optional): The dtype of the RoPE. Defaults to torch.float32.
            device (str, optional): The device of the RoPE. Defaults to 'cpu'.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.ratio = ratio
        self.dynamic = dynamic
        self.dtype = dtype
        self.device = device
        self._set_cos_sin_cache(self.max_seq_len*self.ratio)
        
        
    def _set_cos_sin_cache(self, seq_len):
        base = self.base * (self.ratio**(self.dim/(self.dim-2)))
        beta = (base ** (torch.arange(0,self.dim,2,device=self.device,dtype=self.dtype) / self.dim)).to(self.dtype)
        theta = 1.0 / beta
        n_theta = torch.outer(torch.arange(0,seq_len,device=self.device,dtype=self.dtype),theta)
        n_theta = torch.cat((n_theta,n_theta),dim=-1)
        cos = nn.Parameter(n_theta.cos().to(self.dtype).to(self.device),requires_grad=False)
        sin = nn.Parameter(n_theta.sin().to(self.dtype).to(self.device),requires_grad=False)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The forward pass of the NTK-aware RoPE module
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
            offset(int): offset of input tensor
            
        Returns:
            output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        """
        ratio_ = self.ratio
        es = self.max_seq_len * ratio_
        s_ = input.shape[1] + offset
        if s_ > es:
            k_ = get_min_even_k(s_,self.max_seq_len)
            ratio_ = k_
            if self.dynamic is True:
                self.ratio = k_
                self._set_cos_sin_cache(self.max_seq_len*self.ratio)
                return apply_rotary_pos_emb(input,self.cos_cached[offset:s_],self.sin_cached[offset:s_])
            base = self.base * (ratio_**(self.dim/(self.dim-2)))
            beta = (base ** (torch.arange(0,self.dim,2,device=self.device,dtype=self.dtype) / self.dim)).to(self.dtype)
            theta = 1.0 / beta
            n_theta = torch.outer(torch.arange(offset,s_,device=self.device,dtype=self.dtype),theta)
            n_theta = torch.cat((n_theta,n_theta),dim=-1)
            cos = n_theta.cos().to(self.dtype).to(self.device)
            sin = n_theta.sin().to(self.dtype).to(self.device)
            return apply_rotary_pos_emb(input,cos,sin)
        return apply_rotary_pos_emb(input,self.cos_cached[offset:s_],self.sin_cached[offset:s_])
    
def get_min_even_k(s, ms):
    k = (s + ms - 1) // ms
    if k % 2 == 0:
        return k
    else:
        return k + 1
