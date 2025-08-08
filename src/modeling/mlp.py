from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"

activation_dict = {
    MLPActivationType.RELU : F.relu,
    MLPActivationType.GELU : F.gelu,
    MLPActivationType.SILU : F.silu,
    MLPActivationType.SIGMOID : F.sigmoid,
    MLPActivationType.BILINEAR : lambda x:x,
}

class DenseMLPWithLoRA(nn.Module):
    """Dense MLP module with LoRA adapters
    This is a GLU-style dense MLP layer with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Dense MLP module with LoRA adapters
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = "silu"): activation type
            init_base_seed(int, default = 42): seed for base weight initialization
            lora_rank(int, default = 0): lora rank, if 0, then no lora to apply
            lora_alpha(Optional[float], default = None): lora alpha, if None, then set to lora_rank
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        self.h = hidden_size
        self.ffh = ffh_size
        self.activate = activation_type
        self.init_base_seed = init_base_seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_rank if lora_alpha is None else lora_alpha
        self.lora_init_base_seed = lora_init_base_seed
        self.lora_dropout_seed = lora_dropout_seed
        self.dtype = dtype
        self.device = device
        self.dropout = nn.Dropout(lora_dropout_rate)
        self.reset_parameters()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Dense MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        input_dtype = input.dtype
        input_device = input.device
        input = input.to(dtype=self.dtype,device=self.device)
        mlp_x = (activation_dict[self.activate](input@self.gate_projection.to(input.dtype).to(input.device)) * \
            (input@self.up_projection.to(input.dtype).to(input.device))) @ self.down_projection.to(input.dtype).to(input.device)
        if self.lora_rank == 0: 
            return mlp_x
        torch.manual_seed(self.lora_dropout_seed)
        mlp_lora_x = mlp_x + self.dropout(self.lora_alpha/self.lora_rank * ((input@self.lora_A.to(input.dtype).to(input.device))@self.lora_B.to(input.dtype).to(input.device)))
        input = input.to(input_dtype).to(input_device)
        return mlp_lora_x.to(input_dtype).to(input_device)
        
        
    def reset_parameters(self):
        """Initialize the weights of the Dense MLP module with LoRA adapters
        from a normal distribution (or a uniform distribution for lora weights)
        """
        self.up_projection = nn.Parameter(torch.zeros(self.h,self.ffh,dtype=self.dtype,device=self.device))
        self.gate_projection = nn.Parameter(torch.zeros(self.h,self.ffh,dtype=self.dtype,device=self.device))
        self.down_projection = nn.Parameter(torch.zeros(self.ffh,self.h,dtype=self.dtype,device=self.device))
        if self.lora_rank != 0:
            self.lora_A = nn.Parameter(torch.zeros(self.h,self.lora_rank,dtype=self.dtype,device=self.device))
            self.lora_B = nn.Parameter(torch.zeros(self.lora_rank,self.h,dtype=self.dtype,device=self.device))
        if self.activate is MLPActivationType.SIGMOID or self.activate is MLPActivationType.BILINEAR:
            # Xavier Initialization
            for i,proj in enumerate([self.up_projection,self.gate_projection,self.down_projection]):
                torch.manual_seed(self.init_base_seed+1+i)
                nn.init.xavier_normal_(proj.T)
            if self.lora_rank != 0:
                for i,proj in enumerate([self.lora_A,self.lora_B]):
                    torch.manual_seed(self.lora_init_base_seed+1+i)
                    nn.init.xavier_uniform_(proj.T)
        else:
            # Kaiming Initialization
            for i,proj in enumerate([self.up_projection,self.gate_projection,self.down_projection]):
                torch.manual_seed(self.init_base_seed+1+i)
                nn.init.kaiming_normal_(proj.T, mode='fan_in', nonlinearity='relu')
            if self.lora_rank != 0:
                for i,proj in enumerate([self.lora_A,self.lora_B]):
                    torch.manual_seed(self.lora_init_base_seed+1+i)
                    nn.init.kaiming_uniform_(proj.T, mode='fan_in', nonlinearity='relu')
    
class SparseMLPWithLoRA(nn.Module):
    """Sparse MLP module with LoRA adapters
    This is a GLU-style sparse MLP layer with LoRA adapters, \
        where the sparcity is implemented as Mixture of Experts (MoE), \
            and each expert is a dense MLP with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        num_experts: int = 1,
        moe_topk: int = 1,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Sparse MLP module with LoRA adapters
        
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = MLPActivationType.SILU): activation type
            num_experts(int, default = 1): number of (global) experts, which can deduce expert_size = ffh_size // num_experts
            moe_topk(int, default = 1): topk-routing for MoE to control the sparcity
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean for the initialization
            init_std(float, default = 1.0): std for the initialization
            init_base_seed(int, default = 42): seed for the initialization
            lora_rank(int, default = 0): lora rank
            lora_alpha(Optional[float], default = None): lora alpha
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        self.h = hidden_size
        self.ne = num_experts
        self.top_k = moe_topk
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_base_seed = init_base_seed
        self.dtype = dtype
        self.device = device
        self.e = ffh_size // self.ne
        self.nle = self.ne // world_size
        self.start_pos, self.end_pos = rank * self.nle, (rank+1) * self.nle - 1
        self.experts = nn.ModuleList([
            DenseMLPWithLoRA(hidden_size,self.e,activation_type,init_base_seed+i+self.start_pos,
                             lora_rank,lora_alpha,lora_dropout_rate,
                             lora_dropout_seed+i+self.start_pos,lora_init_base_seed+i+self.start_pos,
                             dtype,device) 
                                      for i in range(self.nle)])
        self.reset_parameters()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Sparse MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        input_dtype = input.dtype
        input_device = input.device
        b,s,h = input.shape
        input = input.to(dtype=self.dtype,device=self.device).view(b*s,h)
        P = F.softmax(input.to(torch.float32)@self.G.to(input.device),dim=-1)
        router_weights, selected_experts = torch.topk(P,k=self.top_k,dim=-1)
        router_weights /= torch.sum(router_weights,dim=-1,keepdim=True)
        output = torch.zeros((b*s,h),dtype=input.dtype,device=input.device)
        expert_mask = F.one_hot(selected_experts, num_classes=self.ne).permute(2,1,0)
        for expert_idx in range(self.nle):
            expert_layer = self.experts[expert_idx]
            expert_idx = expert_idx + self.start_pos
            weight_idx, token_idx = torch.where(expert_mask[expert_idx])
            current_state = input[None,token_idx].view(-1,h)
            current_hidden_state = expert_layer(current_state) * router_weights[token_idx, weight_idx, None].to(input.dtype)
            output.index_add_(0,token_idx,current_hidden_state)
        return output.to(input_dtype).to(input_device).view(b,s,h)
        
    def reset_parameters(self):
        """Initialize the weights of each local expert from its own distribution \
            and the gating layer from a normal distribution
        """
        for expert in self.experts:
            expert.reset_parameters()
        self.G = nn.Parameter(torch.zeros((self.h,self.ne),dtype=torch.float32,device=self.device))
        torch.manual_seed(self.init_base_seed)
        nn.init.normal_(self.G,mean=self.init_mean,std=self.init_std)
