import torch
from torch import nn

class Attention(nn.Module):
    """Implementation of Attention from `Attention is all you need` paper"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dims = self.embed_dim // self.num_heads
        self.scale = self.head_dims ** -0.5                                                             # Scale factor for attention weights
        self.dropout = config.attention_dropout

        # Ensure all linear layers are moved to CUDA
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim).cuda()
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim).cuda()
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim).cuda()
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim).cuda()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention mechanism on `hidden_states` tensor
        Args:
            hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            attn_output: torch.Tensor of shape (batch_size, seq_len, hidden_size)
            attn_weights: torch.Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        if not hidden_states.is_cuda:
            raise ValueError("Input tensor `hidden_states` must be on CUDA.")
        
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dims).transpose(1, 2)

        # Compute attention weights
        key_states_T = key_states.transpose(2, 3)
        attn_weights = (query_states @ key_states_T) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, "
                             f"but is {attn_weights.size()}")

        # Apply softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Compute attention output
        attn_output = attn_weights @ value_states

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dims):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dims)}, "
                             f"but is {attn_output.size()}")

        # Reshape back to original size
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class MLP(nn.Module):
    """Implementation of MultiLayer Perceptron"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Move layers to CUDA
        self.fc1 = nn.Linear(config.hidden_size, config.intermidiate_size).cuda()
        self.fc2 = nn.Linear(config.intermidiate_size, config.hidden_size).cuda()

    def forward(self, hidden_size: torch.Tensor) -> torch.Tensor:
        """
        Compute MLP on `hidden_size` tensor
        Args:
            hidden_size: torch.Tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size)
        """
        if not hidden_size.is_cuda:
            raise ValueError("Input tensor `hidden_size` must be on CUDA.")

        hidden_states = self.fc1(hidden_size)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)

        return hidden_states
