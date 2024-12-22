"""Gemma Model"""

import math
import torch
from torch import nn
from torch.nn import functional as f
from typing import Optional,List,Dict,Tuple
from .vision_encoder import VisionConfigs, VisionModel

class KVCache():
    """
    KVCache is a class that stores the key and value states for each layer of the model.
    
    1. The key and value states are stored in a list of tensors, where each tensor is of shape [batch_size, num_heads_kv, seq_len, head_dim].
    2. The key and value states are stored in the same list, so the key and value states for layer i are stored at index i in the list
    """
    def __init__(self) -> None:
        """
        Initializes the KVCache object.
        """
        self.key_cache: List[torch.Tensor] = []                 # Create an empty list to store the key states
        self.value_cache: List[torch.Tensor] = []               # Create an empty list to store the value states
    
    def num_items(self) -> int:
        """
        Returns the number of items in the cache.
        """
        if len(self.key_cache) == 0:
            return 0 
        else:
            return self.key_cache[0].shape[-2]                  # key_cache = [batch_size, num_heads_kv , seq_len, head_dim]
        
    def update(self,key_state:torch.Tensor,value_state: torch.Tensor,layer_idx: int) -> Tuple[torch.Tensor,torch.Tensor] :
        """
        Updates the key and value states for the given layer index.

        Args:
            key_state: torch.Tensor of shape [batch_size, num_heads_kv, seq_len, head_dim]
            value_state: torch.Tensor of shape [batch_size, num_heads_kv, seq_len, head_dim]
            layer_idx: int, the index of the layer to update the key and value states for
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated key and value states for the given layer index
        """
        if len(key_state) <= layer_idx:
            self.key_cache.append(key_state)
            self.value_cache.append(value_state)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache][layer_idx],key_state, dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache][layer_idx],value_state, dim=-2)
        
        return self.key_cache[layer_idx],self.value_cache[layer_idx]

class TextConfig(nn.Module):
    def __init__(self,vocab_size,hidden_size,intermidiate_size,num_hidden_layers,num_attention_heads,num_key_value_heads,head_dims=256,max_position_embeddings=8192,
                 rms_norm_eps = 1e-6,rope_theta = 10000.0,attention_bias = False,attention_dropout = 0.0,pad_token_id = None,**kwargs,):
        """
        The TextConfig class defines the configuration for the Gemma model.

        Args:
            vocab_size: int, the size of the vocabulary
            hidden_size: int, the hidden size of the model
            intermidiate_size: int, the size of the intermediate layer in the MLP
            num_hidden_layers: int, the number of hidden layers in the model
            num_attention_heads: int, the number of attention heads in the model
            num_key_value_heads: int, the number of key-value heads in the model
            head_dims: int, the dimension of the attention heads
            max_position_embeddings: int, the maximum position embeddings
            rms_norm_eps: float, the epsilon value for the RMSNorm layer
            rope_theta: float, the theta value for the Rotary Positional Encoding
            attention_bias: bool, whether to use bias in the attention layer
            attention_dropout: float, the dropout rate for the attention layer
            pad_token_id: int, the token id for the padding
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermidiate_size = intermidiate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dims = head_dims
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
    
class RMSNorm(nn.Module):
    """
    RMSNorm is a class that implements the Root Mean Square Layer Normalization.
    
    1. The RMSNorm layer normalizes the input tensor by the root mean square of the input tensor.
    2. The RMSNorm layer also learns a scaling parameter for the normalized tensor.
    """
    def __init__(self,dim: int, eps: float = 1e-6):
        """
        Initializes the RMSNorm layer.
        """
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.zeros(dim))                           # Initialize the scaling parameter to zeros
    
    def _norm(self,x):
        """
        Normalizes the input tensor by the root mean square of the input tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self,x):
        """
        Forward pass of the RMSNorm layer.
        """
        output = self._norm(x.float())
        output = output * (1 + self.weights.float())
        return output.type_as(x)
    
class RotaryEmbedding(nn.Module):
    """
    RotaryEmbedding is a class that implements the Rotary Positional Encoding from the paper "Enhanced Transformer with Rotary Position Embedding".
    Allows the model to learn the positional encoding by adding sinusoidal and cosinusoidal functions to the input tensor.
    Improved version of positional and absolute positional encodings.

    Args:
        dim: int, the dimension of the input tensor
        max_position_embeddings: int, the maximum position embeddings
        base: int, the base value for the sinusoidal and cosinusoidal functions
        device: str, the device to run the model on
    """
    def __init__(self,dim,max_position_embeddings=2048,base=10000,device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Why do we need inv_freq? : We need inv_freq to compute the sinusoidal and cosinusoidal functions for the positional encoding.
        inv_freq = 1.0 / (self.base ** (torch.arange(0,self.dim,2,dtype =torch.int64).float() / self.dim)) # Compute the inverse frequency
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self,x,position_ids,seq_len=None):
        """
        Forward pass of the Rotary Positional Encoding layer.
        
        Args:
            x: torch.Tensor, the input tensor
            position_ids: torch.Tensor, the position ids
            seq_len: int, the sequence length
        """
        self.inv_freq.to(x.device)                                                                      # Move the inv_freq tensor to the device
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)   # Expand the inv_freq tensor
        position_ids_expanded = position_ids[:, None, :].float()                                        # Expand the position ids tensor
        device_type = x.device.type                                                                     
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"   # Check if the device type is "mps" or not
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    """
    Rotate the input tensor by half.
    
    Args:
        x: torch.Tensor, the input tensor
    """
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply the Rotary Positional Encoding to the query and key tensors.

    Args:
        q: torch.Tensor, the query tensor
        k: torch.Tensor, the key tensor
        cos: torch.Tensor, the cosinusoidal tensor
        sin: torch.Tensor, the sinusoidal tensor
        unsqueeze_dim: int, the dimension to unsqueeze the tensors
    """
    cos = cos.unsqueeze(unsqueeze_dim)                                  # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)                                  # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class TextMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for the Gemma model.

    1. The MLP consists of two linear layers with a GELU activation function.
    2. The MLP projects the input tensor to an intermediate size and then back to the hidden size.
    3. The MLP also uses the RMSNorm layer to normalize the output tensor.
    """
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.up_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.down_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False) 

    def forward(self,x):
        y = self.gate_proj(x)                        # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        y = nn.functional.gelu(y,approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        j = self.up_proj(x)                          # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        z = y * j                                    # [Batch_Size, Seq_Len, Intermediate_Size]
        z = self.down_proj(z)                        # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return z
    
def repeate_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key and value states for the given number of repetitions.

    Args:
        hidden_states: torch.Tensor, the key and value states
        n_rep: int, the number of repetitions
    
    Returns:
        torch.Tensor: The repeated key and value states
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape                                        # Get the shape of the key and value states
    if n_rep == 1:                                                                                          # If the number of repetitions is 1
        return hidden_states                                                                               
    else:
        hidden_states = hidden_states[:,:,None,:,:].expand(batch, num_key_value_heads, n_rep, slen, head_dim) # Expand the key and value states tensor to the given number of repetitions
        return hidden_states.reshape(batch, num_key_value_heads*n_rep, slen, head_dim)                        # Reshape the key and value states tensor to the given number of repetitions
    
class TextAttention(nn.Module):
    """
    Attention Module for Gemma Model Consisting of RoPE and KV Cache.
    """
    def __init__(self, config:TextConfig, layer_idx:Optional[int]=None):
        """
        """
        super().__init__()
        self.config = config                                                                                            # Get the configuration
        self.layer_idx = layer_idx                                                                                      # Get the layer index
        self.attention_dropout = config.attention_dropout                                                               # Get the attention dropout rate                                  
        self.hidden_size = config.hidden_size                                                                           # Get the hidden size
        self.num_heads = config.num_attention_heads                                                                     # Get the number of attention heads
        self.head_dim = config.head_dims                                                                                # Get the dimension of the attention heads
        self.num_key_value_heads = config.num_key_value_heads                                                           # Get the number of key-value heads 
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads                                          # Get the number of key-value groups 
        self.max_position_embeddings = config.max_position_embeddings                                                   # Get the maximum position embeddings
        self.rope_theta = config.rope_theta                                                                             # Get the theta value for the Rotary Positional Encoding
        self.is_causal = True                                                                                           # Set the causal attention flag to True
        assert self.hidden_size % self.num_heads == 0                                                                   # Check if the hidden size is divisible by the number of attention heads
 
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias=config.attention_bias)             # Initalize the query projection layer with hidden_size, num_heads*head_dim and bias
        
        # Why num_key_value_heads? : Because we want to have separate key and value projections for each key-value head for KV Cache.
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias=config.attention_bias)   # Initalize the key projection layer with hidden_size, num_key_value_heads*head_dim and bias
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias=config.attention_bias)   # Initalize the value projection layer with hidden_size, num_key_value_heads*head_dim and bias
        self.o_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias= config.attention_bias)            # Initalize the output projection layer with hidden_size, num_heads*head_dim and bias
        self.rotary_pos_emb = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta) # Initalize the Rotary Positional Encoding layer

    def forward(self, hidden_states:torch.Tensor, attention_mask:Optional[torch.Tensor]=None, position_ids:Optional[torch.Tensor]=None, kv_cache:Optional[KVCache]=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of cached self attention.
        """
        bsz, q_len = hidden_states.size()                                                                         # [batch_size, seq_len, hidden_size]

        query_states = self.q_proj(hidden_states)                                                                 # Generate Query state
        key_states = self.k_proj(hidden_states)                                                                   # Generate Key state
        value_states = self.v_proj(hidden_states)                                                                 # Generate Value state

        query_states = query_states.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)                   # Update Shapes
        key_states = key_states.view(bsz,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)             # Update Shapes
        value_states = value_states.view(bsz,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)         # Update Shapes

        cos, sin = self.rotary_pos_emb(value_states, position_ids, seq_len=None)                                  # Generate RoPE Embeddings for it 

        query_states, key_states = apply_rotary_pos_emb(query_states,key_states,cos,sin)                          # Apply RoPE 

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states,value_states,self.layer_idx)                    # Update KVCache

        key_states = repeate_kv(key_states, self.num_key_value_heads)
        value_states = repeate_kv(value_states, self.num_key_value_heads)

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)          # (Q.K^T)/sqrt(head_dim)

        assert attention_mask is not None
        attn_weights = attention_mask + attn_weights                                                             # Add Attention mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)   # Apply softmax
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)     # Apply Attention Dropout
        attn_output = torch.matmul(attn_weights,value_states)                                                    # Attention_weights . V

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`Attention Output` should be of the size {(bsz, self.num_heads, q_len, self.head_dim)}, but is of size {(attn_output.size())}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)                                                                   # Resize and send out weights

        return attn_output, attn_weights

class TextDecoderLayer(nn.Module):
    """
    TextDecoderLayer is a class that implements a single layer of the Gemma model.
    
    Args:
        config: TextConfig, the configuration of the model
        layer_idx: int, the index of the layer
    """
    def __init__(self, config:TextConfig, layer_idx:int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = TextAttention(config=self.config,layer_idx=self.layer_idx)
        self.mlp = TextMLP(config=self.config)
        self.input_layer_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.post_attn_layer_norm = RMSNorm(self.config.hidden_size,self.config.rms_norm_eps)

    def forward(self,
                hidden_state:torch.Tensor,
                attention_mask:Optional[torch.Tensor] = None,
                position_ids:Optional[torch.LongTensor]= None,
                kv_cache: Optional[KVCache] = None,
                ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass of the TextDecoderLayer.

        Args:
            hidden_state: torch.Tensor, the input tensor
            attention_mask: torch.Tensor, the attention mask
            position_ids: torch.LongTensor, the position ids
            kv_cache: KVCache, the key-value cache
        """
        residual = hidden_state                                                               # Store the input tensor in the residual variable
        hidden_state = self.input_layer_norm(hidden_state)                                    # Apply Layer Normalization
        hidden_state,_ = self.self_attn(hidden_state,attention_mask,position_ids,kv_cache)    # Apply Self Attention
        hidden_state = residual + hidden_state                                                # Add the residual tensor to the output tensor
        residual = hidden_state                                                               # Store the output tensor in the residual variable
        hidden_state = self.post_attn_layer_norm(hidden_state)                                # Apply Layer Normalization
        hidden_state = self.mlp(hidden_state)                                                 # Apply the MLP
        hidden_state = residual + hidden_state                                                # Add the residual tensor to the output tensor
        return hidden_state

class GemmaModel(nn.Module):
    """
    GemmaModel is a class that implements the Gemma model.
    
    Args:
        config: TextConfig, the configuration of the model
    """
    def __init__(self,config:TextConfig):
        """
        Initializes the GemmaModel.
        """
        super().__init__()
        self.config = config                                                                        # Get the configuration
        self.padding_idx = config.pad_token_id                                                      # Get the padding index
        self.vocab_size = config.vocab_size                                                         # Get the vocabulary size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)   # Initialize the embedding layer
        self.layers = nn.ModuleList(
            [TextDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )                                                                                           # Initialize the `n` decoder layers.
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)                                # Initialize the RMSNorm layer

    def get_input_embedding(self):
        """Returns the input embedding layer."""
        return self.embed_tokens
    
    def forward(self,
                attention_mask:Optional[torch.Tensor]=None,
                position_ids:Optional[torch.LongTensor]=None,
                input_embeds:Optional[torch.FloatTensor]=None,
                kv_cache:Optional[KVCache]=None) -> torch.FloatTensor:
        """
        Forward pass of the GemmaModel.
        """
        hidden_states = input_embeds                                                                # Get the input embeddings
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)          # Compute the normalizer
        hidden_states = hidden_states + normalizer                                                  # Add the normalizer to the input embeddings

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states,attention_mask,position_ids,kv_cache)       # Pass the normalized input to the `n` decoder layers and get the output tensor
        
        hidden_states = self.norm(hidden_states)                                                    # Apply the RMSNorm layer to the output tensor                          
        return hidden_states
    
class GemmaForCausalLM(nn.Module):
    """
    GemmaForCausalLM is a class that implements the Gemma model for causal language modeling.
    """
    def __init__(self, config):
        """
        Initializes the GemmaForCausalLM.
        """
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = self.config.vocab_size

        # Note : What is LM head ?
        # The lm_head is a linear layer that takes the last hidden states from GemmaModel and projects them into vocabulary logits for language modeling.
        # Essentially, it maps the hidden representations to a distribution over the full vocabulary, enabling next-token prediction and related tasks.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embedding(self):
        """
        Returns the input embedding layer.
        """
        return self.model.embed_tokens
    
    def tie_weights(self):
        """
        Ties the weights of the model.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def foraward(self,
                 attention_mask:Optional[torch.Tensor]=None,
                 position_ids:Optional[torch.LongTensor]=None,
                 input_embeds:Optional[torch.FloatTensor]=None,
                 kv_cache:Optional[KVCache]=None) -> Tuple:
        """
        Forward pass of the GemmaForCausalLM.
        """
        output = self.model(
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            position_ids=position_ids,
            kv_cache=kv_cache
        )

        hidden_states = output
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }
        if kv_cache is not None:
            return_data['kv_cache'] = kv_cache

        return return_data

        

