"""SigLip Implementation"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
from modules.attention import Attention, MLP

#------------------------------  Vision Configs---------------------------------------#
class VisionConfigs:
    """
    Configuration class to store parameters for the Vision Transformer model.

    ```
    Example:
    config = VisionConfigs(
        hidden_size=768,
        intermidiate_size=3072,
        num_hidden_layers=12,
        num_attnention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=None,
        device="cuda"
    )
    ```
    """
    def __init__(self,hidden_size=768,intermidiate_size=3072,num_hidden_layers=12,num_attnention_heads=12,num_channels=3,image_size=224,patch_size=16,
            layer_norm_eps=1e-6,attention_dropout=0.0,num_image_tokens: int = None,device: str = "cuda",**kwargs):
        super().__init__()
        self.hidden_size = hidden_size                                 # Hidden size of the model
        self.intermidiate_size = intermidiate_size                     # Intermediate size of the model
        self.num_hidden_layers = num_hidden_layers                     # Number of hidden layers
        self.num_attention_heads = num_attnention_heads                # Number of attention heads
        self.num_channels = num_channels                               # Number of channels in the input image
        self.image_size = image_size                                   # Size of the input image
        self.patch_size = patch_size                                   # Size of the patch
        self.layer_norm_eps = layer_norm_eps                           # Epsilon value for layer normalization
        self.attention_dropout = attention_dropout                     # Dropout probability for attention layers
        self.num_image_tokens = num_image_tokens                       # Number of image tokens
        self.device = torch.device(device)                             # Set device

#------------------------------  Vision Embeddings---------------------------------------#
class VisionEmbeddings(nn.Module):
    """
    Converts input images into patch embeddings with positional information.

    Args:
        config: VisionConfigs object containing configuration parameters.
    """
    def __init__(self, config: VisionConfigs):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid"
        ).to(config.device)  # Move to device

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim).to(config.device)

        # Register buffer to store position ids and ensure that it is not updated during backpropagation.
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)).to(config.device),
            persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel values to patch embeddings with positional information.
        """
        pixel_values = pixel_values.to(self.config.device)                          # Ensure input tensor is on the correct device
        embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)  # Flatten and transpose
        embeddings += self.position_embedding(self.position_ids)                    # Add positional embeddings
        return embeddings

#------------------------------  Vision Encoder---------------------------------------#
class VisionEncoder(nn.Module):
    """
    Vision Encoder module with self-attention and feed-forward layers.
    """
    def __init__(self, config: VisionConfigs):
        super().__init__()
        self.config = config

        self.self_attn = Attention(config).to(config.device)
        self.mlp = MLP(config).to(config.device)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps).to(config.device)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps).to(config.device)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Encoder module.
        """
        hidden_state = hidden_state.to(self.config.device)               
        
        residual = hidden_state
        hidden_state = self.layer_norm1(hidden_state)
        hidden_state, _ = self.self_attn(hidden_state)
        hidden_state += residual
 
        residual2 = hidden_state
        hidden_state = self.layer_norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state += residual2

        return hidden_state

#------------------------------  Vision Transformer---------------------------------------#
class VisionTransformer(nn.Module):
    """
    Vision Transformer Block by combining embeddings and encoder layers.
    """
    def __init__(self, config: VisionConfigs):
        super().__init__()
        self.config = config

        self.embeddings = VisionEmbeddings(config).to(config.device)
        self.encoder = VisionEncoder(config).to(config.device)
        self.post_layernorms = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps).to(config.device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer model.
        """
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        return self.post_layernorms(last_hidden_state)

#------------------------------  Vision Model---------------------------------------#
class VisionModel(nn.Module):
    """
    Complete Vision Model encapsulating the Vision Transformer.
    """
    def __init__(self, config: VisionConfigs):
        super().__init__()
        self.config = config
        self.vision_model = VisionTransformer(config).to(config.device)

    def forward(self, pixel_values: torch.Tensor) -> Tuple:
        """
        Forward pass of the Vision Model.
        """
        return self.vision_model(pixel_values)