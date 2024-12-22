"""PaliGemma Model for multimodal data.

Note: The file uses some uncommon PyTorch Features which are not commonly know and hence have explained them below.

1. torch.full : This function is used to create a tensor of a given shape and fill it with a given value. 

2. torch.masked_scatter : This function is used to copy the elements from the source tensor to the destination tensor based on the mask tensor.

3. torch.finfo : This function is used to get the floating point number limits for a given data type.

4. torch.cumsum : This function is used to compute the cumulative sum of the elements of a tensor in the specified dimension.

5. torch.unsqueeze : This function is used to add a new dimension to the tensor at the specified position.

6. torch.masked_fill : This function is used to fill the elements of the tensor with a given value based on the mask tensor.

7. torch.expand : This function is used to expand the dimensions of the tensor based on the given size.

8. torch.where : This function is used to select elements from two tensors based on the condition provided.

9. torch.shape : This function is used to get the shape of the tensor.

10. torch.device : This function is used to get the device on which the tensor is stored.
"""

import math
import torch
import numpy as np
from torch import nn
from PIL import Image
from .utils import process_images, add_image_tokens_to_prompt
from torch.nn import functional as f
from typing import Optional,List,Dict,Tuple
from .vision_encoder import VisionConfigs, VisionModel
from .text_encoder import KVCache, TextConfig, GemmaForCausalLM
from transformers import AutoTokenizer
import json
import glob
import os
from safetensors import safe_open

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


#-------------------------------------------------- MultiModal Config --------------------------------------------------#
class MultiModalConfig(nn.Module):
    """
    The MultiModalConfig class is used to store the configuration for the multimodal model.
    """
    def __init__(self,vision_config=None,text_config=None,ignore_index=-100,image_token_index=256000,vocab_size=257152,projection_dim=2048,hidden_size=2048,pad_tokens=None,**kwargs):
        """
        The constructor for the MultiModalConfig class.
        
        Args:
        vision_config (Dict) : The configuration for the vision model.
        text_config (Dict) : The configuration for the text model.
        ignore_index (int) : The index to ignore.
        image_token_index (int) : The index for the image token.
        vocab_size (int) : The size of the vocabulary.
        projection_dim (int) : The dimension for the projection.
        hidden_size (int) : The size of the hidden layer.
        pad_tokens (List) : The list of pad tokens.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_tokens = pad_tokens
        self.vision_config = vision_config
        self.text_config = text_config
        self.is_encoder_decoder = False

        self.vision_config = VisionConfigs(**vision_config)
        self.text_config = TextConfig(**text_config)
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = self.projection_dim
#-------------------------------------------------- MultiModal Projector --------------------------------------------------#
class PaliGemmaMultiModalProjector(nn.Module):
    """
    The PaliGemmaMultiModalProjector class is used to project the image features to the hidden size of the model.
    """
    def __init__(self, config:MultiModalConfig):
        super().__init__()
        self.Linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=False)

    def forward(self, image_features):
        return self.Linear(image_features)
#-------------------------------------------------- MultiModal PreProcessor --------------------------------------------------#
class PaliGemmaProcessor(nn.Module):
    """
    Input Processor for PaliGemma Model.
    """
    IMAGE_TOKEN = "<image>"
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)] # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)] # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text:List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True)-> dict:
        """
        """
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."
        pixel_values = process_images(images,size=(self.image_size, self.image_size),resample=Image.Resampling.BICUBIC,rescale_factor=1 / 255.0,image_mean=IMAGENET_STANDARD_MEAN,image_std=IMAGENET_STANDARD_STD)
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [  add_image_tokens_to_prompt(prefix_prompt=prompt,
                                                      bos_token=self.tokenizer.bos_token,
                                                      image_seq_len=self.image_seq_length,
                                                      image_token=self.IMAGE_TOKEN) 
                        for prompt in text]

        inputs = self.tokenizer(input_strings, return_tensor="pt", padding= padding, truncation=truncation)
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
#-------------------------------------------------- MultiModal Model --------------------------------------------------#
class PaliGemmaForConditionalGeneration(nn.Module):
    """
    The PaliGemmaForConditionalGeneration class is used to generate the output based on the input.
    """
    def __init__(self, config:MultiModalConfig):
        """
        The constructor for the PaliGemmaForConditionalGeneration class.
        
        Args:
        config (MultiModalConfig) : The configuration object for the multimodal model.
        """
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(self.config.vision_config)
        self.multimodal_projector = PaliGemmaMultiModalProjector(self.config)
        self.vocab_size = self.config.vocab_size

        self.language_model = GemmaForCausalLM(self.config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        """
        This function is used to tie the weights of the language model.
        """
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_tokens(self,
                                           image_features:torch.Tensor,
                                           input_embds:torch.Tensor,
                                           input_ids: torch.Tensor,
                                           attention_mask: torch.Tensor,
                                           kv_cache: Optional[KVCache] = None):
        """
        This function is used to merge the input embeddings with the image features.
        Working Explained Step by Step:
        1. Scale the image features.
        2. Create a final embedding tensor with the same shape as the input embeddings.
        3. Create masks for text, image and padding tokens. 
        3a. How is the mask created? : The mask is created by checking if the input_ids are not equal to the image token index or the pad token index.
        3b. Why do we create the mask like this? : We create the mask like this to make sure that the image tokens and the pad tokens are not considered for attention.
        4. Expand the masks to the same shape as the image features.
        5. Merge the text and image tokens.
        6. Create a causal mask.
        7. Create position ids based on the attention mask.
        8. Return the final embedding, causal mask and position ids.
        """
        
        _,_,embed_dim = image_features.shape()                                                          # Get the shape of the image features
        batch_size,seq_len = input_ids.shape()                                                          # Get the shape of the input ids
        dtype,device = input_embds.dtype,input_embds.device                                             # Get the dtype and device of the input embeddings

        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features/(self.config.hidden_size ** 0.5)                         # Scale the image features
        final_embedding = torch.zeros(batch_size,seq_len,embed_dim,dtype=dtype,device=device)           # Create a final embedding tensor with the same shape as the input embeddings
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)     # Create a mask for text tokens
        image_mask = input_ids == self.config.image_token_index                                         # Create a mask for image tokens
        pad_mask = input_ids == self.pad_token_id                                                       # Create a mask for padding tokens

        # Note : Why are we doing this ?
        # We are doing this to make sure that the image tokens and the pad tokens are not considered for attention.
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)                          # Expand the text mask to the same shape as the image features
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)                            # Expand the pad mask to the same shape as the image features
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)                        # Expand the image mask to the same shape as the image features

        # Merge the text and image tokens
        final_embedding = torch.where(text_mask_expanded, input_embds, final_embedding)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #----------------------------------------- Create Attention Mask --------------------------------------------#
        min_dtype = torch.finfo(dtype).min  # Smallest representable number
        q_len = input_embds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            casual_mask = torch.full((batch_size, q_len, q_len), min_dtype, dtype=dtype, device=device)
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full((batch_size, q_len, kv_len), min_dtype, dtype=dtype, device=device)

        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)


        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids
    
    def forward(self,input_ids:torch.LongTensor=None,
                pixel_values:torch.FloatTensor=None,
                attention_mask:Optional[torch.Tensor]=None,
                kv_cache:Optional[KVCache]=None) -> Tuple:
        """
        The forward function for the PaliGemmaForConditionalGeneration class.
        """
        # Make sure that the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings   shape: (Batch_Size, Seq_Len, Hidden_Size)
        input_embds = self.language_model.get_input_embedding()(input_ids)

        # 2. Merge text and images  [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(input_embds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multimodal_projector(selected_image_feature)
        # Merge the embeddings of the text tokens and the image tokens
        input_embds, attention_mask, position_ids = self._merge_input_ids_with_image_tokens(image_features, input_embds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(attention_mask,position_ids,input_embds,kv_cache)

        return outputs

#-------------------------------------------------- MultiModal Model --------------------------------------------------#
def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Loads the Huggingface model weights into the architecture for PaliGemma.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = MultiModalConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
