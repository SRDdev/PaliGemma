"""
This file contains helper functions
"""
import yaml
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from glob import glob
from safetensors import safe_open
import os
import json
from typing import Tuple
from multimodal import PaliGemmaForConditionalGeneration, MultiModalConfig
from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

def load_image(image_path: str) -> torch.Tensor:
    """
    Load an image from the given path and transform it to a tensor suitable for the Vision Model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing the image to match the model input
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Normalize(           # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    image_tensor = transform(image).unsqueeze(0)   # Add batch dimension
    return image_tensor

def load_config(config_path="config.yaml"):
    """Load configurations from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Rescales the input image to match the requirements.
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def resize(image: Image,size: Tuple[int, int],resample: Image.Resampling = None,reducing_gap: Optional[int] = None,) -> np.ndarray:
    """
    Resizes the input image as per the requirements.
    """
    height, width = size
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    return resized_image

def normalize(image: np.ndarray,mean: Union[float, Iterable[float]],std: Union[float, Iterable[float]],) -> np.ndarray:
    """
    Normalize the image based on given mean and std.
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Applies all the above functions on the image at once and returns the final image which is ready to send into model.
    """
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

