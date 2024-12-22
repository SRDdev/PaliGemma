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

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
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