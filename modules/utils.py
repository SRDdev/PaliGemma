"""
This file contains helper functions
"""
import yaml
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

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
