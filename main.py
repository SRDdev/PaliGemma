import cv2
import torch
from modules.utils import load_image
from modules.vision_encoder import VisionModel, VisionConfigs

def main():
    config = VisionConfigs()
    model = VisionModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    image_tensor = load_image("data/train/Image.jpg").to(device)

    with torch.no_grad():
        output = model(image_tensor)

    print("=" * 100)
    print(f"Output shape: {output.shape}")

    vision_output = output.detach().cpu().reshape(224, 224, 3)
    cv2.imwrite("vision_output.png", vision_output.numpy())
    print("Image saved !!!")
    print("=" * 100)

if __name__ == "__main__":
    main()
