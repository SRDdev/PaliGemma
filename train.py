import os
import cv2
import torch
import yaml
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from modules.utils import load_config
from modules.vision_encoder import VisionModel, VisionConfigs
from modules.dataset_loader import ImageDataset

class Trainer:
    """Trainer class to encapsulate the training logic."""
    def __init__(self, config):
        self.config = config

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")

        # Initialize model and move to device
        model_config = VisionConfigs(
            hidden_size=config['hidden_size'],
            num_attention_heads=config['num_attention_heads'],
            intermediate_size=config['intermediate_size'],
            attention_dropout=config['attention_dropout']
        )
        self.model = VisionModel(model_config).to(self.device)

        # Set up transformations, dataset, and dataloader
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
        ])
        dataset = ImageDataset(image_dir=config['image_dir'], transform=self.transform)
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), config['learning_rate'])

    def train(self):
        """Train the model."""
        self.model.train()
        for epoch in range(self.config['epochs']):
            epoch_loss = 0.0
            for batch_idx, images in enumerate(self.dataloader):
                images = images.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, images)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.config['epochs']}], Loss: {epoch_loss / len(self.dataloader):.4f}")

    def save_sample_output(self):
        """Save a sample output from the trained model."""
        with torch.no_grad():
            sample_image = next(iter(self.dataloader))[0].unsqueeze(0).to(self.device)
            output = self.model(sample_image).cpu().reshape(
                self.config['image_size'], self.config['image_size'], 3
            )
            cv2.imwrite(self.config['output_image'], output.numpy())
            print(f"Sample image saved to {self.config['output_image']}")

def main():
    config = load_config()

    trainer = Trainer(config)
    trainer.train()

    trainer.save_sample_output()

if __name__ == "__main__":
    main()
