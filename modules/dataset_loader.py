from torch.utils.data import Dataset, DataLoader
import cv2
import os

class ImageDataset(Dataset):
    """
    Custom dataset for loading images from a directory.
    ```
    Example:
    image_dir = 'path/to/image/directory'
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = ImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(batch.shape)
    ```
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        if self.transform:
            image = self.transform(image)

        return image
