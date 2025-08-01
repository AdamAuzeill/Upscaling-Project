import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from src.CNN_model import UpscaleCNN
from src.EDSR_model import EDSR


# Custom Dataset for your cat images
class CatDataset(Dataset):
    def __init__(self, image_dir, transform=None, low_res_transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
        self.low_res_transform = low_res_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        high_res_image = Image.open(img_path).convert("RGB")

        low_res_image = high_res_image.copy()

        if self.transform:
            high_res_image = self.transform(high_res_image)

        if self.low_res_transform:
            low_res_image = self.low_res_transform(low_res_image)

        return low_res_image, high_res_image

def train_model(dataset_path, model_save_path, model_name='cnn', num_epochs=100, learning_rate=0.001):
    """
    Trains the model and saves it.
    """
    high_res_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    low_res_transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])

    dataset = CatDataset(image_dir=dataset_path, transform=high_res_transform, low_res_transform=low_res_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    if model_name == 'cnn':
        model = UpscaleCNN()
    elif model_name == 'edsr':
        model = EDSR()
    else:
        raise ValueError("Invalid model name. Choose 'cnn' or 'edsr'.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for low_res, high_res in dataloader:
            inputs = low_res
            targets = high_res

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train an upscaling model.")
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'edsr'], help='The model to train.')
    args = parser.parse_args()

    model_save_path = f'models/upscale_{args.model}.pth'

    # Example usage:
    train_model('dataset', model_save_path, model_name=args.model)
