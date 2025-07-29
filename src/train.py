import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# Define the CNN model
class UpscaleCNN(nn.Module):
    def __init__(self):
        super(UpscaleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, padding=2)
        self.upsample = nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        x = self.upsample(x)
        return x

class CatDataset(Dataset):
    def __init__(self, image_dir, transform=None, low_res_transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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

def train_model(dataset_path, model_save_path, num_epochs=100, learning_rate=0.001, low_res_size=64, batch_size=32):
    """
    Trains the CNN model and saves it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    high_res_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    low_res_transform = transforms.Compose([
        transforms.Resize((low_res_size, low_res_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),  
    ])

    dataset = CatDataset(image_dir=dataset_path, transform=high_res_transform, low_res_transform=low_res_transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=8)

    model = UpscaleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")
        for low_res, high_res in progress_bar:
            inputs = low_res.to(device)
            targets = high_res.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train_model('dataset', 'models/upscale_cnn.pth', low_res_size=64)
