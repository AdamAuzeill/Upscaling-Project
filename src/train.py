import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ----------------------------
# Dataset pré-calculé (crop fixe + lr/up)
# ----------------------------
class FixedCropEnhancementDataset(Dataset):
    def __init__(self, folder, hr_size=128, lr_size=None):
        """
        Dataset pour super-résolution avec crop fixe.

        Args:
            folder (str): dossier contenant les images
            hr_size (int): taille (carrée) des patches haute résolution
            lr_size (int, optionnel): taille basse résolution avant upscale.
                                      Si None, on prend hr_size // 2 (x2 upscale).
        """
        self.hr_size = hr_size
        self.lr_size = lr_size if lr_size is not None else hr_size // 2

        self.paths = sorted(
            glob.glob(os.path.join(folder, '*.jpg')) +
            glob.glob(os.path.join(folder, '*.jpeg')) +
            glob.glob(os.path.join(folder, '*.png'))
        )
        self.to_tensor = T.ToTensor()


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if img.width < self.hr_size or img.height < self.hr_size:
            raise RuntimeError(f"Image trop petite : {self.paths[idx]}")

        # Crop centré
        left = (img.width - self.hr_size) // 2
        top = (img.height - self.hr_size) // 2
        hr_patch = img.crop((left, top, left + self.hr_size, top + self.hr_size))

        # Downscale puis upscale
        lr_patch = hr_patch.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_patch = lr_patch.resize((self.hr_size, self.hr_size), Image.BICUBIC)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)



class PrecomputedRAMDataset(Dataset):
    def __init__(self, base_dataset):
        print("Préchargement des données en RAM, ça peut prendre quelques secondes...")
        self.data = []
        for i in tqdm(range(len(base_dataset))):
            self.data.append(base_dataset[i])
        print(f"Chargé {len(self.data)} images en RAM")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------
# Modèle CNN
# ----------------------------
class CNNEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Boucle d'entraînement
# ----------------------------
def train_model(data_folder,
                hr_size=128,
                lr_size=None,
                epochs=20,
                batch_size=16,
                learning_rate=1e-4,
                save_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + préchargement en RAM
    base_dataset = FixedCropEnhancementDataset(data_folder, hr_size=hr_size, lr_size=lr_size)
    dataset = PrecomputedRAMDataset(base_dataset)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    # Modèle + optimiseur + fonction de perte
    model = CNNEnhancer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for lr_imgs, hr_imgs in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            preds = model(lr_imgs)
            loss = loss_fn(preds, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * lr_imgs.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"\n Modèle sauvegardé dans {save_path}")


# ------------
# Exécution
# ------------
if __name__ == "__main__":
    DATA_FOLDER = "dataset"
    train_model(
        data_folder=DATA_FOLDER,
        hr_size=256,
        lr_size=128,
        epochs=2,
        batch_size=16,
        learning_rate=1e-4,
        save_path="/content/drive/MyDrive/Colab Notebooks/models/cnn_enhancer.pth"
    )
