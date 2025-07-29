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
    def __init__(self, folder, crop_size=12):
        self.crop_size = crop_size
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
        if img.width < self.crop_size or img.height < self.crop_size:
            raise RuntimeError(f"Image trop petite : {self.paths[idx]}")

        # Crop un peu aléatoire
        left = (img.width - self.crop_size) // 2
        top = (img.height - self.crop_size) // 2
        hr_patch = img.crop((left, top, left + self.crop_size, top + self.crop_size))

        # Downscale puis upscale
        lr_patch = hr_patch.resize((self.crop_size // 2, self.crop_size // 2), Image.BICUBIC)
        lr_patch = lr_patch.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)


class PrecomputedRAMDataset(Dataset):
    def __init__(self, base_dataset):
        print("Préchargement des données en RAM, ça peut prendre quelques secondes...")
        self.data = []
        for i in tqdm(range(len(base_dataset))):
            self.data.append(base_dataset[i])
        print(f"✅ Chargé {len(self.data)} images en RAM")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------
# Modèle CNN simple
# ----------------------------
class CNNEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 5, padding=2)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Boucle d'entraînement
# ----------------------------
def train_model(data_folder,
                crop_size=128,
                epochs=20,
                batch_size=16,
                lr=1e-4,
                save_path=None):

    # On utilise le GPU si possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Création du dataset et chargement de celui-ci
    base_dataset = FixedCropEnhancementDataset(data_folder, crop_size)
    dataset = PrecomputedRAMDataset(base_dataset)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = CNNEnhancer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0

        # Boucle sur les batch
        for lr_imgs, hr_imgs in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            preds = model(lr_imgs)
            loss = loss_fn(preds, hr_imgs)

            optimizer.zero_grad() # On remet les gradients à 0
            loss.backward() # On calcule la dérivée partielle (gradient) pour chaque poids
            optimizer.step() # On ajuste les poids en fonction du gradient

            running_loss += loss.item() * lr_imgs.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Modèle sauvegardé dans {save_path}")


# ------------
# Exécution
# ------------
if __name__ == "__main__":
    DATA_FOLDER = "dataset"
    train_model(
        data_folder=DATA_FOLDER,
        crop_size=128,
        epochs=2,
        batch_size=8,
        lr=2e-4,
        save_path="models/cnn_enhancer.pth"
    )
