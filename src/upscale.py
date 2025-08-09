import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from train import CNNEnhancer

def enhance_image(image_path, model_path, output_path_hr, output_path_lr, hr_size=256,
                  lr_size=None):
    """
    Applique un modèle de super-résolution sur une image.

    Args:
        image_path (str): chemin de l'image originale.
        model_path (str): chemin vers le modèle entraîné (.pth).
        output_path_hr (str): chemin pour sauvegarder la sortie haute résolution.
        output_path_lr (str): chemin pour sauvegarder l'image LR upscalée (entrée réseau).
        lr_size (int, optionnel): taille basse résolution avant upscale.
                                  Si None, prend orig_width // 2 et orig_height // 2 (x2).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Charger l’image d’origine
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # remet l’image dans le bon sens
    orig_size = img.size  # (width, height)

    # 2. Créer l’image basse résolution cohérente avec l'entraînement
    if lr_size is None:
        # comportement ancien : ×2
        lr_img = img.resize((lr_size[0] // 2, lr_size[1] // 2), Image.BICUBIC)
    else:
        # downscale à lr_size, puis upscale à hr size
        lr_img = img.resize((lr_size, lr_size), Image.BICUBIC)

    lr_img_upscaled = lr_img.resize((hr_size, hr_size), Image.BICUBIC)

    # 3. Transformer en tenseur pour PyTorch
    to_tensor = T.ToTensor()
    input_tensor = to_tensor(lr_img_upscaled).unsqueeze(0).to(device)

    # 4. Charger le modèle
    model = CNNEnhancer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. Prédiction
    with torch.no_grad():
        output_tensor = model(input_tensor).clamp(0, 1).squeeze(0).cpu()

    # 6. Reconvertir en images
    to_pil = T.ToPILImage()
    enhanced_img = to_pil(output_tensor)
    lr_img_upscaled_pil = to_pil(input_tensor.squeeze(0).cpu())

    # 7. Sauvegarde
    enhanced_img.save(output_path_hr)
    lr_img_upscaled_pil.save(output_path_lr)

    print(f"✅ Image améliorée sauvegardée : {output_path_hr}")
    print(f"🔍 Image basse résolution (entrée du modèle) sauvegardée : {output_path_lr}")


enhance_image(
    image_path="dataset/2021-12-06_16-24-28_000.jpeg",
    model_path="models/cnn_enhancer.pth",
    output_path_hr="results/enhanced_hr.png",
    output_path_lr="results/input_lr.png",
    hr_size=256,
    lr_size=128  # correspond à l'entraînement
)