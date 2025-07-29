import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from train import CNNEnhancer

def enhance_image(image_path, model_path, output_path_hr, output_path_lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Charger l’image d’origine
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # remet l’image dans le bon sens

    # 2. Créer l’image basse résolution (downscale + upscale)
    orig_size = img.size  # (width, height)
    lr_img = img.resize((orig_size[0] // 2, orig_size[1] // 2), Image.BICUBIC)
    lr_img_upscaled = lr_img.resize(orig_size, Image.BICUBIC)

    # 3. Transformer en tenseur pour PyTorch
    to_tensor = T.ToTensor()
    input_tensor = to_tensor(lr_img_upscaled).unsqueeze(0).to(device)

    # 4. Charger le modèle
    model = CNNEnhancer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. Prédiction
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu()

    # 6. Reconvertir les tenseurs en images
    to_pil = T.ToPILImage()
    enhanced_img = to_pil(output_tensor)
    lr_img_upscaled_pil = to_pil(input_tensor.squeeze(0).cpu())

    # 7. Sauvegarde
    enhanced_img.save(output_path_hr)
    lr_img_upscaled_pil.save(output_path_lr)

    print(f"✅ Image améliorée sauvegardée : {output_path_hr}")
    print(f"🔍 Image basse résolution (entrée du modèle) sauvegardée : {output_path_lr}")


enhance_image("dataset/2021-12-06_16-24-28_000.jpeg", "cnn_enhancer.pth", "results/enhanced_hr.png", "results/input_lr.png")