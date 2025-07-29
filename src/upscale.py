import torch
from torchvision import transforms
from PIL import Image
from train import UpscaleCNN
import os
from datetime import datetime

def process_and_save_images(model_path, input_image_path, downscale_size=(64, 64)):
    """
    Processes an input image: saves original, downscaled, and upscaled versions.
    """
    # Generate output folder name based on current date and time
    output_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join("images", output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Load original image
    image = Image.open(input_image_path).convert("RGB")
    image.save(os.path.join(output_folder, "original.png"))

    # Downscale image
    downscale_transform = transforms.Resize(downscale_size, interpolation=Image.BICUBIC)
    downscaled_image = downscale_transform(image)
    downscaled_image.save(os.path.join(output_folder, "downscaled.png"))

    # Prepare tensor for model
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(downscaled_image).unsqueeze(0)

    # Load model
    model = UpscaleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Upscale image
    with torch.no_grad():
        output_tensor = model(image_tensor)
    upscaled_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    upscaled_image.save(os.path.join(output_folder, "upscaled.png"))

    print(f"Images saved in {output_folder}")

if __name__ == '__main__':
    input_image_path = "dataset/2025-02-02_11-22-37_263.jpeg"
    model_path = "models/upscale_cnn.pth"

    process_and_save_images(model_path, input_image_path, downscale_size=(128, 128))