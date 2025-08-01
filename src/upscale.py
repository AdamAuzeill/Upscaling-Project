import torch
from torchvision import transforms
from PIL import Image
from src.CNN_model import UpscaleCNN
from src.EDSR_model import EDSR
import os

def upscale_image(model_path, image_path, output_path, model_name='cnn'):
    """
    Upscales an image using the trained model.
    """
    # Load the model
    if model_name == 'cnn':
        model = UpscaleCNN()
    elif model_name == 'edsr':
        model = EDSR()
    else:
        raise ValueError("Invalid model name. Choose 'cnn' or 'edsr'.")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Upscale the image
    with torch.no_grad():
        output_tensor = model(image_tensor)

    # Save the upscaled image
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    output_image.save(output_path)
    print(f"Upscaled image saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Upscale an image.")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('output_path', type=str, help='Path to save the upscaled image')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'edsr'], help='The model to use for upscaling.')
    args = parser.parse_args()

    model_path = f'models/upscale_{args.model}.pth'

    # Example usage:
    upscale_image(model_path, args.image_path, args.output_path, model_name=args.model)
