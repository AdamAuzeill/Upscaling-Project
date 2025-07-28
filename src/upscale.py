import torch
from torchvision import transforms
from PIL import Image
from src.train import UpscaleCNN
import os

def upscale_image(model_path, image_path, output_path):
    """
    Upscales an image using the trained model.
    """
    # Load the model
    model = UpscaleCNN()
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
    # Example usage:
    # Ensure a dummy model and image exist for demonstration
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('data/dummy_images'):
        os.makedirs('data/dummy_images')
    if not os.path.exists('results'):
        os.makedirs('results')

    # Create a dummy model and image if they don't exist
    if not os.path.exists('models/upscale_cnn.pth'):
        # Create and save a dummy model
        dummy_model = UpscaleCNN()
        torch.save(dummy_model.state_dict(), 'models/upscale_cnn.pth')

    if not os.path.exists('data/dummy_images/cat_to_upscale.png'):
        dummy_image = Image.new('RGB', (64, 64), color = 'blue')
        dummy_image.save('data/dummy_images/cat_to_upscale.png')

    upscale_image('models/upscale_cnn.pth', 'data/dummy_images/cat_to_upscale.png', 'results/upscaled_cat.png')
