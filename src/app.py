import argparse
from src.image_utils import downscale_image, blur_image
from src.train import train_model
from src.upscale import upscale_image
from PIL import Image
import os

def compare_images(image_path1, image_path2):
    """
    Compares two images and shows them side-by-side.
    """
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    dst = Image.new('RGB', (img1.width + img2.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    dst.show()

def main():
    parser = argparse.ArgumentParser(description="Image Upscaling and Processing Tool")
    subparsers = parser.add_subparsers(dest='command')

    # Train command
    parser_train = subparsers.add_parser('train', help='Train the upscaling model')
    parser_train.add_argument('dataset_path', type=str, help='Path to the training dataset')
    parser_train.add_argument('--model', type=str, default='cnn', choices=['cnn', 'edsr'], help='The model to train.')
    parser_train.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser_train.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Upscale command
    parser_upscale = subparsers.add_parser('upscale', help='Upscale an image')
    parser_upscale.add_argument('image_path', type=str, help='Path to the input image')
    parser_upscale.add_argument('output_path', type=str, help='Path to save the upscaled image')
    parser_upscale.add_argument('--model', type=str, default='cnn', choices=['cnn', 'edsr'], help='The model to use for upscaling.')

    # Downscale command
    parser_downscale = subparsers.add_parser('downscale', help='Downscale an image')
    parser_downscale.add_argument('image_path', type=str, help='Path to the input image')
    parser_downscale.add_argument('output_path', type=str, help='Path to save the downscaled image')
    parser_downscale.add_argument('--size', type=int, nargs=2, default=[128, 128], help='Target size (width, height)')

    # Blur command
    parser_blur = subparsers.add_parser('blur', help='Blur an image')
    parser_blur.add_argument('image_path', type=str, help='Path to the input image')
    parser_blur.add_argument('output_path', type=str, help='Path to save the blurred image')
    parser_blur.add_argument('--radius', type=int, default=2, help='Blur radius')

    # Compare command
    parser_compare = subparsers.add_parser('compare', help='Compare two images')
    parser_compare.add_argument('image1_path', type=str, help='Path to the first image')
    parser_compare.add_argument('image2_path', type=str, help='Path to the second image')

    args = parser.parse_args()

    if args.command == 'train':
        model_save_path = f'models/upscale_{args.model}.pth'
        train_model(args.dataset_path, model_save_path, model_name=args.model, num_epochs=args.epochs, learning_rate=args.lr)
    elif args.command == 'upscale':
        model_path = f'models/upscale_{args.model}.pth'
        upscale_image(model_path, args.image_path, args.output_path, model_name=args.model)
    elif args.command == 'downscale':
        downscale_image(args.image_path, args.output_path, tuple(args.size))
    elif args.command == 'blur':
        blur_image(args.image_path, args.output_path, args.radius)
    elif args.command == 'compare':
        compare_images(args.image1_path, args.image2_path)
    else:
        parser.print_help()

if __name__ == '__main__':
    # Create dummy directories and files for demonstration if they don't exist
    if not os.path.exists('data/dummy_images'):
        os.makedirs('data/dummy_images')
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')

    main()
