from PIL import Image, ImageFilter

def downscale_image(image_path, output_path, size):
    """
    Downscales an image to a given size.
    """
    with Image.open(image_path) as img:
        img.thumbnail(size)
        img.save(output_path)

def blur_image(image_path, output_path, radius=2):
    """
    Applies a Gaussian blur to an image.
    """
    with Image.open(image_path) as img:
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
        blurred_img.save(output_path)
