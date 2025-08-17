from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def image_patch(img, patch_size =(100, 100), stride = 50):

    img_w, img_h = img.size
    print(f"Image dimensions: width={img_w}, height={img_h}")
    patches = []

    for i in range(0, img_h - patch_size[1] + 1, stride):
        for j in range(0, img_w - patch_size[0] + 1, stride):
            patch = img.crop((j, i, j + patch_size[0], i + patch_size[1]))
            patches.append((patch, (j, i)))
    
    return patches

def main():

    print("Starting the object detection process...")

    img_path = r"C:\Users\sahas\OneDrive\Desktop\GenMatch\Photo of a dog.jpg"

    try:

        img = Image.open(img_path)
        print(f"Image opened successfully: {img_path}")

        patches = image_patch(img)
        print(f"Extracted {len(patches)} patches from the image.")

    except FileNotFoundError:
        print(f"Error opening image: {img_path}")
        return

if main() == "__main__":
    main()