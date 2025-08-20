from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
TF_ENABLE_ONEDNN_OPTS=0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# This function extracts patches from an image and returns them along with their coordinates.
def image_patch(img, patch_size =(100, 100), stride = 2):

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

    score_patches = []
    prompt = ["a photo of a human", "a photo of a dog"]

    try:

        # Open the image
        img = Image.open(img_path)
        print(f"Image opened successfully: {img_path}")

        # Extract patches from the image
        patches = image_patch(img)
        print(f"Extracted {len(patches)} patches from the image.")

        # Process all patches with the CLIP model to get the probabilities
        patch_batch = [p for p, (x, y) in patches]
        input = processor(text=prompt, images=patch_batch, return_tensors="pt", padding=True)
        input = {k: v.to(device) for k, v in input.items()}
        with torch.no_grad():
            output = model(**input)

        logits = output.logits_per_image
        prob = logits.softmax(dim=1)

        for i, (patch, (x, y)) in enumerate(patches):
            score = prob[i][0].item()
            score_patches.append((patch, (x, y), score))

        # Create heatmap based on scores
        img_h, img_w = img.size
        pat_h, pat_w = patches[0][0].size

        heatmap = np.zeros((img_h, img_w))

        for _, (x, y), score in score_patches:
            heatmap[y:y + pat_h, x:x + pat_w] += score
        
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(heatmap, cmap='viridis', alpha=0.6)
        ax.axis('off')
        plt.show()


    except FileNotFoundError:
        print(f"Error opening image: {img_path}")
        return

if main() == "__main__":
    main()