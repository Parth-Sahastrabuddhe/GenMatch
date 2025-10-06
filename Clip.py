from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
TF_ENABLE_ONEDNN_OPTS=0

device = "cuda" if torch.cuda.is_available() else "cpu"
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

def bounding_box(img, heatmap):

    img_copy = np.array(img).copy()
    found = False

    normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    _, binary = cv2.threshold(normalized, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        found = True
    
    return img_copy, found

# def main():

#     print("Starting the object detection process...")

#     img_path = r"C:\Users\sahas\OneDrive\Desktop\GenMatch\Photo of a dog.jpg"

#     score_patches = []
#     prompt = ["a photo of a human", "a close up of a dog's face"]

#     try:

#         # Open the image
#         img = Image.open(img_path)
#         print(f"Image opened successfully: {img_path}")

#         # Extract patches from the image
#         patches = image_patch(img)
#         print(f"Extracted {len(patches)} patches from the image.")

#         # Process all patches with the CLIP model to get the probabilities
#         patch_batch = [p for p, (x, y) in patches]
#         input = processor(text=prompt, images=patch_batch, return_tensors="pt", padding=True)
#         input = {k: v.to(device) for k, v in input.items()}
#         with torch.no_grad():
#             output = model(**input)

#         logits = output.logits_per_image
#         prob = logits.softmax(dim=1)

#         for i, (patch, (x, y)) in enumerate(patches):
#             score = prob[i][0].item()
#             score_patches.append((patch, (x, y), score))

#         # Create heatmap based on scores
#         img_h, img_w = img.size
#         pat_h, pat_w = patches[0][0].size

#         heatmap = np.zeros((img_h, img_w))

#         for _, (x, y), score in score_patches:
#             heatmap[y:y + pat_h, x:x + pat_w] += score
        
#         fig, ax = plt.subplots()
#         ax.imshow(img)
#         ax.imshow(heatmap, cmap='viridis', alpha=0.6)
#         ax.axis('off')
#         plt.show()

#         print("Genrating images with bounding box")

#         box_img = bounding_box(img, heatmap)

#         plt.imshow(box_img)
#         plt.axis('off')
#         plt.show()


#     except FileNotFoundError:
#         print(f"Error opening image: {img_path}")
#         return

# if __name__ == "__main__":
#     main()

def run_detection_pipeline(input_image, text_prompt):

    print("Starting the object detection process...")

    img = input_image
    prompt = [text_prompt, "a photo of a blank background"]
    score_patches = []
    all_scores = []
    
    patches = image_patch(img)
    print(f"Extracted {len(patches)} patches from the image.")

    patch_batch = [p for p, (x, y) in patches]
    input_data = processor(text=prompt, images=patch_batch, return_tensors="pt", padding=True)
    input_data = {k: v.to(device) for k, v in input_data.items()}
    with torch.no_grad():
        output = model(**input_data)

    logits = output.logits_per_image
    prob = logits.softmax(dim=1)

    for i, (patch, (x, y)) in enumerate(patches):
        score = prob[i][0].item()
        score_patches.append((patch, (x, y), score))
        all_scores.append(score)
    
    confidence_threshold = 0.20
    max_score = max(all_scores) if all_scores else 0
    print(f"Max confidence score: {max_score:.4f}")

    if max_score < confidence_threshold:
        msg = f"Could not find '{text_prompt}' with enough confidence."
        return msg, input_image

    img_h, img_w = img.size

    if not patches:
        print("Warning: No patches were extracted from the image.")
        return img

    pat_h, pat_w = patches[0][0].size
    heatmap = np.zeros((img_h, img_w))

    for _, (x, y), score in score_patches:
        heatmap[y:y + pat_h, x:x + pat_w] += score
    
    print("Generating image with bounding box...")
    box_img, found = bounding_box(img, heatmap)

    if not found:
        msg = "No object detected matching the prompt."
    else:
        msg = "Object detected and highlighted."

    return msg, Image.fromarray(box_img)