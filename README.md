# 💡 GenMatch: Open-Vocabulary Object Detector

A proof-of-concept implementation of an open-vocabulary object detector that can locate any object in an image using natural language descriptions, powered by OpenAI's CLIP model.

---

## ## Overview

Traditional object detectors are limited to a fixed set of pre-defined categories they were trained on (e.g., "cat," "dog," "car"). This project explores a modern, "zero-shot" approach where no re-training is needed to find new objects. You can simply provide an image and a text prompt, such as "a photo of a red fire hydrant," and the model will attempt to locate it.

This repository contains the implementation for my personal project aimed at exploring the intersection of vision-language models and classic computer vision tasks.

---

## ## Features

* ✅ **Open-Vocabulary Detection:** Locate any object that can be described with text, not just a fixed list of classes.
* ✅ **Zero-Shot:** Requires no fine-tuning or re-training to find new types of objects.
* ✅ **Visual Heatmap:** Generates an intuitive heatmap to show the most probable location of the object.
* ✅ **Bounding Box Generation:** Processes the heatmap to draw a final bounding box around the detected object.

---

## ## How It Works

This initial version (V1.0) uses a patch-based scoring method with a pre-trained CLIP model.

1.  **Image Patching:** The input image is broken down into a grid of smaller, overlapping patches using a sliding window.
2.  **CLIP Scoring:** Each individual patch is passed through the CLIP model along with the user's text prompt. CLIP calculates a similarity score indicating how well the content of the patch matches the text description.
3.  **Heatmap Generation:** The scores for all patches are projected onto a blank canvas, creating a heatmap where the brightest areas correspond to the patches with the highest similarity scores.
4.  **Bounding Box Extraction:** The final heatmap is processed to find the area of highest intensity, and a bounding box is drawn around it.



---

## ## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [Link to Your GitHub Repo]
    cd [Your-Repo-Name]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ## Usage

To run the detector, use the `Clip.py` script from the command line.

```bash
python Clip.py --image "path/to/your/image.jpg" --prompt "a photo of a dog"
