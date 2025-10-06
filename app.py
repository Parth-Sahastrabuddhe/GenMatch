import gradio as gr
from Clip import run_detection_pipeline

print("Loading the application...")

iface = gr.Interface(
    fn=run_detection_pipeline,
    inputs = [
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Text Prompt", placeholder="e.g., a photo of a dog's face")
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Image(type="pil", label="Detection Result")
        ],
    title="GenMatch: Open-Vocabulary Object Detector",
    description="Upload an image and type what you want to find. The model will draw a box around it.",
)

# Launch the app
if __name__ == "__main__":
    iface.launch(debug=True)