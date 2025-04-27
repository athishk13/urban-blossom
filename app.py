import os
import io
import base64
import requests
import gradio as gr
from uuid import uuid4
from pathlib import Path
from PIL import Image
from detection.detect_graffiti import detect, create_mask

SD_API_URL = "https://stablediffusionapi.com/api/v3/inpaint"
SD_API_KEY = os.getenv("SD_API_KEY") 

def encode_image_to_base64(path: Path) -> str:
    """Read an image file and return its base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def generate_murals(image: Image.Image, prompt: str):
    upload_path = Path("temp_upload.jpg")
    image.save(upload_path)

    boxes = detect(upload_path)
    mask_path = Path("temp_mask.png")
    create_mask(upload_path, boxes, mask_path)

    payload = {
        "key":         SD_API_KEY,
        "init_image":  encode_image_to_base64(upload_path),
        "mask_image":  encode_image_to_base64(mask_path),
        "prompt":      prompt,
        "n_samples":   4
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(SD_API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    outputs_b64 = resp.json()["output"] 

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_paths = []
    for b64_str in outputs_b64:
        data = base64.b64decode(b64_str)
        fn = out_dir / f"mural_{uuid4().hex}.png"
        fn.write_bytes(data)
        out_paths.append(str(fn))

    return out_paths


# ── Gradio UI ────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Graffiti Mural Generator")
    with gr.Row():
        with gr.Column():
            upload_img = gr.Image(type="pil", label="Upload Graffiti Wall")
        with gr.Column():
            theme   = gr.Textbox(label="Mural Theme", placeholder="e.g. floral, abstract")
            gen_btn = gr.Button("Generate Murals", variant="primary")

    gallery  = gr.Gallery(label="Generated Murals", columns=4, show_label=False)
    expanded = gr.Image(label="Click to Expand")

    gen_btn.click(fn=generate_murals, inputs=[upload_img, theme], outputs=gallery)
    gallery.select(fn=lambda evt: evt.value, outputs=expanded)

if __name__ == "__main__":
    demo.launch()