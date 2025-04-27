# inpainting/stable_diffusion_client.py
import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()                                #loading SD key from .env
API_KEY      = os.getenv("SD_API_KEY")
API_URL      = "https://stablediffusionapi.com/api/v3/inpaint"

def inpaint(init_image_path, mask_image_path, prompt, 
            width=512, height=512, steps=30, guidance=7.5, strength=0.7):
    # 1. Read & encode
    init_b64 = base64.b64encode(Path(init_image_path).read_bytes()).decode()
    mask_b64 = base64.b64encode(Path(mask_image_path).read_bytes()).decode()

    # 2. Build payload
    payload = {
      "key": API_KEY,
      "prompt": prompt,
      "init_image": init_b64,
      "mask_image": mask_b64,
      "width": str(width),
      "height": str(height),
      "samples": "1",
      "num_inference_steps": str(steps),
      "guidance_scale": guidance,
      "strength": strength,
      "base64": "yes"
    }

    # 3. Call API
    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # 4. Decode & save
    out_data = base64.b64decode(data["output"][0])
    out_path = Path("images/outputs") / f"{Path(init_image_path).stem}_mural.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out_data)
    return str(out_path)
