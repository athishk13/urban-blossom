import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()                               
API_KEY      = os.getenv("SD_API_KEY")
ENDPOINT = "https://stablediffusionapi.com/api/v3/inpaint"

PROMPT   = ("A colorful, detailed urban mural painted realistically on a "
            "concrete wall, vibrant colors, street art style.")


def encode_image_to_base64(path: Path) -> str:
    """Load an image file and return a data-URI style base64 string."""
    img_bytes = path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def inpaint(
    image_path: Path,
    mask_path:  Path,
    prompt:     str,
    out_path:   Path
) -> Path:
    """Call Stable Diffusion inpaint endpoint and write the returned image."""
    payload = {
        "key":    API_KEY,
        "image":  encode_image_to_base64(image_path),
        "mask":   encode_image_to_base64(mask_path),
        "prompt": prompt,
    }

    resp = requests.post(ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()

    img_b64 = data["output"][0]["base64"]
    img_bytes = base64.b64decode(img_b64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(img_bytes)
    return out_path

if __name__ == "__main__":
    from detection.detect_graffiti import detect, create_mask

    RAW_DIR   = Path("images/raw")
    MASKS_DIR = Path("images/masks")
    OUT_DIR   = Path("images/inpainted")

    for img_path in RAW_DIR.glob("*.jpg"):
        print(f"\n→ Processing {img_path.name}")

        boxes     = detect(img_path)
        mask_path = MASKS_DIR / f"{img_path.stem}_mask.png"
        create_mask(img_path, boxes, mask_path)

        out_file = OUT_DIR / f"{img_path.stem}_inpainted.png"
        result   = inpaint(img_path, mask_path, PROMPT, out_file)
        print(f"✔️  Inpainted image saved to {result}")