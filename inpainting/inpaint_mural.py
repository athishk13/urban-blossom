from pathlib import Path
from extraction.extract_bboxes import make_mask
from detection.detect_graffiti import detect
from inpainting.stable_diffusion_client import inpaint


def run_pipeline_for(image_path):
    boxes = detect(str(image_path))
    if not boxes:
        print(f"No graffiti detected in {image_path.name}, skipping.")
        return

    mask_path = Path("images/masks") / f"{image_path.stem}_mask.png"
    make_mask(str(image_path), boxes, str(mask_path))
    print(f" Mask saved → {mask_path}")

    out_file = inpaint(str(image_path), str(mask_path), prompt) 
    print(f" Mural output → {out_file}")

if __name__ == "__main__":
    raw_dir = Path("images/raw")
    default_prompt = (
        "A colorful, detailed urban mural painted realistically on a concrete wall, "
        "vibrant colors, street art style."
    )
    for img in raw_dir.glob("*.*"):
        print(f"\nProcessing {img.name} …")
        run_pipeline_for(img, default_prompt)