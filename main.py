# main.py
from inpainting.inpaint_mural import run_pipeline_for
from pathlib import Path

if __name__ == "__main__":
    for img in Path("images/raw").glob("*.*"):
        run_pipeline_for(img)
