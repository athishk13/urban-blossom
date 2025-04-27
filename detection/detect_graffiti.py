# detection/detect_graffiti.py

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

MODEL_PATH = Path('runs/train_graffiti/exp50_tuned_v3/weights/best.pt')
model = YOLO(str(MODEL_PATH))


def detect(image_path, conf_threshold=0.05):
    """
    Runs YOLOv8 inference on `image_path` and returns a list of
    (x1, y1, x2, y2) tuples above the confidence threshold.
    """
    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def draw_boxes(image_path, boxes, out_path):
    """
    Draws green rectangles over each box on the image and saves it to `out_path`.
    """
    img = cv2.imread(str(image_path))
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), img)
    return str(out_file)


def create_mask(image_path, boxes, mask_path):
    """
    Creates a binary mask where graffiti boxes are white (255) on black (0).
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
    mask_file = Path(mask_path)
    mask_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_file), mask)
    return str(mask_file)


if __name__ == "__main__":
    RAW_DIR   = Path("images/raw")
    OUT_DIR   = Path("images/outputs")
    MASKS_DIR = Path("images/masks")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for img_path in RAW_DIR.iterdir():
        if img_path.name.startswith(".") or img_path.suffix.lower() not in VALID_EXTS:
            continue

        print(f"Processing {img_path.name}…")
        boxes = detect(img_path)
        print(" Detected boxes:", boxes)

        debug_file = OUT_DIR / f"{img_path.stem}_boxes.png"
        draw_boxes(img_path, boxes, debug_file)
        print(f" Debug image saved to: {debug_file}")

        mask_file = MASKS_DIR / f"{img_path.stem}_mask.png"
        create_mask(img_path, boxes, mask_file)
        print(f" Mask image saved to:  {mask_file}\n")