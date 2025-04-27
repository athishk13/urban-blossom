
import json
import shutil
from pathlib import Path

JSON_DIR    = Path("labeled_graffiti_training_data")  # LabelMe JSONs
IMG_DIR     = Path("BIGgraffiti_images")         

# YOLO expects images + labels
YOLO_DS     = Path("images/yolo_dataset")
YOLO_IMGS   = YOLO_DS / "images"
YOLO_LABELS = YOLO_DS / "labels"
YOLO_IMGS.mkdir(parents=True, exist_ok=True)
YOLO_LABELS.mkdir(parents=True, exist_ok=True)

# Define labelme groupID
CLASSES = ["graffiti-art"]

def labelme_to_yolo(json_path: Path) -> str:
    """
    Reads a LabelMe .json annotation and converts it to YOLO format:
    <class_id> <x_center> <y_center> <width> <height>
    All coordinates normalized to [0,1].
    """
    data = json.loads(json_path.read_text())
    iw, ih = data["imageWidth"], data["imageHeight"]
    lines = []

    for shape in data["shapes"]:
        if shape["shape_type"] != "rectangle":
            continue
        cls_id = CLASSES.index(shape["label"])
        (x1, y1), (x2, y2) = shape["points"]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        xc = ((x_min + x_max) / 2) / iw
        yc = ((y_min + y_max) / 2) / ih
        w  = (x_max - x_min) / iw
        h  = (y_max - y_min) / ih

        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)

def main():
    for json_file in JSON_DIR.glob("*.json"):
        stem = json_file.stem

        candidates = [IMG_DIR / f"{stem}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")]
        src_img = next((p for p in candidates if p.exists()), None)
        if not src_img:
            print(f"⚠️  No image for {json_file.name}, tried: {[p.name for p in candidates]}")
            continue

        dst_img = YOLO_IMGS / src_img.name
        shutil.copy(src_img, dst_img)

        yolo_txt = labelme_to_yolo(json_file)
        (YOLO_LABELS / f"{stem}.txt").write_text(yolo_txt)

        print(f"✅  {src_img.name} → {stem}.txt")

if __name__ == "__main__":
    main()
