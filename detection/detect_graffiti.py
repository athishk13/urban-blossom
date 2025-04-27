# detection/detect_graffiti.py
from ultralytics import YOLO
from pathlib import Path

# (1) Load a pretrained or your fine-tuned weights
model = YOLO('yolov8n.pt')  # or 'path/to/your-finetuned.pt'

def detect(image_path, conf_threshold=0.25):
    """
    Runs YOLOv8 inference on image_path.
    Returns a list of (x1, y1, x2, y2) integers for detections above conf_threshold.
    """
    img = str(Path(image_path))
    results = model.predict(source=img, conf=conf_threshold, verbose=False)[0]
    
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

if __name__ == "__main__":
    # quick smoke test
    test_img = "images/raw/ketchup-graffiti.webp"
    print("Detected boxes:", detect(test_img))
