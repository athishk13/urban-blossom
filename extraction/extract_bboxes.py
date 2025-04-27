
import cv2
import numpy as np
from pathlib import Path

def make_mask(image_path, boxes, out_path):
    """
    Given an image and a list of boxes [(x1,y1,x2,y2),…],
    writes a mask PNG where box areas are white and the rest is black.
    """
    img = cv2.imread(str(Path(image_path)))
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)
    return str(out_path)
