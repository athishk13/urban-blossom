# quick_eval.py
from ultralytics import YOLO

model = YOLO('runs/train_graffiti/exp50_tuned_v3/weights/best.pt', task='detect')

results = model.val(
    data={
      'train': 'images/train',
      'val':   'images/val',
      'nc':     1,
      'names': ['graffiti']
    },
    verbose=True
)

p, r, m50, m5095 = results.box.stats[-1]
print(f"Precision:   {p:.3f}")
print(f"Recall:      {r:.3f}")
print(f"mAP@0.5:     {m50:.3f}")
print(f"mAP@0.5–0.95:{m5095:.3f}")
