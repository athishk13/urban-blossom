import yaml
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset

with open('data/graffiti.yaml') as f:
    data_dict = yaml.safe_load(f)

train_ds = YOLODataset(
    path       = data_dict['train'],   
    imgsz      = 640,
    batch_size = 4,
    augment    = False,
    rect       = False,
    cache      = False,
    single_cls = False,
    stride     = 32,
    pad        = 0.0,
    prefix     = '',
    shuffle    = False,
    data       = data_dict          
)

loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)

#Initialize model & trainer
model   = YOLO('yolov8n.pt')
trainer = model._smart_load('trainer')(model=model, args=model.args)

#Pull one batch and run 5 manual training steps
imgs, labels, _, _ = next(iter(loader))
for step in range(5):
    losses = trainer.run_step(imgs, labels)
    print(f"step {step} → box={losses['box']:.4f}, "
          f"cls={losses['cls']:.4f}, dfl={losses['dfl']:.4f}")
