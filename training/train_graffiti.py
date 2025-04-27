from ultralytics import YOLO
from pathlib import Path

def train_graffiti(
    base_model: str = 'yolov8n.pt',
    data_yaml:   str = None,
    epochs:      int = 50,
    patience:    int = 20,
    imgsz:       int = 640,
    batch:       int = 8,
    optimizer:   str = 'Adam',     
    lr0:         float = 1e-4,        # lower initial LR
    lrf:         float = 0.05,        # sharper decay to min LR
    momentum:    float = 0.937,
    weight_decay:float = 1e-3,        # stronger regularization
    warmup_epochs: float = 5.0,       # longer warm-up
    freeze:      int = 10,            # freeze backbone for first epochs
    clip:        bool = True,         
    clip_norm:   float = 5.0,         
    project:     str = 'runs/train_graffiti',
    name:        str = 'exp50_tuned_v4'
):
    # Computing absolute path to data/graffiti.yaml
    BASE = Path(__file__).resolve().parent.parent
    data_yaml = str(BASE / 'data' / 'graffiti.yaml')

    # Initialize model
    model = YOLO(base_model)

    #  revised hyperparameters
    model.train(
        data=data_yaml,
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        freeze=freeze,
        clip=clip,
        clip_norm=clip_norm,

        augment=False, #stability
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0,

        project=project,
        name=name
    )

    print(f"🚀 Training complete! Weights saved to {project}/{name}/weights/best.pt")

if __name__ == '__main__':
    train_graffiti()