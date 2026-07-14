import torch
import os
from datetime import datetime

torch.set_float32_matmul_precision('high')
from ultralytics import RTDETR


MODEL_TYPE = "GMB-DETR"
DATASET_NAME = "RSOD"
TRAIN_MODE = "Scratch"
BATCH_SIZE = 8
EPOCHS = 350

model_yaml_path = "/home/a/projects/RT-DETR/ultralytics-main/fdconv/fdconv-yaml/GMB-DETR.yaml"
data_yaml_path = "/home/a/projects/RT-DETR/ultralytics-main/datasets/RSOD.v3i.yolo26/data.yaml"
PROJECT_DIR = "/home/a/projects/RT-DETR/ultralytics-main/fdconv/Train_results/Train_RSOD"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{MODEL_TYPE}_{DATASET_NAME}_{TRAIN_MODE}_bs{BATCH_SIZE}_ep{EPOCHS}_{timestamp}"
    final_save_dir = os.path.join(PROJECT_DIR, experiment_name)

    model = RTDETR(model_yaml_path)

    results = model.train(

        data=data_yaml_path,  #
        imgsz=640,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=0,
        workers=8,


        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        cos_lr=True,


        warmup_epochs=10.0,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,


        amp=False,
        patience=0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        seed=0,
        deterministic=True,


        mosaic=1.0,
        close_mosaic=30,
        copy_paste=0.0,
        mixup=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        auto_augment="randaugment",
        erasing=0.4,


        project=PROJECT_DIR,
        name=experiment_name,
        plots=True,
        save=True,
        save_period=-1
    )


if __name__ == '__main__':
    main()
