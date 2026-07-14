import torch
import os
from datetime import datetime
import torchvision

# ==========================================

# ==========================================
original_nms = torchvision.ops.nms

def safe_cpu_nms(boxes, scores, iou_threshold):
    device = boxes.device
    keep = original_nms(
        boxes.detach().cpu(),
        scores.detach().cpu(),
        iou_threshold
    )
    return keep.to(device)

torchvision.ops.nms = safe_cpu_nms

# ==========================================

# ==========================================
torch.set_float32_matmul_precision('high')

from ultralytics import RTDETR


# ==========================================

# ==========================================
MODEL_TYPE = "GMB-DETR"
DATASET_NAME = "Visdrone"
BATCH_SIZE = 16


WEIGHT_PATH = "/home/a/projects/RT-DETR/ultralytics-main/fdconv/Train_results/Train_Visdrone/RT-DETR-ResNet101_Visdrone_Scratch_bs2_ep350_20260521_184253/weights/best.pt"
DATA_YAML_PATH = "/home/a/projects/RT-DETR/ultralytics-main/datasets/GMB-DETR-Visdrone2019/GMB-DETR-Visdrone2019.yaml"
SPLIT = "test"
PROJECT_DIR = "/home/a/projects/RT-DETR/ultralytics-main/fdconv/Test_Results/Test_VisDrone"

# ==========================================

# ==========================================
def safe_get(arr, i):
    try:
        return float(arr[i])
    except:
        return 0.0

# ==========================================

# ==========================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Eval_{MODEL_TYPE}_{DATASET_NAME}_{SPLIT}_bs{BATCH_SIZE}_{timestamp}"

    print("=" * 60)
    print("🚀 启动测试（支持完整 COCO 指标输出）")
    print("=" * 60)

    model = RTDETR(WEIGHT_PATH)

    metrics = model.val(
        data=DATA_YAML_PATH,
        split=SPLIT,
        imgsz=640,
        batch=BATCH_SIZE,
        device=0,
        workers=8,
        half=False,
        plots=True,
        save=True,
        save_json=True,
        save_txt=True,
        save_conf=True,
        conf=0.001,
        iou=0.6,
        project=PROJECT_DIR,
        name=experiment_name
    )

    # ==========================================

    # ==========================================
    print("\n📊 COCO-style 全局指标：")
    print(f"mAP@50:        {metrics.box.map50:.4f}")
    print(f"mAP@50-95:     {metrics.box.map:.4f}")


    if hasattr(metrics.box, "map75"):
        print(f"AP@75:         {metrics.box.map75:.4f}")


    if hasattr(metrics.box, "map_small"):
        print(f"AP small (APs): {metrics.box.map_small:.4f}")
        print(f"AP medium(APm): {metrics.box.map_medium:.4f}")
        print(f"AP large (APl): {metrics.box.map_large:.4f}")
    else:
        print("⚠️ 当前 Ultralytics 版本未直接提供 APs/APm/APl（需要 COCO API）")

    # ==========================================

    # ==========================================
    print("\n📋 每类别详细指标：")
    print(f"{'Class':20s} {'P':>8} {'R':>8} {'mAP50':>10} {'mAP50-95':>12}")

    names = metrics.names


    if isinstance(names, dict):
        name_list = [names[i] for i in range(len(names))]
    else:
        name_list = names

    for i, name in enumerate(name_list):
        p = safe_get(metrics.box.p, i)
        r = safe_get(metrics.box.r, i)
        ap50 = safe_get(metrics.box.ap50, i)
        ap = safe_get(metrics.box.ap, i)

        print(f"{name:20s} {p:8.3f} {r:8.3f} {ap50:10.3f} {ap:12.3f}")

    print("\n✅ 测试完成！")

# ==========================================

# ==========================================
if __name__ == '__main__':
    main()
