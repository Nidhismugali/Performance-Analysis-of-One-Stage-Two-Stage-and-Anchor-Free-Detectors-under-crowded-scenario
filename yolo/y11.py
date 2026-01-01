# -----------------------
# YOLOv10 & YOLOv11 Training - Windows / Local PC
# -----------------------

import os
import torch
from ultralytics import YOLO

def train_model(model_name, weights_path, yaml_path, dataset_dir, device):
    print(f"\n🚀 Training {model_name} on dataset...\n")
    model = YOLO(weights_path)
    model.train(
        data=yaml_path,
        epochs=5,
        imgsz=640,
        batch=4,                      # adjust based on GPU memory
        device=device,
        name=f"{model_name}_crowdhuman_person_head",
        project=os.path.join(dataset_dir, "yolo_runs")
    )

def main():
    # -----------------------
    # 1️⃣ Dataset paths
    # -----------------------
    dataset_dir = "C:/Users/mayan/Downloads/crowdhuman.v2i.yolov9"  # update if different
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    # -----------------------
    # 2️⃣ Create/overwrite data.yaml
    # -----------------------
    yaml_content = f"""
train: {dataset_dir}/train/images
val: {dataset_dir}/valid/images
test: {dataset_dir}/test/images

nc: 2
names:
  0: head
  1: person
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # -----------------------
    # 3️⃣ Device selection
    # -----------------------
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # -----------------------
    # 4️⃣ Train YOLOv10n
    # -----------------------
    train_model(
        model_name="yolov10n",
        weights_path="yolov10n.pt",
        yaml_path=yaml_path,
        dataset_dir=dataset_dir,
        device=device
    )

    # -----------------------
    # 5️⃣ Train YOLOv11n
    # -----------------------
    train_model(
        model_name="yolov11n",
        weights_path="yolo11n.pt",
        yaml_path=yaml_path,
        dataset_dir=dataset_dir,
        device=device
    )

    print("\n✅ Training complete for both YOLOv10n and YOLOv11n!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Windows safe
    main()
