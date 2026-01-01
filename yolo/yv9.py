# -----------------------
# YOLOv10 Training Only - Windows / Local PC
# -----------------------

import os
import torch
from ultralytics import YOLO

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
    # 4️⃣ Load YOLOv10 pre-trained model
    # -----------------------
    # Automatically downloads yolov10n.pt if not present
    model = YOLO("yolov10n.pt")

    # -----------------------
    # 5️⃣ Train on dataset
    # -----------------------
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=4,                 # adjust based on GPU memory
        device=device,           # dynamic device
        name="crowdhuman_person_head",
        project=os.path.join(dataset_dir, "yolo_runs")
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Windows safe
    main()
