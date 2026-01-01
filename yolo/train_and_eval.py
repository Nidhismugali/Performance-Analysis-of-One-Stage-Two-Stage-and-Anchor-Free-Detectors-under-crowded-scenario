# --------------------------------------------------------
# YOLOv10n & YOLOv11n Training + Automatic Evaluation
# Organized Runs & Colored Metric Comparison
# --------------------------------------------------------

import os
import torch
import json
from ultralytics import YOLO
from colorama import Fore, Style, init
from tabulate import tabulate
import shutil

init(autoreset=True)


def train_and_save(model_name, weights_path, yaml_path, save_dir, device):
    """Train a YOLO model and save its best weights with a custom name"""
    print(f"\n🚀 Training {Fore.CYAN}{model_name}{Style.RESET_ALL} ...\n")

    model = YOLO(weights_path)
    results = model.train(
        data=yaml_path,
        epochs=2,               # reduce for quick test
        imgsz=640,
        batch=4,
        device=device,
        name=model_name,
        project=save_dir,
    )

    # Path to best weights from Ultralytics run
    best_path = os.path.join(save_dir, model_name, "weights", "best.pt")

    # Create custom-named copies
    custom_best = os.path.join(save_dir, f"best{'10' if '10' in model_name else '11'}.pt")
    if os.path.exists(best_path):
        shutil.copy(best_path, custom_best)
        print(f"{Fore.GREEN}✅ Best weights saved as: {custom_best}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}⚠️ Could not find best.pt at {best_path}{Style.RESET_ALL}")

    return custom_best


def evaluate_model(model_name, weights_path, yaml_path, device):
    """Evaluate a trained YOLO model and extract key metrics"""
    print(f"\n📊 Evaluating {Fore.CYAN}{model_name}{Style.RESET_ALL} ({weights_path}) ...\n")

    model = YOLO(weights_path)
    results = model.val(data=yaml_path, device=device)

    metrics_dict = results.results_dict

    precision = metrics_dict.get("metrics/precision(B)", 0)
    recall = metrics_dict.get("metrics/recall(B)", 0)
    map50 = metrics_dict.get("metrics/mAP50(B)", 0)
    map95 = metrics_dict.get("metrics/mAP50-95(B)", 0)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    return {
        "Precision": precision,
        "Recall": recall,
        "mAP50": map50,
        "mAP50-95": map95,
        "F1 Score": f1,
    }


def compare_metrics(metrics_v10, metrics_v11):
    """Print a colored table comparing YOLOv10n and YOLOv11n metrics"""
    headers = ["Metric", "YOLOv10n", "YOLOv11n", "Better Model"]
    table = []

    for key in metrics_v10.keys():
        v10 = metrics_v10[key]
        v11 = metrics_v11[key]
        if v10 > v11:
            better = Fore.GREEN + "YOLOv10n"
            v10_col, v11_col = Fore.GREEN, Fore.RED
        elif v11 > v10:
            better = Fore.GREEN + "YOLOv11n"
            v10_col, v11_col = Fore.RED, Fore.GREEN
        else:
            better = Fore.YELLOW + "Equal"
            v10_col = v11_col = Fore.YELLOW

        table.append([key, f"{v10_col}{v10:.4f}", f"{v11_col}{v11:.4f}", better])

    print("\n" + Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "📈 YOLOv10n vs YOLOv11n Metrics Comparison")
    print("=" * 60 + "\n")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    os.makedirs("yolo_runs", exist_ok=True)
    with open("yolo_runs/metrics_comparison.json", "w") as f:
        json.dump({"YOLOv10n": metrics_v10, "YOLOv11n": metrics_v11}, f, indent=4)
    print(f"\n💾 Metrics saved at: {Fore.YELLOW}yolo_runs/metrics_comparison.json{Style.RESET_ALL}\n")


def main():
    # Dataset setup
    base_dir = os.getcwd()  # current working dir (C:\PROJECTS\yolov9)
    dataset_dir = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9"
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    yaml_content = f"""
train: {dataset_dir}/train/images
val: {dataset_dir}/valid/images
test: {dataset_dir}/test/images

nc: 2
names:
  0: head
  1: person
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n🧠 Using device: {Fore.YELLOW}{device}{Style.RESET_ALL}")

    # Separate directories for each version
    yolo10_dir = os.path.join(base_dir, "YOLOv10")
    yolo11_dir = os.path.join(base_dir, "YOLOv11")
    os.makedirs(yolo10_dir, exist_ok=True)
    os.makedirs(yolo11_dir, exist_ok=True)

    # Train both models
    best10 = train_and_save("yolov10n", "yolov10n.pt", yaml_path, yolo10_dir, device)
    best11 = train_and_save("yolov11n", "yolo11n.pt", yaml_path, yolo11_dir, device)

    # Evaluate both
    metrics_v10 = evaluate_model("YOLOv10n", best10, yaml_path, device)
    metrics_v11 = evaluate_model("YOLOv11n", best11, yaml_path, device)

    # Compare
    compare_metrics(metrics_v10, metrics_v11)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
