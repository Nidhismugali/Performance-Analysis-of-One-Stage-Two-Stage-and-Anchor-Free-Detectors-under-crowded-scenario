from ultralytics import YOLO
import torch
import os
from tabulate import tabulate
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)  # initialize colorama

def show_metrics_and_curves(metrics, run_dir):
    # 1️⃣ Overall metrics
    overall_table = [
        ["Precision", float(np.mean(metrics.box.p))],
        ["Recall", float(np.mean(metrics.box.r))],
        ["mAP50", float(np.mean(metrics.box.map50))],
        ["mAP50-95", float(np.mean(metrics.box.map))],
        ["F1 Score", float(np.mean(metrics.box.f1)) if hasattr(metrics.box, 'f1') else
         2*float(np.mean(metrics.box.p))*float(np.mean(metrics.box.r))/(float(np.mean(metrics.box.p))+float(np.mean(metrics.box.r))+1e-9)],
    ]
    print(Fore.CYAN + "\n📊 Validation Metrics (Overall):")
    print(tabulate(overall_table, headers=["Metric", "Value"], floatfmt=".4f"))

    # 2️⃣ Per-class metrics
    print(Fore.MAGENTA + "\n🔎 Per-Class Results:")
    class_names = metrics.names if hasattr(metrics, "names") else {0: "class0"}
    per_class_table = []

    def safe_get(val, i):
        if isinstance(val, (float, np.float32, np.float64)):
            return float(val)
        elif hasattr(val, "__len__") and i < len(val):
            return float(val[i])
        else:
            return 0.0

    for i, cls_name in class_names.items():
        p      = safe_get(metrics.box.p, i)
        r      = safe_get(metrics.box.r, i)
        m50    = safe_get(metrics.box.map50, i)
        m5095  = safe_get(metrics.box.map, i)
        f1     = 2*p*r/(p+r+1e-9)
        per_class_table.append([cls_name, p, r, m50, m5095, f1])

    print(tabulate(per_class_table, headers=["Class", "Precision", "Recall", "mAP50", "mAP50-95", "F1"], floatfmt=".4f"))

    # 3️⃣ Curves/images in the run folder
    print(Fore.YELLOW + "\n📈 Curves & Images in this run folder:")
    curve_folder = run_dir
    curve_files = sorted([f for f in os.listdir(curve_folder) if f.lower().endswith((".jpg", ".png"))])
    if curve_files:
        curve_table = [[f] for f in curve_files]
        print(tabulate(curve_table, headers=["File"], tablefmt="fancy_grid"))
    else:
        print("No curve images found in this run folder.")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(Fore.GREEN + f"Using device: {device}")

    # Auto pick latest run folder
    base_path = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9\yolo_runs"
    latest_run = os.path.join(base_path, "crowdhuman_person_head6")
    weights = os.path.join(latest_run, "weights", "best.pt")

    if not os.path.exists(weights):
        raise FileNotFoundError(f"❌ Weights not found: {weights}")

    print(Fore.GREEN + f"✅ Using weights: {weights}")
    model = YOLO(weights)

    yaml_path = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9\data.yaml"

    # Run validation
    metrics = model.val(data=yaml_path, device=device, workers=0)

    # Show metrics and curves
    show_metrics_and_curves(metrics, latest_run)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
