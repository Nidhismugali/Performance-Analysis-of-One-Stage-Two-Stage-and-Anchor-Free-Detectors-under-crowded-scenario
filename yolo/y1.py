from ultralytics import YOLO
import torch
import os
from tabulate import tabulate
import numpy as np
from colorama import Fore, Style

def safe_get(arr, idx):
    """Safely get value from array-like or scalar."""
    try:
        if hasattr(arr, '__getitem__'):
            return float(arr[idx])
        else:
            return float(arr)
    except:
        return 0.0

def color_metric(val, best_val, worst_val):
    """Return colored string based on best/worst metric."""
    if val == best_val:
        return f"{Style.BRIGHT}{Fore.GREEN}{val:.4f}{Style.RESET_ALL}"
    elif val == worst_val:
        return f"{Style.BRIGHT}{Fore.RED}{val:.4f}{Style.RESET_ALL}"
    else:
        return f"{val:.4f}"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.CYAN}Using device: {device}{Style.RESET_ALL}")

    base_path = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9\yolo_runs"
    head_folders = [f for f in os.listdir(base_path) if f.startswith("crowdhuman_person_head") and any(c.isdigit() for c in f)]
    if not head_folders:
        raise FileNotFoundError("No valid 'crowdhuman_person_head' folders found!")

    # Sort by number safely
    head_folders = sorted(head_folders, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    latest_run = os.path.join(base_path, head_folders[-1])
    weights = os.path.join(latest_run, "weights", "best.pt")

    if not os.path.exists(weights):
        raise FileNotFoundError(f"❌ Weights not found: {weights}")

    print(f"{Fore.GREEN}✅ Using weights: {weights}{Style.RESET_ALL}")
    
    model = YOLO(weights)
    yaml_path = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9\data.yaml"

    print(f"{Fore.YELLOW}\n🔄 Running validation...{Style.RESET_ALL}")
    metrics = model.val(data=yaml_path, device=device, workers=0)

    # Overall metrics
    precision_all = float(np.mean(metrics.box.p))
    recall_all    = float(np.mean(metrics.box.r))
    map50_all     = float(np.mean(metrics.box.map50))
    map50_95_all  = float(np.mean(metrics.box.map))
    f1_all        = 2 * precision_all * recall_all / (precision_all + recall_all + 1e-9)

    table_data = [
        [f"{Fore.MAGENTA}Precision{Style.RESET_ALL}", precision_all],
        [f"{Fore.MAGENTA}Recall{Style.RESET_ALL}", recall_all],
        [f"{Fore.MAGENTA}mAP50{Style.RESET_ALL}", map50_all],
        [f"{Fore.MAGENTA}mAP50-95{Style.RESET_ALL}", map50_95_all],
        [f"{Fore.MAGENTA}F1 Score{Style.RESET_ALL}", f1_all],
    ]
    print(f"{Fore.MAGENTA}\n📊 Validation Metrics:{Style.RESET_ALL}")
    print(tabulate(table_data, headers=["Metric", "Value"], floatfmt=".4f"))

    # Per-class metrics
    print(f"{Fore.BLUE}\n🔎 Per-Class Results:{Style.RESET_ALL}")
    class_names = metrics.names if hasattr(metrics, "names") else {0: "class0"}
    per_class_data = []

    # Collect all metrics first to determine best/worst per column
    precisions, recalls, m50s, m5095s, f1s = [], [], [], [], []
    for i, cls_name in class_names.items():
        p = safe_get(metrics.box.p, i)
        r = safe_get(metrics.box.r, i)
        m50 = safe_get(metrics.box.map50, i)
        m5095 = safe_get(metrics.box.map, i)
        f1 = 2 * p * r / (p + r + 1e-9)

        precisions.append(p)
        recalls.append(r)
        m50s.append(m50)
        m5095s.append(m5095)
        f1s.append(f1)
        per_class_data.append([cls_name, p, r, m50, m5095, f1])

    # Determine best/worst per column
    bests = [max(precisions), max(recalls), max(m50s), max(m5095s), max(f1s)]
    worsts = [min(precisions), min(recalls), min(m50s), min(m5095s), min(f1s)]

    # Apply coloring
    colored_table = []
    for row in per_class_data:
        cls_name = f"{Fore.CYAN}{row[0]}{Style.RESET_ALL}"
        p_colored     = color_metric(row[1], bests[0], worsts[0])
        r_colored     = color_metric(row[2], bests[1], worsts[1])
        m50_colored   = color_metric(row[3], bests[2], worsts[2])
        m5095_colored = color_metric(row[4], bests[3], worsts[3])
        f1_colored    = color_metric(row[5], bests[4], worsts[4])
        colored_table.append([cls_name, p_colored, r_colored, m50_colored, m5095_colored, f1_colored])

    print(tabulate(colored_table, headers=["Class", "Precision", "Recall", "mAP50", "mAP50-95", "F1"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
