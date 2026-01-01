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

def get_latest_weight(weights_folder):
    """Return the latest .pt file in the folder (by modified time)."""
    pt_files = [f for f in os.listdir(weights_folder) if f.endswith(".pt")]
    if not pt_files:
        return None
    pt_files = sorted(pt_files, key=lambda x: os.path.getmtime(os.path.join(weights_folder, x)), reverse=True)
    return os.path.join(weights_folder, pt_files[0])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.CYAN}Using device: {device}{Style.RESET_ALL}")

    base_path = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9\yolo_runs"
    head_folders = [f for f in os.listdir(base_path) if f.startswith("crowdhuman_person_head") and any(c.isdigit() for c in f)]
    if not head_folders:
        raise FileNotFoundError("No valid 'crowdhuman_person_head' folders found!")

    # Sort by number safely
    head_folders = sorted(head_folders, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    yaml_path = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9\data.yaml"

    combined_table = []

    for head in head_folders:
        latest_run = os.path.join(base_path, head)
        weights_folder = os.path.join(latest_run, "weights")
        weights = os.path.join(weights_folder, "best.pt")

        if not os.path.exists(weights):
            weights = get_latest_weight(weights_folder)
            if not weights:
                print(f"{Fore.YELLOW}⚠️ No weights found for {head}, skipping...{Style.RESET_ALL}")
                continue

        print(f"{Fore.GREEN}✅ Using weights for {head}: {weights}{Style.RESET_ALL}")
        model = YOLO(weights)

        print(f"{Fore.YELLOW}🔄 Running validation for {head}...{Style.RESET_ALL}")
        metrics = model.val(data=yaml_path, device=device, workers=0)

        # Overall metrics
        precision_all = float(np.mean(metrics.box.p))
        recall_all    = float(np.mean(metrics.box.r))
        map50_all     = float(np.mean(metrics.box.map50))
        map50_95_all  = float(np.mean(metrics.box.map))
        f1_all        = 2 * precision_all * recall_all / (precision_all + recall_all + 1e-9)

        combined_table.append([head, precision_all, recall_all, map50_all, map50_95_all, f1_all])

    # Determine best/worst per column for coloring
    precisions = [row[1] for row in combined_table]
    recalls    = [row[2] for row in combined_table]
    m50s       = [row[3] for row in combined_table]
    m5095s     = [row[4] for row in combined_table]
    f1s        = [row[5] for row in combined_table]

    bests  = [max(precisions), max(recalls), max(m50s), max(m5095s), max(f1s)]
    worsts = [min(precisions), min(recalls), min(m50s), min(m5095s), min(f1s)]

    # Apply coloring
    colored_table = []
    for row in combined_table:
        head_name = f"{Fore.CYAN}{row[0]}{Style.RESET_ALL}"
        p_colored     = color_metric(row[1], bests[0], worsts[0])
        r_colored     = color_metric(row[2], bests[1], worsts[1])
        m50_colored   = color_metric(row[3], bests[2], worsts[2])
        m5095_colored = color_metric(row[4], bests[3], worsts[3])
        f1_colored    = color_metric(row[5], bests[4], worsts[4])
        colored_table.append([head_name, p_colored, r_colored, m50_colored, m5095_colored, f1_colored])

    print(f"{Fore.MAGENTA}\n📊 Combined Head Metrics:{Style.RESET_ALL}")
    print(tabulate(colored_table, headers=["Head", "Precision", "Recall", "mAP50", "mAP50-95", "F1"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
