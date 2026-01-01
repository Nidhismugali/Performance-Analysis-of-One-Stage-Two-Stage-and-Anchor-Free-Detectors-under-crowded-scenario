# --------------------------------------------------------
# YOLOv10n & YOLOv11n Evaluation + Metrics Comparison Only
# --------------------------------------------------------

import os
import torch
from ultralytics import YOLO
from colorama import Fore, Style, init
from tabulate import tabulate
import json

init(autoreset=True)


def evaluate_model(model_name, weights_path, yaml_path, device):
    """Evaluate trained YOLO model and return metrics dict"""
    print(f"\n📊 Evaluating model {Fore.CYAN}{model_name}{Style.RESET_ALL} using weights: {weights_path}\n")

    model = YOLO(weights_path)
    results = model.val(data=yaml_path, device=device)

    # Extract key metrics
    metrics = {
        "Precision": results.box.map_dict["precision"],
        "Recall": results.box.map_dict["recall"],
        "mAP50": results.box.map_dict["map50"],
        "mAP50-95": results.box.map_dict["map"],
        "F1 Score": (2 * results.box.map_dict["precision"] * results.box.map_dict["recall"]) /
                    (results.box.map_dict["precision"] + results.box.map_dict["recall"] + 1e-6)
    }

    print(f"{Fore.GREEN}✅ Evaluation complete for {model_name}!{Style.RESET_ALL}\n")
    return metrics


def compare_metrics(metrics_v10, metrics_v11):
    """Pretty print metrics comparison table"""
    headers = ["Metric", "YOLOv10n", "YOLOv11n", "Better Model"]
    table = []

    for key in metrics_v10.keys():
        val10 = metrics_v10[key]
        val11 = metrics_v11[key]

        if val10 > val11:
            better = Fore.CYAN + "YOLOv10n"
        elif val11 > val10:
            better = Fore.MAGENTA + "YOLOv11n"
        else:
            better = Fore.YELLOW + "Equal"

        table.append([
            key,
            f"{Fore.CYAN}{val10:.4f}",
            f"{Fore.MAGENTA}{val11:.4f}",
            better
        ])

    print("\n" + Fore.GREEN + "=" * 60)
    print(Fore.GREEN + "📈 YOLOv10n vs YOLOv11n Evaluation Metrics Comparison")
    print("=" * 60 + "\n")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    # Save as JSON
    out_path = "yolo_runs/metrics_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"YOLOv10n": metrics_v10, "YOLOv11n": metrics_v11}, f, indent=4)
    print(f"\n💾 Metrics saved at: {Fore.YELLOW}{out_path}{Style.RESET_ALL}\n")


def main():
    # -----------------------
    # Dataset paths
    # -----------------------
    dataset_dir = "C:/Users/mayan/Downloads/crowdhuman.v2i.yolov9"
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
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # -----------------------
    # Device selection
    # -----------------------
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\n🧠 Using device: {Fore.YELLOW}{device}{Style.RESET_ALL}\n")

    # -----------------------
    # Paths to trained weights (update these!)
    # -----------------------
    yolov10_best = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9\yolo_runs\yolov10n_crowdhuman_person_head\weights\best.pt"
    yolov11_best = r"C:\Users\mayan\Downloads\crowdhuman.v2i.yolov9\yolo_runs\yolov11n_crowdhuman_person_head\weights\best.pt"


    # -----------------------
    # Evaluate both models
    # -----------------------
    metrics_v10 = evaluate_model("yolov10n", yolov10_best, yaml_path, device)
    metrics_v11 = evaluate_model("yolov11n", yolov11_best, yaml_path, device)

    # -----------------------
    # Compare results
    # -----------------------
    compare_metrics(metrics_v10, metrics_v11)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
