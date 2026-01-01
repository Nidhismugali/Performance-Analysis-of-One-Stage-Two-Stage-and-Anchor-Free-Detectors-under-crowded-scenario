# tools/eval_prf.py
#
# Extended detection metrics (COCO + Precision/Recall/F1/Accuracy)
# with multi-checkpoint comparison and colored summary table.
#
# Usage:
#   (openmmlab) python tools/eval_prf.py
#
# Features:
#   - Choose checkpoints from:
#       * work_dirs/centernet_crowdhuman_2cls_simple_finetune
#       * work_dirs/centernet_crowdhuman_2cls_strongmix
#       * any custom config + checkpoint path(s)
#   - Evaluate multiple checkpoints in one go
#   - At the end, print a comparison table
#     (best values in green, worst in red if colorama is installed)

import os
import re
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from pycocotools.coco import COCO

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric

from mmdet.utils import register_all_modules

# Optional color support
try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
except Exception:  # pragma: no cover
    class _Dummy:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = _Dummy()
    Style = _Dummy()


# ===================== Custom metric: PR/F1/Accuracy =====================

@METRICS.register_module()
class DetPRFMetric(BaseMetric):
    """Compute per-class and overall Precision / Recall / F1 / Accuracy for
    detection using COCO-style annotations."""

    def __init__(
        self,
        ann_file: str,
        iou_thr: float = 0.5,
        score_thr: float = 0.05,
        classwise: bool = True,
        prefix: Optional[str] = "prf",
    ):
        super().__init__(collect_device="cpu", prefix=prefix)
        self.ann_file = ann_file
        self.iou_thr = float(iou_thr)
        self.score_thr = float(score_thr)
        self.classwise = classwise

        self.coco_gt = COCO(self.ann_file)
        self.cat_ids = sorted(self.coco_gt.getCatIds())
        cats = self.coco_gt.loadCats(self.cat_ids)
        self.class_names = [c["name"] for c in cats]
        self.img_ids = set(self.coco_gt.getImgIds())

    def process(self, data_batch: Dict, data_samples: List[Any]) -> None:
        """Collect predictions for each image."""
        import torch

        def to_numpy(x):
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        for sample in data_samples:
            img_id = None

            # 1) attribute
            if hasattr(sample, "img_id"):
                img_id = getattr(sample, "img_id", None)

            # 2) metainfo on DetDataSample
            if img_id is None and hasattr(sample, "metainfo"):
                meta = getattr(sample, "metainfo")
                try:
                    if hasattr(meta, "get"):
                        img_id = meta.get("img_id", meta.get("img_id_", None))
                    else:
                        img_id = meta["img_id"] if "img_id" in meta else None
                except Exception:
                    pass

            # 3) plain dict
            if img_id is None and isinstance(sample, dict):
                meta = sample.get("metainfo", sample.get("meta", {}))
                if isinstance(meta, dict):
                    img_id = meta.get("img_id", meta.get("img_id_", None))
                if img_id is None:
                    img_id = sample.get("img_id", sample.get("img_id_", None))

            if img_id is None:
                continue

            try:
                img_id = int(img_id)
            except Exception:
                continue

            if hasattr(sample, "pred_instances"):
                pred = sample.pred_instances
            elif isinstance(sample, dict) and "pred_instances" in sample:
                pred = sample["pred_instances"]
            else:
                continue

            if hasattr(pred, "bboxes"):
                bboxes = pred.bboxes
                scores = pred.scores
                labels = pred.labels
            elif isinstance(pred, dict):
                bboxes = pred.get("bboxes", None)
                scores = pred.get("scores", None)
                labels = pred.get("labels", None)
            else:
                continue

            if bboxes is None or scores is None or labels is None:
                continue

            bboxes = to_numpy(bboxes)
            scores = to_numpy(scores)
            labels = to_numpy(labels)

            if self.score_thr is not None:
                keep = scores >= self.score_thr
                bboxes = bboxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            img_results: List[Dict[str, Any]] = []

            for box, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                label = int(label)
                if label < 0 or label >= len(self.cat_ids):
                    continue
                cat_id = int(self.cat_ids[label])

                rec = dict(
                    image_id=int(img_id),
                    bbox=[float(x1), float(y1), float(w), float(h)],
                    score=float(score),
                    category_id=cat_id,
                )

                if rec["image_id"] in self.img_ids:
                    img_results.append(rec)

            self.results.append(img_results)

    @staticmethod
    def _iou_xywh(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)

        x1, y1, w1, h1 = box
        x2 = x1 + w1
        y2 = y1 + h1

        xx1 = np.maximum(x1, boxes[:, 0])
        yy1 = np.maximum(y1, boxes[:, 1])
        xx2 = np.minimum(x2, boxes[:, 0] + boxes[:, 2])
        yy2 = np.minimum(y2, boxes[:, 1] + boxes[:, 3])

        inter_w = np.maximum(xx2 - xx1, 0.0)
        inter_h = np.maximum(yy2 - yy1, 0.0)
        inter = inter_w * inter_h

        area1 = w1 * h1
        area2 = boxes[:, 2] * boxes[:, 3]
        union = area1 + area2 - inter
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        return iou

    def compute_metrics(self, results: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Compute PR/F1/Accuracy given collected predictions."""
        dets: List[Dict[str, Any]] = []
        for img_list in results:
            for d in img_list:
                if d["image_id"] in self.img_ids:
                    dets.append(d)

        num_classes = len(self.cat_ids)
        tp = np.zeros(num_classes, dtype=np.int64)
        fp = np.zeros(num_classes, dtype=np.int64)
        fn = np.zeros(num_classes, dtype=np.int64)

        if len(dets) == 0:
            metrics: Dict[str, float] = dict(
                overall_precision=0.0,
                overall_recall=0.0,
                overall_f1=0.0,
                overall_accuracy=0.0,
            )
            if self.classwise:
                for cname in self.class_names:
                    cname = str(cname)
                    metrics[f"{cname}_precision"] = 0.0
                    metrics[f"{cname}_recall"] = 0.0
                    metrics[f"{cname}_f1"] = 0.0
                    metrics[f"{cname}_accuracy"] = 0.0
            return metrics

        coco_dt = self.coco_gt.loadRes(dets)

        for cls_idx, cat_id in enumerate(self.cat_ids):
            img_ids = self.coco_gt.getImgIds(catIds=[cat_id])
            for img_id in img_ids:
                gt_ann_ids = self.coco_gt.getAnnIds(
                    imgIds=[img_id], catIds=[cat_id], iscrowd=None
                )
                gt_anns = self.coco_gt.loadAnns(gt_ann_ids)
                gt_boxes = np.array(
                    [a["bbox"] for a in gt_anns], dtype=np.float32
                )  # xywh
                num_gt = len(gt_boxes)
                gt_matched = np.zeros(num_gt, dtype=bool)

                dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
                dt_anns = coco_dt.loadAnns(dt_ann_ids)
                dt_anns = sorted(dt_anns, key=lambda x: -x["score"])

                for dt in dt_anns:
                    dt_box = np.array(dt["bbox"], dtype=np.float32)
                    if num_gt == 0:
                        fp[cls_idx] += 1
                        continue
                    ious = self._iou_xywh(dt_box, gt_boxes)
                    max_iou = float(ious.max())
                    max_idx = int(ious.argmax())
                    if max_iou >= self.iou_thr and not gt_matched[max_idx]:
                        tp[cls_idx] += 1
                        gt_matched[max_idx] = True
                    else:
                        fp[cls_idx] += 1

                fn[cls_idx] += int((~gt_matched).sum())

        eps = 1e-12
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / np.maximum(tp + fn, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
        accuracy = tp / np.maximum(tp + fp + fn, 1)

        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        micro_precision = tp_sum / max(tp_sum + fp_sum, 1)
        micro_recall = tp_sum / max(tp_sum + fn_sum, 1)
        micro_f1 = 2 * micro_precision * micro_recall / max(
            micro_precision + micro_recall, eps
        )
        micro_accuracy = tp_sum / max(tp_sum + fp_sum + fn_sum, 1)

        metrics: Dict[str, float] = dict(
            overall_precision=float(micro_precision),
            overall_recall=float(micro_recall),
            overall_f1=float(micro_f1),
            overall_accuracy=float(micro_accuracy),
        )

        if self.classwise:
            for i, cname in enumerate(self.class_names):
                cname = str(cname)
                metrics[f"{cname}_precision"] = float(precision[i])
                metrics[f"{cname}_recall"] = float(recall[i])
                metrics[f"{cname}_f1"] = float(f1[i])
                metrics[f"{cname}_accuracy"] = float(accuracy[i])

        return metrics


# ===================== Experiment description & helpers =====================

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Experiment:
    name: str
    config_path: Path
    work_dir: Path


def _get_experiments() -> List[Experiment]:
    simple_cfg = ROOT / "configs" / "centernet" / "centernet_crowdhuman_2cls_simple.py"

    return [
        Experiment(
            name="simple_finetune",
            config_path=simple_cfg,
            work_dir=ROOT / "work_dirs" / "centernet_crowdhuman_2cls_simple_finetune",
        ),
        Experiment(
            name="strongmix",
            config_path=simple_cfg,  # still use simple config for eval
            work_dir=ROOT / "work_dirs" / "centernet_crowdhuman_2cls_strongmix",
        ),
    ]


@dataclass
class EvalJob:
    label: str
    config_path: Path
    ckpt_path: Path
    work_dir: Path


def _list_checkpoints(work_dir: Path) -> List[Path]:
    pattern = work_dir / "epoch_*.pth"
    ckpts = sorted(glob.glob(str(pattern)))

    def _epoch_num(p: str) -> int:
        m = re.search(r"epoch_(\d+)\.pth", p)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 0
        return 0

    ckpts.sort(key=_epoch_num)
    return [Path(p) for p in ckpts]


def _select_experiment(experiments: List[Experiment]) -> Optional[Experiment]:
    print("Which experiment do you want to evaluate for this group?")
    for i, exp in enumerate(experiments):
        print(f"  {i + 1}) {exp.name:<12} ({exp.work_dir.as_posix()})")
    print("  3) custom config + checkpoint(s)")
    choice = input("Enter choice (1/2/3) [default 1]: ").strip()

    if choice == "" or choice == "1":
        return experiments[0]
    if choice == "2":
        return experiments[1]
    if choice == "3":
        return None
    return experiments[0]


def _select_jobs() -> List[EvalJob]:
    experiments = _get_experiments()
    jobs: List[EvalJob] = []

    print("==== Extended detection metrics (COCO + PRF) ====")

    while True:
        print(f"\nConfigure evaluation group #{len(jobs) + 1}")
        exp = _select_experiment(experiments)

        if exp is None:
            cfg_path_str = input(
                "Enter custom config path (relative or absolute): "
            ).strip()
            if not cfg_path_str:
                print("No config given, skipping this group.")
            else:
                cfg_path = Path(cfg_path_str)
                if not cfg_path.is_file():
                    print(f"[WARN] Config not found: {cfg_path}")
                else:
                    ckpt_str = input(
                        "Enter checkpoint path(s), separated by ';' if multiple: "
                    ).strip()
                    if not ckpt_str:
                        print("[WARN] No checkpoint given, skipping this group.")
                    else:
                        for part in ckpt_str.split(";"):
                            p = Path(part.strip().strip('"').strip("'"))
                            if not p.is_file():
                                print(f"[WARN] Checkpoint not found: {p}")
                                continue
                            label = f"custom:{p.stem}"
                            jobs.append(
                                EvalJob(
                                    label=label,
                                    config_path=cfg_path,
                                    ckpt_path=p,
                                    work_dir=ROOT / "work_dirs" / "eval_prf_custom",
                                )
                            )
        else:
            ckpts = _list_checkpoints(exp.work_dir)
            if not ckpts:
                print(f"[WARN] No checkpoints like epoch_*.pth in {exp.work_dir}")
            else:
                print(f"\nFound checkpoints in {exp.work_dir}:")
                for idx, p in enumerate(ckpts):
                    m = re.search(r"epoch_(\d+)\.pth", p.name)
                    ep = m.group(1) if m else "?"
                    print(f"  {idx}: epoch {ep:>4} -> {p.as_posix()}")

                default_idx = len(ckpts) - 1
                idx_str = input(
                    "Enter checkpoint indices to evaluate "
                    "(e.g. 0,2,5 or 'all') [default last]: "
                ).strip()

                indices: List[int] = []
                if idx_str == "" or idx_str.lower() == "last":
                    indices = [default_idx]
                elif idx_str.lower() in {"all", "*"}:
                    indices = list(range(len(ckpts)))
                else:
                    for part in idx_str.split(","):
                        try:
                            v = int(part.strip())
                            if 0 <= v < len(ckpts):
                                indices.append(v)
                        except ValueError:
                            continue
                    if not indices:
                        indices = [default_idx]

                for i in indices:
                    ckpt = ckpts[i]
                    m = re.search(r"epoch_(\d+)\.pth", ckpt.name)
                    ep = m.group(1) if m else "?"
                    label = f"{exp.name}:epoch_{ep}"
                    jobs.append(
                        EvalJob(
                            label=label,
                            config_path=exp.config_path,
                            ckpt_path=ckpt,
                            work_dir=exp.work_dir,
                        )
                    )

        more = input("Add another experiment group? (y/N): ").strip().lower()
        if more not in {"y", "yes"}:
            break

    if not jobs:
        print("No evaluation jobs configured, aborting.")
    return jobs


def _resolve_ann_file(cfg: Config) -> str:
    """Try to figure out annotation file path from cfg."""
    ann_file = None

    if hasattr(cfg, "val_evaluator") and getattr(cfg.val_evaluator, "ann_file", None):
        ann_file = cfg.val_evaluator.ann_file
    elif hasattr(cfg, "test_evaluator"):
        te = cfg.test_evaluator
        if isinstance(te, dict) and "ann_file" in te:
            ann_file = te["ann_file"]
        elif isinstance(te, list):
            for m in te:
                if isinstance(m, dict) and "ann_file" in m:
                    ann_file = m["ann_file"]
                    break

    if ann_file is None and hasattr(cfg, "test_dataloader"):
        ds = cfg.test_dataloader.get("dataset", None)
        if ds is not None:
            data_root = ds.get("data_root", "")
            ann_rel = ds.get("ann_file", "")
            ann_file = os.path.join(data_root, ann_rel)

    if ann_file is None:
        raise RuntimeError("Could not infer ann_file from config for PRF metric.")

    return os.path.abspath(ann_file)


# ===================== Main =====================

def _color_value(val: Optional[float], vmin: Optional[float], vmax: Optional[float]) -> str:
    if val is None:
        return "   -    "
    s = f"{val:.4f}"
    if vmin is None or vmax is None or vmax <= vmin:
        return s
    if abs(val - vmax) <= 1e-12:
        return f"{Fore.GREEN}{s}{Style.RESET_ALL}"
    if abs(val - vmin) <= 1e-12:
        return f"{Fore.RED}{s}{Style.RESET_ALL}"
    return s


def main():
    # Disable interactive questions inside configs while evaluating
    os.environ["MMDET_INTERACTIVE"] = "0"

    register_all_modules()

    jobs = _select_jobs()
    if not jobs:
        return

    all_results: List[Dict[str, Any]] = []

    for idx, job in enumerate(jobs, start=1):
        print("\n============================================================")
        print(f"Run {idx}/{len(jobs)}: {job.label}")
        print("------------------------------------------------------------")
        cfg = Config.fromfile(str(job.config_path))
        work_dir = job.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg.work_dir = str(work_dir)
        cfg.load_from = str(job.ckpt_path)

        if not hasattr(cfg, "test_cfg"):
            cfg.test_cfg = dict(type="TestLoop")

        ann_file = _resolve_ann_file(cfg)

        cfg.test_evaluator = [
            dict(
                type="CocoMetric",
                ann_file=ann_file,
                metric="bbox",
            ),
            dict(
                type="DetPRFMetric",
                ann_file=ann_file,
                iou_thr=0.5,
                score_thr=0.05,
                classwise=True,
            ),
        ]

        print(f"Config    : {job.config_path.as_posix()}")
        print(f"Checkpoint: {job.ckpt_path.as_posix()}")
        print(f"Work dir  : {work_dir.as_posix()}")
        print(f"Ann file  : {ann_file}")

        runner = Runner.from_cfg(cfg)
        metrics = runner.test()
        all_results.append({"job": job, "metrics": metrics})

    # ================= Summary table =================
    print("\n==================== Evaluation summary ====================")

    rows = []
    for entry in all_results:
        job: EvalJob = entry["job"]
        m: Dict[str, Any] = entry["metrics"]

        # Try several possible keys for mAP
        coco_map = None
        for key in ("coco/bbox_mAP", "bbox_mAP"):
            if key in m and isinstance(m[key], (int, float)):
                coco_map = float(m[key])
                break

        overall_prec = m.get("prf/overall_precision", None)
        overall_rec = m.get("prf/overall_recall", None)
        overall_f1 = m.get("prf/overall_f1", None)

        rows.append(
            dict(
                label=job.label,
                ckpt=job.ckpt_path.name,
                coco_map=coco_map,
                prec=overall_prec,
                rec=overall_rec,
                f1=overall_f1,
            )
        )

    if not rows:
        print("No metrics collected.")
        return

    def _metric_bounds(field: str):
        vals = [r[field] for r in rows if r[field] is not None]
        if not vals:
            return None, None
        return min(vals), max(vals)

    map_min, map_max = _metric_bounds("coco_map")
    rec_min, rec_max = _metric_bounds("rec")
    f1_min, f1_max = _metric_bounds("f1")

    header = (
        f"{'Run label':30s}  {'Checkpoint':20s}  "
        f"{'mAP':>8s}  {'Recall':>8s}  {'F1':>8s}"
    )
    print("\n" + header)
    print("-" * len(header))

    for r in rows:
        label = r["label"][:30]
        ckpt = r["ckpt"][:20]
        map_str = _color_value(r["coco_map"], map_min, map_max)
        rec_str = _color_value(r["rec"], rec_min, rec_max)
        f1_str = _color_value(r["f1"], f1_min, f1_max)
        print(f"{label:30s}  {ckpt:20s}  {map_str:>8s}  {rec_str:>8s}  {f1_str:>8s}")


if __name__ == "__main__":
    main()
