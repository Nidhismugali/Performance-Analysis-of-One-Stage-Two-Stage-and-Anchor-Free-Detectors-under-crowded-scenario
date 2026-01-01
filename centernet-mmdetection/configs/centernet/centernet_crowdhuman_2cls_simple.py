# configs/centernet/centernet_crowdhuman_2cls_simple.py
# Simple CenterNet config for CrowdHuman with interactive CLI.

import os
from typing import List, Optional

# ======== Small helper functions (NO `with` statements here) ========

def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except Exception:
        return ""

def _safe_input_int(prompt: str, default: int, min_val: int = 1) -> int:
    raw = _safe_input(prompt).strip()
    if raw == "":
        return default
    try:
        v = int(raw)
        if v < min_val:
            return default
        return v
    except ValueError:
        return default

def _safe_input_choice(prompt: str, valid: List[str], default: str) -> str:
    raw = _safe_input(prompt).strip()
    if raw == "":
        return default
    return raw if raw in valid else default

def _build_ann_file_for_split(data_root: str, split_choice: str) -> str:
    """Return annotation JSON path for selected CrowdHuman split."""
    base_ann_rel = "crowdhuman_coco/annotations/train.json"
    base_ann = os.path.join(data_root, base_ann_rel)
    base_dir = os.path.dirname(os.path.abspath(base_ann))

    t01 = os.path.join(base_dir, "train01.json")
    t02 = os.path.join(base_dir, "train02.json")
    t03 = os.path.join(base_dir, "train03.json")

    has01 = os.path.isfile(t01)
    has02 = os.path.isfile(t02)
    has03 = os.path.isfile(t03)

    if not (has01 or has02 or has03):
        # Just use the full train.json
        return base_ann_rel.replace("\\", "/")

    mapping = {
        "1": [t01],
        "2": [t02],
        "3": [t03],
        "4": [p for p, ok in ((t01, has01), (t02, has02)) if ok],
        "5": [p for p, ok in ((t02, has02), (t03, has03)) if ok],
        "6": [p for p, ok in ((t03, has03), (t01, has01)) if ok],
        "7": [p for p, ok in ((t01, has01), (t02, has02), (t03, has03)) if ok],
    }
    files = mapping.get(split_choice, mapping["7"])

    # If more than one file is requested (1+2, 2+3, 3+1, ALL),
    # we just fall back to the full train.json for simplicity.
    if len(files) != 1:
        return base_ann_rel.replace("\\", "/")

    rel = os.path.relpath(files[0], data_root)
    return rel.replace("\\", "/")

def _enumerate_checkpoints(exp_dir: str):
    """Return sorted list of epoch_*.pth inside exp_dir."""
    if not os.path.isdir(exp_dir):
        return []
    names = [
        n for n in os.listdir(exp_dir)
        if n.endswith(".pth") and n.startswith("epoch_")
    ]

    def _epoch_num(name: str) -> int:
        try:
            core = name[len("epoch_"):-len(".pth")]
            return int(core)
        except Exception:
            return 0

    names.sort(key=_epoch_num)
    return [os.path.join(exp_dir, n) for n in names]

def _pick_checkpoint() -> Optional[str]:
    """Ask the user which checkpoint (if any) to load."""
    print()
    print("Start from which weights?")
    print("  0) scratch (no detection ckpt, only backbone)")
    print("  1) simple    (work_dirs/centernet_crowdhuman_2cls_simple)")
    print("  2) strongmix (work_dirs/centernet_crowdhuman_2cls_strongmix)")
    print("  3) custom .pth path")
    choice = _safe_input_choice(
        "Enter choice [default 0]: ",
        ["0", "1", "2", "3"],
        "0",
    )

    if choice == "0":
        return None

    if choice == "3":
        path = _safe_input("Path to .pth file: ").strip().strip('"').strip("'")
        if path and os.path.isfile(path):
            return os.path.abspath(path)
        print(f"[Config] WARNING: custom path '{path}' not found. Training from scratch.")
        return None

    base = {
        "1": "work_dirs/centernet_crowdhuman_2cls_simple",
        "2": "work_dirs/centernet_crowdhuman_2cls_strongmix",
    }[choice]
    exp_dir = os.path.abspath(base)
    ckpts = _enumerate_checkpoints(exp_dir)
    if not ckpts:
        print(f"[Config] No epoch_*.pth found in {exp_dir}. Training from scratch.")
        return None

    print(f"[Config] Found {len(ckpts)} checkpoints in {exp_dir}:")
    for idx, p in enumerate(ckpts):
        print(f"  {idx}: {os.path.basename(p)}")

    default_idx = len(ckpts) - 1
    idx = _safe_input_int(
        f"Select checkpoint index [0-{len(ckpts)-1}, default {default_idx}]: ",
        default=default_idx,
        min_val=0,
    )
    if idx >= len(ckpts):
        idx = len(ckpts) - 1
    chosen = os.path.abspath(ckpts[idx])
    print(f"[Config] Using checkpoint: {chosen}")
    return chosen

def _build_work_dir(load_from: Optional[str]) -> str:
    """Decide a work_dir for this run."""
    base_simple = "work_dirs/centernet_crowdhuman_2cls_simple"
    if not load_from:
        return base_simple
    parent = os.path.dirname(os.path.dirname(os.path.abspath(load_from)))
    src_exp = os.path.basename(os.path.dirname(os.path.abspath(load_from)))
    new_name = src_exp + "_finetune"
    candidate = os.path.join(parent, new_name)
    return candidate

# ======== INTERACTIVE PROMPTS (gated by env var) ========

data_root = "data/"
_interactive = os.environ.get("MMDET_INTERACTIVE", "1") == "1"

if _interactive:
    print("==== Simple CenterNet (CrowdHuman) interactive config ====")
    _max_epochs = _safe_input_int("Max epochs [default 3]: ", default=3, min_val=1)
    _batch_size = _safe_input_int("Batch size [default 4]: ", default=4, min_val=1)
    _split_choice = _safe_input_choice(
        "Which CrowdHuman train split? 1=train01, 2=train02, 3=train03, "
        "4=1+2, 5=2+3, 6=3+1, 7=ALL [default 7]: ",
        valid=["1", "2", "3", "4", "5", "6", "7"],
        default="7",
    )
    _load_from_ckpt = _pick_checkpoint()
    _train_ann_file = _build_ann_file_for_split(data_root, _split_choice)
    work_dir = _build_work_dir(_load_from_ckpt).replace("\\", "/")
    print(f"[Config] Train ann_file = {_train_ann_file}")
    print(f"[Config] Max epochs = {_max_epochs}, batch_size = {_batch_size}")
    print(f"[Config] Starting from ckpt: {_load_from_ckpt if _load_from_ckpt else 'SCRATCH'}")
    print(f"[Config] work_dir = {work_dir}")
    print()
else:
    # Non-interactive defaults, used e.g. during evaluation
    _max_epochs = 3
    _batch_size = 4
    _split_choice = "7"
    _load_from_ckpt = None
    _train_ann_file = _build_ann_file_for_split(data_root, _split_choice)
    work_dir = "work_dirs/centernet_crowdhuman_2cls_simple".replace("\\", "/")

# ===================== Standard MMDet config =====================

default_scope = "mmdet"

dataset_type = "CocoDataset"

metainfo = dict(
    classes=("person", "head"),
    palette=[(220, 20, 60), (0, 255, 0)],
)

# No augmentation, fixed resize (needed because CenterNet expects fixed-ish size)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader = dict(
    batch_size=_batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=_train_ann_file,
        data_prefix=dict(img="CrowdHuman/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="crowdhuman_coco/annotations/val.json",
        data_prefix=dict(img="CrowdHuman/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "crowdhuman_coco/annotations/val.json",
    metric="bbox",
)
test_evaluator = val_evaluator

model = dict(
    type="CenterNet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=18,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet18"),
    ),
    neck=None,
    bbox_head=dict(
        type="CenterNetHead",
        num_classes=2,
        in_channels=512,
        feat_channels=512,
        loss_center_heatmap=dict(type="GaussianFocalLoss", loss_weight=1.0),
        loss_wh=dict(type="L1Loss", loss_weight=0.1),
        loss_offset=dict(type="L1Loss", loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        topk=100,
        local_maximum_kernel=3,
        max_per_img=100,
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
)

param_scheduler = [
    dict(
        type="MultiStepLR",
        by_epoch=True,
        milestones=[90, 120],
        gamma=0.1,
    )
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=_max_epochs,
    val_interval=1,
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

auto_scale_lr = dict(base_batch_size=16)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(
    type="LogProcessor",
    window_size=50,
    by_epoch=True,
)

log_level = "INFO"

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    name="visualizer",
    vis_backends=vis_backends,
)

# Keep ALL checkpoints so you don't lose epochs
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=-1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

launcher = "none"

# Detection ckpt to start from (if any), chosen in the interactive block
load_from = _load_from_ckpt
# work_dir already set above, either interactive or default
