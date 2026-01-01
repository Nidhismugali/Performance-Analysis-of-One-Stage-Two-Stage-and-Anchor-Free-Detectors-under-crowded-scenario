# configs/centernet/centernet_crowdhuman_2cls_strongmix.py

import os
import glob
import json

default_scope = 'mmdet'

data_root = 'data/'

# ==================== Small helper functions (NO context managers) ====================

def _ask_int(prompt, default):
    try:
        s = input(f"{prompt} [default {default}]: ").strip()
        if s == '':
            return default
        return int(s)
    except Exception:
        return default

def _list_checkpoints(work_dir):
    pattern = os.path.join(work_dir, 'epoch_*.pth')
    paths = sorted(glob.glob(pattern))
    items = []
    for p in paths:
        name = os.path.basename(p)
        try:
            num_str = name.replace('epoch_', '').replace('.pth', '')
            epoch = int(num_str)
        except Exception:
            epoch = -1
        items.append((epoch, p))
    items.sort(key=lambda x: x[0])
    return items

def _choose_checkpoint():
    print("\n==== Stage-1 (simple) / Stage-2 (strongmix) checkpoint picker ====")
    print("Where do you want to load weights from?")
    print("  1) simple    (work_dirs/centernet_crowdhuman_2cls_simple)")
    print("  2) strongmix (work_dirs/centernet_crowdhuman_2cls_strongmix)")
    print("  3) none      (start from scratch)")
    choice = _ask_int("Enter choice (1/2/3)", 1)

    if choice == 3:
        print(">> No checkpoint will be loaded (training from scratch).")
        return None

    if choice == 1:
        base_dir = 'work_dirs/centernet_crowdhuman_2cls_simple'
        label = 'simple'
    else:
        base_dir = 'work_dirs/centernet_crowdhuman_2cls_strongmix'
        label = 'strongmix'

    ckpts = _list_checkpoints(base_dir)
    if not ckpts:
        print(f">> No checkpoints found under {base_dir}, training from scratch.")
        return None

    print(f"\nFound checkpoints in {base_dir}:")
    for idx, (ep, path) in enumerate(ckpts):
        print(f"  {idx}: epoch {ep:3d} -> {path}")

    idx = _ask_int("Select checkpoint index", len(ckpts) - 1)
    if idx < 0 or idx >= len(ckpts):
        idx = len(ckpts) - 1

    chosen = ckpts[idx][1]
    print(f">> Using {label} checkpoint: {chosen}")
    return chosen

def _choose_train_subset():
    base_ann_rel = 'crowdhuman_coco/annotations/train.json'
    base_ann_path = os.path.join(data_root, base_ann_rel)
    if not os.path.exists(base_ann_path):
        print(f">> WARNING: {base_ann_path} not found, using it directly in dataset.")
        return base_ann_rel

    # Load COCO JSON (NO context manager)
    f = open(base_ann_path, 'r', encoding='utf-8')
    coco = json.load(f)
    f.close()

    images = coco.get('images', [])
    # Group by split name based on file_name prefix
    buckets = {
        'train01': [],
        'train02': [],
        'train03': [],
        'other': [],
    }

    for img in images:
        fn = img.get('file_name', '')
        if fn.startswith('CrowdHuman_train01/'):
            buckets['train01'].append(img)
        elif fn.startswith('CrowdHuman_train02/'):
            buckets['train02'].append(img)
        elif fn.startswith('CrowdHuman_train03/'):
            buckets['train03'].append(img)
        else:
            buckets['other'].append(img)

    total = len(images)
    print("\n==== CrowdHuman train split chooser ====")
    print(f"Total images in train.json: {total}")
    print("  train01:", len(buckets['train01']))
    print("  train02:", len(buckets['train02']))
    print("  train03:", len(buckets['train03']))
    print("  other  :", len(buckets['other']))

    print("\nWhich part of the train set do you want to use?")
    print("  1) ALL (train01 + train02 + train03 + other)  [default]")
    print("  2) train01 only")
    print("  3) train02 only")
    print("  4) train03 only")

    choice = _ask_int("Enter choice (1/2/3/4)", 1)

    if choice <= 1:
        print(">> Using FULL train.json for training.")
        return base_ann_rel

    if choice == 2:
        label = 'train01'
        selected_imgs = buckets['train01']
    elif choice == 3:
        label = 'train02'
        selected_imgs = buckets['train02']
    else:
        label = 'train03'
        selected_imgs = buckets['train03']

    if not selected_imgs:
        print(">> Selected split has 0 images, falling back to FULL train.json.")
        return base_ann_rel

    selected_ids = set()
    for img in selected_imgs:
        img_id = img.get('id', None)
        if img_id is not None:
            selected_ids.add(img_id)

    annotations = coco.get('annotations', [])
    new_annotations = []
    for ann in annotations:
        if ann.get('image_id', None) in selected_ids:
            new_annotations.append(ann)

    new_images = selected_imgs

    # Build new COCO dict, preserving other keys (info, licenses, categories, etc.)
    new_coco = {}
    for k in coco.keys():
        if k == 'images' or k == 'annotations':
            continue
        new_coco[k] = coco[k]
    new_coco['images'] = new_images
    new_coco['annotations'] = new_annotations

    out_rel = 'crowdhuman_coco/annotations/train_subset_' + label + '.json'
    out_path = os.path.join(data_root, out_rel)

    # Save JSON (NO context manager)
    d = os.path.dirname(out_path)
    if not os.path.exists(d):
        os.makedirs(d)
    f2 = open(out_path, 'w', encoding='utf-8')
    json.dump(new_coco, f2)
    f2.close()

    print(f">> Wrote subset annotation: {out_path}")
    print(f"   #images: {len(new_images)}, #anns: {len(new_annotations)}")
    return out_rel

# Pick checkpoint & train subset once at config load time
_load_from_ckpt = _choose_checkpoint()
_train_ann_rel = _choose_train_subset()

# ==================== Model ====================

model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=None,
    ),
    neck=None,
    bbox_head=dict(
        type='CenterNetHead',
        in_channels=512,
        feat_channels=512,
        num_classes=2,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        topk=100,
        local_maximum_kernel=3,
        max_per_img=100),
)

# ==================== Dataset & Pipelines ====================

dataset_type = 'CocoDataset'

metainfo = dict(
    classes=('person', 'head'),
    palette=[(220, 20, 60), (0, 255, 0)],
)

backend_args = None

# ----- Strong augmentation pipeline (Mosaic + MixUp) -----

train_pipeline_strong = [
    dict(
        type='Mosaic',
        img_scale=(512, 512),
        pad_val=114.0),
    # MixUp removed to reduce memory load
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Resize',
        scale=(512, 512),
        keep_ratio=True),
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape',
                   'img_shape', 'scale_factor')),
]


# MultiImageMixDataset wraps the base COCO dataset for Mosaic/MixUp
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=_train_ann_rel,
        data_prefix=dict(img='CrowdHuman/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        backend_args=backend_args,
    ),
    pipeline=train_pipeline_strong,
)

# ----- Val / Test (no strong augs) -----

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='Resize',
        scale=(512, 512),
        keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape',
                   'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='crowdhuman_coco/annotations/val.json',
        data_prefix=dict(img='CrowdHuman/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

# ==================== Evaluation ====================

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'crowdhuman_coco/annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = val_evaluator

# ==================== Training Loop & Schedules ====================

# Ask for epochs and val interval interactively
_max_epochs = _ask_int("How many epochs for STRONGMIX stage?", 3)
_val_interval = _ask_int("Run validation every how many epochs?", 1)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=_max_epochs,
    val_interval=_val_interval,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[90, 120],
        gamma=0.1),
]

# ==================== Runtime ====================

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
launcher = 'none'

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends,
)

work_dir = 'work_dirs/centernet_crowdhuman_2cls_strongmix'

# Final: connect chosen checkpoint (or None) for load_from
load_from = _load_from_ckpt

auto_scale_lr = dict(base_batch_size=16)
