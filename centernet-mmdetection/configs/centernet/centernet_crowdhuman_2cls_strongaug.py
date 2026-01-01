# configs/centernet/centernet_crowdhuman_2cls_strongmix.py

default_scope = 'mmdet'

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
        # We will load the FULL model from a checkpoint via load_from below,
        # so no need for ImageNet init here.
        init_cfg=None,
    ),
    neck=None,
    bbox_head=dict(
        type='CenterNetHead',
        in_channels=512,
        feat_channels=512,
        num_classes=2,
        loss_center_heatmap=dict(
            type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(
            type='L1Loss', loss_weight=0.1),
        loss_offset=dict(
            type='L1Loss', loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        topk=100,
        local_maximum_kernel=3,
        max_per_img=100),
)

# ==================== Dataset & Pipelines ====================

dataset_type = 'CocoDataset'
data_root = 'data/'

metainfo = dict(
    classes=('person', 'head'),
    palette=[(220, 20, 60), (0, 255, 0)],
)

backend_args = None

# ----- Strong augmentation pipeline (Mosaic + MixUp) -----

train_pipeline_strong = [
    # 4-image mosaic
    dict(
        type='Mosaic',
        img_scale=(512, 512),
        pad_val=114.0),

    # Mix images again (2-image blend)
    dict(
        type='MixUp',
        img_scale=(512, 512),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),

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
        ann_file='crowdhuman_coco/annotations/train.json',
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

# ----- Val / Test (no crazy augs, same as simple config) -----

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
    batch_size=16,
    num_workers=8,
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

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=3,      # stage-2 epochs (you can change)
    val_interval=1,
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

# ======== IMPORTANT: load last simple-stage checkpoint here ========
# You already tested with:
#   work_dirs/centernet_crowdhuman_2cls_simple/epoch_1.pth
# If later you train more epochs, just change this to epoch_3.pth etc.
load_from = 'work_dirs/centernet_crowdhuman_2cls_simple/epoch_1.pth'

auto_scale_lr = dict(base_batch_size=16)
