_base_ = ['../_base_/default_runtime.py']

# ---------------------- MODEL ----------------------
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
        depth=101,
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
    ),
    neck=dict(
        type='CTResNetNeck',
        in_channels=2048,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4)
    ),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,
        in_channels=64,
        feat_channels=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(
        score_thr=0.05,
        topk=300,
        max_per_img=300,
        local_maximum_kernel=3
    )
)

# ---------------------- DATASET ----------------------
dataset_type = 'CocoDataset'
data_root = 'data/crowdhuman.v2i.coco-mmdetection/'

# ---------------------- SAFE TRAIN PIPELINE ----------------------
# --- STRONG & SAFE ALBUMENTATIONS BLOCK ---
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),

    # photometric & small geometric augmentations (albumentations)
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomBrightnessContrast', p=0.5),
            dict(type='HueSaturationValue', p=0.5),
            dict(
                type='Affine',  # prefer Affine instead of ShiftScaleRotate warning
                p=0.5,
                rotate=(-10, 10),
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                shear=(-2, 2),
                mode=0
            ),
            # optional: coarse dropout / cutout with safe params
            dict(type='CoarseDropout', max_holes=8, max_height=30, max_width=30, p=0.3),
        ],
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
        bbox_params=dict(
            type='BboxParams',
            format='coco',           # your annotations are COCO coords (x,y,w,h)
            label_fields=['gt_bboxes_labels'],
            min_visibility=0.0,
            clip=True
        )
    ),

    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),

    # <-- IMPORTANT: filter invalid bboxes after ALL aug/resize -->
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),

    dict(type='PackDetInputs')
]


# ---------------------- VAL / TEST PIPELINE ----------------------
val_test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=0, seg=255)),
    dict(type='EnsureBorderExists'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape',
                   'img_shape', 'scale_factor', 'border')
    )
]

# ---------------------- DATALOADERS ----------------------
train_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=('person',)),
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        metainfo=dict(classes=('person',)),
        pipeline=val_test_pipeline
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=dict(classes=('person',)),
        pipeline=val_test_pipeline
    )
)

# ---------------------- EVALUATORS ----------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox'
)
test_evaluator = val_evaluator

# ---------------------- TRAINING SETTINGS ----------------------
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1./1000,
         by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True,
         begin=0, end=80, milestones=[50, 70], gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(interval=10),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(cudnn_benchmark=True)
fp16 = dict(loss_scale='dynamic')

log_level = 'INFO'
load_from = None
resume = False
work_dir = './work_dirs/centernet_r101_persononly_crowdhuman_v2i_final'
