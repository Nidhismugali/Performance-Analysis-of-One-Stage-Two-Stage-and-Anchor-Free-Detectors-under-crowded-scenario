# configs/centernet/centernet_r101_persononly_crowdhuman_v2i_best.py
# CenterNet (ResNet-101) — person-only, CrowdHuman.v2i (COCO-style)
_base_ = ['../_base_/default_runtime.py']

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
        depth=101,  # upgraded backbone
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='CTResNetNeck',
        in_channels=2048,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4)),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,
        in_channels=64,
        feat_channels=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(topk=300, local_maximum_kernel=3, max_per_img=300, score_thr=0.05)
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/crowdhuman.v2i.coco-mmdetection/'

train_dataloader = dict(
    batch_size=4,               # safe default for 6GB VRAM; increase to 6 if OOM-free
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
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),

            # MixUp first (works in this order for your mmdet version)
            dict(type='MixUp', img_scale=(640, 640), ratio_range=(0.8, 1.6)),

            # Mosaic next
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),

            # Random center crop / pad
            dict(
                type='RandomCenterCropPad',
                crop_size=(640, 640),
                ratios=(0.8, 1.25),
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
            ),

            # Multi-scale resize
            dict(keep_ratio=True, scale=[(512,512), (640,640), (720,720)], type='Resize'),

            # Albumentations photometric / geometric augmentations
            dict(
                type='Albu',
                transforms=[
                    dict(type='RandomBrightnessContrast', p=0.5),
                    dict(type='HueSaturationValue', p=0.5),
                    dict(
                        type='ShiftScaleRotate',
                        p=0.5,
                        rotate_limit=10,
                        scale_limit=0.1,
                        shift_limit=0.0625,
                        border_mode=0
                    ),
                ],
                keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
                bbox_params=dict(
                    type='BboxParams',
                    format='coco',
                    label_fields=['gt_bboxes_labels']
                )
            ),

        ]


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
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size_divisor=32, pad_val=dict(img=0, seg=255)),
            dict(type='EnsureBorderExists'),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'border'))
        ]
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
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size_divisor=32, pad_val=dict(img=0, seg=255)),
            dict(type='EnsureBorderExists'),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'border'))
        ]
    )
)

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'valid/_annotations.coco.json', metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1.0e-4),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 1000, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=80, by_epoch=True, milestones=[50, 70], gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=10),
    visualization=dict(type='DetVisualizationHook')
)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# runtime opts
env_cfg = dict(cudnn_benchmark=True)
log_level = 'INFO'
load_from = None
resume = False

work_dir = './work_dirs/centernet_r101_persononly_crowdhuman_v2i_best'
