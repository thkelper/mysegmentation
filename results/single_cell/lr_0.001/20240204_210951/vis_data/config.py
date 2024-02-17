batch_size = 8
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/home/yangchangpeng/wing_studio/data/'
dataset_type = 'CollagenSegDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=638, save_best='mIoU', type='CheckpointHook'),
    logger=dict(interval=638, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    512,
    512,
)
img_suffix = '.tif'
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr = 0.001
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        out_channels=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        compact=False,
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        mask_size=(
            97,
            97,
        ),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        normalization_factor=1.0,
        num_classes=2,
        out_channels=1,
        psa_softmax=True,
        psa_type='bi-direction',
        shrink_factor=2,
        type='PSAHead'),
    pretrained='open-mmlab://resnet50_v1c',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 2
num_workers = 4
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005)
out_channels = 1
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=6380,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
seg_map_suffix = '.png'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='20240201_ASMC_single_cell/val.txt',
        data_prefix=dict(
            img_path='20240201_ASMC_single_cell/',
            seg_map_path='20240201_ASMC_single_cell/'),
        data_root='/home/yangchangpeng/wing_studio/data/',
        img_suffix='.tif',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='CollagenSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=6380, type='IterBasedTrainLoop', val_interval=638)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='20240201_ASMC_single_cell/train.txt',
        data_prefix=dict(
            img_path='20240201_ASMC_single_cell/',
            seg_map_path='20240201_ASMC_single_cell/'),
        data_root='/home/yangchangpeng/wing_studio/data/',
        img_suffix='.tif',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='CollagenSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
use_sigmoid = True
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='20240201_ASMC_single_cell/val.txt',
        data_prefix=dict(
            img_path='20240201_ASMC_single_cell/',
            seg_map_path='20240201_ASMC_single_cell/'),
        data_root='/home/yangchangpeng/wing_studio/data/',
        img_suffix='.tif',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='CollagenSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(type='WandbVisBackend'),
    ])
work_dir = './results/single_cell/lr_0.001'
