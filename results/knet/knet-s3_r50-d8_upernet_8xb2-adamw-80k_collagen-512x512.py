conv_kernel_size = 1
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
data_root = '/home/yangchangpeng/wing_studio/data/col_dataset'
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
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
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
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        dilations=(
            1,
            1,
            1,
            1,
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
            2,
            2,
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
        kernel_generate_head=dict(
            align_corners=False,
            channels=512,
            dropout_ratio=0.1,
            in_channels=[
                256,
                512,
                1024,
                2048,
            ],
            in_index=[
                0,
                1,
                2,
                3,
            ],
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=2,
            pool_scales=(
                1,
                2,
                3,
                6,
            ),
            type='UPerHead'),
        kernel_update_head=[
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=1,
                    type='KernelUpdator'),
                num_classes=2,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=1,
                    type='KernelUpdator'),
                num_classes=2,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=1,
                    type='KernelUpdator'),
                num_classes=2,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
        ],
        num_stages=3,
        type='IterativeDecodeHead'),
    pretrained='open-mmlab://resnet50_v1c',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 2
num_stages = 3
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
out_channels = 1
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=6380,
        milestones=[
            5000,
            6000,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.txt',
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/home/yangchangpeng/wing_studio/data/col_dataset',
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
    batch_size=2,
    dataset=dict(
        ann_file='train.txt',
        data_prefix=dict(
            img_path='img_dir/train/', seg_map_path='ann_dir/train/'),
        data_root='/home/yangchangpeng/wing_studio/data/col_dataset',
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
    num_workers=2,
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
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.txt',
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/home/yangchangpeng/wing_studio/data/col_dataset',
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
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './results/knet'
