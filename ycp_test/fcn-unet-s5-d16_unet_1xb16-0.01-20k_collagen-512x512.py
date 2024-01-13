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
    checkpoint=dict(by_epoch=False, interval=319, type='CheckpointHook'),
    logger=dict(interval=319, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, type='SegVisualizationHook'))
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
load_from = 'ycp_test/iter_3190.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=128,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        out_channels=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        base_channels=64,
        conv_cfg=None,
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        type='UNet',
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
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
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=64,
        in_index=4,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        out_channels=1,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        256,
        256,
    ), mode='slide', stride=(
        170,
        170,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=1000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
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
    num_workers=1,
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
train_cfg = dict(max_iters=3190, type='IterBasedTrainLoop', val_interval=319)
train_dataloader = dict(
    batch_size=4,
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
    num_workers=1,
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
    num_workers=1,
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
    save_dir='vis_ycp',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './ycp_test'
