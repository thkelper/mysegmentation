crop_size = (
    512,
    1024,
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
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/home/yangchangpeng/wing_studio/data/col_train_dataset'
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
load_from = '/home/yangchangpeng/wing_studio/mysegmentation/results/hrnet/best_mIoU_iter_5104.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    18,
                    36,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    18,
                    36,
                    72,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    18,
                    36,
                    72,
                    144,
                ),
                num_modules=3)),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        type='HRNet'),
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
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=270,
        concat_input=False,
        dropout_ratio=-1,
        in_channels=[
            18,
            36,
            72,
            144,
        ],
        in_index=(
            0,
            1,
            2,
            3,
        ),
        input_transform='resize_concat',
        kernel_size=1,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        out_channels=1,
        type='FCNHead'),
    pretrained='open-mmlab://msra/hrnetv2_w18',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 2
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
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
train_cfg = dict(max_iters=6380, type='IterBasedTrainLoop', val_interval=638)
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
use_sigmoid = True
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
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './results/hrnet'
