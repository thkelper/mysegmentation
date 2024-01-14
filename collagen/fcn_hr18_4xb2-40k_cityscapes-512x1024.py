_base_ = [
    './models/fcn_hr18.py', './datasets/collagen_512x512.py',
    './default_runtime.py', './schedules/schedule_6380.py'
]

#######################################################################
#                        PART 1 Modified Settings                     #
#######################################################################
crop_size = (512, 1024)
num_classes=2
out_channels=1
use_sigmoid=True
work_dir = "./results/hrnet"
#######################################################################
#                      PART 2  Dataset & Dataloader                   #
#######################################################################
dataset_type = 'CollagenSegDataset'
data_root = '/home/yangchangpeng/wing_studio/data/col_dataset'
img_scale = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotationsBinary', binary=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='LoadAnnotations'),
    # dict(type='LoadAnnotationsBinary', binary=True),
    dict(type='PackSegInputs')
] 
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        seg_map_suffix=".png",
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img_path='img_dir/train/', seg_map_path='ann_dir/train/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        seg_map_suffix=".png",
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])

#######################################################################
#                             PART 3  Model                           #
#######################################################################
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
            decode_head=dict(
                num_classes=num_classes,
                out_channels=out_channels,
                loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=use_sigmoid, loss_weight=1.0))
            )
#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=6380,
        by_epoch=False)
]
# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=6380, val_interval=638)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=638, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=638, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
