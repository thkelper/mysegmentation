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



