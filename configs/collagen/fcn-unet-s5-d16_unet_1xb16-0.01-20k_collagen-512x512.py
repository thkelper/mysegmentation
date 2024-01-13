import sys
_base_ = [
    './collagen_512x512.py',
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

img_scale = (512, 512)
data_preprocessor = dict(size=img_scale)
optimizer = dict(lr=0.01)
optim_wrapper = dict(optimizer=optimizer)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=1000,
        by_epoch=False)
]
# training schedule for 10 epochs
train_cfg = dict(type='IterBasedTrainLoop', max_iters=3190, val_interval=319)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=319, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=319))



work_dir="./ycp_test"