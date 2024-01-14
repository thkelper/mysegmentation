_base_ = [
    './models/pspnet_r50-d8.py',
    './collagen_512x512.py', './default_runtime.py',
    './schedules/schedule_6380.py'
]
crop_size = (769, 769)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))


work_dir = "./results/pspnet"