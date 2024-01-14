_base_ = [
    './models/psanet_r50-d8.py',
    '.collagen_512x512.py', './default_runtime.py',
    './schedules/schedule_6380.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=1),
    auxiliary_head=dict(num_classes=1))
