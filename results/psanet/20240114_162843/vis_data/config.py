crop_size = (
    480,
    480,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        127.5,
        127.5,
        127.5,
    ],
    pad_val=0,
    seg_pad_val=0,
    size=(
        480,
        480,
    ),
    std=[
        127.5,
        127.5,
        127.5,
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
find_unused_parameters = True
img_scale = (
    512,
    512,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        class_embed_path=
        'https://download.openmmlab.com/mmsegmentation/v0.5/vpd/nyu_class_embeddings.pth',
        class_embed_select=True,
        diffusion_cfg=dict(
            base_learning_rate=0.0001,
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/vpd/stable_diffusion_v1-5_pretrain_third_party.pth',
            params=dict(
                channels=4,
                cond_stage_config=dict(
                    target='ldm.modules.encoders.modules.AbstractEncoder'),
                cond_stage_key='txt',
                cond_stage_trainable=False,
                conditioning_key='crossattn',
                first_stage_config=dict(
                    params=dict(
                        ddconfig=dict(
                            attn_resolutions=[],
                            ch=128,
                            ch_mult=[
                                1,
                                2,
                                4,
                                4,
                            ],
                            double_z=True,
                            dropout=0.0,
                            in_channels=3,
                            num_res_blocks=2,
                            out_ch=3,
                            resolution=256,
                            z_channels=4),
                        embed_dim=4,
                        lossconfig=dict(target='torch.nn.Identity'),
                        monitor='val/rec_loss'),
                    target='ldm.models.autoencoder.AutoencoderKL'),
                first_stage_key='jpg',
                image_size=64,
                linear_end=0.012,
                linear_start=0.00085,
                log_every_t=200,
                monitor='val/loss_simple_ema',
                num_timesteps_cond=1,
                scale_factor=0.18215,
                scheduler_config=dict(
                    params=dict(
                        cycle_lengths=[
                            10000000000000,
                        ],
                        f_max=[
                            1.0,
                        ],
                        f_min=[
                            1.0,
                        ],
                        f_start=[
                            1e-06,
                        ],
                        warm_up_steps=[
                            10000,
                        ]),
                    target='ldm.lr_scheduler.LambdaLinearScheduler'),
                timesteps=1000,
                unet_config=dict(
                    params=dict(
                        attention_resolutions=[
                            4,
                            2,
                            1,
                        ],
                        channel_mult=[
                            1,
                            2,
                            4,
                            4,
                        ],
                        context_dim=768,
                        image_size=32,
                        in_channels=4,
                        legacy=False,
                        model_channels=320,
                        num_heads=8,
                        num_res_blocks=2,
                        out_channels=4,
                        transformer_depth=1,
                        use_checkpoint=True,
                        use_spatial_transformer=True),
                    target='ldm.modules.diffusionmodules.openaimodel.UNetModel'
                ),
                use_ema=False),
            target='ldm.models.diffusion.ddpm.LatentDiffusion'),
        pad_shape=512,
        type='VPD',
        unet_cfg=dict(use_attn=False)),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        pad_val=0,
        seg_pad_val=0,
        size=(
            480,
            480,
        ),
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        fmap_border=(
            1,
            1,
        ),
        in_channels=[
            320,
            640,
            1280,
            1280,
        ],
        max_depth=10,
        num_classes=2,
        out_channels=1,
        type='VPDDepthHead'),
    test_cfg=dict(
        crop_size=(
            480,
            480,
        ), mode='slide_flip', stride=(
            160,
            160,
        )),
    type='DepthEstimator')
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
stable_diffusion_cfg = dict(
    base_learning_rate=0.0001,
    checkpoint=
    'https://download.openmmlab.com/mmsegmentation/v0.5/vpd/stable_diffusion_v1-5_pretrain_third_party.pth',
    params=dict(
        channels=4,
        cond_stage_config=dict(
            target='ldm.modules.encoders.modules.AbstractEncoder'),
        cond_stage_key='txt',
        cond_stage_trainable=False,
        conditioning_key='crossattn',
        first_stage_config=dict(
            params=dict(
                ddconfig=dict(
                    attn_resolutions=[],
                    ch=128,
                    ch_mult=[
                        1,
                        2,
                        4,
                        4,
                    ],
                    double_z=True,
                    dropout=0.0,
                    in_channels=3,
                    num_res_blocks=2,
                    out_ch=3,
                    resolution=256,
                    z_channels=4),
                embed_dim=4,
                lossconfig=dict(target='torch.nn.Identity'),
                monitor='val/rec_loss'),
            target='ldm.models.autoencoder.AutoencoderKL'),
        first_stage_key='jpg',
        image_size=64,
        linear_end=0.012,
        linear_start=0.00085,
        log_every_t=200,
        monitor='val/loss_simple_ema',
        num_timesteps_cond=1,
        scale_factor=0.18215,
        scheduler_config=dict(
            params=dict(
                cycle_lengths=[
                    10000000000000,
                ],
                f_max=[
                    1.0,
                ],
                f_min=[
                    1.0,
                ],
                f_start=[
                    1e-06,
                ],
                warm_up_steps=[
                    10000,
                ]),
            target='ldm.lr_scheduler.LambdaLinearScheduler'),
        timesteps=1000,
        unet_config=dict(
            params=dict(
                attention_resolutions=[
                    4,
                    2,
                    1,
                ],
                channel_mult=[
                    1,
                    2,
                    4,
                    4,
                ],
                context_dim=768,
                image_size=32,
                in_channels=4,
                legacy=False,
                model_channels=320,
                num_heads=8,
                num_res_blocks=2,
                out_channels=4,
                transformer_depth=1,
                use_checkpoint=True,
                use_spatial_transformer=True),
            target='ldm.modules.diffusionmodules.openaimodel.UNetModel'),
        use_ema=False),
    target='ldm.models.diffusion.ddpm.LatentDiffusion')
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
work_dir = './results/psanet'
