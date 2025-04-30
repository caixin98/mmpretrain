_base_ = '../../_base_/default_runtime.py'

optical = dict(
    type='SoftPsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    requires_grad=True,
    use_stn=False,
    down="resize",
    noise_type="gaussian",
    n_psf_mask=1)

find_unused_parameters = True

env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=True,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'Celeb'
num_classes = 93955


train_pipeline = [
            dict(type='LoadImageFromFile2'),
            dict(type='Resize', scale=(172, 172)),
            dict(type='CenterCrop', crop_size=(112, 96)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='Propagated',
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]
                ),
            dict(
                    type='TorchAffineRTS',
                    scale_factor=0.2,
                    translate=(0.2, 0.2),
                    prob=1.0,
                ),
            # dict(type='Collect', keys=['img', 'gt_label', 'affine_matrix']),
            dict(type='PackInputs',algorithm_keys = ["affine_matrix"] ),
        ]

val_pipeline = [
            dict(type='LoadImagePair'),
            dict(
                type='FlipPair',
                keys=['img1', 'img2'],
                keys_flip=['img1_flip', 'img2_flip']),
            dict(
                type='Propagated',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]),
            dict(
                    type='TorchAffineRTS',
                    scale_factor=0.2,
                    translate=(0.2, 0.2),
                    prob=1.0,
                ),
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                out_key='img')
        ]

train_dataloader = dict(
    batch_size=160,
    num_workers=4,
    dataset = dict(
        type='Celeb',
        img_prefix='/mnt/workspace/RawSense/data/celebrity/',
        imglist_root=
        '/mnt/workspace/RawSense/data/celebrity/celebrity_data.txt',
        label_root='/mnt/workspace/RawSense/data/celebrity/celebrity_label.txt',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler',shuffle=True))
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset = dict(
        type='LFW',
        img_prefix='/mnt/workspace/RawSense/data/lfw/lfw-112X96',
        pair_file='/mnt/workspace/RawSense/data/lfw/pairs.txt',
        pipeline=val_pipeline,
   ),
    sampler=dict(type='DefaultSampler',shuffle=False))
test_dataloader = val_dataloader

custom_hooks = [
    dict(type='VisualConvHook'),
    dict(type='VisualAfterOpticalHook')
]
# model settings
model = dict(
    type='AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT_optical',
        optical=optical,
        image_size=168),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(11, 11)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=1, norm_type=2))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=False,
        end = 20000,
      ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
      ),
]
train_cfg = dict(by_epoch=True,max_epochs=100,val_interval=1)
val_cfg = dict()
test_cfg = dict()
val_evaluator = dict(type='Accuracy')  
test_evaluator = val_evaluator
visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)

default_scope = 'mmpretrain'