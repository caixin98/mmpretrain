import torch
optical = dict(
    type='BinaryPsfConv',
    feature_size=3.45e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    requires_grad=True,
    down="resize",
    n_psf_mask=1)
propagated_args = dict(
    mask2sensor=0.002,
    scene2mask=0.4,
    object_height=0.27,
    sensor='IMX250',
    single_psf=False,
    grayscale=False,
    input_dim=[112, 96, 3],
    output_dim=[308, 257, 3],
    dtype_out=torch.float32)
checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'Celeb'
num_classes = 93955
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(172, 172)),
    dict(type='Pad_celeb', size=(180, 172), padding=(0, 8, 0, 0)),
    dict(type='CenterCrop', crop_size=(112, 96)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Propagated',
        mask2sensor=0.002,
        scene2mask=0.4,
        object_height=0.27,
        sensor='IMX250',
        single_psf=False,
        grayscale=False,
        input_dim=[112, 96, 3],
        output_dim=[308, 257, 3],
        dtype_out=torch.float32),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
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
        output_dim=[308, 257, 3],
        dtype_out=torch.float32),
    dict(type='ToTensor', keys=['fold', 'label']),
    dict(
        type='StackImagePair',
        keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
        out_key='img'),
    dict(type='Collect', keys=['img', 'fold', 'label'])
]
train_dir = '/mnt/workspace/RawSense/data/celebrity/'
train_imglist = '/mnt/workspace/RawSense/data/celebrity/celebrity_data.txt'
train_ann_file = '/mnt/workspace/RawSense/data/celebrity/celebrity_label.txt'
val_dir = '/mnt/workspace/RawSense/data/lfw/'
data = dict(
    workers_per_gpu=2,
    train=dict(
        type='Celeb',
        img_prefix='/mnt/workspace/RawSense/data/celebrity/',
        imglist_root=
        '/mnt/workspace/RawSense/data/celebrity/celebrity_data.txt',
        label_root='/mnt/workspace/RawSense/data/celebrity/celebrity_label.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(172, 172)),
            dict(type='Pad_celeb', size=(180, 172), padding=(0, 8, 0, 0)),
            dict(type='CenterCrop', crop_size=(112, 96)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Propagated',
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3],
                dtype_out=torch.float32),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='LFW',
        img_prefix='/mnt/workspace/RawSense/data/lfw/lfw-112X96',
        pair_file='/mnt/workspace/RawSense/data/lfw/pairs.txt',
        pipeline=[
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
                output_dim=[308, 257, 3],
                dtype_out=torch.float32),
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                out_key='img'),
            dict(type='Collect', keys=['img', 'fold', 'label'])
        ]),
    test=dict(
        type='LFW',
        img_prefix='/mnt/workspace/RawSense/data/lfw/lfw-112X96',
        pair_file='/mnt/workspace/RawSense/data/lfw/pairs.txt',
        pipeline=[
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
                output_dim=[308, 257, 3],
                dtype_out=torch.float32),
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                out_key='img'),
            dict(type='Collect', keys=['img', 'fold', 'label'])
        ]),
    train_dataloader=dict(samples_per_gpu=128),
    val_dataloader=dict(samples_per_gpu=32))
evaluation = dict(interval=1, metric='accuracy')
custom_hooks = [
    dict(type='VisualConvHook'),
    dict(type='VisualAfterOpticalHook')
]
model = dict(
    type='FaceImageClassifier',
    backbone=dict(
        type='MobileFaceNet_feature_optical',
        optical=dict(
            type='BinaryPsfConv',
            feature_size=3.45e-05,
            sensor='IMX250',
            input_shape=[3, 308, 257],
            down = "resize",
            scene2mask=0.4,
            mask2sensor=0.002,
            target_dim=[112, 96],
            requires_grad=True,
            n_psf_mask=1,
            mask_kernel_shape=[27])),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=512,
        out_channels=128,
        kernel_size=(8, 6)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)))
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=20000,
    warmup_ratio=0.25)
runner = dict(type='EpochBasedRunner', max_epochs=30)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
