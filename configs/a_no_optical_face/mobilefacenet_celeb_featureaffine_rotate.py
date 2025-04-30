import numpy as np
# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


# dataset settings
dataset_type = 'Celeb'
num_classes = 93955
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#W = np.array([[0.714, 0.714, 0],[-0.714, 0.714, 0]]).astype(np.float) 

memcached_root = '/mnt/lustre/share/memcached_client/'
train_pipeline = [
    dict(type='LoadImageFromFile',
         file_client_args=dict(backend='petrel')
         ),
    dict(type='Resize', size=(172, 172)),
    dict(type='Pad_celeb', size=(180, 172), padding=(0,8,0,0)),
    #dict(type='Pad', size=(180,172)),
    dict(type='CenterCrop', crop_size=(112, 96)),
    #dict(type='RandomResizedCrop', size=(112, 96)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='AffineRTS', angle=45.0, prob=1.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='TorchAffineRTS', angle=(0.0,90.0), prob=1.0, scale_factor=0.0, translate=(0.0, 0.0)),
    dict(type='Collect', keys=['img', 'gt_label', 'affine_matrix'])
]

test_pipeline = [
    dict(type='LoadImagePair'),
    dict(type='FlipPair', keys=['img1', 'img2'], keys_flip=['img1_flip', 'img2_flip']),
    # dict(type='AffineRTS', angle=45.0, prob=1.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img1', 'img1_flip', 'img2', 'img2_flip']),
    dict(type='ToTensor', keys=['fold', 'label']),
    dict(type='TorchAffineRTS', angle=(0.0,90.0), prob=1.0, scale_factor=0.0, translate=(0.0, 0.0)),
    dict(type='StackImagePair', keys=['img1', 'img1_flip', 'img2', 'img2_flip'], out_key='img'),
    dict(type='Collect', keys=['img', 'fold', 'label', 'affine_matrix'])
]

train_dir = 'sh1984:s3://VerifyInsight/Trillion_Pairs/celebrity/'
train_imglist = '/mnt/lustre/share_data/liuxin1/celeb_meta/celebrity_data.txt'
train_ann_file = '/mnt/lustre/share_data/liuxin1/celeb_meta/celebrity_label.txt'
# train_dir = '/mnt/lustre/shiwanxin.vendor/codes/MobileFaceNet_Pytorch-master/data/CASIA/CASIA-WebFace-112X96'
val_dir = '/mnt/lustre/shiwanxin.vendor/codes/MobileFaceNet_Pytorch-master/data/lfw/'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix=train_dir,
        imglist_root=train_imglist,
        label_root=train_ann_file,
        pipeline=train_pipeline),
    val=dict(
        type='LFW',
        img_prefix=val_dir + 'lfw-112X96',
        pair_file=val_dir + 'pairs.txt',
        pipeline=test_pipeline),
    test=dict(
        type='LFW',
        img_prefix=val_dir + 'lfw-112X96',
        pair_file=val_dir + 'pairs.txt',
        pipeline=test_pipeline),
)
evaluation = dict(interval=1, metric='accuracy')


# model settings
model = dict(
    type='AffineFaceImageClassifier',
    backbone=dict(type='MobileFaceNet_feature'),
    neck=dict(type='GlobalDepthWiseNeck',
              in_channels=512,
              out_channels=128,
              kernel_size=(7, 6)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=num_classes)
    )
)


# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
#lr_config = dict(
   # policy='CosineAnnealing',
   # min_lr=0,
   # warmup='linear',
   # warmup_iters=20000,
   # warmup_ratio=0.25)
lr_config = None
runner = dict(type='EpochBasedRunner', max_epochs=100)
