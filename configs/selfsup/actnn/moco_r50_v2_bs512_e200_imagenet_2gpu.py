_base_ = '../../base.py'
conv_cfg = dict(type='QConv2d')
norm_cfg = dict(type='QBN2d', requires_grad=True)
act_cfg = dict(type='QReLU')
# model settings
model = dict(
    type='MOCO',
    pretrained=None,
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    neck=dict(
        type='NonLinearNeckV1ActNN',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    return_label=False,
    memcached=False,
    mclient_path='/dev/shm/mclient/')
dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
    dict(type='RandomHorizontalFlip'),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=256,  # 256*2=512 (~10G)
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            root='data/imagenet/',
            list_file='data/imagenet/train_images.txt',
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ))
# optimizer
optimizer = dict(type='SGD', lr=0.06, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                entity='actnn',
                project='mmssl',
                name='moco_r50_v2_bs512_e200_imagenet_2gpu'
            )
        )
    ]
)
