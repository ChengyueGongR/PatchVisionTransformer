_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/cityscapes_1025x1025.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        ape=False,
        drop_path_rate=0.5,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=20, 
        dropout_ratio=0.1,
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=20,
        loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=1e-2,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 4 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
