# ------------------------------------------------------------
# Combined SegFormer‐MiT-B5 config for fine‐tuning on “Malezas”
# - Backbone: MiT‐B5 (pretrained)
# - Decoder: SegFormerHead with num_classes=5
# - Backbone frozen (lr_mult=0)
# - Crop size: 512×512, sliding‐window inference
# - Scheduler: 40k total iters, LinearLR warmup + PolyLR
# ------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
    '../_base_/datasets/malezas.py'  # Dataset config (5 classes) with pipelines
]

# ============= Data Preprocessor (mean/std/pad + crop size) =============
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

# ============= Model Definition =============
# Here define the path to the weights (.pth) to load from and start the fine tuning
checkpoint = '.../modelo3.pth'

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,  # we rely on init_cfg below
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        # ────── HERE IS THE ONLY CHANGE ──────
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
            prefix='backbone.'
        )
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode='whole'
    )
)


# ============= Optimizer & Freezing Backbone =============
# We freeze the backbone by setting lr_mult=0.0 for all backbone params.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            # Freeze entire backbone
            'backbone': dict(lr_mult=0.0),
            # Keep norm layers’ weight decay at zero
            'norm': dict(decay_mult=0.0),
            # Give head a higher lr (10×)
            'decode_head': dict(lr_mult=10.0)
        }
    )
)

# ============= Learning Rate Scheduler =============
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False
    )
]

# ============= DataLoader Settings =============
train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader


# ============= Scheduler Settings, shorter intervals =============

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, save_best='mIoU'),
)
