_base_ = [
    '../_base_/datasets/rf2021-3d-ped.py',
    # '../_base_/schedules/cyclic_40e.py', 
    '../_base_/schedules/cosine.py',
    '../_base_/default_runtime.py'
]

voxel_size = [0.16, 0.16, 0.1]

model = dict(
    type='SASSD',
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 80000)),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoderSASSD',
        in_channels=4,
        sparse_shape=[41, 1024, 768],
        order=('conv', 'norm', 'act'),
        base_channels=16,
        output_channels=64,
        encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        conv_out_features=((3,1,1),(2,1,1)),
        pointwise_size=160),
    backbone=dict(
        type='SECOND',
        in_channels=128,
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        out_channels=[128, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[256, 256, 256]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=768,
        feat_channels=768,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [-60, -60, -0.6, 60, 60, -0.6]          
            ],
            sizes=[ 
                [0.84, 0.91, 1.74]   # Pedestrian
            ],  
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)
        ),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
