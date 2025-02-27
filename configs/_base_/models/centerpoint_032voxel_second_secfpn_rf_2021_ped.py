voxel_size = [0.32, 0.32, 0.1]

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1],
        voxel_size=voxel_size,
        max_voxels=(40000, 40000)),

    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),

    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 512, 384],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),


    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),


    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=['Pedestrian'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),  
            #sub-voxel location refinement R2
            #height above ground R1
            #3D size R3
            #yaw rotation angle R2
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            # post_center_range=[-60, -103.84, -3, 62.88, 60, 1],
            post_center_range=[-60, -60, -3, 60, 60, 1],
            max_num=100,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7,
			pc_range=[-60, -103.84]
            ),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),


    train_cfg=dict(
	
        pts=dict(
            point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1],
            grid_size=[384, 512, 41],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            point_cloud_range=[-60, -103.84, -3, 62.88, 60, 1],
            post_center_limit_range=[-60, -60, -3, 60, 60, 1],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=512,
            nms_thr=0)))














