
# Copyright (c) OpenMMLab. All rights reserved.
from ast import comprehension
import mmcv
import numpy as np
import pytest
import torch
import pickle
from pypcd import pypcd
import open3d as o3d


from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.bbox import Coord3DMode
from mmdet3d.core.points import DepthPoints, LiDARPoints
# yapf: disable
# from mmdet3d.datasets import (AffineResize, BackgroundPointsFilter,
#                               GlobalAlignment, GlobalRotScaleTrans,
#                               ObjectNameFilter, ObjectNoise, ObjectRangeFilter,
#                               ObjectSample, PointSample, PointShuffle,
#                               PointsRangeFilter, RandomDropPointsColor,
#                               RandomFlip3D, RandomJitterPoints,
#                               RandomShiftScale, VoxelBasedPointSampler)
from mmdet3d.datasets import (RandomFlip3D)

HEADER =  '''\
VERSION 0.7
FIELDS x y z intensity ring
SIZE 4 4 4 1 1 
TYPE F F F U U 
COUNT 1 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA binary_compressed
'''


def read_pcd_data(pcd_path):
        points_pcd = pypcd.PointCloud.from_path(pcd_path)
        pcd_x = points_pcd.pc_data["x"].copy()
        pcd_y = points_pcd.pc_data["y"].copy()
        pcd_z = points_pcd.pc_data["z"].copy()
        pcd_intensity = points_pcd.pc_data["intensity"].copy().astype(np.float32)/256
        pcd_tensor = np.array([pcd_x, pcd_y, pcd_z, pcd_intensity], dtype=np.float32).T
        return points_pcd, pcd_tensor   

def test_random_flip_3d():
    pcd_object, points = read_pcd_data("./data/rf2021/NIA_tracking_data/data/10002/lidar_half_filtered/10002_000.pcd")
    # pcd_object1, points1 = read_pcd_data("../data/result.pcd")

    points = LiDARPoints(points, points_dim=4)
    points.flip()
    point = points.tensor.numpy()

    pcd_object.pc_data["x"] = point[:,0]
    pcd_object.pc_data["y"] = point[:,1]
    pcd_object.pc_data["z"] = point[:,2]
    pcd_object.pc_data["intensity"] = point[:,3]
    pcd_object.save_pcd('./data/result.pcd', comprehension='binary_compressed')

    # points = input_dict['points'].tensor.numpy()
    # gt_bboxes_3d = input_dict['gt_bboxes_3d'].tensor
    # with open('result.pcd', 'w') as wb:
    #     wb.write(HEADER.format(len(point), len(point))+'\n') 
    #     np.savetxt(wb,point,delimiter =  '' , fmt=  '%f %f %f %u')
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point)
    # o3d.io.write_point_cloud("flip_one.pcd", pcd)

if __name__ == '__main__':
    test_random_flip_3d()