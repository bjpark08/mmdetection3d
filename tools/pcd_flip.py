
# Copyright (c) OpenMMLab. All rights reserved.
from ast import comprehension
import mmcv
import numpy as np
import pytest
import torch
import pickle
from pypcd import pypcd
import open3d as o3d
from os import path as osp
from pathlib import Path
from collections import deque
from tqdm import tqdm
import math
import os



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
    root_path = Path("data")
    pcd_dir = osp.join(root_path, "rf2021", "NIA_tracking_data", "data")
    save_pcd_dir = osp.join(root_path, "rf2021", "NIA_tracking_data_fliped", "data")
    label_dir = osp.join(root_path, "NIA_2021_label", "label")
    # pcd_new_dir = osp.join(root_path, )
    if osp.isdir(pcd_dir):
        folder_list = sorted(os.listdir(pcd_dir), key=lambda x:int(x))
        for fol in tqdm(folder_list):
            pcd_data_dir = osp.join(pcd_dir, fol, "lidar_half_filtered")  
            save_pcd_data_dir = osp.join(save_pcd_dir, fol)
            if not osp.exists(save_pcd_data_dir):
                os.mkdir(save_pcd_data_dir)
                os.mkdir(osp.join(save_pcd_data_dir, "lidar_half_filtered"))

            save_pcd_data_dir = osp.join(save_pcd_data_dir, "lidar_half_filtered")
            veh_label_dir = osp.join(label_dir, fol, "car_label")
            ped_label_dir = osp.join(label_dir, fol, "ped_label")  

            for pcd_file in sorted(os.listdir(pcd_data_dir)):
                pcd_file_path = osp.join(pcd_data_dir, pcd_file)
                save_pcd_file_path = osp.join(save_pcd_data_dir, pcd_file)
                # veh_label_file_path = osp.join(veh_label_dir, pcd_file[:-4] + ".txt")
                # ped_label_file_path = osp.join(ped_label_dir, pcd_file[:-4] + ".txt")
                if osp.exists(save_pcd_file_path):
                    continue
                if osp.exists(pcd_file_path):
                    pcd_object, points = read_pcd_data(pcd_file_path)
                    points = LiDARPoints(points, points_dim=4)
                    points.flip()
                    point = points.tensor.numpy()
                    pcd_object.pc_data["x"] = point[:,0]
                    pcd_object.pc_data["y"] = point[:,1]
                    pcd_object.pc_data["z"] = point[:,2]
                    pcd_object.pc_data["intensity"] = point[:,3]
                    # save_pcd_file_path_name = osp.join(save_pcd_file_path, '.pcd')
                    pcd_object.save_pcd(save_pcd_file_path, comprehension='binary_compressed')

    #                 annot_deque.append(annot_dict)
    #                 sample_idx += 1
    # else:
    #     print("no data dir")
    #     print("Please check data dir path")
    #     exit()

    # annot_list = list(annot_deque)
    # total_len = len(annot_list)
    # train_len = int(total_len * 0.8)
    # val_len = int(total_len * 0.1)
    # rf_infos_train = annot_list[:train_len]
    # filename = root_path / f'{info_prefix}_infos_train.pkl'
    # print(f'RF2021 info train file is saved to {filename}')
    # mmcv.dump(rf_infos_train, filename)

    # rf_infos_val = annot_list[train_len:train_len + val_len]
    # filename = root_path / f'{info_prefix}_infos_val.pkl'
    # print(f'RF2021 info val file is saved to {filename}')
    # mmcv.dump(rf_infos_val, filename)

    # rf_infos_test = annot_list[train_len + val_len:]
    # filename = root_path / f'{info_prefix}_infos_test.pkl'
    # print(f'RF2021 info test file is saved to {filename}')
    # mmcv.dump(rf_infos_test, filename)



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