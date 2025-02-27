# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp


from tools.data_converter import indoor_converter as indoor
from tools.data_converter import kitti_converter as kitti
from tools.data_converter import lyft_converter as lyft_converter
from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)

from pypcd import pypcd
import numpy as np
import struct
import random

def read_PC(pcd_file):
    """
        parameter
            .pcd format file.

        return
            np.array, nx3, xyz coordinates of given pointcloud points.
    """
    if pcd_file[-3:]=="bin":
        size_float = 4
        list_pcd = []
        with open(pcd_file, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)
        np_pcd = np.asarray(list_pcd)
        return np_pcd

    points_pcd = pypcd.PointCloud.from_path(pcd_file)
    x = points_pcd.pc_data["x"].copy()
    y = points_pcd.pc_data["y"].copy()
    z = points_pcd.pc_data["z"].copy()
    return np.array([x,y,z]).T

def cross(a, b):
    """
        Cross product of 2d points.

        parameter
            a: np.array, size nx2, point vectors to check 
            b: np.array, size 1x2, line vectors  
        
        return
            c: np.array, size nx2, 
            if c[i] is positive, a[i] is to the right of b, 
            elif c[i] is negative, a[i] is to the left of b. 
    """
    c = a[:,0]*b[1] - a[:,1]*b[0]
    return c 

def check_inside_convex_polygon(points, vertices):
    """
        The function to check indices of points which are inside of a given convex polygon.
        
        parameter
            points: np.array, size nx2, 2d points vectors.
            vertices: np.array, size mx2, vertices of convex polygon, assume that the order of vertices is unknown. 
        
        return 
            indices of the points that are inside of the given convex polygon.
    """
    roi1, roi2 = True, True
    vertex_num = len(vertices)
    for i, v in enumerate(vertices):
        roi1 &= (cross(points - v, vertices[(i + 1)%vertex_num] - v) > 0)
        roi2 &= (cross(points - v, vertices[(i + 1)%vertex_num] - v) < 0)
    return roi1 | roi2

def get_original_box_xy_coords(box_info):
    cx,cy,cz,w,l,h,theta = box_info
    rotation_mtx_T = np.array([[np.cos(theta),np.sin(theta)],
                                [-np.sin(theta),np.cos(theta)]])
    xys = np.array([[-w/2, -l/2],[w/2, -l/2],[w/2, l/2],[-w/2, l/2]])
    rotated_xys = xys @ rotation_mtx_T
    rotated_xys[:,0] += cx
    rotated_xys[:,1] += cy
    return rotated_xys

def get_estimated_z_h(roi_path, box_annot):
    points_xyz = read_PC(roi_path)
    cz_list,h_list = [], []
    Z_MIN_DEFALUT = -1
    H_DEFAULT = 2.0
    PED_H_DEFAULT = 1.7

    for single_annot in box_annot:
        label_cls = single_annot[0]
        box_coor = single_annot[1:].astype(np.float32)
        rotated_xys = get_original_box_xy_coords(box_coor)
        roi_points = points_xyz[check_inside_convex_polygon(points_xyz[:, :2], rotated_xys)]
        try:
            q75, q25 = np.percentile(roi_points[:, 2], [75,25])
            iqr = q75 - q25
            upper_valid = roi_points[:, 2] <= q75 + 1.5 * iqr
            lower_valid = roi_points[:, 2] >= q25 - 1.5 * iqr
            roi_points = roi_points[upper_valid & lower_valid]
            q95, q5 = np.percentile(roi_points[:, 2], [95,5])
            z1, z2 = q5-0.3, q95+0.3

            h = 1.9+0.2*random.random() if label_cls != 'Pedestrian' else 1.6+0.2*random.random()

            # 이 부분을 object class에 따라 구분
            if label_cls == 'Pedestrian' and abs(z2 - z1) > 2.5:
                z2 = z1 + h
            elif label_cls != 'Pedestrian' and abs(z2 - z1) > 4.2:
                z2 = z1 + h

            # z1, z2 = min(roi_points[:,2]), max(roi_points[:,2])
            cz, h = int((z1+z2)/2 * 100)/100, int(abs(z2-z1)*100)/100
        except:
            # edge case, if xy-plane labeling is invalid, roi_buf could be empty.
            # then just set z, h as default value
            cz, h = Z_MIN_DEFALUT, H_DEFAULT
            if label_cls == "Pedestrian":
                h = PED_H_DEFAULT
        cz_list.append(cz)
        h_list.append(h)
    return np.array(cz_list), np.array(h_list)

def additional_height_modifier(box_annot):
    default_h=1.75

    is_ped = (box_annot[:, 0] == 'Pedestrian')
    is_veh = (box_annot[:, 0] != 'Pedestrian')
    annots = box_annot[:, 1:].astype(np.float32)
    
    ped_valid_height_cond = (annots[:, 5] <= 2) & (annots[:, 5] >= 1.5)
    veh_valid_height_cond = (annots[:, 5] <= 3) & (annots[:, 5] >= 1.5)
    ped_invalid_height_cond = (annots[:, 5] > 2.5) | (annots[:, 5] < 0.8)
    veh_invalid_height_cond = (annots[:, 5] > 4.5) | (annots[:, 5] < 1)

    valid_height_cond = (is_ped & ped_valid_height_cond) | (is_veh & veh_valid_height_cond)
    invalid_height_cond = (is_ped & ped_invalid_height_cond) | (is_veh & veh_invalid_height_cond)

    #Kitti dataset은 2, RFdataset은 5
    if sum(valid_height_cond) < 5: return

    floors = annots[valid_height_cond, 2] - 0.5 * annots[valid_height_cond, 5]
    max_floor = np.max(floors)
    min_floor = np.min(floors)
    floor = np.mean(floors)

    floating_in_the_air_cond = invalid_height_cond & ((annots[:, 2] + 0.5*annots[:, 5]) > floor + default_h) 
    under_the_floor_cond = invalid_height_cond & ((annots[:, 2] - 0.5*annots[:, 5]) < floor)
    other_abnormal_cond = (invalid_height_cond & 
                            ((annots[:, 2] + 0.5*annots[:, 5]) <=  floor + default_h) &
                            ((annots[:, 2] - 0.5*annots[:, 5]) >= floor))

    annots[other_abnormal_cond, 2] = floor + default_h*0.5
    annots[floating_in_the_air_cond, 2] += annots[floating_in_the_air_cond, 5]*0.5 - default_h*0.5
    annots[under_the_floor_cond, 2] += -annots[under_the_floor_cond, 5]*0.5 + default_h*0.5
    annots[invalid_height_cond, 5] = default_h

    box_annot[:, 1:] = annots

def rf2021_data_prep(root_path,
                     info_prefix,
                     seq):
    """ Prepare data related to RF2021 dataset.

    1. loop pcd file directory.
    2. vehicle (car) 과 Pedestrian의 label의 형태를 (cls, x, y, z, dx, dy, dz, theta)로 통일
        * 각도를 바꾸는 이유는 우리 데이터셋의 좌표계와 mmdetection3d의 좌표계가 갖는 각도에 대한 기준이 달라서 이를 맞춰줌
    3. z값이 제대로 안만들어져 있는 경우에는 해당 box내에 포인트들을 적절히 봐서 임의로 z 값 채움
    4. custom3DDataset의 형태로 저장. train,val,test 나눠서 pickle 파일로 저장. 

    """

    # 원본 label 데이터 (이전 형식 txt, nan 포함된 label)에서 relabeling으로 돌릴 pkl 파일을 만드는 함수
    # Relabeling 후에 제대로 된 label값은 create_data.py 사용

    import os
    from pathlib import Path
    import numpy as np
    import mmcv
    from collections import deque
    from tqdm import tqdm
    import math
    import pickle

    root_path = Path(root_path)

    pcd_dir = osp.join(root_path, "NIA_tracking_data", "data") 
    NIA_dir = osp.join(root_path, "NIA_2021_label_final")  #원본 label 데이터 폴더
    label_dir = osp.join(NIA_dir, "label")
    sample_idx = 0
    annot_deque = deque([])
    
    if seq!=-1:
        seq_dir = osp.join(NIA_dir, "sequence_divisions", "sequence_set_"+str(seq))
        with open(osp.join(seq_dir,'sequence_train_set.pkl'),'rb') as f:
            train_set=pickle.load(f)

        with open(osp.join(seq_dir,'sequence_test_set.pkl'),'rb') as f:
            test_set=pickle.load(f)   

        train_idx=0
        test_idx=0
    
    if osp.isdir(pcd_dir):
        folder_list = sorted(os.listdir(pcd_dir), key=lambda x:int(x))
        for fol in tqdm(folder_list):
            if seq!=-1:
                if train_idx<len(train_set) and train_set[train_idx]==int(fol):
                    train_idx+=1
                elif test_idx<len(test_set) and test_set[test_idx]==int(fol):
                    test_idx+=1
                else:
                    continue
                    
            pcd_data_dir = osp.join(pcd_dir, fol, "lidar_half_filtered")
            veh_label_dir = osp.join(label_dir, fol, "car_label")
            ped_label_dir = osp.join(label_dir, fol, "ped_label")
            for pcd_file in sorted(os.listdir(pcd_data_dir)):
                pcd_file_path = osp.join(pcd_data_dir, pcd_file)
                veh_label_file_path = osp.join(veh_label_dir, pcd_file[:-4] + ".txt")
                ped_label_file_path = osp.join(ped_label_dir, pcd_file[:-4] + ".txt")
                annot_veh = np.array([])
                annot_ped = np.array([])
                if osp.exists(veh_label_file_path):
                    if os.path.getsize(veh_label_file_path):
                        annot_veh = np.loadtxt(veh_label_file_path, dtype=np.object_).reshape(-1, 8)
                        annot_veh[annot_veh == 'nan'] = '-1.00'

                        #txt 형식에 따라 주석 풀어줄 것.
                        annot_veh[:, [1,2,3,4,5,6]] = annot_veh[:, [4,5,6,2,1,3]]
                        annot_veh[:, 7] = math.pi/2 - annot_veh[:, 7].astype(np.float32)

                        #Relabeling을 위한 데이터 제작시 Cyclist를 Car로 통합. 새로 만든 txt로 pkl 만들때는 주석 처리해야함.
                        #원리상 Ped는 뒤에 붙어있어야하는데 Cyclist를 넣으면 Ped가 중간중간에 끼게 되어 (car개수)+(ped번호)로 index를 구할 수 없게 됨.
                        annot_veh[annot_veh == 'Cyclist'] = 'Car'           

                if osp.exists(ped_label_file_path):
                    if os.path.getsize(ped_label_file_path):
                        annot_ped = np.loadtxt(ped_label_file_path, dtype=np.object_).reshape(-1, 6)
                        annot_ped[annot_ped == 'nan'] = '-1.00'

                        #Relabeling을 위한 데이터 제작시 width, length에 랜덤한 값 넣어줌.
                        #랜덤하게 조정할 필요 없다면 아래 다섯줄 주석
                        annot_ped[annot_ped[:, 4] == '-1.00', 4] = str(0.65 + random.random() * 0.1)
                        annot_ped[annot_ped[:, 5] ==  '-1.00', 5] = str(0.65 + random.random() * 0.1)  
                        annot_cls = np.array([["Pedestrian"] for _ in range(len(annot_ped))])
                        annot_angle = np.array([[0] for _ in range(len(annot_ped))])
                        annot_ped = np.hstack((annot_cls, annot_ped, annot_angle))

                if len(annot_veh)>0 and len(annot_ped)>0:
                    annot = np.vstack((annot_veh, annot_ped))
                elif len(annot_veh)>0:
                    annot = np.copy(annot_veh)
                elif len(annot_ped)>0:
                    annot = np.copy(annot_ped)                           
                
                if len(annot):
                    invalid_cond = (annot[:, 3] == '-1.00') & (annot[:, 6] == '-1.00')
                    cz, h = get_estimated_z_h(pcd_file_path, annot[invalid_cond]) # 일부 z값이 없는 애들은 추가로 만들어줌
                    annot[invalid_cond, 3] = cz
                    annot[invalid_cond, 6] = h
                    additional_height_modifier(annot) # z값을 1차적으로 다 채워놓은 상태에서, invalid 해보이는 것들을 추가적으로 수정
                    pcd_file_path = "/".join(pcd_file_path.split("/")[2:])
                    annot_dict = dict(
                        sample_idx= sample_idx,
                        lidar_points= {'lidar_path': pcd_file_path},
                        annos= {'box_type_3d': 'LiDAR',
                                'gt_bboxes_3d': annot[:, 1:].astype(np.float32),
                                'gt_names': annot[:, 0]
                                }
                    )
                    annot_deque.append(annot_dict)
                    sample_idx += 1
    else:
        print("no data dir")
        print("Please check data dir path")
        exit()

    annot_list = list(annot_deque)
    rf_infos_train = []
    rf_infos_test = []

    train_idx=0
    test_idx=0
    
    cur_seq=0
    next_seq=0

    if seq!=-1:
        for i in range(len(annot_list)):
            cur_seq=int(annot_list[i]['lidar_points']['lidar_path'][23:28])
            seq_change=False

            if i<len(annot_list)-1:
                next_seq=int(annot_list[i+1]['lidar_points']['lidar_path'][23:28])
                if cur_seq != next_seq:
                    seq_change=True

            if train_idx<len(train_set) and train_set[train_idx]==cur_seq:
                rf_infos_train.append(annot_list[i])
                if seq_change:
                    train_idx+=1

            elif test_idx<len(test_set) and test_set[test_idx]==cur_seq:
                rf_infos_test.append(annot_list[i])
                if seq_change:
                    test_idx+=1
    else:
        total_len = len(annot_list)
        train_len = int(total_len * 0.9)
        rf_infos_train = annot_list[:train_len]
        rf_infos_test = annot_list[train_len:]

    filename = root_path / f'{info_prefix}_infos_train.pkl'
    print(f'RF2021 info train file is saved to {filename}')
    mmcv.dump(rf_infos_train, filename)
    
    filename = root_path / f'{info_prefix}_infos_test.pkl'
    print(f'RF2021 info test file is saved to {filename}')
    mmcv.dump(rf_infos_test, filename)

    filename = root_path / f'{info_prefix}_infos_train_small.pkl'
    print(f'RF2021 info train(interval 20) file is saved to {filename}')
    mmcv.dump(rf_infos_train[1::20], filename)

    #create_groundtruth_database('Custom3DDataset', root_path, info_prefix,
    #                        root_path / f'{info_prefix}_infos_train_small.pkl')

def weak_kitti_data_prep(root_path,
                    info_prefix):
    """ change kitti data to RF2021 dataset format.

    1. loop pcd file directory.
    2. vehicle (car) 과 Pedestrian의 label의 형태를 (cls, x, y, z, dx, dy, dz, theta)로 통일
        * 각도를 바꾸는 이유는 우리 데이터셋의 좌표계와 mmdetection3d의 좌표계가 갖는 각도에 대한 기준이 달라서 이를 맞춰줌
    3. z값이 제대로 안만들어져 있는 경우에는 해당 box내에 포인트들을 적절히 봐서 임의로 z 값 채움
    4. custom3DDataset의 형태로 저장. train,val,test 나눠서 pickle 파일로 저장. 

    """
    import os
    from pathlib import Path
    import numpy as np
    import mmcv
    from collections import deque
    from tqdm import tqdm
    import math
    import pickle
    import random

    ped_xyset_ratio=60 # 보행자 레이블 중, 수정할 비율
    all_hset_ratio=100 # get_estiamted_z로 높이를 수정할 비율

    root_path = Path(root_path)

    #gt_database만 만들기 위한 임시 code
    #create_groundtruth_database('Custom3DDataset', root_path, f'{info_prefix}_{ped_xyset_ratio}',
    #                       root_path / f'{info_prefix}_{ped_xyset_ratio}_infos_train.pkl')
    #exit()

    #kitti의 형식을 그대로 가져온 뒤 kitti의 데이터 변환 함수를 그대로 통과시키고 그 값을 rfdataset 형식으로 바꾼다.
    data={}
    train_data_dir = osp.join(root_path,"datasets/kitti","kitti_infos_train.pkl")
    val_data_dir = osp.join(root_path,"datasets/kitti","kitti_infos_val.pkl")

    with open(train_data_dir,'rb') as f:
	    data['train'] = pickle.load(f)
    with open(val_data_dir,'rb') as f:
        data['val'] = pickle.load(f)

    sample_idx = 0
    annot_deque_train = deque([])
    annot_deque_val = deque([])
    
    for setting in data.keys():
        for info in tqdm(data[setting]):
            rect = info['calib']['R0_rect'].astype(np.float32)
            Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

            annos = info['annos']
            loc = annos['location']
            dims = annos['dimensions']
            rots = annos['rotation_y']
            gt_names = np.array(annos['name'][:]).astype(np.object_)
            gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                        axis=1).astype(np.float32)

            gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
                Box3DMode.LIDAR, np.linalg.inv(rect @ Trv2c))
            gt_bboxes_3d = gt_bboxes_3d.tensor.numpy()
            gt_bboxes_3d[:,2] += gt_bboxes_3d[:,5]/2.0
            
            #kitti와 rfdataset label 통합 (Car/Pedestrian/Cyclist/(Dontcare))
            gt_names[gt_names == 'Van'] = 'Car'
            gt_names[gt_names == 'Truck'] = 'Car'
            gt_names[gt_names == 'Tram'] = 'Car'
            gt_names[gt_names == 'Misc'] = 'DontCare'
            gt_names[gt_names == 'Person_sitting'] = 'DontCare'

            annot = np.hstack((gt_names.reshape(-1,1), gt_bboxes_3d))
            annot = np.delete(annot,gt_names=='DontCare',axis=0)

            if len(annot):
                bin_dir=osp.join("training/velodyne_reduced/",info['point_cloud']['velodyne_path'][-10:])

                #if setting == 'train': (double set으로 하기 위해 train/val 구분 X)
                annot_ped = (annot[:, 0] == 'Pedestrian')
                annot_xyset_rand = np.array(random.sample(range(1,101),len(annot[:, 0]))) <= ped_xyset_ratio
                annot_hset_rand = np.array(random.sample(range(1,101),len(annot[:, 0]))) <= all_hset_ratio

                annot_xy_change = annot_ped * annot_xyset_rand
                annot[annot_xy_change, 4] = str(0.6 + random.random() * 0.1)
                annot[annot_xy_change, 5] = str(0.8 + random.random() * 0.1)

                cz, h = get_estimated_z_h(osp.join(root_path,bin_dir), annot[annot_hset_rand])
                annot[annot_hset_rand, 3] = cz
                annot[annot_hset_rand, 6] = h
                additional_height_modifier(annot)

                #ped를 뒤쪽에 모아서 둠. relabeling 함수 특성상 ped가 뒤쪽에 모여있어야 제대로 돌아가기 때문.
                annot_nonped = annot[annot[:,0] != 'Pedestrian']
                annot_ped = annot[annot[:,0] == 'Pedestrian']
                annot = np.vstack((annot_nonped, annot_ped))

                annot_dict = dict(
                    sample_idx= sample_idx,
                    lidar_points= {'lidar_path': bin_dir },
                    annos= {'box_type_3d': 'LiDAR',
                            'gt_bboxes_3d': annot[:, 1:].astype(np.float32),
                            'gt_names': annot[:, 0]
                            }
                )

                if setting == 'train':
                    annot_deque_train.append(annot_dict)
                elif setting == 'val':
                    annot_deque_val.append(annot_dict)
                sample_idx += 1

    annot_list_train = list(annot_deque_train)
    annot_list_val = list(annot_deque_val)

    weak_kitti_infos_train = annot_list_train[:]
    filename = root_path / f'{info_prefix}_{ped_xyset_ratio}_infos_train.pkl'
    print(f'Weak Kitti info train file is saved to {filename}')
    mmcv.dump(weak_kitti_infos_train, filename)

    weak_kitti_infos_val = annot_list_val[:]
    filename = root_path / f'{info_prefix}_{ped_xyset_ratio}_infos_val.pkl'
    print(f'Weak Kitti info val file is saved to {filename}')
    mmcv.dump(weak_kitti_infos_val, filename)

    create_groundtruth_database('Custom3DDataset', root_path, f'{info_prefix}_{ped_xyset_ratio}',
                           root_path / f'{info_prefix}_{ped_xyset_ratio}_infos_train.pkl')

def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path,
                                  f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)

    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
        return

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    nuscenes_converter.export_2d_annotation(
        root_path, info_train_path, version=version)
    nuscenes_converter.export_2d_annotation(
        root_path, info_val_path, version=version)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{out_dir}/{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 5. Here we store pose information of these frames
            for later use.
    """
    from tools.data_converter import waymo_converter as waymo

    splits = ['training', 'validation', 'testing']
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split == 'testing'))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(
        out_dir, info_prefix, max_sweeps=max_sweeps, workers=workers)
    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--seq',
    type=int,
    default=-1,
    help='separating sequence fairly via Ped counts. -1 for nonuse.'
)
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            with_plane=args.with_plane)
    elif args.dataset == 'weak_kitti':
        weak_kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag
        )
    elif args.dataset == 'rf2021':
        rf2021_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            seq=args.seq
        )
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
