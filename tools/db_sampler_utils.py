import numpy as np
import pickle
import os    
from pathlib import Path
import mmcv
from pypcd import pypcd
import open3d as o3d
from os import path as osp
from mmdet3d.core.visualizer import (show_multi_modality_result, show_result, show_seg_result)
from mmdet3d.apis import init_model, inference_detector


def read_pcd_data(pcd_path):
    points_pcd = pypcd.PointCloud.from_path(pcd_path)
    pcd_x = points_pcd.pc_data["x"].copy()
    pcd_y = points_pcd.pc_data["y"].copy()
    pcd_z = points_pcd.pc_data["z"].copy()
    pcd_intensity = points_pcd.pc_data["intensity"].copy().astype(np.float32)/256
    pcd_tensor = np.array([pcd_x, pcd_y, pcd_z, pcd_intensity], dtype=np.float32).T
    return pcd_tensor   

def add_more_bus_sample():
    filepath = './rf2021_dbinfos_train.pkl'
    with open(filepath,'rb') as f:
        datas=pickle.load(f)

    cars = datas['Car']
    add_car_tensor = []
    add_car_size = []
    num_points = []
    image_idxs = []
    idx = 0
    flag = True
    last_image_idx = cars[-1]["image_idx"]

    for data in cars:
        if(np.absolute(data['box3d_lidar'][4]) >= 10 and data['num_points_in_gt'] >= 100):
            points = np.fromfile(data['path'], np.float32).reshape(-1, 4)
            pcd_x = points[:, 0].copy()
            pcd_y = points[:, 1].copy()
            pcd_z = points[:, 2].copy()      
            pcd_intensity = points[:, 3].copy().astype(np.float32)/256
            pcd_tensor = np.array([pcd_x, pcd_y, pcd_z, pcd_intensity], dtype=np.float32).T
            num_points.append(data['num_points_in_gt'])
            if(idx != 0):
                if(np.absolute(image_idxs[idx-1] - data['image_idx']) > 50): #시간별로 인접한 PCD 파일의 중복을 피하기 위함
                    add_car_tensor.append(pcd_tensor)
                    image_idxs.append(data['image_idx'])
                    box_coors = data['box3d_lidar']
                    box_coors[0] = box_coors[0] * 1.1
                    box_coors[1] = box_coors[1] * 1.1
                    add_car_size.append(box_coors)
                    idx = idx + 1
            else:
                add_car_tensor.append(pcd_tensor)
                image_idxs.append(data['image_idx'])
                box_coors = data['box3d_lidar']
                box_coors[0] = box_coors[0] * 1.1
                box_coors[1] = box_coors[1] * 1.1
                add_car_size.append(box_coors)
                idx = idx + 1

    for i in range(len(add_car_tensor)):
        pcd_x = add_car_tensor[i][0] * 1.1
        pcd_y = add_car_tensor[i][1] * 1.1
        pcd_tensor = np.array([pcd_x, pcd_y, add_car_tensor[i][2], add_car_tensor[i][3]], dtype = np.float32).T
        pcd_tensor = pcd_tensor.reshape(-1)
        last_image_idx = last_image_idx + 1
        with open('./new_bin/' + str(last_image_idx) + '_Car_1.bin', 'w') as f:
            pcd_tensor.tofile(f)
    

    last_image_idx = cars[-1]["image_idx"] + 1
    add_car_dict = []
    for i in range(len(add_car_size)):
        path = 'new_bin/' + str(last_image_idx) + '_Car_1.bin'
        image_idx = last_image_idx
        last_image_idx = last_image_idx + 1
        gt_idx = 1
        box3d_lidar = add_car_size[i]
        add_car_dict.append({'name': 'Car', 'path': path, 'image_idx': image_idx, 'gt_idx': 1, 'box3d_lidar': box3d_lidar, 'num_points_in_gt': num_points[i], 'difficulty': 0, 'group_id': None})


    with open('rf2021_dbinfos_train.pkl', 'rb') as f:
        data = pickle.load(f)
    
    data['Car'] = data['Car'] + add_car_dict
    augmented_dbinfos = {'Car': data['Car'], 'Pedestrian': data['Pedestrian']}
    with open('bus_augmented_rf2021_dbinfos_train.pkl', 'wb') as f:
        pickle.dump(augmented_dbinfos, f)

def db_infos_interpret():
    filepath = './data/rf2021/rf2021_dbinfos_train.pkl'

    with open(filepath,'rb') as f:
        dbinfos=pickle.load(f)

    # dbinfos.keys() = ['Car', 'Pedestrian', 'Cyclist']
    # ex) cars[10000] = {'name': 'Car', 'path': 'rf2021_gt_database/480_Car_1.bin', 'image_idx': 480, 'gt_idx': 1, 'box3d_lidar': array([  0.53   , -31.23   ,  -2.325  ,   2.01   ,   4.76   ,   1.47   ,
    #  0.04079], dtype=float32), 'num_points_in_gt': 313, 'difficulty': 0, 'group_id': 10483}
    cars = dbinfos['Car']   
    peds = dbinfos['Pedestrian']  
    
    from collections import deque
    car_point = deque([])
    cnt, condition = 0, 0
    
    for car in cars:
        # car['box3d_lidar'][3]: 짧은 변
        # car['box3d_lidar'][4]: 긴 변
        if(np.absolute(car['box3d_lidar'][4]) >= 10 and np.absolute(car['box3d_lidar'][3]) >= 2):
            # 해당 PCD 파일의 위치를 알고싶을때 print("Folder is ", 10002 + car['image_idx'] // 200, " + ", car['image_idx'] - ((car['image_idx'] // 200) * 200) - 1)
            cnt = cnt + 1
            car_point.append(car['num_points_in_gt'])
            if(np.absolute(car['box3d_lidar'][1]) < 30 and np.absolute(car['box3d_lidar'][0]) < 30):
                condition = condition + 1

    car_len = len(cars)
    print("|--------------------------------------------------")
    print("|Total num of Car: ", car_len)
    print("|--------------------------------------------------")
    print("|Car (size over width {}, length {})".format(2, 10))
    print("|cnt: {0:0.4f}%\n| in range of |cx| < {1}, |cy| < {2}: {3:0.4f}%".format(cnt/car_len, 30, 30, condition/car_len))
    print("|--------------------------------------------------")
    import pandas as pd
    car_point = pd.DataFrame(np.array(car_point))
    print("car_point's stats: ")
    print(car_point.describe())
    
def sample_show():
    filepath = './data/sampler.pkl' # 만약 sampler.pkl 파일이 로컬 ./data 경로에 없으면 있는 사람에게 문의
    filepath_pcd = './data/rf2021/rf2021_infos_train.pkl'
    
    with open(filepath,'rb') as f:
        datas=pickle.load(f)

    with open(filepath_pcd, 'rb') as f:
        train_scene_infos = pickle.load(f)
    
    for data, scene_info in zip(datas, train_scene_infos):
        gt_points = data['points'].tensor.numpy()
        pcd_path = "./data/rf2021/" + scene_info['lidar_points']['lidar_path']
        scene_points = read_pcd_data(pcd_path)
        points = np.vstack((gt_points, scene_points))
        show_result(points, None, data['gt_bboxes_3d'], 
                out_dir = './data',
                filename = 'sampler_visualize',
                show=True,
                snapshot=False,
                pred_labels=None)

if __name__ == '__main__':
    # sample_show()
    db_infos_interpret()