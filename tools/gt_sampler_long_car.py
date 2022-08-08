import numpy as np
import pickle
import pypcd
import os    
from pathlib import Path


def read_pcd_data(pcd_path):
        points_pcd = pypcd.PointCloud.from_path(pcd_path)
        pcd_x = points_pcd.pc_data["x"].copy()
        pcd_y = points_pcd.pc_data["y"].copy()
        pcd_z = points_pcd.pc_data["z"].copy()
        pcd_intensity = points_pcd.pc_data["intensity"].copy().astype(np.float32)/256
        pcd_tensor = np.array([pcd_x, pcd_y, pcd_z, pcd_intensity], dtype=np.float32).T
        return points_pcd, pcd_tensor   
    
if __name__ == '__main__':    

    filepath = './rf2021_dbinfos_train.pkl'
    with open(filepath,'rb') as f:
        datas=pickle.load(f)

    cars = datas['Car']
    add_car = []
    add_car_size = []
    num_points = []
    image_idxs = []
    idx = 0
    flag = True
    last_image_idx = cars[-1]["image_idx"]

    for data in cars:
        # car_point.append(car['num_points_in_gt'])
        # car_length.append(car['box3d_lidar'][4])

        if(np.absolute(data['box3d_lidar'][4]) >= 10 and data['num_points_in_gt'] >= 100):
            points = np.fromfile(data['path'], np.float32).reshape(-1, 4)
            # points = np.fromfile(data['path'], np.float32)
            pcd_x = points[:, 0].copy()
            pcd_y = points[:, 1].copy()
            pcd_z = points[:, 2].copy()      
            pcd_intensity = points[:, 3].copy().astype(np.float32)/256
            pcd_tensor = np.array([pcd_x, pcd_y, pcd_z, pcd_intensity], dtype=np.float32).T
            num_points.append(data['num_points_in_gt'])
            if(idx != 0):
                if(np.absolute(image_idxs[idx-1] - data['image_idx']) > 50):
                    add_car.append(pcd_tensor)
                    image_idxs.append(data['image_idx'])
                    temp = data['box3d_lidar']
                    temp[0] = temp[0] * 1.5
                    temp[1] = temp[1] * 1.5
                    add_car_size.append(temp)
                    idx = idx + 1
            else:
                add_car.append(pcd_tensor)
                image_idxs.append(data['image_idx'])
                temp = data['box3d_lidar']
                temp[0] = temp[0] * 1.5
                temp[1] = temp[1] * 1.5
                add_car_size.append(temp)
                idx = idx + 1
    
    # car_idx = []

    # # for i in range(len(add_car)):
    # #     if(i % 10 == 0):
    # #         car_idx.append(add_car[i])
    # # print(len(car_idx))

    # print(len(car_idx))

    for i in range(len(add_car)):
        pcd_x = add_car[i][0] * 1.5 
        pcd_y = add_car[i][1] * 1.5
        pcd_tensor = np.array([pcd_x, pcd_y, add_car[i][2], add_car[i][3]], dtype = np.float32).T
        pcd_tensor = pcd_tensor.reshape(-1)
        last_image_idx = last_image_idx + 1
        # with open('./new_bin/' + str(last_image_idx) + '_Car_1.bin','wb') as rf:
	    #     pickle.dump(pcd_tensor, rf)
        with open('./new_bin/' + str(last_image_idx) + '_Car_1.bin', 'w') as f:
            pcd_tensor.tofile(f)
    

    # last_image_idx = cars[-1]["image_idx"] + 1
    # temp = []
    # for i in range(len(add_car_size)):
    #     path = 'new_bin/' + str(last_image_idx) + '_Car_1.bin'
    #     image_idx = last_image_idx
    #     last_image_idx = last_image_idx + 1
    #     gt_idx = 1
    #     box3d_lidar = add_car_size[i]
    #     temp.append({'name': 'Car', 'path': path, 'image_idx': image_idx, 'gt_idx': 1, 'box3d_lidar': box3d_lidar, 'num_points_in_gt': num_points[i], 'difficulty': 0, 'group_id': None})


    # with open('rf2021_dbinfos_train.pkl', 'rb') as f:
    #     data = pickle.load(f)
    
    # data['Car'] = data['Car'] + temp
    # temp = {'Car': data['Car'], 'Pedestrian': data['Pedestrian'], 'Pedestr': data['Pedestr']}
    # with open('new_rf2021_dbinfos_train.pkl', 'wb') as f:
    #     pickle.dump(temp, f)