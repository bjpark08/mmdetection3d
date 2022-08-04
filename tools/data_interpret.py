import pickle
import numpy as np
import mmcv
from pypcd import pypcd
import open3d as o3d

def read_pcd_data(pcd_path):
        points_pcd = pypcd.PointCloud.from_path(pcd_path)
        pcd_x = points_pcd.pc_data["x"].copy()
        pcd_y = points_pcd.pc_data["y"].copy()
        pcd_z = points_pcd.pc_data["z"].copy()
        pcd_intensity = points_pcd.pc_data["intensity"].copy().astype(np.float32)/256
        pcd_tensor = np.array([pcd_x, pcd_y, pcd_z, pcd_intensity], dtype=np.float32).T
        return points_pcd, pcd_tensor  

if __name__ == '__main__':

    filepath = './data/rf2021/rf2021_dbinfos_train.pkl'

    with open(filepath,'rb') as f:
        datas=pickle.load(f)

    cars = datas['Car']    #list
    peds = datas['Pedestrian']   
    car_point = []
    car_total_num = 0
    cnt = 0
    flag = True
    car_length = []
    car_point_condition = []
    car_size = []

    # ped_point = []
    # for ped in peds: 
    #     ped_point.append(ped['num_points_in_gt'])
    #     if(flag == True):
    #         points_pcd, pcd_tensor = read_pcd_data(ped['./data/rf2021/rf2021_gt_database/0_Car_0.bin'])
    #         flag = False

    for car in cars:
        car_point.append(car['num_points_in_gt'])
        car_length.append(car['box3d_lidar'][4])


        if(np.absolute(car['box3d_lidar'][4]) >= 15):
            cnt = cnt + 1
            if(flag == True):
                print(car)
                print("Folder is ", 10002 + car['image_idx'] // 200, " + ", car['image_idx'] - ((car['image_idx'] // 200) * 200) - 1)
                flag = False
            
        # # if(car['num_points_in_gt'] <= 30 and np.absolute(car['box3d_lidar'][1]) < 10 and np.absolute(car['box3d_lidar'][0]) < 10):
        # if(np.absolute(car['box3d_lidar'][4]) >= 25 and np.absolute(car['box3d_lidar'][3]) > 10):
        #     if(flag == True):
        #         print(car)
        #         print('\n')
        #         print("Folder is ", 10002 + car['image_idx'] // 200, " + ", car['image_idx'] - ((car['image_idx'] // 200) * 200) - 1)
        #         flag = False
        #     cnt = cnt + 1
        #     if(np.absolute(car['box3d_lidar'][1]) < 30 and np.absolute(car['box3d_lidar'][0]) < 30):
        #         condition = condition + 1

    # car_point = np.array(car_point)
    # car_length = np.array(car_length)
    # car_point_condition = np.array(car_point_condition)
    # car_size = np.array(car_size)
    # car_size = np.absolute(car_size)
    # print("Total num of Car ", len(cars))
    # print("Condition ", cnt, " ", cnt / len(cars) * 100)
    # print("number of object is ", cnt,  " ", cnt / len(cars) * 100, "%")
    # print("number of conditioned object is ", condition, " ", condition / len(cars) * 100, "%")
    # print(car_point.mean())
    # print(car_point_condition.mean())
    # print(np.mean(car_size, axis = 0))
