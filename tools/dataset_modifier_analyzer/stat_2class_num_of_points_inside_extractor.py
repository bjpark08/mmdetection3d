import pickle
import numpy as np
import open3d as o3d
from pypcd import pypcd

def extract_ped(datas,min_cnt,pcd_cnt):
    result=[]
    for data in datas:
        car_cnt=0
        ped_cnt=0
        for label in data['annos']['gt_names']:
            if label=='Car':
                car_cnt+=1
            if label=='Pedestrian':
                ped_cnt+=1
        if (car_cnt>=min_cnt and ped_cnt>=min_cnt):
            result.append(data)
            if len(result)==pcd_cnt:
                break
    return result

def extract_ped_pointless(datas,point_cnt):
    result=[]
    cnt=0
    for data in datas:
        cnt+=1
        if cnt%100==0:
            print(cnt)
        cur_data=data
        delete_list=[]
        for i in range(len(cur_data['annos']['gt_bboxes_3d'])):
            center = cur_data['annos']['gt_bboxes_3d'][i, 0:3]
            dim = cur_data['annos']['gt_bboxes_3d'][i, 3:6]
            yaw = np.zeros(3)
            yaw[2] = cur_data['annos']['gt_bboxes_3d'][i, 6]
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)

            box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)

            pcd = o3d.io.read_point_cloud(root_path + cur_data['lidar_points']['lidar_path'])
            
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            if len(indices)<point_cnt:
                delete_list.append(i)
        cur_data['annos']['gt_bboxes_3d']=np.delete(cur_data['annos']['gt_bboxes_3d'],delete_list,axis=0)
        cur_data['annos']['gt_names']=np.delete(cur_data['annos']['gt_names'],delete_list,axis=0)
        result.append(cur_data)
    return result

root_path = 'data/rf2021/'
file_name='data/rf2021/rf2021_infos_train'

with open(file_name+'.pkl','rb') as f:
	datas=pickle.load(f)

min_cnt=10
point_cnt=10
pcd_cnt=100
filtered_datas=extract_ped(datas,min_cnt,pcd_cnt)
filtered_datas=extract_ped_pointless(filtered_datas,point_cnt)

print(min_cnt, point_cnt, pcd_cnt)


with open(file_name+'_2class_'+str(min_cnt)+'_points_'+str(point_cnt)+'_size'+str(pcd_cnt)+'.pkl','wb') as rf:
    pickle.dump(filtered_datas,rf,protocol=pickle.HIGHEST_PROTOCOL)
