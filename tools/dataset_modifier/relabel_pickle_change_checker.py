import pickle
import numpy as np

car_cnt=0
ped_cnt=0
car_change=0
ped_change=0

file_name='../../data/rf2021/rf2021_infos_test_height_make'
mode='pred' # or 'mid'

with open(file_name+'.pkl','rb') as f:
    former_datas=pickle.load(f)

with open(file_name+'_changed_'+mode+'.pkl','rb') as f:
    cur_datas=pickle.load(f)

max_diff=0
a=0
b=0
assert len(former_datas)==len(cur_datas)
for i in range(len(cur_datas)):
    former_data=former_datas[i]
    cur_data=cur_datas[i]
    assert len(former_data['annos']['gt_bboxes_3d'])==len(cur_data['annos']['gt_bboxes_3d'])
    for j in range(len(cur_data['annos']['gt_names'])):
        assert former_data['annos']['gt_names'][j]==cur_data['annos']['gt_names'][j]
        label=cur_data['annos']['gt_names'][j]
        if label=='Car':
            car_cnt+=1
        elif label=='Pedestrian':
            ped_cnt+=1

        former_box=former_data['annos']['gt_bboxes_3d'][j]
        cur_box=cur_data['annos']['gt_bboxes_3d'][j]
        for k in range(7):
            if former_box[k]!=cur_box[k]:
                max_diff=max(max_diff,abs(former_box[k]-cur_box[k]))
                if abs(former_box[k]-cur_box[k])>2 and 3<=k<6:
                    print(k, former_box[k], cur_box[k], former_data['lidar_points']['lidar_path'])
                if label=='Car':
                    car_change+=1
                    break
                elif label=='Pedestrian':
                    ped_change+=1
                    break

print(car_cnt, ped_cnt)
print(car_change, ped_change)
print(car_change/car_cnt, ped_change/ped_cnt)
print(max_diff)