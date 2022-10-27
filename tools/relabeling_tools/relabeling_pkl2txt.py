import pickle
import math
import numpy as np
import os

from os import path as osp
from tqdm import tqdm

#사용한 pkl 파일을 다시 txt로 바꿔주는 코드
#원래 NIA_2021_label과 다른 점은 아래와 같다.
#ped의 경우 원래는 ped가 없으면 txt 파일이 아예 만들어지지 않았다. 여기서는 하나하나 다 만듬
#실수형 자료(class를 제외한 전부)가 원래는 1.50처럼 유효숫자가 무조건 세 개였는데 지금은 1.5가 되는 등 0이 될 때 자릿수가 줄어든 경우가 있음

filename_train='data/rf2021/rf2021_relabeled_infos_train.pkl'
#filename_val='data/rf2021/datasets/rf2021/rf2021_infos_val_full.pkl'
#filename_test='data/rf2021/datasets/rf2021/rf2021_infos_test_full.pkl'

root_path = 'data/rf2021/'
NIA_path = root_path + 'NIA_2021_label_relabeled/'
label_dir = NIA_path + 'label/'

with open(filename_train,'rb') as f1:
    data_train=pickle.load(f1)

#with open(filename_val,'rb') as f2:
#    data_val=pickle.load(f2)

#with open(filename_test,'rb') as f3:
#    data_test=pickle.load(f3)

data_full = data_train #+ data_val + data_test

######## 다른 파일로 돌릴 시 위의 부분들을 고쳐줄 것. 경우에 따라 val과 test가 없으므로 해당 부분을 지우고 돌릴 것

for data in tqdm(data_full):
    seq_num=data['lidar_points']['lidar_path'][-13:-8]
    frame_num=data['lidar_points']['lidar_path'][-7:-4]
    
    veh_path = label_dir + seq_num + "/car_label/"
    ped_path = label_dir + seq_num + "/ped_label/"
    ped_cv_path = label_dir + seq_num + "/ped_cv_point/"

    annot = np.hstack((data['annos']['gt_names'].reshape(-1,1),data['annos']['gt_bboxes_3d']))
    veh_annot = annot[annot[:, 0] != 'Pedestrian',:]
    ped_annot = annot[annot[:, 0] == 'Pedestrian',1:7]
    ped_cv_annot = annot[annot[:, 0] == 'Pedestrian',1:3]

    if not osp.exists(veh_path):
        os.makedirs(veh_path)
    if not osp.exists(ped_path):
        os.makedirs(ped_path)
    if not osp.exists(ped_cv_path):
        os.makedirs(ped_cv_path)

    veh_annot[veh_annot[:,0] != 'Car',0] = 'Cyclist'
    veh_annot[:,[1,2,3,4,5,6]]=veh_annot[:,[5,4,6,1,2,3]]
    veh_annot[:,7]=np.around(math.pi/2 - veh_annot[:,7].astype(np.float32),2)
    
    np.savetxt(veh_path+f"{seq_num}_{frame_num}.txt", veh_annot, fmt='%s', delimiter=' ')
    np.savetxt(ped_path+f"{seq_num}_{frame_num}.txt", ped_annot, fmt='%s', delimiter=' ')
    np.savetxt(ped_cv_path+f"{seq_num}_{frame_num}.txt", ped_cv_annot, fmt='%s', delimiter=' ')


#만든 label 폴더들마다 Scene의 Property들을 넣는다.
sequence_max = 3000
#min_ped = 0 (조건 삭제, 원래는 해당 seq의 평균 ped의 최소값. 예를 들어 min_ped가 1이면 ped갯수의 평균이 1미만인 seq는 학습 및 평가 데이터셋에서 제외됨.)

# [seq번호, seq의 scene 갯수, seq의 전체 veh 수, seq의 전체 ped 수, seq의 전체 큰 차 수]  
object_cnt=[[0,0,0,0,0] for i in range(sequence_max)]
car_all=0
ped_all=0
car_big_all=0

folder_list = sorted(os.listdir(label_dir), key=lambda x:int(x))
for fol in tqdm(folder_list):
    veh_label_dir = osp.join(label_dir, fol, "car_label")
    ped_label_dir = osp.join(label_dir, fol, "ped_label")
    seq_info_dir = osp.join(label_dir, fol)
    fol = int(fol) - 10002
    object_cnt[fol][1] = len(os.listdir(veh_label_dir))
    object_cnt[fol][0] = fol + 10002
    if osp.exists(veh_label_dir):
        for veh_file in sorted(os.listdir(veh_label_dir)):
            veh_label_file_path = osp.join(veh_label_dir, veh_file)
            if osp.exists(veh_label_file_path):
                if os.path.getsize(veh_label_file_path):
                    annot_veh = np.loadtxt(veh_label_file_path, dtype=np.object_).reshape(-1, 8)

                    object_cnt[fol][2] += len(annot_veh)

                    big_cars = annot_veh[:,3].astype(np.float32) > 3
                    object_cnt[fol][4] += len(annot_veh[big_cars,:])

                    car_all+=len(annot_veh)
                    car_big_all+=len(annot_veh[big_cars,:])

    if osp.exists(ped_label_dir):
        for ped_file in sorted(os.listdir(ped_label_dir)):
            ped_label_file_path = osp.join(ped_label_dir, ped_file)
            if osp.exists(ped_label_file_path):
                if os.path.getsize(ped_label_file_path):
                    annot_ped = np.loadtxt(ped_label_file_path, dtype=np.object_).reshape(-1, 6)
                    object_cnt[fol][3] += len(annot_ped)
                    ped_all+=len(annot_ped)

    seq_info = np.array(object_cnt[fol])
    seq_mean_info = seq_info[2:5]/seq_info[1]
    seq_info = np.hstack((seq_info, seq_mean_info))

    np.savetxt(seq_info_dir+f"/{int(seq_info[0])}.txt", seq_info, fmt='%s', delimiter=' ')

object_cnt = np.array(object_cnt).astype(np.float32)
object_cnt = object_cnt[object_cnt[:,1]>0, :]
object_cnt[:,2:5]=np.true_divide(object_cnt[:,2:5],object_cnt[:,1].reshape(-1,1))

#for seq_data in object_cnt:
#    print(str(seq_data[0]])+"\t"+str(object_cnt[i][1])+"\t"+str(round(object_cnt[i][2],2))+"\t"+str(round(object_cnt[i][3],2))"\t"+str(round(object_cnt[i][4],2)))

object_cnt = list(object_cnt)

print(f"Vehicle : {car_all}, Pedestrian : {ped_all}, Big Vehicle : {car_big_all}")

for option in [(2,'Num of Vehicles'),(3,'Num of Pedestrians'),(4,'Num of Big Vehicles')]:
    object_cnt.sort(key=lambda x:x[option[0]])
    all_set=[]
    train_set=[]
    test_set=[]

    for i in range(len(object_cnt)):
        all_set.append(int(object_cnt[i][0]))
        if i%10==9:
            test_set.append(int(object_cnt[i][0]))
        else:
            train_set.append(int(object_cnt[i][0]))

    all_set.sort()
    train_set.sort()
    test_set.sort()

    print(f"All of {len(object_cnt)} sequences, Divide with Option {option[0]-1}, {option[1]}")
    print(f"Train set (90%) {len(train_set)} and Test set (10%) {len(test_set)}")
    print(train_set[:10])
    print(test_set[:10])

    if not os.path.exists(NIA_path + f"sequence_divisions/sequence_set_{option[0]-1}"):
        os.makedirs(NIA_path + f"/sequence_divisions/sequence_set_{option[0]-1}")

    with open(NIA_path + f"sequence_divisions/sequence_set_{option[0]-1}/sequence_train_set.pkl",'wb') as rf:
        pickle.dump(train_set,rf)

    with open(NIA_path + f"sequence_divisions/sequence_set_{option[0]-1}/sequence_test_set.pkl",'wb') as rf:
        pickle.dump(test_set,rf)