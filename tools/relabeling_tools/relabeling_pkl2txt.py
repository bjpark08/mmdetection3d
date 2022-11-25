import pickle
import math
import numpy as np
import os

from os import path as osp
from tqdm import tqdm

root_path = 'data/rf2021/'
# Relabeling 과정에서는 Cyclist를 Car에 포함시켜서 label을 Car로 통일시키지만
# 다시 원래대로 돌릴 때에는 원래대로 Cyclist로 돌려놔야하므로 원본 label 쪽에서 Cyclist의 정보를 얻어와야함.
NIA_original_label_path = root_path + 'NIA_2021_label/' + 'label/'

# single set 사용
filename_train='data/rf2021/relabeling_single_set/rf2021_final_infos_merged.pkl'
NIA_path = root_path + 'NIA_2021_label_relabeled_single_set/'
label_dir = NIA_path + 'label/'

# double set 사용
# filename_train='data/rf2021/relabeling_double_set/rf2021_final_infos_train_merged.pkl'
# NIA_path = root_path + 'NIA_2021_label_relabeled_double_set/'
# label_dir = NIA_path + 'label/'

with open(filename_train,'rb') as f1:
    data_full=pickle.load(f1)


######## 다른 파일로 돌릴 시 위의 부분들을 고쳐줄 것.

for data in tqdm(data_full):
    seq_num=data['lidar_points']['lidar_path'][-13:-8]
    frame_num=data['lidar_points']['lidar_path'][-7:-4]
    
    veh_path = label_dir + seq_num + "/car_label/"
    ped_path = label_dir + seq_num + "/ped_label/"
    ped_cv_path = label_dir + seq_num + "/ped_cv_point/"

    #veh, ped 관계없이 class,cx,cy,cz,l,w,h,theta로 통일
    annot = np.hstack((data['annos']['gt_names'].reshape(-1,1),data['annos']['gt_bboxes_3d']))
    veh_annot = annot[annot[:, 0] != 'Pedestrian',:]
    ped_annot = annot[annot[:, 0] == 'Pedestrian',:]
    ped_cv_annot = annot[annot[:, 0] == 'Pedestrian',1:3]

    if not osp.exists(veh_path):
        os.makedirs(veh_path)
    if not osp.exists(ped_path):
        os.makedirs(ped_path)
    if not osp.exists(ped_cv_path):
        os.makedirs(ped_cv_path)

    original_veh_path = NIA_original_label_path + seq_num + "/car_label/" + f"{seq_num}_{frame_num}.txt"
    if osp.exists(original_veh_path):
        if os.path.getsize(original_veh_path):
            veh_original_annot = np.loadtxt(original_veh_path, dtype=np.object_).reshape(-1, 8)
            # relabeling이 Cyclist를 Car에 포함해서 했으면 class 명만 옮긴다
            # relabeling이 Cyclist를 Ped에 포함해서 했으면 Ped쪽 Cyclist를 Car쪽으로 옮긴다
            if len(veh_annot[:,0])==len(veh_original_annot):
                veh_annot[:,0] = veh_original_annot[:,0]
            else:
                noncar_cnt = sum(veh_original_annot[:, 0] != 'Car')
                noncar_annot = ped_annot[0:noncar_cnt,:]
                noncar_annot[:,0] = 'Cyclist'
                veh_annot = np.vstack((veh_annot, noncar_annot))
                ped_annot = ped_annot[noncar_cnt:,:]
                
            

    #txt 형식 변경으로 제거
    #veh_annot[:,[1,2,3,4,5,6]]=veh_annot[:,[5,4,6,1,2,3]]
    #veh_annot[:,7]=np.around(math.pi/2 - veh_annot[:,7].astype(np.float32),2)
    
    np.savetxt(veh_path+f"{seq_num}_{frame_num}.txt", veh_annot, fmt='%s', delimiter=' ')
    np.savetxt(ped_path+f"{seq_num}_{frame_num}.txt", ped_annot, fmt='%s', delimiter=' ')
    np.savetxt(ped_cv_path+f"{seq_num}_{frame_num}.txt", ped_cv_annot, fmt='%s', delimiter=' ')

#만든 label 폴더들마다 Scene의 Property들을 넣는다.
sequence_max = 3000

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
                    annot_ped = np.loadtxt(ped_label_file_path, dtype=np.object_).reshape(-1, 8)
                    object_cnt[fol][3] += len(annot_ped)
                    ped_all+=len(annot_ped)

    seq_info = np.array(object_cnt[fol])
    seq_mean_info = seq_info[2:5]/seq_info[1]
    seq_info = np.hstack((seq_info, seq_mean_info))

    np.savetxt(seq_info_dir+f"/{int(seq_info[0])}.txt", seq_info, fmt='%s', delimiter=' ')

object_cnt = np.array(object_cnt).astype(np.float32)
object_cnt = object_cnt[object_cnt[:,1]>0, :]
object_cnt[:,2:5]=np.true_divide(object_cnt[:,2:5],object_cnt[:,1].reshape(-1,1))

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