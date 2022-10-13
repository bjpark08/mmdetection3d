#사용한 pkl 파일을 다시 txt로 바꿔주는 코드
#pkl 파일은 filename_XXX로 계속 추가해나갈 수 있다. 후반 실험은 val과 test가 없었으므로 해당 부분을 지우고 돌릴 것
#rootpath에서 새 label txt 폴더의 이름을 고쳐줄 수 있다.

#원래 NIA_2021_label과 다른 점은 아래와 같다.
#ped의 경우 원래는 ped가 없으면 아예 만들어지지 않았다. 여기서는 하나하나 다 만듬
#실수형 자료(class를 제외한 전부)가 원래는 1.50처럼 유효숫자가 무조건 세 개였는데 지금은 1.5가 되는 등 0이 될 때 자릿수가 줄어든 경우가 있음
#label/(seq) 폴더에 있는 ped_cv_points는 만들지 않음. 어차피 사용되자 없는 것으로 보임

import pickle
import math
import numpy as np
import os

from os import path as osp
from tqdm import tqdm

filename_train='data/rf2021/relabeling_test/rf2021_infos_train_full.pkl'
filename_val='data/rf2021/rf2021_infos_val_full.pkl'
filename_test='data/rf2021/rf2021_infos_test_full.pkl'
root_path = 'data/rf2021/NIA_2021_label_new/label/'

######## 다른 파일로 돌릴 시 위의 부분들을 고쳐줄 것

with open(filename_train,'rb') as f1:
    data_train=pickle.load(f1)

with open(filename_val,'rb') as f2:
    data_val=pickle.load(f2)

with open(filename_test,'rb') as f3:
    data_test=pickle.load(f3)

data_full = data_train + data_val + data_test

for data in tqdm(data_full):
    seq_num=data['lidar_points']['lidar_path'][-13:-8]
    frame_num=data['lidar_points']['lidar_path'][-7:-4]
    
    veh_path = root_path + seq_num + "/car_label/"
    ped_path = root_path + seq_num + "/ped_label/"

    annot = np.hstack((data['annos']['gt_names'].reshape(-1,1),data['annos']['gt_bboxes_3d']))
    veh_annot = annot[annot[:, 0] != 'Pedestrian',:]
    ped_annot = annot[annot[:, 0] == 'Pedestrian',1:7]

    if not osp.exists(veh_path):
        os.makedirs(veh_path)
    if not osp.exists(ped_path):
        os.makedirs(ped_path)

    veh_annot[veh_annot[:,0] != 'Car',0] = 'Cyclist'
    veh_annot[:,[1,2,3,4,5,6]]=veh_annot[:,[5,4,6,1,2,3]]
    veh_annot[:,7]=np.around(math.pi/2 - veh_annot[:,7].astype(np.float32),2)
    
    np.savetxt(veh_path+f"{seq_num}_{frame_num}.txt", veh_annot, fmt='%s', delimiter=' ')
    np.savetxt(ped_path+f"{seq_num}_{frame_num}.txt", ped_annot, fmt='%s', delimiter=' ')
