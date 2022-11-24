# data/rf2021/{dir_path}에 있는 train1_file, train2_file을 이용하여 double set iteration을 돌리는 함수
# weak_kitti에서는 train set이 애초에 작기 때문에 그냥 small이 아닌 전체로 gt_database를 만든다.

import pickle
import os, sys

from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)

os.environ['MKL_THREADING_LAYER'] = 'GNU'

iteration=5
data_root = 'data/kitti_relabeling/'
dir_path = 'relabeling_20/'

original_data = 'original_data/'

percent = '20'
prefix = 'weak_kitti_'+ percent
dbinfos_file = prefix + '_dbinfos_train.pkl'
gt_data_file = prefix + '_gt_database'

train1_file = prefix + '_infos_train1.pkl'
checkpoint1_file = 'checkpoints/iter1_0.pth'

train2_file = prefix + '_infos_train2.pkl'
checkpoint2_file = 'checkpoints/iter2_0.pth'

filename_train = dir_path + original_data + prefix + '_infos_train.pkl'
filename_val = dir_path + original_data + prefix + '_infos_val.pkl'

config = 'configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_weak_kitti.py'
work_dir = 'work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_weak_kitti/epoch_20.pth'

######## 다른 파일로 돌릴 시 위의 부분들을 고쳐줄 것

mkdir_message=f"mkdir {data_root + dir_path}"
#os.system(mkdir_message)
print(f"Relabeling Results will be saved in {data_root + dir_path}")


#dir path의 original data 폴더에 원본 데이터를 넣어주면 전부 하나로 합친뒤 반으로 나눠서 train1, train2로 만든다.
print("==============Cutting Original Dataset in Half==============")
with open(data_root + filename_train,'rb') as f1:
    data_train=pickle.load(f1)

with open(data_root + filename_val,'rb') as f1:
    data_val=pickle.load(f1)

data=data_train+data_val
data1=data[:len(data)//2]
data2=data[len(data)//2:]

with open(data_root + dir_path + train1_file,'wb') as f1:
    pickle.dump(data1,f1)

with open(data_root + dir_path + train2_file,'wb') as f1:
    pickle.dump(data2,f1)

#### Process Start

cur_train1 = train1_file
cur_checkpoint1 = checkpoint1_file

cur_train2 = train2_file
cur_checkpoint2 = checkpoint2_file

for i in range(iteration):
    next_train1 = train1_file[:-4]+f'_{str(i+1)}.pkl'
    next_checkpoint1 = checkpoint1_file[:-6]+f'_{str(i+1)}.pth'

    next_train2 = train2_file[:-4]+f'_{str(i+1)}.pkl'
    next_checkpoint2 = checkpoint2_file[:-6]+f'_{str(i+1)}.pth'


    print(f"==============First Training of iteration {i+1}==============")
    train_message = \
        f"./tools/dist_train.sh {config} 2 --training-pkl {dir_path + cur_train1} --no-validate"
    move_checkpoint_message = \
        f"mv {work_dir} {cur_checkpoint1}"
    
    print(train_message)
    os.system(train_message)
    print(move_checkpoint_message)
    os.system(move_checkpoint_message)

    erase_gt_database_message = \
        f"rm -rf {data_root + dbinfos_file} {data_root + gt_data_file}"
        
    print(erase_gt_database_message)
    os.system(erase_gt_database_message)

    create_groundtruth_database('Custom3DDataset', data_root, prefix, data_root + dir_path + cur_train2)


    print(f"==============Second Training of iteration {i+1}==============")
    train_message = \
        f"./tools/dist_train.sh {config} 2 --training-pkl {dir_path + cur_train2} --no-validate"
    move_checkpoint_message = \
        f"mv {work_dir} {cur_checkpoint2}"
    
    print(train_message)
    os.system(train_message)
    print(move_checkpoint_message)
    os.system(move_checkpoint_message)



    print(f"==============Relabeling Process of iteration {i+1}==============")
    relabel_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint1} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {dir_path + cur_train2}"
    
    print(relabel_train_message)
    os.system(relabel_train_message)
    

    move_train_pkl_message = \
        f"mv {data_root + dir_path + cur_train2[:-4] + '_changed_pred.pkl'} {data_root + dir_path + next_train2}"
    
    print(move_train_pkl_message)
    os.system(move_train_pkl_message)
    

   

    relabel_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint2} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {dir_path + cur_train1}"
    
    print(relabel_train_message)
    os.system(relabel_train_message)
    
    move_train_pkl_message = \
        f"mv {data_root + dir_path + cur_train1[:-4] + '_changed_pred.pkl'} {data_root + dir_path + next_train1}"
    
    print(move_train_pkl_message)
    os.system(move_train_pkl_message)
    

    erase_gt_database_message = \
        f"rm -rf {data_root + dbinfos_file} {data_root + gt_data_file}"
        
    print(erase_gt_database_message)
    os.system(erase_gt_database_message)

    create_groundtruth_database('Custom3DDataset', data_root, prefix, data_root + dir_path + next_train1)


    print(f"==============Validation Process of iteration {i+1}==============")
    relabel_valid1_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint1} 2 --eval mAP --show-dir results --validation-pkl {dir_path + cur_train2}"
    relabel_valid2_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint2} 2 --eval mAP --show-dir results --validation-pkl {dir_path + cur_train1}"

    print(relabel_valid1_message)
    os.system(relabel_valid1_message)
    print(relabel_valid2_message)
    os.system(relabel_valid2_message)

    cur_train1=next_train1
    cur_checkpoint1=next_checkpoint1

    cur_train2=next_train2
    cur_checkpoint2=next_checkpoint2