# data/rf2021/{dir_path}에 있는 train1_file, train2_file, small_train1_file, small_train2_file을 이용하여 double set iteration을 돌리는 함수

import pickle
import os, sys

from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)

os.environ['MKL_THREADING_LAYER'] = 'GNU'

iteration=5
data_root = 'data/rf2021/'
dir_path = 'relabeling_final_1104/'
original_data = 'original_data/'

prefix = 'rf2021_final'
dbinfos_file = 'rf2021_final_dbinfos_train.pkl'
gt_data_file = 'rf2021_final_gt_database'

filename_train = 'rf2021_final_infos_train.pkl'
filename_test = 'rf2021_final_infos_test.pkl'

train1_file = 'rf2021_final_infos_train1.pkl'
small_train1_file = 'rf2021_final_infos_train1_small.pkl'
checkpoint1_file = 'checkpoints/iter1_0.pth'

train2_file = 'rf2021_final_infos_train2.pkl'
small_train2_file = 'rf2021_final_infos_train2_small.pkl'
checkpoint2_file = 'checkpoints/iter2_0.pth'

config = 'configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_rf_2021.py'
work_dir = 'work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_rf_2021/epoch_20.pth'

######## 다른 파일로 돌릴 시 위의 부분들을 고쳐줄 것

#mkdir_message=f"mkdir {data_root + dir_path}"
#os.system(mkdir_message)
print(f"Relabeling Results will be saved in {data_root + dir_path}")


# print("==============Cutting Original Dataset in Half==============")
# with open(data_root + dir_path + original_data + filename_train,'rb') as f1:
#     data_train=pickle.load(f1)

# with open(data_root + dir_path + original_data + filename_test,'rb') as f1:
#     data_test=pickle.load(f1)

# data=data_train+data_test
# data1=data[:len(data)//2]
# data2=data[len(data)//2:]

# with open(data_root + dir_path + train1_file,'wb') as f1:
#     pickle.dump(data1,f1)

# with open(data_root + dir_path + train2_file,'wb') as f1:
#     pickle.dump(data2,f1)

#### Process Start

cur_train1 = train1_file
cur_small_train1 = small_train1_file
cur_checkpoint1 = checkpoint1_file

cur_train2 = train2_file
cur_small_train2 = small_train2_file
cur_checkpoint2 = checkpoint2_file

for i in range(iteration):
    next_train1 = train1_file[:-4]+f'_{str(i+1)}.pkl'
    next_small_train1 = small_train1_file[:-4]+f'_{str(i+1)}.pkl'
    next_checkpoint1 = checkpoint1_file[:-6]+f'_{str(i+1)}.pth'

    next_train2 = train2_file[:-4]+f'_{str(i+1)}.pkl'
    next_small_train2 = small_train2_file[:-4]+f'_{str(i+1)}.pkl'
    next_checkpoint2 = checkpoint2_file[:-6]+f'_{str(i+1)}.pth'


    print(f"==============First Training of iteration {i+1}==============")
    create_groundtruth_database('Custom3DDataset', data_root, prefix, data_root + dir_path + cur_small_train1)
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


    print(f"==============Second Training of iteration {i+1}==============")
    create_groundtruth_database('Custom3DDataset', data_root, prefix, data_root + dir_path + cur_small_train2)
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
    relabel_small_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint1} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {dir_path + cur_small_train2}"

    print(relabel_train_message)
    os.system(relabel_train_message)
    print(relabel_small_train_message)
    os.system(relabel_small_train_message)


    move_train_pkl_message = \
        f"mv {data_root + dir_path + cur_train2[:-4] + '_changed_pred.pkl'} {data_root + dir_path + next_train2}"
    move_small_train_pkl_message = \
        f"mv {data_root + dir_path + cur_small_train2[:-4] + '_changed_pred.pkl'} {data_root + dir_path + next_small_train2}"

    print(move_train_pkl_message)
    os.system(move_train_pkl_message)
    print(move_small_train_pkl_message)
    os.system(move_small_train_pkl_message)


   

    relabel_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint2} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {dir_path + cur_train1}"
    relabel_small_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint2} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {dir_path + cur_small_train1}"

    print(relabel_train_message)
    os.system(relabel_train_message)
    print(relabel_small_train_message)
    os.system(relabel_small_train_message)


    move_train_pkl_message = \
        f"mv {data_root + dir_path + cur_train1[:-4] + '_changed_pred.pkl'} {data_root + dir_path + next_train1}"
    move_small_train_pkl_message = \
        f"mv {data_root + dir_path + cur_small_train1[:-4] + '_changed_pred.pkl'} {data_root + dir_path + next_small_train1}"

    print(move_train_pkl_message)
    os.system(move_train_pkl_message)
    print(move_small_train_pkl_message)
    os.system(move_small_train_pkl_message)
 

    erase_gt_database_message = \
        f"rm -rf {data_root + dbinfos_file} {data_root + gt_data_file}"
        
    print(erase_gt_database_message)
    os.system(erase_gt_database_message)


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
    cur_small_train1=next_small_train1
    cur_checkpoint1=next_checkpoint1

    cur_train2=next_train2
    cur_small_train2=next_small_train2
    cur_checkpoint2=next_checkpoint2