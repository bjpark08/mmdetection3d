import os, sys

from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)

os.environ['MKL_THREADING_LAYER'] = 'GNU'

iteration=5
data_root = 'data/rf2021/'

train_file = 'rf2021_seq_infos_train.pkl'
#train_file = 'rf2021_seq_infos_train.pkl'
#val_file = 'rf2021_infos_val_height_make.pkl'  # don't use val anymore
test_file = 'rf2021_seq_infos_test.pkl'
small_train_file = 'rf2021_seq_infos_train_small.pkl'
checkpoint_file = 'checkpoints/iter_0.pth'

config = 'configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_rf_2021.py'
work_dir = 'work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_rf_2021/epoch_20.pth'

dir_path='./data/rf2021/relabeling_results/'
mkdir_message=f"mkdir {dir_path}"
#os.system(mkdir_message)
print("Relabeling Results will be saved in data/rf2021/relabeling_results")

cur_train = train_file
#cur_val = val_file
cur_test = test_file
cur_small_train = small_train_file
cur_checkpoint = checkpoint_file

for i in range(iteration):
    next_train = train_file[:-4]+f'_{str(i+1)}.pkl'
    #next_val = val_file[:-4]+f'_{str(i+1)}.pkl'
    next_test = test_file[:-4]+f'_{str(i+1)}.pkl'
    next_small_train = small_train_file[:-4]+f'_{str(i+1)}.pkl'
    next_checkpoint = checkpoint_file[:-6]+f'_{str(i+1)}.pth'

    train_message = \
        f"./tools/dist_train.sh {config} 2 --training-pkl {'relabeling_results/' + cur_train} --no-validate"
    move_checkpoint_message = \
        f"mv {work_dir} {cur_checkpoint}"
    
    print(train_message)
    os.system(train_message)
    print(move_checkpoint_message)
    os.system(move_checkpoint_message)



    relabel_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results/' + cur_train}"
    #relabel_val_message = \
    #    f"./tools/dist_test.sh {config} {cur_checkpoint} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results/' + cur_val}"
    relabel_test_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results/' + cur_test}"
    relabel_small_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results/' + cur_small_train}"

    print(relabel_train_message)
    os.system(relabel_train_message)
    #print(relabel_val_message)
    #os.system(relabel_val_message)
    print(relabel_test_message)
    os.system(relabel_test_message)
    print(relabel_small_train_message)
    os.system(relabel_small_train_message)



    move_train_pkl_message = \
        f"mv {dir_path + cur_train[:-4] + '_changed_pred.pkl'} {dir_path + next_train}"
    #move_val_pkl_message = \
    #    f"mv {dir_path + cur_val[:-4] + '_changed_pred.pkl'} {dir_path + next_val}"
    move_test_pkl_message = \
        f"mv {dir_path + cur_test[:-4] + '_changed_pred.pkl'} {dir_path + next_test}"
    move_small_train_pkl_message = \
        f"mv {dir_path + cur_small_train[:-4] + '_changed_pred.pkl'} {dir_path + next_small_train}"

    print(move_train_pkl_message)
    os.system(move_train_pkl_message)
    #print(move_val_pkl_message)
    #os.system(move_val_pkl_message)
    print(move_test_pkl_message)
    os.system(move_test_pkl_message)
    print(move_small_train_pkl_message)
    os.system(move_small_train_pkl_message)
 


    erase_gt_database_message = \
        f"rm -rf {data_root + 'rf2021_seq_dbinfos_train.pkl'} {data_root + 'rf2021_seq_gt_database'}"
        
    #print(erase_gt_database_message)
    #os.system(erase_gt_database_message)



    create_groundtruth_database('Custom3DDataset', 'data/rf2021', 'rf2021_seq', dir_path + next_small_train)

    cur_train=next_train
    #cur_val=next_val
    cur_test=next_test
    cur_small_train=next_small_train
    cur_checkpoint=next_checkpoint




