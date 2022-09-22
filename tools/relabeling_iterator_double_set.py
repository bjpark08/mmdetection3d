import os, sys

from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)

os.environ['MKL_THREADING_LAYER'] = 'GNU'

iteration=5
data_root = 'data/rf2021/'

train1_file = 'rf2021_seq_infos_train1.pkl'
test1_file = 'rf2021_seq_infos_test1.pkl'
small_train1_file = 'rf2021_seq_infos_train1_small.pkl'
checkpoint1_file = 'checkpoints/iter1_0.pth'

train2_file = 'rf2021_seq_infos_train2.pkl'
test2_file = 'rf2021_seq_infos_test2.pkl'
small_train2_file = 'rf2021_seq_infos_train2_small.pkl'
checkpoint1_file = 'checkpoints/iter2_0.pth'

config = 'configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_rf_2021.py'
work_dir = 'work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_rf_2021/epoch_20.pth'

dir_path='./data/rf2021/relabeling_results_double_set/'
mkdir_message=f"mkdir {dir_path}"
#os.system(mkdir_message)
print(f"Relabeling Results will be saved in {dir_path}")

cur_train1 = train1_file
cur_test1 = test1_file
cur_small_train1 = small_train1_file
cur_checkpoint1 = checkpoint1_file

cur_train2 = train2_file
cur_test2 = test2_file
cur_small_train2 = small_train2_file
cur_checkpoint2 = checkpoint2_file

for i in range(iteration):
    next_train1 = train1_file[:-4]+f'_{str(i+1)}.pkl'
    next_test1 = test1_file[:-4]+f'_{str(i+1)}.pkl'
    next_small_train1 = small_train1_file[:-4]+f'_{str(i+1)}.pkl'
    next_checkpoint1 = checkpoint1_file[:-6]+f'_{str(i+1)}.pth'

    next_train2 = train2_file[:-4]+f'_{str(i+1)}.pkl'
    next_test2 = test2_file[:-4]+f'_{str(i+1)}.pkl'
    next_small_train2 = small_train2_file[:-4]+f'_{str(i+1)}.pkl'
    next_checkpoint2 = checkpoint2_file[:-6]+f'_{str(i+1)}.pth'


    print(f"==============First Half of iteration {i+1}==============")
    train_message = \
        f"./tools/dist_train.sh {config} 2 --training-pkl {'relabeling_results_double_set/' + cur_train1} --no-validate"
    move_checkpoint_message = \
        f"mv {work_dir} {cur_checkpoint1}"
    
    print(train_message)
    os.system(train_message)
    print(move_checkpoint_message)
    os.system(move_checkpoint_message)


    relabel_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint1} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results_double_set/' + cur_train2}"
    relabel_test_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint1} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results_double_set/' + cur_test2}"
    relabel_small_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint1} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results_double_set/' + cur_small_train2}"

    print(relabel_train_message)
    os.system(relabel_train_message)
    print(relabel_test_message)
    os.system(relabel_test_message)
    print(relabel_small_train_message)
    os.system(relabel_small_train_message)


    move_train_pkl_message = \
        f"mv {dir_path + cur_train2[:-4] + '_changed_pred.pkl'} {dir_path + next_train2}"
    move_test_pkl_message = \
        f"mv {dir_path + cur_test2[:-4] + '_changed_pred.pkl'} {dir_path + next_test2}"
    move_small_train_pkl_message = \
        f"mv {dir_path + cur_small_train2[:-4] + '_changed_pred.pkl'} {dir_path + next_small_train2}"

    print(move_train_pkl_message)
    os.system(move_train_pkl_message)
    print(move_test_pkl_message)
    os.system(move_test_pkl_message)
    print(move_small_train_pkl_message)
    os.system(move_small_train_pkl_message)
 

    erase_gt_database_message = \
        f"rm -rf {data_root + 'rf2021_seq_dbinfos_train.pkl'} {data_root + 'rf2021_seq_gt_database'}"
        
    #print(erase_gt_database_message)
    #os.system(erase_gt_database_message)



    create_groundtruth_database('Custom3DDataset', 'data/rf2021', 'rf2021_seq', dir_path + next_small_train2)


    print(f"==============Second Half of iteration {i+1}==============")
    train_message = \
        f"./tools/dist_train.sh {config} 2 --training-pkl {'relabeling_results_double_set/' + next_train2} --no-validate"
    move_checkpoint_message = \
        f"mv {work_dir} {cur_checkpoint2}"
    
    print(train_message)
    os.system(train_message)
    print(move_checkpoint_message)
    os.system(move_checkpoint_message)


    relabel_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint2} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results_double_set/' + cur_train1}"
    relabel_test_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint2} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results_double_set/' + cur_test1}"
    relabel_small_train_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint2} 2 --eval mAP --show-dir results --relabeling --relabeling-pkl {'relabeling_results_double_set/' + cur_small_train1}"

    print(relabel_train_message)
    os.system(relabel_train_message)
    print(relabel_test_message)
    os.system(relabel_test_message)
    print(relabel_small_train_message)
    os.system(relabel_small_train_message)


    move_train_pkl_message = \
        f"mv {dir_path + cur_train2[:-4] + '_changed_pred.pkl'} {dir_path + next_train1}"
    move_test_pkl_message = \
        f"mv {dir_path + cur_test2[:-4] + '_changed_pred.pkl'} {dir_path + next_test1}"
    move_small_train_pkl_message = \
        f"mv {dir_path + cur_small_train2[:-4] + '_changed_pred.pkl'} {dir_path + next_small_train1}"

    print(move_train_pkl_message)
    os.system(move_train_pkl_message)
    print(move_test_pkl_message)
    os.system(move_test_pkl_message)
    print(move_small_train_pkl_message)
    os.system(move_small_train_pkl_message)
 

    erase_gt_database_message = \
        f"rm -rf {data_root + 'rf2021_seq_dbinfos_train.pkl'} {data_root + 'rf2021_seq_gt_database'}"
        
    #print(erase_gt_database_message)
    #os.system(erase_gt_database_message)



    create_groundtruth_database('Custom3DDataset', 'data/rf2021', 'rf2021_seq', dir_path + next_small_train1)


    cur_train1=next_train1
    cur_test1=next_test1
    cur_small_train1=next_small_train1
    cur_checkpoint1=next_checkpoint1

    cur_train2=next_train2
    cur_test2=next_test2
    cur_small_train2=next_small_train2
    cur_checkpoint2=next_checkpoint2