import os, sys

from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)

os.environ['MKL_THREADING_LAYER'] = 'GNU'

iteration=5
data_root = 'data/rf2021/'

train1_file = 'rf2021_seq_infos_train1.pkl'
small_train1_file = 'rf2021_seq_infos_train1_small.pkl'
checkpoint1_file = 'checkpoints/iter1_0.pth'

train2_file = 'rf2021_seq_infos_train2.pkl'
small_train2_file = 'rf2021_seq_infos_train2_small.pkl'
checkpoint2_file = 'checkpoints/iter2_0.pth'

config = 'configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_rf_2021.py'
work_dir = 'work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_rf_2021/epoch_20.pth'

dir_path='./data/rf2021/relabeling_results_double_set/'

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

    print(f"==============Validation Process of iteration {i+1}==============")
    relabel_valid1_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint1} 2 --eval mAP --show-dir results --validation-pkl {'relabeling_results_double_set/' + next_train2}"
    relabel_valid2_message = \
        f"./tools/dist_test.sh {config} {cur_checkpoint2} 2 --eval mAP --show-dir results --validation-pkl {'relabeling_results_double_set/' + next_train1}"

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