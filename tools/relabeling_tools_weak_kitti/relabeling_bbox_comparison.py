# data/rf2021/{dir_path}에 있는 train1_file, train2_file을 이용하여 double set iteration을 돌리는 함수
# weak_kitti에서는 train set이 애초에 작기 때문에 그냥 small이 아닌 전체로 gt_database를 만든다.

import pickle
import os, sys

from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)

os.environ['MKL_THREADING_LAYER'] = 'GNU'

iteration=5
percent = '0'
data_root = 'data/kitti_relabeling/'
dir_path = 'relabeling_' + percent + '/'

ori1='ori_kitti_infos_train1.pkl'
ori2='ori_kitti_infos_train2.pkl'

prefix = 'weak_kitti_'+ percent

train1_file = prefix + '_infos_train1.pkl'
train2_file = prefix + '_infos_train2.pkl'

config = 'configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_weak_kitti.py'

######## 다른 파일로 돌릴 시 위의 부분들을 고쳐줄 것

#### Process Start

cur_train1 = train1_file
cur_train2 = train2_file

for i in range(iteration):
    next_train1 = train1_file[:-4]+f'_{str(i+1)}.pkl'
    next_train2 = train2_file[:-4]+f'_{str(i+1)}.pkl'

    print(f"==============BBox Comparison of iteration {i+1}==============")
    bbox_compare1_message = \
        f"python tools/relabeling_tools_weak_kitti/bbox_compare.py {config} --eval mAP --show-dir results --relabeled-pkl {dir_path + cur_train1} --original-pkl {ori1}"
    bbox_compare2_message = \
        f"python tools/relabeling_tools_weak_kitti/bbox_compare.py {config} --eval mAP --show-dir results --relabeled-pkl {dir_path + cur_train2} --original-pkl {ori2}"

    print(bbox_compare1_message)
    os.system(bbox_compare1_message)
    print(bbox_compare2_message)
    os.system(bbox_compare2_message)

    cur_train1=next_train1
    cur_train2=next_train2