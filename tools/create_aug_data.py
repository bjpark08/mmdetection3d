# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path
from os import path as osp
import pickle
import numpy as np

from tools.data_converter import indoor_converter as indoor
from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

root_path = Path(args.root_path)

create_groundtruth_database('Custom3DDataset', Path(args.root_path), args.extra_tag, Path(args.root_path) / f'{args.extra_tag}_infos_train_height_make.pkl')