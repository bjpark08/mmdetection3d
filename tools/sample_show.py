import pickle
import numpy as np
import os
from os import path as osp
from mmdet3d.core.visualizer import (show_multi_modality_result, show_result, show_seg_result)
from mmdet3d.apis import init_model, inference_detector


if __name__ == '__main__':
    filepath = './data/sampler.pkl'
    filepath_pcd = './data/rf2021/rf2021_infos_train.pkl'
    
    with open(filepath,'rb') as f:
        datas=pickle.load(f)

    with open(filepath_pcd, 'rb') as a:
        pcd = pickle.load(a)

    for data in datas:
        point = data['points'].info.numpy() 
        show_result(point, None, data['gt_bboxes_3d'], 
                out_dir = './data',
                filename = 'sampler_visualize',
                show=True,
                snapshot=False,
                pred_labels=None)