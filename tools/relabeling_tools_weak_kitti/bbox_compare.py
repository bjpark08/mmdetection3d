# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction

import mmdet
from mmdet3d.datasets import build_dataset
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet.datasets import replace_ImageToTensor


if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--relabeled-pkl',
        type=str,
        default=None,
        help="pkl file which is relabeled and to be evaluated"
    )
    parser.add_argument(
        '--original-pkl',
        type=str,
        default=None,
        help="pkl file which is original bbox from kitti labeling data"
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    cfg.data.test['ann_file'] = cfg.data.test['data_root'] + args.relabeled_pkl
    relabeled_dataset = build_dataset(cfg.data.test)

    cfg.data.test['ann_file'] = cfg.data.test['data_root'] + args.original_pkl
    original_dataset = build_dataset(cfg.data.test)

    original_datas = [info['annos'] for info in original_dataset.data_infos]
    outputs = []
    for i in range(len(original_datas)):
        ref = original_datas[i]
        n = len(ref['gt_names'])
        output = {}
        output_pts = {}

        output_pts['boxes_3d'] = LiDARInstance3DBoxes(ref['gt_bboxes_3d'], origin=(0.5, 0.5, 0.5))

        output_pts['scores_3d'] = torch.tensor([1]*n)

        labels = ref['gt_names']
        label_3d = [-1] * n
        for i in range(n):
            if labels[i] == 'Car':
                label_3d[i] = 0
            elif labels[i] == 'Pedestrian':
                label_3d[i] = 1
        output_pts['labels_3d'] = torch.tensor(label_3d)

        output['pts_bbox']=output_pts
        outputs.append(output)
    
    kwargs = {} if args.eval_options is None else args.eval_options
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    print(relabeled_dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
