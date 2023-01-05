# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test, vis_pose_result
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

import json
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    '''
    if not distributed:
        model = MMDataParallel(model, device_ids=[args.gpu_id])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    '''
    with open("predictions_pose.json", 'r') as f:
        outputs = json.load(f)

    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))


    np_outputs = []
    for output in outputs:
        np_dict = {}
        np_dict['preds'] = np.array(output['preds'])
        np_dict['boxes'] = np.array(output['boxes'])
        np_dict['image_paths'] = ['data/coco/val2017/'+output['image_paths'][0]]
        np_dict['bbox_ids'] = np.array(output['bbox_ids'])
        np_dict['output_heatmap'] = output['output_heatmap']
        np_outputs.append(np_dict)

    for i in range(len(np_outputs)):
        print(np_outputs[i]['preds'])
        print(np_outputs[i]['boxes'])
        print((np_outputs[i]['image_paths']))
        print((np_outputs[i]['bbox_ids']))
    print(len(np_outputs))

    
    for output in np_outputs:
        save_img = cv2.imread(output['image_paths'][0])
        vis_boxes = output['boxes']
        if len(vis_boxes) > 0:
            for vis_box in vis_boxes:
                x1 = int(vis_box[0] - vis_box[2]*100)
                y1 = int(vis_box[1] - vis_box[3]*100)
                x2 = int(vis_box[0] + vis_box[2]*100)
                y2 = int(vis_box[1] + vis_box[3]*100)
                save_img = cv2.rectangle(save_img, (x1, y1), \
                            (x2, y2), (200,200,200), 3, cv2.LINE_AA)
            for vis_joint in output['preds'][0]:
                x1 = int(vis_joint[0] - 1)
                y1 = int(vis_joint[1] - 1)
                x2 = int(vis_joint[0] + 2)
                y2 = int(vis_joint[1] + 2)
                save_img = cv2.rectangle(save_img, (x1, y1), \
                            (x2, y2), (0,0,200), 3, cv2.LINE_AA)
            
        cv2.imwrite("test_imgs/"+output['image_paths'][0].split('/')[-1], save_img)

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(np_outputs, args.out)

        #print(outputs)#[{'preds': [32, 17, 3], 'boxes' : [32, 6], 'image_paths': [32], 'bbox_ids': [32], 'output_heatmap': None} ...]
        results = dataset.evaluate(np_outputs, cfg.work_dir, **eval_config)
        print(results)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
