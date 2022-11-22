# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import warnings
from collections import OrderedDict

import numpy as np
from mmcv import Config, deprecated_api_warning
from scipy.io import loadmat, savemat

from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class TopDownVGGBBCDataset(Kpt2dSviewRgbImgTopDownDataset):
    """VGG Youtube Dataset for top-down pose estimation.

1. bbcpose.mat:
        Contains all dataset metadata in one variable.
        demo.m shows an example of how to use this data.


        bbcpose =
        1x92 struct array with fields:
                videoName               - original video name
                type                    - train/test/val
                source                  - which annotation method has been used
                train_frames    - train frame numbers
                train_joints    - training joints
                test_frames             - manual ground truth frames (if any -- only for test frames)
                test_joints             - manual ground truth joints (if any -- only for test frames)

    VGG keypoint indexes::

        0: 'head',
        1: 'right_wrist',
        2: 'left_wrist',
        3: 'right_elbow',
        4: 'left_elbow',
        5: 'right_shoulder',
        6: 'left_shoulder'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/vgg.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            coco_style=False,
            test_mode=test_mode)

        self.db = self._get_db()
        self.image_set = set(x['image_file'] for x in self.db)
        self.num_images = len(self.image_set)

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        # create train/val split
        mat = loadmat(self.ann_file)
        anno = mat['bbcpose'][0][:20]

        gt_db = []
        bbox_id = 0
        for index, a_raw in enumerate(anno):
            a = a_raw
            #bbcpose =
            #1x92 struct array with fields:
            #    0.videoName               - original video name
            #    1.type                    - train/test/val
            #    2.source                  - which annotation method has been used
            #    3.train_frames    - train frame numbers
            #    4.train_joints    - training joints
            #    5.test_frames             - manual ground truth frames (if any -- only for test frames)
            #    6.test_joints             - manual ground truth joints (if any -- only for test frames)
            image_dir = str(index+1)
            
            if not self.test_mode:#train
                #if a[1][0] == 'train':
                key_points_ = np.transpose(a[4], (2, 1, 0))#[2,7,N]->[N,7,2]
                target_frames = a[3][0]
            else:#testval
                if a[1][0] in ['val','test']:
                    key_points_ = np.transpose(a[6], (2, 1, 0))#[2,7,N]->[N,7,2]
                    target_frames = a[5][0]
                else:
                    key_points_ = np.array([])
                    target_frames = np.array([])

            for frame_i, frame_name in enumerate(target_frames):
                image_file_name = str(int(frame_name))+".jpg"
                image_name = osp.join(image_dir, image_file_name)                

                key_points = key_points_[frame_i]
                #print(key_points)            

                center = np.average(key_points, axis=0)#, dtype=np.float32)
                scale = (np.max(key_points, axis=0) - np.min(key_points, axis=0))/100.

                # Adjust center/scale slightly to avoid cropping limbs
                if center[0] != -1:
                    center[1] = center[1] + 15 * scale[1]

                # MPII uses matlab format, index is 1-based,
                # we should first convert to 0-based index
                center = center - 1
                
                num_joints = 7
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)#code from coco

                if not self.test_mode:
                    joints = key_points
                    assert len(joints) == self.ann_info['num_joints'], \
                        f'joint num diff: {len(joints)}' + \
                        f' vs {self.ann_info["num_joints"]}'

                    joints_3d[:, 0:2] = joints[:, 0:2] - 1
                    joints_3d_visible[:, :2] = 1

                image_file = osp.join(self.img_prefix, image_name)
                gt_db.append({
                    'image_file': image_file,
                    'bbox_id': bbox_id,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox_score': 1
                })
                #print(str(bbox_id), " : ", image_file)
                #print(str(bbox_id), " : ", center)
                #print(str(bbox_id), " : ", scale)
                #print(str(bbox_id), " : ", joints_3d)
                #print(str(bbox_id), " : ", joints_3d_visible)
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='PCKh', **kwargs):
        """Evaluate PCKh for MPII dataset. Adapted from
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['/val2017/000000\
                    397133.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap.
            res_folder (str, optional): The folder to save the testing
                results. Default: None.
            metric (str | list[str]): Metrics to be performed.
                Defaults: 'PCKh'.

        Returns:
            dict: PCKh for each joint
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCKh']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        kpts = []
        for result in results:
            preds = result['preds']
            bbox_ids = result['bbox_ids']
            batch_size = len(bbox_ids)
            for i in range(batch_size):
                kpts.append({'keypoints': preds[i], 'bbox_id': bbox_ids[i]})
        kpts = self._sort_and_unique_bboxes(kpts)

        preds = np.stack([kpt['keypoints'] for kpt in kpts])

        # convert 0-based index to 1-based index,
        # and get the first two dimensions.
        preds = preds[..., :2] + 1.0

        if res_folder:
            pred_file = osp.join(res_folder, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        SC_BIAS = 0.6
        threshold = 0.5

        mat = loadmat(self.ann_file)
        mat_data = mat['bbcpose'][0][10:20]
        loc_array = []
        for i in range(mat_data.shape[0]):# for test+val set
            loc_array.append(np.transpose(mat_data[i][6], (2,1,0)))#10*[2,7,200]->10*[200,7,2]
        loc_array = np.array(loc_array)#10*[200,7,2]->[10*200,7,2]
        pos_gt_src = np.transpose(np.concatenate(loc_array,axis=0), (1,2,0))#[10*200,7,2]->[7,2,10*200]

        pos_pred_src = np.transpose(preds, [1, 2, 0])#[K,2,N]

        #jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)#[K,N]

        num_joints = 7#np.sum(jnt_visible, axis=1)
        
        brief_head_box = (np.max(pos_gt_src, axis=0) - np.min(pos_gt_src, axis=0))/14.#[2,5000]
        brief_head_size = np.multiply(brief_head_box[0],brief_head_box[1])
        headsizes = brief_head_size*SC_BIAS
        scale = headsizes * np.ones((num_joints, 1), dtype=np.float32)
        scaled_uv_err = uv_err / scale
        less_than_threshold = (scaled_uv_err <= threshold)# * jnt_visible
        PCKh = 100. * np.sum(less_than_threshold, axis=1) / pos_gt_src.shape[2]#[K]

        # saved
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), num_joints), dtype=np.float32)

        for r, threshold in enumerate(rng):
            less_than_threshold = (scaled_uv_err <= threshold)# * jnt_visible
            pckAll[r, :] = 100. * np.sum(
                less_than_threshold, axis=1) / pos_gt_src.shape[2]


        #PCKh = np.ma.array(PCKh, mask=False)
        #PCKh.mask[6:8] = True

        #jnt_count = np.ma.array(jnt_count, mask=False)
        #jnt_count.mask[6:8] = True
        #jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [('Head', PCKh[0]),
                      ('Shoulder', 0.5 * (PCKh[5] + PCKh[6])),
                      ('Elbow', 0.5 * (PCKh[3] + PCKh[4])),
                      ('Wrist', 0.5 * (PCKh[1] + PCKh[2])),
                      ('PCKh', np.sum(PCKh)/num_joints), #* jnt_ratio)),
                      ('PCKh@0.1', np.sum(pckAll[10, :])/num_joints)]# * jnt_ratio))]
        name_value = OrderedDict(name_value)

        return name_value

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
