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
class TopDownVGGDataset(Kpt2dSviewRgbImgTopDownDataset):
    """VGG Youtube Dataset for top-down pose estimation.

    The videos were downloaded from YouTube and automatically scaled and cropped (using an upper body detector) so that the shoulder width is 100 pixels wide (on average across the video). 100 frames from each of the scaled and cropped videos were randomly chosen and manually annotated with upper body joints. The Head, Right wrist, Left wrist, Right elbow, Left elbow, Right shoulder and Left shoulder were annotated.

    Annotated frames are contained within the GT_frames.zip package.

    The matlab file YouTube_Pose_dataset.mat contains a structure array called 'data'. There are 50 elements in the array, one for each video of the dataset. Each element is structured as follows:

    data(i).url - string for the youtube weblink for video i
    data(i).videoname - string for the code name of the youtube video
    data(i).locs - 2 by 7 by 100 array containing 2D locations for the ground truth upper body joints. Row 1 are x values and Row 2 are y values. Columns are formatted from left to right as: Head, Right wrist, Left wrist, Right elbow, Left elbow, Right shoulder and Left shoulder (Person centric).
    data(i).frameids = 1 by 100 array containing the frame indicies which were annotated.
    data(i).label_names - cell array of strings for corresponding body joint labels
    data(i).crop - 1 by 4 array giving the crop bounding box [topx topy botx boty] from the original video
    data(i).scale - value the video should be scaled by
    data(i).imgPath - cell array containing paths to the pre scaled and cropped annotated frames
    data(i).origRes - 1 by 2 array [height,width] resolution of original video
    data(i).isYouTubeSubset - boolean, true if video belongs to the YouTube Subset dataset

    e.g. data(i).imgPath{f} refers to video i and frame f, with frame id data(i).frameids(f) and joint locations data(i).locs(:,:,f)

    Note: ground truth body joint locations correspond to cropped and scaled videos (original videos are first cropped, then scaled using the appropriate values).

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
        mat_data = mat['data'][0]
        mat_list = []
        for i in range(mat_data.shape[0]):
            mat_list.append(np.asarray(mat_data[i]))
        anno = np.array(mat_list)[0]
        #numpy data of 50 videos

        gt_db = []
        bbox_id = 0
        for a_raw in anno:
            a = list(a_raw)
            #list with length 9
            #0.url - string for the youtube weblink for video i
            #1.videoname - string for the code name of the youtube video
            #2.locs - 2 by 7 by 100 array containing 2D locations for the ground truth upper body joints. Row 1 are x values and Row 2 are y values. Columns are formatted from left to right as: Head, Right wrist, Left wrist, Right elbow, Left elbow, Right shoulder and Left shoulder (Person centric).
            #3.frameids = 1 by 100 array containing the frame indicies which were annotated.
            #4.label_names - cell array of strings for corresponding body joint labels
            #5.scale - value the video should be scaled by
            #6.? - 1 by 4 array giving the crop bounding box [topx topy botx boty] from the original video (MAYBE)
            #7.origRes - 1 by 2 array [height,width] resolution of original video
            #8.isYouTubeSubset - boolean, true if video belongs to the YouTube Subset dataset
            image_dir = a[1][0]
            
            key_points_ = np.transpose(a[2], (2, 1, 0))#[2,7,100]->[100,7,2]

            for frame_i in range(100):
                frame_name = a[3][0][frame_i]
                image_file_name = "frame_"+(6-len(str(frame_name[frame_i])))*"0"+str(frame_name[frame_i])+".jpg"
                image_name = osp.join(image_dir, image_file_name)                

                key_points = key_points_[frame_i]
                head_kp = key_points[0]
                right_wrist_kp = key_points[1]
                left_wrist_kp = key_points[2]
                right_elbow_kp = key_points[3]
                left_elbow_kp = key_points[4]
                right_shoulder_kp = key_points[5]
                left_shoulder_kp = key_points[6]                

                center = np.average(key_points, axis=0, dtype=np.float32)
                scale = (np.max(key_points, axis=0, dtype=np.float32) - np.min(key_points, axis=0, dtype=np.float32))*2

                # Adjust center/scale slightly to avoid cropping limbs
                if center[0] != -1:
                    center[1] = center[1] + 15 * scale[1]

                # MPII uses matlab format, index is 1-based,
                # we should first convert to 0-based index
                center = center - 1
                
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)#code from coco

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

        gt_file = osp.join(osp.dirname(self.ann_file), 'mpii_gt_val.mat')
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = headsizes * np.ones((len(uv_err), 1), dtype=np.float32)
        scaled_uv_err = uv_err / scale
        scaled_uv_err = scaled_uv_err * jnt_visible
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
        PCKh = 100. * np.sum(less_than_threshold, axis=1) / jnt_count

        # save
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), 16), dtype=np.float32)

        for r, threshold in enumerate(rng):
            less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
            pckAll[r, :] = 100. * np.sum(
                less_than_threshold, axis=1) / jnt_count

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [('Head', PCKh[head]),
                      ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
                      ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
                      ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
                      ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
                      ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
                      ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
                      ('PCKh', np.sum(PCKh * jnt_ratio)),
                      ('PCKh@0.1', np.sum(pckAll[10, :] * jnt_ratio))]
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
