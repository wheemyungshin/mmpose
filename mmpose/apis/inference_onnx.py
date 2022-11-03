# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings
from collections import defaultdict

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.utils.misc import deprecated_api_warning
from PIL import Image

from mmpose.core.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.core import imshow_bboxes, imshow_keypoints

from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose

@deprecated_api_warning(name_dict=dict(img_or_path='imgs_or_paths'))
def inference_top_down_pose_model_onnx(ort_session,
                                  imgs_or_paths,
                                  person_results=None,
                                  bbox_thr=None,
                                  format='xywh',
                                  dataset='TopDownCocoDataset',
                                  dataset_info=None,
                                  return_heatmap=False,
                                  outputs=None,
                                  config=None):
    """Inference a single image with a list of person bounding boxes. Support
    single-frame and multi-frame inference setting.

    Note:
        - num_frames: F
        - num_people: P
        - num_keypoints: K
        - bbox height: H
        - bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        imgs_or_paths (str | np.ndarray | list(str) | list(np.ndarray)):
            Image filename(s) or loaded image(s).
        person_results (list(dict), optional): a list of detected persons that
            contains ``bbox`` and/or ``track_id``:

            - ``bbox`` (4, ) or (5, ): The person bounding box, which contains
                4 box coordinates (and score).
            - ``track_id`` (int): The unique id for each human instance. If
                not provided, a dummy person result with a bbox covering
                the entire image will be used. Default: None.
        bbox_thr (float | None): Threshold for bounding boxes. Only bboxes
            with higher scores will be fed into the pose detector.
            If bbox_thr is None, all boxes will be used.
        format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.

            - `xyxy` means (left, top, right, bottom),
            - `xywh` means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
            It is deprecated. Please use dataset_info instead.
        dataset_info (DatasetInfo): A class containing all dataset info.
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned. Default: None.

    Returns:
        tuple:
        - pose_results (list[dict]): The bbox & pose info. \
            Each item in the list is a dictionary, \
            containing the bbox: (left, top, right, bottom, [score]) \
            and the pose (ndarray[Kx3]): x, y, score.
        - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
            torch.Tensor[N, K, H, W]]]): \
            Output feature maps from layers specified in `outputs`. \
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # decide whether to use multi frames for inference
    use_multi_frames = False

    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    pose_results = []
    returned_outputs = []

    if person_results is None:
        # create dummy person results
        sample = imgs_or_paths[0] if use_multi_frames else imgs_or_paths
        if isinstance(sample, str):
            width, height = Image.open(sample).size
        else:
            height, width = sample.shape[:2]
        person_results = [{'bbox': np.array([0, 0, width, height])}]

    if len(person_results) == 0:
        return pose_results

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        person_results = [person_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = bbox_xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = bbox_xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return []

    # poses is results['pred'] # N x 17x 3
    poses = _inference_single_pose_model_onnx(
        ort_session,
        imgs_or_paths,
        bboxes_xywh,
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        use_multi_frames=use_multi_frames,
        config=config)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy))
    for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                              bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results

def _inference_single_pose_model_onnx(ort_session,
                                 imgs_or_paths,
                                 bboxes,
                                 dataset='TopDownCocoDataset',
                                 dataset_info=None,
                                 return_heatmap=False,
                                 use_multi_frames=False,
                                 config=None):
    """Inference human bounding boxes.

    Note:
        - num_frames: F
        - num_bboxes: N
        - num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        imgs_or_paths (list(str) | list(np.ndarray)): Image filename(s) or
            loaded image(s)
        bboxes (list | np.ndarray): All bounding boxes (with scores),
            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
            where N is number of bounding boxes.
        dataset (str): Dataset name. Deprecated.
        dataset_info (DatasetInfo): A class containing all dataset info.
        return_heatmap (bool): Flag to return heatmap, default: False
        use_multi_frames (bool): Flag to use multi frames for inference

    Returns:
        ndarray[NxKx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    cfg = config
    device = -1

    if use_multi_frames:
        assert 'frame_weight_test' in cfg.data.test.data_cfg
        # use multi frames for inference
        # the number of input frames must equal to frame weight in the config
        assert len(imgs_or_paths) == len(
            cfg.data.test.data_cfg.frame_weight_test)

    # build the data pipeline
    _test_pipeline = copy.deepcopy(cfg.test_pipeline)

    has_bbox_xywh2cs = False
    for transform in _test_pipeline:
        if transform['type'] == 'TopDownGetBboxCenterScale':
            has_bbox_xywh2cs = True
            break
    if not has_bbox_xywh2cs:
        _test_pipeline.insert(
            0, dict(type='TopDownGetBboxCenterScale', padding=1.25))
    test_pipeline = Compose(_test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    if dataset_info is not None:
        dataset_name = dataset_info.dataset_name
        flip_pairs = dataset_info.flip_pairs
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        if dataset in ('TopDownCocoDataset', 'TopDownOCHumanDataset',
                       'AnimalMacaqueDataset'):
            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                          [13, 14], [15, 16]]
        elif dataset == 'TopDownCocoWholeBodyDataset':
            body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                    [13, 14], [15, 16]]
            foot = [[17, 20], [18, 21], [19, 22]]

            face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                    [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                    [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                    [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                    [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

            hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                    [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                    [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                    [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                    [111, 132]]
            flip_pairs = body + foot + face + hand
        elif dataset == 'TopDownAicDataset':
            flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
        elif dataset == 'TopDownMpiiDataset':
            flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        elif dataset == 'TopDownMpiiTrbDataset':
            flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
                          [14, 15], [16, 22], [28, 34], [17, 23], [29, 35],
                          [18, 24], [30, 36], [19, 25], [31, 37], [20, 26],
                          [32, 38], [21, 27], [33, 39]]
        elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                         'PanopticDataset', 'InterHand2DDataset'):
            flip_pairs = []
        elif dataset in 'Face300WDataset':
            flip_pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11],
                          [6, 10], [7, 9], [17, 26], [18, 25], [19, 24],
                          [20, 23], [21, 22], [31, 35], [32, 34], [36, 45],
                          [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                          [48, 54], [49, 53], [50, 52], [61, 63], [60, 64],
                          [67, 65], [58, 56], [59, 55]]

        elif dataset in 'FaceAFLWDataset':
            flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                          [12, 14], [15, 17]]

        elif dataset in 'FaceCOFWDataset':
            flip_pairs = [[0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11],
                          [12, 14], [16, 17], [13, 15], [18, 19], [22, 23]]

        elif dataset in 'FaceWFLWDataset':
            flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
                          [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
                          [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],
                          [33, 46], [34, 45], [35, 44], [36, 43], [37, 42],
                          [38, 50], [39, 49], [40, 48], [41, 47], [60, 72],
                          [61, 71], [62, 70], [63, 69], [64, 68], [65, 75],
                          [66, 74], [67, 73], [55, 59], [56, 58], [76, 82],
                          [77, 81], [78, 80], [87, 83], [86, 84], [88, 92],
                          [89, 91], [95, 93], [96, 97]]

        elif dataset in 'AnimalFlyDataset':
            flip_pairs = [[1, 2], [6, 18], [7, 19], [8, 20], [9, 21], [10, 22],
                          [11, 23], [12, 24], [13, 25], [14, 26], [15, 27],
                          [16, 28], [17, 29], [30, 31]]
        elif dataset in 'AnimalHorse10Dataset':
            flip_pairs = []

        elif dataset in 'AnimalLocustDataset':
            flip_pairs = [[5, 20], [6, 21], [7, 22], [8, 23], [9, 24],
                          [10, 25], [11, 26], [12, 27], [13, 28], [14, 29],
                          [15, 30], [16, 31], [17, 32], [18, 33], [19, 34]]

        elif dataset in 'AnimalZebraDataset':
            flip_pairs = [[3, 4], [5, 6]]

        elif dataset in 'AnimalPoseDataset':
            flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15],
                          [16, 17], [18, 19]]
        else:
            raise NotImplementedError()
        dataset_name = dataset

    batch_data = []
    for bbox in bboxes:
        # prepare data
        data = {
            'bbox':
            bbox,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset_name,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs
            }
        }

        if use_multi_frames:
            # weight for different frames in multi-frame inference setting
            data['frame_weight'] = cfg.data.test.data_cfg.frame_weight_test
            if isinstance(imgs_or_paths[0], np.ndarray):
                data['img'] = imgs_or_paths
            else:
                data['image_file'] = imgs_or_paths
        else:
            if isinstance(imgs_or_paths, np.ndarray):
                data['img'] = imgs_or_paths
            else:
                data['image_file'] = imgs_or_paths

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, [device])[0]

    results = []
    for i in range(len(batch_data['img_metas'])):
        ort_inputs = {ort_session.get_inputs()[0].name: batch_data['img'][i].unsqueeze(0).numpy()}
        ort_outs = ort_session.run(None, ort_inputs)[0]

        result = decode_heatmap([batch_data['img_metas'][i]], ort_outs, cfg)['preds'][0]
        results.append(result)

    # forward the model
    #with torch.no_grad():
    #    result = model(
    #        img=batch_data['img'],
    #        img_metas=batch_data['img_metas'],
    #        return_loss=False,
    #        return_heatmap=return_heatmap)

    return results


def decode_heatmap(img_metas, output, config, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=config.model.test_cfg.get('unbiased_decoding', False),
            post_process=config.model.test_cfg.get('post_process', 'default'),
            kernel=config.model.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=config.model.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=config.model.test_cfg.get('use_udp', False),
            target_type=config.model.test_cfg.get('target_type', 'GaussianHeatmap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

def vis_pose_result_onnx(ort_session,
                    img,
                    result,
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset='TopDownCocoDataset',
                    dataset_info=None,
                    show=False,
                    out_file=None,
                    config=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """

    # get dataset info
    if config is not None:
        dataset_info = DatasetInfo(config.dataset_info)

    if dataset_info is not None:
        skeleton = dataset_info.skeleton
        pose_kpt_color = dataset_info.pose_kpt_color
        pose_link_color = dataset_info.pose_link_color
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])

        if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
            # show the results
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]

            pose_link_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
            ]]

        elif dataset == 'TopDownCocoWholeBodyDataset':
            # show the results
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2],
                        [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
                        [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
                        [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                        [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                        [102, 103], [91, 104], [104, 105], [105, 106],
                        [106, 107], [91, 108], [108, 109], [109, 110],
                        [110, 111], [112, 113], [113, 114], [114, 115],
                        [115, 116], [112, 117], [117, 118], [118, 119],
                        [119, 120], [112, 121], [121, 122], [122, 123],
                        [123, 124], [112, 125], [125, 126], [126, 127],
                        [127, 128], [112, 129], [129, 130], [130, 131],
                        [131, 132]]

            pose_link_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ] + [16, 16, 16, 16, 16, 16] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
            pose_kpt_color = palette[
                [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]

        elif dataset == 'TopDownAicDataset':
            skeleton = [[2, 1], [1, 0], [0, 13], [13, 3], [3, 4], [4, 5],
                        [8, 7], [7, 6], [6, 9], [9, 10], [10, 11], [12, 13],
                        [0, 6], [3, 9]]

            pose_link_color = palette[[
                9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 0, 7, 7
            ]]
            pose_kpt_color = palette[[
                9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 0, 0
            ]]

        elif dataset == 'TopDownMpiiDataset':
            skeleton = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                        [7, 8], [8, 9], [8, 12], [12, 11], [11, 10], [8, 13],
                        [13, 14], [14, 15]]

            pose_link_color = palette[[
                16, 16, 16, 16, 16, 16, 7, 7, 0, 9, 9, 9, 9, 9, 9
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 16, 7, 7, 0, 0, 9, 9, 9, 9, 9, 9
            ]]

        elif dataset == 'TopDownMpiiTrbDataset':
            skeleton = [[12, 13], [13, 0], [13, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [0, 6], [1, 7], [6, 7], [6, 8], [7,
                                                                 9], [8, 10],
                        [9, 11], [14, 15], [16, 17], [18, 19], [20, 21],
                        [22, 23], [24, 25], [26, 27], [28, 29], [30, 31],
                        [32, 33], [34, 35], [36, 37], [38, 39]]

            pose_link_color = palette[[16] * 14 + [19] * 13]
            pose_kpt_color = palette[[16] * 14 + [0] * 26]

        elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                         'PanopticDataset'):
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
                        [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
                        [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
                        [18, 19], [19, 20]]

            pose_link_color = palette[[
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
            pose_kpt_color = palette[[
                0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16,
                16, 16
            ]]

        elif dataset == 'InterHand2DDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9],
                        [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
                        [16, 17], [17, 18], [18, 19], [3, 20], [7, 20],
                        [11, 20], [15, 20], [19, 20]]

            pose_link_color = palette[[
                0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12,
                16
            ]]
            pose_kpt_color = palette[[
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16, 0
            ]]

        elif dataset == 'Face300WDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 68]
            kpt_score_thr = 0

        elif dataset == 'FaceAFLWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 19]
            kpt_score_thr = 0

        elif dataset == 'FaceCOFWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 29]
            kpt_score_thr = 0

        elif dataset == 'FaceWFLWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 98]
            kpt_score_thr = 0

        elif dataset == 'AnimalHorse10Dataset':
            skeleton = [[0, 1], [1, 12], [12, 16], [16, 21], [21, 17],
                        [17, 11], [11, 10], [10, 8], [8, 9], [9, 12], [2, 3],
                        [3, 4], [5, 6], [6, 7], [13, 14], [14, 15], [18, 19],
                        [19, 20]]

            pose_link_color = palette[[4] * 10 + [6] * 2 + [6] * 2 + [7] * 2 +
                                      [7] * 2]
            pose_kpt_color = palette[[
                4, 4, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 7, 7, 7, 4, 4, 7, 7, 7,
                4
            ]]

        elif dataset == 'AnimalFlyDataset':
            skeleton = [[1, 0], [2, 0], [3, 0], [4, 3], [5, 4], [7, 6], [8, 7],
                        [9, 8], [11, 10], [12, 11], [13, 12], [15, 14],
                        [16, 15], [17, 16], [19, 18], [20, 19], [21, 20],
                        [23, 22], [24, 23], [25, 24], [27, 26], [28, 27],
                        [29, 28], [30, 3], [31, 3]]

            pose_link_color = palette[[0] * 25]
            pose_kpt_color = palette[[0] * 32]

        elif dataset == 'AnimalLocustDataset':
            skeleton = [[1, 0], [2, 1], [3, 2], [4, 3], [6, 5], [7, 6], [9, 8],
                        [10, 9], [11, 10], [13, 12], [14, 13], [15, 14],
                        [17, 16], [18, 17], [19, 18], [21, 20], [22, 21],
                        [24, 23], [25, 24], [26, 25], [28, 27], [29, 28],
                        [30, 29], [32, 31], [33, 32], [34, 33]]

            pose_link_color = palette[[0] * 26]
            pose_kpt_color = palette[[0] * 35]

        elif dataset == 'AnimalZebraDataset':
            skeleton = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 7], [6, 7], [7, 2],
                        [8, 7]]

            pose_link_color = palette[[0] * 8]
            pose_kpt_color = palette[[0] * 9]

        elif dataset in 'AnimalPoseDataset':
            skeleton = [[0, 1], [0, 2], [1, 3], [0, 4], [1, 4], [4, 5], [5, 7],
                        [6, 7], [5, 8], [8, 12], [12, 16], [5, 9], [9, 13],
                        [13, 17], [6, 10], [10, 14], [14, 18], [6, 11],
                        [11, 15], [15, 19]]

            pose_link_color = palette[[0] * 20]
            pose_kpt_color = palette[[0] * 20]
        else:
            NotImplementedError()

    img = show_result_(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img

def show_result_(img,
                result,
                skeleton=None,
                kpt_score_thr=0.3,
                bbox_color='green',
                pose_kpt_color=None,
                pose_link_color=None,
                text_color='white',
                radius=4,
                thickness=1,
                font_scale=0.5,
                bbox_thickness=1,
                win_name='',
                show=False,
                show_keypoint_weight=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        skeleton (list[list]): The connection of keypoints.
            skeleton is 0-based indexing.
        kpt_score_thr (float, optional): Minimum score of keypoints
            to be shown. Default: 0.3.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
            If None, do not draw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links.
            If None, do not draw links.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        win_name (str): The window name.
        show (bool): Whether to show the image. Default: False.
        show_keypoint_weight (bool): Whether to change the transparency
            using the predicted confidence scores of keypoints.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        Tensor: Visualized img, only if not `show` or `out_file`.
    """
    img = mmcv.imread(img)
    img = img.copy()

    bbox_result = []
    bbox_labels = []
    pose_result = []
    for res in result:
        if 'bbox' in res:
            bbox_result.append(res['bbox'])
            bbox_labels.append(res.get('label', None))
        pose_result.append(res['keypoints'])

    if bbox_result:
        bboxes = np.vstack(bbox_result)
        # draw bounding boxes
        imshow_bboxes(
            img,
            bboxes,
            labels=bbox_labels,
            colors=bbox_color,
            text_color=text_color,
            thickness=bbox_thickness,
            font_scale=font_scale,
            show=False)

    if pose_result:
        img = imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                            pose_kpt_color, pose_link_color, radius,
                            thickness)

    if show:
        imshow(img, win_name, wait_time)

    if out_file is not None:
        imwrite(img, out_file)

    return img