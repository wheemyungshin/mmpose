# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import numpy as np


def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path (video file or dir)')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--min-bbox-size',
        type=float,
        default=0,
        help='Bounding box size threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--min-points',
        type=int,
        default=0,
        help='Minimum predicted pose joints threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--resize-w',
        type=int,
        default=0)
    parser.add_argument(
        '--resize-h',
        type=int,
        default=0)


    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # load video
    video_path_list = []
    if os.path.isdir(args.video_path):
        print("Video Path is Dir.")
        for video_file in os.listdir(args.video_path):
            video_path = os.path.join(args.video_path, video_file)
            video_path_list.append(video_path)
    else:
        video_path_list = [args.video_path]

    # read video
    for video_path in video_path_list:
        video = mmcv.VideoReader(video_path)
        assert video.opened, f'Faild to load video file {video_path}'

        if args.out_video_root == '':
            save_out_video = False
        else:
            os.makedirs(args.out_video_root, exist_ok=True)
            save_out_video = True

        if save_out_video:
            fps = video.fps
            if args.resize_h == 0 or args.resize_w == 0:
                size = (video.width, video.height)
            else:
                size = (args.resize_w, args.resize_h)
            
            print("SIZE: ", size)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(args.out_video_root,
                            f'vis_{os.path.basename(video_path)}'), fourcc,
                fps, size)
        print("Loading...:", video_path)

        # whether to return heatmap, optional
        return_heatmap = False

        # return the output of some desired layers,
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        print('Running inference...')
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            if not (args.resize_h == 0 or args.resize_w == 0):
                cur_frame = cv2.resize(cur_frame, (args.resize_w, args.resize_h), interpolation=cv2.INTER_AREA)

            # get the detection results of current frame
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, cur_frame)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            filtered_person_results = []
            for person_result in person_results:
                person_size = (person_result['bbox'][2]-person_result['bbox'][0])*(person_result['bbox'][3]-person_result['bbox'][1])
                if person_size >= args.min_bbox_size:
                    filtered_person_results.append(person_result)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                cur_frame,
                filtered_person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            new_pose_results = []
            for pose_result in pose_results:
                kpoints = pose_result['keypoints'][:,-1]
                # left right division
                # even for right and odd for left, and zero is for nose
                if np.sum(kpoints[2::2] > args.kpt_thr) >= args.min_points or np.sum(kpoints[1::2] > args.kpt_thr) >= args.min_points:
                    new_pose_results.append(pose_result)

            # show the results
            vis_frame = vis_pose_result(
                pose_model,
                cur_frame,
                new_pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False)

            if args.show:
                cv2.imshow('Frame', vis_frame)

            if save_out_video:
                videoWriter.write(vis_frame)

            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if save_out_video:
            videoWriter.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
