# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, 
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

import numpy as np
import json

from collections import defaultdict

def load_person_detection_results(args):
    """Load coco person detection results."""
    num_joints = 17
    all_boxes = None
    with open(args.person_results, 'r') as f:
        all_boxes = json.load(f)

    if not all_boxes:
        raise ValueError('=> Load %s fail!' % args.person_results)

    print(f'=> Total boxes: {len(all_boxes)}')
    
    person_results = defaultdict(list)
    for det_res in all_boxes:
        if det_res['category_id'] != 1:
            continue

        box = det_res['bbox']
        score = det_res['score']
        box_5 = [box[0], box[1], box[2], box[3], score]

        person = {}
        person['bbox'] = box_5
        person_results[det_res['video_path']].append((person, det_res['image_id']))
    return person_results

def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('person_results', help='Json file for detection')
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
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--resize-w',
        type=int,
        default=0)
    parser.add_argument(
        '--resize-h',
        type=int,
        default=0)

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')

    print('Initializing model...')
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

    #read person results
    person_results_dict = load_person_detection_results(args)

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

        # frame index offsets for inference, used in multi-frame inference setting
        if args.use_multi_frames:
            assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
            indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

        # whether to return heatmap, optional
        return_heatmap = False

        # return the output of some desired layers,
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None


        print('Running inference...')
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            if not (args.resize_h == 0 or args.resize_w == 0):
                cur_frame = cv2.resize(cur_frame, (args.resize_w, args.resize_h), interpolation=cv2.INTER_AREA)

            person_results = person_results_dict[video_path.split("/")[-1]]

            filtered_person_results = []
            for person_result, person_image_id in person_results:
                if person_image_id == frame_id:
                    #print("person_result: ", person_result)
                    #print("person_image_id: ", person_image_id)
                    person_size = (person_result['bbox'][2]-person_result['bbox'][0])*(person_result['bbox'][3]-person_result['bbox'][1])
                    if person_size >= args.min_bbox_size:
                        filtered_person_results.append(person_result)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                frames if args.use_multi_frames else cur_frame,
                filtered_person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            new_pose_results = []
            for pose_result in pose_results:
                if np.sum(pose_result['keypoints'][:,-1] > args.kpt_thr) >= args.min_points:
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
