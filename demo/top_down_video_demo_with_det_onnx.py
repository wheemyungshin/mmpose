# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (inference_top_down_pose_model_onnx,
                         vis_pose_result, vis_pose_result_onnx, inference_detector_onnx)
from mmpose.datasets import DatasetInfo


import numpy as np
import json

from collections import defaultdict

import onnx
import onnxruntime

import time

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
    """
    parser = ArgumentParser()	
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_onnx_file', help='onnx file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_onnx_file', help='onnx file for pose')
    parser.add_argument('--input-path', type=str, help='Video path (video or image / file or dir)')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--output-path',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
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
    parser.add_argument(
        '--detection-frame-stride',
        type=int,
        default=1)
    parser.add_argument(
        '--save-np',
        action='store_true',
        default=False,
        help='whether to save raw numpy outputs.')

    args = parser.parse_args()

    assert args.show or (args.output_path != '')
    assert args.det_config is not None
    assert args.pose_config is not None
    assert args.det_onnx_file is not None
    
    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file    
    det_model = onnx.load(args.det_onnx_file)
    onnx.checker.check_model(det_model)

    det_ort_session = onnxruntime.InferenceSession(args.det_onnx_file)    

    det_config = args.det_config
    if isinstance(det_config, str):
        det_config = mmcv.Config.fromfile(det_config)
    elif not isinstance(det_config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(det_config)}')

    pose_config = args.pose_config
    if isinstance(pose_config, str):
        pose_config = mmcv.Config.fromfile(pose_config)
    elif not isinstance(pose_config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(pose_config)}')

    pose_model = onnx.load(args.pose_onnx_file)
    onnx.checker.check_model(pose_model)

    pose_ort_session = onnxruntime.InferenceSession(args.pose_onnx_file)

    dataset = pose_config.data['test']['type']
    # get datasetinfo
    dataset_info = pose_config.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # load video
    input_path_list = []
    if os.path.isdir(args.input_path):
        print("Video Path is Dir.")
        for video_file in os.listdir(args.input_path):
            input_path = os.path.join(args.input_path, video_file)
            input_path_list.append(input_path)
    else:
        input_path_list = [args.input_path]

    # read videos / images
    start_time = time.time()
    last_time = start_time
    for input_path in input_path_list:
        input_type = None
        if os.path.splitext(input_path)[1] in [".mp4", ".MP4"]:
            print("Video file detected: {}".format(input_path))
            input_type = "video"
        elif os.path.splitext(input_path)[1] in [".jpg", ".png", ".jpeg", ".JPEG"]:
            print("Image file detected: {}".format(input_path), end="")
            input_type = "image"
        else:
            raise TypeError('the input file type is not supported: ', os.path.splitext(input_path)[1])

        if args.output_path == '':
            save_output = False
        else:
            os.makedirs(args.output_path, exist_ok=True)
            save_output = True

        if input_type == "video":
            video = mmcv.VideoReader(input_path)
            assert video.opened, f'Faild to load video file {input_path}'

            if save_output:
                fps = video.fps
                if args.resize_h == 0 or args.resize_w == 0:
                    size = (video.width, video.height)
                else:
                    size = (args.resize_w, args.resize_h)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                videoWriter = cv2.VideoWriter(
                    os.path.join(args.output_path,
                                f'vis_{os.path.basename(input_path)}'), fourcc,
                    fps, size)

            # return the output of some desired layers,
            # e.g. use ('backbone', ) to return backbone feature
            input_dataset = mmcv.track_iter_progress(video)
        
        elif input_type == "image":
            input_dataset = [cv2.imread(input_path)]

        person_results = None
        for frame_id, cur_frame in enumerate(input_dataset):
            if not (args.resize_h == 0 or args.resize_w == 0):
                cur_frame = cv2.resize(cur_frame, (args.resize_w, args.resize_h), interpolation=cv2.INTER_AREA)

            # get the detection results of current frame
            # the resulting box is (x1, y1, x2, y2)
            if frame_id == 0 or frame_id % args.detection_frame_stride == 0:
                person_results = inference_detector_onnx(det_ort_session, cur_frame, config=det_config, size=(args.resize_h, args.resize_w))
                
            #print("Before: ", person_results)
            filtered_person_results = []
            for person_result in person_results:
                person_size = (person_result['bbox'][2]-person_result['bbox'][0])*(person_result['bbox'][3]-person_result['bbox'][1])
                if person_size >= args.min_bbox_size:
                    filtered_person_results.append(person_result)
            #print("After: ", filtered_person_results)

            # test a single image, with a list of bboxes.
            pose_results, raw_results = inference_top_down_pose_model_onnx(
                pose_ort_session,
                cur_frame,
                filtered_person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None,
                config=pose_config,
                size=(args.resize_h, args.resize_w))

            #print("Before: ", pose_results)
            new_pose_results = []	
            for pose_result in pose_results:	
                kpoints = pose_result['keypoints'][:,-1]	
                # left right division / face division
                # even for right and odd for left, and zero is for nose	
                if np.sum(kpoints[2::2] > args.kpt_thr) >= args.min_points or np.sum(kpoints[1::2] > args.kpt_thr) >= args.min_points or \
                    np.sum(kpoints[:5] > args.kpt_thr) >= args.min_points :	
                    new_pose_results.append(pose_result)
            #print("After: ", new_pose_results)

            # show the results
            vis_frame = vis_pose_result_onnx(
                pose_ort_session,
                cur_frame,
                new_pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False,
                config=pose_config)

            if args.show:
                cv2.imshow('Frame', vis_frame)

            if save_output:
                if input_type == "video":
                    videoWriter.write(vis_frame)
                elif input_type == "image":
                    cv2.imwrite(os.path.join(args.output_path, f'vis_{os.path.basename(input_path)}'), vis_frame)
                    if raw_results is not None and args.save_np:
                        if len(raw_results[0])==2:
                            raw_results_save_x=[]
                            raw_results_save_y=[]
                            for raw_result in raw_results:
                                raw_results_save_x.append(raw_result[1])
                                raw_results_save_y.append(raw_result[0])
                            np.save(os.path.join(args.output_path, f'raw_x_{os.path.splitext(os.path.basename(input_path))[0]}'), raw_results_save_x)
                            np.save(os.path.join(args.output_path, f'raw_y_{os.path.splitext(os.path.basename(input_path))[0]}'), raw_results_save_y)
                        else:
                            raw_results_save=[]
                            for raw_result in raw_results:
                                raw_results_save.append(raw_result[0])
                            np.save(os.path.join(args.output_path, f'raw_{os.path.splitext(os.path.basename(input_path))[0]}'), raw_results_save)

            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if save_output and input_type == "video":
            videoWriter.release()
        if args.show:
            cv2.destroyAllWindows()
        print(" (Elapsed: {:.3f}s / Total: {:.3f}s)".format(time.time()-last_time,time.time()-start_time))
        last_time = time.time()


if __name__ == '__main__':
    main()
