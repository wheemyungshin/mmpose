# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
import numpy as np

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from collections import defaultdict

#def concat_tile(im_tiles):
#    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_tiles])

def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
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
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
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

    # read video

    video_dict = defaultdict(list)
    for video_file in os.listdir(args.video_path):
        if video_file.endswith(('.MP4','.mp4')):
            angle = int(video_file[-19])
            print(video_file)
            print(video_file[-17:-4])
            print(angle)
            dict_name = video_file[:-19]+video_file[-17:-10]
            print(dict_name)
            if len(video_dict[dict_name]) == 0:
                video_dict[dict_name] = [None, None, None, None, None, None, None, None, None]
            video_dict[dict_name][angle] = video_file            


    print(video_dict)

    for video_angles in video_dict.values():
        video_list_9view = [[], [], [],
                            [], [], [],
                            [], [], []]
        video = None
        for angle_num, video_file in enumerate(video_angles):
            if video_file is not None:
                video_path = os.path.join(args.video_path, video_file)
                video = mmcv.VideoReader(video_path)
                assert video.opened, f'Faild to load video file {video_path}'

                if args.out_video_root == '':
                    save_out_video = False
                else:
                    os.makedirs(args.out_video_root, exist_ok=True)
                    save_out_video = True

                if save_out_video:
                    fps = video.fps
                    size = (video.width, video.height)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    videoWriter = cv2.VideoWriter(
                        os.path.join(args.out_video_root,
                                    f'vis_{os.path.basename(video_path)}'), fourcc,
                        fps, size)

                # frame index offsets for inference, used in multi-frame inference setting
                if args.use_multi_frames:
                    assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
                    indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

                # whether to return heatmap, optional
                return_heatmap = False

                # return the output of some desired layers,
                # e.g. use ('backbone', ) to return backbone feature
                output_layer_names = None

                print('Running inference...:', video_path)
                for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
                    if ((frame_id < 1000) or (8000 < frame_id < 9000) or (10500 < frame_id < 11500) or (13000 < frame_id < 14000)):
                        # get the detection results of current frame
                        # the resulting box is (x1, y1, x2, y2)
                        mmdet_results = inference_detector(det_model, cur_frame)

                        # keep the person class bounding boxes.
                        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

                        if args.use_multi_frames:
                            frames = collect_multi_frames(video, frame_id, indices,
                                                        args.online)

                        # test a single image, with a list of bboxes.
                        pose_results, returned_outputs = inference_top_down_pose_model(
                            pose_model,
                            frames if args.use_multi_frames else cur_frame,
                            person_results,
                            bbox_thr=args.bbox_thr,
                            format='xyxy',
                            dataset=dataset,
                            dataset_info=dataset_info,
                            return_heatmap=return_heatmap,
                            outputs=output_layer_names)

                        # show the results
                        vis_frame = vis_pose_result(
                            pose_model,
                            cur_frame,
                            pose_results,
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
                        resize_vis_frame = cv2.resize(vis_frame, dsize=(0, 0), fx=0.333, fy=0.333, interpolation=cv2.INTER_AREA)
                        video_list_9view[angle_num].append(resize_vis_frame)


                        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                if save_out_video:
                    videoWriter.release()
                if args.show:
                    cv2.destroyAllWindows()

        if args.out_video_root == '':
            save_out_video = False
        else:
            os.makedirs(args.out_video_root, exist_ok=True)
            save_out_video = True
        
        
        if save_out_video and video is not None:
            fps = video.fps
            size = (video.width, video.height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(args.out_video_root,
                            f'9view_{os.path.basename(video_path)}'), fourcc,
                fps, size)
            
            for frame_index in range(len(video_list_9view[0])):
                #tmp_video_list_9view = []
                #for i in range(9):
                #    if len(video_list_9view[i])==0:
                #        tmp_video_list_9view.append(np.zeros([int(video.width*0.333), int(video.height*0.333), 3], dtype=np.uint8))
                #    else:
                #        tmp_video_list_9view.append(video_list_9view[i][frame_index])

                #im_tiles = [[tmp_video_list_9view[0], tmp_video_list_9view[1], tmp_video_list_9view[2]],
                #            [tmp_video_list_9view[3], tmp_video_list_9view[4], tmp_video_list_9view[5]],
                #            [tmp_video_list_9view[6], tmp_video_list_9view[7], tmp_video_list_9view[8]]]
                #img_9view = np.zeros([video.width, video.height, 3], dtype=np.uint8)
                #img_9v,iew_ = concat_tile(im_tiles)
                #img_9view[:img_9view_.shape[0],:img_9view_.shape[1],:img_9view_.shape[2]] = img_9view_

                img_9view = np.zeros([video.height, video.width, 3], dtype=np.uint8)

                cut_h = int(img_9view.shape[0]*0.333)
                cut_w = int(img_9view.shape[1]*0.333)
                if len(video_list_9view[0]) > frame_index:
                    img_9view[:cut_h, :cut_w, :img_9view.shape[2]] \
                     = video_list_9view[0][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]
                if len(video_list_9view[1]) > frame_index:
                    img_9view[:cut_h, cut_w:cut_w*2, :img_9view.shape[2]] \
                     = video_list_9view[1][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]
                if len(video_list_9view[2]) > frame_index:
                    img_9view[:cut_h, cut_w*2:cut_w*3, :img_9view.shape[2]] \
                     = video_list_9view[2][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]

                if len(video_list_9view[3]) > frame_index:
                    img_9view[cut_h:cut_h*2, :cut_w, :img_9view.shape[2]] \
                     = video_list_9view[3][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]
                if len(video_list_9view[4]) > frame_index:
                    img_9view[cut_h:cut_h*2, cut_w:cut_w*2, :img_9view.shape[2]] \
                     = video_list_9view[4][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]
                if len(video_list_9view[5]) > frame_index:
                    img_9view[cut_h:cut_h*2, cut_w*2:cut_w*3, :img_9view.shape[2]] \
                     = video_list_9view[5][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]

                if len(video_list_9view[6]) > frame_index:
                    img_9view[cut_h*2:cut_h*3, :cut_w, :img_9view.shape[2]] \
                     = video_list_9view[6][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]
                if len(video_list_9view[7]) > frame_index:
                    img_9view[cut_h*2:cut_h*3, cut_w:cut_w*2, :img_9view.shape[2]] \
                     = video_list_9view[7][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]
                if len(video_list_9view[8]) > frame_index:
                    img_9view[cut_h*2:cut_h*3, cut_w*2:cut_w*3, :img_9view.shape[2]] \
                     = video_list_9view[8][frame_index][:cut_h, :cut_w, :img_9view.shape[2]]
                videoWriter.write(img_9view)
            videoWriter.release()


if __name__ == '__main__':
    main()
