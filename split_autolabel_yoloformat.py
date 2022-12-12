# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

import json


def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--source-path', type=str, help='Video path (video file or dir)')
    parser.add_argument('--auto-label-path', type=str, help='Video path (video file or dir)')
    parser.add_argument('--new-target-path', type=str, help='Video path (video file or dir)')
    parser.add_argument('--only-box-num', type=int, default=-999, help='Select frames with this number of boxes only)')
    parser.add_argument(
        '--frame-ratio',
        type=int,
        default=1,
        help="if frame ratio is 0, the ratio is the video's fps")
    
    args = parser.parse_args()

    # load video
    video_path_list = []
    if os.path.isdir(args.source_path):
        print("Video Path is Dir.")
        for video_file in os.listdir(args.source_path):
            if os.path.splitext(video_file)[-1] in [".mp4", ".MP4"]: 
                video_path = os.path.join(args.source_path, video_file)
                video_path_list.append(video_path)
    else:
        video_path_list = [args.source_path]
    
    auto_label_path = args.auto_label_path
    auto_label_last_folder = auto_label_path.split('/')[-1]

    save_image_dir = os.path.join(args.new_target_path, 'images', auto_label_last_folder)
    save_label_dir = os.path.join(args.new_target_path, 'labels', auto_label_last_folder)
    print(save_image_dir)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)
    
    # read video
    for video_path in video_path_list:
        rawname = os.path.splitext(os.path.basename(video_path))[0]
        video = mmcv.VideoReader(video_path)
        if not video.opened:
            print(f'Faild to load video file {video_path}')
            continue

        fps = video.fps
        print("Loading...:", video_path)

        # whether to return heatmap, optional
        return_heatmap = False

        # return the output of some desired layers,
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        print('Running inference...')
        if args.frame_ratio == 0:
            frame_ratio = int(fps)
        else:
            frame_ratio = args.frame_ratio
        
        json_file = os.path.join(auto_label_path, rawname+'.json')
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            img_size = json_data['size']
        
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            if frame_id % frame_ratio == 0:
                all_boxes = []
                for instance in json_data[str(frame_id)]:
                    all_boxes.append(instance['bbox'])

                print(len(all_boxes))
                if args.only_box_num < 0 or len(all_boxes) == args.only_box_num:
                    print(all_boxes)
                    save_label_name = os.path.join(save_label_dir, rawname+'_'+str(frame_id)+'.txt')
                    labels_f = open(save_label_name, 'w')
                    for i, box in enumerate(all_boxes):
                        line = ["0"]
                        line.append(str(max(min(round(box[0]/img_size[0], 6),1),0)))   
                        line.append(str(max(min(round(box[1]/img_size[1], 6),1),0)))   
                        line.append(str(max(min(round(box[2]/img_size[0], 6),1),0)))   
                        line.append(str(max(min(round(box[3]/img_size[1], 6),1),0))) 

                        labels_f.write(' '.join(line))                       
                        labels_f.write('\n')
                    labels_f.close()
            
                    save_image_name = os.path.join(save_image_dir, rawname+'_'+str(frame_id)+'.jpg')
                    cv2.imwrite(save_image_name, cur_frame)

if __name__ == '__main__':
    main()
