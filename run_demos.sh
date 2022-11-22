search_dir=./onnx_models
for entry in "$search_dir"/*
do
	echo "$entry"
	
	filename=$(basename -- "$entry")
	extension="${filename##*.}"
	filename="${filename%.*}"
	
	if [[ $entry == *"128x96"* ]];
       	then
		size="128x96"
	else
		size="256x192"
	fi
	if [[ $entry == *"simcc"* ]];
	then
		thr=0.025
	else
		thr=0.35
	fi
	echo "$size"
	echo "$filename"
	cfgname=${filename}
	cfgname="${cfgname%%${size}*}coco_${size}${cfgname##*${size}}"
	echo "$cfgname"

	python3 demo/top_down_video_demo_with_det_onnx.py demo/mmdetection_cfg/yolov7_coco.py onnx_models/yolov7_${size}.onnx configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/${cfgname}_sample32.py ${entry} --input-path onnx_inout_samples/inputs/ --output-path onnx_inout_samples/outputs/${filename} --resize-w ${size:0:3} --resize-h ${size:4} --bbox-thr 0.35 --kpt-thr ${thr} --save-np
done
