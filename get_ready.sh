conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
/home/wmshin/miniconda3/envs/openmmlab/bin/pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
/home/wmshin/miniconda3/envs/openmmlab/bin/pip install -r requirements.txt
/home/wmshin/miniconda3/envs/openmmlab/bin/pip install -v -e .
