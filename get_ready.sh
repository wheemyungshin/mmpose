conda create --name openmmlab-cuda113 python=3.8 -y
conda activate openmmlab-cuda113
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
../miniconda3/envs/openmmlab-cuda113/bin/pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
../miniconda3/envs/openmmlab-cuda113/bin/pip install -r requirements.txt
../miniconda3/envs/openmmlab-cuda113/bin/pip install -v -e .
