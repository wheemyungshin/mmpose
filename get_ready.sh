conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
../miniconda3/envs/openmmlab/bin/pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
../miniconda3/envs/openmmlab/bin/pip install -r requirements.txt
../miniconda3/envs/openmmlab/bin/pip install -v -e .
