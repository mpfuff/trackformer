* get cuda version: nvcc --version

conda info --envs


## Python 3.7 Environment
conda create --name mp_trackf_p37 --clone mxnet_p37 \
conda activate mp_trackf_p37 \

7P PC:
- 12 Minuten pro Epoche Scene32, 3015 Images
- conda create --name mp_trackf_p37  python=3.7 \
- source activate mp_trackf_p37 \
- conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
- conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
- conda install cudatoolkit=10.0 -c pytorch
- conda install cudatoolkit=10.2 -c pytorch
- conda install cudatoolkit=11.3 -c pytorch
- conda install cudatoolkit -c pytorch
- conda uninstall cudatoolkit
- conda install pytorch torchvision cudatoolkit  -c pytorch
- pip install opencv-python==4.2.0.34
-
- pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

 \
pip3 install -r requirements.txt \

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
anaconda-project 0.9.1 requires ruamel-yaml, which is not installed.
shap 0.40.0 requires packaging>20.9, but you have packaging 20.4 which is incompatible.
bokeh 2.4.2 requires typing-extensions>=3.10.0, but you have typing-extensions 3.7.4.3 which is incompatible.
aiobotocore 1.3.0 requires botocore<1.20.50,>=1.20.49, but you have botocore 1.23.17 which is incompatible.
```
pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI' \
python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install

## SSH

/home/ubuntu/projects/pytorch/tracking/trackformer/mp_scripts


## Visdom
pip install visdom

conda info --envs  
conda activate mp_trackf_p37  
visdom -port 8090


### Training
python src/train.py with \
deformable \
tracking \
mot17 \
full_res \
resume=models/mot17_train_deformable_private/checkpoint.pth \
output_dir=models/custom_dataset_train_deformable \
mot_path=/home/ubuntu/projects/pytorch/tracking/data/cyclist \
train_split=train \
val_split=val \
no_vis=True \
epochs=2 \

python src/train.py with \
deformable \
tracking \
mot17 \
full_res \
resume=models/mot17_train_deformable_private/checkpoint.pth \
output_dir=models/custom_dataset_train_deformable \
mot_path=/Users/matthias/projects/ml/vision/detection/tracking/data/cyclist \
train_split=train \
val_split=val \
epochs=20 \




scp -r -i ~/.ssh/mampf.pem /Users/matthias/projects/ml/vision/detection/tracking/data/cyclist ubuntu@18.198.208.89:/home/ubuntu/projects/pytorch/tracking/data


/home/ubuntu/projects/pytorch/tracking/trackformer/mp_scripts/python





python src/track_cyclists.py with \
dataset_name=DEMO \
data_root_dir=/home/ubuntu/projects/pytorch/tracking/data/cyclist/val/test \
output_dir=data/cycl \
write_images=pretty



python src/track.py with \
dataset_name=DEMO_1 \
data_root_dir=data/snakeboard \
output_dir=data/snakeboard \
write_images=pretty


with
deformable
tracking
mot17
resume=models/mot17_train_deformable_private_v2/checkpoint.pth
output_dir=models/mot17_train_deformable_private_v3
epochs=2

- actual run config:  
with
deformable
tracking
mot17
resume=models/mot17_train_deformable_private/checkpoint.pth
output_dir=models/cyclists_dataset_train_deformable
mot_path=/home/ubuntu/projects/pytorch/tracking/data/cyclist
train_split=train
val_split=val
epochs=10
val_interval=2
tracking_eval=False

## Monitoring

tcp://6.tcp.ngrok.io:11270 -> localhost:8090

- GPU  
-- cat /usr/local/cuda/version.txt  
-- nvcc --version  
-- https://unix.stackexchange.com/questions/38560/gpu-usage-monitoring-cuda  
-- htop  
-- nvidia-smi  
-- nvidia-smi -L  
-- watch -d -n 0.5 nvidia-smi
