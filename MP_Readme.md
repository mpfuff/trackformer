* get cuda version: nvcc --version

## PYthon 3.7 Environment
conda create --name mp_trackf_p37 --clone mxnet_p37 \
conda activate mp_trackf_p37 \
pip install torch==1.5.1 torchvision==0.6.1 \
pip3 install -r requirements.txt \

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
anaconda-project 0.9.1 requires ruamel-yaml, which is not installed.
shap 0.40.0 requires packaging>20.9, but you have packaging 20.4 which is incompatible.
bokeh 2.4.2 requires typing-extensions>=3.10.0, but you have typing-extensions 3.7.4.3 which is incompatible.
aiobotocore 1.3.0 requires botocore<1.20.50,>=1.20.49, but you have botocore 1.23.17 which is incompatible.
```
pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI' \
python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install \

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






python src/track.py with \
dataset_name=DEMO_1 \
data_root_dir=data/snakeboard \
output_dir=data/snakeboard \
write_images=pretty
