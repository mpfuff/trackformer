


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
