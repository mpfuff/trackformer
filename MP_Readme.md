


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


