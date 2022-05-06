from mp_scripts.converter.fahrradar.convert_fahrradar_util import convert_fahrradar


input_path =  "/Users/matthias/projects/ml/vision/detection/tracking/data/fahrradar/coco/scene_31_test_preview/annotations/"
# output_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/scene32/annotations/"
output_path = input_path

in_train = {
    "in_file": "train_in.json",
    "out_file": "train.json",
}

files_list = [in_train, ]

convert_fahrradar(input_path, output_path, files_list)
