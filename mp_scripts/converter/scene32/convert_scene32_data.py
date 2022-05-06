from mp_scripts.converter.scene32.convert_scene32_util import convert_scene32


input_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/scene32/ff1/coco/"
output_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/scene32/annotations/"
# output_path = input_path

in_train = {
    "in_file": "coco_track.json",
    "out_file": "scene32.json",
}

files_list = [in_train, ]

convert_scene32(input_path, output_path, files_list)
