import os, json

from mp_scripts.convert_scene32_util import convert_scene32


input_path = "/home/ubuntu/projects/pytorch/tracking/data/scene32/annotations/"
output_path = "/home/ubuntu/projects/pytorch/tracking/data/scene32_mod/annotations/"
output_path = input_path

in_train = {
    "in_file": "coco_track.json",
    "out_file": "scene32_track.json",
}

files_list = [in_train, ]

convert_scene32(input_path, output_path, files_list)
