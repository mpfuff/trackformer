import os, json

from mp_scripts.convert_util import convert_all

images_base_num = 887

input_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/cyclist/annotations/"
output_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/cyclist_mod/annotations/"
output_path = input_path

convert_all(input_path, output_path)
