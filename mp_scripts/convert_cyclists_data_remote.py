import os, json

from convert_util import convert_all

images_base_num = 887

in_path = "/home/ubuntu/projects/pytorch/tracking/data/cyclist/annotations/"
out_path = "/home/ubuntu/projects/pytorch/tracking/data/cyclist_mod/annotations/"
out_path = in_path


convert_all(in_path, out_path)
