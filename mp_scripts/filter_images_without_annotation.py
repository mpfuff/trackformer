import os
import shutil

from mp_scripts.filter_images_util import filter_and_rename_images
from mp_scripts.mp_util.MpFileUtil import MpFileUtil

# from .mp_util.MpFileUtil import MpFileUtil

fu = MpFileUtil()

in_json_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/scene32/annotations/"
in_json_file = "scene32_track.json"
in_images_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/scene32/images/"
out_images_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/scene32/aws/train//"

fu.make_dir(out_images_path)

filter_and_rename_images(in_json_path=in_json_path, in_images_path=in_images_path, out_images_path=out_images_path,
                         in_json_file=in_json_file)
