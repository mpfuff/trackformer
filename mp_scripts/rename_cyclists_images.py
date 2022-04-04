
import os
import shutil
from mp_scripts.mp_util.MpFileUtil import MpFileUtil

fu = MpFileUtil()

in_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/cyclists_test/"
out_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/cyclists_test_renamed/"

fu.make_dir(out_path)


def rename_image_files(in_dir: str, out_dir: str):
    all_files = fu.list_all_files_in_dir(in_dir)
    all_files_sorted = sorted(all_files)


    for i, one_image in enumerate(all_files_sorted):
        in_file_path = in_dir + one_image
        num = f"{i:06d}"
        out_file_path = out_dir + num + ".jpeg"
        # os.rename(in_file_path, out_file_path)
        shutil.copy(in_file_path, out_file_path)


rename_image_files(in_path, out_path)