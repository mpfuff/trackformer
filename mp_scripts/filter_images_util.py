import os, json
import shutil

images_base_num = 887


def filter_and_rename_images(in_json_path, in_images_path, out_images_path, in_json_file: str):
    in_json_path_file = in_json_path + in_json_file

    with open(in_json_path_file, 'r') as f:
        data = json.load(f)
        all_annotation_image_ids = set()
        annotations = data['annotations']

        for ann in annotations:
            # ann["image_id"] = ann["image_id"] - images_base_num
            all_annotation_image_ids.add(ann["image_id"])


        images = data['images']

        remaining_images = []
        i=0
        for img in images:
            old_img_id = img["id"]
            if old_img_id not in all_annotation_image_ids:
                continue

            remaining_images.append(img)
            old_file_name = img["file_name"]
            num = i + 1
            new_file_name = f"{num:06d}.jpg"
            in_file_path = in_images_path + old_file_name
            out_file_path = out_images_path + new_file_name
            shutil.copy(in_file_path, out_file_path)
            i += 1


