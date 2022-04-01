import os, json

images_base_num = 887


def convert_files(in_path, out_path, in_file: str, out_file: str):
    in_file_path = in_path + in_file

    with open(in_file_path, 'r') as f:
        data = json.load(f)
        all_annotation_image_ids = set()
        # data['id_man'] = 134 # <--- add `id` value.

        categories = data['categories']
        categories[0]['id'] = 1
        categories[0]['supercategory'] = "cyclist"
        data['categories'] = categories

        annotations = data['annotations']

        for ann in annotations:
            # ann["image_id"] = ann["image_id"] - images_base_num
            all_annotation_image_ids.add(ann["image_id"])

            ann['category_id'] = 1

            bbox = ann["bbox"]
            for i, x in enumerate(bbox):
                bbox[i] = int(bbox[i])

        images = data['images']


        remaining_images = []
        for img in images:
            old_img_id = img["id"]
            if old_img_id not in all_annotation_image_ids:
                continue

            remaining_images.append(img)

        data['images'] = remaining_images
        images = data['images']
        seq_length = len(remaining_images)
        new_image_id = 0

        for i, img in enumerate(images):
            old_img_id = img["id"]


            for ann in annotations:
                if ann["image_id"] == old_img_id:
                    ann["image_id"] = new_image_id

            img["id"] = new_image_id
            img["seq_length"] = seq_length
            img["first_frame_image_id"] = 0
            img["prev_image_id"] = new_image_id - 1
            img["next_image_id"] = new_image_id + 1
            img["frame_id"] = new_image_id
            new_image_id = new_image_id + 1

        data['images'] = images
        # f.seek(0)        # <--- should reset file position to the beginning.
        # json.dump(data, f, indent=4)
        # f.truncate()     # remove remaining part

    # write to new file
    out_file = os.path.join(os.path.dirname(out_path), out_file)
    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)


def convert_all(in_path, out_path):
    in_train = {
        "in_file": "cyc_train.json",
        "out_file": "train.json",
        # "seq_length": 885
    }
    in_val = {
        "in_file": "cyc_val.json",
        "out_file": "val.json",
        # "seq_length": 886
    }

    files_list = [in_train, in_val]

    for f in files_list:
        convert_files(in_path, out_path, **f)
