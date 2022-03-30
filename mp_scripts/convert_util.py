import os, json

images_base_num = 887

def convert_files(in_path, out_path, in_file: str, out_file: str, seq_length: int):
    in_file_path = in_path + in_file

    with open(in_file_path, 'r') as f:
        data = json.load(f)
        all_annotation_image_ids = set()
        # data['id_man'] = 134 # <--- add `id` value.
        annotations = data['annotations']
        for ann in annotations:
            ann["image_id"] = ann["image_id"] - images_base_num
            all_annotation_image_ids.add(ann["image_id"])
            bbox = ann["bbox"]
            for i, x in enumerate(bbox):
                bbox[i] = int(bbox[i])

        images = data['images']
        for i, img in enumerate(images):
            img["id"] = img["id"] - images_base_num
            if img["id"] not in all_annotation_image_ids:
                images.pop(i)
                continue

            img["seq_length"] = seq_length
            img["first_frame_image_id"] = 0
            img["prev_image_id"] = img["prev_image_id"] - images_base_num
            img["next_image_id"] = img["next_image_id"] - images_base_num
            img["frame_id"] = img["frame_id"] - images_base_num

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
        "seq_length": 885
    }
    in_val = {
        "in_file": "cyc_val.json",
        "out_file": "val.json",
        "seq_length": 886
    }

    files_list = [in_train, in_val]

    for f in files_list:
        convert_files(in_path, out_path, **f)

