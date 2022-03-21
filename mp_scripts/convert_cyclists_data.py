import os, json


in_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/cyclist/annotations/"
out_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/cyclist_mod/annotations/"
out_path = in_path

# in_file = "cyc_val.json"
# out_file = "val.json"
# seq_length = 886

in_file = "cyc_train.json"
out_file = "train.json"
seq_length = 885

in_file_path = in_path + in_file
out_file_path = out_path + in_file


with open(in_file_path, 'r') as f:
    data = json.load(f)
    # data['id_man'] = 134 # <--- add `id` value.
    images = data['images']
    for img in images:
        img["seq_length"] = seq_length
        img["first_frame_image_id"] = 887

    # f.seek(0)        # <--- should reset file position to the beginning.
    # json.dump(data, f, indent=4)
    # f.truncate()     # remove remaining part

# write to new file
out_file = os.path.join(os.path.dirname(out_path), out_file)
with open(out_file, 'w') as f:
    json.dump(data, f, indent=4)






