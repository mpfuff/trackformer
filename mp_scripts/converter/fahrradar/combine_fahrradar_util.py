import os, json

images_base_num = 887


def combine_files(scenes, out_path, in_file: str, out_file: str):

    data_out = {}
    images_base = 0
    annotations_base = 0
    i = 1
    for scene in scenes:
        input_path = f"/Users/matthias/projects/ml/vision/detection/tracking/data/fahrradar/coco/{scene}/annotations/"
        in_file_path = input_path + in_file
        with open(in_file_path, 'r') as f:
            data = json.load(f)
            # all_annotation_image_ids = set()
            # data['id_man'] = 134 # <--- add `id` value.

            categories = data['categories']
            annotations = data['annotations']
            images = data['images']

            if i == 1:
                data_out['categories'] = categories
                data_out['annotations'] = annotations
                data_out['images'] = images
            else:
                for annotation in annotations:
                    annotation['id'] = annotation['id'] + annotations_base
                    annotation['image_id'] = annotation['image_id'] + images_base
                    data_out['annotations'].append(annotation)
                for image in images:
                    image['first_frame_image_id'] = images_base
                    image['id'] = image['id'] + images_base
                    image['frame_id'] = image['frame_id'] + images_base
                    data_out['images'].append(image)

            seq_length = images[0]['seq_length']
            images_base = images_base + seq_length
            annotations_length = annotations[-1]['id']
            annotations_base = annotations_base + annotations_length
        i += 1



    # write to new file
    out_file = os.path.join(os.path.dirname(out_path), out_file)
    with open(out_file, 'w') as f:
        json.dump(data_out, f, indent=4)


def combine_fahrradar(scenes, out_path, files_list):
    for f in files_list:
        combine_files(scenes, out_path, **f)
