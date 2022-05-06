from mp_scripts.converter.fahrradar.combine_fahrradar_util import combine_fahrradar

scene_31 = "scene_31_test_preview"
scene_32 = "scene_32_testdata_HD"
scene_41 = "scene_41_testdata_4K"
scene_42 = "scene_42_testdata_HD"
# scene = "scene_50_testdata_416p"

scenes = []
scenes.append(scene_31)
scenes.append(scene_32)
# scenes.append(scene_41)
# scenes.append(scene_42)

# input_path = f"/Users/matthias/projects/ml/vision/detection/tracking/data/fahrradar/coco/{scene}/annotations/"
output_path = "/Users/matthias/projects/ml/vision/detection/tracking/data/fahrradar/coco/combined/annotations/"
# output_path = input_path

in_train = {
    "in_file": "train.json",
    "out_file": "combined.json",
}

files_list = [in_train, ]

combine_fahrradar(scenes, output_path, files_list)
