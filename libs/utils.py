import os
import pickle
import wget
from zipfile import ZipFile
import json
from config import * 
from PIL import Image, ImageDraw

cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
data_dir = os.path.join(parent_dir, "dataset")
img_id_pickle = os.path.join(data_dir, "processed_img_id")
keypoints_pickle = os.path.join(data_dir, "processed_keypoints")
img_dir = os.path.join(data_dir, "train2017")
processed_dir = os.path.join(data_dir, "processed_img")
annotaion_dir = os.path.join(data_dir, "annotations")
coco_dataset_url = "http://images.cocodataset.org/zips/train2017.zip"
coco_annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def load_processed_img_id():
    reset()
    with open(img_id_pickle, "rb") as f:
        processed_img_id = pickle.load(f)

    with open(keypoints_pickle, "rb") as f:
        processed_keypoints = pickle.load(f)

    if len(processed_img_id) == 0:
        print("Download need")
        process_img() 
        print("done")
    else:
        print("finished loaded")

    return processed_img_id, processed_keypoints

def process_img():
    processed_img_id = []
    processed_keypoints = []

    if True:
        print("Downloading coco dataset")
        download_coco()

    original_image_id, original_keypoints, original_bbox = read_annotation_json()
  
    Config.num_keypoints = len(original_keypoints[0]) // 3

    print(original_image_id[:10])
    print(Config.num_keypoints)

    for i in range(len(train_image_id)):
        if i > 1000:
            break
        ind = i
        img = Image.open(original_image_id[ind])
        w, h = img.size
        keyp = original_keypoints[ind]
        bx, by, bw, bh = original_bbox[ind]
        bsz, cx, cy = max(bw, bh) * Config.bbx_multiplier, bx + bw//2, by + bh//2

        if bsz < 224 :
            continue
        if num_key < 6:
            continue

        lx, rx = cx - bsz // 2, cx + bsz//2
        ly, ry = cy - bsz //2 , cy + bsz //2

        roi = img.crop((lx, ly, rx, ry))

        points = []
        for i in range(Config.num_keypoints):
            x, y, v = keyp[3 * i : 3 *(i+1)]
            if v < 1:
                points.extend((x,y,v))
            else:
                x -= cx - bsz // 2
                y -= cy - bsz// 2
                points.extend((x,y, v))
        roi = roi.resize(Config.image_shape)
        normalized_keypoint = []
        for i in range(Config.num_keypoints):
            x, y, v = points[3 * i : 3 * (i + 1)]
            x /= bsz
            y /= bsz
            normalized_keypoint.extend((x,y, v))

        new_path =  os.path.join(processed_dir, '%012d.jpg' % (ind))
        print(new_path)
        roi.save(new_path)
        processed_keypoints.append(normalized_keypoint)
        processed_img_id.append(new_path)

    with open(img_id_pickle, "wb") as f:
        pickle.dump(processed_img_id, f)

    with open(keypoints_pickle, "wb") as f:
        pickle.dump(processed_keypoints, f)

def download_coco():
    wget.download(coco_annotation_url, data_dir)
    with ZipFile(os.path.join(data_dir, "annotations_trainval2017.zip")) as zf:
        zf.extractall(data_dir)
    wget.download(coco_dataset_url, data_dir)

    with ZipFile(os.path.join(data_dir, "train2017.zip")) as zf:
        zf.extractall(data_dir)


def read_annotation_json():
    with open(os.path.join(annotaion_dir, "person_keypoints_train2017.json"), "r") as f:
        annotations = json.load(f)

    annotations = annotations["annotations"]
    original_image_id = []
    original_keypoints = []
    original_bbox = []

    for a in annotations:
        image_id, keyp, bbox, num = a["image_id"], a["keypoints"], a["bbox"], a["num_keypoints"]
        image_id = os.path.join(img_dir, '%012d.jpg' % (image_id))
        # print(image_id)
        # print(keyp)
        # print(bbox)
        keyp = remove_ignored_keypoints(keyp) 
        original_image_id.append(image_id)
        original_keypoints.append(keyp)
        original_bbox.append(bbox)

    return original_image_id, original_keypoints, original_bbox


def reset():
    processed_img_id = []
    processed_keypoints = []

    with open(img_id_pickle, "wb") as f:
        pickle.dump(processed_img_id, f)

    with open(keypoints_pickle, "wb") as f:
        pickle.dump(processed_keypoints, f)
