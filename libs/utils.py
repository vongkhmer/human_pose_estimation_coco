import os
import pickle
import wget
from zipfile import ZipFile
import json
from config import * 
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage

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
coco_val_url = "http://images.cocodataset.org/zips/val2017.zip"

def load_processed_img_id():
    with open(img_id_pickle, "rb") as f:
        processed_img_id = pickle.load(f)

    with open(keypoints_pickle, "rb") as f:
        processed_keypoints = pickle.load(f)

    if len(processed_img_id) == 0:
        print("Training data is needed. Processing data for training..")
        processed_img_id, processed_keypoints = process_img() 
        print("done")
    else:
        print("finished loaded")
    Config.num_keypoints = len(processed_keypoints["train"][0]) // 3

    return processed_img_id, processed_keypoints

def process_img():
    processed_img_id = []
    processed_keypoints = []
    with open(os.path.join(data_dir, "download_flag"), "rb") as f:
        try:
            downloaded = pickle.load(f)
        except:
            downloaded = False

    if not downloaded:
        print("Downloading coco dataset")
        download_coco()

    img_id_data, keypoints_data, bbox_info_data, num_keypoints_data = read_annotation_json()

    print("Start processing images according to configuration..")

    processed_img_id = {"train": [], "val":[]}
    processed_keypoints = {"train": [], "val": []}
                        
    for data_type in ["train", "val"]:
        for ind in range(len(img_id_data[data_type])):
            img = Image.open(img_id_data[data_type][ind])
            w, h = img.size
            keyp = keypoints_data[data_type][ind]
            num_key = num_keypoints_data[data_type][ind]
            bx, by, bw, bh = bbox_info_data[data_type][ind]
            bsz, cx, cy = max(bw, bh) * Config.bbx_multiplier, bx + bw//2, by + bh//2

            if bsz < 224 :
                continue
            if num_key < 4:
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

            new_path =  os.path.join(processed_dir, f"{data_type}_" + '%012d.jpg' % (ind))
            print(new_path)
            roi.save(new_path)
            processed_keypoints[data_type].append(normalized_keypoint)
            processed_img_id[data_type].append(new_path)

    print(f"processed img data for training {len(processed_img_id['train'])}, for val {len(processed_img_id['val'])}")

    #save processed id list 
    with open(img_id_pickle, "wb") as f:
        pickle.dump(processed_img_id, f)

    with open(keypoints_pickle, "wb") as f: 
        pickle.dump(processed_keypoints, f)

    return processed_img_id, processed_keypoints

def download_coco():
    print("Downloading annotation file...")
    wget.download(coco_annotation_url, data_dir)

    print("\nExtracting annotations...")
    with ZipFile(os.path.join(data_dir, "annotations_trainval2017.zip")) as zf:
        zf.extractall(data_dir)
    print("Done.")

    print("Downloading training data.")
    wget.download(coco_dataset_url, data_dir)

    print("\nExtracting training data...")
    with ZipFile(os.path.join(data_dir, "train2017.zip")) as zf:
        zf.extractall(data_dir)
    print("Done.")

    print("Downloading validation data..")
    wget.download(coco_val_url, data_dir)

    print("\nExtracting validation data...")
    with ZipFile(os.path.join(data_dir, "val2017.zip")) as zf:
        zf.extractall(data_dir)
    print("Done.")

    downloaded = True
    with open(os.path.join(data_dir, "download_flag"), "wb") as f:
        pickle.dump(downloaded, f)


def read_annotation_json():
    img_id_data = {"train": [], "val":[]}
    keypoints_data = {"train" : [], "val": []}
    bbox_info_data = {"train": [], "val": []}
    num_keypoints_data = {"train" : [], "val" : []}

    for data_type in ["train", "val"]:
        json_file = os.path.join(annotaion_dir, f"person_keypoints_{data_type}2017.json")
        with open(json_file) as jsf:
            annotations = json.load(jsf)
        annotations = annotations["annotations"]
        for a in annotations:
            image_id, keyp, bbox, num = a["image_id"], a["keypoints"], a["bbox"], a["num_keypoints"]
            image_id = os.path.join(data_dir, os.path.join(f"{data_type}2017", '%012d.jpg' % (image_id)))
            # print(image_id)
            # print(keyp)
            # print(bbox)
            num = 0

            keyp = remove_ignored_keypoints(keyp) 

            for i in range(len(keyp) // 3):
                _, _, v = keyp[ 3 * i: 3 * i + 3]
                if v > 0.5:
                    num += 1

            img_id_data[data_type].append(image_id)
            keypoints_data[data_type].append(keyp)
            bbox_info_data[data_type].append(bbox)
            num_keypoints_data[data_type].append(num)

        Config.num_keypoints = len(keypoints_data["train"][0]) // 3
    return img_id_data, keypoints_data, bbox_info_data, num_keypoints_data


def reset():
    processed_img_id = []
    processed_keypoints = []
    downloaded = False 

    with open(img_id_pickle, "wb") as f:
        pickle.dump(processed_img_id, f)

    with open(keypoints_pickle, "wb") as f:
        pickle.dump(processed_keypoints, f)
    
    with open(os.path.join(data_dir, "download_flag"), "wb") as f:
        pickle.dump(downloaded, f)

def draw_bbox(im, x, y, w, h):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    draw.rectangle([x, y, x+w, y +h])
    return im

def draw_keypoints(im, keypoints, radius=2):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    sz = len(keypoints) // 3
    points = []
    for i in range(sz):
        x, y, v = keypoints[3 * i : 3 *(i+1)]
        if v < 0.5:
            continue
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(255, 0, 0), outline=(255, 0, 0))
    return im

def draw_normalized_keypoints(im, keypoints, radius = 2):
    im = im.copy()
    if not isinstance(im, Image.Image):
        im = im / np.max(im) * 255
        im = Image.fromarray(im.astype('uint8'))
    w, h = im.size
    draw = ImageDraw.Draw(im)
    sz = len(keypoints) // 3
    for i in range(sz):
        x, y, v = keypoints[3 * i : 3 *(i+1)]
        x *= w 
        y *= h
        if v  < 0.5:
            continue
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(255, 0, 0), outline=(255, 0, 0))
    return im

def heatmap_to_keypoints(heatmap, threshold = 0.8):
    keyp = []
    _, hw, hh = heatmap.shape
    for i in range(Config.num_keypoints):
        hmp = heatmap[i,:,:]
        if np.max(hmp) < threshold:
            keyp.extend([0,0, 0])
        else:
            hmp = hmp / np.max(hmp)
            hmp = np.where(hmp > 0.9, hmp, 0)
            cy, cx = ndimage.center_of_mass(hmp)
            keyp.extend([cx / hw, cy /hh, 2])
    return keyp

def draw_heatmap(im, heatmap, threshold = 0.8):
    im = np.array(im)
    norm_keyp = heatmap_to_keypoints(heatmap, threshold)
    return draw_normalized_keypoints(im, norm_keyp) 