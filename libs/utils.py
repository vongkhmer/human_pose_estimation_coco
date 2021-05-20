import os
import pickle
import wget
from zipfile import ZipFile
import json
from config import * 
from PIL import Image, ImageDraw
import numpy as np

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
        processed_img_id, processed_keypoints = process_img() 
        print("done")
    else:
        print("finished loaded")

    return processed_img_id, processed_keypoints

def process_img():
    processed_img_id = []
    processed_keypoints = []
    with open(os.path.join(data_dir, "download_flag"), "rb") as f:
        downloaded = pickle.load(f)

    if not downloaded:
        print("Downloading coco dataset")
        #download_coco()

    original_image_id, original_keypoints, original_bbox, num_keypoints = read_annotation_json()
  
    Config.num_keypoints = len(original_keypoints[0]) // 3

    print(original_image_id[:10])
    print(Config.num_keypoints)

    print("Start processing images according to configuration..")

    for i in range(len(original_image_id)):
        if i > 1000:
            break
        ind = i
        img = Image.open(original_image_id[ind])
        w, h = img.size
        keyp = original_keypoints[ind]
        num_key = num_keypoints[ind]

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
    print("total processed : ", len(processed_keypoints))

    return processed_img_id, processed_keypoints

def download_coco():
    print("Downloading annotation file...")
    wget.download(coco_annotation_url, data_dir)

    print("Extracting annotations...")
    with ZipFile(os.path.join(data_dir, "annotations_trainval2017.zip")) as zf:
        zf.extractall(data_dir)
    print("\nDone.")

    print("Downloading training data.")
    wget.download(coco_dataset_url, data_dir)

    print("Extracting training data...")
    with ZipFile(os.path.join(data_dir, "train2017.zip")) as zf:
        zf.extractall(data_dir)
    print("\nDone.")
    
    download = True
    with open(os.path.join(data_dir, "download_flag"), "wb") as f:
        pickle.dump(downloaded, f)


def read_annotation_json():
    print("Reading annotation json..")
    with open(os.path.join(annotaion_dir, "person_keypoints_train2017.json"), "r") as f:
        annotations = json.load(f)

    annotations = annotations["annotations"]
    original_image_id = []
    original_keypoints = []
    original_bbox = []
    num_keypoints = []


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
        num_keypoints.append(num)
    
    print("Done.")


    return original_image_id, original_keypoints, original_bbox, num_keypoints


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
        if np.max(hmp) < threshold * 255:
            keyp.extend([0,0, 0])
        else:
            hmp = hmp / np.max(hmp) * 255
            hmp = np.where(hmp > 0.9 * 255, hmp, 0)
            cy, cx = ndimage.center_of_mass(hmp)
            keyp.extend([cx / hw, cy /hh, 2])
    return keyp

def draw_heatmap(im, heatmap, threshold = 0.8):
    # im = im.resize(Config.heatmap_shape)
    im = np.array(im)
    norm_keyp = heatmap_to_keypoints(heatmap, threshold)
    # for j in range(Config.num_keypoints):
    #   hmp = heatmap[:,:,j]
    #   hmp = np.where(hmp > 100, hmp, 0)
    #   mx = max(255, np.max(hmp))
    #   mask = hmp / mx * 255
    #   mask = Image.fromarray(mask.astype('uint8'))
    #   mask = mask.resize(im.shape[:-1])
    #   mask = np.array(mask)
    #   im[:,:,0] = np.where(mask[:,:,0] > 100, mask[:,:,0], im[:,:,0])
    #   im[:,:,1] = np.where(mask[:,:,0] > 100, 0, im[:,:,1])
    #   im[:,:,2] = np.where(mask[:,:,0] > 100, 0, im[:,:,2])
    return draw_normalized_keypoints(im, norm_keyp) 