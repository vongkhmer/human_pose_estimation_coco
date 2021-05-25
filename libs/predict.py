from config import * 
from PIL import Image, ImageDraw
import numpy as np
from utils import * 
from model import * 
import sys

def load_img(fn):
    im = Image.open(fn)
    original_im = im
    w, h = im.size
    mx = max(w, h)
    w = w * 256 // mx
    h = h * 256 // mx
    padx = (256 - w) // 2
    pady = (256 - h) // 2
    cx = w //2
    cy = h // 2
    new_w, new_h = 224, 224
    im = im.resize((w, h))
    # original_im = im.crop((cx -128, cy -128, cx + 128, cy + 128))
    im = im.crop((cx - new_w// 2, cy - new_h // 2, cx + new_w // 2, cy +  new_h // 2))
    print(im.size)
    display(im)
    im = resnet_preprocess(im)
    return im.unsqueeze(0), original_im, padx, pady

def predict(model, fn):
    model.eval()
    X, original_im, padx, pady = load_img(fn)
    outputs = human_pose_model(X.to(device))
    X = X.squeeze()
    outputs = outputs.to('cpu')
    outputs = outputs.squeeze()
    keyp = heatmap_to_keypoints(outputs.detach().numpy(), threshold = 0.4)

    for i in range(Config.num_keypoints):
        kx, ky, _ = keyp[3 * i : 3 * i + 3]
        kx = kx * 224 + 16
        ky = ky * 224 + 16
        kx = (kx - padx) / (256 - 2 * padx) 
        ky = (ky - pady) / (256 -  2 * pady)
        keyp[3*i] = kx
        keyp[3 *i + 1] = ky

    result_img = draw_normalized_keypoints(original_im, keyp)
    return keyp, result_img

if __name__ == "__main__":
    model_name = sys.argv[1]
    image_name = sys.argv[2]
    output_image_name = sys.argv[3]

    print(f"Initializing model {model_name}")

    keyp, result_img = predict(model_name, image_name)

    print("Prediction result : ", keyp)
    print(f"Check outputs image @ test/{output_image_name}")