from config import * 
from PIL import Image, ImageDraw
import numpy as np
from utils import * 
from model import * 
from pose_data_loader import *
import sys
import torch

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
    im = resnet_preprocess(im)
    return im.unsqueeze(0), original_im, padx, pady

def predict(model, fn, device):
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
    model_name = os.path.join(models_dir, sys.argv[1])
    image_name = os.path.join(test_dir, sys.argv[2])
    output_image_name = os.path.join(test_dir, sys.argv[3])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Initializing model {model_name}")
    human_pose_model = HumanPose()
    human_pose_model.load_state_dict(torch.load( model_name, map_location=device))

    keyp, result_img = predict(human_pose_model, image_name, device)

    print("Prediction result : ", keyp)
    print(f"Check outputs image @ test/{output_image_name}")