from config import Config
from pose_data_loader import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np

resnet_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def train():
    processed_img_id, processed_keypoints = load_processed_img_id()

    print(len(processed_keypoints))
    print(processed_keypoints[0])

    human_pose_dataset = HumanPoseDataset(processed_img_id, 
                                      processed_keypoints, 
                                      Config.heatmap_radius, 
                                      Config.heatmap_shape, 
                                      resnet_preprocess, 
                                      Config.num_keypoints, 
                                      Config.sigma)
    it = iter(human_pose_dataset)
    X, Y, V = next(it)
    print(X.shape)
    print(Y.shape)
    print(V.shape)

    X = invTrans(X)
    plt.imshow(X.permute(1,2,0))
    plt.savefig("test-img.png")

    X = X.permute(1, 2, 0).numpy()

    X = draw_heatmap(X, Y)

    plt.imshow(X)
    plt.savefig("test-key.png")

    hmp = np.sum(Y, axis = 0)
    plt.imshow(hmp)
    plt.savefig("test-hmp.png")


    X, Y, V = next(it)
    print(X.shape)
    print(Y.shape)
    print(V.shape)

    X = invTrans(X)
    plt.imshow(X.permute(1,2,0))
    plt.savefig("test-img2.png")

    X = X.permute(1, 2, 0).numpy()

    X = draw_heatmap(X, Y)

    plt.imshow(X)
    plt.savefig("test-key2.png")

    hmp = np.sum(Y, axis = 0)
    plt.imshow(hmp)
    plt.savefig("test-hmp2.png")



if __name__ == "__main__":
    # reset()
    train()