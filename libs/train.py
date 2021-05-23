from config import Config
from pose_data_loader import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from model import *
import torch
from tqdm import tqdm

def train():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print("using GPU :", use_cuda)

    processed_img_id, processed_keypoints = load_processed_img_id()

    human_pose_model = HumanPose()
    X = torch.rand(1, 3, *Config.image_shape)
    Y = human_pose_model(X)
    Config.heatmap_shape = list(Y.shape)[2:]


    # print(len(processed_keypoints))
    # print(processed_keypoints[0])

    print("Config")
    print(f"Num keypoints : {Config.num_keypoints}")
    print(f"Heatmap shape : {Config.heatmap_shape}")

    random_flip = RandomFlip(Config.sym_pairs)
    random_translate = RandomTranslate()
    random_rotate = RandomRotate()
    composed = transforms.Compose([random_flip, random_translate, random_rotate])

    Config.sigma = 2 
    Config.heatmap_radius = 3 * Config.sigma

    train_dataset = HumanPoseDataset(processed_img_id["train"], 
                                        processed_keypoints["train"], 
                                        Config.heatmap_radius, 
                                        Config.heatmap_shape, 
                                        resnet_preprocess, 
                                        Config.num_keypoints, 
                                        Config.sigma, composed)

    val_dataset = HumanPoseDataset(processed_img_id["val"], 
                                      processed_keypoints["val"], 
                                      Config.heatmap_radius, 
                                      Config.heatmap_shape, 
                                      resnet_preprocess, 
                                      Config.num_keypoints, 
                                      Config.sigma)
    it = iter(train_dataset)
    X, Y, V = next(it)
    print(X.shape)
    print(Y.shape)
    print(V.shape)

    X = invTrans(X)
    plt.imshow(X.permute(1,2,0))
    plt.savefig("test-img.png")
    plt.clf()

    X = X.permute(1, 2, 0).numpy()

    X = draw_heatmap(X, Y)

    plt.imshow(X)
    plt.savefig("test-key.png")
    plt.clf()

    hmp = np.sum(Y, axis = 0)
    plt.imshow(hmp)
    plt.savefig("test-hmp.png")
    plt.clf()


    X, Y, V = next(it)
    print(X.shape)
    print(Y.shape)
    print(V.shape)

    X = invTrans(X)
    plt.imshow(X.permute(1,2,0))
    plt.savefig("test-img2.png")
    plt.clf()

    X = X.permute(1, 2, 0).numpy()

    X = draw_heatmap(X, Y)

    plt.imshow(X)
    plt.savefig("test-key2.png")
    plt.clf()

    hmp = np.sum(Y, axis = 0)
    plt.imshow(hmp)
    plt.savefig("test-hmp2.png")
    plt.clf()


    it = iter(val_dataset)
    X, Y, V = next(it)
    print(X.shape)
    print(Y.shape)
    print(V.shape)

    X = invTrans(X)
    plt.imshow(X.permute(1,2,0))
    plt.savefig("val-img.png")
    plt.clf()

    X = X.permute(1, 2, 0).numpy()

    X = draw_heatmap(X, Y)

    plt.imshow(X)
    plt.savefig("val-key.png")
    plt.clf()

    hmp = np.sum(Y, axis = 0)
    plt.imshow(hmp)
    plt.savefig("val-hmp.png")
    plt.clf()


    X, Y, V = next(it)
    print(X.shape)
    print(Y.shape)
    print(V.shape)

    X = invTrans(X)
    plt.imshow(X.permute(1,2,0))
    plt.savefig("val-img2.png")
    plt.clf()

    X = X.permute(1, 2, 0).numpy()

    X = draw_heatmap(X, Y)

    plt.imshow(X)
    plt.savefig("val-key2.png")
    plt.clf()

    hmp = np.sum(Y, axis = 0)
    plt.imshow(hmp)
    plt.savefig("val-hmp2.png")
    plt.clf()

    human_pose_model.to(device)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, human_pose_model.parameters()), lr=1e-4)

    print("Training first 10 epochs with resnet weight frozen...")


if __name__ == "__main__":
    # reset()
    train()