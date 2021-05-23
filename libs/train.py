from torchvision import models
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
    NUM_TEST = 10 
    for test in range(NUM_TEST):

        X, Y, V = next(it)
        print(X.shape)
        print(Y.shape)
        print(V.shape)
        X = invTrans(X)
        X = X.permute(1, 2, 0).numpy()
        X = draw_heatmap(X, Y)
        plt.imshow(X)
        plt.savefig(f"train-{test}.png")
        plt.clf()

    it = iter(val_dataset)

    for test in range(NUM_TEST):

        X, Y, V = next(it)
        print(X.shape)
        print(Y.shape)
        print(V.shape)
        X = invTrans(X)
        X = X.permute(1, 2, 0).numpy()
        X = draw_heatmap(X, Y)
        plt.imshow(X)
        plt.savefig(f"val-{test}.png")
        plt.clf()
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size =64,shuffle = True, num_workers =2, pin_memory = True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, num_workers=2, pin_memory=True)

    human_pose_model.to(device)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, human_pose_model.parameters()), lr=1e-4)

    print("Training first 10 epochs with resnet weight frozen...")

    loss_hist = {"train":[], "val" : []}
    NUM_EPOCH = 100
    start_epoch = 0
    end_epoch = start_epoch + NUM_EPOCH + 1
    best_val_loss = 1e5

    for epoch in range(start_epoch, end_epoch):
        total_loss = 0
        total_batch = 0
        if epoch == 10:
            print("Unfreeze layer 2 last layers...")
            human_pose_model.unfreeze(6)
            optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, human_pose_model.parameters()), lr=1e-4)

        if epoch == 60:
            print("Unfreeze another layer...")
            human_pose_model.unfreeze(5)
            optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, human_pose_model.parameters()), lr=5e-6)

        human_pose_model.train()
        print(f"Epoch {epoch} : training...")
        for local_batch, local_labels, local_masks in tqdm(train_data_loader):
            local_batch, local_labels, local_masks = local_batch.to(device), local_labels.to(device), local_masks.to(device)

            optimizer.zero_grad()
            outputs = human_pose_model(local_batch)
            outputs = outputs * local_masks
            loss = loss_function(outputs, local_labels)
            total_batch += 1
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss = total_loss / total_batch
        print(f"Training loss {train_loss}")
        loss_hist["train"].append(train_loss)

        print(f"Eval on validation set...")

        human_pose_model.eval()
        total_loss = 0
        total_batch = 0
        for local_batch, local_labels, local_masks in tqdm(val_data_loader):
            local_batch, local_labels, local_masks = local_batch.to(device), local_labels.to(device), local_masks.to(device)

            outputs = human_pose_model(local_batch)
            outputs = outputs * local_masks
            loss = loss_function(outputs, local_labels)
            total_batch += 1
            total_loss += float(loss)

        val_loss = total_loss / total_batch
        print(f"Val loss {val_loss}")
        loss_hist["val"].append(val_loss)

        if epoch % 10 == 0:
            torch.save(human_pose_model.state_dict(), os.path.join(models_dir, f"pose_model_with_val_e_{epoch }"))

        if val_loss < best_val_loss:
            torch.save(human_pose_model.state_dict(), os.path.join(models_dir, "pose_model_with_val_best_val_loss"))
            best_val_loss = val_loss
        print(f"Best val loss {best_val_loss}")


if __name__ == "__main__":
    # reset()
    train()