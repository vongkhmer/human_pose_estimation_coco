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

    humanpose_data_loader = torch.utils.data.DataLoader(human_pose_dataset, 
                                                        batch_size =64,
                                                        shuffle = True, 
                                                        num_workers =2, 
                                                        pin_memory = True)

    human_pose_model.to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, human_pose_model.parameters()), lr=1e-4)
    loss_hist = []
    NUM_EPOCH = 50
    start_epoch = 31 
    end_epoch = start_epoch + NUM_EPOCH + 1
    for epoch in range(start_epoch, end_epoch):
        total_loss = 0
        total_batch = 0
        for local_batch, local_labels, local_masks in tqdm(humanpose_data_loader):
            local_batch, local_labels, local_masks = local_batch.to(device), local_labels.to(device), local_masks.to(device)

            optimizer.zero_grad()
            outputs = human_pose_model(local_batch)
            outputs = outputs * local_masks
            loss = loss_function(outputs, local_labels)
            total_batch += 1
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} loss {total_loss / total_batch}")
        loss_hist.append(total_loss / total_batch)
        # if epoch % 10 == 0:
        #     torch.save(deconv.state_dict(), f"/content/drive/MyDrive/models/masked_deconv_model_e_{epoch }")



if __name__ == "__main__":
    # reset()
    train()