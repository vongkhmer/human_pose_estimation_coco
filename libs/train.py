from config import Config
from pose_data_loader import *
from utils import *

resnet_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train():
    processed_img_id, processed_keypoints = load_processed_img_id()

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


if __name__ == "__main__":
    train()