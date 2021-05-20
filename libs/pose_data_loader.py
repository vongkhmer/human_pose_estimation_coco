from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms, datasets
import numpy as np

class HumanPoseDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, keypoints_labels, heatmap_radius, heatmap_shape, preprocess, num_keypoints, sigma):
            'Initialization'
            self.keypoints_labels = keypoints_labels
            self.list_IDs = list_IDs
            self.preprocess = preprocess
            self.heatmap_shape = heatmap_shape
            self.heatmap_radius = heatmap_radius
            self.sigma = sigma
            self.num_keypoints = num_keypoints
            self.grid = self.create_grid()
            self.keypoints_vec, self.visible_vec = self.create_keypoints_vec()
            self.offset = np.exp(-0.5 * (self.heatmap_radius ** 2) / self.sigma ** 2)

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]

            # Load data and get label
            X = Image.open(ID).convert("RGB")
            # Z = X
            X = self.preprocess(X)
            # y = self.keypoints_labels[index]
            hmp = self.create_gaussian_heatmap(self.keypoints_vec[index])
            hmp = 255.0 * hmp * self.visible_vec[index]
            return X, hmp, self.visible_vec[index]
            
    def create_grid(self):
        ret = np.zeros([self.num_keypoints,] + self.heatmap_shape + [2, ], dtype=np.float32)
        for i in range(self.heatmap_shape[0]):
            for j in range(self.heatmap_shape[1]):
                ret[:, i, j, :] = [j, i]
        return ret

    def create_gaussian_heatmap(self, centers):
        # center = center.reshape((1, 1, 2))
        r_square = np.sum((self.grid - centers) ** 2, axis=-1)
        hmp = np.exp( - 0.5 * r_square / self.sigma ** 2 )
        hmp = np.where(hmp > self.offset, hmp, 0)
        return hmp
    
    def create_keypoints_vec(self):
        keypoints = []
        visibility = []
        data_len = len(self.keypoints_labels)
        for i in range(data_len):
            label = self.keypoints_labels[i]
            key = np.zeros([self.num_keypoints, 1, 1, 2], dtype=np.float32)
            vis = np.zeros([self.num_keypoints, 1,1], dtype=np.float32)
            for j in range(self.num_keypoints):
                x,y, v = label[3 * j : 3 * j + 3]
                key[j, 0, 0, :] = [x * self.heatmap_shape[0], y * self.heatmap_shape[1]]
                vis[j, 0,0] = 1 if v > 0 else 0
            keypoints.append(key)
            visibility.append(vis)
        return keypoints, visibility
