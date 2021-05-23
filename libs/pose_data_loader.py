from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms, datasets
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

class HumanPoseDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, keypoints_labels, heatmap_radius, heatmap_shape, preprocess, num_keypoints, sigma, data_transform = None):
        'Initialization'
        self.keypoints_labels = keypoints_labels.copy()
        self.list_IDs = list_IDs.copy()
        self.preprocess = preprocess
        self.heatmap_shape = heatmap_shape
        self.heatmap_radius = heatmap_radius
        self.sigma = sigma
        self.num_keypoints = num_keypoints
        self.grid = self.create_grid()
        # self.keypoints_vec, self.visible_vec = self.create_keypoints_vec()
        self.offset = np.exp(-0.5 * (self.heatmap_radius ** 2) / self.sigma ** 2)
        self.data_transform = data_transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = Image.open(ID).convert("RGB")
        label = self.keypoints_labels[index].copy()
        
        if self.data_transform:
          X, label = self.data_transform((X, label))

        X = self.preprocess(X)
        keyp, visibility = self.create_keypoints_vec(label)
        hmp = self.create_gaussian_heatmap(keyp)
        hmp = hmp * visibility 
        return X, hmp, visibility 
        
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
  
  def create_keypoints_vec(self, label):
    key = np.zeros([self.num_keypoints, 1, 1, 2], dtype=np.float32)
    vis = np.zeros([self.num_keypoints, 1,1], dtype=np.float32)
    for j in range(self.num_keypoints):
      x,y, v = label[3 * j : 3 * j + 3]
      x = np.floor(x * self.heatmap_shape[0] + 0.5)
      y = np.floor(y * self.heatmap_shape[1] + 0.5) 
      key[j, 0, 0, :] = [x , y ]
      vis[j, 0,0] = 1 if v > 0.5 else 0
    return key, vis

class RandomFlip(object):

    def __init__(self, sym_pair, flip_prob = 0.3):
      self.flip_prob = flip_prob
      self.sym_pair = sym_pair


    def __call__(self, sample):
        img, label = sample
        prob = float(torch.rand(1))
        if prob > self.flip_prob:
          return img, label

        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        new_label = label.copy()

        for i, j in self.sym_pair:
          new_label[3 * i : 3 * i + 3] = label[3 * j : 3 * j + 3]
          new_label[3 * j : 3 * j + 3] = label[3 * i : 3 * i + 3] 

        for i in range(len(new_label)//3):
          x, y, v = new_label[3 * i : 3 * i + 3]
          x = 1 - x
          new_label[3 * i] = x
        return img, new_label

class RandomTranslate(object):
  def __init__(self, trans_prob = 0.3, max_dist = 20):
    self.trans_prob = trans_prob
    self.max_dist = max_dist 

  def __call__(self, sample):
    img, label = sample
    prob = float(torch.rand(1))
    if prob > self.trans_prob:
      return sample
    w, h = img.size
    dx, dy = map(int, torch.randint(-self.max_dist, self.max_dist, (2,)))
    img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    new_label = label.copy()
    for i in range(len(new_label)//3):
          x, y, v = new_label[3 * i : 3 * i + 3]
          x -= dx / w
          y -= dy / h
          new_label[3 *i : 3 * i + 3] = x, y , v
    return img, new_label

class RandomRotate(object):
  def __init__(self, rot_prob = 0.3, max_angle = 15):
    self.rot_prob = rot_prob
    self.max_angle = max_angle

  def __call__(self, sample):
    img, label = sample
    prob = float(torch.rand(1))
    if prob > self.rot_prob:
      return sample
    angle = float(torch.rand(1)) * self.max_angle * 3.14 / 180
    a = np.cos(angle) 
    b = -np.sin(angle)
    c = 0
    d = np.sin(angle)
    e = np.cos(angle)
    f = 0
    new_label = label.copy()
    img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
    for i in range(len(new_label)//3):
          x, y, v = new_label[3 * i : 3 * i + 3]
          x = a * x - b * y + c
          y = -d *x + e * y + c
          new_label[3 *i : 3 * i + 3] = x, y , v
    return img, new_label