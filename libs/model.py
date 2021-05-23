
import torchvision.models as models
import torch.nn as nn

class Deconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = (4,4)
        self.kernel_num = 256
        self.padding = 1
        self.stride = 2
        self.num_joints = Config.num_keypoints
        self.conv1 = nn.ConvTranspose2d(in_channels=2048, out_channels=self.kernel_num, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.kernel_num, out_channels=self.kernel_num, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.kernel_num, out_channels=self.kernel_num, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.kernel_num)
        self.bn2 = nn.BatchNorm2d(self.kernel_num)
        self.bn3 = nn.BatchNorm2d(self.kernel_num)
        self.final_deconv = nn.Conv2d(in_channels=self.kernel_num, out_channels=self.num_joints, kernel_size=(1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        out = self.final_deconv(x)
        return out 


class HumanPose(nn.Module):
  def __init__(self):
    super().__init__()
    resnet = models.resnet101(pretrained=True)
    resnet_bottleneck = nn.Sequential(*(list(resnet.children())[:-2]))
    self.bottleneck = resnet_bottleneck
    self.deconv = Deconv()

    for child in self.bottleneck.children():
        for param in child.parameters():
            param.requires_grad = False

  def forward(self, x):
    features = self.bottleneck(x)
    heatmap = self.deconv(features)
    return heatmap

  def unfreeze(self, start_from=1000):
    layer_num = 0
    for child in self.bottleneck.children():
      layer_num += 1
      if layer_num >  start_from:
        for param in child.parameters():
          param.requires_grad = True