from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy

'''
def get_resnet18():
  model = nn.Sequential(
          nn.Conv2d(n_channels, 64, kernel_size=7, stride=1, padding='same', bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          ResidualStack(64, 64, 1, 6),
          ResidualStack(64, 128, 2, 8),
          ResidualStack(128, 256, 2, 12),
          ResidualStack(256, 512, 1, 6),
          nn.AdaptiveAvgPool2d(1),
          Lambda(lambda x: x.squeeze()),
          nn.Linear(512, n_classes)
      )
  return model
'''
class Resnet18MLMI(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding='same', bias=False)
        self.rs1 = ResidualStack(64, 64, 1, 6)
        self.rs2 = ResidualStack(64, 128, 2, 8)
        self.rs3 = ResidualStack(128, 256, 2, 12)
        self.rs4 = ResidualStack(256, 512, 1, 6)
        
        self.l2norm = Normalize(2)
        
        self.adapt = nn.AdaptiveAvgPool2d(1)
        self.lambda = Lambda(lambda x: x.squeeze())
        self.fc = nn.Linear(512, n_classes)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.proj = cfg.proj
        if self.proj: #use an extra projection layer
            self.fc1 = nn.Linear(512, 2048)
            # self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.fc2 = nn.Linear(2048, 64)
        else:
            self.fc4 = nn.Linear(512, 64)
        
        def forward(self, x):
            x = self.relu(self.bn(self.conv1(x)))
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.adapt(x)
            feat = self.lambda(x)
            out = self.fc(x)
            if self.proj:
                feat = F.relu(self.fc1(feat))
                feat = self.fc2(feat)
                feat = self.l2norm(feat)
                return out, feat
            return out

class ResidualStack(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        first = [ResidualBlock(in_channels, out_channels, stride)]
        rest = [ResidualBlock(out_channels, out_channels) for i in range(num_blocks - 1)]
        self.modules_list = nn.Sequential(*(first + rest))
        
    def forward(self, input):
        return self.modules_list(input)
  
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
        
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()      
        self.info = 'in ' +  str(in_channels) + ', out ' + str(out_channels) + ', stride ' + str(stride)
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, 0, out_channels - in_channels), mode="constant", value=0))
        else:
            self.skip = nn.Sequential()
 
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        bn1 = nn.BatchNorm2d(out_channels)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        bn2 = nn.BatchNorm2d(out_channels)
        self.l1 = nn.Sequential(conv1, bn1)
        self.l2 = nn.Sequential(conv2, bn2)

    def forward(self, input):
        skip = self.skip(input)
        x = self.l1(input)
        x = F.relu(x)
        x = self.l2(x)
        return F.relu(x + skip)
        
class ResidualStack(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        first = [ResidualBlock(in_channels, out_channels, stride)]
        rest = [ResidualBlock(out_channels, out_channels) for i in range(num_blocks - 1)]
        self.modules_list = nn.Sequential(*(first + rest))
        
    def forward(self, input):
        return self.modules_list(input)
    