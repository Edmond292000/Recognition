import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt


# Encoder
# ---- Conv -> Relu -> Pool -> BN ----
class BasicBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicBlock1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)
        out = self.pool(out)
        out = self.bn(out)
        return out


# ---- Conv -> Relu -> BN ----
class BasicBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicBlock2, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)
        out = self.bn(out)
        return out
