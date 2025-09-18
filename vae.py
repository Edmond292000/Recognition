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

# hyperparameter
latent_dim = 256


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


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # layer 1
        self.layer1 = BasicBlock1(1, 96, 5, 1)

        # layer 2
        self.layer2 = BasicBlock2(96, 128, 3, 1)

        # layer 3
        self.layer3 = BasicBlock1(128, 192, 3, 1)

        # layer 4
        self.layer4 = BasicBlock2(192, 256, 3, 1)

        # fully connected
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 11 * 11, 4096)
        self.fc_mu = nn.Linear(4096, latent_dim)
        self.fc_logvar = nn.Linear(4096, latent_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.flatten(x)
        h = F.relu(self.fc(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  # log(var^2)

        return mu, logvar

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps



