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

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 256 * 11 * 11)
        self.bn2 = nn.BatchNorm1d(256 * 11 * 11)

        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.unpool1 = nn.Upsample(scale_factor=2, mode="nearest")

        self.deconv2 = nn.ConvTranspose2d(256, 192, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(192)

        self.unpool2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.deconv3 = nn.ConvTranspose2d(192, 128, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 1, kernel_size=5, stride=1)

    def forward(self, z):
        z = F.relu(self.bn1(self.fc1(z)))
        z = F.relu(self.bn2(self.fc2(z)))

        z = z.view(-1, 256, 11, 11)

        z = F.relu(self.bn3(self.deconv1(z)))
        z = self.unpool1(z)

        z = F.relu(self.bn4(self.deconv2(z)))
        z = self.unpool2(z)

        z = F.relu(self.bn5(self.deconv3(z)))
        z = torch.sigmoid(self.deconv4(z))

        return z


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    beta: weight for KL divergence (beta-VAE)
    """
    # Reconstruction loss (Binary Cross Entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD, BCE, KLD


