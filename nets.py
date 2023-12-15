import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_size=784, h1_dim=512, h2_dim=256):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, h1_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h1_dim, h2_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h2_dim, 1)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, image_size=784, h1_dim=128, h2_dim=256, h3_dim=512, h4_dim=1024, noise_size=100, image_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.image_shape = image_shape
        self.model = nn.Sequential(
            nn.Linear(noise_size, h1_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h2_dim, h3_dim),
            nn.BatchNorm1d(h3_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h3_dim, h4_dim),
            nn.BatchNorm1d(h4_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h4_dim, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.model(z)
        return z.view(-1, self.image_shape)

