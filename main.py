import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from nets import Generator, Discriminator
from opacus import PrivacyEngine

# MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# init
generator = Generator()
discriminator = Discriminator()

# optimizer
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# hyperparams
num_epochs = 100
clip_value = 0.01  # clipping gradients

# train
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)

        # noise
        z = torch.randn(batch_size, 100)

        # generator
        fake_images = generator(z)

        # discriminator
        real_images = real_images.view(batch_size, -1)
        real_validity = discriminator(real_images)
        fake_validity = discriminator(fake_images.detach())

        # using Wasserstein distance as value func
        wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)

        # loss
        discriminator_loss = -wasserstein_distance
        generator_loss = wasserstein_distance

        # update
        optimizer_D.zero_grad()
        discriminator_loss.backward()
        optimizer_D.step()

        # grad clipping for Lipschitz property
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # update
        optimizer_G.zero_grad()
        generator_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(train_loader)}] "
                  f"Generator Loss: {generator_loss.item():.4f} "
                  f"Discriminator Loss: {discriminator_loss.item():.4f}")

# save generator
torch.save(generator.state_dict(), 'wgan_generator.pth')