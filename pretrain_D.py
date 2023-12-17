import torch
import torch.optim as optim
from nets import Generator


def pretrain_discriminator(D, data_loader, epochs, batch_size, noise_dim, cuda):
    D.train()
    preG = Generator()
    if cuda:
        preG = preG.cuda()
    optimizerD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(preG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(data_loader):
            # load real data
            if cuda:
                real_imgs = real_imgs.cuda()

            D.zero_grad()
            real_decision = D(real_imgs)
            real_loss = -torch.mean(real_decision)

            # fake data
            noise = torch.randn(batch_size, noise_dim)
            if cuda:
                noise = noise.cuda()
            fake_data = preG(noise)
            fake_decision = D(fake_data)
            fake_loss = torch.mean(fake_decision)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD.step()

            preG.zero_grad()
            fake_data = preG(noise)
            decision = D(fake_data)
            g_loss = -torch.mean(decision)
            g_loss.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {real_loss.item()}")

    del preG
