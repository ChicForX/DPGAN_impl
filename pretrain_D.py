import torch
import torch.optim as optim
from nets import Generator
def pretrain_discriminator(D, data_loader, epochs, batch_size, noise_dim, cuda):
    D.train()
    PreG = Generator()
    if cuda:
        PreG = PreG.cuda()
    optimizerD = optim.Adam(D.parameters(), lr=1e-4)
    optimizerG = optim.Adam(PreG.parameters(), lr=1e-4)

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
            fake_data = PreG(noise)
            fake_decision = D(fake_data)
            fake_loss = torch.mean(fake_decision)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD.step()

            PreG.zero_grad()
            fake_data = PreG(noise)
            decision = D(fake_data)
            g_loss = -torch.mean(decision)
            g_loss.backward()
            optimizerG.step()

            if i % 100 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {real_loss.item()}")
