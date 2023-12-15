import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from nets import Discriminator, Generator
import utils
from opacus import PrivacyEngine
import argparse
import config

if __name__ == "__main__":

    # choose model and hyperparams by argv param
    parser = argparse.ArgumentParser()
    parser.add_argument('config_model', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.config_model == 1:  # wgan-gp
        cfg = config.config_wgan_gp
    elif args.config_model == 2:  # dp-wgan-gp
        cfg = config.config_dp_wgan_gp
    elif args.config_model == 3:  # gs-wgan
        cfg = config.config_gs_wgan

    # Configure GPU or CPU settings
    cuda = True if torch.cuda.is_available() else False

    # Set hyperparameters
    total_epochs = cfg['total_epochs']
    batch_size = 64
    lr_D = cfg['lr_D']
    lr_G = cfg['lr_G']
    num_workers = cfg['num_workers']  # 0 for opacus
    noise_dim = 100
    image_size = 784
    image_width = 28
    channel = 1
    a = 10  # gradient penalty
    clip_value = 0.01
    dataset_dir = "./dataset/MNIST"
    gen_images_dir = cfg['gen_images_dir']
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(gen_images_dir, exist_ok=True)
    image_shape = (channel, image_width, image_width)

    # Get data
    transform = transforms.Compose(
            [transforms.Resize(image_width),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        )
    dataset = datasets.MNIST(root=dataset_dir,
                             train=True,
                             transform=transform,
                             download=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)

    # model
    D = Discriminator()
    G = Generator()
    if cuda:
        D = D.cuda()
        G = G.cuda()

    # optimizer
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G)

    # differential privacy
    if args.config_model == 2:  # dp-wgan-gp
        privacy_engine = PrivacyEngine(accountant='rdp')
        model, optimizer_D, dataloader = privacy_engine.make_private(
            module=D,
            optimizer=optimizer_D,
            data_loader=dataloader,
            noise_multiplier=0.2,
            max_grad_norm=5.0,
            grad_sample_mode="functorch"  # functorch based per-sample gradient computation (no hooks, no warnings)
        )

    # train
    for epoch in range(total_epochs):
        pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{total_epochs}', postfix=dict,
                    mininterval=0.3)
        LD = 0
        LG = 0
        for i, (real_imgs, _) in enumerate(dataloader):
            bs = real_imgs.shape[0]
            if cuda:
                real_imgs = real_imgs.cuda()

            # discriminator
            optimizer_D.zero_grad()
            z = torch.randn((bs, noise_dim))
            if cuda:
                z = z.cuda()
            fake_imgs = G(z).detach()
            if args.config_model == 2:
                # gradient penalty is included in opacus
                loss_D = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs))
            else:
                gp = utils.grad_penalty(D, real_imgs, fake_imgs, cuda)
                # gradient penalty
                loss_D = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs)) + a * gp

            loss_D.backward()
            optimizer_D.step()
            LD += loss_D.item()

            # generator
            optimizer_G.zero_grad()
            gen_imgs = G(z)
            loss_G = -torch.mean(D(gen_imgs))
            loss_G.backward()
            optimizer_G.step()
            LG += loss_G.item()

            pbar.set_postfix(**{'D_loss': loss_D.item(), 'G_loss': loss_G.item()})
            pbar.update(1)

        pbar.close()
        print("total_D_loss:%.4f,total_G_loss:%.4f" % (
            LD / len(dataloader), LG / len(dataloader)))
        if args.config_model == 2:
            epsilon = privacy_engine.accountant.get_epsilon(delta=cfg['delta'])
            print(f"ε = {epsilon}, δ = {cfg['delta']}")
        # save images 5*5
        save_image(gen_imgs.data[:25], "%s/ep%d.png" % (gen_images_dir, (epoch + 1)), nrow=5,
                   normalize=True)