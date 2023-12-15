import torch
import matplotlib.pyplot as plt
import numpy as np

# gradient penalty
def grad_penalty(discriminator, real_images, fake_images, cuda):
    epsilon = torch.rand(size=(real_images.shape[0], 1, 1, 1))
    if cuda:
        epsilon = epsilon.cuda()
        fake_images = fake_images.cuda()
    interpolates = epsilon * real_images + (1 - epsilon) * fake_images
    interpolates.requires_grad_()
    d_interpolates = discriminator(interpolates)
    fake = torch.ones_like(d_interpolates)
    if cuda:
        fake = fake.cuda()
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=fake,
                                    create_graph=True, retain_graph=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#pruning


#visualization


def plot_sampled_images(real_images, fake_images, num_samples=10):
    # sample from batch
    sampled_indices = np.random.choice(range(real_images.size(0)), num_samples, replace=False)
    real_samples = real_images[sampled_indices].detach().cpu().view(num_samples, 28, 28)
    fake_samples = fake_images[sampled_indices].detach().cpu().view(num_samples, 28, 28)

    fig, axes = plt.subplots(4, num_samples // 2, figsize=((num_samples // 2) * 2, 8))

    for i in range(num_samples // 2):
        # real
        axes[0, i].imshow(real_samples[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(real_samples[num_samples // 2 + i], cmap='gray')
        axes[1, i].axis('off')

        # fake
        axes[2, i].imshow(fake_samples[i], cmap='gray')
        axes[2, i].axis('off')
        axes[3, i].imshow(fake_samples[num_samples // 2 + i], cmap='gray')
        axes[3, i].axis('off')

    plt.show()


