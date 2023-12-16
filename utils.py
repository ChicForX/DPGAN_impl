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

# pruning
# TODO

# gradient hook: adding noise only. grad clipping is replaced by grad penalty.
def create_dp_hook(module, grad_input, grad_output):
    global dynamic_hook
    return dynamic_hook(module, grad_input, grad_output)

def dummy_hook(module, grad_input, grad_output):
    pass

def dp_hook(module, grad_in, grad_out, sensitivity):
    global noise_multiplier
    grad_wrt_image = grad_in[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]

    # reshape
    grad_wrt_image = grad_wrt_image.view(batchsize, -1)

    # add noise, regarding noise_multiplier and SENSITIVITY
    noise = noise_multiplier * sensitivity * torch.randn_like(grad_wrt_image)
    grad_wrt_image = grad_wrt_image + noise

    # reshape to original grad shape
    grad_in_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_in) - 1):
        grad_in_new.append(grad_in[i + 1])

    return tuple(grad_in_new)
