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
def master_hook_adder(module, grad_input, grad_output):
    global dynamic_hook
    return dynamic_hook(module, grad_input, grad_output)

def dummy_hook(module, grad_input, grad_output):
    pass

def dp_hook(sensitivity, noise_multiplier, clip_bound_batch):
    def hook_function(module, grad_input, grad_output):
        grad_wrt_input = grad_input[0]
        grad_input_shape = grad_wrt_input.size()
        batch_size = grad_input_shape[0]

        # reshape
        grad_wrt_input = grad_wrt_input.view(batch_size, -1)

        # clipping
        clip_bound = clip_bound_batch / batch_size
        grad_input_norm = torch.norm(grad_wrt_input, p=2, dim=1)
        clip_coef = clip_bound / (grad_input_norm + 1e-10)
        clip_coef = clip_coef.unsqueeze(-1)
        grad_wrt_input = clip_coef * grad_wrt_input

        # add noise, considering noise_multiplier and sensitivity
        noise = clip_bound * noise_multiplier * sensitivity * torch.randn_like(grad_wrt_input)
        grad_wrt_input = grad_wrt_input + noise

        # reshape to the original grad shape
        grad_in_new = [grad_wrt_input.view(grad_input_shape)]
        for i in range(1, len(grad_input)):
            grad_in_new.append(grad_input[i])

        return tuple(grad_in_new)

    return hook_function
