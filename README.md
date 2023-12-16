# Implement of WGAN-GP, DP-WGAN and GS-WGAN

Implements of 3 models.

WGAN-GP: GAN using Wasserstein distance as loss and gradient penalty as constraints.

DP-WGAN: WGAN using PrivateEngine in Opacus package, in which gradient clipping is included.

GS-WGAN: WGAN using DP hook, where noise is added, and gradient penalty. Pretrain of discriminator is adopted.

Different from original GS-WGAN source code, gradient clipping is not adopted, since gradient penalty affored a strict constraint already, according to paper.1.

## Startup Command

Using an additional parameter config_model with values 1, 2, 3, corresponds to the three models WGAN-GP, DP-WGAN, and GS-WGAN respectively. For example: py .\main.py 1.

## Reference

paper:

1.WGAN-GP | Improved Training of Wasserstein GANs(Gulrajani et al., 2017)

2.DP-WGAN | Differentially Private Generative Adversarial Network(Xie et al., 2018)

3.GS-WGAN | GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators(Chen et al., 2021) | [Source Code](https://github.com/DingfanChen/GS-WGAN
)
