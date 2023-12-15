config_wgan_gp = {
    'lr_D': 4e-3,
    'lr_G': 1e-3,
    'gen_images_dir': "./gen_images_wgan",
    'num_workers': 8,
    'total_epochs': 50,
    'batch_size': 64
}

config_dp_wgan_gp = {
    'lr_D': 5e-4,
    'lr_G': 2e-4,
    'gen_images_dir': "./gen_images_dpwgan",
    'num_workers': 0,
    'total_epochs': 100,
    'batch_size': 128,
    'delta': 1e-4
}

config_gs_wgan = {
    'lr_D': 4e-3,
    'lr_G': 1e-3,
    'gen_images_dir': "./gen_images_gswgan",
    'num_workers': 0,
    'total_epochs': 50,
    'batch_size': 64
}
