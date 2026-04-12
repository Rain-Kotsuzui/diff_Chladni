from torchmetrics.functional import structural_similarity_index_measure as ssim
import numpy as np

def ssim_loss(display_pattern, target_pattern):
    
    similarity_index = ssim(display_pattern, target_pattern, data_range=1.0)
    
    return 1.0 - similarity_index


def physics_informed_loss(amp, target_pattern):
    energy_field = amp ** 2
    
    energy_on_lines = np.sum(energy_field * target_pattern)
    
    total_energy = np.sum(energy_field) + 1e-10
    
    loss = energy_on_lines / total_energy
    
    return loss

import torch
import lpips

def lpips_loss(display_pattern, target_pattern, loss_model = lpips.LPIPS(net='vgg').cuda()):
    display_3ch = display_pattern.repeat(1, 3, 1, 1)
    target_3ch = target_pattern.repeat(1, 3, 1, 1)

    img1 = display_3ch* 2.0 - 1.0
    img2 = target_3ch* 2.0 - 1.0
    
    dist = loss_model(img1, img2)
    
    return dist.mean()
