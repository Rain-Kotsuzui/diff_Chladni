from torchmetrics.functional import structural_similarity_index_measure as ssim
import numpy as np

def ssim_loss(display_pattern, target_pattern):
    
    similarity_index = ssim(display_pattern, target_pattern, data_range=1.0)
    
    return 1.0 - similarity_index


def physics_informed_loss(energy_field, target_pattern):
    # max_energy = torch.max(energy_field)
    # energy_field = energy_field / (max_energy + 1e-8)
    # energy_on_target = torch.sum(energy_field * target_pattern)
    
    # energy_off_target = torch.sum(energy_field * (1.0 - target_pattern))
    
    # loss = energy_on_target / (energy_off_target + 1e-8)

    avg_energy = torch.mean(energy_field) + 1e-8
    norm_energy = energy_field / avg_energy
    
    loss = torch.mean(norm_energy * target_pattern) - 0.5 * torch.mean(norm_energy * (1.0 - target_pattern))
    return loss

def void_protection_loss(h_tensor, target_pattern, safe_h=0.001):
     
    h_2d = h_tensor.reshape(target_pattern.shape)
    
    eps = 1e-5
    violation = 1.0 / (h_2d + eps) 
    
    mask = (target_pattern < 0.5).int()
    
    penalty = (violation * target_pattern)*mask
    return torch.mean(penalty)

import torch
import lpips

def lpips_loss(display_pattern, target_pattern, loss_model = lpips.LPIPS(net='vgg').cuda()):
    display_3ch = display_pattern.repeat(1, 3, 1, 1)
    target_3ch = target_pattern.repeat(1, 3, 1, 1)

    img1 = display_3ch* 2.0 - 1.0
    img2 = target_3ch* 2.0 - 1.0
    
    dist = loss_model(img1, img2)
    
    return dist.mean()
