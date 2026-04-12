import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from parameter import PlateParams
from direct_solver import DifferentiableDirectSolver 
from loss import ssim_loss, lpips_loss 
import torch.optim as optim 
from mpl_toolkits.axes_grid1 import make_axes_locatable

wp.init()

def compute_loss_and_adjoint_force(wr_np, wi_np, N, target_pattern):
 
    wr = torch.from_numpy(wr_np).cuda().requires_grad_(True)
    wi = torch.from_numpy(wi_np).cuda().requires_grad_(True)
    
    target = torch.from_numpy(target_pattern.copy()).cuda().reshape(1, 1, N, N)

    amp = torch.sqrt(wr**2 + wi**2 + 1e-12).reshape(1, 1, N, N)
    
    with torch.no_grad():
        clip_max = torch.quantile(amp, 0.98)
        if clip_max < 1e-10: clip_max = 1e-10
    
    display = torch.exp(-(amp / (clip_max * 0.15 + 1e-12))**2)

    l1 = ssim_loss(display, target.reshape(1, 1, N, N))
    l3 = lpips_loss(display, target.reshape(1, 1, N, N))
    
    total_loss = l1 + l3.mean()

    if total_loss.dim() > 0:
        total_loss = total_loss.mean()

    total_loss.backward()
    
    return total_loss.item(), wr.grad.cpu().numpy(), wi.grad.cpu().numpy()

def generate_gaussian_force(N, pos_normalized, sigma=0.02):
    x = torch.linspace(0, 1, N, device="cuda")
    y = torch.linspace(0, 1, N, device="cuda")
    Y, X = torch.meshgrid(y, x, indexing='ij') 
    
    dist_sq = (X - pos_normalized[0])**2 + (Y - pos_normalized[1])**2
    force = torch.exp(-dist_sq / (2 * sigma**2))
    
    return force.flatten()

def main():
    N = 128       
    L = 0.3
    p = PlateParams()
    p.E, p.nu, p.rho = 70.0e9, 0.33, 2700.0
    p.dx = p.dy = L / N
    p.eta = 1e-4

    chladni_solver = DifferentiableDirectSolver(p, N, N)


    h_np = np.ones(N * N, dtype=np.float32) * 0.001
    
    fr_np = np.zeros(N * N, dtype=np.float32)
    fi_wp = wp.zeros(N * N, dtype=float, device="cuda")


    pos_tensor = torch.tensor([0.5, 0.5], device="cuda", requires_grad=True)
    target_freq = 1250.0 
    freq_tensor = torch.tensor([target_freq], device="cuda", requires_grad=True)
    
    optimizer = optim.Adam([
        {'params': pos_tensor, 'lr': 2e-3},  
        {'params': freq_tensor, 'lr': 1.0}  
    ])

    # idx_center = (N // 2) * N + (N // 2)

    # fr_np[idx_center] = 1.0 
    # fr_wp = wp.from_numpy(fr_np, device="cuda")


    image_path = "target_processed.jpg"
    target_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if target_raw is None: raise FileNotFoundError("Target image not found")
    target_resized = cv2.resize(target_raw, (N, N)).astype(np.float32) / 255.0
    
    target_pattern = 1.0 - target_resized
    # target_pattern = np.flipud(target_pattern)

    plt.ion() 
    fig, axes  = plt.subplots(2, 2, figsize=(8, 6), facecolor='black')
    
    ax1, ax2, ax3, ax4 = axes.flatten()
    im1 = ax1.imshow(np.zeros((N, N)), cmap='magma', origin='lower', extent=[0, L*100, 0, L*100], vmin=0, vmax=1)
    im2 = ax2.imshow(np.zeros((N, N)), cmap='viridis', origin='lower', extent=[0, L*100, 0, L*100])
    im3 = ax3.imshow(h_np.reshape(N, N)*1000, cmap='plasma', origin='lower', extent=[0, L*100, 0, L*100])
    im4 = ax4.imshow(np.zeros((N, N)), cmap='inferno', origin='lower', 
                     extent=[0, L*100, 0, L*100], vmin=0, vmax=1.1)
    def add_aligned_colorbar(im, ax, label=""):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(label, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        return cax
    
    add_aligned_colorbar(im1, ax1, label='Normalized Amp')
    add_aligned_colorbar(im2, ax2, label='Log Amp')
    add_aligned_colorbar(im3, ax3, label='Thickness (mm)')
    add_aligned_colorbar(im4, ax4, label='Force Amp')

    for ax in [ax1, ax2, ax3,ax4]:
        ax.tick_params(colors='white')
        ax.set_facecolor('black')

    ax1.set_title("Simulated Pattern", color='white', pad=10)
    ax2.set_title("Log Amplitude", color='white', pad=10)
    ax3.set_title("Plate Thickness", color='white', pad=10)
    ax4.set_title("Excitation Force", color='white', pad=10)

    target_freq = 1250.0 
    p.omega = 2.0 * np.pi * target_freq
    
    
    learning_rate_h = 1e-4
    
    
    for step in range(100000):
        optimizer.zero_grad()

        curr_freq = freq_tensor.item()
        p.omega = 2.0 * np.pi * curr_freq
        
        force_tensor = generate_gaussian_force(N, pos_tensor)
        fr_np = force_tensor.detach().cpu().numpy()
        
        h_wp = wp.from_numpy(h_np, device="cuda")
        fr_wp = wp.from_numpy(fr_np.astype(np.float32), device="cuda")
        
        try:
            w_complex = chladni_solver.solve(h_wp, fr_wp, fi_wp)
            wr_np, wi_np = w_complex.real, w_complex.imag
            
            curr_loss, g_wr, g_wi = compute_loss_and_adjoint_force(
                wr_np, wi_np, N, target_pattern
            )
            
            grad_w_complex = g_wr + 1j * g_wi


            grad_h, grad_fr_np, grad_omega = chladni_solver.compute_adjoint_gradient(h_wp, w_complex, grad_w_complex)
            
            # 厚度 
            grad_norm = np.max(np.abs(grad_h)) + 1e-10
            h_np -= learning_rate_h * (grad_h / grad_norm) 
            h_np = np.clip(h_np, 0.0002, 0.004)

            # 位置和频率
            grad_fr_tensor = torch.from_numpy(grad_fr_np).cuda()
            force_tensor.backward(grad_fr_tensor)
        
            # 频率梯度: dL/dfreq = dL/domega * (2*pi)
            freq_tensor.grad = torch.tensor([grad_omega * 2.0 * np.pi], device="cuda")

            
            optimizer.step()

            
            with torch.no_grad():
                pos_tensor.clamp_(0.05, 0.95)
                freq_tensor.clamp_(10.0, 20000.0)

            print(f"Step {step:04d} | Loss: {curr_loss:.6f} | Freq: {curr_freq:.1f}Hz | Pos: ({pos_tensor[0].item():.3f}, {pos_tensor[1].item():.3f})")
            if step %5 ==0:
                amp = np.abs(w_complex).reshape(N, N)
                clip_max = np.percentile(amp, 98)
                display = np.exp(-(amp / (clip_max * 0.15 + 1e-15))**2)
                
                im1.set_data(display)
                im2.set_data(np.log10(amp + 1e-20))
                im2.set_clim(np.min(np.log10(amp+1e-20)), np.max(np.log10(amp+1e-20)))
                im3.set_data(h_np.reshape(N, N) * 1000.0)
                im4.set_data(fr_np.reshape(N, N))

                ax1.set_title(f"Step: {step} | Loss: {curr_loss:.4f}\nFreq: {curr_freq:.1f} Hz | Pos: ({pos_tensor[0].item():.2f}, {pos_tensor[1].item():.2f})", color='white')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

        except Exception as e:
            print(f" {step} 出错: {e}")
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()