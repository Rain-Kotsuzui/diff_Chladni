import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from parameter import PlateParams
from direct_solver import DifferentiableDirectSolver 
from loss import ssim_loss, lpips_loss ,physics_informed_loss,void_protection_loss
import torch.optim as optim 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

import scipy.ndimage as ndimage

wp.init()

def compute_loss_and_adjoint_force(h_tensor,wr_np, wi_np, N, target_pattern):
 
    wr = torch.from_numpy(wr_np).cuda().float().requires_grad_(True)
    wi = torch.from_numpy(wi_np).cuda().float().requires_grad_(True)
    
    target = torch.from_numpy(target_pattern.copy()).cuda().float().reshape(1, 1, N, N)

    amp = torch.sqrt(wr**2 + wi**2 + 1e-12).reshape(1, 1, N, N)
    
    with torch.no_grad():
        clip_max = torch.max(amp)
        if clip_max < 1e-10: clip_max = 1e-10
    
    display = torch.exp(-(amp / (clip_max * 0.15 + 1e-12))**2)
    
    energy_field = (wr**2 + wi**2).reshape(1, 1, N, N)

    l2 = physics_informed_loss(energy_field, target)*0.001

    l1 = ssim_loss(display, target.reshape(1, 1, N, N))*100.0
    l3 = lpips_loss(display, target.reshape(1, 1, N, N))*100.0

    total_loss = l1+ l3.mean()+l2

    if total_loss.dim() > 0:
        total_loss = total_loss.mean()

    total_loss.backward()
    
    return total_loss.item(), wr.grad.cpu().numpy(), wi.grad.cpu().numpy(), l1.item(),l2.item(), l3.mean().item()

def generate_gaussian_force(N, positions, sigma=0.02):
    x = torch.linspace(0, 1, N, device="cuda")
    y = torch.linspace(0, 1, N, device="cuda")
    Y, X = torch.meshgrid(y, x, indexing='ij') 
    
    total_force = torch.zeros(N * N, device="cuda")

    # dist_sq = (X - pos_normalized[0])**2 + (Y - pos_normalized[1])**2
    # force = torch.exp(-dist_sq / (2 * sigma**2))
    for i in range(positions.shape[0]):
        dist_sq = (X - positions[i, 0])**2 + (Y - positions[i, 1])**2
        force = torch.exp(-dist_sq / (2 * sigma**2))
        total_force += force.flatten()
    return total_force

def main(resume_step=None,num_sources=1):
    output_dir = f"./train_results/sources_{num_sources}"
    os.makedirs(output_dir, exist_ok=True)
    

    N = 64       
    L = 0.3
    p = PlateParams()
    p.E, p.nu, p.rho = 70.0e9, 0.33, 2700.0
    p.dx = p.dy = L / N
    p.eta = 1e-4

    chladni_solver = DifferentiableDirectSolver(p, N, N)


    h_np = np.ones(N * N, dtype=np.float32) * 0.005
    
    h_tensor = torch.full((N * N,), 0.005, device="cuda", requires_grad=True, dtype=torch.float32)

    fr_np = np.zeros(N * N, dtype=np.float32)
    fi_wp = wp.zeros(N * N, dtype=float, device="cuda")


    init_positions = torch.rand((num_sources, 2), device="cuda").float() 
    # init_positions = torch.tensor([[0.5, 0.5]], device="cuda").float()
    
    pos_tensor = init_positions.clone().detach().requires_grad_(True).float() 
    

    target_freq = 1250.0 
    freq_tensor = torch.tensor([target_freq], device="cuda", requires_grad=True)
    
    optimizer = optim.Adam([
        {'params': [h_tensor], 'lr': 0.01},
        {'params': pos_tensor, 'lr': 0.1},  
        {'params': freq_tensor, 'lr': 10.0}  
    ])



    image_path = "target_processed.jpg"
    target_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if target_raw is None: raise FileNotFoundError("Target image not found")
    target_resized  = cv2.resize(target_raw, (N, N)).astype(np.float32) / 255.0
    target_pattern = 1.0 - target_resized[::-1, :] # 左右颠倒

    plt.ion() 
    fig, axes  = plt.subplots(2, 2, figsize=(8, 6), facecolor='black')
    
    ax1, ax2, ax3, ax4 = axes.flatten()
    im1 = ax1.imshow(np.zeros((N, N)), cmap='magma', origin='lower', extent=[0, L*100, 0, L*100], vmin=0, vmax=1)
    im2 = ax2.imshow(np.zeros((N, N)), cmap='viridis', origin='lower', extent=[0, L*100, 0, L*100])
    im3 = ax3.imshow(h_np.reshape(N, N)*1000, cmap='plasma', origin='lower', extent=[0, L*100, 0, L*100],vmin=0.0, vmax=6.0)
    

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
    ax1.set_facecolor('#00FF00')
    ax3.set_facecolor('#00FF00')

    ax1.set_title("Simulated Pattern", color='white', pad=10)
    ax2.set_title("Log Amplitude", color='white', pad=10)
    ax3.set_title("Plate Thickness", color='white', pad=10)
    ax4.set_title("Excitation Force", color='white', pad=10)

    target_freq = 1250.0 
    p.omega = 2.0 * np.pi * target_freq
    
    
    loss_history = []
    loss_history_l1 = []
    loss_history_l2 = []
    loss_history_l3 = []
    loss_history_l4 = []
    step_history = []
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4), facecolor='#F0F0F0')
    line_loss, = ax_loss.plot([], [], 'r-', linewidth=2, label='Total Loss')
    line_loss_l1, = ax_loss.plot([], [], 'g-', linewidth=2, label='L1 Loss')
    line_loss_l2, = ax_loss.plot([], [], 'y-', linewidth=2, label='L2 Loss')
    line_loss_l3, = ax_loss.plot([], [], 'b-', linewidth=2, label='L3 Loss')
    line_loss_l4, = ax_loss.plot([], [], 'm-', linewidth=2, label='Void Protection Loss')

    ax_loss.set_title("Training Loss History")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale('log')
    # ax_loss.set_ylim(1e-1, 1e2)
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    ax_loss.legend()
    
    if resume_step is not None:
        checkpoint_dir = os.path.join(output_dir, f"{resume_step}")
        if os.path.exists(checkpoint_dir):
            print(f"Resuming training from step {resume_step}...")
            h_path = os.path.join(checkpoint_dir, "h.npy")
            if os.path.exists(h_path):
                h_np = np.load(h_path).flatten().astype(np.float32)
                with torch.no_grad():
                    h_tensor.copy_(torch.from_numpy(h_np).cuda())
                print(f"Loaded h from {h_path}")
            
            force_path = os.path.join(checkpoint_dir, "force.npz")
            if os.path.exists(force_path):
                data = np.load(force_path)
                loaded_pos = data['pos']
                loaded_freq = data['freq']
                with torch.no_grad():
                    pos_tensor.copy_(torch.tensor(loaded_pos, device="cuda"))
                    freq_tensor.copy_(torch.tensor(loaded_freq, device="cuda"))
                print(f"Loaded pos: {loaded_pos}, freq: {loaded_freq}")
        else:
            print(f"Checkpoint directory {checkpoint_dir} not found. Starting from scratch.")
    else:
        random_noise = np.random.rand(N * N).astype(np.float32)
    
        noise_2d = random_noise.reshape(N, N)
    
        smoothed_noise = ndimage.gaussian_filter(noise_2d, sigma=2.0)
    
        smoothed_noise = (smoothed_noise - smoothed_noise.min()) / (smoothed_noise.max() - smoothed_noise.min())
        smoothed_noise = np.arctan((smoothed_noise - 0.5) * 60)*2 / 3.1415926 + 0.5
        h_min, h_max = 0.0000, 0.006
        h_np = (smoothed_noise * (h_max - h_min) + h_min).flatten().astype(np.float32)

        with torch.no_grad():
            h_tensor.copy_(torch.from_numpy(h_np).cuda())

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    for step in range(100000):
        optimizer.zero_grad()

        curr_freq = freq_tensor.item()
        p.omega = 2.0 * np.pi * curr_freq
        
        force_tensor = generate_gaussian_force(N, pos_tensor)
        fr_np = force_tensor.detach().cpu().numpy()
        
        h_wp = wp.from_torch(h_tensor)

        fr_wp = wp.from_numpy(fr_np.astype(np.float32), device="cuda")
        
        try:
            w_complex = chladni_solver.solve(h_wp, fr_wp, fi_wp)
            wr_np, wi_np = w_complex.real, w_complex.imag
            
            curr_loss, g_wr, g_wi,l1,l2,l3 = compute_loss_and_adjoint_force(
                h_tensor,wr_np, wi_np, N, target_pattern
            )
            
            l4 = (0)*void_protection_loss(h_tensor,torch.from_numpy(target_pattern.copy()).cuda().float().reshape(1, 1, N, N))
            l4.backward()
            
            grad_protect = h_tensor.grad.clone()


            grad_w_complex = g_wr + 1j * g_wi


            grad_h, grad_fr_np, grad_omega = chladni_solver.compute_adjoint_gradient(h_wp, w_complex, grad_w_complex)
            
            gh_torch = torch.from_numpy(grad_h).cuda().float().reshape(1, 1, N, N)

            
            target_mask = torch.from_numpy(target_pattern).cuda().float().reshape(N, N)
            
            active_protection_mask = (target_mask < 0.5) & (gh_torch > 0)
            gh_torch[active_protection_mask] = 0.0


            gh_smooth = torch.nn.functional.avg_pool2d(gh_torch, kernel_size=3, stride=1, padding=1)
            gh_phys_final = gh_smooth / (torch.norm(gh_smooth) + 1e-10)

            
            h_tensor.grad =gh_phys_final.flatten()+grad_protect
            

            # 厚度 
            # grad_norm = np.max(np.abs(grad_h)) + 1e-10
            # h_np -= learning_rate_h * (grad_h / grad_norm) 


            # grad_h_2d = grad_h.reshape(N, N)

            # grad_h_smooth = ndimage.gaussian_filter(grad_h_2d, sigma=0.1)

            # h_np -= learning_rate_h * grad_h_smooth.flatten() 
            # h_np = np.clip(h_np, 0.000, 0.010)


            # 位置和频率
            grad_fr_tensor = torch.from_numpy(grad_fr_np).cuda().float()
            force_tensor.backward(grad_fr_tensor)
        
            # 频率梯度: dL/dfreq = dL/domega * (2*pi)
            freq_tensor.grad = torch.tensor([grad_omega * 2.0 * np.pi], device="cuda",dtype=torch.float32)

            
            # scheduler.step(curr_loss)
            optimizer.step()

            
            with torch.no_grad():
                pos_tensor.clamp_(0.05, 0.95)
                freq_tensor.clamp_(10.0, 20000.0)
                h_tensor.clamp_(0.0001/2, 0.006)
                h_np = h_tensor.cpu().numpy()

            print(f"Step {step:04d} | Loss: {curr_loss:.6f} | Freq: {curr_freq:.1f}Hz)")
            loss_history.append(curr_loss)
            loss_history_l1.append(l1)
            loss_history_l2.append(l2)
            loss_history_l3.append(l3)
            loss_history_l4.append(l4.item())
            step_history.append(step)
            line_loss.set_data(step_history, loss_history)
            line_loss_l1.set_data(step_history, loss_history_l1)
            line_loss_l2.set_data(step_history, loss_history_l2)
            line_loss_l3.set_data(step_history, loss_history_l3)
            line_loss_l4.set_data(step_history, loss_history_l4)
            
            ax_loss.relim()
            ax_loss.autoscale_view()
            
            fig_loss.canvas.draw_idle()
            fig_loss.canvas.flush_events()


            if step %1 ==0:
                amp = np.abs(w_complex).reshape(N, N)
                clip_max = np.max(amp)
                display = np.exp(-(amp / (clip_max * 0.15 + 1e-15))**2)
                

                masked_h = np.ma.masked_where(h_np.reshape(N, N) < 0.0001, h_np.reshape(N, N))
                masked_display = np.ma.masked_where(h_np.reshape(N, N) < 0.0001, display)

                im1.set_data(masked_display)
                im2.set_data(np.log10(amp + 1e-20))
                im2.set_clim(np.min(np.log10(amp+1e-20)), np.max(np.log10(amp+1e-20)))
                
                im3.set_data(masked_h*1000.0)
                # im3.set_data(h_np.reshape(N, N) * 1000.0)

                im4.set_data(fr_np.reshape(N, N))

                ax1.set_title(f"Step: {step} | Loss: {curr_loss:.4f}\nFreq: {curr_freq:.1f} Hz )", color='white')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                if step % 100 == 0:
                    step_dir = os.path.join(output_dir, f"{step}")
                    os.makedirs(step_dir, exist_ok=True)
            

                    h_to_save = (h_np.reshape(N, N)).astype(np.float32)
                    np.save(os.path.join(step_dir, f"h.npy"), h_to_save)

                    np.savez(
                        os.path.join(step_dir, "force.npz"),
                        pos=pos_tensor.detach().cpu().numpy(),  # 位置 [x, y]
                        freq=freq_tensor.detach().cpu().numpy() # 频率 [f]
                    )
                    print(f"Checkpoint saved at step {step}")

                # plt.pause(0.001)

        except Exception as e:
            print(f" {step} 出错: {e}")
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main(None,2)