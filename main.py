import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from parameter import PlateParams
from direct_solver import DifferentiableDirectSolver
from loss import ssim_loss, lpips_loss, physics_informed_loss, void_protection_loss, quantile_physics_loss
import torch.optim as optim
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import torch.nn.functional as F
import scipy.ndimage as ndimage
from device_manager import TORCH_DEVICE, WARP_DEVICE, init_devices, to_device

# 初始化设备
init_devices()

MIN_H = 0.0001
MAX_H = 0.01
max_steps = 100000
FREQ_MIN = 10.0
FREQ_MAX = 5000.0
FREQ_SWEEP_POINTS = 1000
FREQ_SWEEP_INTERVAL = 200


def compute_loss_and_adjoint_force(h_tensor, wr_np, wi_np, N, target_pattern):
    wr = torch.from_numpy(wr_np).to(TORCH_DEVICE).float().requires_grad_(True)
    wi = torch.from_numpy(wi_np).to(TORCH_DEVICE).float().requires_grad_(True)

    target = torch.from_numpy(target_pattern.copy()).to(TORCH_DEVICE).float().reshape(1, 1, N, N)

    amp = torch.sqrt(wr**2 + wi**2+1e-12).reshape(1, 1, N, N)
    h_2d = h_tensor.reshape(1, 1, N, N)

    material_mask = torch.sigmoid((h_2d - MIN_H) * 1e10)
    penalty_value = 1000.0
    effective_amp = (amp * material_mask +
                     penalty_value * (1.0 - material_mask))

    # effective_amp = amp

    with torch.no_grad():
        clip_max = torch.max(amp)
        if clip_max < 1e-10:
            clip_max = 1e-10

    display = torch.exp(-(effective_amp / (clip_max * 0.15 + 1e-12))**2)

    l2 = physics_informed_loss(effective_amp, target)

    l1 = ssim_loss(display, target.reshape(1, 1, N, N))
    l3 = quantile_physics_loss(effective_amp, target, q=0.10)
    

    total_loss = l1 + l3.mean()+l2

    if total_loss.dim() > 0:
        total_loss = total_loss.mean()

    # total_loss.backward()
    total_loss.backward(retain_graph=True)
    return total_loss.item(), wr.grad.cpu().numpy(), wi.grad.cpu().numpy(), l1.item(), l2.item(), l3.mean().item()


def evaluate_total_loss_no_grad(h_tensor, wr_np, wi_np, N, target_pattern):
    with torch.no_grad():
        wr = torch.from_numpy(wr_np).to(TORCH_DEVICE).float()
        wi = torch.from_numpy(wi_np).to(TORCH_DEVICE).float()
        target = torch.from_numpy(target_pattern.copy()).to(TORCH_DEVICE).float().reshape(1, 1, N, N)

        amp = torch.sqrt(wr**2 + wi**2 + 1e-12).reshape(1, 1, N, N)
        h_2d = h_tensor.reshape(1, 1, N, N)

        material_mask = torch.sigmoid((h_2d - MIN_H) * 1e10)
        penalty_value = 1000.0
        effective_amp = amp * material_mask + \
            penalty_value * (1.0 - material_mask)

        clip_max = torch.max(amp).clamp(min=1e-10)
        display = torch.exp(-(effective_amp / (clip_max * 0.15 + 1e-12))**2)

        l2 = physics_informed_loss(effective_amp, target)
        l1 = ssim_loss(display, target)
        l3 = quantile_physics_loss(effective_amp, target, q=0.10)
        l4 = void_protection_loss(h_tensor, target)

        total_loss = l1 + l2 + l3.mean() + l4
        return total_loss.item()


def generate_gaussian_force(N, positions, sigma=0.02):
    x = torch.linspace(0, 1, N, device=TORCH_DEVICE)
    y = torch.linspace(0, 1, N, device=TORCH_DEVICE)
    Y, X = torch.meshgrid(y, x, indexing='ij') 
    
    total_force = torch.zeros(N * N, device=TORCH_DEVICE)

    # dist_sq = (X - pos_normalized[0])**2 + (Y - pos_normalized[1])**2
    # force = torch.exp(-dist_sq / (2 * sigma**2))
    for i in range(positions.shape[0]):
        dist_sq = (X - positions[i, 0])**2 + (Y - positions[i, 1])**2
        force = torch.exp(-dist_sq / (2 * sigma**2))*10.0
        total_force += force.flatten()
    return total_force


def apply_diffusion(rho, sigma):
    s = float(sigma)
    if s <= 0.1:
        return rho

    k_size = int(4 * sigma + 1)
    if k_size % 2 == 0:
        k_size += 1

    x = torch.arange(k_size).float() - k_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = gauss.view(1, 1, -1, 1) * gauss.view(1, 1, 1, -1)
    kernel = kernel.to(rho.device)

    return F.conv2d(rho, kernel, padding=k_size // 2)


def main(resume_step=None, num_sources=1):
    output_dir = f"./train_results/sources_{num_sources}"
    os.makedirs(output_dir, exist_ok=True)

    N = 64
    L = 0.3
    p = PlateParams()
    p.E, p.nu, p.rho = 70.0e9, 0.33, 2700.0
    p.dx = p.dy = L / N
    p.eta = 1e-6

    chladni_solver = DifferentiableDirectSolver(p, N, N)

    h_np = np.ones(N * N, dtype=np.float32) * 0.005
    
    h_tensor = torch.full((N * N,), 0.005, device=TORCH_DEVICE, requires_grad=True, dtype=torch.float32)

    fr_np = np.zeros(N * N, dtype=np.float32)
    fi_wp = wp.zeros(N * N, dtype=float, device=WARP_DEVICE)

    init_positions = torch.rand((num_sources, 2), device=TORCH_DEVICE).float() 
    # init_positions = torch.tensor([[0.5, 0.5]], device=TORCH_DEVICE).float()
    
    pos_tensor = init_positions.clone().detach().requires_grad_(True).float() 

    target_freq = 1250.0 
    freq_tensor = torch.tensor([target_freq], device=TORCH_DEVICE, requires_grad=True)

    image_path = "target_processed.jpg"
    target_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if target_raw is None:
        raise FileNotFoundError("Target image not found")
    target_resized = cv2.resize(target_raw, (N, N)).astype(np.float32) / 255.0
    target_pattern = target_resized[::-1, :]  # 左右颠倒

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), facecolor='black')

    ax1, ax2, ax3, ax4 = axes.flatten()
    im1 = ax1.imshow(np.zeros((N, N)), cmap='magma', origin='lower', extent=[
                     0, L*100, 0, L*100], vmin=0, vmax=1)
    im2 = ax2.imshow(np.zeros((N, N)), cmap='viridis',
                     origin='lower', extent=[0, L*100, 0, L*100])
    im3 = ax3.imshow(h_np.reshape(N, N)*1000, cmap='plasma',
                     origin='lower', extent=[0, L*100, 0, L*100], vmin=0.0, vmax=10.0)

    x_coords = np.linspace(0, L*100, N)
    y_coords = np.linspace(0, L*100, N)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

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
    add_aligned_colorbar(im2, ax2, label='Amp')
    add_aligned_colorbar(im3, ax3, label='Thickness (mm)')
    add_aligned_colorbar(im4, ax4, label='Force Amp')

    for ax in [ax1, ax2, ax3, ax4]:
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
    line_loss_l4, = ax_loss.plot(
        [], [], 'm-', linewidth=2, label='Void Protection Loss')

    ax_loss.set_title("Training Loss History")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale('log')
    # ax_loss.set_ylim(1e-1, 1e5)
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    ax_loss.legend()

    # if resume_step is not None:
    #     checkpoint_dir = os.path.join(output_dir, f"{resume_step}")
    #     if os.path.exists(checkpoint_dir):
    #         print(f"Resuming training from step {resume_step}...")
    #         h_path = os.path.join(checkpoint_dir, "h.npy")
    #         if os.path.exists(h_path):
    #             h_np = np.load(h_path).flatten().astype(np.float32)
    #             with torch.no_grad():
    #                 h_tensor.copy_(torch.from_numpy(h_np).cuda())
    #             print(f"Loaded h from {h_path}")

    #         force_path = os.path.join(checkpoint_dir, "force.npz")
    #         if os.path.exists(force_path):
    #             data = np.load(force_path)
    #             loaded_pos = data['pos']
    #             loaded_freq = data['freq']
    #             with torch.no_grad():
    #                 pos_tensor.copy_(torch.tensor(loaded_pos, device=TORCH_DEVICE))
    #                 freq_tensor.copy_(torch.tensor(loaded_freq, device=TORCH_DEVICE))
    #             print(f"Loaded pos: {loaded_pos}, freq: {loaded_freq}")
    #     else:
    #         print(f"Checkpoint directory {checkpoint_dir} not found. Starting from scratch.")
    # else:
    #     random_noise = np.random.rand(N * N).astype(np.float32)

    #     noise_2d = random_noise.reshape(N, N)

    #     smoothed_noise = ndimage.gaussian_filter(noise_2d, sigma=1.0)

    #     smoothed_noise = (smoothed_noise - smoothed_noise.min()) / (smoothed_noise.max() - smoothed_noise.min())
    #     smoothed_noise = np.arctan((smoothed_noise - 0.5) * 60)*2 / 3.1415926 + 0.5

    #     h_np = (smoothed_noise * (MAX_H - MIN_H) + MIN_H).flatten().astype(np.float32)

    #     with torch.no_grad():
    #         h_tensor.copy_(torch.from_numpy(h_np).cuda())
    
    rho_tensor = torch.full((1, 1, N, N), 0.5, device=TORCH_DEVICE, requires_grad=True) 
    random_noise = np.random.rand(N * N).astype(np.float32)
 

    noise_2d = random_noise.reshape(N, N)

    smoothed_noise = ndimage.gaussian_filter(noise_2d, sigma=1.0)

    smoothed_noise = (smoothed_noise - smoothed_noise.min()) / \
        (smoothed_noise.max() - smoothed_noise.min())
    # rho_np = np.arctan((smoothed_noise - 0.5) * 60)*2 / 3.1415926 + 0.5
    rho_np = smoothed_noise*0.0+0.5

    with torch.no_grad():
        rho_tensor.copy_(torch.from_numpy(rho_np.flatten()).to(TORCH_DEVICE).view(1, 1, N, N))
    def get_physical_h(rho):
        return rho * (MAX_H - MIN_H/2) + MIN_H/2
    # h_np = (smoothed_noise * (MAX_H - MIN_H) + MIN_H).flatten().astype(np.float32)

    optimizer = optim.Adam([
        {'params': [rho_tensor], 'lr': 0.2},
        {'params': pos_tensor, 'lr': 0.05},
        {'params': freq_tensor, 'lr': 1.0}
    ])

    def update_visualization(h_phys_tensor, w_complex_np, force_np, step_idx, freq_hz, loss_value, phase="Train"):
        h_phys_np = h_phys_tensor.detach().cpu().numpy().reshape(N, N)
        amp = np.abs(w_complex_np).reshape(N, N)
        clip_max = np.max(amp)
        display = np.exp(-(amp / (clip_max * 0.15 + 1e-15))**2)

        norm = np.max(display)
        if norm > 1e-12:
            display = display / norm

        masked_h = np.ma.masked_where(h_phys_np < MIN_H, h_phys_np)
        masked_display = np.ma.masked_where(h_phys_np < MIN_H, display)
        im1.set_data(masked_display)

        material_mask = 1 / (1 + np.exp(-((h_phys_np - MIN_H) * 1e10)))
        penalty_value = 100000.0
        effective_amp = amp * material_mask + \
            penalty_value * (1.0 - material_mask)
        im2.set_data(np.log10((effective_amp + 1e-12)).reshape(N, N))

        for c in ax2.collections:
            c.remove()
        energy_field = (w_complex_np.real**2 +
                        w_complex_np.imag**2).reshape(N, N)
        non_zero_energy = energy_field[energy_field > 1e-18]
        if non_zero_energy.size > 0:
            threshold = np.percentile(non_zero_energy, 10)
            ax2.contour(X_grid, Y_grid, energy_field, levels=[
                        threshold], colors='white', linewidths=1.5)
        im2.set_clim(np.min(np.log10(amp + 1e-12)),
                     np.max(np.log10(amp + 1e-12)))

        im3.set_data(masked_h * 1000.0)
        im4.set_data(force_np.reshape(N, N))

        ax1.set_title(
            f"{phase} | Step: {step_idx} | Loss: {loss_value:.4f}\nFreq: {freq_hz:.1f} Hz", color='white')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    for step in range(int(max_steps)):
        optimizer.zero_grad()

        current_sigma = max(0.0, 1.0 * (1 - step / 5000))

        if step > 0 and step % FREQ_SWEEP_INTERVAL == 0:
            with torch.no_grad():
                h_diffused_eval = apply_diffusion(rho_tensor, current_sigma)
                h_phys_eval = get_physical_h(h_diffused_eval)
                force_eval = generate_gaussian_force(N, pos_tensor)
                fr_eval_np = force_eval.detach().cpu().numpy().astype(np.float32)
                h_eval_wp = wp.from_torch(h_phys_eval.flatten().cpu())
                fr_eval_wp = wp.from_numpy(fr_eval_np, device=WARP_DEVICE)

                best_freq = float(freq_tensor.item())
                best_loss = float("inf")
                freq_candidates = np.linspace(
                    FREQ_MIN, FREQ_MAX, FREQ_SWEEP_POINTS, dtype=np.float32)

                for f in freq_candidates:
                    p.omega = 2.0 * np.pi * float(f)
                    w_complex_eval = chladni_solver.solve(
                        h_eval_wp, fr_eval_wp, fi_wp)
                    loss_eval = evaluate_total_loss_no_grad(
                        h_phys_eval,
                        w_complex_eval.real,
                        w_complex_eval.imag,
                        N,
                        target_pattern
                    )
                    update_visualization(
                        h_phys_eval,
                        w_complex_eval,
                        fr_eval_np,
                        step,
                        float(f),
                        loss_eval,
                        phase="Sweep"
                    )
                    if loss_eval < best_loss:
                        best_loss = loss_eval
                        best_freq = float(f)

                freq_tensor.copy_(torch.tensor([best_freq], device=TORCH_DEVICE, dtype=torch.float32))
                print(f"[Freq Sweep] Step {step:04d} | Best Freq: {best_freq:.2f} Hz | Loss: {best_loss:.6f}")

        curr_freq = freq_tensor.item()
        p.omega = 2.0 * np.pi * curr_freq

        force_tensor = generate_gaussian_force(N, pos_tensor)
        fr_np = force_tensor.detach().cpu().numpy()

        # h_wp = wp.from_torch(h_tensor)
        h_diffused = apply_diffusion(rho_tensor, current_sigma)
        h_phys = get_physical_h(h_diffused)
        
        h_wp = wp.from_torch(h_phys.flatten().cpu())
        fr_wp = wp.from_numpy(fr_np.astype(np.float32), device=WARP_DEVICE)
        
        try:
            w_complex = chladni_solver.solve(h_wp, fr_wp, fi_wp)
            wr_np, wi_np = w_complex.real, w_complex.imag

            curr_loss, g_wr, g_wi, l1, l2, l3 = compute_loss_and_adjoint_force(
                h_phys, wr_np, wi_np, N, target_pattern
            )
            
            l4 = (1.0)*void_protection_loss(h_phys,torch.from_numpy(target_pattern.copy()).to(TORCH_DEVICE).float().reshape(1, 1, N, N))
            # l4.backward()
            (l4).backward(retain_graph=True)

            # grad_protect = rho_tensor.grad.clone()

            grad_w_complex = g_wr + 1j * g_wi

            grad_h, grad_fr_np, grad_omega = chladni_solver.compute_adjoint_gradient(h_wp, w_complex, grad_w_complex)
            gh_torch = torch.from_numpy(grad_h).to(TORCH_DEVICE).float().reshape(1, 1, N, N)
            # target_mask = torch.from_numpy(target_pattern).cuda().float().reshape(N, N)
            # active_protection_mask = (target_mask < 0.5) & (gh_torch > 0)
            # gh_torch[active_protection_mask] = 0.0

            # gh_smooth = torch.nn.functional.avg_pool2d(gh_torch, kernel_size=3, stride=1, padding=1)
            # gh_phys_final = gh_smooth / (torch.abs(gh_smooth).max() + 1e-10)

            gh_torch = gh_torch / (gh_torch.norm() + 1e-8)

            # h_phys.grad =gh_phys_final.flatten()+grad_protect
            h_phys.backward(gradient=gh_torch *0.5)

            with torch.no_grad():
                if step < 10000:
                    low_res_size = 6 
                    low_res_noise = torch.randn(1, 1, low_res_size, low_res_size, device=TORCH_DEVICE)
                    g_noise = F.interpolate(low_res_noise, size=(N, N), mode='bicubic', align_corners=False)
                    current_grad_mag = rho_tensor.grad.abs().mean()
                    noise_strength = (0.1 + current_grad_mag) * \
                        1 * (1 - step / 10000)
                    rho_tensor.grad.add_(g_noise.view_as(
                        rho_tensor.grad) * noise_strength)

            # 厚度
            # grad_norm = np.max(np.abs(grad_h)) + 1e-10
            # h_np -= learning_rate_h * (grad_h / grad_norm)

            # grad_h_2d = grad_h.reshape(N, N)

            # grad_h_smooth = ndimage.gaussian_filter(grad_h_2d, sigma=0.1)

            # h_np -= learning_rate_h * grad_h_smooth.flatten()
            # h_np = np.clip(h_np, 0.000, 0.010)

            # 位置和频率
            grad_fr_tensor = torch.from_numpy(grad_fr_np).to(TORCH_DEVICE).float()
            force_tensor.backward(grad_fr_tensor)
        
            # 频率梯度: dL/dfreq = dL/domega * (2*pi)
            freq_tensor.grad = torch.tensor([grad_omega * 2.0 * np.pi], device=TORCH_DEVICE,dtype=torch.float32)

            # scheduler.step(curr_loss)
            optimizer.step()

            with torch.no_grad():
                pos_tensor.clamp_(0.05, 0.95)
                freq_tensor.clamp_(FREQ_MIN, FREQ_MAX)

                if step % 20 == 0:
                    low_res_size = 8 
                    low_res_noise = torch.randn(1, 1, low_res_size, low_res_size, device=TORCH_DEVICE)
                    continuous_noise = F.interpolate(low_res_noise, size=(N, N), mode='bicubic', align_corners=False)
                    noise_strength = 0.5* (1 - step / 10000) 
                    rho_tensor.add_(continuous_noise.view(1, 1, N, N) * noise_strength)
              
                rho_tensor.clamp_(0, 1)
                # h_tensor.clamp_(MIN_H/2, MAX_H)
                # h_np = h_tensor.cpu().numpy()

            print(
                f"Step {step:04d} | Loss: {curr_loss:.6f} | Freq: {curr_freq:.1f}Hz)")
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

            if step % 1 == 0:
                update_visualization(
                    h_phys, w_complex, fr_np, step, curr_freq, curr_loss, phase="Train")

                if step % 100 == 0:
                    step_dir = os.path.join(output_dir, f"{step}")
                    os.makedirs(step_dir, exist_ok=True)

                    h_to_save = h_phys.detach().cpu().numpy().astype(np.float32)
                    np.save(os.path.join(step_dir, f"h.npy"), h_to_save)

                    np.savez(
                        os.path.join(step_dir, "force.npz"),
                        pos=pos_tensor.detach().cpu().numpy(),  # 位置 [x, y]
                        freq=freq_tensor.detach().cpu().numpy()  # 频率 [f]
                    )
                    print(f"Checkpoint saved at step {step}")

                # plt.pause(0.001)

        except Exception as e:
            print(f" {step} 出错: {e}")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main(None, 1)
