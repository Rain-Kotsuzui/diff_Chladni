import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warp as wp
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 引入你的自定义模块
from parameter import PlateParams
from direct_solver import DifferentiableDirectSolver
from loss import ssim_loss, lpips_loss, physics_informed_loss, void_protection_loss, quantile_physics_loss

wp.init()

MIN_H = 0.0001
MAX_H = 0.01
max_steps = 100000
FREQ_MIN = 10.0
FREQ_MAX = 5000.0
FREQ_SWEEP_POINTS = 1000
FREQ_SWEEP_INTERVAL = 200
N = 64
L = 0.3

class ChladniSolverFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_phys, force, omega, solver, p, fi_wp):

        h_wp = wp.from_torch(h_phys.flatten().contiguous())
        f_wp = wp.from_torch(force.flatten().contiguous())
        p.omega = float(omega.item())

        w_complex = solver.solve(h_wp, f_wp, fi_wp)
        
        wr = torch.from_numpy(w_complex.real).cuda().float().reshape(1, 1, N, N)
        wi = torch.from_numpy(w_complex.imag).cuda().float().reshape(1, 1, N, N)

        ctx.solver = solver
        ctx.h_wp = h_wp
        ctx.w_complex = w_complex
        
        return wr, wi

    @staticmethod
    def backward(ctx, grad_wr, grad_wi):
        solver = ctx.solver
        h_wp = ctx.h_wp
        w_complex = ctx.w_complex

        grad_w_complex = grad_wr.cpu().numpy().flatten() + 1j * grad_wi.cpu().numpy().flatten()

        grad_h, grad_fr, grad_omega = solver.compute_adjoint_gradient(
            h_wp, w_complex, grad_w_complex
        )

        # 转换回 Tensor
        gh_torch = torch.from_numpy(grad_h).cuda().float().reshape(1, 1, N, N)
        gf_torch = torch.from_numpy(grad_fr).cuda().float().reshape(1, 1, N, N)
        gomega_torch = torch.tensor([grad_omega], device="cuda", dtype=torch.float32)

        return gh_torch, gf_torch, gomega_torch, None, None, None

# ==========================================
# 辅助函数
# ==========================================
def generate_gaussian_force(N, positions, sigma=0.02):
    x = torch.linspace(0, 1, N, device="cuda")
    y = torch.linspace(0, 1, N, device="cuda")
    Y, X = torch.meshgrid(y, x, indexing='ij')

    total_force = torch.zeros(N * N, device="cuda")
    for i in range(positions.shape[0]):
        dist_sq = (X - positions[i, 0])**2 + (Y - positions[i, 1])**2
        force = torch.exp(-dist_sq / (2 * sigma**2)) * 10.0
        total_force += force.flatten()
    return total_force.view(1, 1, N, N)

def evaluate_loss(h_phys, amp, target_pattern):
    """统一 Loss 逻辑，修正 1e10 为 1e4 以保护梯度下溢"""
    material_mask = torch.sigmoid((h_phys - MIN_H) * 1e4) 
    
    penalty_value = 1000.0
    effective_amp = amp * material_mask + penalty_value * (1.0 - material_mask)

    clip_max = torch.max(amp).clamp(min=1e-10)
    display = torch.exp(-(effective_amp / (clip_max * 0.15 + 1e-12))**2)

    l1 = ssim_loss(display, target_pattern)
    l2 = physics_informed_loss(effective_amp, target_pattern)
    l3 = quantile_physics_loss(effective_amp, target_pattern, q=0.10).mean()
    l4 = void_protection_loss(h_phys, target_pattern) * 100.0

    total_loss = l1 + l2 + l3 + l4
    return total_loss, l1, l2, l3, l4, display


def main(resume_step=None, num_sources=1):
    output_dir = f"./train_results/sources_{num_sources}"
    os.makedirs(output_dir, exist_ok=True)

    p = PlateParams()
    p.E, p.nu, p.rho = 70.0e9, 0.33, 2700.0
    p.dx = p.dy = L / N
    p.eta = 1e-6
    chladni_solver = DifferentiableDirectSolver(p, N, N)
    fi_wp = wp.zeros(N * N, dtype=float, device="cuda")

    image_path = "target_processed.jpg"
    target_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if target_raw is None:
        raise FileNotFoundError("Target image not found")
    target_resized = cv2.resize(target_raw, (N, N)).astype(np.float32) / 255.0
    target_pattern = torch.from_numpy(target_resized[::-1, :].copy()).cuda().float().reshape(1, 1, N, N)

    # 定义无界的可学习参数 (依靠 sigmoid 映射到物理界限，防截断)
    raw_rho = nn.Parameter(torch.zeros(1, 1, N, N, device="cuda")) # sigmoid(0) = 0.5 初始厚度
    raw_pos = nn.Parameter(torch.zeros(1, 2, device="cuda"))       # 初始位置中心

    optimizer = optim.Adam([
        {'params': [raw_rho], 'lr': 0.1},
        {'params': [raw_pos], 'lr': 0.5}
    ])

    best_freq = 1250.0

    # ==========================================
    # 2. 保留原有的绘图界面初始化
    # ==========================================
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), facecolor='black')
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    im1 = ax1.imshow(np.zeros((N, N)), cmap='magma', origin='lower', extent=[0, L*100, 0, L*100], vmin=0, vmax=1)
    im2 = ax2.imshow(np.zeros((N, N)), cmap='viridis', origin='lower', extent=[0, L*100, 0, L*100])
    im3 = ax3.imshow(np.zeros((N, N)), cmap='plasma', origin='lower', extent=[0, L*100, 0, L*100], vmin=0.0, vmax=10.0)
    im4 = ax4.imshow(np.zeros((N, N)), cmap='inferno', origin='lower', extent=[0, L*100, 0, L*100], vmin=0, vmax=1.1)

    x_coords = np.linspace(0, L*100, N)
    y_coords = np.linspace(0, L*100, N)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

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

    # 绘制 Loss 历史曲线窗口
    loss_history, loss_history_l1, loss_history_l2, loss_history_l3, loss_history_l4, step_history = [], [], [], [], [], []
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
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    ax_loss.legend()

    def update_visualization(h_phys_tensor, w_complex_np, force_np, step_idx, freq_hz, loss_value, phase="Train"):
        h_phys_np = h_phys_tensor.detach().cpu().numpy().reshape(N, N)
        amp = np.abs(w_complex_np).reshape(N, N)
        clip_max = np.max(amp)
        display = np.exp(-(amp / (clip_max * 0.15 + 1e-15))**2)

        norm = np.max(display)
        if norm > 1e-12:
            display = display / norm

        is_void = h_phys_np <= MIN_H
        masked_h = np.ma.masked_where(is_void, h_phys_np)
        masked_display = np.ma.masked_where(is_void, display)
        im1.set_data(masked_display)

        material_mask_vis = (~is_void).astype(np.float32)
        penalty_value = 100000.0
        effective_amp_vis = amp * material_mask_vis + penalty_value * (1.0 - material_mask_vis)
        
        im2.set_data(np.log10((effective_amp_vis + 1e-12)).reshape(N, N))

        for c in ax2.collections:
            c.remove()
        energy_field = (w_complex_np.real**2 + w_complex_np.imag**2).reshape(N, N)
        non_zero_energy = energy_field[energy_field > 1e-18]
        if non_zero_energy.size > 0:
            threshold = np.percentile(non_zero_energy, 10)
            ax2.contour(X_grid, Y_grid, energy_field, levels=[threshold], colors='white', linewidths=1.5)
        
        im2.set_clim(np.min(np.log10(amp + 1e-12)), np.max(np.log10(amp + 1e-12)))
        im3.set_data(masked_h * 1000.0)
        im4.set_data(force_np.reshape(N, N))

        ax1.set_title(f"{phase} | Step: {step_idx} | Loss: {loss_value:.4f}\nFreq: {freq_hz:.1f} Hz", color='white')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()



    for step in range(max_steps):
        
        if step > 0 and step % FREQ_SWEEP_INTERVAL == 0:
            with torch.no_grad():
                h_phys_eval = torch.sigmoid(raw_rho) * (MAX_H - MIN_H) + MIN_H
                pos_eval = torch.sigmoid(raw_pos) * 0.8 + 0.1
                force_eval = generate_gaussian_force(N, pos_eval)
                
                fr_eval_np = force_eval.detach().cpu().numpy().astype(np.float32).flatten()
                h_eval_wp = wp.from_torch(h_phys_eval.flatten().contiguous())
                fr_eval_wp = wp.from_numpy(fr_eval_np, device="cuda")

                best_loss = float("inf")
                freq_candidates = np.linspace(FREQ_MIN, FREQ_MAX, FREQ_SWEEP_POINTS, dtype=np.float32)

                for f in freq_candidates:
                    p.omega = 2.0 * np.pi * float(f)
                    w_complex_eval = chladni_solver.solve(h_eval_wp, fr_eval_wp, fi_wp)
                    
                    wr_eval = torch.from_numpy(w_complex_eval.real).cuda().float().view(1, 1, N, N)
                    wi_eval = torch.from_numpy(w_complex_eval.imag).cuda().float().view(1, 1, N, N)
                    amp_eval = torch.sqrt(wr_eval**2 + wi_eval**2 + 1e-12)
                    
                    loss_eval, _, _, _, _, _ = evaluate_loss(h_phys_eval, amp_eval, target_pattern)
                    
                    if loss_eval.item() < best_loss:
                        best_loss = loss_eval.item()
                        best_freq = float(f)
                        best_w_complex = w_complex_eval
                        
                    update_visualization(h_phys_eval, w_complex_eval, force_eval.detach().cpu().numpy(), step, float(f), loss_eval.item(), phase="Sweep")

                print(f"[Freq Sweep] Step {step:04d} | Best Freq: {best_freq:.2f} Hz | Loss: {best_loss:.6f}")


        optimizer.zero_grad()

        h_phys = torch.sigmoid(raw_rho) * (MAX_H - MIN_H) + MIN_H
        pos = torch.sigmoid(raw_pos) * 0.8 + 0.1
        force_tensor = generate_gaussian_force(N, pos)
        omega = torch.tensor([best_freq * 2.0 * np.pi], device="cuda", requires_grad=True)

        wr, wi = ChladniSolverFunction.apply(h_phys, force_tensor, omega, chladni_solver, p, fi_wp)
        amp = torch.sqrt(wr**2 + wi**2 + 1e-12)
        
        curr_loss, l1, l2, l3, l4, display = evaluate_loss(h_phys, amp, target_pattern)

        curr_loss.backward()

        torch.nn.utils.clip_grad_norm_([raw_rho, raw_pos], max_norm=5.0)

        optimizer.step()

        w_complex_np = wr.detach().cpu().numpy().flatten() + 1j * wi.detach().cpu().numpy().flatten()
        force_np = force_tensor.detach().cpu().numpy()
        
        loss_val = curr_loss.item()
        print(f"Step {step:04d} | Loss: {loss_val:.6f} | Freq: {best_freq:.1f}Hz")
        
        loss_history.append(loss_val)
        loss_history_l1.append(l1.item())
        loss_history_l2.append(l2.item())
        loss_history_l3.append(l3.item())
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
            update_visualization(h_phys, w_complex_np, force_np, step, best_freq, loss_val, phase="Train")

        if step > 0 and step % 100 == 0:
            step_dir = os.path.join(output_dir, f"{step}")
            os.makedirs(step_dir, exist_ok=True)
            np.save(os.path.join(step_dir, f"h.npy"), h_phys.detach().cpu().numpy())
            np.savez(os.path.join(step_dir, "force.npz"), 
                     pos=pos.detach().cpu().numpy(), freq=np.array([best_freq]))

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main(None, 1)