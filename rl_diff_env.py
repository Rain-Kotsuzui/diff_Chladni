
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import warp as wp
from parameter import PlateParams
from direct_solver import DifferentiableDirectSolver, ChladniSolverFunction
from loss import ssim_loss, physics_informed_loss, quantile_physics_loss, void_protection_loss

wp.init()


class DifferentiableChladniEnv:
    def __init__(self, target_image_path="target_processed.jpg", N=64, control_N=8):
        self.N = N
        self.control_N = control_N

        # 物理参数
        self.MIN_H, self.MAX_H = 0.0001, 0.01
        self.p = PlateParams()
        self.p.E, self.p.nu, self.p.rho = 70.0e9, 0.33, 2700.0
        self.p.dx = self.p.dy = 0.3 / N
        self.solver = DifferentiableDirectSolver(self.p, N, N)
        self.fi_wp = wp.zeros(N * N, dtype=float, device="cuda")

        self._load_target(target_image_path)

    def _load_target(self, path):
        target_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.target_pattern = cv2.resize(
            target_raw, (self.N, self.N)).astype(np.float32)[::-1, :] / 255.0
        self.target_torch = torch.from_numpy(
            self.target_pattern.copy()).cuda().float().reshape(1, 1, self.N, self.N)

    def _compute_loss(self, amp_torch, h_phys_tensor):

        target = self.target_torch
        h_2d = h_phys_tensor.reshape(1, 1, self.N, self.N)

        material_mask = torch.sigmoid((h_2d - self.MIN_H) * 1e10)

        penalty_value = 1000.0
        effective_amp = amp_torch * material_mask + \
            penalty_value * (1.0 - material_mask)

        clip_max = torch.max(amp_torch).clamp(min=1e-10)
        display = torch.exp(-(effective_amp / (clip_max * 0.15 + 1e-12))**2)

        l1 = ssim_loss(display, target)

        l2 = physics_informed_loss(effective_amp, target)

        l3 = quantile_physics_loss(effective_amp, target, q=0.10)

        l4 = void_protection_loss(h_phys_tensor, target, self.MIN_H * 2) * 1e3

        total_loss = l1 + l2 + l3 + l4

        loss_details = {
            "l1": l1.item(),
            "l2": l2.item(),
            "l3": l3.item(),
            "l4": l4.item()
        }

        return total_loss, loss_details

    def reset(self):
        # 所有的状态都用 Tensor 维护，且需要 requires_grad=True 的连通性
        self.control_points = torch.ones(
            (1, 1, self.control_N, self.control_N), device="cuda") * 0.5
        self.force_pos = torch.tensor([[0.25, 0.25]], device="cuda")
        self.force_freq = torch.tensor([1250.0], device="cuda")

        obs, loss, details = self._simulate()
        self.last_loss = loss.detach()  # 重置时的初始 Loss 不参与反向传播
        return obs

    # 将 force 转换函数改为完全支持微分的版本
    def generate_gaussian_force(self, positions, sigma=0.02):
        x = torch.linspace(0, 1, self.N, device="cuda")
        y = torch.linspace(0, 1, self.N, device="cuda")
        Y, X = torch.meshgrid(y, x, indexing='ij')

        # 确保 positions 的形状正确
        pos_x, pos_y = positions[0, 0], positions[0, 1]
        dist_sq = (X - pos_x)**2 + (Y - pos_y)**2
        force = torch.exp(-dist_sq / (2 * sigma**2)) * 10.0
        return force.view(1, 1, self.N, self.N)

    def step(self, action_tensor):
        # action_tensor 形状应为 (67,)
        thickness_act = action_tensor[:64].view(1, 1, 8, 8)
        pos_act = action_tensor[64:66].view(1, 2)
        freq_act = action_tensor[66]

        # 用可微的方式更新状态
        # 注意：不要用 in-place 操作 (如 +=)，用重新赋值以保留计算图
        self.control_points = torch.clamp(
            self.control_points + thickness_act * 0.1, 0.0, 1.0)
        self.force_pos = torch.clamp(self.force_pos + pos_act * 0.2, 0.1, 0.9)
        self.force_freq = torch.clamp(
            self.force_freq + freq_act * 100.0, 50.0, 2000.0)

        # 模拟并获取可微损失
        obs, current_loss, loss_details = self._simulate()

        # Reward 也是带有梯度的 Tensor
        reward = (self.last_loss - current_loss) * 100.0
        self.last_loss = current_loss.detach()

        info = {
            "loss": current_loss.detach().item(),
            "freq": self.force_freq.detach().item()
        }

        info.update(loss_details)
        return obs, reward, current_loss, info

    def _simulate(self):
        # 1. 状态映射 (全部通过 PyTorch 算子)
        rho_t = F.interpolate(self.control_points, size=(
            self.N, self.N), mode='bicubic', align_corners=False).clamp(0, 1)
        h_phys_t = rho_t * (self.MAX_H - self.MIN_H/2) + self.MIN_H/2

        force_t = self.generate_gaussian_force(self.force_pos, sigma=0.02)
        omega_t = self.force_freq * 2.0 * np.pi

        # 2. 调用我们封装的 Autograd 算子 !!! 这是核心 !!!
        wr, wi = ChladniSolverFunction.apply(
            h_phys_t, force_t, omega_t, self.solver, self.p, self.fi_wp
        )

        # 3. 计算可微 Loss
        amp_t = torch.sqrt(wr**2 + wi**2 + 1e-12)
        # 沿用你原来的 _compute_loss (确保里面全是 torch 操作)
        total_loss, details = self._compute_loss(amp_t, h_phys_t)

        # 4. 组装 Observation Tensor
        obs = torch.cat([
            self.target_torch,
            rho_t,
            amp_t / (amp_t.max() + 1e-8),
            force_t / (force_t.max() + 1e-8)
        ], dim=1)  # Shape: (1, 4, N, N)

        return obs, total_loss, details
    
    def detach_state(self):
        """切断计算图，但保留当前物理状态供后续步骤继续演化"""
        self.control_points = self.control_points.detach().clone()
        self.force_pos = self.force_pos.detach().clone()
        self.force_freq = self.force_freq.detach().clone()
        self.last_loss = self.last_loss.detach()


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享特征提取器
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        feature_dim = 32 * 16 * 16  # 针对 64x64 的输入

        # Actor 网络输出均值和标准差 (针对 67 个动作参数: 64厚度 + 2位置 + 1频率)
        self.actor_mean = nn.Linear(feature_dim, 67)
        self.actor_log_std = nn.Parameter(torch.zeros(67))

        # Critic 网络输出单个价值
        self.critic = nn.Linear(feature_dim, 1)

    def forward(self, obs, deterministic=False):
        features = self.conv(obs)

        # 计算价值 V(s)
        value = self.critic(features)

        # 计算动作分布
        action_mean = torch.tanh(self.actor_mean(features))  # 将均值压在 [-1, 1]
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

        dist = Normal(action_mean, action_std)

        if deterministic:
            action = action_mean
        else:
            # SHAC 的关键：rsample 保证了梯度能从 action 流回 actor_mean 和 actor_log_std
            action = dist.rsample()

        return action.squeeze(0), value.squeeze(-1)
