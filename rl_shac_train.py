import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rl_diff_env import DifferentiableChladniEnv

class DifferentiableAgent(nn.Module):
    def __init__(self, obs_shape=(4, 64, 64), action_dim=67):
        super().__init__()
        # 沿用你设计的 CustomCNN
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *obs_shape)).shape[1]

        # 共享特征提取
        self.shared_net = nn.Sequential(
            nn.Linear(n_flatten, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        # Actor 输出动作分布 (均值和对数标准差)
        self.actor_mean = nn.Linear(512, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic 输出状态价值 V(s)
        self.critic = nn.Linear(512, 1)

    def forward(self, obs, deterministic=False):
        features = self.cnn(obs)
        shared = self.shared_net(features)

        value = self.critic(shared)

        # 限制均值在 [-1, 1]
        action_mean = torch.tanh(self.actor_mean(shared))
        # action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

        log_std = torch.clamp(self.actor_log_std, min=-20, max=2)
        action_std = torch.exp(log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        

        if deterministic:
            action = action_mean
        else:
            # 关键：rsample() 允许梯度穿过采样过程流回网络权重！(重参数化技巧)
            action = dist.rsample()

        return action.squeeze(0), value.squeeze(-1)

class LiveRenderer:
    def __init__(self, N=64, L=0.3):
        self.N = N
        self.L = L
        self.MIN_H = 0.0001
        self.MAX_H = 0.01
        
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='black')
        self.ax1, self.ax2, self.ax3, self.ax4 = self.axes.flatten()

        extent = [0, self.L * 100, 0, self.L * 100]

        self.im1 = self.ax1.imshow(np.zeros((self.N, self.N)), cmap='magma', origin='lower', extent=extent, vmin=0, vmax=1)
        self.im2 = self.ax2.imshow(np.zeros((self.N, self.N)), cmap='viridis', origin='lower', extent=extent)
        self.im3 = self.ax3.imshow(np.zeros((self.N, self.N)), cmap='plasma', origin='lower', extent=extent, vmin=0, vmax=10)
        self.im4 = self.ax4.imshow(np.zeros((self.N, self.N)), cmap='inferno', origin='lower', extent=extent, vmin=0, vmax=1.1)

        self.ax1.set_facecolor('#00FF00')
        self.ax3.set_facecolor('#00FF00')

        coords = np.linspace(0, self.L * 100, self.N)
        self.X_grid, self.Y_grid = np.meshgrid(coords, coords)
        self.contour_set = None

        def add_colorbar(im, ax, label):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = self.fig.colorbar(im, cax=cax)
            cbar.set_label(label, color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        add_colorbar(self.im1, self.ax1, 'Pattern')
        add_colorbar(self.im2, self.ax2, 'Log Amp')
        add_colorbar(self.im3, self.ax3, 'Thick (mm)')
        add_colorbar(self.im4, self.ax4, 'Force')

        # ==========================================
        # 恢复：初始化 4 个固定的子图标题
        # ==========================================
        titles = ["Pattern (Target Overlay)", "Log Amplitude",
                  "Thickness (mm)", "Excitation Force"]
        for ax, title in zip(self.axes.flatten(), titles):
            ax.set_title(title, color='white', pad=10, fontsize=10)
            ax.tick_params(colors='white')

    def update(self, obs_tensor, infos, step):
        # 将 Tensor 剥离计算图并转为 numpy
        obs = obs_tensor.detach().cpu().numpy()[0]
        
        rho_np = obs[1]
        amp_norm = obs[2]
        force_np = obs[3]

        h_phys = rho_np * (self.MAX_H - self.MIN_H/2) + self.MIN_H/2
        material_mask = (h_phys > self.MIN_H).astype(np.float32)

        display = np.exp(-(amp_norm / 0.15)**2)
        masked_display = np.ma.masked_where(material_mask == 0, display)
        self.im1.set_data(masked_display)

        penalty_val = 1000.0
        effective_amp = amp_norm * material_mask + penalty_val * (1.0 - material_mask)
        self.im2.set_data(np.log10(effective_amp + 1e-12))
        self.im2.set_clim(np.min(np.log10(amp_norm + 1e-12)), 0.5)

        if self.contour_set:
            for c in self.contour_set.collections:
                c.remove()
        energy_field = amp_norm**2
        if np.max(energy_field) > 1e-9:
            threshold = np.percentile(energy_field[energy_field > 0], 10)
            self.contour_set = self.ax2.contour(self.X_grid, self.Y_grid, energy_field, levels=[threshold], colors='white', linewidths=1)

        masked_h = np.ma.masked_where(material_mask == 0, h_phys * 1000.0)
        self.im3.set_data(masked_h)
        self.im4.set_data(force_np)

        # ==========================================
        # 恢复：提取各项 loss 并在 ax1 显示详细状态
        # ==========================================
        # 安全地提取张量或浮点数值
        def get_val(key, default=0.0):
            val = infos.get(key, default)
            return val.item() if hasattr(val, 'item') else val

        total_l = get_val('loss', 0)
        l1 = get_val('l1', 0)
        l2 = get_val('l2', 0)
        l3 = get_val('l3', 0)
        l4 = get_val('l4', 0)
        freq = get_val('freq', 0)

        # 原汁原味的标题格式，带有换行和字体大小设定
        self.ax1.set_title(
            f"Step: {step} | freq: {freq:.2f} | Loss: {total_l:.2f}\n"
            f" l1:{l1:.3f} l2:{l2:.3f} l3:{l3:.3f} l4:{l4:.3f}",
            color='white', fontsize=9
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
def main():
    os.makedirs("./models", exist_ok=True)
    
    env = DifferentiableChladniEnv(N=64, control_N=8)
    agent = DifferentiableAgent().cuda()
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    renderer = LiveRenderer()
    
    # 两个关键长度
    max_ep_steps = 200 # 一个回合总共 200 步 (与你原来的设定一致)
    horizon = 16       # 每 16 步进行一次梯度截断更新
    gamma = 0.99
    
    total_steps = 0
    render_freq = 1

    print("开始可微强化学习 (SHAC) 训练...")
    
    for episode in range(10000):
        # 回合开始，真正地重置环境 (变回平坦的板子)
        obs = env.reset()
        step_in_ep = 0
        
        while step_in_ep < max_ep_steps:
            rewards = []
            values =[]
            
            # 动态计算当前块的长度 (最后一块可能不足 16 步)
            current_horizon = min(horizon, max_ep_steps - step_in_ep)
            
            # ==========================================
            # 1. 展开短时域 (走 16 步)
            # ==========================================
            for t in range(current_horizon):
                total_steps += 1
                step_in_ep += 1
                
                action, value = agent(obs)
                next_obs, reward, current_loss, infos = env.step(action)
                
                rewards.append(reward)
                values.append(value)
                
                obs = next_obs
                
                if total_steps % render_freq == 0:
                    renderer.update(obs, infos, total_steps)

            # ==========================================
            # 2. 计算这 16 步的损失
            # ==========================================
            _, final_value = agent(obs)
            discounted_return = final_value.detach()
            
            actor_loss = 0
            critic_loss = 0
            
            for i in reversed(range(current_horizon)):
                discounted_return = rewards[i] + gamma * discounted_return
                actor_loss -= discounted_return 
                critic_loss += F.mse_loss(values[i], discounted_return.detach())

            loss = (actor_loss + 0.5 * critic_loss) / current_horizon

            # ==========================================
            # 3. 伴随梯度反向传播
            # ==========================================
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
            # optimizer.step()
            has_nan = False
            for param in agent.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan = True
                    break
            
            if has_nan:
                print(f"⚠️ 警告: 步骤 {total_steps} 检测到梯度 NaN！跳过本次网络更新，并重置环境。")
                optimizer.zero_grad() # 清空有毒梯度
                obs = env.reset()     # 环境可能已经处于崩溃状态，直接重置
                step_in_ep = max_ep_steps # 强制结束当前 Episode
            else:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
                optimizer.step()
            
            # ==========================================
            # 4. 【核心！】切断计算图，但游戏继续
            # ==========================================
            # 剥离 obs 的计算图，防止下个块反向传播时追溯到这里
            obs = obs.detach()
            # 剥离环境内部状态的计算图
            env.detach_state() 

        # --- 一个回合 (200步) 结束 ---
        if episode % 10 == 0:
            print(f"Ep: {episode} | PhysLoss: {current_loss.item():.4f} | Freq: {env.force_freq.item():.1f}Hz")
            
        if episode > 0 and episode % 500 == 0:
            torch.save(agent.state_dict(), f"./models/shac_agent_ep{episode}.pth")
            
            # 保存这 200 步结束时雕刻出的最终厚度图
            rho_np = obs[0, 1].cpu().numpy()
            MIN_H, MAX_H = 0.0001, 0.01
            h_phys_np = rho_np * (MAX_H - MIN_H/2) + MIN_H/2
            np.save(f"./models/optimized_h_ep{episode}.npy", h_phys_np)
            print(f"模型及物理厚度图已保存至 ./models/ 目录")


if __name__ == "__main__":
    main()