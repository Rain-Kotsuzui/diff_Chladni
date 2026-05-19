import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rl_env import ChladniRLEnv


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(
                observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations): return self.linear(self.cnn(observations))


class RenderCallback(BaseCallback):
    def __init__(self, render_freq=20, N=64, L=0.3):
        super().__init__()
        self.render_freq = render_freq
        self.N = N
        self.L = L
        self.MIN_H = 0.0001
        self.MAX_H = 0.01
        self.fig = None
        self.contour_set = None

    def _on_training_start(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(
            2, 2, figsize=(10, 8), facecolor='black')
        self.ax1, self.ax2, self.ax3, self.ax4 = self.axes.flatten()

        # 0.3m -> 30cm
        extent = [0, self.L * 100, 0, self.L * 100]

        # 初始化 4 个图像句柄
        self.im1 = self.ax1.imshow(np.zeros(
            (self.N, self.N)), cmap='magma', origin='lower', extent=extent, vmin=0, vmax=1)
        self.im2 = self.ax2.imshow(
            np.zeros((self.N, self.N)), cmap='viridis', origin='lower', extent=extent)
        self.im3 = self.ax3.imshow(np.zeros(
            (self.N, self.N)), cmap='plasma', origin='lower', extent=extent, vmin=0, vmax=10)
        self.im4 = self.ax4.imshow(
            np.zeros((self.N, self.N)), cmap='inferno', origin='lower', extent=[0, self.L*100, 0, self.L*100], vmin=0, vmax=1.1)

        # 严格执行：空洞背景设为绿色
        self.ax1.set_facecolor('#00FF00')
        self.ax3.set_facecolor('#00FF00')

        # 准备等高线坐标网格
        coords = np.linspace(0, self.L * 100, self.N)
        self.X_grid, self.Y_grid = np.meshgrid(coords, coords)

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

        titles = ["Pattern (Target Overlay)", "Log Amplitude",
                  "Thickness (mm)", "Excitation Force"]
        for ax, title in zip(self.axes.flatten(), titles):
            ax.set_title(title, color='white', pad=10, fontsize=10)
            ax.tick_params(colors='white')

    def _on_step(self):
        if self.num_timesteps % self.render_freq == 0:
            # 获取环境数据
            # obs 结构: [Target, Rho, Amp_norm, Force]
            obs = self.locals['new_obs'][0]
            inf = self.locals['infos'][0]

            rho_np = obs[1]
            amp_norm = obs[2]  # 已经是归一化到 [0, 1] 的振幅
            force_np = obs[3]

            # 计算真实的物理厚度
            h_phys = rho_np * (self.MAX_H - self.MIN_H/2) + self.MIN_H/2

            # 1. 物理掩码逻辑
            # 厚度小于等于最小值的区域视为“空洞”
            material_mask = (h_phys > self.MIN_H).astype(np.float32)

            # 2. Pattern (im1) - 绿色背景掩码
            # 使用 0.15 系数使线条锐利
            display = np.exp(-(amp_norm / 0.15)**2)
            masked_display = np.ma.masked_where(material_mask == 0, display)
            self.im1.set_data(masked_display)

            # 3. Log Amplitude (im2) - 配合 Penalty 和等高线
            penalty_val = 1000.0
            # 无材料区施加高振幅惩罚，使节点线在对比中极度黑亮
            effective_amp = amp_norm * material_mask + \
                penalty_val * (1.0 - material_mask)
            self.im2.set_data(np.log10(effective_amp + 1e-12))
            self.im2.set_clim(
                np.min(np.log10(amp_norm + 1e-12)), 0.5)  # 限制上限突出细节

            # 绘制白色能量等高线 (节点区)
            if self.contour_set:
                for c in self.contour_set.collections:
                    c.remove()
            energy_field = amp_norm**2
            if np.max(energy_field) > 1e-9:
                threshold = np.percentile(energy_field[energy_field > 0], 10)
                self.contour_set = self.ax2.contour(self.X_grid, self.Y_grid, energy_field,
                                                    levels=[threshold], colors='white', linewidths=1)

            # 4. Thickness (im3) - 绿色背景掩码
            masked_h = np.ma.masked_where(material_mask == 0, h_phys * 1000.0)
            self.im3.set_data(masked_h)

            # 5. Force (im4)
            self.im4.set_data(force_np)

            # 更新标题文字
            total_l = inf.get('loss', 0)
            l1 = inf.get('l1', 0)
            l2 = inf.get('l2', 0)
            l3 = inf.get('l3', 0)
            l4 = inf.get('l4', 0)
            freq = inf.get('freq', 0)
            self.ax1.set_title(f"Step: {self.num_timesteps} | freq: {freq:.2f} | Loss: {total_l:.2f}\n l1:{l1:.3f} l2:{l2:.3f} l3:{l3:.3f} l4:{l4:.3f}",
                               color='white', fontsize=9)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        return True

    def _on_training_end(self):
        plt.ioff()
        plt.show()


class LogLossCallback(BaseCallback):
    def __init__(self, log_freq=10, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        infos = self.locals.get('infos')
        if infos:
            inf = infos[0]
            for key in ['l1', 'l2', 'l3', 'l4', 'loss']:
                if key in inf:
                    self.logger.record(f"losses/{key}", inf[key])

        return True
import os
def continue_training():
    env = ChladniRLEnv()
    
    checkpoint_path = "./models/rl_5000_steps.zip" 
    
    callbacks = [
        CheckpointCallback(save_freq=5000, save_path="./models/", name_prefix="rl"), 
        RenderCallback(1), 
        LogLossCallback()
    ]

    if os.path.exists(checkpoint_path):
        print(f"正在从存档加载模型: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env)
 

    model.learn(
        total_timesteps=500000, 
        callback=callbacks, 
        reset_num_timesteps=False
    )


def main():
    env = ChladniRLEnv()
    policy_kwargs = dict(features_extractor_class=CustomCNN, net_arch=[
                         dict(pi=[512, 512], vf=[512, 512])])

    model = PPO("CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                n_steps=50,
                tensorboard_log="./logs/")

    callbacks = [CheckpointCallback(
        5000, "./models/"), RenderCallback(1), LogLossCallback()]
    model.learn(total_timesteps=500000, callback=callbacks)


if __name__ == "__main__":
    main()
