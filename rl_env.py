import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
import warp as wp
from parameter import PlateParams
from direct_solver import DifferentiableDirectSolver
from loss import ssim_loss, physics_informed_loss, quantile_physics_loss, void_protection_loss

wp.init()

class ChladniRLEnv(gym.Env):
    def __init__(self, target_image_path="target_processed.jpg", N=64, control_N=8):
        super(ChladniRLEnv, self).__init__()
        self.N = N
        self.control_N = control_N
        self.control_points = np.ones((self.control_N, self.control_N), dtype=np.float32) * 0.5

        self.max_steps = 200
        
        self.action_dim = self.control_N * self.control_N + 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # 观察空间: 4通道 [Target, Rho, Amp, Force]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4, N, N), dtype=np.float32)

        # 物理参数
        self.MIN_H, self.MAX_H = 0.0001, 0.01
        self.init_freq = 1250.0
        self.force_pos = np.array([0.25, 0.25], dtype=np.float32)
        
        self.p = PlateParams()
        self.p.E, self.p.nu, self.p.rho = 70.0e9, 0.33, 2700.0
        self.p.dx = self.p.dy = 0.3 / N
        self.p.omega = 2.0 * np.pi * self.init_freq
        self.solver = DifferentiableDirectSolver(self.p, N, N)
        self.fi_wp = wp.zeros(N * N, dtype=float, device="cuda")
        

        self._load_target(target_image_path)

    def _load_target(self, path):
        target_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if target_raw is None:
            raise FileNotFoundError(f"Target image not found: {path}")
            
        self.target_pattern = cv2.resize(target_raw, (self.N, self.N)).astype(np.float32)[::-1,:] / 255.0
        self.target_torch = torch.from_numpy(self.target_pattern.copy()).cuda().float().reshape(1,1,self.N,self.N)
        dist_map = ndimage.distance_transform_edt(self.target_pattern < 0.5)
        self.dist_map_torch = torch.from_numpy(dist_map / (np.max(dist_map)+1e-8)).cuda().float().reshape(1,1,self.N,self.N)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.control_points = np.ones((self.control_N, self.control_N), dtype=np.float32) * 0.5
        self.force_pos = np.array([0.25, 0.25], dtype=np.float32)
        self.force_freq = 1250.0
        obs, loss, _, _ = self._simulate_and_get_feedback()
        self.last_loss = loss
        return obs, {}

    def step(self, action):
        self.current_step += 1


        thickness_act = action[:64].reshape(8, 8)
        self.control_points = np.clip(self.control_points + thickness_act * 0.1, 0.0, 1.0)
        self.force_pos = np.clip(self.force_pos + action[64:66] * 0.2, 0.1, 0.9)
        self.force_freq = np.clip(self.force_freq + action[66] * 100.0, 50.0, 2000.0)

        obs, current_loss, is_invalid, loss_details = self._simulate_and_get_feedback()
        reward = (self.last_loss - current_loss) * 100.0
        self.last_loss = current_loss
        
        terminated = current_loss < 0.05
        truncated = self.current_step >= self.max_steps
        
        info = {"loss": current_loss, "freq": self.force_freq}
        info.update(loss_details)
        return obs, reward, terminated, truncated, info

    def _compute_loss(self, amp_torch, h_phys_tensor):
   
        target = self.target_torch
        h_2d = h_phys_tensor.reshape(1, 1, self.N, self.N)

        material_mask = torch.sigmoid((h_2d - self.MIN_H) * 1e10)
        
        penalty_value = 1000.0
        effective_amp = amp_torch * material_mask + penalty_value * (1.0 - material_mask)

        clip_max = torch.max(amp_torch).clamp(min=1e-10)
        display = torch.exp(-(effective_amp / (clip_max * 0.15 + 1e-12))**2)

        l1 = ssim_loss(display, target)
        
        
    
        l2 = physics_informed_loss(effective_amp,target)
        
        l3 = quantile_physics_loss(effective_amp, target, q=0.10)
        
        l4 = void_protection_loss(h_phys_tensor, target, self.MIN_H * 2) * 1e3

        total_loss = l1 + l2 + l3 + l4
        
        loss_details = {
            "l1": l1.item(),
            "l2": l2.item(),
            "l3": l3.item(),
            "l4": l4.item()
        }
        
        return total_loss.item(), loss_details
    
    def generate_gaussian_force(self,N, positions, sigma=0.02):
        x = torch.linspace(0, 1, N, device="cuda")
        y = torch.linspace(0, 1, N, device="cuda")
        Y, X = torch.meshgrid(y, x, indexing='ij')

        total_force = torch.zeros(N * N, device="cuda")

        for i in range(positions.shape[0]):
            dist_sq = (X - positions[i, 0])**2 + (Y - positions[i, 1])**2
            force = torch.exp(-dist_sq / (2 * sigma**2))*10.0
            total_force += force.flatten()
        return total_force

    def _simulate_and_get_feedback(self):
        with torch.no_grad():
            ctrl_t = torch.from_numpy(self.control_points).cuda().reshape(1,1,8,8)
            rho_t = F.interpolate(ctrl_t, size=(self.N, self.N), mode='bicubic', align_corners=False).clamp(0,1)
            h_phys_t = rho_t * (self.MAX_H - self.MIN_H/2) + self.MIN_H/2
            h_wp = wp.from_torch(h_phys_t.flatten())
            
            force_t = self.generate_gaussian_force(self.N,torch.from_numpy(self.force_pos).cuda().float().unsqueeze(0) , sigma=0.02)
            fr_wp = wp.from_numpy(force_t.cpu().numpy().flatten().astype(np.float32), device="cuda")

            
            w_c = self.solver.solve(h_wp, fr_wp, self.fi_wp)
            wr = torch.from_numpy(w_c.real).cuda().float()
            wi = torch.from_numpy(w_c.imag).cuda().float()

            amp_t = torch.sqrt(wr**2 + wi**2 + 1e-12).reshape(1, 1, self.N, self.N)
            loss_v, details = self._compute_loss(amp_t, h_phys_t)
            obs = np.stack([self.target_pattern, rho_t.cpu().numpy()[0,0], 
                                (amp_t/(amp_t.max()+1e-8)).cpu().numpy()[0,0], 
                                (force_t/(force_t.max()+1e-8)).cpu().numpy().reshape(64,64)])
            return obs, loss_v, False, details
          