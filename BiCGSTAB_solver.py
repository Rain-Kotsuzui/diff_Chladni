import warp as wp
import numpy as np
from energy import compute_energy_gradient_kernel
from complex import complex_dot_kernel, complex_axpy_kernel
from parameter import PlateParams
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, params, nx, ny):
        self.p = params
        self.nx = nx
        self.ny = ny
        self.n = nx * ny

        self.awr = wp.zeros(self.n, dtype=float, device="cuda")
        self.awi = wp.zeros(self.n, dtype=float, device="cuda")
        self.tr = wp.zeros(self.n, dtype=float, device="cuda")
        self.ti = wp.zeros(self.n, dtype=float, device="cuda")
        self.sr = wp.zeros(self.n, dtype=float, device="cuda")
        self.si = wp.zeros(self.n, dtype=float, device="cuda")

        
        self.diag_r = wp.zeros(self.n, dtype=float, device="cuda")
        self.diag_i = wp.zeros(self.n, dtype=float, device="cuda")
        
        self.tmp_r = wp.zeros(self.n, dtype=float, device="cuda")
        self.tmp_i = wp.zeros(self.n, dtype=float, device="cuda")

    
    def apply_A_preconditioned(self, wr, wi, h):
        """返回 M^-1 * A * w"""
        self.awr.zero_()
        self.awi.zero_()
        wp.launch(compute_energy_gradient_kernel, dim=self.n, 
                  inputs=[wr, wi, h, self.awr, self.awi, self.p, self.nx, self.ny], device="cuda")
        
        wp.launch(apply_jacobi_kernel, dim=self.n,
                  inputs=[self.awr, self.awi, self.diag_r, self.diag_i, self.tmp_r, self.tmp_i], device="cuda")
        
        return self.tmp_r, self.tmp_i
    
    def apply_A(self, wr, wi, h):
        self.awr.zero_()
        self.awi.zero_()
        # 计算A*w
        wp.launch(
            kernel=compute_energy_gradient_kernel,
            dim=self.n,
            inputs=[wr, wi, h, self.awr, self.awi, self.p, self.nx, self.ny],
            device="cuda"
        )
        return self.awr, self.awi

    def complex_dot(self, ar, ai, br, bi):
        res_r = wp.zeros(1, dtype=float, device="cuda")
        res_i = wp.zeros(1, dtype=float, device="cuda")
        wp.launch(complex_dot_kernel, dim=self.n, inputs=[ar, ai, br, bi, res_r, res_i], device="cuda")
        return complex(float(res_r.numpy()[0]), float(res_i.numpy()[0]))

    def solve(self, h, fr, fi, max_iters=1000, tol=1e-7):
        """BiCGSTAB"""
        # wp.launch(compute_diagonal_kernel, dim=self.n, 
        #           inputs=[h, self.diag_r, self.diag_i, self.p, self.nx, self.ny], device="cuda")

        # 1. 初始化 x0 = 0, r0 = f - A*x0 = f
        
        wr = wp.from_numpy(np.random.randn(self.n).astype(np.float32) * 1e-12, device="cuda")
        wi = wp.from_numpy(np.random.randn(self.n).astype(np.float32) * 1e-12, device="cuda")

        rr = wp.clone(fr); ri = wp.clone(fi)
        # wp.launch(apply_jacobi_kernel, dim=self.n, 
        #           inputs=[fr, fi, self.diag_r, self.diag_i, rr, ri], device="cuda")
        
        r0_hat_r = wp.clone(rr); r0_hat_i = wp.clone(ri)
        
        rho = 1.0 + 0.0j; alpha = 1.0 + 0.0j; omega = 1.0 + 0.0j
        vr = wp.zeros(self.n, dtype=float); vi = wp.zeros(self.n, dtype=float)
        pr = wp.zeros(self.n, dtype=float); pi = wp.zeros(self.n, dtype=float)

        # 初始范数用于检查收敛
        initial_norm = self.complex_norm_sq(rr, ri)
        tol_sq = tol ** 2 


        plt.ion()  # 开启交互模式
        fig_res, ax_res = plt.subplots(figsize=(8, 4))
        ax_res.set_yscale('log') # 残差通常跨越多个量级，使用对数坐标
        ax_res.set_title("BiCGSTAB Residual Convergence")
        ax_res.set_xlabel("Iterations")
        ax_res.set_ylabel("Residual Norm (Relative)")
        res_line, = ax_res.plot([], [], 'b-')
        res_history = []
        iter_history = []

        for k in range(max_iters):
            rho_prev = rho
            rho = self.complex_dot(r0_hat_r, r0_hat_i, rr, ri)
            
            if k == 0:
                wp.copy(pr, rr); wp.copy(pi, ri)
            else:
                beta = (rho / rho_prev) * (alpha / omega)
                wp.launch(complex_axpy_kernel, dim=self.n, inputs=[vr, vi, pr, pi, -omega.real, -omega.imag, self.tr, self.ti], device="cuda")
                wp.launch(complex_axpy_kernel, dim=self.n, inputs=[self.tr, self.ti, rr, ri, beta.real, beta.imag, pr, pi], device="cuda")

            # v = A * p
            # vr, vi = self.apply_A_preconditioned(pr, pi, h)
            vr, vi = self.apply_A(pr, pi, h)
            
            # alpha = rho / <r0_hat, v>
            v_dot = self.complex_dot(r0_hat_r, r0_hat_i, vr, vi)
            if abs(v_dot) < 1e-30:
                print("BiCGSTAB Breakdown: v_dot = 0")
                break
            alpha = rho / v_dot
            

            # s = r - alpha * v
            wp.launch(complex_axpy_kernel, dim=self.n, inputs=[vr, vi, rr, ri, -alpha.real, -alpha.imag, self.sr, self.si], device="cuda")
           
            # 检查 s 的范数 (提前退出)
            s_norm_sq = abs(self.complex_dot(self.sr, self.si, self.sr, self.si))
            if s_norm_sq < tol_sq * initial_norm:
                wp.launch(complex_axpy_kernel, dim=self.n, inputs=[pr, pi, wr, wi, alpha.real, alpha.imag, wr, wi], device="cuda")
                print(f"BiCGSTAB converged early at iter {k}")
                break

            # t = A * s
            self.tr, self.ti = self.apply_A(self.sr, self.si, h)

            
            # omega = <t, s> / <t, t>
            t_dot = self.complex_dot(self.tr, self.ti, self.tr, self.ti)
            if abs(t_dot) < 1e-30:
                omega = 0.0j
            else:
                omega = self.complex_dot(self.tr, self.ti, self.sr, self.si) / t_dot
            
            
            # x = x + alpha * p + omega * s
            wp.launch(complex_axpy_kernel, dim=self.n, inputs=[pr, pi, wr, wi, alpha.real, alpha.imag, wr, wi], device="cuda")
            wp.launch(complex_axpy_kernel, dim=self.n, inputs=[self.sr, self.si, wr, wi, omega.real, omega.imag, wr, wi], device="cuda")
            
            # r = s - omega * t
            wp.launch(complex_axpy_kernel, dim=self.n, inputs=[self.tr, self.ti, self.sr, self.si, -omega.real, -omega.imag, rr, ri], device="cuda")
            # 检查收敛
            curr_norm = abs(self.complex_dot(rr, ri, rr, ri))
            if curr_norm < tol_sq * initial_norm:
                print(f"BiCGSTAB converged at iter {k}")
                break
        
            if k % 100 == 0:
                curr_norm_sq = self.complex_norm_sq(rr, ri)
                curr_norm = np.sqrt(curr_norm_sq)
                rel_res = curr_norm / initial_norm
                
                print(f"Iter {k:5d} | Rel Residual: {rel_res:.6e}")
                
                # 更新绘图数据
                res_history.append(rel_res)
                iter_history.append(k)
                res_line.set_data(iter_history, res_history)
                
                # 自动缩放坐标轴
                ax_res.relim()
                ax_res.autoscale_view()
                fig_res.canvas.draw()
                fig_res.canvas.flush_events()
                plt.pause(0.01) # 必须有，让 GUI 线程处理渲染

                # 检查收敛
                if rel_res < tol:
                    print(f"BiCGSTAB converged at iter {k}")
                    break
                    
                # 如果残差爆炸，提前退出
                if rel_res > 1e10:
                    print("BiCGSTAB Diverged (Residual too large)")
                    break
        plt.ioff()
        return wr, wi



    def complex_norm_sq(self, r_r, r_i):
        res_r = wp.zeros(1, dtype=float, device="cuda")
        res_i = wp.zeros(1, dtype=float, device="cuda")
        wp.launch(complex_dot_kernel, dim=self.n, inputs=[r_r, r_i, r_r, r_i, res_r, res_i], device="cuda")

        return float(res_r.numpy()[0]) 

@wp.kernel
def compute_diagonal_kernel(
    h: wp.array(dtype=float),
    diag_r: wp.array(dtype=float), diag_i: wp.array(dtype=float),
    p: PlateParams, nx: int, ny: int
):
    idx = wp.tid()
    e_i = idx // ny
    e_j = idx % ny
    if e_i >= nx or e_j >= ny: return

    hi = h[idx]
    if hi < 1e-6:
        diag_r[idx] = 1.0e6
        diag_i[idx] = 0.0
        return

    D = (p.E * hi * hi * hi) / (12.0 * (1.0 - p.nu * p.nu))
    inv_dx2 = 1.0 / (p.dx * p.dx)
    inv_dy2 = 1.0 / (p.dy * p.dy)
    area = p.dx * p.dy

    k_diag = 4.0 * D * (inv_dx2 * inv_dx2 + inv_dy2 * inv_dy2 + 2.0 * p.nu * inv_dx2 * inv_dy2) * area
    
    mass_diag = - p.rho * hi * (p.omega * p.omega) * area

    # 包含阻尼的复数对角线
    # M_diag = (k_diag + mass_diag) + i * (p.eta * k_diag)
    diag_r[idx] = k_diag + mass_diag
    diag_i[idx] = p.eta * k_diag

@wp.kernel
def apply_jacobi_kernel(
    vr: wp.array(dtype=float), vi: wp.array(dtype=float),
    dr: wp.array(dtype=float), di: wp.array(dtype=float),
    out_r: wp.array(dtype=float), out_i: wp.array(dtype=float)
):
    idx = wp.tid()
    # 复数除法: (vr + i*vi) / (dr + i*di)
    denom = dr[idx] * dr[idx] + di[idx] * di[idx]
    if denom < 1e-12:
        out_r[idx] = vr[idx]
        out_i[idx] = vi[idx]
    else:
        out_r[idx] = (vr[idx] * dr[idx] + vi[idx] * di[idx]) / denom
        out_i[idx] = (vi[idx] * dr[idx] - vr[idx] * di[idx]) / denom