import numpy as np
import warp as wp
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from parameter import PlateParams

@wp.kernel
def assemble_stiffness_hessian_kernel(
    h: wp.array(dtype=float),
    p: PlateParams,
    nx: int, ny: int,
    coo_rows: wp.array(dtype=int),
    coo_cols: wp.array(dtype=int),
    coo_vals_r: wp.array(dtype=float),
    coo_vals_i: wp.array(dtype=float)
):
    idx = wp.tid()
    i = idx // ny
    j = idx % ny
    if i >= nx or j >= ny: return

    hi = h[idx]
    base_ptr = idx * 100 

    if hi < 0.0001:
        coo_rows[base_ptr] = idx
        coo_cols[base_ptr] = idx
        coo_vals_r[base_ptr] = 1.0
        coo_vals_i[base_ptr] = 0.0
        return
    
    D = (p.E * hi**3.0) / (12.0 * (1.0 - p.nu**2.0))
    dx2 = p.dx * p.dx
    dy2 = p.dy * p.dy
    dxdy = p.dx * p.dy
    area = p.dx * p.dy
    

    # 质量
    # Π_mass = -0.5 * ω² * ρ * h * w² * area
    # ∂²Π/∂w² = -ω² * ρ * h * area
    mass_val = - (p.omega**2.0) * p.rho * hi * area
    coo_rows[base_ptr] = idx
    coo_cols[base_ptr] = idx
    coo_vals_r[base_ptr] = mass_val
    
    entry_offset = int(1)

    # 刚度
    # 能量密度 e = 0.5 * D * [ (kxx+kyy)² - 2(1-ν)(kxx*kyy - kxy²) ]
    # 展开后 e = 0.5 * D * [ kxx² + kyy² + 2ν*kxx*kyy + 2(1-ν)kxy² ]
    
    # 预定义 3x3 邻域相对于中心 (i,j) 的局部权重
    # w_idx = [ (i-1,j-1), (i-1,j), (i-1,j+1), (i,j-1), (i,j), (i,j+1), (i+1,j-1), (i+1,j), (i+1,j+1) ]
    for mi in range(-1, 2):
        for mj in range(-1, 2):
            # 节点 m 的全局索引
            rm = i + mi
            cm = j + mj
            if rm < 0 or rm >= nx or cm < 0 or cm >= ny: continue
            idx_m = rm * ny + cm
            if h[idx_m] < 0.0001: continue

            # 计算节点 m 对该点(i,j)处曲率的贡献系数
            # kxx = (w_i+1 - 2w_i + w_i-1) / dx²
            m_kxx = 0.0
            if mj == 0:
                if mi == -1: m_kxx = 1.0/dx2
                elif mi == 0: m_kxx = -2.0/dx2
                elif mi == 1: m_kxx = 1.0/dx2
            
            # kyy = (w_j+1 - 2w_j + w_j-1) / dy²
            m_kyy = 0.0
            if mi == 0:
                if mj == -1: m_kyy = 1.0/dy2
                elif mj == 0: m_kyy = -2.0/dy2
                elif mj == 1: m_kyy = 1.0/dy2
                
            # kxy = (w_i+1,j+1 - w_i-1,j+1 - w_i+1,j-1 + w_i-1,j-1) / 4dxdy
            m_kxy = 0.0
            if mi == -1 and mj == -1: m_kxy = 1.0/(4.0*dxdy)
            elif mi == -1 and mj == 1: m_kxy = -1.0/(4.0*dxdy)
            elif mi == 1 and mj == -1: m_kxy = -1.0/(4.0*dxdy)
            elif mi == 1 and mj == 1: m_kxy = 1.0/(4.0*dxdy)

            for ni in range(-1, 2):
                for nj in range(-1, 2):
                    # 节点 n 的全局索引
                    rn = i + ni
                    cn = j + nj
                    if rn < 0 or rn >= nx or cn < 0 or cn >= ny: continue
                    idx_n = rn * ny + cn
                    
                    if h[idx_n] < 0.0001: continue
                    # 节点 n 对曲率的贡献
                    n_kxx = 0.0
                    if nj == 0:
                        if ni == -1: n_kxx = 1.0/dx2
                        elif ni == 0: n_kxx = -2.0/dx2
                        elif ni == 1: n_kxx = 1.0/dx2
                    
                    n_kyy = 0.0
                    if ni == 0:
                        if nj == -1: n_kyy = 1.0/dy2
                        elif nj == 0: n_kyy = -2.0/dy2
                        elif nj == 1: n_kyy = 1.0/dy2
                        
                    n_kxy = 0.0
                    if ni == -1 and nj == -1: n_kxy = 1.0/(4.0*dxdy)
                    elif ni == -1 and nj == 1: n_kxy = -1.0/(4.0*dxdy)
                    elif ni == 1 and nj == -1: n_kxy = -1.0/(4.0*dxdy)
                    elif ni == 1 and nj == 1: n_kxy = 1.0/(4.0*dxdy)

                    # 根据 Hessian ∂²e/∂wm∂wn 计算贡献
                    # A_mn += D * [ m_kxx*n_kxx + m_kyy*n_kyy + ν(m_kxx*n_kyy + m_kyy*n_kxx) + 2(1-ν)m_kxy*n_kxy ] * area
                    val = D * (
                        m_kxx * n_kxx + 
                        m_kyy * n_kyy + 
                        p.nu * (m_kxx * n_kyy + m_kyy * n_kxx) + 
                        2.0 * (1.0 - p.nu) * m_kxy * n_kxy
                    ) * area
                    
                    if entry_offset < 100:
                        pos = base_ptr + entry_offset
                        coo_rows[pos] = idx_m
                        coo_cols[pos] = idx_n
                        coo_vals_r[pos] = val
                        coo_vals_i[pos] = val * p.eta # 虚部阻尼
                        entry_offset += 1

@wp.kernel
def compute_grad_h_kernel(
    h: wp.array(dtype=float),
    wr: wp.array(dtype=float), wi: wp.array(dtype=float),
    lr: wp.array(dtype=float), li: wp.array(dtype=float),
    p: PlateParams,
    nx: int, ny: int,
    grad_h: wp.array(dtype=float)
):
    idx = wp.tid()
    i = idx // ny
    j = idx % ny
    if i >= nx or j >= ny: return

    hi = h[idx]
    if hi < 0.0001:
        grad_h[idx] = 0.0
        return
    
    # D = (E * h^3) / (12 * (1 - nu^2)) -> dD/dh = 3 * D / h
    dD_dh = (3.0 * p.E * hi**2.0) / (12.0 * (1.0 - p.nu**2.0))
    # dm/dh = rho * area
    dm_dh = p.rho * p.dx * p.dy
    area = p.dx * p.dy
    dx2, dy2, dxdy = p.dx**2.0, p.dy**2.0, p.dx*p.dy

    sum_val = 0.0

    # 质量项导数贡
    # d(Mass)/dh = -omega^2 * dm/dh
    # 贡献 = Re( conj(lambda_i) * (-omega^2 * dm_dh) * w_i )
    mass_grad_val = -(p.omega**2.0) * dm_dh
    # Re( (lr - i*li) * mass_grad_val * (wr + i*wi) )
    sum_val += (lr[idx] * wr[idx] + li[idx] * wi[idx]) * mass_grad_val

    # 刚度项导数
    for mi in range(-1, 2):
        for mj in range(-1, 2):
            rm, cm = i + mi, j + mj
            if rm < 0 or rm >= nx or cm < 0 or cm >= ny: continue
            idx_m = rm * ny + cm
            
            m_kxx = 0.0
            if mj == 0:
                if mi == -1: m_kxx = 1.0/dx2
                elif mi == 0: m_kxx = -2.0/dx2
                elif mi == 1: m_kxx = 1.0/dx2
            m_kyy = 0.0
            if mi == 0:
                if mj == -1: m_kyy = 1.0/dy2
                elif mj == 0: m_kyy = -2.0/dy2
                elif mj == 1: m_kyy = 1.0/dy2
            m_kxy = 0.0
            if mi == -1 and mj == -1: m_kxy = 1.0/(4.0*dxdy)
            elif mi == -1 and mj == 1: m_kxy = -1.0/(4.0*dxdy)
            elif mi == 1 and mj == -1: m_kxy = -1.0/(4.0*dxdy)
            elif mi == 1 and mj == 1: m_kxy = 1.0/(4.0*dxdy)

            for ni in range(-1, 2):
                for nj in range(-1, 2):
                    rn, cn = i + ni, j + nj
                    if rn < 0 or rn >= nx or cn < 0 or cn >= ny: continue
                    idx_n = rn * ny + cn

                    n_kxx = 0.0
                    if nj == 0:
                        if ni == -1: n_kxx = 1.0/dx2
                        elif ni == 0: n_kxx = -2.0/dx2
                        elif ni == 1: n_kxx = 1.0/dx2
                    n_kyy = 0.0
                    if ni == 0:
                        if nj == -1: n_kyy = 1.0/dy2
                        elif nj == 0: n_kyy = -2.0/dy2
                        elif nj == 1: n_kyy = 1.0/dy2
                    n_kxy = 0.0
                    if ni == -1 and nj == -1: n_kxy = 1.0/(4.0*dxdy)
                    elif ni == -1 and nj == 1: n_kxy = -1.0/(4.0*dxdy)
                    elif ni == 1 and nj == -1: n_kxy = -1.0/(4.0*dxdy)
                    elif ni == 1 and nj == 1: n_kxy = 1.0/(4.0*dxdy)

                    # 基础刚度因子 (不含 D)
                    stiff_factor = (m_kxx * n_kxx + m_kyy * n_kyy + p.nu * (m_kxx * n_kyy + m_kyy * n_kxx) + 2.0 * (1.0 - p.nu) * m_kxy * n_kxy) * area
                    
                    # 包含导数 dD/dh 和 复数阻尼 (1 + i*eta)
                    # dK_mn/dh = dD_dh * stiff_factor * (1 + i*eta)
                    dk_real = dD_dh * stiff_factor
                    dk_imag = dk_real * p.eta
                    
                    # 计算伴随项贡献: Re( conj(lambda_m) * dK_mn/dh * w_n )
                    # (lr - i*li) * (dk_real + i*dk_imag) * (wr + i*wi)
                    # 实部 = lr*(dk_real*wr - dk_imag*wi) + li*(dk_real*wi + dk_imag*wr)
                    term_re = lr[idx_m] * (dk_real * wr[idx_n] - dk_imag * wi[idx_n]) + \
                              li[idx_m] * (dk_real * wi[idx_n] + dk_imag * wr[idx_n])
                    
                    sum_val += term_re

    grad_h[idx] = -sum_val

class DifferentiableDirectSolver:
    def __init__(self, params, nx, ny):
        self.params = params
        self.nx, self.ny = nx, ny
        self.n = nx * ny
        self.entries_per_node = 100
        
        self.rows = wp.zeros(self.n * self.entries_per_node, dtype=int)
        self.cols = wp.zeros(self.n * self.entries_per_node, dtype=int)
        self.vals_r = wp.zeros(self.n * self.entries_per_node, dtype=float)
        self.vals_i = wp.zeros(self.n * self.entries_per_node, dtype=float)
        
        self.last_A = None # 缓存矩阵用于反向传播

    def solve(self, h_wp, fr_wp, fi_wp):
        self.rows.fill_(-1)
        self.cols.fill_(-1)
        self.vals_r.zero_()
        self.vals_i.zero_()

        wp.launch(
            kernel=assemble_stiffness_hessian_kernel,
            dim=self.n,
            inputs=[h_wp, self.params, self.nx, self.ny, 
                    self.rows, self.cols, self.vals_r, self.vals_i],
            device="cuda"
        )
        wp.synchronize()

        r_np = self.rows.numpy()
        c_np = self.cols.numpy()
        vr_np = self.vals_r.numpy()
        vi_np = self.vals_i.numpy()
            
        mask = (r_np >= 0)
        A_sparse = sp.coo_matrix(
            (vr_np[mask] + 1j * vi_np[mask], (r_np[mask], c_np[mask])), 
            shape=(self.n, self.n)
        ).tocsr()
        diag = A_sparse.diagonal()
        zero_diag_mask = np.abs(diag) < 0.0001
        if np.any(zero_diag_mask):
            patch = sp.diags(zero_diag_mask.astype(float), 0, shape=(self.n, self.n), format='csr')
            A_sparse = A_sparse + patch

        self.last_A = A_sparse 
    
        f_complex = fr_wp.numpy() + 1j * fi_wp.numpy()
        h_np = h_wp.numpy()

        f_complex[h_np < 0.0001] = 0.0 


        if np.any(zero_diag_mask):
            f_complex[zero_diag_mask] = 0.0
        
        
        w_complex = spla.spsolve(self.last_A, f_complex)
        
        return w_complex

    def solve_adjoint(self, adjoint_force_complex):
        """
        解伴随方程 K^T * lambda = adjoint_force_complex
        """
        rhs = np.conj(adjoint_force_complex)
        lam_complex = spla.spsolve(self.last_A, rhs)
        return lam_complex
    
    def compute_adjoint_gradient(self, h_wp, w_complex, grad_w_complex):
        """
        grad_w_complex: dL/dw (复数向量)
        """
        if self.last_A is None:
            raise RuntimeError("必须先调用 solve 才能计算梯度")
        
        # 求解伴随方程 K^T * lambda = grad_w_complex
        # 这里采用：K * lambda = conj(grad_w)
        lam_complex = self.solve_adjoint(grad_w_complex)
        
        wr_wp = wp.from_numpy(w_complex.real.astype(np.float32), device="cuda")
        wi_wp = wp.from_numpy(w_complex.imag.astype(np.float32), device="cuda")
        lr_wp = wp.from_numpy(lam_complex.real.astype(np.float32), device="cuda")
        li_wp = wp.from_numpy(lam_complex.imag.astype(np.float32), device="cuda")
        grad_h_wp = wp.zeros(self.n, dtype=float, device="cuda")

        wp.launch(
            kernel=compute_grad_h_kernel,
            dim=self.n,
            inputs=[h_wp, wr_wp, wi_wp, lr_wp, li_wp, self.params, self.nx, self.ny, grad_h_wp],
            device="cuda"
        )
        
        grad_fr_np = np.real(lam_complex).astype(np.float32)
        area = self.params.dx * self.params.dy
        h_np = h_wp.numpy()
        
        # M_ii = -omega^2 * rho * h * area
        # dK/domega = -2 * omega * rho * h * area

        dK_domega_diag = -2.0 * self.params.omega * self.params.rho * h_np * area
        
        # dL/domega = Re( lambda^H * (dK/domega * w) )
        grad_omega = np.real(np.sum(np.conj(lam_complex) * dK_domega_diag * w_complex))

        return grad_h_wp.numpy(), grad_fr_np, grad_omega



class DirectSolver:
    def __init__(self, params, nx, ny):
        self.params = params
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.entries_per_node = 100 
        
        self.rows = wp.zeros(self.n * self.entries_per_node, dtype=int)
        self.cols = wp.zeros(self.n * self.entries_per_node, dtype=int)
        self.vals_r = wp.zeros(self.n * self.entries_per_node, dtype=float)
        self.vals_i = wp.zeros(self.n * self.entries_per_node, dtype=float)

    def solve(self, h_wp, fr_wp, fi_wp):
        self.rows.fill_(-1)
        self.cols.fill_(-1)
        self.vals_r.zero_()
        self.vals_i.zero_()

        wp.launch(
            kernel=assemble_stiffness_hessian_kernel,
            dim=self.n,
            inputs=[h_wp, self.params, self.nx, self.ny, 
                    self.rows, self.cols, self.vals_r, self.vals_i],
            device="cuda"
        )
        wp.synchronize()

        r_np = self.rows.numpy()
        c_np = self.cols.numpy()
        vr_np = self.vals_r.numpy()
        vi_np = self.vals_i.numpy()
            
        mask = (r_np >= 0)
        A_sparse = sp.coo_matrix(
            (vr_np[mask] + 1j * vi_np[mask], (r_np[mask], c_np[mask])), 
            shape=(self.n, self.n)
        ).tocsr()
        diag = A_sparse.diagonal()
        zero_diag_mask = np.abs(diag) < 1e-12
        if np.any(zero_diag_mask):
            patch = sp.diags(zero_diag_mask.astype(float), 0, shape=(self.n, self.n), format='csr')
            A_sparse = A_sparse + patch

        f_complex = fr_wp.numpy() + 1j * fi_wp.numpy()

        if np.any(zero_diag_mask):
            f_complex[zero_diag_mask] = 0.0

        w_complex = spla.spsolve(A_sparse, f_complex)
        
        return wp.from_numpy(w_complex.real.astype(np.float32), device="cuda"), \
               wp.from_numpy(w_complex.imag.astype(np.float32), device="cuda")