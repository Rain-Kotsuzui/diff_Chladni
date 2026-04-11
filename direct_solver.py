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
    D = (p.E * hi**3.0) / (12.0 * (1.0 - p.nu**2.0))
    dx2 = p.dx * p.dx
    dy2 = p.dy * p.dy
    dxdy = p.dx * p.dy
    area = p.dx * p.dy
    
    base_ptr = idx * 100 

    # --- 质量  ---
    # Π_mass = -0.5 * ω² * ρ * h * w² * area
    # ∂²Π/∂w² = -ω² * ρ * h * area
    mass_val = - (p.omega**2.0) * p.rho * hi * area
    coo_rows[base_ptr] = idx
    coo_cols[base_ptr] = idx
    coo_vals_r[base_ptr] = mass_val
    
    entry_offset = int(1)

    # --- 刚度 ---
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