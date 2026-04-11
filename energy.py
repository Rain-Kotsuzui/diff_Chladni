import warp as wp
from parameter import PlateParams


@wp.func
def get_index(i: int, j: int, nx: int, ny: int):
    if i < 0 or i >= nx or j < 0 or j >= ny:
        return -1
    return i * ny + j


@wp.func
def fetch_w(r: int, c: int, arr: wp.array(dtype=float), nx: int, ny: int):
    if r < 0:
        return 2.0 * arr[0 * ny + c] - arr[1 * ny + c]
    if r >= nx:
        return 2.0 * arr[(nx - 1) * ny + c] - arr[(nx - 2) * ny + c]
    if c < 0:
        return 2.0 * arr[r * ny + 0] - arr[r * ny + 1]
    if c >= ny:
        return 2.0 * arr[r * ny + (ny - 1)] - arr[r * ny + (ny - 2)]
    return arr[r * ny + c]

@wp.kernel
def compute_energy_gradient_kernel(
    wr: wp.array(dtype=float), wi: wp.array(dtype=float),
    h: wp.array(dtype=float),
    awr: wp.array(dtype=float), awi: wp.array(dtype=float),
    p: PlateParams, nx: int, ny: int
):
    idx = wp.tid()
    e_i = idx // ny
    e_j = idx % ny
    if e_i >= nx or e_j >= ny:
        return

    idx = e_i * ny + e_j

    hi = h[idx]
    if hi < 1e-6:
        wp.atomic_add(awr, idx, wr[idx] * 1.0e6)
        wp.atomic_add(awi, idx, wi[idx] * 1.0e6)
        return

    D = (p.E * hi * hi * hi) / (12.0 * (1.0 - p.nu * p.nu))
    inv_dx2 = 1.0 / (p.dx * p.dx)
    inv_dy2 = 1.0 / (p.dy * p.dy)
    inv_4dxdy = 1.0 / (4.0 * p.dx * p.dy)
    area = p.dx * p.dy

    w_cc = wr[idx]; wi_cc = wi[idx]

    w_l = fetch_w(e_i-1, e_j, wr,nx, ny);  wi_l =  fetch_w(e_i-1, e_j, wi,nx, ny)
    w_r =  fetch_w(e_i+1, e_j, wr,nx, ny);  wi_r =  fetch_w(e_i+1, e_j, wi,nx, ny)
    w_u =  fetch_w(e_i, e_j+1, wr,nx, ny);  wi_u =  fetch_w(e_i, e_j+1, wi,nx, ny)
    w_d =  fetch_w(e_i, e_j-1, wr,nx, ny);  wi_d =  fetch_w(e_i, e_j-1, wi,nx, ny)
    
    w_lu =  fetch_w(e_i-1, e_j+1, wr,nx, ny); wi_lu =  fetch_w(e_i-1, e_j+1, wi,nx, ny)
    w_ru =  fetch_w(e_i+1, e_j+1, wr,nx, ny); wi_ru =  fetch_w(e_i+1, e_j+1, wi,nx, ny)
    w_ld =  fetch_w(e_i-1, e_j-1, wr,nx, ny); wi_ld =  fetch_w(e_i-1, e_j-1, wi,nx, ny)
    w_rd =  fetch_w(e_i+1, e_j-1, wr,nx, ny); wi_rd =  fetch_w(e_i+1, e_j-1, wi,nx, ny)

    kxx_r = (w_r - 2.0 * w_cc + w_l) * inv_dx2
    kyy_r = (w_u - 2.0 * w_cc + w_d) * inv_dy2
    kxy_r = (w_ru - w_lu - w_rd + w_ld) * inv_4dxdy

    kxx_i = (wi_r - 2.0 * wi_cc + wi_l) * inv_dx2
    kyy_i = (wi_u - 2.0 * wi_cc + wi_d) * inv_dy2
    kxy_i = (wi_ru - wi_lu - wi_rd + wi_ld) * inv_4dxdy

    mx_r = D * ((kxx_r + p.nu * kyy_r) - p.eta * (kxx_i + p.nu * kyy_i))
    mx_i = D * ((kxx_i + p.nu * kyy_i) + p.eta * (kxx_r + p.nu * kyy_r))
    my_r = D * ((kyy_r + p.nu * kxx_r) - p.eta * (kyy_i + p.nu * kxx_i))
    my_i = D * ((kyy_i + p.nu * kxx_i) + p.eta * (kyy_r + p.nu * kxx_r))
    mxy_r = D * (1.0 - p.nu) * (kxy_r - p.eta * kxy_i)
    mxy_i = D * (1.0 - p.nu) * (kxy_i + p.eta * kxy_r)

    wp.atomic_add(awr, idx, (mx_r * -2.0 * inv_dx2 + my_r * -2.0 * inv_dy2) * area)
    wp.atomic_add(awi, idx, (mx_i * -2.0 * inv_dx2 + my_i * -2.0 * inv_dy2) * area)

    if e_i > 0:
        wp.atomic_add(awr, idx - ny, mx_r * inv_dx2 * area)
        wp.atomic_add(awi, idx - ny, mx_i * inv_dx2 * area)
    if e_i < nx - 1:
        wp.atomic_add(awr, idx + ny, mx_r * inv_dx2 * area)
        wp.atomic_add(awi, idx + ny, mx_i * inv_dx2 * area)
    if e_j > 0:
        wp.atomic_add(awr, idx - 1, my_r * inv_dy2 * area)
        wp.atomic_add(awi, idx - 1, my_i * inv_dy2 * area)
    if e_j < ny - 1:
        wp.atomic_add(awr, idx + 1, my_r * inv_dy2 * area)
        wp.atomic_add(awi, idx + 1, my_i * inv_dy2 * area)

    # 混合项 Mxy 贡献
    mxy_w = 2.0 * inv_4dxdy * area
    if e_i < nx-1 and e_j < ny-1: wp.atomic_add(awr, idx+ny+1, mxy_r * mxy_w); wp.atomic_add(awi, idx+ny+1, mxy_i * mxy_w)
    if e_i > 0 and e_j < ny-1:    wp.atomic_add(awr, idx-ny+1, mxy_r * -mxy_w); wp.atomic_add(awi, idx-ny+1, mxy_i * -mxy_w)
    if e_i < nx-1 and e_j > 0:    wp.atomic_add(awr, idx+ny-1, mxy_r * -mxy_w); wp.atomic_add(awi, idx+ny-1, mxy_i * -mxy_w)
    if e_i > 0 and e_j > 0:       wp.atomic_add(awr, idx-ny-1, mxy_r * mxy_w); wp.atomic_add(awi, idx-ny-1, mxy_i * mxy_w)
    
    # has_l = (e_i > 0)
    # has_r = (e_i < nx - 1)
    # has_u = (e_j < ny - 1)
    # has_d = (e_j > 0)


    # w_cc = wr[idx]
    # wi_cc = wi[idx]

    # if has_l and has_r:
    #     wl = wr[idx - ny]; wr_val = wr[idx + ny]
    #     wil = wi[idx - ny]; wir = wi[idx + ny]
        
    #     kxx_r = (wr_val - 2.0 * w_cc + wl) * inv_dx2
    #     kxx_i = (wir - 2.0 * wi_cc + wil) * inv_dx2
        
    #     kyy_coupled_r = 0.0; kyy_coupled_i = 0.0
    #     if has_u and has_d:
    #         kyy_coupled_r = (wr[idx + 1] - 2.0 * w_cc + wr[idx - 1]) * inv_dy2
    #         kyy_coupled_i = (wi[idx + 1] - 2.0 * wi_cc + wi[idx - 1]) * inv_dy2
            
    #     mx_r = D * ((kxx_r + p.nu * kyy_coupled_r) - p.eta * (kxx_i + p.nu * kyy_coupled_i))
    #     mx_i = D * ((kxx_i + p.nu * kyy_coupled_i) + p.eta * (kxx_r + p.nu * kyy_coupled_r))
        
    #     # 分发梯度
    #     wp.atomic_add(awr, idx, mx_r * (-2.0 * inv_dx2) * area)
    #     wp.atomic_add(awi, idx, mx_i * (-2.0 * inv_dx2) * area)
    #     wp.atomic_add(awr, idx - ny, mx_r * inv_dx2 * area)
    #     wp.atomic_add(awi, idx - ny, mx_i * inv_dx2 * area)
    #     wp.atomic_add(awr, idx + ny, mx_r * inv_dx2 * area)
    #     wp.atomic_add(awi, idx + ny, mx_i * inv_dx2 * area)

    # if has_u and has_d:
    #     wu = wr[idx + 1]; wd = wr[idx - 1]
    #     wiu = wi[idx + 1]; wid = wi[idx - 1]
        
    #     kyy_r = (wu - 2.0 * w_cc + wd) * inv_dy2
    #     kyy_i = (wiu - 2.0 * wi_cc + wid) * inv_dy2
        
    #     kxx_coupled_r = 0.0; kxx_coupled_i = 0.0
    #     if has_l and has_r:
    #         kxx_coupled_r = (wr[idx + ny] - 2.0 * w_cc + wr[idx - ny]) * inv_dx2
    #         kxx_coupled_i = (wi[idx + ny] - 2.0 * wi_cc + wi[idx - ny]) * inv_dx2
            
    #     my_r = D * ((kyy_r + p.nu * kxx_coupled_r) - p.eta * (kyy_i + p.nu * kxx_coupled_i))
    #     my_i = D * ((kyy_i + p.nu * kxx_coupled_i) + p.eta * (kyy_r + p.nu * kxx_coupled_r))
        
    #     wp.atomic_add(awr, idx, my_r * (-2.0 * inv_dy2) * area)
    #     wp.atomic_add(awi, idx, my_i * (-2.0 * inv_dy2) * area)
    #     wp.atomic_add(awr, idx + 1, my_r * inv_dy2 * area)
    #     wp.atomic_add(awi, idx + 1, my_i * inv_dy2 * area)
    #     wp.atomic_add(awr, idx - 1, my_r * inv_dy2 * area)
    #     wp.atomic_add(awi, idx - 1, my_i * inv_dy2 * area)

    # if has_l and has_r and has_u and has_d:
    #     w_ru = wr[idx + ny + 1]; w_lu = wr[idx - ny + 1]
    #     w_rd = wr[idx + ny - 1]; w_ld = wr[idx - ny - 1]
    #     wi_ru = wi[idx + ny + 1]; wi_lu = wi[idx - ny + 1]
    #     wi_rd = wi[idx + ny - 1]; wi_ld = wi[idx - ny - 1]
        
    #     kxy_r = (w_ru - w_lu - w_rd + w_ld) * inv_4dxdy
    #     kxy_i = (wi_ru - wi_lu - wi_rd + wi_ld) * inv_4dxdy
        
    #     mxy_r = D * (1.0 - p.nu) * (kxy_r - p.eta * kxy_i)
    #     mxy_i = D * (1.0 - p.nu) * (kxy_i + p.eta * kxy_r)
        
    #     mxy_weight = 2.0 * inv_4dxdy * area # 2.0 来自能量泛函中的 2*Mxy*kxy
    #     wp.atomic_add(awr, idx + ny + 1, mxy_r * mxy_weight)
    #     wp.atomic_add(awi, idx + ny + 1, mxy_i * mxy_weight)
    #     wp.atomic_add(awr, idx - ny + 1, mxy_r * -mxy_weight)
    #     wp.atomic_add(awi, idx - ny + 1, mxy_i * -mxy_weight)
    #     wp.atomic_add(awr, idx + ny - 1, mxy_r * -mxy_weight)
    #     wp.atomic_add(awi, idx + ny - 1, mxy_i * -mxy_weight)
    #     wp.atomic_add(awr, idx - ny - 1, mxy_r * mxy_weight)
    #     wp.atomic_add(awi, idx - ny - 1, mxy_i * mxy_weight)

    # # 实部采样
    # w_cc = wr[idx]
    # w_l = fetch_w(e_i-1, e_j,w_cc, wr,nx,ny)
    # w_r = fetch_w(e_i+1, e_j,w_cc, wr,nx,ny)
    # w_u = fetch_w(e_i, e_j+1,w_cc, wr,nx,ny)
    # w_d = fetch_w(e_i, e_j-1,w_cc, wr,nx,ny)
    # w_lu = fetch_w(e_i-1, e_j+1,w_cc, wr,nx,ny)
    # w_ru = fetch_w(e_i+1, e_j+1,w_cc, wr,nx,ny)
    # w_ld = fetch_w(e_i-1, e_j-1,w_cc, wr,nx,ny)
    # w_rd = fetch_w(e_i+1, e_j-1,w_cc, wr,nx,ny)

    # # 虚部采样
    # wi_cc = wi[idx]
    # wi_l = fetch_w(e_i-1, e_j,wi_cc,  wi,nx,ny)
    # wi_r = fetch_w(e_i+1, e_j,wi_cc,  wi,nx,ny)
    # wi_u = fetch_w(e_i, e_j+1,wi_cc,  wi,nx,ny)
    # wi_d = fetch_w(e_i, e_j-1,wi_cc,  wi,nx,ny)
    # wi_lu = fetch_w(e_i-1, e_j+1,wi_cc,  wi,nx,ny)
    # wi_ru = fetch_w(e_i+1, e_j+1,wi_cc,  wi,nx,ny)
    # wi_ld = fetch_w(e_i-1, e_j-1,wi_cc,  wi,nx,ny)
    # wi_rd = fetch_w(e_i+1, e_j-1,wi_cc,  wi,nx,ny)

    # kxx_r = (w_r - 2.0*w_cc + w_l) * inv_dx2
    # kyy_r = (w_u - 2.0*w_cc + w_d) * inv_dy2
    # kxy_r = (w_ru - w_lu - w_rd + w_ld) * inv_4dxdy

    # kxx_i = (wi_r - 2.0*wi_cc + wi_l) * inv_dx2
    # kyy_i = (wi_u - 2.0*wi_cc + wi_d) * inv_dy2
    # kxy_i = (wi_ru - wi_lu - wi_rd + wi_ld) * inv_4dxdy

    # 根据 D_complex = D * (1 + i*eta)
    # M_complex = D_complex * Curvature_complex
    # M_real = D * (k_r - eta * k_i)
    # M_imag = D * (k_i + eta * k_r)

    # # Mx
    # mx_r = D * ((kxx_r + p.nu * kyy_r) - p.eta * (kxx_i + p.nu * kyy_i))
    # mx_i = D * ((kxx_i + p.nu * kyy_i) + p.eta * (kxx_r + p.nu * kyy_r))
    
    # # My
    # my_r = D * ((kyy_r + p.nu * kxx_r) - p.eta * (kyy_i + p.nu * kxx_i))
    # my_i = D * ((kyy_i + p.nu * kxx_i) + p.eta * (kyy_r + p.nu * kxx_r))
    
    # # Mxy
    # mxy_r = D * (1.0 - p.nu) * (kxy_r - p.eta * kxy_i)
    # mxy_i = D * (1.0 - p.nu) * (kxy_i + p.eta * kxy_r)

    # # 分发刚度梯度到 9 个邻域节点 ---
    # # 使用中心差分模板的权重分发梯度
    # # dU/dw_cc = mx*(-2/dx2) + my*(-2/dy2)
    # wp.atomic_add(awr, idx, (mx_r * (-2.0 * inv_dx2) + my_r * (-2.0 * inv_dy2)) * area)
    # wp.atomic_add(awi, idx, (mx_i * (-2.0 * inv_dx2) + my_i * (-2.0 * inv_dy2)) * area)

    # # 左右节点 (受 kxx 影响)
    # l_idx = get_index(e_i-1, e_j, nx, ny)
    # if l_idx != -1:
    #     wp.atomic_add(awr, l_idx, (mx_r * inv_dx2) * area)
    #     wp.atomic_add(awi, l_idx, (mx_i * inv_dx2) * area)
    
    # r_idx = get_index(e_i+1, e_j, nx, ny)
    # if r_idx != -1:
    #     wp.atomic_add(awr, r_idx, (mx_r * inv_dx2) * area)
    #     wp.atomic_add(awi, r_idx, (mx_i * inv_dx2) * area)

    # # 上下节点 (受 kyy 影响)
    # u_idx = get_index(e_i, e_j+1, nx, ny)
    # if u_idx != -1:
    #     wp.atomic_add(awr, u_idx, (my_r * inv_dy2) * area)
    #     wp.atomic_add(awi, u_idx, (my_i * inv_dy2) * area)

    # d_idx = get_index(e_i, e_j-1, nx, ny)
    # if d_idx != -1:
    #     wp.atomic_add(awr, d_idx, (my_r * inv_dy2) * area)
    #     wp.atomic_add(awi, d_idx, (my_i * inv_dy2) * area)

    # # 四角节点 (受 kxy 影响)
    # # 权重分别为: ru(+), lu(-), rd(-), ld(+) 乘以系数 2.0 (因为 Mxy 贡献 2倍能量)
    # mxy_weight = 2.0 * inv_4dxdy * area
    
    # idx_ru = get_index(e_i+1, e_j+1, nx, ny)
    # if idx_ru != -1:
    #     wp.atomic_add(awr, idx_ru, (mxy_r * mxy_weight))
    #     wp.atomic_add(awi, idx_ru, (mxy_i * mxy_weight))

    # idx_lu = get_index(e_i-1, e_j+1, nx, ny)
    # if idx_lu != -1:
    #     wp.atomic_add(awr, idx_lu, (mxy_r * -mxy_weight))
    #     wp.atomic_add(awi, idx_lu, (mxy_i * -mxy_weight))

    # idx_rd = get_index(e_i+1, e_j-1, nx, ny)
    # if idx_rd != -1:
    #     wp.atomic_add(awr, idx_rd, (mxy_r * -mxy_weight))
    #     wp.atomic_add(awi, idx_rd, (mxy_i * -mxy_weight))

    # idx_ld = get_index(e_i-1, e_j-1, nx, ny)
    # if idx_ld != -1:
    #     wp.atomic_add(awr, idx_ld, (mxy_r * mxy_weight))
    #     wp.atomic_add(awi, idx_ld, (mxy_i * mxy_weight))
    
    # 动能项贡献-
    # dT/dw = rho * h * omega^2 * w
    # 注意：动能项是在节点上直接计算，不涉及邻域
    inertia_r = - p.rho * hi * (p.omega * p.omega) * w_cc * area
    inertia_i = - p.rho * hi * (p.omega * p.omega) * wi_cc * area
    
    wp.atomic_add(awr, idx, inertia_r)
    wp.atomic_add(awi, idx, inertia_i)
