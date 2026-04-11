import warp as wp

@wp.kernel
def complex_dot_kernel(
    ar: wp.array(dtype=float), ai: wp.array(dtype=float),
    br: wp.array(dtype=float), bi: wp.array(dtype=float),
    out_r: wp.array(dtype=float), out_i: wp.array(dtype=float)
):
    idx = wp.tid()
    # (ar + i*ai) * (br - i*bi) = (ar*br + ai*bi) + i*(ai*br - ar*bi)
    r = ar[idx] * br[idx] + ai[idx] * bi[idx]
    i = ai[idx] * br[idx] - ar[idx] * bi[idx]
    wp.atomic_add(out_r, 0, r)
    wp.atomic_add(out_i, 0, i)

@wp.kernel
def complex_axpy_kernel(
    xr: wp.array(dtype=float), xi: wp.array(dtype=float),
    yr: wp.array(dtype=float), yi: wp.array(dtype=float),
    alphar: float, alphai: float,
    outr: wp.array(dtype=float), outi: wp.array(dtype=float)
):
    idx = wp.tid()
    # out = y + alpha * x
    # alpha * x = (alphar + i*alphai) * (xr + i*xi) = (alphar*xr - alphai*xi) + i*(alphar*xi + alphai*xr)
    tr = alphar * xr[idx] - alphai * xi[idx]
    ti = alphar * xi[idx] + alphai * xr[idx]
    outr[idx] = yr[idx] + tr
    outi[idx] = yi[idx] + ti