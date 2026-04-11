import warp as wp
import numpy as np


wp.init()

@wp.struct
class PlateParams:
    E: float
    nu: float
    rho: float

    L: float      # 板长
    
    N: int        # 网格数量
    dx: float     # 网格间距
    dy: float     
    eta: float    # 阻尼

    omega: float # 激励频率
