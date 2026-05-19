"""
设备管理器 - 统一管理PyTorch和Warp的设备后端
"""
import torch
import warp as wp

def get_torch_device():
    """获取最优的PyTorch计算设备"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_warp_device():
    """获取Warp设备 - 当前只支持CPU（Mac Metal不支持）"""
    # 检查CUDA
    if wp.is_cuda_available():
        return "cuda"
    # 当前Warp Mac版本不支持Metal，只能用CPU
    return "cpu"

def init_devices():
    """初始化所有设备"""
    # 初始化Warp
    wp.init()
    
    torch_device = get_torch_device()
    warp_device = get_warp_device()
    
    print(f"=== 设备初始化 ===")
    print(f"PyTorch 设备: {torch_device}")
    print(f"Warp    设备: {warp_device}")
    
    return torch_device, warp_device

# 全局设备
TORCH_DEVICE = get_torch_device()
WARP_DEVICE = get_warp_device()

def torch_to_warp_device(torch_dev):
    """将PyTorch设备转换为Warp设备字符串"""
    if torch_dev.type == "cuda":
        return "cuda"
    return "cpu"

def warp_to_torch_device(warp_dev):
    """将Warp设备转换为PyTorch设备"""
    if warp_dev == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")

# 便捷函数
def to_device(tensor):
    """将tensor移到PyTorch设备"""
    if isinstance(tensor, torch.Tensor):
        return tensor.to(TORCH_DEVICE)
    return tensor

def synchronize():
    """同步所有设备"""
    if TORCH_DEVICE.type == "mps":
        torch.mps.synchronize()
    elif TORCH_DEVICE.type == "cuda":
        torch.cuda.synchronize()
