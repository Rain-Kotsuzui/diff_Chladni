"""
Brush Engine for non-uniform plate thickness editing.
Supports Photoshop-like brush stamping with hardness, flow, and preset management.
"""
import numpy as np
import json
import os
from pathlib import Path


PRESETS_DIR = Path(__file__).parent / "brush_presets"


def ensure_presets_dir():
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_stamp(size: int, hardness: float) -> np.ndarray:
    """
    Generate a brush stamp kernel.
    
    Args:
        size: Brush diameter in pixels (grid cells).
        hardness: 0.0 (very soft) to 1.0 (hard edge).
    
    Returns:
        2D float32 array of shape (size, size) with values in [0, 1].
    """
    if size < 1:
        size = 1
    radius = size / 2.0
    y, x = np.ogrid[-radius:radius:size*1j, -radius:radius:size*1j]
    # normalize distance so that edge is at r=1
    d = np.sqrt(x*x + y*y) / radius
    d = np.clip(d, 0.0, 1.0)

    # Hardness mapping:
    # hardness -> 0: very soft gaussian-like falloff
    # hardness -> 1: nearly hard edge
    if hardness >= 0.99:
        # Hard circle
        stamp = np.where(d < 1.0, 1.0, 0.0)
    else:
        # Softness parameter: lower hardness = larger sigma
        # Map hardness [0,1] to sigma [0.5, 0.05]
        sigma = 0.5 - 0.45 * hardness
        stamp = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        stamp = np.clip(stamp, 0.0, 1.0)
        # Zero outside radius for crisp circular boundary
        stamp[d >= 1.0] = 0.0

    # Normalize so that maximum is 1.0
    max_val = stamp.max()
    if max_val > 1e-9:
        stamp = stamp / max_val
    return stamp.astype(np.float32)


def apply_brush(
    h_np: np.ndarray,
    center_row: int,
    center_col: int,
    mode: str,  # 'raise' or 'lower'
    flow: float,
    stamp: np.ndarray,
    delta_h: float = 0.0001,  # base thickness change per application (meters)
    min_h: float = 0.00005,
    max_h: float = 0.01,
) -> np.ndarray:
    """
    Apply a brush stamp to the thickness field at given center pixel.
    
    Args:
        h_np: 2D array of thickness values (meters).
        center_row, center_col: integer center coordinates in h_np index space.
        mode: 'raise' or 'lower'.
        flow: 0.0 to 1.0, brush opacity/flow.
        stamp: 2D kernel from generate_stamp().
        delta_h: base thickness change magnitude per full-strength stamp.
        min_h, max_h: clamp bounds.
    
    Returns:
        Modified h_np (modified in-place and returned).
    """
    if mode not in ("raise", "lower"):
        return h_np

    sign = 1.0 if mode == "raise" else -1.0
    kh, kw = stamp.shape
    half_h = kh // 2
    half_w = kw // 2

    rows = h_np.shape[0]
    cols = h_np.shape[1]

    r0 = max(0, center_row - half_h)
    r1 = min(rows, center_row - half_h + kh)
    c0 = max(0, center_col - half_w)
    c1 = min(cols, center_col - half_w + kw)

    # Compute overlapping slice of stamp
    sr0 = r0 - (center_row - half_h)
    sr1 = sr0 + (r1 - r0)
    sc0 = c0 - (center_col - half_w)
    sc1 = sc0 + (c1 - c0)

    if r1 <= r0 or c1 <= c0:
        return h_np

    patch = stamp[sr0:sr1, sc0:sc1]
    h_np[r0:r1, c0:c1] += sign * flow * delta_h * patch
    np.clip(h_np, min_h, max_h, out=h_np)
    return h_np


def save_preset(name: str, size: int, hardness: float, flow: float, mode: str):
    """Save a brush preset to JSON."""
    ensure_presets_dir()
    path = PRESETS_DIR / f"{name}.json"
    data = {
        "name": name,
        "size": int(size),
        "hardness": float(hardness),
        "flow": float(flow),
        "mode": str(mode),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_preset(name: str) -> dict:
    """Load a brush preset by name. Returns dict with defaults if missing."""
    ensure_presets_dir()
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        return default_preset()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_presets() -> list:
    """Return a list of preset names."""
    ensure_presets_dir()
    names = []
    for p in sorted(PRESETS_DIR.glob("*.json")):
        names.append(p.stem)
    return names


def delete_preset(name: str):
    """Delete a preset by name."""
    ensure_presets_dir()
    path = PRESETS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()


def default_preset() -> dict:
    return {
        "name": "Default",
        "size": 15,
        "hardness": 0.5,
        "flow": 0.5,
        "mode": "raise",
    }
