"""Frame preprocessing: HSV color masks, resize, normalize, frame-stack concat.

Cross-platform — pure numpy + PIL + torch. PIL is a transitive dep of torchvision,
so no extra install needed.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from train.config import DataConfig, MaskSpec


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB → HxWx3 uint8 HSV (PIL convention: 0-255 per channel)."""
    return np.asarray(Image.fromarray(rgb).convert("HSV"))


def build_color_masks(rgb: np.ndarray, specs: list[MaskSpec]) -> np.ndarray:
    """Return (N, H, W) uint8 masks, one per spec, with values in {0, 255}."""
    if not specs:
        return np.zeros((0, rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    masks = np.zeros((len(specs), rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
    for i, spec in enumerate(specs):
        if spec.h_min <= spec.h_max:
            h_ok = (h >= spec.h_min) & (h <= spec.h_max)
        else:
            # Hue wraps around the 0/255 boundary (e.g. red)
            h_ok = (h >= spec.h_min) | (h <= spec.h_max)
        s_ok = (s >= spec.s_min) & (s <= spec.s_max)
        v_ok = (v >= spec.v_min) & (v <= spec.v_max)
        masks[i] = (h_ok & s_ok & v_ok).astype(np.uint8) * 255
    return masks


def _resize(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L").resize((w, h), Image.BILINEAR)
    else:
        img = Image.fromarray(arr).resize((w, h), Image.BILINEAR)
    return np.asarray(img)


def preprocess_stack(
    frames: list[np.ndarray],
    size_hw: tuple[int, int],
    cfg: DataConfig,
) -> torch.Tensor:
    """Build the (K*(3+N), H, W) float32 tensor a tower will see.

    frames: list of K HxWx3 uint8 RGB crops (oldest first, newest last).
    Each frame contributes 3 normalized RGB channels + N binary mask channels.
    """
    mean = np.array(cfg.rgb_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(cfg.rgb_std, dtype=np.float32).reshape(3, 1, 1)

    layers: list[np.ndarray] = []
    for f in frames:
        # RGB: resize → CHW → normalize
        rgb_r = _resize(f, size_hw).astype(np.float32) / 255.0
        rgb_chw = rgb_r.transpose(2, 0, 1)
        rgb_norm = (rgb_chw - mean) / std
        layers.append(rgb_norm)

        if cfg.mask_specs:
            # Build masks at original resolution (cleaner thresholding), then resize
            masks = build_color_masks(f, cfg.mask_specs)
            masks_r = np.stack([_resize(m, size_hw) for m in masks], axis=0)
            layers.append(masks_r.astype(np.float32) / 255.0)

    return torch.from_numpy(np.concatenate(layers, axis=0))


def per_tower_channels(cfg: DataConfig) -> int:
    """How many input channels a tower expects, given the current config."""
    return cfg.frame_stack * (3 + len(cfg.mask_specs))
