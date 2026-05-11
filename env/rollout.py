"""Live rollout of a trained policy.

Usage:
    python -m env.rollout --checkpoint checkpoints/best.pt
    python -m env.rollout --checkpoint checkpoints/best.pt --device mps --steps 500
    python -m env.rollout --checkpoint checkpoints/best.pt --dry-run

macOS / MPS: set PYTORCH_ENABLE_MPS_FALLBACK=1 before launching python if you
hit ops that aren't yet supported on MPS.

Safety:
  - F12 (override via --kill-key) terminates the rollout.
  - Move the cursor to a screen corner to trigger pyautogui FAILSAFE (macOS).
  - All clicks are clamped to the client window inside the env.

DO NOT RUN until you have a trained checkpoint AND the game window open at
the same resolution + UI scale used for the demos that trained it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env.osrs_env import HEAD_ORDER, OSRSFlipEnv
from model.policy import FlipperPolicy
from recorder.config import CaptureConfig
from train.config import DataConfig, MaskSpec


def _pick_device(prefer: str) -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reconstruct_data_cfg(d: dict[str, Any]) -> DataConfig:
    """Rebuild a DataConfig from the dict saved in the checkpoint."""
    cfg = DataConfig()
    tuple_fields = {"ge_resize_hw", "inv_resize_hw", "rgb_mean", "rgb_std"}
    for k, v in d.items():
        if not hasattr(cfg, k):
            continue
        if k == "mask_specs":
            cfg.mask_specs = [MaskSpec(**m) if isinstance(m, dict) else m for m in v]
        elif k in tuple_fields and isinstance(v, list):
            setattr(cfg, k, tuple(v))
        else:
            setattr(cfg, k, v)
    return cfg


def _start_kill_listener(kill_key: str, stop_flag: dict[str, bool]):
    from pynput import keyboard

    def on_press(key):
        name = getattr(key, "name", None) or getattr(key, "char", None)
        if name == kill_key:
            stop_flag["v"] = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _argmax_action(logits: dict[str, torch.Tensor]) -> np.ndarray:
    return np.array(
        [int(logits[h].argmax(dim=-1).item()) for h in HEAD_ORDER],
        dtype=np.int64,
    )


def rollout(args: argparse.Namespace) -> int:
    device = _pick_device(args.device)
    print(f"device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data_cfg = _reconstruct_data_cfg(ckpt["data_cfg"])

    model = FlipperPolicy(data_cfg, pretrained=False).to(device)  # don't redownload ImageNet
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    capture_cfg = CaptureConfig()
    if args.window:
        capture_cfg.window_title_substr = args.window

    env = OSRSFlipEnv(data_cfg=data_cfg, capture_cfg=capture_cfg, capture_fps=args.fps)

    stop_flag = {"v": False}
    listener = _start_kill_listener(args.kill_key, stop_flag)
    print(f"Press {args.kill_key} to stop.")

    obs, _ = env.reset()
    print(f"obs ge.shape={obs['ge'].shape}  inv.shape={obs['inv'].shape}")

    if args.dry_run:
        print("[dry-run] env initialized + one observation captured. Exiting without acting.")
        env.close()
        listener.stop()
        return 0

    t_start = time.monotonic()
    n_steps = 0
    try:
        while n_steps < args.steps and not stop_flag["v"]:
            with torch.no_grad():
                ge = torch.from_numpy(obs["ge"]).unsqueeze(0).to(device)
                inv = torch.from_numpy(obs["inv"]).unsqueeze(0).to(device)
                logits = model(ge, inv)
            action = _argmax_action(logits)
            obs, _reward, _term, _trunc, _info = env.step(action)
            n_steps += 1

            if n_steps % args.log_every == 0:
                elapsed = time.monotonic() - t_start
                rate = n_steps / max(1e-6, elapsed)
                print(f"step {n_steps}  action={action.tolist()}  {rate:.2f} steps/s")
    finally:
        env.close()
        listener.stop()
        print(f"Done. {n_steps} steps in {time.monotonic() - t_start:.1f}s.")
    return 0


def main():
    p = argparse.ArgumentParser(description="OSRS Flipper Bot — live rollout.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--window", default=None, help="Override window title substring.")
    p.add_argument("--kill-key", default="f12")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--dry-run", action="store_true",
                   help="Initialize env, capture one observation, exit without acting.")
    args = p.parse_args()
    sys.exit(rollout(args))


if __name__ == "__main__":
    main()
