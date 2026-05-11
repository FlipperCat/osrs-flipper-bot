"""Behavioral cloning training loop.

Usage:
    python -m train.bc <data_root>
    python -m train.bc <data_root> --device mps --batch-size 16 --epochs 10
    python -m train.bc <data_root> --quick    # tiny smoke test

Device auto: CUDA > MPS (Apple Silicon) > CPU.

macOS / MPS note: some ops fall back to CPU silently with PYTORCH_ENABLE_MPS_FALLBACK=1.
PyTorch reads that var at import time — set it BEFORE launching python:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m train.bc ...

DO NOT RUN until you have at least one recorded demo session.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model.policy import FlipperPolicy
from train.config import (
    BUTTON_VOCAB,
    DWELL_BUCKETS_MS,
    DataConfig,
    KEY_VOCAB,
    REGION_VOCAB,
)
from train.dataset import OSRSFlipDataset


HEAD_NAMES: tuple[str, ...] = ("region", "click_x", "click_y", "button", "key", "dwell")


@dataclass
class BCConfig:
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 3e-4
    backbone_lr_mult: float = 0.1     # backbone gets lr * this once unfrozen
    weight_decay: float = 1e-4
    epochs: int = 20
    warmup_epochs: int = 3            # backbones frozen for this many epochs
    val_fraction: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    save_dir: str = "checkpoints"
    seed: int = 42
    accum_steps: int = 1


# -- device / setup ---------------------------------------------------------

def _pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -- per-session temporal val split ----------------------------------------

def _temporal_split(ds: OSRSFlipDataset, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Hold out the last `val_fraction` of each session as validation.

    Cleaner than a random split: adjacent frames are highly correlated, so a
    random split leaks the future into training. A per-session temporal split
    keeps val in the future of train for every session.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

    # samples is a flat list of (session_idx, frame_idx) in session-major order;
    # within each session, frame_idx is monotonically increasing.
    by_sess: dict[int, list[int]] = {}
    for i, (si, _fi) in enumerate(ds.samples):
        by_sess.setdefault(si, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for si, idx_list in by_sess.items():
        n = len(idx_list)
        cut = max(1, int(n * (1 - val_fraction)))
        train_idx.extend(idx_list[:cut])
        val_idx.extend(idx_list[cut:])

    rng = random.Random(seed)
    rng.shuffle(train_idx)
    return train_idx, val_idx


# -- loss / metrics --------------------------------------------------------

def _compute_losses(
    logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of cross-entropies across heads. All heads are weighted equally for V1."""
    losses: dict[str, torch.Tensor] = {}
    for h in HEAD_NAMES:
        losses[h] = F.cross_entropy(logits[h], batch[h])
    total = sum(losses.values())
    return total, {h: float(v.detach().cpu()) for h, v in losses.items()}


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().detach().cpu())


# -- train / eval epochs ---------------------------------------------------

def _train_one_epoch(
    model: FlipperPolicy,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    bc_cfg: BCConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    n = 0
    sum_total = 0.0
    sum_per_head: dict[str, float] = {h: 0.0 for h in HEAD_NAMES}
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(batch["ge"], batch["inv"])
        total, per_head = _compute_losses(logits, batch)

        (total / bc_cfg.accum_steps).backward()
        if (step + 1) % bc_cfg.accum_steps == 0:
            if bc_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), bc_cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        bs = batch["region"].size(0)
        n += bs
        sum_total += float(total.detach().cpu()) * bs
        for h, v in per_head.items():
            sum_per_head[h] += v * bs

        if step % bc_cfg.log_every == 0:
            elapsed = time.time() - t0
            ips = n / max(1e-6, elapsed)
            print(
                f"  ep{epoch} step{step} loss={float(total):.4f} "
                f"({', '.join(f'{h}={per_head[h]:.3f}' for h in HEAD_NAMES)}) "
                f"{ips:.1f} samples/s"
            )

    return {
        "loss": sum_total / max(1, n),
        **{f"loss_{h}": sum_per_head[h] / max(1, n) for h in HEAD_NAMES},
    }


@torch.no_grad()
def _eval(model: FlipperPolicy, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    n = 0
    sum_total = 0.0
    sum_per_head = {h: 0.0 for h in HEAD_NAMES}
    sum_acc = {h: 0.0 for h in HEAD_NAMES}

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(batch["ge"], batch["inv"])
        total, per_head = _compute_losses(logits, batch)
        bs = batch["region"].size(0)
        n += bs
        sum_total += float(total) * bs
        for h in HEAD_NAMES:
            sum_per_head[h] += per_head[h] * bs
            sum_acc[h] += _accuracy(logits[h], batch[h]) * bs

    if n == 0:
        return {"loss": float("nan")}
    out = {"loss": sum_total / n}
    for h in HEAD_NAMES:
        out[f"loss_{h}"] = sum_per_head[h] / n
        out[f"acc_{h}"] = sum_acc[h] / n
    return out


# -- optimizer / lr handling -----------------------------------------------

def _build_optimizer(model: FlipperPolicy, bc_cfg: BCConfig, unfrozen: bool) -> torch.optim.Optimizer:
    """Build an AdamW with two param groups.

    During warmup the backbone group is lr=0, so it gets no updates even if
    requires_grad is True (it isn't — set_backbone_frozen handles that too).
    After warmup we recreate the optimizer with the backbone group at
    lr * backbone_lr_mult.
    """
    bb_lr = bc_cfg.lr * bc_cfg.backbone_lr_mult if unfrozen else 0.0
    return torch.optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": bb_lr},
            {"params": list(model.head_parameters()), "lr": bc_cfg.lr},
        ],
        weight_decay=bc_cfg.weight_decay,
    )


# -- checkpoint ------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    model: FlipperPolicy,
    opt: torch.optim.Optimizer,
    data_cfg: DataConfig,
    bc_cfg: BCConfig,
    epoch: int,
    val_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "data_cfg": asdict(data_cfg),
            "bc_cfg": asdict(bc_cfg),
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


# -- main ------------------------------------------------------------------

def train_bc(
    data_root: str,
    device: torch.device,
    data_cfg: DataConfig,
    bc_cfg: BCConfig,
) -> int:
    _set_seed(bc_cfg.seed)

    ds = OSRSFlipDataset(data_root, data_cfg)
    print(f"dataset: {len(ds)} samples across {len(ds.sessions)} sessions")
    if len(ds) == 0:
        print("no samples — record some demos first.")
        return 1

    train_idx, val_idx = _temporal_split(ds, bc_cfg.val_fraction, bc_cfg.seed)
    print(f"split: train={len(train_idx)} val={len(val_idx)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=bc_cfg.batch_size,
        shuffle=True,
        num_workers=bc_cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=bc_cfg.batch_size,
        shuffle=False,
        num_workers=bc_cfg.num_workers,
        pin_memory=pin,
    )

    model = FlipperPolicy(data_cfg).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Phase 1: backbones frozen, train heads + trunk only
    model.set_backbone_frozen(True)
    opt = _build_optimizer(model, bc_cfg, unfrozen=False)
    save_dir = Path(bc_cfg.save_dir)

    best_val = math.inf
    for epoch in range(bc_cfg.epochs):
        if epoch == bc_cfg.warmup_epochs:
            print(f"[unfreezing backbones at epoch {epoch}]")
            model.set_backbone_frozen(False)
            opt = _build_optimizer(model, bc_cfg, unfrozen=True)

        print(f"\nepoch {epoch} (backbones={'frozen' if epoch < bc_cfg.warmup_epochs else 'tunable'})")
        train_metrics = _train_one_epoch(model, train_loader, opt, device, bc_cfg, epoch)
        val_metrics = _eval(model, val_loader, device)

        print(f"  train: loss={train_metrics['loss']:.4f}")
        print(f"  val:   loss={val_metrics['loss']:.4f}  "
              + "  ".join(f"acc_{h}={val_metrics.get(f'acc_{h}', float('nan')):.3f}" for h in HEAD_NAMES))

        # Save last + best
        _save_checkpoint(save_dir / "last.pt", model, opt, data_cfg, bc_cfg, epoch, val_metrics)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            _save_checkpoint(save_dir / "best.pt", model, opt, data_cfg, bc_cfg, epoch, val_metrics)
            print(f"  ** new best val loss={best_val:.4f}")

    print(f"\ndone. best val loss = {best_val:.4f}. checkpoints in {save_dir}/")
    return 0


def main():
    p = argparse.ArgumentParser(description="OSRS Flipper Bot — behavioral cloning.")
    p.add_argument("data_root", type=str, help="Directory containing recorded demo sessions.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Smoke test: tiny batch, 1 epoch, 1 step.")
    args = p.parse_args()

    bc_cfg = BCConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        val_fraction=args.val_fraction,
        save_dir=args.save_dir,
        accum_steps=args.accum_steps,
        seed=args.seed,
    )
    if args.quick:
        bc_cfg.batch_size = 2
        bc_cfg.num_workers = 0
        bc_cfg.epochs = 1
        bc_cfg.warmup_epochs = 0
        bc_cfg.log_every = 1

    device = _pick_device(args.device)
    print(f"device: {device}")
    if device.type == "mps" and not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        print("WARNING: running on MPS without PYTORCH_ENABLE_MPS_FALLBACK=1 — "
              "some ops may error. Re-launch with that env var set if you hit issues.")

    sys.exit(train_bc(args.data_root, device, DataConfig(), bc_cfg))


if __name__ == "__main__":
    main()
