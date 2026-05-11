"""OSRS Flipper Bot — supervised dataset for behavioral cloning.

Reads sessions written by `recorder/` and serves (observation, multi-head label)
pairs ready for a PyTorch DataLoader.

Cross-platform — works on Mac and Windows identically. Coords from pynput and
bbox config are in the OS's logical units (points on macOS, pixels on Windows);
the math is the same either way because everything stays in one coordinate
system per session.

Quick inspect:
    python -m train.dataset path/to/data/demos
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from train.config import (
    BUTTON_VOCAB,
    DWELL_BUCKETS_MS,
    DataConfig,
    KEY_VOCAB,
    REGION_VOCAB,
)
from train.transforms import preprocess_stack


REGION_NONE = REGION_VOCAB.index("none")
REGION_MENU = REGION_VOCAB.index("menu")
REGION_INVENTORY = REGION_VOCAB.index("inventory")


@dataclass
class SessionIndex:
    session_dir: Path
    frame_paths: list[Path]
    frame_ts: np.ndarray              # float64, monotonic timestamps per frame
    events: list[dict]                # sorted by timestamp
    client_origin: tuple[int, int]    # (left, top) of client area in screen coords
    ge_bbox: tuple[int, int, int, int]    # client-relative (x1, y1, x2, y2)
    inv_bbox: tuple[int, int, int, int]


def index_session(session_dir: Path) -> SessionIndex:
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))

    frame_paths = sorted((session_dir / "frames").glob("*.npz"))
    ts = np.zeros(len(frame_paths), dtype=np.float64)
    for i, p in enumerate(frame_paths):
        with np.load(p) as z:
            ts[i] = float(z["ts"])

    events: list[dict] = []
    actions_path = session_dir / "actions.jsonl"
    if actions_path.exists():
        with open(actions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        events.sort(key=lambda e: e["ts"])

    cfg = meta["config"]
    client_bbox = meta["client_bbox"]
    return SessionIndex(
        session_dir=session_dir,
        frame_paths=frame_paths,
        frame_ts=ts,
        events=events,
        client_origin=(int(client_bbox[0]), int(client_bbox[1])),
        ge_bbox=tuple(cfg["ge_menu_bbox"]),
        inv_bbox=tuple(cfg["inventory_bbox"]),
    )


def _bin_dwell(ms: float) -> int:
    bucket = 0
    for i, threshold in enumerate(DWELL_BUCKETS_MS):
        if ms >= threshold:
            bucket = i
    return bucket


def _bin_xy(
    rx: float, ry: float, bbox: tuple[int, int, int, int], data_cfg: DataConfig
) -> tuple[int, int] | None:
    x1, y1, x2, y2 = bbox
    if not (x1 <= rx < x2 and y1 <= ry < y2):
        return None
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    bx = min(data_cfg.grid_x - 1, int((rx - x1) / w * data_cfg.grid_x))
    by = min(data_cfg.grid_y - 1, int((ry - y1) / h * data_cfg.grid_y))
    return bx, by


def build_label_for_frame(
    sess: SessionIndex, frame_idx: int, cfg: DataConfig
) -> dict[str, int]:
    """Find the next click/keypress within `lookahead_s` and build a label.

    If no such event exists, the label is the all-zero idle label (region=none,
    button=none, key=none, dwell=0).
    """
    t_frame = float(sess.frame_ts[frame_idx])
    t_end = t_frame + cfg.lookahead_s

    label = {"region": REGION_NONE, "click_x": 0, "click_y": 0,
             "button": 0, "key": 0, "dwell": 0}

    # Linear scan from a starting point. Optimization (binary search) is fine
    # for sub-million-event sessions; we'll add it if we ever need it.
    next_event = None
    for ev in sess.events:
        t = ev["ts"]
        if t < t_frame:
            continue
        if t > t_end:
            break
        t_ev = ev["type"]
        if t_ev == "click" and ev.get("pressed", False):
            next_event = ev
            break
        if t_ev == "key_down":
            next_event = ev
            break

    if next_event is None:
        return label

    if next_event["type"] == "click":
        ox, oy = sess.client_origin
        rx = float(next_event["x"]) - ox
        ry = float(next_event["y"]) - oy

        binned = _bin_xy(rx, ry, sess.ge_bbox, cfg)
        if binned is not None:
            label["region"] = REGION_MENU
        else:
            binned = _bin_xy(rx, ry, sess.inv_bbox, cfg)
            if binned is not None:
                label["region"] = REGION_INVENTORY
            else:
                # Click landed outside both regions; treat as idle for BC.
                return label

        label["click_x"], label["click_y"] = binned
        btn = next_event.get("button", "left")
        label["button"] = BUTTON_VOCAB.index(btn) if btn in BUTTON_VOCAB else 0

    elif next_event["type"] == "key_down":
        key = next_event.get("key") or ""
        label["key"] = KEY_VOCAB.index(key) if key in KEY_VOCAB else 0

    dwell_ms = (float(next_event["ts"]) - t_frame) * 1000.0
    label["dwell"] = _bin_dwell(dwell_ms)
    return label


class OSRSFlipDataset(Dataset):
    """One sample = (GE crop stack, inventory crop stack, multi-head label)."""

    def __init__(self, data_root: Path | str, cfg: DataConfig | None = None):
        self.data_root = Path(data_root)
        self.cfg = cfg or DataConfig()

        self.sessions: list[SessionIndex] = []
        if self.data_root.is_dir():
            for sd in sorted(self.data_root.iterdir()):
                if sd.is_dir() and (sd / "meta.json").exists():
                    self.sessions.append(index_session(sd))

        K = self.cfg.frame_stack
        self.samples: list[tuple[int, int]] = []
        for si, sess in enumerate(self.sessions):
            # Skip the first K-1 frames where we don't have a full stack
            for fi in range(K - 1, len(sess.frame_paths)):
                self.samples.append((si, fi))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        si, fi = self.samples[idx]
        sess = self.sessions[si]
        K = self.cfg.frame_stack

        ge_frames: list[np.ndarray] = []
        inv_frames: list[np.ndarray] = []
        for k in range(K):
            p = sess.frame_paths[fi - (K - 1 - k)]
            with np.load(p) as z:
                ge_frames.append(np.asarray(z["ge"]))
                inv_frames.append(np.asarray(z["inv"]))

        ge = preprocess_stack(ge_frames, self.cfg.ge_resize_hw, self.cfg)
        inv = preprocess_stack(inv_frames, self.cfg.inv_resize_hw, self.cfg)

        label = build_label_for_frame(sess, fi, self.cfg)

        return {
            "ge": ge,
            "inv": inv,
            "region": torch.tensor(label["region"], dtype=torch.long),
            "click_x": torch.tensor(label["click_x"], dtype=torch.long),
            "click_y": torch.tensor(label["click_y"], dtype=torch.long),
            "button": torch.tensor(label["button"], dtype=torch.long),
            "key": torch.tensor(label["key"], dtype=torch.long),
            "dwell": torch.tensor(label["dwell"], dtype=torch.long),
        }


def _inspect(data_root: str) -> int:
    ds = OSRSFlipDataset(data_root)
    print(f"sessions: {len(ds.sessions)}")
    for s in ds.sessions:
        print(f"  {s.session_dir.name}: {len(s.frame_paths)} frames, {len(s.events)} events")
    print(f"samples (frames with full stack): {len(ds)}")
    if len(ds) == 0:
        print("(no samples — record some demos first)")
        return 0

    sample = ds[0]
    print("\nfirst sample shapes:")
    print(f"  ge:  {tuple(sample['ge'].shape)}  dtype={sample['ge'].dtype}")
    print(f"  inv: {tuple(sample['inv'].shape)}  dtype={sample['inv'].dtype}")
    print("first sample label:")
    for k in ("region", "click_x", "click_y", "button", "key", "dwell"):
        print(f"  {k}: {int(sample[k])}")

    # Label-distribution sanity for the first 1024 samples
    n = min(1024, len(ds))
    region_counts = np.zeros(len(REGION_VOCAB), dtype=np.int64)
    button_counts = np.zeros(len(BUTTON_VOCAB), dtype=np.int64)
    key_counts = np.zeros(len(KEY_VOCAB), dtype=np.int64)
    for i in range(n):
        s = ds[i]
        region_counts[int(s["region"])] += 1
        button_counts[int(s["button"])] += 1
        key_counts[int(s["key"])] += 1
    print(f"\nlabel distribution over first {n} samples:")
    print(f"  region: {dict(zip(REGION_VOCAB, region_counts.tolist()))}")
    print(f"  button: {dict(zip(BUTTON_VOCAB, button_counts.tolist()))}")
    nz_keys = {k: int(c) for k, c in zip(KEY_VOCAB, key_counts) if c > 0}
    print(f"  key (non-zero): {nz_keys}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m train.dataset <data_root>")
        sys.exit(2)
    sys.exit(_inspect(sys.argv[1]))
