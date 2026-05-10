"""Per-session frame + action writers.

Layout:
    <output_root>/<YYYYMMDD-HHMMSS>/
        meta.json
        actions.jsonl
        frames/00000000.npz
        frames/00000001.npz
        ...

Frames are stored compressed (np.savez_compressed) with both region crops in a
single archive per timestep. Optional full-client frame is included if requested.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class SessionWriter:
    def __init__(self, root: Path, meta: dict):
        ts_str = time.strftime("%Y%m%d-%H%M%S")
        self.session_dir = Path(root) / ts_str
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.actions_path = self.session_dir / "actions.jsonl"
        self.meta_path = self.session_dir / "meta.json"

        self._actions_fh = open(self.actions_path, "a", encoding="utf-8")
        self._frame_idx = 0

        self.meta = {**meta, "started_at": ts_str}
        self._save_meta()

    def _save_meta(self) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, default=str)

    def write_frame(
        self,
        ts: float,
        ge_crop: np.ndarray,
        inv_crop: np.ndarray,
        full: np.ndarray | None = None,
    ) -> int:
        idx = self._frame_idx
        path = self.frames_dir / f"{idx:08d}.npz"
        kwargs = {"ts": np.float64(ts), "ge": ge_crop, "inv": inv_crop}
        if full is not None:
            kwargs["full"] = full
        np.savez_compressed(path, **kwargs)
        self._frame_idx += 1
        return idx

    def write_actions(self, events: list[dict], frame_idx: int) -> None:
        for ev in events:
            ev["frame_idx"] = frame_idx
            self._actions_fh.write(json.dumps(ev) + "\n")
        self._actions_fh.flush()

    def close(self) -> None:
        self.meta["frames_written"] = self._frame_idx
        self.meta["ended_at"] = time.strftime("%Y%m%d-%H%M%S")
        self._save_meta()
        self._actions_fh.close()
