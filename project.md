# OSRS Flipper Bot — Architecture & Plan

Pixel-only PyTorch vision policy that learns to flip items on the Grand Exchange of a private OSRS server, supervised by a closed-source overlay plugin that visually marks the next correct action.

## Scope

**In scope:**
- Watch the OSRS client window, identify the plugin's overlay cues, and execute the right click + keypress sequence to complete buy/sell flips.
- Two regions of interest: the **trade menu** (GE interface) and **inventory**. Clicks are bounded to those bboxes.
- Behavioral cloning from human demos → optional DAgger → optional PPO with server-DB GP-delta reward.

**Out of scope (explicit):**
- World movement, chat, combat, anything outside the GE flow.
- Reading the plugin's source, memory, files, or any sidecar — pixel-only is a hard constraint.
- Anything pointed at official Jagex worlds. Private test server only.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Screen capture | `dxcam` | ~240 fps on Windows; fastest path |
| Input emulation | `pydirectinput` | DirectInput; survives input filtering that ignores `SendInput` |
| Window mgmt | `pywin32` | Foreground check + bbox lookup for clamp |
| Model framework | PyTorch 2.x | Standard; ConvNeXt / ViT pretrained available |
| Encoder | ConvNeXt-Tiny (ImageNet-pretrained) | Strong on text + colored-overlay regimes; small enough for one GPU |
| RL (later) | `gymnasium` + `stable-baselines3` (PPO) | Stable, debuggable |
| Logging | TensorBoard or Wandb | Either fine |
| Demo store | SQLite + per-frame `.npz` | Small enough, queryable |

All local on the PC. No cloud. Latency in the action loop matters.

## Observation space

Each timestep produces:
- **GE menu crop**: fixed-size RGB tensor (e.g. 256×384), pulled from a static bbox relative to the client window.
- **Inventory crop**: fixed-size RGB tensor (e.g. 256×256).
- **HSV color-mask channels**: for each plugin overlay color, a binary mask threshold per crop. Concatenated as extra input channels → input to each tower is `(3 + N) × H × W`.
- **Frame stack**: last K=4 frames per crop for short temporal context.

Lock UI scale + window resolution at recording time. The two crops + mask channels are the only visual input.

## Action space (discrete, multi-head)

| Head | Values | Notes |
|---|---|---|
| `region` | `{none, menu, inventory}` | Which region the click targets |
| `click_x` | grid over region width (e.g. 32 bins) | Discretized x within the chosen region |
| `click_y` | grid over region height (e.g. 24 bins) | Discretized y within the chosen region |
| `button` | `{none, left, right}` | Click type |
| `key` | `{none, 0–9, backspace, enter, esc, k}` | For price/qty entry |
| `dwell` | `{0, 100ms, 250ms, 500ms}` | Pause after action |

Loss = sum of per-head cross-entropy. `none` actions are valid and abundant (model must learn to wait).

Final pixel coords = region bbox origin + grid bin → clamped to the OSRS window. **Hard clamp at the env layer**, not just trusted from the model.

## Model

**Two crop towers** (shared architecture, separate weights):
- ConvNeXt-Tiny stem + stages, frozen for first ~5 epochs of BC, then unfrozen.
- GAP → 256-d feature per tower per frame.
- Frame stack of 4 → 4×256 = 1024-d per tower.

**Trunk:**
- Concat both towers (2048-d) → 2-layer MLP (1024 → 512) with GELU + LayerNorm.

**Action heads:**
- Six independent linear heads on the 512-d trunk output (see action space table).

**Auxiliary heads (BC only, dropped at inference):**
- `overlay_bbox`: predict (cx, cy, w, h, class) for the plugin's current cue box per region.
- `slot_state`: 8-class softmax × 8 GE slots — `{empty, buying, pending, complete, selling, sold, collect_buy, collect_sell}`.
- `digit_ocr`: predict the digit string currently in the price/qty input (synthetic-font + real-capture training data).

Total loss = primary action loss + 0.1 × (sum of aux losses). Aux signals are crucial without a teacher socket — they force the encoder to build the right representations from pixels alone.

## Training stages

### Stage 1 — Behavioral cloning (BC)

- **Goal:** policy that follows the plugin's overlays in nominal conditions.
- **Data:** 15–30 hours of recorded human play with plugin on, stratified across:
  - In-game time-of-day (lighting changes)
  - All 8 GE slot states (mix of empty / pending / complete)
  - Active flipping AND idle waiting periods
- **Loss:** sum of per-head CE + 0.1 × aux losses.
- **Augmentations:** random window position, brightness/contrast jitter, occlusion patches (chat box, players in the GE area), small UI-scale jitter.
- **Acceptance:** beats scripted baseline on simulated GP/hr in offline replay.

### Stage 2 — DAgger (optional, only if BC drifts in live play)

- Run BC policy live; user takes over on divergence; corrections logged as (frame, expert_action) pairs; retrain.
- Iterate until live behavior is stable.

### Stage 3 — PPO fine-tune (optional, only after Stage 1+2 stable)

- `gymnasium` env wrapping the live game.
- Reward = Δ GP queried directly from the **private server's DB** per step (we own the server) + shaping:
  - `+1` for opening GE
  - `+1` for entering the right item (matched by overlay class at action time)
  - `+5` for a completed flip
  - `−0.1`/sec wasted in idle
  - `−10` for clicks clamped to the bbox boundary (i.e. the model tried to leave the menu)
- Policy initialized from Stage 1 weights. Value head tacked on the trunk.

## Data collection

`recorder/` script:
- 30 fps screen + per-event mouse/keyboard log with monotonic timestamps.
- Saves: `frames/{epoch}.npz` (uint8 RGB crops, both regions), `actions.jsonl` (one row per event), `meta.json` (resolution, UI scale, plugin version hash if surfaced cosmetically — NOT internal).
- One session per file; SQLite manifest indexes sessions for quick query.

Stratify recording sessions by:
- Hour-of-day in-game
- High-volume vs low-volume item categories
- Slow / fast play pace

## Safety

- **bbox clamp:** all click outputs are clamped into the OSRS client window before being dispatched. Enforced at env layer, not relied upon from the model.
- **kill switch:** global hotkey thread (e.g. F12) terminates the policy and releases mouse/keyboard.
- **`pyautogui.FAILSAFE = True`** as a backup — corner-of-screen abort still works since `pydirectinput` doesn't override it.
- **Action rate limit:** max 10 actions/sec, hard cap.
- **Server-side circuit breaker** (later, when RL): if GP delta over N minutes is too negative, kill the session.

## Open decisions

- **Encoder:** ConvNeXt-Tiny (default) vs ViT-S/16. ConvNeXt is the safer first pick.
- **Action discretization grid:** 32×24 or finer? Coarser is easier; finer lets it hit small buttons. Start 32×24.
- **Frame stack vs Transformer over frames:** stack first; revisit if temporal context matters.
- **Reward shaping weights** for PPO — only relevant once we get there.

## Milestones

1. **Recorder working** + 1 hour of demo captured.
2. Demo loader + dataset stats notebook (action distribution, slot state coverage).
3. BC baseline trained, evaluated on held-out demo replay.
4. Live BC rollout in the env with safety wrappers.
5. DAgger loop if needed.
6. PPO env + reward; fine-tune.

## Risks

- **Distribution shift on UI scale / resolution.** Mitigated by locking both at recording; otherwise heavy augmentation.
- **Action-space mismatch:** if the plugin sometimes wants actions outside the trade menu, the bounded action space breaks. Audit demo logs for any out-of-region clicks before committing.
- **Reward signal lag** (RL): server DB writes may lag actions; account for it in reward window.
- **Overlay color collisions** with normal game UI — verify HSV thresholds give clean masks before using as aux input.
