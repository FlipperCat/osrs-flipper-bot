"""Two-tower vision policy.

GE menu crop and inventory crop each go through their own ConvNeXt-Tiny
(via timm, which adapts the 3-channel ImageNet stem to our K*(3+N)-channel
frame-stacked input). The two feature vectors are concatenated, run through
a small trunk MLP, and read out by six independent action heads.

Heads, in order: region, click_x, click_y, button, key, dwell. All categorical.

Aux heads (overlay_bbox / slot_state / digit_ocr) from project.md are not
included here — they require labels the recorder doesn't produce automatically
and are a follow-up once V1 is training well.
"""
from __future__ import annotations

from typing import Iterable

import timm
import torch
from torch import nn

from train.config import (
    BUTTON_VOCAB,
    DWELL_BUCKETS_MS,
    DataConfig,
    KEY_VOCAB,
    REGION_VOCAB,
)
from train.transforms import per_tower_channels


class RegionTower(nn.Module):
    """ConvNeXt-Tiny with the input stem adapted to in_chans channels.

    Returns a (B, feat_dim) pooled feature vector. timm handles the
    in_chans != 3 adaptation by averaging the ImageNet stem weights so
    pretraining is partially retained.
    """

    def __init__(self, in_chans: int, model_name: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,         # drop the ImageNet classifier
            global_pool="avg",     # we want a single pooled vector
        )
        self.feat_dim: int = int(self.backbone.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class Trunk(nn.Module):
    """MLP: concat(ge_feat, inv_feat) -> hidden -> hidden/2."""

    def __init__(self, in_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
        )
        self.out_dim: int = hidden // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyHeads(nn.Module):
    """Six independent linear heads on the trunk output. Returns a dict of logits."""

    def __init__(self, in_dim: int, cfg: DataConfig):
        super().__init__()
        self.region = nn.Linear(in_dim, len(REGION_VOCAB))
        self.click_x = nn.Linear(in_dim, cfg.grid_x)
        self.click_y = nn.Linear(in_dim, cfg.grid_y)
        self.button = nn.Linear(in_dim, len(BUTTON_VOCAB))
        self.key = nn.Linear(in_dim, len(KEY_VOCAB))
        self.dwell = nn.Linear(in_dim, len(DWELL_BUCKETS_MS))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "region": self.region(x),
            "click_x": self.click_x(x),
            "click_y": self.click_y(x),
            "button": self.button(x),
            "key": self.key(x),
            "dwell": self.dwell(x),
        }


class FlipperPolicy(nn.Module):
    """Full model: two towers + trunk + six heads."""

    def __init__(
        self,
        cfg: DataConfig,
        encoder_name: str = "convnext_tiny",
        pretrained: bool = True,
        trunk_hidden: int = 1024,
    ):
        super().__init__()
        in_ch = per_tower_channels(cfg)
        self.ge_tower = RegionTower(in_ch, encoder_name, pretrained=pretrained)
        self.inv_tower = RegionTower(in_ch, encoder_name, pretrained=pretrained)
        self.trunk = Trunk(self.ge_tower.feat_dim + self.inv_tower.feat_dim, hidden=trunk_hidden)
        self.heads = PolicyHeads(self.trunk.out_dim, cfg)

    def forward(self, ge: torch.Tensor, inv: torch.Tensor) -> dict[str, torch.Tensor]:
        f_ge = self.ge_tower(ge)
        f_inv = self.inv_tower(inv)
        x = torch.cat([f_ge, f_inv], dim=-1)
        x = self.trunk(x)
        return self.heads(x)

    # -- training helpers -------------------------------------------------

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.ge_tower.parameters()
        yield from self.inv_tower.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.trunk.parameters()
        yield from self.heads.parameters()

    def set_backbone_frozen(self, frozen: bool) -> None:
        for p in self.backbone_parameters():
            p.requires_grad = not frozen
