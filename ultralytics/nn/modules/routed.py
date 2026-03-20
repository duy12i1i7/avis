# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Routing-aware modules for tiny-object detection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("SFRBottleneck", "SFRC2f", "SparseSubpixelExpert")


class SparseSubpixelExpert(nn.Module):
    """Top-k patch expert that restores subpixel details only where the router requests more compute."""

    def __init__(self, c1: int, patch_size: int = 4, route_ratio: float = 0.25, min_regions: int = 1, e: float = 0.5):
        """Initialize the sparse subpixel expert."""
        super().__init__()
        hidden = max(int(c1 * e), 16)
        self.patch_size = max(int(patch_size), 2)
        self.route_ratio = max(float(route_ratio), 0.0)
        self.min_regions = max(int(min_regions), 1)

        route_hidden = max(hidden // 2, 16)
        self.route_head = nn.Sequential(
            nn.Conv2d(c1, route_hidden, 1, bias=False),
            nn.BatchNorm2d(route_hidden),
            nn.SiLU(),
            nn.Conv2d(route_hidden, 1, 1),
        )
        self.expert = nn.Sequential(
            Conv(c1 * 4, hidden, 1, 1),
            DWConv(hidden, hidden, 3, 1),
            Conv(hidden, c1 * 4, 1, 1, act=False),
        )
        self.last_route_density = 1.0

    def _effective_patch_size(self, h: int, w: int) -> int:
        """Return a valid even patch size for the current spatial shape."""
        p = min(self.patch_size, h, w)
        if p % 2:
            p -= 1
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sparse routed refinement and return a feature residual."""
        b, c, h, w = x.shape
        p = self._effective_patch_size(h, w)
        if self.route_ratio <= 0.0 or p < 2:
            self.last_route_density = 0.0
            return x.new_zeros(x.shape)

        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, hp, wp = x_pad.shape
        gh, gw = hp // p, wp // p
        num_patches = gh * gw
        k = min(num_patches, max(self.min_regions, int(math.ceil(num_patches * self.route_ratio))))
        self.last_route_density = k / max(num_patches, 1)

        pooled = F.avg_pool2d(x_pad, kernel_size=p, stride=p)
        scores = self.route_head(pooled).sigmoid().flatten(1)
        topk_idx = scores.topk(k, dim=1, sorted=False).indices

        patches = x_pad.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, num_patches, c, p, p)
        batch_idx = torch.arange(b, device=x.device).unsqueeze(1)
        selected = patches[batch_idx, topk_idx].reshape(b * k, c, p, p)

        refined = F.pixel_unshuffle(selected, 2)
        refined = self.expert(refined)
        refined = F.pixel_shuffle(refined, 2).view(b, k, c, p, p)
        refined = refined * scores.gather(1, topk_idx).view(b, k, 1, 1, 1)

        delta = x_pad.new_zeros((b, num_patches, c, p, p))
        delta[batch_idx, topk_idx] = refined
        delta = delta.view(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, hp, wp)
        return delta[..., :h, :w]


class SFRBottleneck(nn.Module):
    """A lightweight bottleneck with dense local context and a sparse detail expert."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        e: float = 1.0,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 1,
    ):
        """Initialize the routed bottleneck."""
        super().__init__()
        hidden = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.local = DWConv(hidden, hidden, 3, 1)
        self.expert = SparseSubpixelExpert(
            hidden, patch_size=patch_size, route_ratio=route_ratio, min_regions=min_regions, e=e
        )
        self.cv2 = Conv(hidden, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the routed bottleneck with optional residual connection."""
        y = self.cv1(x)
        y = self.local(y) + self.expert(y)
        y = self.cv2(y)
        return x + y if self.add else y


class SFRC2f(nn.Module):
    """C2f-style block whose repeated units use sparse routed subpixel experts."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        e: float = 0.5,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 1,
    ):
        """Initialize a C2f block with sparse routed bottlenecks."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            SFRBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                e=1.0,
                patch_size=patch_size,
                route_ratio=route_ratio,
                min_regions=min_regions,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the routed C2f block."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
