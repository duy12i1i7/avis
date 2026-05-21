# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Routing-aware modules for tiny-object detection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C2f, C3, Bottleneck, PSABlock
from .conv import Conv, DWConv

__all__ = (
    "ScaleSelectiveFusion",
    "SFRBottleneck",
    "SFRKBlock",
    "SFRC2f",
    "SFRC3k",
    "SFRC3k2",
    "SFRC2fFull",
    "SFRC3kFull",
    "SFRC3k2Full",
    "SparseSubpixelExpert",
    "SparseSubpixelExpertFull",
)


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


class ScaleSelectiveFusion(nn.Module):
    """Inject P2 detail into the P3 branch without introducing a full P2 detection head."""

    def __init__(
        self,
        c_p2: int,
        c_p3: int,
        c2: int,
        e: float = 0.5,
        patch_size: int = 4,
        route_ratio: float = 0.125,
        min_regions: int = 1,
    ):
        """Initialize the selective P2-to-P3 fusion bridge."""
        super().__init__()
        hidden = max(int(c2 * e), 16)
        self.p2_proj = Conv(c_p2 * 4, hidden, 1, 1)
        self.p3_proj = Conv(c_p3, hidden, 1, 1)
        self.router = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.detail = SparseSubpixelExpert(
            hidden, patch_size=patch_size, route_ratio=route_ratio, min_regions=min_regions, e=1.0
        )
        self.out = Conv(hidden, c2, 1, 1)
        self.last_route_density = 1.0

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse high-resolution detail from P2 into the current P3 tensor."""
        p2, p3 = x
        target_hw = (p3.shape[-2] * 2, p3.shape[-1] * 2)
        if p2.shape[-2:] != target_hw:
            p2 = F.interpolate(p2, size=target_hw, mode="bilinear", align_corners=False)

        p2 = F.pixel_unshuffle(p2, 2)
        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        gate = self.router(torch.cat((p2, p3), 1)).sigmoid()
        detail = self.detail(p2)
        self.last_route_density = self.detail.last_route_density
        return self.out(p3 + gate * p2 + detail)


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
        use_local: bool = True,
    ):
        """Initialize the routed bottleneck."""
        super().__init__()
        hidden = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.local = DWConv(hidden, hidden, 3, 1) if use_local else None
        self.expert = SparseSubpixelExpert(
            hidden, patch_size=patch_size, route_ratio=route_ratio, min_regions=min_regions, e=e
        )
        self.cv2 = Conv(hidden, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the routed bottleneck with optional residual connection."""
        y = self.cv1(x)
        y = (self.local(y) if self.local is not None else y) + self.expert(y)
        y = self.cv2(y)
        return x + y if self.add else y


class SFRKBlock(nn.Module):
    """Kernel-aware routed bottleneck used to study SparseSubpixelExpert on C3k-style extractors."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 1,
        use_local: bool = True,
        k: int = 3,
    ):
        """Initialize the kernel-aware routed bottleneck."""
        super().__init__()
        hidden = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.local = Bottleneck(hidden, hidden, shortcut=False, g=g, k=(k, k), e=1.0) if use_local else None
        self.expert = SparseSubpixelExpert(
            hidden, patch_size=patch_size, route_ratio=route_ratio, min_regions=min_regions, e=e
        )
        self.cv2 = Conv(hidden, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the kernel-aware routed bottleneck with optional residual connection."""
        y = self.cv1(x)
        y = (self.local(y) if self.local is not None else y) + self.expert(y)
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
        use_local: bool = True,
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
                use_local=use_local,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the routed C2f block."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SFRC3k(C3):
    """C3k-style block whose repeated units use kernel-aware sparse routed bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 1,
        use_local: bool = True,
        k: int = 3,
        g: int = 1,
    ):
        """Initialize a C3k block with sparse routed kernel-aware bottlenecks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(
                SFRKBlock(
                    c_,
                    c_,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    patch_size=patch_size,
                    route_ratio=route_ratio,
                    min_regions=min_regions,
                    use_local=use_local,
                    k=k,
                )
                for _ in range(n)
            )
        )


class SFRC3k2(C2f):
    """C3k2-style shell whose repeated units host sparse routed C3k blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 1,
        use_local: bool = True,
        k: int = 3,
        c3k: bool = True,
        attn: bool = False,
        g: int = 1,
    ):
        """Initialize a C3k2-style routed block for host-module transfer studies."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                SFRKBlock(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    patch_size=patch_size,
                    route_ratio=route_ratio,
                    min_regions=min_regions,
                    use_local=use_local,
                    k=k,
                ),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else SFRC3k(
                self.c,
                self.c,
                2,
                shortcut=shortcut,
                g=g,
                e=1.0,
                patch_size=patch_size,
                route_ratio=route_ratio,
                min_regions=min_regions,
                use_local=use_local,
                k=k,
            )
            if c3k
            else SFRKBlock(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                e=1.0,
                patch_size=patch_size,
                route_ratio=route_ratio,
                min_regions=min_regions,
                use_local=use_local,
                k=k,
            )
            for _ in range(n)
        )


class SparseSubpixelExpertFull(nn.Module):
    """Dynamic-budget sparse expert with route supervision hooks for full SFR experiments."""

    def __init__(
        self,
        c1: int,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 0,
        e: float = 0.5,
        route_thresh: float = 0.45,
        route_floor: float = 0.0,
    ):
        """Initialize the full sparse expert with dynamic routing controls."""
        super().__init__()
        hidden = max(int(c1 * e), 16)
        self.patch_size = max(int(patch_size), 2)
        self.route_ratio = max(float(route_ratio), 0.0)
        self.min_regions = max(int(min_regions), 0)
        self.route_thresh = float(route_thresh)
        self.route_floor = max(float(route_floor), 0.0)

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
        self.last_route_density = 0.0
        self.last_route_count = 0.0
        self.last_num_patches = 0
        self.last_patch_size = 0
        self.last_grid_shape = (0, 0)
        self.last_input_hw = (0, 0)
        self.last_route_logits: torch.Tensor | None = None
        self.last_route_scores: torch.Tensor | None = None
        self.last_route_mask: torch.Tensor | None = None

    def _effective_patch_size(self, h: int, w: int) -> int:
        """Return a valid even patch size for the current tensor."""
        p = min(self.patch_size, h, w)
        if p % 2:
            p -= 1
        return p

    def _routing_bounds(self, num_patches: int) -> tuple[int, int]:
        """Return per-image minimum and maximum routed patch counts."""
        floor = int(math.ceil(num_patches * self.route_floor))
        k_min = max(self.min_regions, floor)
        k_max = min(num_patches, max(k_min, int(math.ceil(num_patches * self.route_ratio))))
        return k_min, k_max

    def _select_indices(self, scores: torch.Tensor, active: torch.Tensor, k_min: int, k_max: int) -> torch.Tensor:
        """Select routed patch indices for one image using threshold-first dynamic routing."""
        if k_max <= 0:
            return scores.new_empty((0,), dtype=torch.long)

        active_idx = active.nonzero(as_tuple=False).flatten()
        desired = active_idx.numel()
        if desired == 0 and k_min == 0:
            return active_idx

        desired = min(max(desired, k_min), k_max)
        if desired == 0:
            return scores.new_empty((0,), dtype=torch.long)

        if active_idx.numel() >= desired:
            active_scores = scores[active_idx]
            if active_idx.numel() == desired:
                return active_idx
            keep = active_scores.topk(desired, sorted=False).indices
            return active_idx[keep]

        return scores.topk(desired, sorted=False).indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic sparse refinement and return a residual tensor."""
        b, c, h, w = x.shape
        p = self._effective_patch_size(h, w)
        track_routes = self.training
        self.last_input_hw = (h, w)
        self.last_patch_size = p
        self.last_route_logits = None
        self.last_route_scores = None
        self.last_route_mask = None
        self.last_route_density = 0.0
        self.last_route_count = 0.0
        self.last_num_patches = 0
        if self.route_ratio <= 0.0 or p < 2:
            return x.new_zeros(x.shape)

        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, hp, wp = x_pad.shape
        gh, gw = hp // p, wp // p
        num_patches = gh * gw
        self.last_num_patches = num_patches
        self.last_grid_shape = (gh, gw)

        k_min, k_max = self._routing_bounds(num_patches)
        if k_max <= 0:
            return x.new_zeros(x.shape)

        pooled = F.avg_pool2d(x_pad, kernel_size=p, stride=p)
        logits_map = self.route_head(pooled).squeeze(1)
        scores_map = logits_map.sigmoid()
        scores = scores_map.flatten(1)
        active = scores_map > self.route_thresh

        if track_routes:
            self.last_route_logits = logits_map
            self.last_route_scores = scores_map
            self.last_route_mask = torch.zeros_like(scores_map, dtype=torch.bool)

        patches = x_pad.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, num_patches, c, p, p)
        delta = x_pad.new_zeros((b, num_patches, c, p, p))

        total_routed = 0
        for bi in range(b):
            idx = self._select_indices(scores[bi], active[bi].flatten(), k_min, k_max)
            if idx.numel() == 0:
                continue

            selected = patches[bi, idx]
            refined = F.pixel_unshuffle(selected, 2)
            refined = self.expert(refined)
            refined = F.pixel_shuffle(refined, 2)
            refined = refined * scores[bi, idx].view(-1, 1, 1, 1)

            delta[bi, idx] = refined
            if track_routes:
                self.last_route_mask[bi].view(-1)[idx] = True
            total_routed += int(idx.numel())

        self.last_route_count = float(total_routed) / max(b, 1)
        self.last_route_density = float(total_routed) / max(b * num_patches, 1)
        delta = delta.view(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, hp, wp)
        return delta[..., :h, :w]


class SFRFullBottleneck(nn.Module):
    """Full SFR bottleneck with dynamic routing and supervision-ready route logits."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        e: float = 1.0,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 0,
        use_local: bool = True,
        route_thresh: float = 0.45,
        route_floor: float = 0.0,
    ):
        """Initialize the full routed bottleneck."""
        super().__init__()
        hidden = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.local = DWConv(hidden, hidden, 3, 1) if use_local else None
        self.expert = SparseSubpixelExpertFull(
            hidden,
            patch_size=patch_size,
            route_ratio=route_ratio,
            min_regions=min_regions,
            e=e,
            route_thresh=route_thresh,
            route_floor=route_floor,
        )
        self.cv2 = Conv(hidden, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the full routed bottleneck."""
        y = self.cv1(x)
        y = (self.local(y) if self.local is not None else y) + self.expert(y)
        y = self.cv2(y)
        return x + y if self.add else y


class SFRFullKBlock(nn.Module):
    """Kernel-aware full SFR bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 0,
        use_local: bool = True,
        route_thresh: float = 0.45,
        route_floor: float = 0.0,
        k: int = 3,
    ):
        """Initialize the kernel-aware full routed bottleneck."""
        super().__init__()
        hidden = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.local = Bottleneck(hidden, hidden, shortcut=False, g=g, k=(k, k), e=1.0) if use_local else None
        self.expert = SparseSubpixelExpertFull(
            hidden,
            patch_size=patch_size,
            route_ratio=route_ratio,
            min_regions=min_regions,
            e=e,
            route_thresh=route_thresh,
            route_floor=route_floor,
        )
        self.cv2 = Conv(hidden, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the kernel-aware full routed bottleneck."""
        y = self.cv1(x)
        y = (self.local(y) if self.local is not None else y) + self.expert(y)
        y = self.cv2(y)
        return x + y if self.add else y


class SFRC2fFull(nn.Module):
    """C2f-style full SFR block with dynamic routing."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        e: float = 0.5,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 0,
        use_local: bool = True,
        route_thresh: float = 0.45,
        route_floor: float = 0.0,
    ):
        """Initialize the full routed C2f block."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            SFRFullBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                e=1.0,
                patch_size=patch_size,
                route_ratio=route_ratio,
                min_regions=min_regions,
                use_local=use_local,
                route_thresh=route_thresh,
                route_floor=route_floor,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full routed C2f block."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SFRC3kFull(C3):
    """C3k-style full SFR block with dynamic routing."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 0,
        use_local: bool = True,
        route_thresh: float = 0.45,
        route_floor: float = 0.0,
        k: int = 3,
        g: int = 1,
    ):
        """Initialize a full routed C3k block."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(
                SFRFullKBlock(
                    c_,
                    c_,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    patch_size=patch_size,
                    route_ratio=route_ratio,
                    min_regions=min_regions,
                    use_local=use_local,
                    route_thresh=route_thresh,
                    route_floor=route_floor,
                    k=k,
                )
                for _ in range(n)
            )
        )


class SFRC3k2Full(C2f):
    """C3k2-style full SFR shell with dynamic routing."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        patch_size: int = 4,
        route_ratio: float = 0.25,
        min_regions: int = 0,
        use_local: bool = True,
        route_thresh: float = 0.45,
        route_floor: float = 0.0,
        k: int = 3,
        c3k: bool = True,
        attn: bool = False,
        g: int = 1,
    ):
        """Initialize a full routed C3k2 block."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                SFRFullKBlock(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=1.0,
                    patch_size=patch_size,
                    route_ratio=route_ratio,
                    min_regions=min_regions,
                    use_local=use_local,
                    route_thresh=route_thresh,
                    route_floor=route_floor,
                    k=k,
                ),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else SFRC3kFull(
                self.c,
                self.c,
                2,
                shortcut=shortcut,
                g=g,
                e=1.0,
                patch_size=patch_size,
                route_ratio=route_ratio,
                min_regions=min_regions,
                use_local=use_local,
                route_thresh=route_thresh,
                route_floor=route_floor,
                k=k,
            )
            if c3k
            else SFRFullKBlock(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                e=1.0,
                patch_size=patch_size,
                route_ratio=route_ratio,
                min_regions=min_regions,
                use_local=use_local,
                route_thresh=route_thresh,
                route_floor=route_floor,
                k=k,
            )
            for _ in range(n)
        )
