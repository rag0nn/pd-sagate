"""Physics-Prior and QMF-Calibrated Dynamic SA-Gate (PD-SAGate) for RGB-D Fusion.

Synthesises four complementary ideas for robust RGB-D semantic segmentation:
1. **ESANet** (Seichter et al., ICRA 2021): Pretrained ResNet34-NBt1D dual encoder
   with channel-wise Squeeze-and-Excitation (SE) features.
2. **SA-Gate** (Chen et al., ECCV 2020): Two-step *Feature Separation* and
   *Pixel-Level Softmax Aggregation* (``w_rgb + w_depth = 1``).
3. **Physical Depth Sensor Prior**: Explicit missing-hole fraction ``M_hole``
   (``D <= 1e-3``) and valid-pixel Sobel discontinuity ``||grad D||_2`` penalise
   corrupted depth regions directly inside both channel separation and spatial
   Softmax logits.
4. **DynMM** (Xue & Marculescu, CVPRW 2023) & **QMF** (Zhang et al., ICML 2023):
   Sample-adaptive conditional skipping of cross-modal refinement when the
   QMF-calibrated mean depth reliability ``w_bar_depth < tau``.

Ablation modes (``PDSAGateConfig.mode``):
* ``"mask_only"``   (V2: ``Mask-SAGate``)  : ``M_hole`` prior + spatial Softmax aggregation
                                             (``beta=0``, no channel separation, no refinement).
* ``"phys_sagate"`` (V3: ``Phys-SAGate``)  : Full physics prior (``M_hole`` + Sobel) +
                                             SA-Gate separation & Softmax aggregation +
                                             cross-modal refinement (``use_dynamic_skip=False``).
* ``"full"``        (V4: ``PD-SAGate``)    : Full physics prior + SA-Gate separation &
                                             Softmax aggregation + DynMM conditional refinement
                                             skip (``use_dynamic_skip=True``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PDSAGateConfig",
    "PDSAGateFusion",
    "DepthPrior",
    "SpatialQualityHead",
    "compute_depth_prior_maps",
]

_VALID_MODES = ("full", "phys_sagate", "mask_only")


def _inv_softplus(x: float) -> float:
    """Inverse of softplus, used to initialise positive learnable scalars."""
    return math.log(math.expm1(max(x, 1e-6)))


class SpatialQualityHead(nn.Module):
    """Lightweight 3x3 DWConv + GroupNorm + ReLU + 1x1 Conv producing spatial quality logits."""

    def __init__(self, in_ch: int, out_ch: int = 1, groups: int = 8) -> None:
        super().__init__()
        g = min(groups, in_ch)
        while in_ch % g != 0 and g > 1:
            g -= 1
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.gn = nn.GroupNorm(g, in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(F.relu(self.gn(self.dw(x)), inplace=True))


@torch.no_grad()
def compute_depth_prior_maps(
    depth_raw: torch.Tensor,
    size: Tuple[int, int],
    eps_d: float = 1e-3,
    grad_clip: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute physical depth-sensor priors ``(M_hole, ||grad D||_2)`` at feature resolution.

    Args:
        depth_raw: ``B x 1 x H0 x W0`` raw depth in metres (``<= eps_d`` = missing hole).
        size: target ``(H, W)`` of the encoder stage feature map.
        eps_d: threshold below which a depth pixel is considered a missing hole.
        grad_clip: gradient magnitudes (metres / pixel) are clipped and normalised to ``[0, 1]``.

    Returns:
        ``(hole_frac, grad_mag)`` each ``B x 1 x H x W`` in ``[0, 1]``.
    """
    hole = (depth_raw <= eps_d).float()
    kx = depth_raw.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(2, 3)
    d_pad = F.pad(depth_raw, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(d_pad, kx)
    gy = F.conv2d(d_pad, ky)
    valid = 1.0 - F.max_pool2d(hole, 3, stride=1, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12) * valid
    grad = grad.clamp(max=grad_clip) / grad_clip

    if hole.shape[-2:] != tuple(size):
        hole = F.adaptive_avg_pool2d(hole, size)
        grad = F.adaptive_avg_pool2d(grad, size)
    return hole, grad


class DepthPrior(nn.Module):
    """Learnable physical depth penalty ``P_depth = alpha * M_hole + beta * ||grad D||_2``."""

    def __init__(
        self,
        alpha_init: float = 3.0,
        beta_init: float = 0.5,
        eps_d: float = 1e-3,
        use_sobel: bool = True,
    ) -> None:
        super().__init__()
        self._alpha = nn.Parameter(torch.tensor(_inv_softplus(alpha_init)))
        self.use_sobel = use_sobel
        if use_sobel and beta_init > 0:
            self._beta = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
        else:
            self.register_parameter("_beta", None)
        self.eps_d = eps_d

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha)

    @property
    def beta(self) -> torch.Tensor:
        if self._beta is None:
            return self._alpha.new_zeros(())
        return F.softplus(self._beta)

    def forward(
        self, depth_raw: torch.Tensor, size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hole, grad = compute_depth_prior_maps(depth_raw, size, self.eps_d)
        if self.use_sobel and self._beta is not None:
            p_depth = self.alpha * hole + self.beta * grad
        else:
            p_depth = self.alpha * hole
        return p_depth, hole, grad


@dataclass
class PDSAGateConfig:
    mode: str = "full"               # "full" (V4), "phys_sagate" (V3), "mask_only" (V2)
    use_prior: bool = True
    use_dynamic_skip: bool = True
    tau: float = 0.28                # dynamic skip threshold on mean depth weight w_bar_depth
    gate_temperature: float = 0.05   # soft gate temperature during training
    alpha_init: float = 6.0          # initial hole penalty weight
    beta_init: float = 0.15          # initial Sobel discontinuity penalty weight
    gamma_scale_init: float = 2.0    # balanced aggregation scale (w_rgb*r + w_depth*(r + gamma*d))
    eps_d: float = 1e-3              # hole threshold in metres
    refine_type: str = "dw"          # "dw" (depthwise-separable) or "dense"


class PDSAGateFusion(nn.Module):
    """Physics-Prior & QMF-Calibrated Dynamic SA-Gate fusion block."""

    def __init__(self, channels: int, cfg: Optional[PDSAGateConfig] = None, **kwargs) -> None:
        super().__init__()
        cfg = cfg or PDSAGateConfig()
        for k, v in kwargs.items():
            if not hasattr(cfg, k):
                raise TypeError(f"Unknown PDSAGateConfig field '{k}'")
            setattr(cfg, k, v)
        if cfg.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {cfg.mode!r}")
        if cfg.mode != "full":
            cfg.use_dynamic_skip = False
        self.cfg = cfg
        self.channels = channels
        C = channels

        use_sobel = cfg.mode in ("full", "phys_sagate")
        self.has_separation = cfg.mode in ("full", "phys_sagate")
        self.has_refine = cfg.mode in ("full", "phys_sagate")

        # Physical sensor prior P_depth(u, v)
        if cfg.use_prior:
            self.depth_prior = DepthPrior(
                alpha_init=cfg.alpha_init,
                beta_init=cfg.beta_init if use_sobel else 0.0,
                eps_d=cfg.eps_d,
                use_sobel=use_sobel,
            )

        # Step 1: Physics-Assisted Channel Separation (SA-Gate Separation)
        if self.has_separation:
            mid = max(16, C // 4)
            self.sep_rgb = nn.Sequential(
                nn.Linear(C, mid, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(mid, C, bias=True),
            )
            self.sep_depth = nn.Sequential(
                nn.Linear(C + 2, mid, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(mid, C, bias=True),
            )

        # Step 2: Physics-Informed Pixel-Level Softmax Aggregation (SA-Gate Aggregation)
        self.qual_rgb = SpatialQualityHead(C, 1)
        self.qual_depth = SpatialQualityHead(C, 1)
        self.register_buffer("_gamma_scale", torch.tensor(float(cfg.gamma_scale_init)))

        # Step 3: DynMM Conditional Cross-Modal Refinement Block
        if self.has_refine:
            g = min(8, C)
            while C % g != 0 and g > 1:
                g -= 1
            if cfg.refine_type == "dense":
                self.refine = nn.Sequential(
                    nn.Conv2d(2 * C, C, 3, padding=1, bias=False),
                    nn.GroupNorm(g, C),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(C, C, 3, padding=1, bias=False),
                    nn.GroupNorm(g, C),
                )
            elif cfg.refine_type == "dw":
                self.refine = nn.Sequential(
                    nn.Conv2d(2 * C, 2 * C, 3, padding=1, groups=2 * C, bias=False),
                    nn.Conv2d(2 * C, C, 1, bias=False),
                    nn.GroupNorm(g, C),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
                    nn.Conv2d(C, C, 1, bias=False),
                    nn.GroupNorm(g, C),
                )
            else:
                raise ValueError(f"refine_type must be 'dw' or 'dense', got {cfg.refine_type!r}")

        self._init_weights()

    @property
    def gamma_scale(self) -> torch.Tensor:
        return self._gamma_scale

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init last layer of quality heads => clean pixel logits start at 0 (w_rgb = w_depth = 0.5)
        for qh in (self.qual_rgb, self.qual_depth):
            nn.init.zeros_(qh.pw.weight)
            nn.init.zeros_(qh.pw.bias)

        # Zero-init last layer of channel separation MLPs => 2 * sigmoid(0) == 1.0 (exact identity start)
        if self.has_separation:
            nn.init.zeros_(self.sep_rgb[-1].weight)
            nn.init.zeros_(self.sep_rgb[-1].bias)
            nn.init.zeros_(self.sep_depth[-1].weight)
            nn.init.zeros_(self.sep_depth[-1].bias)

        # Zero-init last GroupNorm of residual refinement => identity start
        if self.has_refine:
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    def _soft_gate(self, w_bar_depth: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((w_bar_depth - self.cfg.tau) / self.cfg.gate_temperature)

    def forward(
        self,
        r_se: torch.Tensor,
        d_se: torch.Tensor,
        depth_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if r_se.shape != d_se.shape:
            raise ValueError(f"shape mismatch: rgb {tuple(r_se.shape)} vs depth {tuple(d_se.shape)}")
        B, C, H, W = r_se.shape
        size = (H, W)
        cfg = self.cfg
        aux: Dict[str, torch.Tensor] = {}

        # Physical Depth Prior P_depth(u, v)
        if cfg.use_prior and depth_raw is not None:
            p_depth, hole_frac, grad_mag = self.depth_prior(depth_raw, size)
        else:
            p_depth = r_se.new_zeros((B, 1, H, W))
            hole_frac = r_se.new_zeros((B, 1, H, W))
            grad_mag = r_se.new_zeros((B, 1, H, W))

        # Step 1: Physics-Assisted Channel Separation
        if self.has_separation:
            r_pool = F.adaptive_avg_pool2d(r_se, 1).flatten(1)
            d_pool = F.adaptive_avg_pool2d(d_se, 1).flatten(1)
            h_mean = hole_frac.mean(dim=(2, 3))  # B x 1
            g_mean = grad_mag.mean(dim=(2, 3))   # B x 1
            c_rgb = self.sep_rgb(r_pool).view(B, C, 1, 1)
            c_depth = self.sep_depth(torch.cat([d_pool, h_mean, g_mean], dim=1)).view(B, C, 1, 1)
            r_tilde = r_se * (1.0 + 0.15 * torch.tanh(c_rgb))
            d_tilde = d_se * (1.0 + 0.15 * torch.tanh(c_depth))
        else:
            r_tilde = r_se
            d_tilde = d_se

        # Step 2: Physics-Informed Pixel-Level Softmax Aggregation
        z_rgb = 1.5 * torch.tanh(self.qual_rgb(r_tilde))      # B x 1 x H x W
        z_depth = 1.5 * torch.tanh(self.qual_depth(d_tilde))  # B x 1 x H x W
        z_depth_eff = z_depth - p_depth                       # Penalise holes & noisy depth edges

        logits_pair = torch.cat([z_rgb, z_depth_eff], dim=1)  # B x 2 x H x W
        weights = F.softmax(logits_pair, dim=1)
        w_rgb = weights[:, 0:1]
        w_depth = weights[:, 1:2]

        # Softmax convex combination between pure RGB (r_tilde) and depth-enhanced state (r_tilde + gamma * d_tilde):
        # * When w_rgb = w_depth = 0.5 and gamma = 2.0: f_agg = r_tilde + d_tilde (exact pretrained ESANet SE-add)
        # * When w_depth -> 0 (missing depth hole): f_agg -> r_tilde (clean RGB without scale distortion)
        f_agg = w_rgb * r_tilde + w_depth * (r_tilde + self.gamma_scale * d_tilde)
        w_bar_depth = w_depth.mean(dim=(1, 2, 3))       # B

        aux.update(
            w_rgb=w_rgb,
            w_depth=w_depth,
            w_depth_bar=w_bar_depth,
            hole_frac=hole_frac,
            grad_mag=grad_mag,
            p_depth=p_depth,
            r_tilde=r_tilde,
            d_tilde=d_tilde,
        )

        if not self.has_refine:
            aux["skip_mask"] = torch.zeros(B, dtype=torch.bool, device=r_se.device)
            return f_agg, aux

        # Step 3: DynMM Conditional Cross-Modal Refinement Skip
        if not cfg.use_dynamic_skip:
            refined_delta = 0.1 * torch.tanh(self.refine(torch.cat([f_agg, d_tilde], dim=1)))
            out = f_agg + refined_delta
            aux["skip_mask"] = torch.zeros(B, dtype=torch.bool, device=r_se.device)
            return out, aux

        if self.training:
            gate = self._soft_gate(w_bar_depth).view(B, 1, 1, 1)
            refined_delta = 0.1 * torch.tanh(self.refine(torch.cat([f_agg, d_tilde], dim=1)))
            out = f_agg + gate * refined_delta
            aux["gate"] = gate.flatten()
            aux["skip_mask"] = w_bar_depth < cfg.tau
            return out, aux

        # Inference / Eval: Hard conditional skip (real FLOPs & latency saving)
        keep = w_bar_depth >= cfg.tau
        aux["skip_mask"] = ~keep
        if not keep.any():
            return f_agg, aux
        if keep.all():
            refined_delta = 0.1 * torch.tanh(self.refine(torch.cat([f_agg, d_tilde], dim=1)))
            return f_agg + refined_delta, aux

        out = f_agg.clone()
        idx = keep.nonzero(as_tuple=True)[0]
        fa_sub, dt_sub = f_agg[idx], d_tilde[idx]
        out[idx] = fa_sub + 0.1 * torch.tanh(self.refine(torch.cat([fa_sub, dt_sub], dim=1)))
        return out, aux

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"channels={self.channels}, mode={c.mode}, prior={c.use_prior}, "
            f"dynamic_skip={c.use_dynamic_skip}, tau={c.tau}"
        )
