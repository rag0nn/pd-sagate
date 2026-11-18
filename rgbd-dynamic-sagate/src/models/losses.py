"""Loss functions for PD-SAGate RGB-D semantic segmentation (SPEC_03 §4).

Total training objective:
    ``L_total = L_CE^{multi-scale} + lambda_qmf * L_QMF``

1. **MultiScaleWeightedCE**:
   Official ESANet multi-scale (1x, 1/8, 1/16, 1/32) median-frequency weighted
   cross-entropy loss.

2. **QMFQualityLoss** (Quality-Aware Multimodal Fusion, Theorem 1 — Zhang et al., ICML 2023):
   * ``L_aux = 0.5 * (mean(l_rgb) + mean(l_depth))``: supervises the stage-4
     unimodal auxiliary heads (``1x1 Conv(512, 40)``) to estimate per-sample
     modality losses ``l_rgb^{(b)}`` and ``l_depth^{(b)}``.
   * ``L_rank``: enforces negative covariance ``Cov(w_m, l_m) < 0`` via pairwise
     margin ranking between modality loss difference and Softmax weight difference:
     ``max(0, sign(l_depth - l_rgb) * (w_bar_depth - w_bar_rgb) + margin)``.
   * ``L_hole``: physical hole anchor keeping ``w_depth -> 0`` on missing depth
     pixels (``M_hole > 0.5``) while maintaining balanced weights on valid regions.
   * Combined: ``L_QMF = 0.25 * L_aux + L_rank + 0.5 * L_hole``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets.nyuv2_dataset import IGNORE_INDEX, compute_median_frequency_weights

__all__ = [
    "MultiScaleWeightedCE",
    "QMFQualityLoss",
    "TotalPDSAGateLoss",
    "compute_per_sample_ce",
]


def compute_per_sample_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute per-sample weighted cross-entropy loss ``l^{(b)}`` of shape ``(B,)``."""
    if logits.shape[-2:] != target.shape[-2:]:
        target = (
            F.interpolate(target.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest")
            .squeeze(1)
            .long()
        )
    B = logits.shape[0]
    loss_map = F.cross_entropy(
        logits.float(),
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )  # B x H x W
    valid = (target != ignore_index).float()
    if class_weights is not None:
        w_pix = class_weights[target.clamp_min(0)] * valid
    else:
        w_pix = valid
    per_sample = loss_map.view(B, -1).sum(dim=1) / w_pix.view(B, -1).sum(dim=1).clamp_min(1e-6)
    return per_sample


class MultiScaleWeightedCE(nn.Module):
    """ESANet multi-scale median-frequency weighted cross-entropy loss."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        scale_weights: Sequence[float] = (1.0, 0.4, 0.4, 0.4),
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        if class_weights is None:
            class_weights = compute_median_frequency_weights()
        self.register_buffer("class_weights", class_weights.float())
        self.scale_weights = tuple(scale_weights)
        self.ignore_index = ignore_index

    def _single_scale(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != target.shape[-2:]:
            target = (
                F.interpolate(target.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest")
                .squeeze(1)
                .long()
            )
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        loss_map = F.cross_entropy(
            logits.float(),
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        w_pix = self.class_weights[target.clamp_min(0)] * valid.float()
        return loss_map.sum() / w_pix.sum().clamp_min(1e-6)

    def forward(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self._single_scale(outputs, target)
        total = outputs[0].new_zeros(())
        for w, out in zip(self.scale_weights, outputs):
            total = total + w * self._single_scale(out, target)
        return total


class QMFQualityLoss(nn.Module):
    """QMF Theorem-1 Quality-Consistency Loss + Physical Hole Anchor (SPEC_03 §4).

    Formula:
        ``L_QMF = 0.25 * L_aux + L_rank + 0.5 * L_hole``
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        margin: float = 0.10,
        aux_weight: float = 0.25,
        rank_weight: float = 1.0,
        hole_weight: float = 0.5,
        valid_target: float = 0.48,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        if class_weights is None:
            class_weights = compute_median_frequency_weights()
        self.register_buffer("class_weights", class_weights.float())
        self.margin = margin
        self.aux_weight = aux_weight
        self.rank_weight = rank_weight
        self.hole_weight = hole_weight
        self.valid_target = valid_target
        self.ignore_index = ignore_index

    def forward(
        self,
        aux_list: List[Dict[str, torch.Tensor]],
        target: torch.Tensor,
        enable_qmf_gate_supervision: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not aux_list:
            zero = target.new_zeros((), dtype=torch.float32)
            return zero, {"loss_qmf": 0.0, "loss_aux": 0.0, "loss_rank": 0.0, "loss_hole": 0.0}

        device = target.device
        l_rgb_per_sample = None
        l_dep_per_sample = None
        loss_aux = torch.zeros((), device=device)

        # 1) Unimodal auxiliary loss L_aux from final encoder stage (stage 4)
        for aux in aux_list:
            if "aux_logits_rgb" in aux and "aux_logits_depth" in aux:
                l_rgb_per_sample = compute_per_sample_ce(
                    aux["aux_logits_rgb"], target, self.class_weights, self.ignore_index
                )
                l_dep_per_sample = compute_per_sample_ce(
                    aux["aux_logits_depth"], target, self.class_weights, self.ignore_index
                )
                loss_aux = 0.5 * (l_rgb_per_sample.mean() + l_dep_per_sample.mean())
                break

        if not enable_qmf_gate_supervision:
            # For V2 (mask_only): train detached aux linear probes for eval diagnostics only
            total_qmf = self.aux_weight * loss_aux
            w_holes, w_valids = [], []
            for aux in aux_list:
                if "w_depth" in aux and "hole_frac" in aux:
                    wd = aux["w_depth"].detach().float()
                    hole = aux["hole_frac"].detach().float()
                    if (hole > 0.5).any():
                        w_holes.append(wd[hole > 0.5].mean().item())
                    if (hole < 0.1).any():
                        w_valids.append(wd[hole < 0.1].mean().item())
            return total_qmf, {
                "loss_qmf": float(total_qmf.detach().item()),
                "loss_aux": float(loss_aux.detach().item()),
                "loss_rank": 0.0,
                "loss_hole": 0.0,
                "w_depth_on_holes": float(sum(w_holes) / len(w_holes)) if w_holes else 0.0,
                "w_depth_on_valid": float(sum(w_valids) / len(w_valids)) if w_valids else 0.0,
            }

        # 2) QMF Theorem-1 Negative Covariance Ranking Loss L_rank & Physical Hole Anchor L_hole
        rank_terms = []
        hole_terms = []
        w_holes, w_valids = [], []

        if l_rgb_per_sample is not None and l_dep_per_sample is not None:
            # Center unimodal loss difference around the natural stage-4 structural gap (~0.35)
            # so clean images stay balanced while genuine depth/RGB degradation drives ranking
            loss_diff = (l_dep_per_sample - l_rgb_per_sample - 0.35).detach()
            sign_diff = torch.where(loss_diff.abs() > 0.15, torch.sign(loss_diff), torch.zeros_like(loss_diff))
        else:
            sign_diff = None

        for aux in aux_list:
            if "w_depth" not in aux or "w_rgb" not in aux:
                continue
            w_dep = aux["w_depth"].float()
            w_rgb = aux["w_rgb"].float()
            w_bar_dep = w_dep.mean(dim=(1, 2, 3))
            w_bar_rgb = w_rgb.mean(dim=(1, 2, 3))

            if sign_diff is not None:
                active = sign_diff != 0
                if active.any():
                    rank_stage = F.relu(
                        sign_diff[active] * (w_bar_dep[active] - w_bar_rgb[active]) + self.margin
                    ).mean()
                    rank_terms.append(rank_stage)

            if "hole_frac" in aux:
                hole = aux["hole_frac"].float()
                # Suppress w_depth on known holes (M_hole > 0.5)
                hole_mask = (hole > 0.5).float()
                loss_h = (hole * (w_dep ** 2)).mean()
                if hole_mask.any():
                    loss_h = loss_h + (hole_mask * (w_dep ** 2)).sum() / hole_mask.sum().clamp_min(1.0)
                # Keep valid non-hole pixels anchored near balanced contribution (~0.48)
                valid_mask = (1.0 - hole).clamp_min(0.0)
                w_valid_mean = (w_dep * valid_mask).sum(dim=(1, 2, 3)) / valid_mask.sum(
                    dim=(1, 2, 3)
                ).clamp_min(1.0)
                loss_v = 2.0 * ((w_valid_mean - self.valid_target) ** 2).mean()
                hole_terms.append(loss_h + loss_v)

                if (hole > 0.5).any():
                    w_holes.append(w_dep.detach()[hole > 0.5].mean().item())
                if (hole < 0.1).any():
                    w_valids.append(w_dep.detach()[hole < 0.1].mean().item())

        loss_rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=device)
        loss_hole = torch.stack(hole_terms).mean() if hole_terms else torch.zeros((), device=device)
        total_qmf = (
            self.aux_weight * loss_aux
            + self.rank_weight * loss_rank
            + self.hole_weight * loss_hole
        )

        stats = {
            "loss_qmf": float(total_qmf.detach().item()),
            "loss_aux": float(loss_aux.detach().item()),
            "loss_rank": float(loss_rank.detach().item()),
            "loss_hole": float(loss_hole.detach().item()),
            "w_depth_on_holes": float(sum(w_holes) / len(w_holes)) if w_holes else 0.0,
            "w_depth_on_valid": float(sum(w_valids) / len(w_valids)) if w_valids else 0.0,
        }
        return total_qmf, stats


class TotalPDSAGateLoss(nn.Module):
    """``L_total = L_CE^{multi-scale}(Y, Y_hat) + lambda_qmf * L_QMF``."""

    def __init__(
        self,
        lambda_qmf: float = 0.20,
        class_weights: Optional[torch.Tensor] = None,
        scale_weights: Sequence[float] = (1.0, 0.4, 0.4, 0.4),
        enable_qmf_gate_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.ce = MultiScaleWeightedCE(class_weights=class_weights, scale_weights=scale_weights)
        self.qmf = QMFQualityLoss(class_weights=self.ce.class_weights)
        self.lambda_qmf = lambda_qmf
        self.enable_qmf_gate_supervision = enable_qmf_gate_supervision

    def forward(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        target: torch.Tensor,
        aux_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_ce = self.ce(outputs, target)
        if aux_list and self.lambda_qmf > 0:
            loss_qmf, stats = self.qmf(
                aux_list, target, enable_qmf_gate_supervision=self.enable_qmf_gate_supervision
            )
            total = loss_ce + self.lambda_qmf * loss_qmf
        else:
            total = loss_ce
            stats = {"loss_qmf": 0.0, "loss_aux": 0.0, "loss_rank": 0.0, "loss_hole": 0.0}
        stats["loss_ce"] = float(loss_ce.detach().item())
        stats["loss_total"] = float(total.detach().item())
        return total, stats
