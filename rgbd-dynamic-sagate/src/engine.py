"""Shared evaluation loop (paired clean / corrupted evaluation with PD-SAGate & QMF tracking)."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.nyuv2_dataset import IGNORE_INDEX, N_CLASSES, compute_median_frequency_weights, normalize_rgbd
from .models.losses import compute_per_sample_ce
from .utils.metrics import SegMetrics, compute_qmf_covariance


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    depth_mode: str = "refined",
    amp: bool = False,
    desc: str = "eval",
    collect_gate_stats: bool = True,
) -> Dict[str, object]:
    model.eval()
    prev_qmf_flag = getattr(model, "compute_qmf_aux", False)
    if collect_gate_stats and hasattr(model, "compute_qmf_aux"):
        model.compute_qmf_aux = True

    metrics = SegMetrics(N_CLASSES, device=device)
    class_weights = compute_median_frequency_weights().to(device)

    w_depth_sum: Dict[int, float] = {}
    skip_sum: Dict[int, float] = {}
    qmf_w_rgb: List[float] = []
    qmf_w_depth: List[float] = []
    qmf_l_rgb: List[float] = []
    qmf_l_depth: List[float] = []
    n_batches = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth_m = batch["depth"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        rgb_n, depth_n = normalize_rgbd(rgb, depth_m, depth_mode)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(rgb_n, depth_n, depth_m)
        if logits.shape[-2:] != label.shape[-2:]:
            logits = F.interpolate(logits, label.shape[-2:], mode="bilinear", align_corners=False)
        metrics.update(logits, label)

        if collect_gate_stats:
            stage_w_dep = []
            for aux in getattr(model, "fusion_aux", []):
                s = aux.get("stage", -1)
                if "w_depth_bar" in aux:
                    w_b = aux["w_depth_bar"].float()
                    w_depth_sum[s] = w_depth_sum.get(s, 0.0) + w_b.mean().item()
                    stage_w_dep.append(w_b)
                elif "w_depth" in aux:
                    w_b = aux["w_depth"].float().mean(dim=(1, 2, 3))
                    w_depth_sum[s] = w_depth_sum.get(s, 0.0) + w_b.mean().item()
                    stage_w_dep.append(w_b)
                if "skip_mask" in aux:
                    skip_sum[s] = skip_sum.get(s, 0.0) + aux["skip_mask"].float().mean().item()

                if "aux_logits_rgb" in aux and "aux_logits_depth" in aux and stage_w_dep:
                    l_r = compute_per_sample_ce(
                        aux["aux_logits_rgb"], label, class_weights, IGNORE_INDEX
                    )
                    l_d = compute_per_sample_ce(
                        aux["aux_logits_depth"], label, class_weights, IGNORE_INDEX
                    )
                    w_d_mean = torch.stack(stage_w_dep, dim=0).mean(dim=0)
                    w_r_mean = 1.0 - w_d_mean
                    qmf_w_rgb.extend(w_r_mean.cpu().tolist())
                    qmf_w_depth.extend(w_d_mean.cpu().tolist())
                    qmf_l_rgb.extend(l_r.cpu().tolist())
                    qmf_l_depth.extend(l_d.cpu().tolist())

        model.fusion_aux = []
        n_batches += 1

    if hasattr(model, "compute_qmf_aux"):
        model.compute_qmf_aux = prev_qmf_flag

    res = metrics.compute()
    if w_depth_sum:
        res["mean_w_depth_per_stage"] = {k: v / n_batches for k, v in sorted(w_depth_sum.items())}
    if skip_sum:
        res["skip_rate_per_stage"] = {k: v / n_batches for k, v in sorted(skip_sum.items())}
    if qmf_w_depth:
        res["qmf_stats"] = compute_qmf_covariance(
            qmf_w_rgb, qmf_w_depth, qmf_l_rgb, qmf_l_depth
        )
    return res
