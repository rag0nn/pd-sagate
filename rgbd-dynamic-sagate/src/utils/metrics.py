"""Evaluation metrics: segmentation quality, QMF empirical covariance, calibration, and resource cost."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Dict, Sequence

import numpy as np
import torch

# Key classes highlighted in the report (NYUv2-40 train ids)
KEY_CLASSES = {"wall": 0, "floor": 1, "table": 6, "chair": 4, "window": 8}


class SegMetrics:
    """Confusion-matrix based mIoU / pixel accuracy / class IoU + ECE.

    Matches ESANet's ``MeanIntersectionOverUnion``: void pixels (label < 0) are
    ignored and the mean is taken over classes present in the ground truth.
    """

    def __init__(self, n_classes: int = 40, n_bins: int = 15, device="cpu") -> None:
        self.n = n_classes
        self.cm = torch.zeros(n_classes, n_classes, dtype=torch.int64, device=device)
        self.n_bins = n_bins
        self.bin_conf = torch.zeros(n_bins, dtype=torch.float64, device=device)
        self.bin_acc = torch.zeros(n_bins, dtype=torch.float64, device=device)
        self.bin_cnt = torch.zeros(n_bins, dtype=torch.float64, device=device)
        self.nll_sum = 0.0
        self.n_pix = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, label: torch.Tensor) -> None:
        """``logits``: B x K x H x W (already at label resolution); ``label``: B x H x W."""
        label = label.to(logits.device)
        valid = label >= 0
        prob = logits.float().softmax(1)
        conf, pred = prob.max(1)
        t, p, c = label[valid], pred[valid], conf[valid]
        idx = t * self.n + p
        self.cm += torch.bincount(idx, minlength=self.n ** 2).view(self.n, self.n).to(self.cm.device)
        correct = (t == p).double()
        b = torch.clamp((c * self.n_bins).long(), max=self.n_bins - 1)
        self.bin_cnt += torch.bincount(b, minlength=self.n_bins).double().to(self.bin_cnt.device)
        self.bin_conf += torch.bincount(b, weights=c.double(), minlength=self.n_bins).to(self.bin_conf.device)
        self.bin_acc += torch.bincount(b, weights=correct, minlength=self.n_bins).to(self.bin_acc.device)
        logp = torch.log(prob.clamp_min(1e-12))
        self.nll_sum += -logp.permute(0, 2, 3, 1)[valid].gather(1, t[:, None]).sum().item()
        self.n_pix += int(valid.sum().item())

    def merge(self, other: "SegMetrics") -> None:
        self.cm += other.cm.to(self.cm.device)
        self.bin_cnt += other.bin_cnt.to(self.bin_cnt.device)
        self.bin_conf += other.bin_conf.to(self.bin_conf.device)
        self.bin_acc += other.bin_acc.to(self.bin_acc.device)
        self.nll_sum += other.nll_sum
        self.n_pix += other.n_pix

    def compute(self) -> Dict[str, object]:
        cm = self.cm.double()
        tp = cm.diag()
        gt = cm.sum(1)
        pr = cm.sum(0)
        iou = tp / (gt + pr - tp).clamp_min(1)
        present = gt > 0
        miou = iou[present].mean().item()
        pacc = (tp.sum() / cm.sum().clamp_min(1)).item()
        macc = (tp[present] / gt[present]).mean().item()
        nz = self.bin_cnt > 0
        ece = (
            (self.bin_cnt[nz] / self.bin_cnt.sum())
            * (self.bin_acc[nz] / self.bin_cnt[nz] - self.bin_conf[nz] / self.bin_cnt[nz]).abs()
        ).sum().item()
        out = {
            "mIoU": 100 * miou,
            "pixel_acc": 100 * pacc,
            "mean_acc": 100 * macc,
            "ECE": 100 * ece,
            "NLL": self.nll_sum / max(self.n_pix, 1),
            "class_IoU": (100 * iou).tolist(),
        }
        for name, k in KEY_CLASSES.items():
            out[f"IoU_{name}"] = 100 * iou[k].item()
        return out


# --------------------------------------------------------------------------- #
# QMF Theorem-1 Empirical Covariance
# --------------------------------------------------------------------------- #
def compute_qmf_covariance(
    w_rgb: Sequence[float] | np.ndarray,
    w_depth: Sequence[float] | np.ndarray,
    l_rgb: Sequence[float] | np.ndarray,
    l_depth: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    """Compute empirical covariance ``Cov(w_m, l_m)`` (QMF Theorem 1, Zhang et al. ICML 2023).

    For each sample ``b``, across the two modalities ``m in {rgb, depth}``:
        ``cov_b = 0.5 * (w_depth[b] - w_rgb[b]) * (l_depth[b] - l_rgb[b])``

    Also computes the population covariance across all ``2N`` ``(w_m, l_m)`` pairs.
    Negative values (``Cov(w_m, l_m) < 0``) confirm that the Softmax gate assigns
    strictly lower weight to whichever modality incurs higher loss.
    """
    wr = np.asarray(w_rgb, dtype=np.float64)
    wd = np.asarray(w_depth, dtype=np.float64)
    lr = np.asarray(l_rgb, dtype=np.float64)
    ld = np.asarray(l_depth, dtype=np.float64)
    if wr.size == 0:
        return {"qmf_cov_pair": 0.0, "qmf_cov_pooled": 0.0, "l_rgb_mean": 0.0, "l_depth_mean": 0.0}

    pair_cov = 0.5 * (wd - wr) * (ld - lr)
    w_all = np.concatenate([wr, wd])
    l_all = np.concatenate([lr, ld])
    pooled_cov = float(np.mean((w_all - w_all.mean()) * (l_all - l_all.mean())))
    return {
        "qmf_cov_pair": float(np.mean(pair_cov)),
        "qmf_cov_pooled": pooled_cov,
        "l_rgb_mean": float(np.mean(lr)),
        "l_depth_mean": float(np.mean(ld)),
        "w_rgb_mean": float(np.mean(wr)),
        "w_depth_mean": float(np.mean(wd)),
    }


# --------------------------------------------------------------------------- #
# Resource metrics
# --------------------------------------------------------------------------- #
def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_flops(model: torch.nn.Module, inputs: Sequence[torch.Tensor]) -> float:
    """Forward FLOPs (GFLOPs, 1 MAC = 2 FLOPs) via ``torch.utils.flop_counter``."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    with FlopCounterMode(display=False) as fc:
        model(*inputs)
    model.train(was_training)
    return fc.get_total_flops() / 1e9


@torch.inference_mode()
def measure_latency(
    model: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    n_warmup: int = 20,
    n_iter: int = 100,
    amp: bool = False,
) -> Dict[str, float]:
    """Mean / std latency (ms/frame), FPS and peak VRAM (MB) of a forward pass."""
    model.eval()
    dev = inputs[0].device
    cuda = dev.type == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.float16) if (amp and cuda) else nullcontext()
    with ctx:
        for _ in range(n_warmup):
            model(*inputs)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(n_iter):
            if cuda:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                model(*inputs)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                model(*inputs)
                times.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(times)
    bs = inputs[0].shape[0]
    return {
        "latency_ms": float(t.mean() / bs),
        "latency_std_ms": float(t.std() / bs),
        "fps": float(1e3 * bs / t.mean()),
        "peak_vram_MB": float(torch.cuda.max_memory_allocated() / 2**20) if cuda else float("nan"),
    }
