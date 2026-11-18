"""Visualisation utilities for RGB-D segmentation, Softmax modality weights, and 11-scenario grids."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..datasets.corruptions import Corruption, default_eval_suite, sample_generator
from ..datasets.nyuv2_dataset import N_CLASSES

# Official NYUv2 40-class color palette (from ESANet NYUv2Base.CLASS_COLORS_40)
_PALETTE_40 = np.array([
    [0, 0, 0],       # void
    [174, 199, 232], [152, 223, 138], [31, 119, 180],  [255, 187, 120], [188, 189, 34],
    [140, 86, 75],   [255, 152, 150], [214, 39, 40],   [197, 176, 213], [148, 103, 189],
    [196, 156, 148], [23, 190, 207],  [178, 76, 76],   [247, 182, 210], [66, 188, 102],
    [219, 219, 141], [140, 57, 197],  [202, 185, 52],  [51, 176, 203],  [200, 54, 131],
    [92, 193, 61],   [78, 71, 183],   [172, 114, 82],  [255, 127, 14],  [91, 163, 138],
    [153, 98, 156],  [140, 153, 101], [158, 218, 229], [100, 125, 154], [178, 127, 135],
    [120, 185, 128], [146, 111, 194], [44, 160, 44],   [112, 128, 144], [96, 207, 209],
    [227, 119, 194], [213, 92, 176],  [94, 106, 211],  [82, 84, 163],   [100, 85, 144],
], dtype=np.uint8)


def colorize_label(label: np.ndarray) -> np.ndarray:
    """Convert ``H x W`` label in ``{-1, 0..39}`` to ``H x W x 3`` uint8 RGB."""
    idx = np.clip(label + 1, 0, N_CLASSES)
    return _PALETTE_40[idx]


def colorize_depth(depth_m: np.ndarray, vmin: float = 0.4, vmax: float = 6.0) -> np.ndarray:
    """Colorize ``H x W`` depth in metres; holes (``<= 1e-3``) are drawn in black."""
    cmap = plt.get_cmap("turbo")
    valid = depth_m > 1e-3
    norm = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    rgba[~valid] = 0
    return rgba


def save_qualitative_panel(
    out_path: str,
    rgb: torch.Tensor,
    depth_m: torch.Tensor,
    label: torch.Tensor,
    pred_base: torch.Tensor,
    pred_pdsagate: torch.Tensor,
    w_depth: Optional[torch.Tensor] = None,
    w_rgb: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
) -> None:
    """Save a 7-column publication-ready comparison figure (SPEC_04 §3.1)."""
    H, W = label.shape[-2:]
    rgb_np = (rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    dep_np = depth_m.detach().cpu().squeeze().numpy()
    lab_np = label.detach().cpu().numpy()
    pb_np = pred_base.detach().cpu().numpy()
    pp_np = pred_pdsagate.detach().cpu().numpy()

    void = lab_np < 0
    gt_col = colorize_label(lab_np)
    pb_col = colorize_label(pb_np)
    pp_col = colorize_label(pp_np)
    pb_col[void] = 0
    pp_col[void] = 0

    fig, axes = plt.subplots(1, 7, figsize=(21, 3.3), dpi=180)
    panels = [
        (rgb_np, "RGB Girdisi"),
        (colorize_depth(dep_np), "Derinlik (m, siyah=delik)"),
        (gt_col, "Gerçek Etiket (GT)"),
        (pb_col, "ESANet (SE Baseline)"),
        (pp_col, "PD-SAGate (Önerilen)"),
    ]
    for ax, (img, ttl) in zip(axes[:5], panels):
        ax.imshow(img)
        ax.set_title(ttl, fontsize=9.5, fontweight="bold")
        ax.axis("off")

    if w_depth is not None:
        wd_map = F.interpolate(
            w_depth.detach().cpu().float().mean(1, keepdim=True),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        im_d = axes[5].imshow(wd_map, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[5].set_title(r"Depth Ağırlığı $w_{\mathrm{depth}}$", fontsize=9.5, fontweight="bold")
        plt.colorbar(im_d, ax=axes[5], fraction=0.046, pad=0.04)
    axes[5].axis("off")

    if w_rgb is not None:
        wr_map = F.interpolate(
            w_rgb.detach().cpu().float().mean(1, keepdim=True),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        im_r = axes[6].imshow(wr_map, cmap="magma", vmin=0.0, vmax=1.0)
        axes[6].set_title(r"RGB Ağırlığı $w_{\mathrm{rgb}}$", fontsize=9.5, fontweight="bold")
        plt.colorbar(im_r, ax=axes[6], fraction=0.046, pad=0.04)
    axes[6].axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


SCENARIO_TITLES_TR = {
    "clean": "1. Temiz (Clean)",
    "depth_holes@s1": "2. Depth Holes s1 (%15 Delik)",
    "depth_holes@s2": "3. Depth Holes s2 (%30 Delik)",
    "depth_holes@s3": "4. Depth Holes s3 (%50 Delik)",
    "axial_noise@s1": "5. Axial Noise s1 (2x Kinect z²)",
    "axial_noise@s2": "6. Axial Noise s2 (5x Kinect z²)",
    "axial_noise@s3": "7. Axial Noise s3 (10x Kinect z²)",
    "low_light@s2": "8. Low-Light s2 (γ=2.5 + Gürültü)",
    "low_light@s3": "9. Low-Light s3 (γ=3.2 Aşırı Karanlık)",
    "motion_blur@s2": "10. Motion Blur s2 (L=9 Bulanıklık)",
    "mixed@s2": "11. Mixed s2 (Sol Karanlık / Sağ Delik)",
}


def save_corruptions_11_scenarios_grid(
    rgb_clean: torch.Tensor,
    depth_clean: torch.Tensor,
    label: torch.Tensor,
    frame_index: int,
    out_dir: str,
    seed: int = 2024,
) -> List[dict]:
    """Generate ``fig_corruptions_11_scenarios.{pdf,png}`` showing a single NYUv2 test image
    under all 11 evaluation scenarios (Clean + 10 Corruptions) as RGB & Depth pairs (SPEC_04 §3.1).
    """
    os.makedirs(out_dir, exist_ok=True)
    suite: List[Corruption] = default_eval_suite()
    assert len(suite) == 11, f"Expected 11 scenarios, got {len(suite)}"

    # Layout: 3 rows x 4 cols = 12 cells (11 scenarios + 1 Ground Truth & Legend cell)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10.2), dpi=200)
    axes_flat = axes.flatten()

    clean_rgb_np = (rgb_clean.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    clean_dep_np = depth_clean.squeeze().cpu().numpy()
    lab_np = label.cpu().numpy()

    scenario_stats = []
    for idx, c in enumerate(suite):
        g = sample_generator(seed, frame_index)
        rgb_c, depth_c = c(rgb_clean.clone(), depth_clean.clone(), g)
        rgb_np = (rgb_c.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        dep_np = depth_c.squeeze().cpu().numpy()
        dep_col = colorize_depth(dep_np)

        # Concatenate RGB (left) and Depth (right) side-by-side with a thin white separator
        sep = np.full((rgb_np.shape[0], 6, 3), 255, dtype=np.uint8)
        pair_img = np.concatenate([rgb_np, sep, dep_col], axis=1)

        hole_pct = float((dep_np <= 1e-3).mean() * 100.0)
        valid_mask = (dep_np > 1e-3) & (clean_dep_np > 1e-3)
        depth_rmse_cm = (
            float(np.sqrt(np.mean((dep_np[valid_mask] - clean_dep_np[valid_mask]) ** 2)) * 100.0)
            if valid_mask.any()
            else 0.0
        )
        rgb_mean = float(rgb_c.mean().item())

        scenario_stats.append(
            {
                "tag": c.tag,
                "title": SCENARIO_TITLES_TR.get(c.tag, c.tag),
                "hole_pct": hole_pct,
                "depth_rmse_cm": depth_rmse_cm,
                "rgb_mean_intensity": rgb_mean,
            }
        )

        ax = axes_flat[idx]
        ax.imshow(pair_img)
        title_str = SCENARIO_TITLES_TR.get(c.tag, c.tag)
        sub_str = f"RGB Ort: {rgb_mean:.2f} | Delik: %{hole_pct:.1f} | ΔD RMSE: {depth_rmse_cm:.1f} cm"
        ax.set_title(f"{title_str}\n{sub_str}", fontsize=8.8, fontweight="bold")
        ax.axis("off")

    # 12th cell: Ground Truth semantic mask + Clean Depth side-by-side reference
    ax_gt = axes_flat[11]
    gt_col = colorize_label(lab_np)
    sep = np.full((gt_col.shape[0], 6, 3), 255, dtype=np.uint8)
    ref_pair = np.concatenate([gt_col, sep, colorize_depth(clean_dep_np)], axis=1)
    ax_gt.imshow(ref_pair)
    ax_gt.set_title(
        f"Referans: Gerçek Etiket (Sol) & Temiz Derinlik (Sağ)\nNYUv2 Test Örneği (Frame #{frame_index}, 40 Sınıf)",
        fontsize=8.8,
        fontweight="bold",
        color="#1f4e78",
    )
    ax_gt.axis("off")

    fig.suptitle(
        f"NYU Depth V2 Tekil Test Sahnesi Üzerinde 11 Değerlendirme Senaryosu (Sol: RGB, Sağ: Derinlik Haritası — Frame #{frame_index})",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_corruptions_11_scenarios.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_corruptions_11_scenarios.png"), bbox_inches="tight")
    plt.close(fig)
    return scenario_stats
