"""Generate publication-ready figures, 11-scenario visual grid, architecture diagram, and LaTeX tables (SPEC_04 §3)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.nyuv2_dataset import NYUv2  # noqa: E402
from src.utils.visualization import save_corruptions_11_scenarios_grid  # noqa: E402

SCENARIO_LABELS = {
    "clean": "Temiz (Clean NYUv2 Test)",
    "depth_holes@s1": "Depth Holes s1 (%15 Delik)",
    "depth_holes@s2": "Depth Holes s2 (%30 Delik)",
    "depth_holes@s3": "Depth Holes s3 (%50 Delik)",
    "axial_noise@s1": "Axial Noise s1 (2x Kinect $z^2$)",
    "axial_noise@s2": "Axial Noise s2 (5x Kinect $z^2$)",
    "axial_noise@s3": "Axial Noise s3 (10x Kinect $z^2$)",
    "low_light@s2": "Low-Light RGB s2 ($\\gamma=2.5$)",
    "low_light@s3": "Low-Light RGB s3 ($\\gamma=3.2$)",
    "motion_blur@s2": "Motion Blur RGB s2 ($L=9$)",
    "mixed@s2": "Mixed Cross-Modal s2",
}

VARIANT_LABELS = {
    "V1_SE_Baseline": "V1: ESANet (SE Baseline)",
    "V2_Mask_SAGate": "V2: Mask-SAGate (Sadece M_hole)",
    "V3_Phys_SAGate": "V3: Phys-SAGate (Skip Yok)",
    "V4_PD_SAGate": "V4: PD-SAGate (Tam Model, Önerilen)",
}


def fmt_mean_std(d: Dict[str, object], key: str, bold: bool = False) -> str:
    m = d.get(f"{key}_mean", d.get(key, 0.0))
    s = d.get(f"{key}_std", 0.0)
    txt = f"{m:.2f} $\\pm$ {s:.2f}" if s > 1e-4 else f"{m:.2f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def draw_architecture_diagram(fig_dir: str) -> None:
    """Generate ESANet Fig-2 style clean architecture diagram for PD-SAGate."""
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 6.2), dpi=220)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def add_box(x, y, w, h, text, fc="#e8f0fe", ec="#1a73e8", fontsize=8.8, bold=True):
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color="#111111",
        )

    def add_arrow(x0, y0, x1, y1, color="#333333", style="->"):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle=style, color=color, lw=1.6),
        )

    # Title
    ax.text(
        7.25,
        5.92,
        "Önerilen Mimari: ESANet-R34-NBt1D Omurgası Üzerinde Fizik-Öncüllü ve QMF-Kalibreli Dinamik SA-Gate (PD-SAGate)",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#0d3b66",
    )

    # Left: Dual Encoder Inputs & Pretrained SE
    add_box(0.3, 4.3, 2.0, 0.95, "RGB Girdisi\n(3 × 480 × 640)\nResNet34-NBt1D\n+ Ön-Eğitimli SE", fc="#dceefb", ec="#1d4ed8")
    add_box(0.3, 2.5, 2.0, 0.95, "Derinlik Girdisi\n(1 × 480 × 640)\nResNet34-NBt1D\n+ Ön-Eğitimli SE", fc="#e0f2fe", ec="#0284c7")
    add_box(
        0.3,
        0.65,
        2.0,
        1.15,
        "Fiziksel Sensör Önseli\n(DepthPrior, @no_grad)\n• Delik: M_hole = I(D ≤ 10⁻³)\n• Kenar: ||∇D||₂ (Sobel)\nP_depth = α·M_hole + β·||∇D||₂",
        fc="#fef3c7",
        ec="#d97706",
        fontsize=8.0,
    )

    # Step 1: Physics-Assisted Channel Separation
    add_box(
        2.9,
        2.6,
        2.5,
        2.6,
        "Adım 1: Fizik-Destekli\nKanal Ayırma (SA-Gate)\n\n"
        "• c_rgb = MLP(GAP(r_se))\n"
        "• c_depth = MLP([GAP(d_se),\n   Mean(M_hole), Mean(||∇D||₂)])\n\n"
        "r̃ = r_se ⊙ (1 + 0.15 tanh(c_rgb))\n"
        "d̃ = d_se ⊙ (1 + 0.15 tanh(c_depth))",
        fc="#f3e8ff",
        ec="#7e22ce",
        fontsize=8.0,
    )

    # Step 2: Physics-Informed Pixel Softmax Aggregation
    add_box(
        6.0,
        2.6,
        2.85,
        2.6,
        "Adım 2: Fizik-Öncüllü Piksel\nSoftmax Toplama (SA-Gate)\n\n"
        "• z_rgb = 1.5 tanh(SpatialQual(r̃))\n"
        "• z_depth^eff = 1.5 tanh(SQ(d̃)) − P_depth\n"
        "• [w_rgb, w_depth] = Softmax(z)\n"
        "   (w_rgb + w_depth = 1)\n\n"
        "F_agg = w_rgb⊙r̃ + w_depth⊙(r̃ + γd̃)",
        fc="#dcfce7",
        ec="#15803d",
        fontsize=8.0,
    )

    # Step 3: DynMM Conditional Refinement Skip
    add_box(
        9.45,
        2.6,
        2.55,
        2.6,
        "Adım 3: DynMM Koşullu\nÇapraz İyileştirme Atlaması\n\n"
        "Ortalama Derinlik Güveni:\n"
        "w̄_depth = Mean_hw(w_depth)\n\n"
        "• w̄_depth ≥ τ (0.28):\n"
        "  F_out = F_agg + 0.1 tanh(Refine)\n"
        "• w̄_depth < τ (Bozuk Depth):\n"
        "  Refine ATLANIR (F_out = F_agg)",
        fc="#ffe4e6",
        ec="#be123c",
        fontsize=8.1,
    )

    # Right: Context Module & Multi-Scale Decoder
    add_box(
        12.45,
        3.25,
        1.8,
        1.5,
        "PPM Bağlam\nModülü &\nÇok Ölçekli\nDecoder\n(1x, 1/8, 1/16, 1/32)\n40 Sınıf Çıktı",
        fc="#e0e7ff",
        ec="#3730a3",
        fontsize=8.4,
    )

    # Bottom: QMF Theorem-1 Quality Calibration Loss (Training Only)
    add_box(
        3.5,
        0.55,
        9.0,
        1.35,
        "Eğitimde QMF Teorem-1 Kalite-Tutarlılığı ve Sıralama Kaybı (Çıkarımda Sıfır Ek Maliyet)\n"
        "• Son aşama (s=4) tekil yardımcı başlıklar: aux_head_rgb(r̃), aux_head_depth(d̃) → l_rgb^(b), l_depth^(b)\n"
        "• Negatif Kovaryans Sıralama Kaybı (Cov(w_m, l_m) < 0):  L_rank = max(0, sign(l_depth − l_rgb)·(w̄_depth − w̄_rgb) + 0.10)\n"
        "• Toplam Kayıp:  L_total = L_CE^multi-scale + λ_qmf · (0.25 L_aux + L_rank + 0.5 L_hole)",
        fc="#fff7ed",
        ec="#c2410c",
        fontsize=8.0,
    )

    # Arrows
    add_arrow(2.3, 4.75, 2.9, 4.4)
    add_arrow(2.3, 2.95, 2.9, 3.3)
    add_arrow(2.3, 1.25, 2.9, 2.8, color="#b45309")
    add_arrow(2.3, 1.25, 6.0, 2.8, color="#b45309")
    add_arrow(5.4, 3.9, 6.0, 3.9)
    add_arrow(8.85, 3.9, 9.45, 3.9)
    add_arrow(12.0, 3.9, 12.45, 3.9)
    add_arrow(7.4, 2.6, 7.4, 1.9, color="#c2410c", style="<->")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_architecture_pdsagate.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "fig_architecture_pdsagate.png"), bbox_inches="tight")
    plt.close(fig)


def plot_robustness_curves(variants: Dict[str, Dict[str, Dict[str, object]]], fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), dpi=200)

    styles = {
        "V1_SE_Baseline": ("#d62728", "o--", "V1: ESANet (SE Baseline)"),
        "V2_Mask_SAGate": ("#ff7f0e", "s-.", "V2: Mask-SAGate"),
        "V3_Phys_SAGate": ("#1f77b4", "^-", "V3: Phys-SAGate (Skip Yok)"),
        "V4_PD_SAGate": ("#2ca02c", "D-", "V4: PD-SAGate (Tam Model)"),
    }

    # Panel A: Missing Depth Holes
    hole_tags = ["clean", "depth_holes@s1", "depth_holes@s2", "depth_holes@s3"]
    hole_x = [0, 15, 30, 50]
    for v_key, (col, marker, label) in styles.items():
        if v_key not in variants:
            continue
        vd = variants[v_key]
        ys = [vd[t]["mIoU_mean"] for t in hole_tags if t in vd]
        es = [vd[t].get("mIoU_std", 0.0) for t in hole_tags if t in vd]
        axes[0].errorbar(
            hole_x[: len(ys)], ys, yerr=es, fmt=marker, color=col, label=label, linewidth=2.0, capsize=3
        )
    axes[0].set_title("(a) Eksik Derinlik Deliklerine Dayanıklılık ($D=0$)", fontsize=10.0, fontweight="bold")
    axes[0].set_xlabel("Eksik Derinlik Oranı (%)", fontsize=9.2)
    axes[0].set_ylabel("mIoU (%) — NYUv2 Test", fontsize=9.2)
    axes[0].set_xticks(hole_x)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.0, loc="lower left")

    # Panel B: Axial Depth Noise
    ax_tags = ["clean", "axial_noise@s1", "axial_noise@s2", "axial_noise@s3"]
    ax_x = [0, 2, 5, 10]
    for v_key, (col, marker, label) in styles.items():
        if v_key not in variants:
            continue
        vd = variants[v_key]
        ys = [vd[t]["mIoU_mean"] for t in ax_tags if t in vd]
        es = [vd[t].get("mIoU_std", 0.0) for t in ax_tags if t in vd]
        axes[1].errorbar(
            ax_x[: len(ys)], ys, yerr=es, fmt=marker, color=col, label=label, linewidth=2.0, capsize=3
        )
    axes[1].set_title(r"(b) Aksiyal Sensör Gürültüsüne Dayanıklılık ($\sigma_z \propto z^2$)", fontsize=10.0, fontweight="bold")
    axes[1].set_xlabel("Gürültü Çarpanı", fontsize=9.2)
    axes[1].set_ylabel("mIoU (%) — NYUv2 Test", fontsize=9.2)
    axes[1].set_xticks(ax_x)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.0, loc="lower left")

    # Panel C: RGB Low-Light, Motion Blur & Mixed Corruptions
    rgb_tags = ["clean", "low_light@s2", "low_light@s3", "motion_blur@s2", "mixed@s2"]
    rgb_lbls = ["Clean", "LowLight\n(γ=2.5)", "LowLight\n(γ=3.2)", "Motion\nBlur", "Mixed\n(RGB+D)"]
    rx = np.arange(len(rgb_tags))
    for v_key, (col, marker, label) in styles.items():
        if v_key not in variants:
            continue
        vd = variants[v_key]
        ys = [vd[t]["mIoU_mean"] for t in rgb_tags if t in vd]
        es = [vd[t].get("mIoU_std", 0.0) for t in rgb_tags if t in vd]
        axes[2].errorbar(
            rx[: len(ys)], ys, yerr=es, fmt=marker, color=col, label=label, linewidth=2.0, capsize=3
        )
    axes[2].set_title("(c) Düşük Işık, Bulanıklık ve Karma Bozulmalar", fontsize=10.0, fontweight="bold")
    axes[2].set_xticks(rx)
    axes[2].set_xticklabels(rgb_lbls, fontsize=8.5)
    axes[2].set_ylabel("mIoU (%) — NYUv2 Test", fontsize=9.2)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=8.0, loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_robustness_curves.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "fig_robustness_curves.png"), bbox_inches="tight")
    plt.close(fig)


def plot_ablation_and_qmf(variants: Dict[str, Dict[str, Dict[str, object]]], fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    tags = ["clean", "depth_holes@s2", "depth_holes@s3", "axial_noise@s2", "low_light@s2", "mixed@s2"]
    labels = ["Clean", "Holes 30%", "Holes 50%", "Axial 5x", "Low-Light", "Mixed"]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.3), dpi=200)
    x = np.arange(len(tags))
    width = 0.20
    v_order = ["V1_SE_Baseline", "V2_Mask_SAGate", "V3_Phys_SAGate", "V4_PD_SAGate"]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    # Panel A: 4-Variant Ablation Bars
    for i, (vk, col) in enumerate(zip(v_order, colors)):
        if vk not in variants:
            continue
        vd = variants[vk]
        vals = [vd[t]["mIoU_mean"] for t in tags if t in vd]
        errs = [vd[t].get("mIoU_std", 0.0) for t in tags if t in vd]
        axes[0].bar(
            x[: len(vals)] + (i - 1.5) * width,
            vals,
            width,
            yerr=errs,
            label=VARIANT_LABELS[vk],
            color=col,
            alpha=0.9,
            capsize=2,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylabel("mIoU (%)", fontsize=9.2)
    axes[0].set_title("(a) 4 Varyantlı Ablasyon Karşılaştırması", fontsize=10.0, fontweight="bold")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=7.5, loc="lower left")

    # Panel B: Stage-wise Mean Depth Softmax Weight w_bar_depth across scenarios for V4
    if "V4_PD_SAGate" in variants:
        v4 = variants["V4_PD_SAGate"]
        stages = [0, 1, 2, 3, 4]
        for t, lbl, mk in [
            ("clean", "Clean Depth", "o-"),
            ("depth_holes@s1", "Depth Holes 15%", "s--"),
            ("depth_holes@s2", "Depth Holes 30%", "^-."),
            ("depth_holes@s3", "Depth Holes 50%", "v:"),
            ("low_light@s2", "Low-Light RGB (γ=2.5)", "D-"),
        ]:
            if t in v4 and "mean_w_depth_per_stage" in v4[t]:
                ws = [
                    v4[t]["mean_w_depth_per_stage"].get(
                        str(s), v4[t]["mean_w_depth_per_stage"].get(s, 0.0)
                    )
                    for s in stages
                ]
                axes[1].plot([s + 1 for s in stages], ws, mk, linewidth=2.0, label=lbl)
        axes[1].axhline(0.28, color="crimson", linestyle="--", linewidth=1.3, label=r"Skip Eşiği $\tau=0.28$")
        axes[1].set_xticks([1, 2, 3, 4, 5])
        axes[1].set_xticklabels(
            ["Stage 0\n(1/4)", "Stage 1\n(1/4)", "Stage 2\n(1/8)", "Stage 3\n(1/16)", "Stage 4\n(1/32)"],
            fontsize=8.2,
        )
        axes[1].set_ylabel(r"Ortalama Derinlik Ağırlığı $\bar{w}_{\mathrm{depth}}$", fontsize=9.2)
        axes[1].set_title(r"(b) Aşama Bazlı $\bar{w}_{\mathrm{depth}}$ ve Dinamik Eşik", fontsize=10.0, fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(fontsize=7.8, loc="best")

    # Panel C: QMF Theorem-1 Empirical Covariance Cov(w_m, l_m) < 0 verification
    qmf_tags = ["clean", "depth_holes@s1", "depth_holes@s2", "depth_holes@s3", "axial_noise@s2", "mixed@s2"]
    qmf_lbls = ["Clean", "Holes\n15%", "Holes\n30%", "Holes\n50%", "Axial\n5x", "Mixed"]
    qx = np.arange(len(qmf_tags))
    q_width = 0.26
    for idx_v, (vk, col, vshort) in enumerate(
        [
            ("V2_Mask_SAGate", "#ff7f0e", "V2: Mask-SAGate (QMF Yok)"),
            ("V3_Phys_SAGate", "#1f77b4", "V3: Phys-SAGate (QMF Var)"),
            ("V4_PD_SAGate", "#2ca02c", "V4: PD-SAGate (Önerilen)"),
        ]
    ):
        if vk not in variants:
            continue
        vd = variants[vk]
        covs = [
            vd[t].get("qmf_stats", {}).get("qmf_cov_pair", 0.0) for t in qmf_tags if t in vd
        ]
        if covs:
            axes[2].bar(
                qx[: len(covs)] + (idx_v - 1.0) * q_width,
                covs,
                q_width,
                label=vshort,
                color=col,
                alpha=0.9,
            )
    axes[2].axhline(0.0, color="black", linewidth=1.1, linestyle="-")
    axes[2].set_xticks(qx)
    axes[2].set_xticklabels(qmf_lbls, fontsize=8.3)
    axes[2].set_ylabel(r"Ampirik Kovaryans $\mathrm{Cov}(w_m, \ell_m)$", fontsize=9.2)
    axes[2].set_title(r"(c) QMF Teorem-1 Doğrulaması ($\mathrm{Cov}(w_m, \ell_m) < 0$)", fontsize=10.0, fontweight="bold")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=7.8, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_ablation_and_qmf.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "fig_ablation_and_qmf.png"), bbox_inches="tight")
    plt.close(fig)


def write_latex_tables(data: Dict[str, object], out_tex: str) -> None:
    variants = data["variants"]
    costs = data["costs"]
    lines = []

    # Table 1: Main 11-scenario ablation table
    lines.append("% Auto-generated by scripts/generate_plots_and_tables.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{NYU Depth V2 (654 test görüntüsü) üzerinde 4 ablasyon varyantının 11 değerlendirme senaryosundaki (Temiz + 10 Bozulma) mIoU (\\%) performansı ve ortalama korunma oranı.}"
    )
    lines.append("\\label{tab:ablation_main}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Değerlendirme Senaryosu} & \\textbf{V1: ESANet (SE)} & \\textbf{V2: Mask-SAGate} & \\textbf{V3: Phys-SAGate} & \\textbf{V4: PD-SAGate (Önerilen)} \\\\"
    )
    lines.append("\\midrule")

    v_keys = ["V1_SE_Baseline", "V2_Mask_SAGate", "V3_Phys_SAGate", "V4_PD_SAGate"]
    for tag, label in SCENARIO_LABELS.items():
        vals = [
            variants[vk][tag]["mIoU_mean"] if (vk in variants and tag in variants[vk]) else -1.0
            for vk in v_keys
        ]
        best_idx = int(np.argmax(vals))
        cells = []
        for idx, vk in enumerate(v_keys):
            if vk in variants and tag in variants[vk]:
                cells.append(fmt_mean_std(variants[vk][tag], "mIoU", bold=(idx == best_idx)))
            else:
                cells.append("-")
        lbl_tex = label.replace("%", "\\%")
        lines.append(f"{lbl_tex} & " + " & ".join(cells) + " \\\\")

    # Average over the 10 corrupted scenarios + retention ratio
    corrupt_tags = [t for t in SCENARIO_LABELS if t != "clean"]
    lines.append("\\midrule")
    avg_cells, ret_cells = [], []
    avg_vals = []
    for vk in v_keys:
        if vk in variants:
            c_mious = [variants[vk][t]["mIoU_mean"] for t in corrupt_tags if t in variants[vk]]
            avg_vals.append(float(np.mean(c_mious)) if c_mious else -1.0)
        else:
            avg_vals.append(-1.0)
    best_avg = int(np.argmax(avg_vals))
    for idx, vk in enumerate(v_keys):
        if avg_vals[idx] > 0:
            clean_m = variants[vk]["clean"]["mIoU_mean"]
            ret = 100.0 * avg_vals[idx] / max(clean_m, 1e-6)
            a_str = f"{avg_vals[idx]:.2f}"
            r_str = f"{ret:.1f}\\%"
            if idx == best_avg:
                a_str = f"\\textbf{{{a_str}}}"
                r_str = f"\\textbf{{{r_str}}}"
            avg_cells.append(a_str)
            ret_cells.append(r_str)
        else:
            avg_cells.append("-")
            ret_cells.append("-")
    lines.append("\\textbf{10 Bozulma Ortalaması (mIoU)} & " + " & ".join(avg_cells) + " \\\\")
    lines.append(
        "\\textbf{Ortalama Korunma Oranı ($\\text{mIoU}_{\\text{bozuk}} / \\text{mIoU}_{\\text{temiz}}$)} & "
        + " & ".join(ret_cells)
        + " \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}\n")

    # Table 2: Hardware & Resource Cost
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Donanım ve kaynak verimliliği ölçütleri ($640\\times 480$ tek kare çıkarım, NVIDIA RTX 3050 Laptop GPU, FP16 AMP).}"
    )
    lines.append("\\label{tab:resource_cost}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Model Varyantı} & \\textbf{Parametre (M)} & \\textbf{GFLOPs} & \\textbf{Gecikme (ms)} & \\textbf{FPS} & \\textbf{Tepe VRAM (MB)} \\\\"
    )
    lines.append("\\midrule")
    for ck, clabel in [
        ("V1_SE_Baseline", "V1: ESANet-R34 SE (Baseline)"),
        ("V2_Mask_SAGate", "V2: Mask-SAGate (Sadece $M_{\\mathrm{hole}}$)"),
        ("V3_Phys_SAGate", "V3: Phys-SAGate (Skip Yok)"),
        ("V4_PD_SAGate_clean", "V4: PD-SAGate (Tam İyileştirme / Temiz)"),
        ("V4_PD_SAGate_skipped", "V4: PD-SAGate (Dinamik Atlama Aktif / Bozuk Depth)"),
    ]:
        if ck not in costs:
            continue
        c = costs[ck]
        lines.append(
            f"{clabel} & {c['params_M']:.2f} & {c['GFLOPs']:.2f} & "
            f"{c['latency_ms']:.2f} $\\pm$ {c['latency_std_ms']:.2f} & {c['fps']:.1f} & {c['peak_vram_MB']:.1f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}\n")

    os.makedirs(os.path.dirname(os.path.abspath(out_tex)), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print("Wrote LaTeX tables to", out_tex)


def generate_11_scenarios_figure(fig_dir: str, sample_idx: int = 12) -> None:
    ds = NYUv2("test", depth_mode="refined", preload=False, indices=[sample_idx])
    sample = ds[0]
    stats = save_corruptions_11_scenarios_grid(
        sample["rgb"],
        sample["depth"],
        sample["label"],
        frame_index=sample["index"],
        out_dir=fig_dir,
    )
    ds.close()
    summary_path = os.path.join(fig_dir, "corruptions_11_scenarios_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"sample_test_idx": sample_idx, "frame_index": sample["index"], "scenarios": stats}, f, indent=2)
    print("Saved 11-scenario visual grid and summary to", fig_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample_idx", type=int, default=12, help="NYUv2 test index for 11-scenario grid")
    p.add_argument("--only_static", action="store_true", help="Only generate 11-scenario grid & architecture diagram")
    args = p.parse_args()

    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    draw_architecture_diagram(fig_dir)
    generate_11_scenarios_figure(fig_dir, sample_idx=args.sample_idx)

    if args.only_static:
        return

    abl_json = os.path.join(ROOT, "results", "ablations", "ablation_results.json")
    out_tex = os.path.join(ROOT, "report", "generated_tables.tex")
    if not os.path.exists(abl_json):
        raise FileNotFoundError(f"Run scripts/run_ablations.py first; missing {abl_json}")
    data = json.load(open(abl_json))
    plot_robustness_curves(data["variants"], fig_dir)
    plot_ablation_and_qmf(data["variants"], fig_dir)
    write_latex_tables(data, out_tex)


if __name__ == "__main__":
    main()
