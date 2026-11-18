"""Stage 3: 4-variant ablation study, QMF Theorem-1 verification, cost profiling, and qualitative figures (SPEC_04 §2).

Evaluates the four variants across all 11 evaluation scenarios (Clean + 10 Corruptions)
on NYU Depth V2 (654 test images):

* **V1 — Baseline (``ESANet-SE``)**: official pretrained SE-add fusion.
* **V2 — ``Mask-SAGate`` (``mask_only``)**: physical ``M_hole`` prior + Softmax aggregation.
* **V3 — ``Phys-SAGate`` (``phys_sagate``)**: full physics prior (``M_hole`` + Sobel) +
  SA-Gate channel separation & Softmax aggregation + QMF loss (no dynamic skip).
* **V4 — ``PD-SAGate`` (``pd_sagate``, Full Model)**: V3 + DynMM conditional refinement
  skip (``w_bar_depth < tau``), evaluated across ``seeds=[42, 123, 999]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.corruptions import Corruption, default_eval_suite  # noqa: E402
from src.datasets.nyuv2_dataset import NYUv2, normalize_rgbd  # noqa: E402
from src.engine import evaluate  # noqa: E402
from src.models.esanet_pd_sagate import DEFAULT_CKPT, build_model  # noqa: E402
from src.utils.metrics import count_flops, count_parameters, measure_latency  # noqa: E402
from src.utils.visualization import save_qualitative_panel  # noqa: E402


def load_trained_variant(ckpt_path: str, dev: torch.device, tau_override: float | None = None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("pdsagate_cfg", {}))
    if tau_override is not None:
        cfg["tau"] = tau_override
    model = build_model(
        fusion=ckpt["fusion"],
        pretrained=False,
        pdsagate_stages=tuple(ckpt.get("pdsagate_stages", [0, 1, 2, 3, 4])),
        pdsagate_cfg=cfg,
    ).to(dev)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def measure_variant_cost(
    model: torch.nn.Module, dev: torch.device, amp: bool = True, corrupt_depth: bool = False
):
    x_rgb = torch.randn(1, 3, 480, 640, device=dev)
    x_d = torch.randn(1, 1, 480, 640, device=dev)
    d_m = (
        torch.zeros(1, 1, 480, 640, device=dev)
        if corrupt_depth
        else torch.rand(1, 1, 480, 640, device=dev) * 4.0 + 0.5
    )
    if corrupt_depth:
        x_d.fill_(-2.0)
    cost = {
        "params_M": count_parameters(model),
        "GFLOPs": count_flops(model, (x_rgb, x_d, d_m)),
    }
    cost.update(measure_latency(model, (x_rgb, x_d, d_m), n_warmup=15, n_iter=60, amp=amp))
    return cost


def eval_suite_on_model(
    model: torch.nn.Module,
    ds: NYUv2,
    suite: List[Corruption],
    dev: torch.device,
    batch_size: int = 8,
    amp: bool = True,
    tag_prefix: str = "",
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for c in suite:
        ds.corruption = c
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        res = evaluate(model, loader, dev, ds.depth_mode, amp=amp, desc=f"{tag_prefix}:{c.tag}")
        out[c.tag] = res
        qmf_cov = res.get("qmf_stats", {}).get("qmf_cov_pair", 0.0)
        print(
            f"  [{tag_prefix:16s} | {c.tag:18s}] mIoU={res['mIoU']:5.2f}%  "
            f"PixAcc={res['pixel_acc']:5.2f}%  ECE={res['ECE']:5.2f}%  Cov(w,l)={qmf_cov:+.4f}",
            flush=True,
        )
    return out


def aggregate_seeds(seed_runs: Dict[int, Dict[str, Dict[str, object]]]) -> Dict[str, Dict[str, object]]:
    tags = list(next(iter(seed_runs.values())).keys())
    scalar_keys = [
        "mIoU",
        "pixel_acc",
        "mean_acc",
        "ECE",
        "NLL",
        "IoU_wall",
        "IoU_floor",
        "IoU_table",
        "IoU_chair",
        "IoU_window",
    ]
    agg: Dict[str, Dict[str, object]] = {}
    clean_miou_per_seed = {
        s: seed_runs[s]["clean"]["mIoU"] for s in seed_runs if "clean" in seed_runs[s]
    }

    for t in tags:
        agg[t] = {}
        for k in scalar_keys:
            vals = [seed_runs[s][t][k] for s in seed_runs]
            agg[t][f"{k}_mean"] = float(np.mean(vals))
            agg[t][f"{k}_std"] = float(np.std(vals, ddof=0))

        if clean_miou_per_seed:
            ret_vals = [
                100.0 * seed_runs[s][t]["mIoU"] / max(clean_miou_per_seed[s], 1e-6)
                for s in seed_runs
                if s in clean_miou_per_seed
            ]
            agg[t]["retention_pct_mean"] = float(np.mean(ret_vals))
            agg[t]["retention_pct_std"] = float(np.std(ret_vals, ddof=0))

        first = next(iter(seed_runs.values()))[t]
        if "mean_w_depth_per_stage" in first:
            stages = first["mean_w_depth_per_stage"].keys()
            agg[t]["mean_w_depth_per_stage"] = {
                int(s): float(
                    np.mean(
                        [
                            seed_runs[sd][t]["mean_w_depth_per_stage"].get(
                                s, seed_runs[sd][t]["mean_w_depth_per_stage"].get(str(s), 0.0)
                            )
                            for sd in seed_runs
                        ]
                    )
                )
                for s in stages
            }
        if "skip_rate_per_stage" in first:
            stages = first["skip_rate_per_stage"].keys()
            agg[t]["skip_rate_per_stage"] = {
                int(s): float(
                    np.mean(
                        [
                            seed_runs[sd][t]["skip_rate_per_stage"].get(
                                s, seed_runs[sd][t]["skip_rate_per_stage"].get(str(s), 0.0)
                            )
                            for sd in seed_runs
                        ]
                    )
                )
                for s in stages
            }
        if "qmf_stats" in first:
            qkeys = first["qmf_stats"].keys()
            agg[t]["qmf_stats"] = {
                qk: float(np.mean([seed_runs[sd][t]["qmf_stats"][qk] for sd in seed_runs]))
                for qk in qkeys
            }
    return agg


@torch.no_grad()
def export_qualitative_examples(
    ds: NYUv2,
    base_model: torch.nn.Module,
    pds_model: torch.nn.Module,
    dev: torch.device,
    out_dir: str,
    sample_indices: List[int] = [0, 12, 25, 48, 90, 140],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    scenarios = [
        Corruption("clean"),
        Corruption("depth_holes", 2),
        Corruption("depth_holes", 3),
        Corruption("mixed", 2),
    ]
    for c in scenarios:
        ds.corruption = c
        for idx in sample_indices:
            sample = ds[idx]
            rgb = sample["rgb"].unsqueeze(0).to(dev)
            depth = sample["depth"].unsqueeze(0).to(dev)
            label = sample["label"].to(dev)
            rgb_n, depth_n = normalize_rgbd(rgb, depth, ds.depth_mode)

            log_b = base_model(rgb_n, depth_n, depth)
            log_p = pds_model(rgb_n, depth_n, depth)
            pred_b = log_b.argmax(1)[0]
            pred_p = log_p.argmax(1)[0]

            aux1 = pds_model.fusion_aux[1] if len(pds_model.fusion_aux) > 1 else pds_model.fusion_aux[0]
            w_depth_map = aux1.get("w_depth")
            w_rgb_map = aux1.get("w_rgb")

            fname = os.path.join(out_dir, f"qual_{c.tag}_idx{idx:03d}.png")
            save_qualitative_panel(
                fname,
                sample["rgb"],
                sample["depth"],
                label,
                pred_b,
                pred_p,
                w_depth=w_depth_map,
                w_rgb=w_rgb_map,
                title=f"NYUv2 Test #{idx} (Frame {sample['index']}) — Senaryo: {c.tag}",
            )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2_seeds", type=int, nargs="+", default=[42])
    p.add_argument("--v3_seeds", type=int, nargs="+", default=[42])
    p.add_argument("--v4_seeds", type=int, nargs="+", default=[42, 123, 999])
    p.add_argument("--ckpt_dir", default=os.path.join(ROOT, "results", "checkpoints"))
    p.add_argument("--baseline_dir", default=os.path.join(ROOT, "results", "baseline"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "ablations"))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ds = NYUv2("test", depth_mode="refined", preload=True)
    suite = default_eval_suite()

    out_json = os.path.join(args.out_dir, "ablation_results.json")
    all_results: Dict[str, object] = (
        json.load(open(out_json)) if os.path.exists(out_json) else {"costs": {}, "variants": {}}
    )

    # 1) Variant 1: Baseline (ESANet SE-Fusion)
    print("=== Variant 1: ESANet SE-Fusion (Baseline) ===")
    if "V1_SE_Baseline" not in all_results["variants"] or len(all_results["variants"]["V1_SE_Baseline"]) < len(suite):
        m_base = build_model("se", pretrained=True, ckpt_path=DEFAULT_CKPT).to(dev).eval()
        all_results["costs"]["V1_SE_Baseline"] = measure_variant_cost(m_base, dev, amp=args.amp)
        v1_tags = {}
        for c in suite:
            cached = os.path.join(args.baseline_dir, f"{c.tag}.json")
            if os.path.exists(cached):
                v1_tags[c.tag] = json.load(open(cached))
            else:
                ds.corruption = c
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
                res = evaluate(m_base, loader, dev, ds.depth_mode, amp=args.amp, desc=f"V1_SE:{c.tag}")
                os.makedirs(args.baseline_dir, exist_ok=True)
                json.dump(res, open(cached, "w"), indent=2)
                v1_tags[c.tag] = res
            print(
                f"  [V1_SE            | {c.tag:18s}] mIoU={v1_tags[c.tag]['mIoU']:5.2f}%  PixAcc={v1_tags[c.tag]['pixel_acc']:5.2f}%",
                flush=True,
            )
        all_results["variants"]["V1_SE_Baseline"] = aggregate_seeds({42: v1_tags})
        del m_base
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        json.dump(all_results, open(out_json, "w"), indent=2)

    # 2) Variant 2: Mask-SAGate (mask_only)
    print("=== Variant 2: Mask-SAGate (M_hole + Softmax) ===")
    v2_seeds: Dict[int, Dict[str, Dict[str, object]]] = {
        int(k): v for k, v in all_results.get("v2_per_seed", {}).items()
    }
    for s in args.v2_seeds:
        if s in v2_seeds and len(v2_seeds[s]) == len(suite):
            print(f"  [V2_s{s} cached]")
            continue
        p_ckpt = os.path.join(args.ckpt_dir, f"mask_only_seed{s}.pth")
        if os.path.exists(p_ckpt):
            m_v2 = load_trained_variant(p_ckpt, dev)
        else:
            m_v2 = build_model("mask_only", pretrained=True, ckpt_path=DEFAULT_CKPT).to(dev).eval()
        if "V2_Mask_SAGate" not in all_results["costs"]:
            all_results["costs"]["V2_Mask_SAGate"] = measure_variant_cost(m_v2, dev, amp=args.amp)
        v2_seeds[s] = eval_suite_on_model(m_v2, ds, suite, dev, args.batch_size, args.amp, f"V2_s{s}")
        del m_v2
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if v2_seeds:
        all_results["variants"]["V2_Mask_SAGate"] = aggregate_seeds(v2_seeds)
        all_results["v2_per_seed"] = v2_seeds
        json.dump(all_results, open(out_json, "w"), indent=2)

    # 3) Variant 3: Phys-SAGate (No Dynamic Skip)
    print("=== Variant 3: Phys-SAGate (Physics + SA-Gate + QMF, No Skip) ===")
    v3_seeds: Dict[int, Dict[str, Dict[str, object]]] = {
        int(k): v for k, v in all_results.get("v3_per_seed", {}).items()
    }
    for s in args.v3_seeds:
        if s in v3_seeds and len(v3_seeds[s]) == len(suite):
            print(f"  [V3_s{s} cached]")
            continue
        p_ckpt = os.path.join(args.ckpt_dir, f"phys_sagate_seed{s}.pth")
        if not os.path.exists(p_ckpt):
            continue
        m_v3 = load_trained_variant(p_ckpt, dev)
        if "V3_Phys_SAGate" not in all_results["costs"]:
            all_results["costs"]["V3_Phys_SAGate"] = measure_variant_cost(m_v3, dev, amp=args.amp)
        v3_seeds[s] = eval_suite_on_model(m_v3, ds, suite, dev, args.batch_size, args.amp, f"V3_s{s}")
        del m_v3
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if v3_seeds:
        all_results["variants"]["V3_Phys_SAGate"] = aggregate_seeds(v3_seeds)
        all_results["v3_per_seed"] = v3_seeds
        json.dump(all_results, open(out_json, "w"), indent=2)

    # 4) Variant 4: Full PD-SAGate (Physics + SA-Gate + QMF + DynMM Skip)
    print("=== Variant 4: Full PD-SAGate (Ours) ===")
    v4_seeds: Dict[int, Dict[str, Dict[str, object]]] = {
        int(k): v for k, v in all_results.get("v4_per_seed", {}).items()
    }
    for s in args.v4_seeds:
        if s in v4_seeds and len(v4_seeds[s]) == len(suite):
            print(f"  [V4_s{s} cached]")
            continue
        p_ckpt = os.path.join(args.ckpt_dir, f"pd_sagate_seed{s}.pth")
        if not os.path.exists(p_ckpt):
            continue
        m_v4 = load_trained_variant(p_ckpt, dev)
        if "V4_PD_SAGate_clean" not in all_results["costs"]:
            all_results["costs"]["V4_PD_SAGate_clean"] = measure_variant_cost(
                m_v4, dev, amp=args.amp, corrupt_depth=False
            )
            m_skip = load_trained_variant(p_ckpt, dev, tau_override=0.99)
            all_results["costs"]["V4_PD_SAGate_skipped"] = measure_variant_cost(
                m_skip, dev, amp=args.amp, corrupt_depth=True
            )
            del m_skip
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        v4_seeds[s] = eval_suite_on_model(m_v4, ds, suite, dev, args.batch_size, args.amp, f"V4_s{s}")
        del m_v4
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if v4_seeds:
        all_results["variants"]["V4_PD_SAGate"] = aggregate_seeds(v4_seeds)
        all_results["v4_per_seed"] = v4_seeds

    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved ablation results to", out_json)

    p_ckpt0 = os.path.join(args.ckpt_dir, f"pd_sagate_seed{args.v4_seeds[0]}.pth")
    if os.path.exists(p_ckpt0):
        print("Generating qualitative comparison panels...")
        m_base = build_model("se", pretrained=True, ckpt_path=DEFAULT_CKPT).to(dev).eval()
        m_v4_first = load_trained_variant(p_ckpt0, dev)
        export_qualitative_examples(
            ds, m_base, m_v4_first, dev, os.path.join(ROOT, "results", "figures", "qualitative")
        )


if __name__ == "__main__":
    main()
