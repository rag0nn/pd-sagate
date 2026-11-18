"""Stage 1: Verify and cache ESANet-R34-NBt1D baseline (V1) across the 11 evaluation scenarios.

Example::

    python scripts/eval_baseline.py --amp
    python scripts/eval_baseline.py --suite clean

Outputs ``results/baseline/<tag>.json`` and ``results/baseline/summary.csv``.
The official ESANet NYUv2 checkpoint reports **50.30% mIoU** on the clean test split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.corruptions import Corruption, default_eval_suite  # noqa: E402
from src.datasets.nyuv2_dataset import NYUv2  # noqa: E402
from src.engine import evaluate  # noqa: E402
from src.models.esanet_pd_sagate import DEFAULT_CKPT, build_model  # noqa: E402
from src.utils.metrics import count_flops, count_parameters, measure_latency  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--fusion", default="se")
    p.add_argument("--depth_mode", default="refined", choices=["refined", "raw"])
    p.add_argument("--suite", default="default", choices=["default", "clean"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--preload", action="store_true", default=True)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--limit", type=int, default=None, help="debug: first N test images")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "baseline"))
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = build_model(a.fusion, pretrained=True, ckpt_path=a.ckpt).to(dev).eval()
    print("loaded checkpoint:", getattr(model, "load_info", {}))

    cost_file = os.path.join(a.out, "cost.json")
    if not os.path.exists(cost_file) or a.limit:
        x_rgb = torch.randn(1, 3, 480, 640, device=dev)
        x_d = torch.randn(1, 1, 480, 640, device=dev)
        d_m = torch.rand(1, 1, 480, 640, device=dev) * 5.0
        res_cost = {
            "params_M": count_parameters(model),
            "GFLOPs": count_flops(model, (x_rgb, x_d, d_m)),
        }
        res_cost.update(measure_latency(model, (x_rgb, x_d, d_m), amp=a.amp))
        res_cost["device"] = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
        json.dump(res_cost, open(cost_file, "w"), indent=2)
    else:
        res_cost = json.load(open(cost_file))
    print("cost:", json.dumps(res_cost, indent=2))

    idx = list(range(a.limit)) if a.limit else None
    ds = NYUv2("test", depth_mode=a.depth_mode, preload=a.preload, indices=idx)
    suite = [Corruption("clean")] if a.suite == "clean" else default_eval_suite()
    rows = []
    for c in suite:
        out_file = os.path.join(a.out, f"{c.tag}.json")
        if os.path.exists(out_file) and not a.limit:
            r = json.load(open(out_file))
            rows.append(r)
            print(
                f"{c.tag:22s} [cached] mIoU {r['mIoU']:6.2f}  PixAcc {r['pixel_acc']:6.2f}  ECE {r['ECE']:5.2f}",
                flush=True,
            )
            continue
        ds.corruption = c
        loader = DataLoader(
            ds,
            batch_size=a.batch_size,
            shuffle=False,
            num_workers=a.workers,
            pin_memory=True,
            persistent_workers=False,
        )
        t0 = time.time()
        r = evaluate(model, loader, dev, a.depth_mode, amp=a.amp, desc=c.tag)
        r.update(
            corruption=c.name,
            severity=c.severity,
            params=c.params,
            n_images=len(ds),
            seconds=time.time() - t0,
        )
        json.dump(r, open(out_file, "w"), indent=2)
        rows.append(r)
        print(
            f"{c.tag:22s} mIoU {r['mIoU']:6.2f}  PixAcc {r['pixel_acc']:6.2f}  ECE {r['ECE']:5.2f}"
            f"  ({r['seconds']:.0f}s)",
            flush=True,
        )

    with open(os.path.join(a.out, "summary.csv"), "w") as f:
        keys = [
            "corruption",
            "severity",
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
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(
                ",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n"
            )
    print("saved to", a.out)


if __name__ == "__main__":
    main()
