"""Stage 2: Train / fine-tune PD-SAGate variants (V2, V3, V4) on NYU Depth V2 (SPEC_04 §1).

Supports both individual variant runs and the complete Balanced Pack (``--balanced_pack``):
* V2 (``mask_only``): ``seed=42``, 8 epochs
* V3 (``phys_sagate``, no skip): ``seed=42``, 8 epochs
* V4 (``pd_sagate``, full model): ``seeds=[42, 123, 999]``, 8 epochs each

Disk safety rule: Only a single final ``.pth`` checkpoint is written per run;
no intermediate epoch checkpoints are ever stored on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.corruptions import (  # noqa: E402
    axial_depth_noise,
    depth_holes,
    low_light,
    mixed_corruption,
)
from src.datasets.nyuv2_dataset import NYUv2, normalize_rgbd  # noqa: E402
from src.engine import evaluate  # noqa: E402
from src.models.esanet_pd_sagate import DEFAULT_CKPT, build_model  # noqa: E402
from src.models.losses import TotalPDSAGateLoss  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_training_corruption(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    rng: random.Random,
    gen: torch.Generator,
    prob: float = 0.65,
):
    """Stochastically corrupt 65% of mini-batch samples (holes, axial noise, low-light, mixed)."""
    B = rgb.shape[0]
    rgb_out = rgb.clone()
    depth_out = depth.clone()
    for b in range(B):
        if rng.random() >= prob:
            continue
        choice = rng.choice(["holes", "holes_heavy", "axial", "low_light", "mixed"])
        if choice == "holes":
            ratio = rng.uniform(0.12, 0.35)
            depth_out[b] = depth_holes(depth_out[b], ratio=ratio, generator=gen)
        elif choice == "holes_heavy":
            ratio = rng.uniform(0.35, 0.55)
            depth_out[b] = depth_holes(depth_out[b], ratio=ratio, generator=gen)
        elif choice == "axial":
            scale = rng.uniform(2.0, 8.0)
            depth_out[b] = axial_depth_noise(depth_out[b], scale=scale, generator=gen)
            if rng.random() < 0.5:
                depth_out[b] = depth_holes(depth_out[b], ratio=0.15, generator=gen)
        elif choice == "low_light":
            gamma = rng.uniform(1.8, 3.0)
            rgb_out[b] = low_light(rgb_out[b], gamma=gamma, read_noise=0.03, generator=gen)
        elif choice == "mixed":
            rgb_out[b], depth_out[b] = mixed_corruption(
                rgb_out[b], depth_out[b], hole_ratio=0.45, gamma=2.5, generator=gen
            )
    return rgb_out, depth_out


def configure_trainable_params(
    model: nn.Module,
    lr_fusion: float,
    lr_decoder: float,
    weight_decay: float,
    train_decoder: bool = True,
):
    """Freeze RGB/Depth ResNet backbones and pretrained SE weights; train PD-SAGate + aux heads + skip projections."""
    fusion_params, dec_params = [], []
    for name, p in model.named_parameters():
        if any(name.startswith(f"se_layer{s}.gate.") for s in range(5)) or name.startswith(
            ("aux_head_rgb.", "aux_head_depth.")
        ):
            p.requires_grad = True
            fusion_params.append(p)
        elif train_decoder and any(
            name.startswith(prefix) for prefix in ("skip_layer",)
        ):
            p.requires_grad = True
            dec_params.append(p)
        else:
            p.requires_grad = False

    groups = [{"params": fusion_params, "lr": lr_fusion, "weight_decay": weight_decay}]
    if dec_params:
        groups.append({"params": dec_params, "lr": lr_decoder, "weight_decay": weight_decay})
    optimizer = torch.optim.AdamW(groups)
    return optimizer, len(fusion_params), len(dec_params)


def freeze_backbone_bn(model: nn.Module) -> None:
    """Keep frozen ResNet and decoder BatchNorm layers in eval mode so running stats do not drift."""
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and not any(
            name.startswith(f"se_layer{s}.gate") for s in range(5)
        ):
            m.eval()


def train_one_run(
    args,
    fusion: str,
    seed: int,
    train_ds: NYUv2,
    test_ds: NYUv2,
    dev: torch.device,
) -> Dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_out = os.path.join(args.out_dir, f"{fusion}_seed{seed}.pth")
    log_file = os.path.join(args.out_dir, f"train_log_{fusion}_seed{seed}.json")

    if os.path.exists(ckpt_out) and not args.force:
        ckpt = torch.load(ckpt_out, map_location="cpu", weights_only=False)
        model = build_model(
            fusion=ckpt["fusion"],
            pretrained=False,
            pdsagate_stages=tuple(ckpt.get("pdsagate_stages", args.stages)),
            pdsagate_cfg=ckpt.get("pdsagate_cfg", {}),
        ).to(dev)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        test_ds.corruption = None
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
        clean_res = evaluate(model, test_loader, dev, args.depth_mode, amp=args.amp, desc=f"eval-clean-{fusion}-s{seed}")
        print(
            f"[{fusion} seed={seed} cached] Clean mIoU: {clean_res['mIoU']:.2f}% | PixAcc: {clean_res['pixel_acc']:.2f}%",
            flush=True,
        )
        log_out = {
            "seed": seed,
            "fusion": fusion,
            "ckpt_path": ckpt_out,
            "clean_mIoU": clean_res["mIoU"],
            "clean_pixel_acc": clean_res["pixel_acc"],
            "qmf_stats": clean_res.get("qmf_stats", {}),
            "history": ckpt.get("history", []),
        }
        with open(log_file, "w") as f:
            json.dump(log_out, f, indent=2)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return log_out

    set_seed(seed)
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)

    pdsagate_cfg = {
        "tau": args.tau,
        "alpha_init": args.alpha_init,
        "beta_init": args.beta_init if fusion != "mask_only" else 0.0,
        "gamma_scale_init": args.gamma_scale_init,
        "refine_type": args.refine_type,
        "use_prior": not args.no_prior,
        "use_dynamic_skip": (fusion == "pd_sagate") and (not args.no_skip),
    }
    model = build_model(
        fusion=fusion,
        pretrained=True,
        ckpt_path=args.ckpt,
        pdsagate_stages=tuple(args.stages),
        pdsagate_cfg=pdsagate_cfg,
    ).to(dev)

    optimizer, n_f, n_d = configure_trainable_params(
        model, args.lr_fusion, args.lr_decoder, args.weight_decay, train_decoder=args.train_decoder
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[{fusion} seed={seed}] trainable_params={n_trainable/1e6:.3f}M (tensors: fusion+aux={n_f}, dec={n_d})",
        flush=True,
    )

    enable_qmf_gate = fusion in ("phys_sagate", "pd_sagate")
    criterion = TotalPDSAGateLoss(
        lambda_qmf=args.lambda_qmf,
        enable_qmf_gate_supervision=enable_qmf_gate,
    ).to(dev)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    total_steps = args.epochs * max(1, (len(train_loader) + args.accum_steps - 1) // args.accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")

    history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze_backbone_bn(model)
        optimizer.zero_grad(set_to_none=True)
        ep_loss, ep_ce, ep_qmf, ep_rank = 0.0, 0.0, 0.0, 0.0
        w_hole_list, w_val_list = [], []
        pbar = tqdm(train_loader, desc=f"{fusion} s{seed} ep {epoch}/{args.epochs}", leave=False)

        for step, batch in enumerate(pbar, 1):
            rgb = batch["rgb"]
            depth_m = batch["depth"]
            label = batch["label"].to(dev, non_blocking=True)

            rgb_c, depth_c = apply_training_corruption(
                rgb, depth_m, rng, gen, prob=args.corruption_prob
            )
            rgb_c = rgb_c.to(dev, non_blocking=True)
            depth_c = depth_c.to(dev, non_blocking=True)
            rgb_n, depth_n = normalize_rgbd(rgb_c, depth_c, args.depth_mode)

            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp and dev.type == "cuda"):
                outputs = model(rgb_n, depth_n, depth_c)
                loss, stats = criterion(outputs, label, model.fusion_aux)
                loss_scaled = loss / args.accum_steps

            scaler.scale(loss_scaled).backward()
            if step % args.accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=5.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            ep_loss += stats["loss_total"]
            ep_ce += stats["loss_ce"]
            ep_qmf += stats["loss_qmf"]
            ep_rank += stats.get("loss_rank", 0.0)
            if stats.get("w_depth_on_holes", 0.0) > 0:
                w_hole_list.append(stats["w_depth_on_holes"])
            if stats.get("w_depth_on_valid", 0.0) > 0:
                w_val_list.append(stats["w_depth_on_valid"])
            pbar.set_postfix(ce=f"{stats['loss_ce']:.3f}", qmf=f"{stats['loss_qmf']:.3f}")
            model.fusion_aux = []

        n_b = len(train_loader)
        rec = {
            "epoch": epoch,
            "loss_total": ep_loss / n_b,
            "loss_ce": ep_ce / n_b,
            "loss_qmf": ep_qmf / n_b,
            "loss_rank": ep_rank / n_b,
            "w_depth_on_holes": float(np.mean(w_hole_list)) if w_hole_list else 0.0,
            "w_depth_on_valid": float(np.mean(w_val_list)) if w_val_list else 0.0,
            "elapsed_s": time.time() - t_start,
        }
        history.append(rec)
        print(
            f"  [{fusion} s{seed} ep {epoch:02d}] loss={rec['loss_total']:.4f} "
            f"(ce={rec['loss_ce']:.4f}, qmf={rec['loss_qmf']:.4f}, rank={rec['loss_rank']:.4f}) "
            f"w_dep(valid={rec['w_depth_on_valid']:.3f}, hole={rec['w_depth_on_holes']:.3f}) ({rec['elapsed_s']:.0f}s)",
            flush=True,
        )

    # Save single final checkpoint (disk protection: no intermediate checkpoints)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "fusion": fusion,
            "pdsagate_stages": list(args.stages),
            "pdsagate_cfg": pdsagate_cfg,
            "seed": seed,
            "epochs": args.epochs,
            "history": history,
        },
        ckpt_out,
    )

    del optimizer, scheduler, scaler, criterion
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    test_ds.corruption = None
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    clean_res = evaluate(
        model, test_loader, dev, args.depth_mode, amp=args.amp, desc=f"eval-clean-{fusion}-s{seed}"
    )
    print(
        f"[{fusion} seed={seed}] Clean mIoU: {clean_res['mIoU']:.2f}% | PixAcc: {clean_res['pixel_acc']:.2f}%",
        flush=True,
    )

    log_out = {
        "seed": seed,
        "fusion": fusion,
        "ckpt_path": ckpt_out,
        "trainable_params_M": n_trainable / 1e6,
        "clean_mIoU": clean_res["mIoU"],
        "clean_pixel_acc": clean_res["pixel_acc"],
        "qmf_stats": clean_res.get("qmf_stats", {}),
        "history": history,
    }
    with open(log_file, "w") as f:
        json.dump(log_out, f, indent=2)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return log_out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fusion",
        default="pd_sagate",
        choices=["pd_sagate", "phys_sagate", "mask_only"],
    )
    p.add_argument(
        "--balanced_pack",
        action="store_true",
        help="Run the full SPEC_04 Balanced Pack: V2(s42), V3(s42), V4(s42,123,999)",
    )
    p.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--lr_fusion", type=float, default=2e-4)
    p.add_argument("--lr_decoder", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_qmf", type=float, default=0.10)
    p.add_argument("--corruption_prob", type=float, default=0.65)
    p.add_argument("--tau", type=float, default=0.36)
    p.add_argument("--alpha_init", type=float, default=6.0)
    p.add_argument("--beta_init", type=float, default=0.15)
    p.add_argument("--gamma_scale_init", type=float, default=2.0)
    p.add_argument("--refine_type", default="dw", choices=["dw", "dense"])
    p.add_argument("--no_prior", action="store_true")
    p.add_argument("--no_skip", action="store_true")
    p.add_argument("--train_decoder", action="store_true", default=False)
    p.add_argument("--no_train_decoder", action="store_false", dest="train_decoder")
    p.add_argument("--depth_mode", default="refined", choices=["refined", "raw"])
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--limit_train", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "checkpoints"))
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_idx = list(range(args.limit_train)) if args.limit_train else None
    train_ds = NYUv2("train", depth_mode=args.depth_mode, train_aug=True, preload=True, indices=train_idx)
    test_ds = NYUv2("test", depth_mode=args.depth_mode, train_aug=False, preload=True)

    if args.balanced_pack:
        schedule = [
            ("mask_only", 42),
            ("phys_sagate", 42),
            ("pd_sagate", 42),
            ("pd_sagate", 123),
            ("pd_sagate", 999),
        ]
    else:
        schedule = [(args.fusion, s) for s in args.seeds]

    summary = []
    for fusion, seed in schedule:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        res = train_one_run(args, fusion, seed, train_ds, test_ds, dev)
        summary.append(res)

    tag = "balanced_pack" if args.balanced_pack else args.fusion
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
