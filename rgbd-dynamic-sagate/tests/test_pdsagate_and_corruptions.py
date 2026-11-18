"""Unit tests for PD-SAGate fusion, QMF loss, 11 sensor corruptions, and zero-CKF/Kalman cleanliness (SPEC_03 §5).

Run either with ``pytest -q tests`` or directly ``python tests/test_pdsagate_and_corruptions.py``.
"""

from __future__ import annotations

import os
import re
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.datasets.corruptions import (  # noqa: E402
    Corruption,
    default_eval_suite,
    depth_holes,
    sample_generator,
)
from src.models.esanet_pd_sagate import build_model  # noqa: E402
from src.models.losses import TotalPDSAGateLoss  # noqa: E402
from src.models.pd_sagate_fusion import PDSAGateFusion  # noqa: E402
from src.utils.metrics import compute_qmf_covariance, count_flops  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, H, W = 2, 480, 640
STAGES = [(64, 4), (64, 4), (128, 8), (256, 16), (512, 32)]


def synthetic_rgbd(b: int = B, h: int = H, w: int = W, seed: int = 0):
    """Smooth random RGB in [0,1] and a piece-wise planar room depth (0.5-6 m)."""
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(b, 3, h // 32, w // 32, generator=g)
    rgb = torch.nn.functional.interpolate(
        low, size=(h, w), mode="bicubic", align_corners=False
    ).clamp(0, 1)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    depth = 1.0 + 4.0 * yy[None, None].expand(b, 1, h, w).clone()
    depth[..., h // 3 : 2 * h // 3, w // 4 : w // 2] = 1.2
    depth[..., : h // 10, :] = 0.0  # native missing band at top
    return rgb.to(DEVICE), depth.to(DEVICE)


def stage_features(rgb, depth, channels, stride, seed=0):
    g = torch.Generator().manual_seed(seed)
    h, w = rgb.shape[-2] // stride, rgb.shape[-1] // stride
    xr = torch.randn(rgb.shape[0], channels, h, w, generator=g).to(DEVICE)
    xd = torch.randn(rgb.shape[0], channels, h, w, generator=g).to(DEVICE)
    return xr.requires_grad_(), xd.requires_grad_()


# --------------------------------------------------------------------------- #
# 1) Corruption 11-Scenario Suite Tests
# --------------------------------------------------------------------------- #
def test_11_scenarios_shapes_ranges_and_determinism():
    rgb, depth = synthetic_rgbd()
    suite = default_eval_suite()
    assert len(suite) == 11, f"Expected 11 scenarios in default_eval_suite, got {len(suite)}"
    for c in suite:
        r1, d1 = c(rgb, depth, sample_generator(42, 7))
        r2, d2 = c(rgb, depth, sample_generator(42, 7))
        assert r1.shape == rgb.shape and d1.shape == depth.shape, c.tag
        assert r1.min() >= 0 and r1.max() <= 1, c.tag
        assert d1.min() >= 0 and torch.isfinite(d1).all(), c.tag
        assert torch.equal(r1, r2) and torch.equal(d1, d2), f"non-deterministic: {c.tag}"
        assert (d1[depth == 0] == 0).all(), c.tag


def test_depth_hole_ratio():
    _, depth = synthetic_rgbd()
    for ratio in (0.15, 0.30, 0.50):
        out, mask = depth_holes(depth, ratio, generator=sample_generator(1, 0), return_mask=True)
        frac = mask.float().mean(dim=(1, 2, 3))
        assert torch.allclose(frac, torch.full_like(frac, ratio), atol=0.01), (ratio, frac)
        assert (out[mask] == 0).all()


# --------------------------------------------------------------------------- #
# 2) PD-SAGate Shape, Softmax Partition-of-Unity & Gradient Flow Tests
# --------------------------------------------------------------------------- #
def test_pdsagate_all_modes_and_gradients():
    rgb, depth = synthetic_rgbd()
    for mode in ("full", "phys_sagate", "mask_only"):
        for c, s in STAGES:
            m = PDSAGateFusion(c, mode=mode).to(DEVICE).train()
            xr, xd = stage_features(rgb, depth, c, s)
            out, aux = m(xr, xd, depth)
            assert out.shape == xr.shape, (mode, c)
            # Softmax partition of unity: w_rgb + w_depth == 1
            w_sum = aux["w_rgb"] + aux["w_depth"]
            assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5)
            out.pow(2).mean().backward()
            for name, p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), f"{mode}:{name}"
            assert torch.isfinite(xr.grad).all() and torch.isfinite(xd.grad).all()


def test_esanet_pdsagate_and_qmf_loss_forward_backward():
    rgb, depth = synthetic_rgbd()
    label = torch.randint(-1, 40, (B, H, W), device=DEVICE)
    model = build_model("pd_sagate", pretrained=False).to(DEVICE).train()
    criterion = TotalPDSAGateLoss(lambda_qmf=0.20).to(DEVICE)

    outputs = model(rgb, depth, depth)
    assert isinstance(outputs, (tuple, list)) and outputs[0].shape == (B, 40, H, W)
    assert len(model.fusion_aux) == 5
    assert "aux_logits_rgb" in model.fusion_aux[4] and "aux_logits_depth" in model.fusion_aux[4]

    loss, stats = criterion(outputs, label, model.fusion_aux)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if name.startswith(("se_layer0.gate.", "aux_head_rgb.", "aux_head_depth.")):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


# --------------------------------------------------------------------------- #
# 3) Physical Prior & QMF Covariance Tests (M_hole = 1 => w_depth << w_rgb)
# --------------------------------------------------------------------------- #
def test_physical_prior_suppresses_holes():
    torch.manual_seed(0)
    rgb, depth = synthetic_rgbd()
    m = PDSAGateFusion(64, mode="full").to(DEVICE).eval()
    xr, xd = stage_features(rgb, depth, 64, 4)
    with torch.no_grad():
        _, aux_clean = m(xr, xd, depth)
        _, aux_dead = m(xr, xd, torch.zeros_like(depth))  # all depth pixels dead (M_hole = 1)

    w_dead = aux_dead["w_depth"].mean().item()
    w_rgb_dead = aux_dead["w_rgb"].mean().item()
    assert w_dead < 0.10 and w_dead < w_rgb_dead, f"Expected w_depth << w_rgb on dead depth, got {w_dead}"

    hole_mask = aux_clean["hole_frac"] > 0.99
    valid_mask = aux_clean["hole_frac"] < 0.01
    w_on_hole = aux_clean["w_depth"][hole_mask].mean().item()
    w_on_valid = aux_clean["w_depth"][valid_mask].mean().item()
    assert w_on_hole < 0.10 and w_on_valid > 0.40, (w_on_hole, w_on_valid)

    # Verify QMF empirical covariance utility yields negative covariance when higher loss gets lower weight
    cov_stats = compute_qmf_covariance(
        w_rgb=[0.85, 0.80, 0.30],
        w_depth=[0.15, 0.20, 0.70],
        l_rgb=[1.10, 1.20, 2.50],
        l_depth=[2.80, 2.60, 1.05],
    )
    assert cov_stats["qmf_cov_pair"] < 0.0 and cov_stats["qmf_cov_pooled"] < 0.0


# --------------------------------------------------------------------------- #
# 4) Real Conditional Skip Test (FLOPs & Sub-batch Execution)
# --------------------------------------------------------------------------- #
def test_dynamic_skip_really_skips_refine_and_saves_flops():
    rgb, depth = synthetic_rgbd(b=1)
    xr, xd = stage_features(rgb, depth, 128, 8)

    m_keep = PDSAGateFusion(128, mode="full", tau=0.01).to(DEVICE).eval()  # never skips on clean
    m_skip = PDSAGateFusion(128, mode="full", tau=0.99).to(DEVICE).eval()  # always skips
    m_skip.load_state_dict(m_keep.state_dict())

    # Hook onto m_skip.refine to verify it is NOT called at all when skipped
    called = {"count": 0}
    hook = m_skip.refine.register_forward_hook(lambda *_: called.__setitem__("count", called["count"] + 1))
    with torch.no_grad():
        out_skip, aux_skip = m_skip(xr, xd, depth)
        out_keep, aux_keep = m_keep(xr, xd, depth)
    hook.remove()

    assert aux_skip["skip_mask"].all(), "Expected skip_mask == True when w_bar_depth < tau"
    assert (~aux_keep["skip_mask"]).all(), "Expected skip_mask == False when w_bar_depth >= tau"
    assert called["count"] == 0, f"Refine block should not be called when skipped, called {called['count']} times"

    flops_keep = count_flops(m_keep, (xr, xd, depth))
    flops_skip = count_flops(m_skip, (xr, xd, depth))
    assert flops_skip < flops_keep, f"Expected flops_skip ({flops_skip}) < flops_keep ({flops_keep})"


# --------------------------------------------------------------------------- #
# 5) Zero-CKF / Zero-Kalman Cleanliness Check (Red Line Enforcement)
# --------------------------------------------------------------------------- #
def test_zero_ckf_and_kalman_symbols_in_codebase():
    forbidden = re.compile(r"\b(ckf|cubature|kalman|xi_plus|xi_minus|var_fused|ck_gate|ckgate)\b", re.IGNORECASE)
    scan_dirs = [
        os.path.join(ROOT, "src"),
        os.path.join(ROOT, "scripts"),
        os.path.join(ROOT, "configs"),
        os.path.join(ROOT, "report"),
    ]
    violations = []
    for d in scan_dirs:
        for dirpath, _, filenames in os.walk(d):
            if "__pycache__" in dirpath:
                continue
            for fn in filenames:
                if not fn.endswith((".py", ".yaml", ".yml", ".sh", ".tex")):
                    continue
                fpath = os.path.join(dirpath, fn)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                matches = forbidden.findall(text)
                if matches:
                    violations.append((os.path.relpath(fpath, ROOT), list(set(matches))))
    assert not violations, f"Forbidden CKF/Kalman terms found in codebase: {violations}"


if __name__ == "__main__":
    t0 = time.time()
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"All {len(tests)} test suites passed in {time.time() - t0:.2f}s on {DEVICE}.")
