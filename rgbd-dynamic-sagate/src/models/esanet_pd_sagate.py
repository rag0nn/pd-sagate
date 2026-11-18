"""ESANet-R34-NBt1D backbone wrapped with PD-SAGate fusion blocks and unimodal auxiliary heads.

The encoder, context module (PPM), and multi-scale decoder are the unmodified
official ESANet modules (:mod:`src.models.esanet`). Only the five RGB-D fusion
blocks (``se_layer0..4``) are wrapped with :class:`PDSAGateFusion`:

==================  =============================================================
``fusion``          Variant description
==================  =============================================================
``se``              V1: Official ESANet ``SqueezeAndExciteFusionAdd`` (Baseline)
``mask_only``       V2: ``Mask-SAGate`` (``M_hole`` prior + Softmax aggregation)
``phys_sagate``     V3: ``Phys-SAGate`` (Physics prior + SA-Gate + QMF, no skip)
``pd_sagate``       V4: ``PD-SAGate`` (Full model + DynMM conditional skip)
==================  =============================================================
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .esanet.model import ESANet
from .pd_sagate_fusion import PDSAGateConfig, PDSAGateFusion

DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "third_party",
    "ESANet",
    "trained_models",
    "nyuv2",
    "r34_NBt1D.pth",
)

ESANET_NYUV2_KWARGS = dict(
    height=480,
    width=640,
    num_classes=40,
    encoder_rgb="resnet34",
    encoder_depth="resnet34",
    encoder_block="NonBottleneck1D",
    channels_decoder=[512, 256, 128],
    nr_decoder_blocks=[3, 3, 3],
    activation="relu",
    encoder_decoder_fusion="add",
    context_module="ppm",
    fuse_depth_in_rgb_encoder="SE-add",
    upsampling="learned-3x3-zeropad",
)

FUSIONS = ("se", "mask_only", "phys_sagate", "pd_sagate")
_MODE_MAP = {
    "mask_only": "mask_only",
    "phys_sagate": "phys_sagate",
    "pd_sagate": "full",
}


class ResidualSEPDSAGateWrapper(nn.Module):
    """Wraps a pretrained ESANet ``SqueezeAndExciteFusionAdd`` with :class:`PDSAGateFusion`.

    Preserves the pretrained ``se_rgb`` and ``se_depth`` channel attention layers
    and feeds their outputs ``(r_se, d_se)`` plus raw depth ``depth_m`` into
    :class:`PDSAGateFusion`.
    """

    def __init__(self, se_block: nn.Module, gate: PDSAGateFusion) -> None:
        super().__init__()
        self.se_rgb = se_block.se_rgb
        self.se_depth = se_block.se_depth
        self.gate = gate

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_m: Optional[torch.Tensor] = None,
    ):
        r_se = self.se_rgb(rgb)
        d_se = self.se_depth(depth)
        return self.gate(r_se, d_se, depth_m)


class ESANetPDSAGate(ESANet):
    """ESANet-R34-NBt1D with PD-SAGate fusion stages and training-only unimodal QMF heads."""

    def __init__(
        self,
        fusion: str = "se",
        pdsagate_stages: Sequence[int] = (0, 1, 2, 3, 4),
        pdsagate_cfg: Optional[dict] = None,
        **esanet_kwargs,
    ) -> None:
        kw = dict(ESANET_NYUV2_KWARGS)
        kw.update(esanet_kwargs)
        kw.setdefault("pretrained_on_imagenet", False)
        super().__init__(**kw)
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        self.fusion = fusion
        self.pdsagate_stages = tuple(pdsagate_stages) if fusion != "se" else ()
        self.fusion_aux: List[dict] = []
        self.compute_qmf_aux: bool = False  # Can be enabled during eval to measure QMF Cov(w_m, l_m)

        if fusion != "se":
            chans = [
                64,
                self.encoder_rgb.down_4_channels_out,
                self.encoder_rgb.down_8_channels_out,
                self.encoder_rgb.down_16_channels_out,
                self.encoder_rgb.down_32_channels_out,
            ]
            for s in self.pdsagate_stages:
                cfg_dict = dict(pdsagate_cfg or {})
                cfg_dict["mode"] = _MODE_MAP[fusion]
                cfg = PDSAGateConfig(**cfg_dict)
                old_se = getattr(self, f"se_layer{s}")
                setattr(
                    self,
                    f"se_layer{s}",
                    ResidualSEPDSAGateWrapper(old_se, PDSAGateFusion(chans[s], cfg)),
                )
            # Lightweight 1x1 unimodal classification heads at stage 4 (512 -> 40) for QMF Theorem-1
            c_last = chans[4]
            num_classes = kw["num_classes"]
            self.aux_head_rgb = nn.Conv2d(c_last, num_classes, kernel_size=1, bias=True)
            self.aux_head_depth = nn.Conv2d(c_last, num_classes, kernel_size=1, bias=True)
            nn.init.normal_(self.aux_head_rgb.weight, std=0.01)
            nn.init.zeros_(self.aux_head_rgb.bias)
            nn.init.normal_(self.aux_head_depth.weight, std=0.01)
            nn.init.zeros_(self.aux_head_depth.bias)

    def _fuse(self, s: int, rgb: torch.Tensor, depth: torch.Tensor, depth_m: Optional[torch.Tensor]):
        layer = getattr(self, f"se_layer{s}")
        if isinstance(layer, (PDSAGateFusion, ResidualSEPDSAGateWrapper)):
            out, aux = layer(rgb, depth, depth_m)
            aux["stage"] = s
            r_t = aux.pop("r_tilde", None)
            d_t = aux.pop("d_tilde", None)
            if s == 4 and hasattr(self, "aux_head_rgb") and (self.training or self.compute_qmf_aux):
                if self.fusion == "mask_only":
                    r_t = r_t.detach()
                    d_t = d_t.detach()
                aux["aux_logits_rgb"] = self.aux_head_rgb(r_t)
                aux["aux_logits_depth"] = self.aux_head_depth(d_t)
            self.fusion_aux.append(aux)
            return out
        return layer(rgb, depth)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, depth_m: Optional[torch.Tensor] = None):
        self.fusion_aux = []
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)
        fuse = self._fuse(0, rgb, depth, depth_m)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        fuse = self._fuse(1, rgb, depth, depth_m)
        skip1 = self.skip_layer1(fuse)

        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        fuse = self._fuse(2, rgb, depth, depth_m)
        skip2 = self.skip_layer2(fuse)

        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        fuse = self._fuse(3, rgb, depth, depth_m)
        skip3 = self.skip_layer3(fuse)

        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        fuse = self._fuse(4, rgb, depth, depth_m)

        out = self.context_module(fuse)
        return self.decoder(enc_outs=[out, skip3, skip2, skip1])


def load_esanet_checkpoint(
    model: nn.Module, ckpt_path: str = DEFAULT_CKPT, strict_backbone: bool = True
):
    """Load official ESANet NYUv2 weights (including pretrained ``se_rgb`` / ``se_depth``)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    own = model.state_dict()
    load = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(load, strict=False)
    allowed_prefixes = tuple(
        [f"se_layer{s}.gate." for s in getattr(model, "pdsagate_stages", ())]
        + ["aux_head_rgb.", "aux_head_depth."]
    )
    missing_backbone = [k for k in missing if not k.startswith(allowed_prefixes)]
    if strict_backbone and (missing_backbone or unexpected):
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing_backbone[:5]} unexpected={unexpected[:5]}"
        )
    return {
        "loaded": len(load),
        "new_params": [k for k in own if k not in load],
        "epoch": ckpt.get("epoch"),
    }


def build_model(
    fusion: str = "se",
    pretrained: bool = True,
    ckpt_path: str = DEFAULT_CKPT,
    pdsagate_stages: Sequence[int] = (0, 1, 2, 3, 4),
    pdsagate_cfg: Optional[dict] = None,
    **esanet_kwargs,
) -> ESANetPDSAGate:
    model = ESANetPDSAGate(
        fusion=fusion,
        pdsagate_stages=pdsagate_stages,
        pdsagate_cfg=pdsagate_cfg,
        **esanet_kwargs,
    )
    if pretrained:
        info = load_esanet_checkpoint(model, ckpt_path)
        model.load_info = info
    return model


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
