# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher, scale_adaptive_weight


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _apply_ablation_preset(
    preset: str,
    reg_loss_mode: str,
    match_reg_mode: str,
    quality_label_mode: str,
    use_density_matcher: bool,
) -> tuple[str, str, str, bool]:
    if not preset:
        return reg_loss_mode, match_reg_mode, quality_label_mode, use_density_matcher

    presets = {
        "official": ("giou", "giou", "iou", False),
        "fixed_nwd": ("fixed_nwd_giou", "fixed_nwd_giou", "iou", False),
        "sa": ("sa_nwd_giou", "sa_nwd_giou", "iou", False),
        "quality": ("giou", "giou", "scale_hybrid", False),
        "density": ("giou", "giou", "iou", True),
        "sa_quality": ("sa_nwd_giou", "sa_nwd_giou", "scale_hybrid", False),
        "sa_density": ("sa_nwd_giou", "sa_nwd_giou", "iou", True),
        "quality_density": ("giou", "giou", "scale_hybrid", True),
        "full": ("sa_nwd_giou", "sa_nwd_giou", "scale_hybrid", True),
    }
    if preset not in presets:
        raise ValueError(f"Unknown GMB_ABLATION_PRESET={preset!r}. Available presets: {sorted(presets)}.")
    return presets[preset]


def _reshape_constant_for_pairs(
    constant: float | torch.Tensor,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    constant = torch.as_tensor(constant, device=reference.device, dtype=reference.dtype).clamp_min(eps)
    if constant.ndim == 1 and reference.ndim == 2 and reference.shape[-1] == 1:
        if constant.shape[0] == reference.shape[0]:
            constant = constant.unsqueeze(-1)
    return constant


def wasserstein_score(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    eps: float = 1e-7,
    constant: float | torch.Tensor = 0.05,
) -> torch.Tensor:
    """Compute matched normalized Gaussian Wasserstein similarity."""
    if xywh:
        x1, y1, w1, h1 = box1.chunk(4, -1)
        x2, y2, w2, h2 = box2.chunk(4, -1)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = (b1_x2 - b1_x1).clamp_min(eps), (b1_y2 - b1_y1).clamp_min(eps)
        w2, h2 = (b2_x2 - b2_x1).clamp_min(eps), (b2_y2 - b2_y1).clamp_min(eps)
        x1, y1 = b1_x1 + w1 / 2, b1_y1 + h1 / 2
        x2, y2 = b2_x1 + w2 / 2, b2_y1 + h2 / 2

    center_distance = (x1 - x2).pow(2) + (y1 - y2).pow(2)
    wh_distance = ((w1 - w2).pow(2) + (h1 - h2).pow(2)) / 4.0
    wasserstein_2 = center_distance + wh_distance + eps
    constant = _reshape_constant_for_pairs(constant, wasserstein_2, eps)
    return torch.exp(-torch.sqrt(wasserstein_2) / constant).squeeze(-1)


def wasserstein_loss(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    eps: float = 1e-7,
    constant: float | torch.Tensor = 0.05,
) -> torch.Tensor:
    """Compute matched normalized Gaussian Wasserstein loss."""
    return 1.0 - wasserstein_score(box1, box2, xywh=xywh, eps=eps, constant=constant)


class DETRLoss(nn.Module):
    """DETR loss with scale-adaptive NWD, quality labels, and density-aware matching."""

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
        reg_loss_mode: str = "sa_nwd_giou",
        match_reg_mode: str = "sa_nwd_giou",
        quality_label_mode: str = "scale_hybrid",
        nwd_constant: float = 0.05,
        nwd_alpha: float = 0.50,
        nwd_alpha_min: float = 0.0,
        nwd_alpha_max: float = 0.70,
        quality_nwd_alpha: float = 0.50,
        quality_alpha_min: float = 0.0,
        quality_alpha_max: float = 0.70,
        small_object_scale: float = 0.030,
        scale_temperature: float = 0.010,
        use_density_matcher: bool = True,
        density_gain: float = 1.0,
        density_tau: float = 0.030,
        density_temperature: float = 0.010,
    ):
        super().__init__()

        reg_loss_mode, match_reg_mode, quality_label_mode, use_density_matcher = _apply_ablation_preset(
            os.getenv("GMB_ABLATION_PRESET", ""),
            reg_loss_mode,
            match_reg_mode,
            quality_label_mode,
            use_density_matcher,
        )
        reg_loss_mode = _env_str("GMB_REG_LOSS_MODE", reg_loss_mode)
        match_reg_mode = _env_str("GMB_MATCH_REG_MODE", match_reg_mode)
        quality_label_mode = _env_str("GMB_QUALITY_LABEL_MODE", quality_label_mode)
        nwd_constant = _env_float("GMB_NWD_CONSTANT", nwd_constant)
        nwd_alpha = _env_float("GMB_NWD_ALPHA", nwd_alpha)
        nwd_alpha_min = _env_float("GMB_NWD_ALPHA_MIN", nwd_alpha_min)
        nwd_alpha_max = _env_float("GMB_NWD_ALPHA_MAX", nwd_alpha_max)
        quality_nwd_alpha = _env_float("GMB_QUALITY_NWD_ALPHA", quality_nwd_alpha)
        quality_alpha_min = _env_float("GMB_QUALITY_ALPHA_MIN", quality_alpha_min)
        quality_alpha_max = _env_float("GMB_QUALITY_ALPHA_MAX", quality_alpha_max)
        small_object_scale = _env_float("GMB_SMALL_OBJECT_SCALE", small_object_scale)
        scale_temperature = _env_float("GMB_SCALE_TEMPERATURE", scale_temperature)
        use_density_matcher = _env_bool("GMB_USE_DENSITY_MATCHER", use_density_matcher)
        density_gain = _env_float("GMB_DENSITY_GAIN", density_gain)
        density_tau = _env_float("GMB_DENSITY_TAU", density_tau)
        density_temperature = _env_float("GMB_DENSITY_TEMPERATURE", density_temperature)

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(
            cost_gain={"class": 2, "bbox": 5, "giou": 2},
            match_reg_mode=match_reg_mode,
            nwd_constant=nwd_constant,
            nwd_alpha=nwd_alpha,
            nwd_alpha_min=nwd_alpha_min,
            nwd_alpha_max=nwd_alpha_max,
            small_object_scale=small_object_scale,
            scale_temperature=scale_temperature,
            use_density_matcher=use_density_matcher,
            density_gain=density_gain,
            density_tau=density_tau,
            density_temperature=density_temperature,
        )
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

        self.reg_loss_mode = reg_loss_mode
        self.quality_label_mode = quality_label_mode
        self.nwd_constant = nwd_constant
        self.nwd_alpha = nwd_alpha
        self.nwd_alpha_min = nwd_alpha_min
        self.nwd_alpha_max = nwd_alpha_max
        self.quality_nwd_alpha = quality_nwd_alpha
        self.quality_alpha_min = quality_alpha_min
        self.quality_alpha_max = quality_alpha_max
        self.small_object_scale = small_object_scale
        self.scale_temperature = scale_temperature

    def _get_loss_class(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        gt_scores: torch.Tensor,
        num_gts: int,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]

        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _adaptive_nwd_alpha(
        self,
        gt_bboxes: torch.Tensor,
        min_weight: float | None = None,
        max_weight: float | None = None,
    ) -> torch.Tensor:
        return scale_adaptive_weight(
            gt_bboxes,
            small_object_scale=self.small_object_scale,
            scale_temperature=self.scale_temperature,
            min_weight=self.nwd_alpha_min if min_weight is None else min_weight,
            max_weight=self.nwd_alpha_max if max_weight is None else max_weight,
        )

    def _hybrid_reg_loss(self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        giou_loss = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True).squeeze(-1)

        if self.reg_loss_mode == "giou":
            return giou_loss

        nwd_loss = wasserstein_loss(pred_bboxes, gt_bboxes, xywh=True, constant=self.nwd_constant)
        if self.reg_loss_mode == "nwd":
            return nwd_loss
        if self.reg_loss_mode == "fixed_nwd_giou":
            alpha = gt_bboxes.new_full((gt_bboxes.shape[0],), self.nwd_alpha)
        elif self.reg_loss_mode == "sa_nwd_giou":
            alpha = self._adaptive_nwd_alpha(gt_bboxes)
        else:
            raise ValueError(
                "reg_loss_mode must be one of: 'giou', 'nwd', 'fixed_nwd_giou', 'sa_nwd_giou'."
            )
        return alpha * nwd_loss + (1.0 - alpha) * giou_loss

    def _get_loss_bbox(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * self._hybrid_reg_loss(pred_bboxes, gt_bboxes).sum() / len(gt_bboxes)
        return {k: v.squeeze() for k, v in loss.items()}

    def _quality_scores(self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        iou_quality = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1).clamp_(0.0, 1.0)

        if self.quality_label_mode == "iou":
            return iou_quality

        nwd_quality = wasserstein_score(
            pred_bboxes.detach(),
            gt_bboxes,
            xywh=True,
            constant=self.nwd_constant,
        ).clamp_(0.0, 1.0)
        if self.quality_label_mode == "nwd":
            return nwd_quality
        if self.quality_label_mode == "fixed_hybrid":
            beta = gt_bboxes.new_full((gt_bboxes.shape[0],), self.quality_nwd_alpha)
        elif self.quality_label_mode == "scale_hybrid":
            beta = self._adaptive_nwd_alpha(
                gt_bboxes,
                min_weight=self.quality_alpha_min,
                max_weight=self.quality_alpha_max,
            )
        else:
            raise ValueError(
                "quality_label_mode must be one of: 'iou', 'nwd', 'fixed_hybrid', 'scale_hybrid'."
            )
        return ((1.0 - beta) * iou_quality + beta * nwd_quality).clamp_(0.0, 1.0)

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]

        return {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }

    @staticmethod
    def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes_matched, gt_bboxes_matched = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes_matched):
            gt_scores[idx] = self._quality_scores(pred_bboxes_matched, gt_bboxes_matched)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes_matched), postfix),
            **self._get_loss_bbox(pred_bboxes_matched, gt_bboxes_matched, postfix),
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor],
        dn_num_group: int,
        gt_groups: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), (
                    f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                )
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
