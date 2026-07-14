# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


def scale_adaptive_weight(
    gt_bboxes: torch.Tensor,
    small_object_scale: float = 0.030,
    scale_temperature: float = 0.010,
    min_weight: float = 0.0,
    max_weight: float = 0.70,
) -> torch.Tensor:
    """Return a per-GT gate that is large for small boxes and small for large boxes."""
    if gt_bboxes.numel() == 0:
        return gt_bboxes.new_zeros((0,))

    wh = gt_bboxes[:, 2:].clamp_min(0.0)
    scale = torch.sqrt((wh[:, 0] * wh[:, 1]).clamp_min(1e-12))
    gate = torch.sigmoid((small_object_scale - scale) / max(scale_temperature, 1e-7))
    return min_weight + (max_weight - min_weight) * gate


def density_adaptive_weight(
    gt_bboxes: torch.Tensor,
    gt_groups: list[int],
    density_tau: float = 0.030,
    density_temperature: float = 0.010,
    max_weight: float = 1.0,
) -> torch.Tensor:
    """Return a per-GT density gate based on nearest-neighbor center distance inside each image."""
    if gt_bboxes.numel() == 0:
        return gt_bboxes.new_zeros((0,))

    weights = gt_bboxes.new_zeros((gt_bboxes.shape[0],))
    start = 0
    for num_gt in gt_groups:
        end = start + num_gt
        if num_gt > 1:
            centers = gt_bboxes[start:end, :2]
            dist = torch.cdist(centers, centers, p=2)
            dist = dist.masked_fill(torch.eye(num_gt, dtype=torch.bool, device=gt_bboxes.device), float("inf"))
            nearest = dist.min(dim=1).values
            weights[start:end] = max_weight * torch.sigmoid(
                (density_tau - nearest) / max(density_temperature, 1e-7)
            )
        start = end
    return weights


def pairwise_nwd(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    constant: float | torch.Tensor = 0.05,
) -> torch.Tensor:
    """Pairwise normalized Gaussian Wasserstein similarity for xywh boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    cx1, cy1, w1, h1 = boxes1.split(1, dim=-1)
    cx2, cy2, w2, h2 = boxes2.split(1, dim=-1)

    center_distance = (cx1 - cx2.T).pow(2) + (cy1 - cy2.T).pow(2)
    wh_distance = ((w1 - w2.T).pow(2) + (h1 - h2.T).pow(2)) / 4.0
    wasserstein_2 = center_distance + wh_distance + eps

    constant = torch.as_tensor(constant, device=boxes1.device, dtype=boxes1.dtype).clamp_min(eps)
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)


class HungarianMatcher(nn.Module):
    """Hungarian matcher with ablation-friendly UAV small-object matching options."""

    def __init__(
        self,
        cost_gain: dict[str, float] | None = None,
        use_fl: bool = True,
        with_mask: bool = False,
        num_sample_points: int = 12544,
        alpha: float = 0.25,
        gamma: float = 2.0,
        match_reg_mode: str = "sa_nwd_giou",
        nwd_constant: float = 0.05,
        nwd_alpha: float = 0.50,
        nwd_alpha_min: float = 0.0,
        nwd_alpha_max: float = 0.70,
        small_object_scale: float = 0.030,
        scale_temperature: float = 0.010,
        use_density_matcher: bool = True,
        density_gain: float = 1.0,
        density_tau: float = 0.030,
        density_temperature: float = 0.010,
    ):
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

        self.match_reg_mode = match_reg_mode
        self.nwd_constant = nwd_constant
        self.nwd_alpha = nwd_alpha
        self.nwd_alpha_min = nwd_alpha_min
        self.nwd_alpha_max = nwd_alpha_max
        self.small_object_scale = small_object_scale
        self.scale_temperature = scale_temperature

        self.use_density_matcher = use_density_matcher
        self.density_gain = density_gain
        self.density_tau = density_tau
        self.density_temperature = density_temperature

    def _reg_cost(self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        giou_cost = 1.0 - bbox_iou(
            pred_bboxes.unsqueeze(1),
            gt_bboxes.unsqueeze(0),
            xywh=True,
            GIoU=True,
        ).squeeze(-1)

        if self.match_reg_mode == "giou":
            return giou_cost

        nwd_cost = 1.0 - pairwise_nwd(pred_bboxes, gt_bboxes, constant=self.nwd_constant)
        if self.match_reg_mode == "nwd":
            return nwd_cost
        if self.match_reg_mode == "fixed_nwd_giou":
            alpha = gt_bboxes.new_full((gt_bboxes.shape[0],), self.nwd_alpha)
        elif self.match_reg_mode == "sa_nwd_giou":
            alpha = scale_adaptive_weight(
                gt_bboxes,
                small_object_scale=self.small_object_scale,
                scale_temperature=self.scale_temperature,
                min_weight=self.nwd_alpha_min,
                max_weight=self.nwd_alpha_max,
            )
        else:
            raise ValueError(
                "match_reg_mode must be one of: 'giou', 'nwd', 'fixed_nwd_giou', 'sa_nwd_giou'."
            )
        return alpha.unsqueeze(0) * nwd_cost + (1.0 - alpha.unsqueeze(0)) * giou_cost

    def _density_cost(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_groups: list[int],
    ) -> torch.Tensor:
        if not self.use_density_matcher or self.density_gain <= 0:
            return pred_bboxes.new_zeros((pred_bboxes.shape[0], gt_bboxes.shape[0]))

        density = density_adaptive_weight(
            gt_bboxes,
            gt_groups,
            density_tau=self.density_tau,
            density_temperature=self.density_temperature,
        )
        center_cost = (pred_bboxes[:, None, :2] - gt_bboxes[None, :, :2]).abs().sum(-1)
        return self.density_gain * center_cost * density.unsqueeze(0)

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * pred_scores.pow(self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * (1 - pred_scores).pow(self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)
        cost_reg = self._reg_cost(pred_bboxes, gt_bboxes)

        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_reg
            + self._density_cost(pred_bboxes, gt_bboxes, gt_groups)
        )

        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups_tensor = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups_tensor[k])
            for k, (i, j) in enumerate(indices)
        ]

    def _cost_mask(
        self,
        bs: int,
        num_gts: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute mask matching cost. Kept for compatibility with RT-DETR segmentation variants."""
        assert masks is not None and gt_mask is not None, "Make sure the input has `mask` and `gt_mask`."

        sample_points = torch.rand([bs, 1, self.num_sample_points, 2], device=masks.device)
        sample_points = 2.0 * sample_points - 1.0

        out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
        out_mask = out_mask.flatten(0, 1)

        tgt_mask = torch.cat(gt_mask).unsqueeze(1)
        sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
        tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze(2).squeeze(1)

        with torch.amp.autocast("cuda", enabled=False):
            pos_cost_mask = F.binary_cross_entropy_with_logits(
                out_mask, torch.ones_like(out_mask), reduction="none"
            )
            neg_cost_mask = F.binary_cross_entropy_with_logits(
                out_mask, torch.zeros_like(out_mask), reduction="none"
            )
            cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
            cost_mask /= self.num_sample_points

            out_mask = F.sigmoid(out_mask)
            numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
            denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)

            C = self.cost_gain["mask"] * cost_mask + self.cost_gain["dice"] * cost_dice
        return C


def get_cdn_group(
    batch: dict[str, Any],
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_dn: int = 100,
    cls_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    training: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    bs = len(gt_groups)
    gt_cls = batch["cls"]
    gt_bbox = batch["bboxes"]
    b_idx = batch["batch_idx"]

    dn_cls = gt_cls.repeat(2 * num_group)
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)

    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        mask = torch.rand(dn_cls.shape, device=dn_cls.device) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)

    num_dn = int(max_nums * 2 * num_group)
    dn_cls_embed = class_embed[dn_cls]
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)

    map_indices = torch.cat([torch.arange(num, dtype=torch.long, device=gt_cls.device) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    attn_mask[num_dn:, :num_dn] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )
