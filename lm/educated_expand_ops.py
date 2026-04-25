"""
Standalone educated neuron-split ops (factored from ``RescalableModule`` in models.py).

Used by ``inflate_vit_educated`` for timm ViT; no optimizer state.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

SQRT_2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))


def expand_weight_matrix(
    W: torch.Tensor,
    in_expand_idx: Optional[torch.Tensor] = None,
    out_expand_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Dict[int, int]]]:
    """Duplicate selected output rows and/or input columns; return mapping like models.py."""
    mapping: Dict[str, Dict[int, int]] = {"in": {}, "out": {}}
    W = W.clone()
    if out_expand_idx is not None and len(out_expand_idx) > 0:
        if int(out_expand_idx.max()) >= W.size(0):
            raise IndexError(
                f"out_expand_idx max {int(out_expand_idx.max())} >= W.size(0) {W.size(0)}"
            )
        if int(out_expand_idx.min()) < 0:
            raise IndexError(f"out_expand_idx min {int(out_expand_idx.min())} < 0")
        first_new_row = W.size(0)
        W = torch.cat([W, W[out_expand_idx]], dim=0)
        for i, old in enumerate(out_expand_idx):
            mapping["out"][int(old)] = int(first_new_row + i)

    if in_expand_idx is not None and len(in_expand_idx) > 0:
        if int(in_expand_idx.max()) >= W.size(1):
            raise IndexError(
                f"in_expand_idx max {int(in_expand_idx.max())} >= W.size(1) {W.size(1)}"
            )
        if int(in_expand_idx.min()) < 0:
            raise IndexError(f"in_expand_idx min {int(in_expand_idx.min())} < 0")
        first_new_col = W.size(1)
        in_expand_idx = in_expand_idx.to(W.device)
        W = torch.cat([W, W[:, in_expand_idx]], dim=1)
        for i, old in enumerate(in_expand_idx):
            mapping["in"][int(old)] = int(first_new_col + i)

    return W, mapping


def case_5_scaling(W: torch.Tensor, mapping: Dict[str, Dict[int, int]]) -> torch.Tensor:
    affected_cols = list(mapping["in"].values()) + list(mapping["in"].keys())
    if affected_cols:
        W = W.clone()
        W[:, affected_cols] /= 2.0
    return W


def case_4_scaling(W: torch.Tensor, mapping: Dict[str, Dict[int, int]], growth_factor: float) -> torch.Tensor:
    """Attention-style quadrant scaling (matches RescalableModule.case_4_scaling)."""
    W = W.T.clone()
    old_rows = list(mapping["in"].keys())
    old_cols = list(mapping["out"].keys())
    new_rows = list(mapping["in"].values())
    new_cols = list(mapping["out"].values())
    affected_rows = old_rows + new_rows
    affected_cols = old_cols + new_cols
    scale_factor = growth_factor ** 0.25
    W *= scale_factor

    row_only_mask = torch.zeros(W.shape, dtype=torch.bool, device=W.device)
    col_only_mask = torch.zeros(W.shape, dtype=torch.bool, device=W.device)
    both_mask = torch.zeros(W.shape, dtype=torch.bool, device=W.device)

    non_affected_cols = list(set(range(W.shape[1])) - set(affected_cols))
    non_affected_rows = list(set(range(W.shape[0])) - set(affected_rows))

    if affected_rows and non_affected_cols:
        rows_idx, cols_idx = torch.meshgrid(
            torch.tensor(affected_rows, device=W.device),
            torch.tensor(non_affected_cols, device=W.device),
            indexing="ij",
        )
        row_only_mask[rows_idx, cols_idx] = True

    if non_affected_rows and affected_cols:
        rows_idx, cols_idx = torch.meshgrid(
            torch.tensor(non_affected_rows, device=W.device),
            torch.tensor(affected_cols, device=W.device),
            indexing="ij",
        )
        col_only_mask[rows_idx, cols_idx] = True

    if affected_rows and affected_cols:
        rows_idx, cols_idx = torch.meshgrid(
            torch.tensor(affected_rows, device=W.device),
            torch.tensor(affected_cols, device=W.device),
            indexing="ij",
        )
        both_mask[rows_idx, cols_idx] = True

    if col_only_mask.any():
        W[col_only_mask] /= SQRT_2.to(W.device)
    if row_only_mask.any():
        W[row_only_mask] /= 2.0
    if both_mask.any():
        W[both_mask] /= 2.0 * SQRT_2.to(W.device)
    return W.T


def apply_educated_noise(
    W: torch.Tensor,
    mapping: Dict[str, Dict[int, int]],
    std: float,
    noise_sigma: float,
) -> torch.Tensor:
    """Structured noise on quadrants (matches RescalableModule.apply_noise)."""
    if noise_sigma == 0.0 or std == 0.0:
        return W

    old_rows: List[int] = list(mapping["out"].keys())
    old_cols: List[int] = list(mapping["in"].keys())
    new_rows: List[int] = list(mapping["out"].values())
    new_cols: List[int] = list(mapping["in"].values())

    if not old_rows:
        old_rows = list(range(0, W.shape[0]))
    if not old_cols:
        old_cols = list(range(0, W.shape[1]))

    if not old_rows and not old_cols:
        return W

    eps_TL_shape = (len(old_rows), len(old_cols))
    eps_TL = torch.randn(eps_TL_shape, device=W.device, dtype=W.dtype) * std * noise_sigma

    noise = torch.zeros_like(W)

    if old_rows and old_cols:
        rows_idx, cols_idx = torch.meshgrid(
            torch.tensor(old_rows, device=W.device),
            torch.tensor(old_cols, device=W.device),
            indexing="ij",
        )
        noise[rows_idx, cols_idx] = eps_TL

    if old_rows and new_cols:
        rows_idx, cols_idx = torch.meshgrid(
            torch.tensor(old_rows, device=W.device),
            torch.tensor(new_cols, device=W.device),
            indexing="ij",
        )
        for i, new_col in enumerate(new_cols):
            if i < len(old_cols):
                noise[old_rows, new_col] = -eps_TL[:, i]

    if new_rows and old_cols:
        eps_BL_shape = (len(new_rows), len(old_cols))
        eps_BL = torch.randn(eps_BL_shape, device=W.device, dtype=W.dtype) * std * noise_sigma
        rows_idx, cols_idx = torch.meshgrid(
            torch.tensor(new_rows, device=W.device),
            torch.tensor(old_cols, device=W.device),
            indexing="ij",
        )
        noise[rows_idx, cols_idx] = eps_BL

    if new_rows and new_cols:
        rows_idx, cols_idx = torch.meshgrid(
            torch.tensor(new_rows, device=W.device),
            torch.tensor(new_cols, device=W.device),
            indexing="ij",
        )
        for i, new_col in enumerate(new_cols):
            if i < len(old_cols):
                noise[new_rows, new_col] = -eps_BL[:, i]

    return W + noise


def expand_bias_1d(
    bias: Optional[torch.Tensor],
    out_expand_idx: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if bias is None or out_expand_idx is None or len(out_expand_idx) == 0:
        return bias
    b = bias.detach().clone()
    return torch.cat([b, b[out_expand_idx]], dim=0)


def educated_expand_patch_proj(
    weight_2d: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_expand_idx: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Output-channel dup only (``GrowablePatchEmbedding.grow``): no case-5, no noise."""
    W_old = weight_2d.detach().clone()
    W_new, _ = expand_weight_matrix(W_old, None, out_expand_idx)
    b_new = expand_bias_1d(bias, out_expand_idx)
    return W_new, b_new


def make_width_expand_indices(
    dim: int,
    growth_factor: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random subset of columns/rows to duplicate (same count as GrowableSequential)."""
    num_expand = int(dim * growth_factor - dim)
    if num_expand <= 0:
        return torch.tensor([], dtype=torch.long)
    perm = torch.randperm(dim, device="cpu", generator=generator)
    return perm[:num_expand].long()


def educated_linear_case5(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    in_expand_idx: Optional[torch.Tensor],
    out_expand_idx: Optional[torch.Tensor],
    growth_factor: float,
    noise_sigma: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    W_old = weight.detach().clone()
    std_scale = W_old.std().item() if W_old.numel() > 0 else 0.0
    W_new, mapping = expand_weight_matrix(W_old, in_expand_idx, out_expand_idx)
    W_new = case_5_scaling(W_new, mapping)
    # matches expand_case_5: apply_noise(..., W_old.std() * noise_factor) with noise_factor=1
    W_new = apply_educated_noise(W_new, mapping, std_scale, noise_sigma)
    b_new = expand_bias_1d(bias, out_expand_idx)
    return W_new, b_new


def educated_linear_case4(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    in_expand_idx: Optional[torch.Tensor],
    out_expand_idx: Optional[torch.Tensor],
    growth_factor: float,
    noise_sigma: float,
    noise_sigma_attention: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    W_old = weight.detach().clone()
    std_scale = W_old.std().item() if W_old.numel() > 0 else 0.0
    W_new, mapping = expand_weight_matrix(W_old, in_expand_idx, out_expand_idx)
    W_new = case_4_scaling(W_new, mapping, growth_factor)
    # matches expand_attention_score_layer: apply_noise(..., noise_sigma_attention * W_old.std())
    W_new = apply_educated_noise(
        W_new, mapping, noise_sigma_attention * std_scale, noise_sigma
    )
    b_new = expand_bias_1d(bias, out_expand_idx)
    return W_new, b_new
