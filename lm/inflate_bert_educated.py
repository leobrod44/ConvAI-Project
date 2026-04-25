# This python file is for expanding Pre-LN model, note currently we only support unchanged head dimension (dim of each head)
# Check Pre-LN model defined in preln_bert.py
# In this file, inflate and expansion are used interchangeably

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers import (
    BertConfig,
    BertForMaskedLM,
)
from preln_bert import (
    modBertOnlyMLMHead, modBertAttention, modBertLayer, modBertEmbeddings,
    modBertLMPredictionHead, modBertModel, modBertForMaskedLM,
)
from timm.layers.weight_init import trunc_normal_
from transformers.models.bert.modeling_bert import (
    BertOnlyMLMHead, BertAttention, BertLayer, BertEmbeddings,
    BertLMPredictionHead, BertModel,
)
import random

BertLayerNorm = torch.nn.LayerNorm

import torch.nn.init as init
from torch import Tensor
from typing import Tuple, Union, Optional

try:
    from .educated_expand_ops import apply_educated_noise
except ImportError:
    from educated_expand_ops import apply_educated_noise


# ---------------------------------------------------------------------------
# Noise helper for symmetric splits
# ---------------------------------------------------------------------------

def _apply_symmetric_in_split_noise(
    weight: Tensor,
    *,
    in_features_old: int,
    in_divisor: int,
    noise_sigma: float,
    scale: float,
    in_pattern: str,
) -> Tensor:
    h, o, i_new = weight.shape
    if in_divisor < 2 or i_new != in_divisor * in_features_old:
        return weight
    w2 = weight.reshape(h * o, i_new)
    std_scale = float(w2.std().item()) if w2.numel() > 0 else 0.0
    if std_scale == 0.0:
        return weight
    if in_pattern == "symmetric_scaled":
        std_scale = std_scale * float(scale)
    out2 = w2
    in_old = in_features_old
    for b in range(in_divisor - 1):
        mapping_in = {j + b * in_old: j + (b + 1) * in_old for j in range(in_old)}
        out2 = apply_educated_noise(
            out2,
            {"in": mapping_in, "out": {}},
            std_scale,
            noise_sigma,
        )
    return out2.reshape(h, o, i_new)


# ---------------------------------------------------------------------------
# Optimizer state inflation helper
# ---------------------------------------------------------------------------

def _inflate_optimizer_state(
    orig_param: Optional[nn.Parameter],
    inf_param: Optional[nn.Parameter],
    orig_optimizer: Optional[torch.optim.Optimizer],
    inf_optimizer: Optional[torch.optim.Optimizer],
    *,
    # Shape info needed to reconstruct the block structure
    head_old: int,
    out_old: int,
    in_old: int,
    head_new: int,
    out_new: int,
    in_new: int,
    # Patterns — same ones used for the weight
    head_pattern: str,
    out_pattern: str,
    in_pattern: str,
    scalecirc: float,
    noise_sigma: float,
    # Whether this is a 3-D (MHA) or 2-D (linear) param
    flag_with_head: bool,
):
    """
    Inflate Adam optimizer state (exp_avg, exp_avg_sq) from orig_param to
    inf_param using the identical block structure that was applied to the
    weights in inflate_fc_nonint_heads / project_linear.

    Rules
    -----
    - exp_avg   : inflated with the same pattern as the weight (moment
                  lives in the same space as the gradient, which lives in
                  the same space as the weight).
    - exp_avg_sq: inflated with the *absolute* version of the pattern —
                  we never want signed cancellation in the second moment
                  because it tracks squared magnitudes.  Concretely:
                    symmetric / symmetric_scaled -> replicate + scale,
                    no signed noise (noise_sigma=0 for sq).
    - step      : reset to 0 (fresh start requested by caller).
    """
    if orig_optimizer is None or inf_optimizer is None:
        return
    if orig_param is None or inf_param is None:
        return
    if orig_param not in orig_optimizer.state:
        return  # orig param never stepped — nothing to transfer

    orig_state = orig_optimizer.state[orig_param]

    # Ensure inf_optimizer has a state dict for inf_param (it might be empty
    # if inf_optimizer was just constructed and never stepped).
    if inf_param not in inf_optimizer.state:
        inf_optimizer.state[inf_param] = {}

    inf_state = inf_optimizer.state[inf_param]

    # Always give a fresh step counter
    if "step" in orig_state:
        inf_state["step"] = torch.zeros_like(orig_state["step"])
    else:
        inf_state["step"] = torch.tensor(0, dtype=torch.long)

    in_divisor = in_new // in_old  # exact multiple guaranteed by caller
    out_divisor = out_new // out_old
    head_divisor = head_new // head_old

    for key in ("exp_avg", "exp_avg_sq"):
        if key not in orig_state:
            continue

        s = orig_state[key].detach().clone()  # original state tensor

        # Reshape to 3-D [head, out, in] to mirror project_linear's view
        if flag_with_head:
            # s is [head_old * out_old, in_old] stored flat
            s = s.reshape(head_old, out_old, in_old)
        else:
            s = s.unsqueeze(0)  # [1, out_old, in_old]

        # --- expand out dim ---
        s = inflate(s, out_new, dim=1, pattern=out_pattern)
        # --- expand head dim ---
        s = inflate(s, head_new, dim=0, pattern=head_pattern)
        # s is now [head_new, out_new, in_old]

        # --- expand in dim ---
        if in_pattern in ("symmetric", "symmetric_scaled"):
            # Replicate along in-dim then scale — same as weight path.
            # For exp_avg_sq we skip signed noise (always non-negative).
            w = s.unsqueeze(2).repeat(1, 1, in_divisor, 1)
            if in_pattern == "symmetric":
                w = w / in_divisor
            else:
                w = w * (scalecirc / in_divisor)
            s = w.reshape(head_new, out_new, in_new)

            # For exp_avg only: inject the same ±e noise so the momentum
            # is consistent with the weight perturbation direction.
            # For exp_avg_sq: no signed noise (squared moment must stay ≥ 0
            # and we don't want cancellation artifacts).
            if key == "exp_avg" and noise_sigma > 0.0:
                s = _apply_symmetric_in_split_noise(
                    s,
                    in_features_old=in_old,
                    in_divisor=in_divisor,
                    noise_sigma=noise_sigma,
                    scale=scalecirc,
                    in_pattern=in_pattern,
                )
        else:
            # Classic circular/average/zero — just replicate
            s = inflate(s, in_new, dim=2, pattern=in_pattern)

        # Flatten back to 2-D
        if flag_with_head:
            s = s.reshape(head_new * out_new, in_new)
        else:
            s = s.squeeze(0)  # [out_new, in_new]

        inf_state[key] = s.to(inf_param.device)

    # Copy any remaining scalar / non-tensor state entries
    for k, v in orig_state.items():
        if k not in ("exp_avg", "exp_avg_sq", "step"):
            inf_state[k] = v


def _inflate_optimizer_state_1d(
    orig_param: Optional[nn.Parameter],
    inf_param: Optional[nn.Parameter],
    orig_optimizer: Optional[torch.optim.Optimizer],
    inf_optimizer: Optional[torch.optim.Optimizer],
    *,
    features_new: int,
    pattern: str,
):
    """
    Inflate Adam optimizer state for 1-D parameters (biases, LayerNorm
    weight/bias) and 2-D embedding tables (inflated along dim=1).

    Handles three shapes:
      - 1-D [features_old]              -> [features_new]          (bias / LN)
      - 2-D [vocab, features_old]       -> [vocab, features_new]   (embeddings)

    exp_avg_sq never receives signed noise — it is always non-negative.
    step is reset to 0 (fresh start).
    """
    if orig_optimizer is None or inf_optimizer is None:
        return
    if orig_param is None or inf_param is None:
        return
    if orig_param not in orig_optimizer.state:
        return

    orig_state = orig_optimizer.state[orig_param]

    if inf_param not in inf_optimizer.state:
        inf_optimizer.state[inf_param] = {}
    inf_state = inf_optimizer.state[inf_param]

    if "step" in orig_state:
        inf_state["step"] = torch.zeros_like(orig_state["step"])
    else:
        inf_state["step"] = torch.tensor(0, dtype=torch.long)

    for key in ("exp_avg", "exp_avg_sq"):
        if key not in orig_state:
            continue
        s = orig_state[key].detach().clone()
        if s.dim() == 1:
            # bias / LN param: [features_old] -> [features_new]
            s = inflate(s, features_new, dim=0, pattern=pattern)
        elif s.dim() == 2:
            # embedding table: [vocab, features_old] -> [vocab, features_new]
            s = inflate(s, features_new, dim=1, pattern=pattern)
        inf_state[key] = s.to(inf_param.device)

    for k, v in orig_state.items():
        if k not in ("exp_avg", "exp_avg_sq", "step"):
            inf_state[k] = v


# ---------------------------------------------------------------------------
# Core tensor inflation
# ---------------------------------------------------------------------------

def inflate(
    features: Tensor,
    features_new: int,
    dim: int = 1,
    pattern: str = 'circular',
) -> Tensor:
    """Expand a weight/feature tensor along ``dim`` to size ``features_new``."""
    device = features.device
    features_dim = features.dim()
    features_old = features.size(dim)

    if dim >= features_dim:
        raise ValueError('The specified dimension exceeds the feature dimension.')
    if features_old > features_new:
        raise ValueError('The inflated feature size is smaller than the original one.')

    divisor = features_new // features_old
    residue = features_new % features_old

    assert pattern in [
        'circular', 'average', 'zero', 'gauss', 'null', 'unif',
        'ones', 'unif01', 'symmetric', 'symmetric_scaled',
    ], 'The expansion pattern "{}" is not supported.'.format(pattern)

    if pattern == 'null':
        return torch.zeros(
            features.size()[:dim] + (features_new,) + features.size()[dim + 1:]
        ).to(device)

    features = features.repeat([1] * dim + [divisor] + [1] * (features_dim - 1 - dim))

    if residue > 0:
        idx = torch.tensor(range(residue)).to(device)
        if pattern in ('circular', 'symmetric', 'symmetric_scaled'):
            features_ = features.index_select(dim, idx)
        elif pattern == 'average':
            features_ = (
                torch.ones(features.size()[:dim] + (residue,) + features.size()[dim + 1:]).to(device)
                * torch.mean(features, dim=dim, keepdim=True)
            )
        elif pattern == 'gauss':
            features_ = (
                torch.randn(features.size()[:dim] + (residue,) + features.size()[dim + 1:]).to(device)
                * torch.mean(features, dim=dim, keepdim=True)
            )
        elif pattern == 'unif':
            print('Use 1.0 for unif')
            features_ = 2 * torch.rand(
                features.size()[:dim] + (residue,) + features.size()[dim + 1:]
            ).to(device) - 1
        elif pattern == 'ones':
            features_ = torch.ones(
                features.size()[:dim] + (residue,) + features.size()[dim + 1:]
            ).to(device)
        elif pattern == 'unif01':
            print('Unif 0-1')
            features_ = torch.rand(
                features.size()[:dim] + (residue,) + features.size()[dim + 1:]
            ).to(device)
        else:  # zero
            features_ = torch.zeros(
                features.size()[:dim] + (residue,) + features.size()[dim + 1:]
            ).to(device)
        features = torch.cat((features, features_.to(device)), dim=dim)

    return features


# ---------------------------------------------------------------------------
# Core linear-layer projection
# ---------------------------------------------------------------------------

def project_linear(
    weight: Tensor,
    bias: Tensor,
    weight_: Tensor,
    head_pattern: str = 'circular',
    in_pattern: str = 'circular',
    out_pattern: str = 'circular',
    circ_mode: str = 'projection',
    cancel: bool = False,
    scalezero: float = 1.0,
    scalecirc: float = 1.0,
    scalecancel: float = 1.0,
    noise_sigma: float = 0.0,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Expand a linear layer weight (and bias) from shape
    [head_old, out_old, in_old] to [head_new, out_new, in_new].
    """
    head_old, out_old, in_old = weight.shape
    head_new, out_new, in_new = weight_.shape

    if head_old > head_new or out_old > out_new or in_old > in_new:
        raise ValueError('The expanded model is smaller than the original model.')

    in_divisor = in_new // in_old
    in_residue = in_new % in_old

    if 'circular' in out_pattern:
        out_pattern = 'circular'
    if 'circular' in head_pattern:
        head_pattern = 'circular'

    assert out_pattern in ('circular', 'average', 'zero', 'symmetric', 'symmetric_scaled'), \
        'Unsupported out_pattern: {}'.format(out_pattern)
    assert in_pattern in ('circular', 'average', 'zero', 'symmetric', 'symmetric_scaled'), \
        'Unsupported in_pattern: {}'.format(in_pattern)
    assert circ_mode in ('comp', 'projection'), \
        'Unsupported circ_mode: {}'.format(circ_mode)

    # --- Step 1: expand out and head dims ---
    weight = inflate(weight, out_new, dim=1, pattern=out_pattern)
    weight = inflate(weight, head_new, dim=0, pattern=head_pattern)

    # ==========================================================================
    # SYMMETRIC PATH
    # ==========================================================================
    if in_pattern in ('symmetric', 'symmetric_scaled'):
        if in_residue != 0:
            raise NotImplementedError(
                'symmetric/symmetric_scaled requires in_new to be an exact multiple '
                'of in_old. Got in_old={}, in_new={} (residue={}).'.format(
                    in_old, in_new, in_residue)
            )

        print('Use symmetric (net2net split): in_divisor={}, scalecirc={}, noise_sigma={}'.format(
            in_divisor, scalecirc, noise_sigma))

        w = weight.unsqueeze(2).repeat(1, 1, in_divisor, 1)

        if in_pattern == 'symmetric':
            w = w / in_divisor
        else:
            w = w * (scalecirc / in_divisor)

        final_weight = w.reshape(head_new, out_new, in_new)

        if noise_sigma > 0.0:
            final_weight = _apply_symmetric_in_split_noise(
                final_weight,
                in_features_old=in_old,
                in_divisor=in_divisor,
                noise_sigma=noise_sigma,
                scale=scalecirc,
                in_pattern=in_pattern,
            )

        if bias is not None:
            bias = inflate(bias, out_new, dim=1, pattern=out_pattern)
            bias = inflate(bias, head_new, dim=0, pattern=head_pattern)

        return final_weight, bias

    # ==========================================================================
    # CLASSIC PATHS (circular / average / zero)
    # ==========================================================================
    if in_residue == 0:
        print('Width is divisible.')

        if cancel:
            assert weight.abs().sum() == 0
            print('cancelzero, R.V. scale={}'.format(scalecancel))
            weight_ = weight_ * scalecancel
            weight_ = weight_.reshape(head_new, out_new, in_divisor, in_old)
            weight = weight_ - (weight_.sum(dim=2) - weight).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
            final_weight = weight.reshape(head_new, out_new, in_new)

        elif circ_mode == 'projection':
            print('projection circular (from {}) R.V. scale={}'.format(in_pattern, scalecirc))
            weight_ = weight_ * scalecirc
            weight_ = weight_.reshape(head_new, out_new, in_divisor, in_old)
            weight = weight_ - (weight_.sum(dim=2) - weight).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
            final_weight = weight.reshape(head_new, out_new, in_new)

        elif circ_mode == 'comp':
            print('comp circular (from {}) R.V. scale={}'.format(in_pattern, scalecirc))
            assert in_divisor == 2, 'comp circ mode only supports in_divisor == 2'
            weight_ = weight_.reshape(head_new, out_new, in_divisor, in_old)[:, :, 0, :] * scalecirc
            weight_compensate = (weight - weight_).detach().clone()
            final_weight = torch.cat([weight_compensate, weight_], dim=2).reshape(head_new, out_new, in_new)

        else:
            raise ValueError('Unsupported circ_mode: {}'.format(circ_mode))

    else:
        if in_pattern == 'circular':
            if cancel or circ_mode == 'projection':
                if cancel:
                    assert weight.abs().sum() == 0
                    print('cancelzero (residue), R.V. scale={}'.format(scalecancel))
                    weight_ = weight_ * scalecancel
                else:
                    print('projection circular (residue) R.V. scale={}'.format(scalecirc))
                    weight_ = weight_ * scalecirc
                weight_0_, weight_r_ = weight_.split([in_divisor * in_old, in_residue], dim=2)
                weight_0_ = weight_0_.reshape(head_new, out_new, in_divisor, in_old)
                weight_0_, weight_1_ = weight_0_.split([in_residue, in_old - in_residue], dim=3)
                weight_0_ = torch.cat((weight_0_, weight_r_.unsqueeze(2)), dim=2)

                weight_0, weight_1 = weight.split([in_residue, in_old - in_residue], dim=2)
                weight_0 = weight_0_ - (weight_0_.sum(dim=2) - weight_0).unsqueeze(2).repeat(1, 1, in_divisor + 1, 1) / (in_divisor + 1)
                weight_1 = weight_1_ - (weight_1_.sum(dim=2) - weight_1).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor

                weight_0, weight_r = weight_0.split([in_divisor, 1], dim=2)
                weight_0 = torch.cat((weight_0, weight_1), dim=3)
                weight_r = weight_r.squeeze(2)

            elif circ_mode == 'comp':
                print('comp circular (residue) scale={}'.format(scalecirc))
                weight_0_, weight_r_ = weight_.split([in_divisor * in_old, in_residue], dim=2)
                weight_0_ = weight_0_.reshape(head_new, out_new, in_divisor, in_old)

                weight_0, weight_1 = weight.split([in_residue, in_old - in_residue], dim=2)
                assert weight_r_.size() == weight_0.size()
                weight_r_ = weight_r_ * scalecirc
                weight_0_, weight_1_ = weight_0_.split([in_residue, in_old - in_residue], dim=3)
                weight_0_cat = torch.cat((weight_0_, weight_r_.unsqueeze(2)), dim=2)

                weight_0 = weight_0_ - (weight_0_cat.sum(dim=2) - weight_0).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
                weight_1 = weight_1_ - (weight_1_.sum(dim=2) - weight_1).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
                weight_0 = torch.cat((weight_0, weight_1), dim=3)
                weight_r = weight_r_
            else:
                raise

        else:
            weight_0_, weight_r_ = weight_.split([in_divisor * in_old, in_residue], dim=2)
            weight_0_ = weight_0_.reshape(head_new, out_new, in_divisor, in_old)
            weight_0 = weight_0_ - (weight_0_.sum(dim=2) - weight).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor

            if in_pattern == 'average':
                print('average residue expansion.')
                weight_r = weight_r_ - weight_r_.sum(dim=2).unsqueeze(2).repeat(1, 1, in_residue) / in_residue
            elif in_pattern == 'zero':
                print('zero residue expansion, scale={}'.format(scalezero))
                weight_r = weight_r_ * scalezero
            else:
                raise

        weight_0 = weight_0.reshape(head_new, out_new, in_divisor * in_old)
        final_weight = torch.cat((weight_0, weight_r), dim=2)

    if bias is not None:
        bias = inflate(bias, out_new, dim=1, pattern=out_pattern)
        bias = inflate(bias, head_new, dim=0, pattern=head_pattern)

    return final_weight, bias


# ---------------------------------------------------------------------------
# inflate_fc_nonint_heads — dispatch wrapper + optimizer state transfer
# ---------------------------------------------------------------------------

def inflate_fc_nonint_heads(
    orig_weight, orig_bias,
    new_heads, new_out_channels, new_in_channels,
    heads_pattern, out_pattern, in_pattern,
    mode, device='cuda', inf_weight=None,
    AKI_weight=None, AKI_bias=None, indices=None,
    scalezero=1.0, scalecancel=1.0, scalecirc=1.0,
    circ_mode='projection', noise_sigma: float = 0.0,
    # --- optimizer state transfer ---
    orig_param: Optional[nn.Parameter] = None,
    inf_param: Optional[nn.Parameter] = None,
    orig_optimizer: Optional[torch.optim.Optimizer] = None,
    inf_optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Function-preserving expansion of a fully-connected or multi-head-attention
    weight tensor, with optional in-place optimizer state transfer.

    New parameters
    --------------
    orig_param      : the nn.Parameter in orig_bert whose state lives in orig_optimizer
    inf_param       : the nn.Parameter in inf_bert that inf_optimizer should track
    orig_optimizer  : AdamW / Adam optimizer associated with orig_bert (already stepped)
    inf_optimizer   : freshly constructed optimizer associated with inf_bert
                      (its state for inf_param will be populated here)
    """
    flag_with_head = True
    if len(orig_weight.size()) == 2:
        assert new_heads == 1, 'weight only has 2 dims; heads expansion not supported'
        flag_with_head = False
        orig_weight = orig_weight.unsqueeze(0)
        orig_bias   = orig_bias.unsqueeze(0) if orig_bias is not None else None
        if inf_weight  is not None: inf_weight  = inf_weight.unsqueeze(0)
        if AKI_weight  is not None: AKI_weight  = AKI_weight.unsqueeze(0)
        if AKI_bias    is not None: AKI_bias    = AKI_bias.unsqueeze(0)
        heads, out_channels, in_channels = orig_weight.size()
    else:
        heads, out_channels, in_channels = orig_weight.size()
        assert new_out_channels == out_channels, \
            'out_channels must not change when increasing number of heads'

    assert in_pattern in (
        'circular', 'zero', 'average', 'symmetric', 'symmetric_scaled'
    ), 'Unsupported in_pattern: {}'.format(in_pattern)
    assert mode in (
        'proj', 'net2net', 'allzero', 'cancelzero', 'AKI', 'AKIproj', 'nearzero'
    ), 'Unsupported mode: {}'.format(mode)

    # ------------------------------------------------------------------
    # Mode pre-processing
    # ------------------------------------------------------------------
    if mode == 'net2net':
        print('Use net2net')
        assert circ_mode == 'projection'
        inf_weight = torch.zeros_like(inf_weight)

    elif mode in ('AKI', 'AKIproj'):
        assert AKI_weight is not None
        if orig_bias is not None:
            assert AKI_bias is not None
        if mode == 'AKI':
            inf_weight = torch.zeros_like(inf_weight)
            assert circ_mode == 'projection'
            print('AKI, in_pattern={}'.format(in_pattern))
        else:
            print('AKIproj, in_pattern={}'.format(in_pattern))
        if flag_with_head:
            delta = new_heads - heads
            orig_weight = torch.cat([orig_weight, AKI_weight[:delta].detach().clone()], dim=0)
            orig_bias   = torch.cat([orig_bias,   AKI_bias[:delta].detach().clone()], dim=0) if orig_bias is not None else None
        else:
            delta = new_out_channels - out_channels
            orig_weight = torch.cat([orig_weight, AKI_weight[:, :delta].detach().clone()], dim=1)
            orig_bias   = torch.cat([orig_bias,   AKI_bias[:, :delta].detach().clone()], dim=1) if orig_bias is not None else None

    elif mode == 'cancelzero':
        print('Use cancelzero')
        orig_weight = torch.zeros_like(orig_weight)
        if orig_bias is not None:
            orig_bias = torch.zeros_like(orig_bias)

    elif mode == 'allzero':
        print('Use allzero')
        zero_weight = torch.zeros_like(inf_weight)
        zero_bias   = None
        if orig_bias is not None:
            h_new, o_new, _ = inf_weight.shape
            zero_bias = torch.zeros(h_new * o_new).to(device)
        if not flag_with_head:
            zero_weight = zero_weight.squeeze(0)
        # allzero: optimizer state for inf_param is left as zeros (fresh start)
        return zero_weight, zero_bias

    elif mode == 'nearzero':
        print('Use nearzero')
        small_weight = inf_weight.detach().clone() / 10
        small_bias   = None
        if orig_bias is not None:
            h_new, o_new, _ = inf_weight.shape
            small_bias = torch.zeros(h_new * o_new).to(device)
        if not flag_with_head:
            small_weight = small_weight.squeeze(0)
        return small_weight, small_bias

    elif mode == 'proj':
        print('proj, in_pattern={}, circ_mode={}'.format(in_pattern, circ_mode))

    # ------------------------------------------------------------------
    # Capture pre-expansion shape for optimizer state inflation
    # ------------------------------------------------------------------
    head_old, out_old, in_old = orig_weight.shape
    head_new = new_heads
    out_new  = new_out_channels
    in_new   = new_in_channels

    # ------------------------------------------------------------------
    # Core expansion
    # ------------------------------------------------------------------
    expand_matrix, stack_b = project_linear(
        orig_weight, orig_bias, inf_weight,
        head_pattern=heads_pattern,
        in_pattern=in_pattern,
        out_pattern=out_pattern,
        circ_mode=circ_mode,
        cancel=(mode == 'cancelzero'),
        scalezero=scalezero,
        scalecancel=scalecancel,
        scalecirc=scalecirc,
        noise_sigma=noise_sigma,
    )

    # ------------------------------------------------------------------
    # Optimizer state transfer (Option A: done right here after expansion)
    # ------------------------------------------------------------------
    _inflate_optimizer_state(
        orig_param=orig_param,
        inf_param=inf_param,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
        head_old=head_old,
        out_old=out_old,
        in_old=in_old,
        head_new=head_new,
        out_new=out_new,
        in_new=in_new,
        head_pattern=heads_pattern,
        out_pattern=out_pattern,
        in_pattern=in_pattern,
        scalecirc=scalecirc,
        noise_sigma=noise_sigma,
        flag_with_head=flag_with_head,
    )

    if not flag_with_head:
        assert expand_matrix.size(0) == 1
        expand_matrix = expand_matrix.squeeze(0)
        if stack_b is not None:
            stack_b = stack_b.squeeze(0).detach().clone()

    return expand_matrix, stack_b


# ---------------------------------------------------------------------------
# Embedding / LayerNorm helpers
# ---------------------------------------------------------------------------

def _apply_embedding_noise(tensor, d_old: int, noise_sigma: float):
    if noise_sigma <= 0.0:
        return tensor
    import torch
    N, d_new = tensor.shape
    D = d_new // d_old
    if D < 2:
        return tensor
    std_scale = tensor.std().item()
    if std_scale == 0.0:
        return tensor
    eps_scale = std_scale * noise_sigma
    out = tensor.clone()
    for b in range(D - 1):
        eps = torch.randn(N, d_old, dtype=tensor.dtype, device=tensor.device) * eps_scale
        out[:, b * d_old       : (b + 1) * d_old] -= eps
        out[:, (b + 1) * d_old : (b + 2) * d_old] += eps
    return out


def inflate_BertEmbeddings(
    orig_layer, inf_layer,
    pattern='average', ln_pattern='average', ln_bias_pattern='average',
    device='cuda', noise_sigma: float = 0.0,
):
    assert isinstance(orig_layer, (BertEmbeddings, modBertEmbeddings))
    assert isinstance(inf_layer,  (BertEmbeddings, modBertEmbeddings))
    assert hasattr(orig_layer, 'LayerNorm') == hasattr(inf_layer, 'LayerNorm')

    def _inflate_embed(orig_w, new_size):
        inflated = inflate(orig_w.detach().clone(), new_size, dim=1, pattern=pattern)
        return _apply_embedding_noise(inflated, orig_w.size(1), noise_sigma)

    with torch.no_grad():
        inf_layer.word_embeddings.weight.data = _inflate_embed(
            orig_layer.word_embeddings.weight,
            inf_layer.word_embeddings.weight.size(1),
        )
        inf_layer.position_embeddings.weight.data = _inflate_embed(
            orig_layer.position_embeddings.weight,
            inf_layer.position_embeddings.weight.size(1),
        )
        inf_layer.token_type_embeddings.weight.data = _inflate_embed(
            orig_layer.token_type_embeddings.weight,
            inf_layer.token_type_embeddings.weight.size(1),
        )

    if hasattr(orig_layer, 'LayerNorm'):
        inflate_ln(
            orig_layer.LayerNorm, inf_layer.LayerNorm,
            pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device,
        )


def inflate_ln(
    orig_ln_layer, inf_ln_layer,
    pattern='zero', bias_pattern='average',
    scale_for_bert=False, device='cuda',
    orig_optimizer: Optional[torch.optim.Optimizer] = None,
    inf_optimizer: Optional[torch.optim.Optimizer] = None,
):
    assert isinstance(orig_ln_layer, nn.LayerNorm)
    assert isinstance(inf_ln_layer,  nn.LayerNorm)
    if orig_ln_layer.elementwise_affine:
        features_new = inf_ln_layer.weight.size(0)
        features_old = orig_ln_layer.weight.size(0)
        scale = (features_new // features_old) * (features_old / features_new)
        adjust = 1.0 / (features_new // features_old) if scale_for_bert else 1.0
        with torch.no_grad():
            if inf_ln_layer.weight is not None:
                inf_ln_layer.weight.data = (
                    inflate(orig_ln_layer.weight.data, features_new, dim=0, pattern=pattern)
                    * math.sqrt(scale) * adjust
                ).detach().clone()
            if inf_ln_layer.bias is not None:
                inf_ln_layer.bias.data = (
                    inflate(orig_ln_layer.bias.data, features_new, dim=0, pattern=bias_pattern)
                    * adjust
                ).detach().clone()
        _inflate_optimizer_state_1d(
            orig_ln_layer.weight, inf_ln_layer.weight,
            orig_optimizer, inf_optimizer,
            features_new=features_new, pattern=pattern,
        )
        if orig_ln_layer.bias is not None and inf_ln_layer.bias is not None:
            _inflate_optimizer_state_1d(
                orig_ln_layer.bias, inf_ln_layer.bias,
                orig_optimizer, inf_optimizer,
                features_new=features_new, pattern=bias_pattern,
            )
    inf_ln_layer.eps = orig_ln_layer.eps * scale


def inflate_ln_bert2bert(
    orig_ln_layer, inf_ln_layer,
    pattern='zero', bias_pattern='average',
    scale_for_bert=False, device='cuda',
    orig_optimizer: Optional[torch.optim.Optimizer] = None,
    inf_optimizer: Optional[torch.optim.Optimizer] = None,
):
    assert isinstance(orig_ln_layer, nn.LayerNorm)
    assert isinstance(inf_ln_layer,  nn.LayerNorm)
    if orig_ln_layer.elementwise_affine:
        features_new = inf_ln_layer.weight.size(0)
        features_old = orig_ln_layer.weight.size(0)
        adjust = 1.0 / (features_new // features_old) if scale_for_bert else 1.0
        with torch.no_grad():
            if inf_ln_layer.weight is not None:
                inf_ln_layer.weight.data = (
                    inflate(orig_ln_layer.weight.data, features_new, dim=0, pattern=pattern)
                    * adjust
                ).detach().clone()
            if inf_ln_layer.bias is not None:
                inf_ln_layer.bias.data = (
                    inflate(orig_ln_layer.bias.data, features_new, dim=0, pattern=bias_pattern)
                    * adjust
                ).detach().clone()
        _inflate_optimizer_state_1d(
            orig_ln_layer.weight, inf_ln_layer.weight,
            orig_optimizer, inf_optimizer,
            features_new=features_new, pattern=pattern,
        )
        if orig_ln_layer.bias is not None and inf_ln_layer.bias is not None:
            _inflate_optimizer_state_1d(
                orig_ln_layer.bias, inf_ln_layer.bias,
                orig_optimizer, inf_optimizer,
                features_new=features_new, pattern=bias_pattern,
            )
    inf_ln_layer.eps = orig_ln_layer.eps


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------

def inflate_modBertAttention(
    orig_att, inf_att,
    kqv_heads_pattern='circular',
    kqv_out_pattern='circular',
    kqv_in_pattern='zero',
    proj_out_pattern='average',
    ln_pattern='average',
    ln_bias_pattern='average',
    mode='proj',
    device='cuda',
    out_mode=None,
    AKI_att=None,
    indices=None,
    scalezero=1.0, scalecancel=1.0, scalecirc=1.0,
    circ_mode='projection',
    noise_sigma: float = 0.0,
    orig_optimizer: Optional[torch.optim.Optimizer] = None,
    inf_optimizer: Optional[torch.optim.Optimizer] = None,
):
    assert isinstance(orig_att, modBertAttention)
    assert isinstance(inf_att,  modBertAttention)
    if 'AKI' in mode:
        assert AKI_att is not None
    if AKI_att is not None:
        assert isinstance(AKI_att, modBertAttention)
    assert orig_att.self.attention_head_size == inf_att.self.attention_head_size, \
        'Currently only support inflation with same attention_head_size'

    if out_mode is None:
        out_mode = mode

    head_dim      = orig_att.self.attention_head_size
    heads         = orig_att.self.num_attention_heads
    inf_heads     = inf_att.self.num_attention_heads
    embed_dim     = orig_att.self.all_head_size
    inf_embed_dim = inf_att.self.all_head_size

    W_q = orig_att.self.query.weight.detach().clone()
    W_k = orig_att.self.key.weight.detach().clone()
    W_v = orig_att.self.value.weight.detach().clone()

    W_q_inf = inf_att.self.query.weight.detach().clone()
    W_k_inf = inf_att.self.key.weight.detach().clone()
    W_v_inf = inf_att.self.value.weight.detach().clone()

    if orig_att.self.query.bias is not None:
        reshape_q_bias = orig_att.self.query.bias.view(heads, -1)
        reshape_k_bias = orig_att.self.key.bias.view(heads, -1)
        reshape_v_bias = orig_att.self.value.bias.view(heads, -1)
    else:
        reshape_q_bias = reshape_k_bias = reshape_v_bias = None

    proj_weight = orig_att.output.dense.weight.detach().clone()
    proj_bias   = orig_att.output.dense.bias.detach().clone() \
                  if orig_att.output.dense.bias is not None else None

    reshape_W_q = W_q.view(heads, -1, embed_dim)
    reshape_W_k = W_k.view(heads, -1, embed_dim)
    reshape_W_v = W_v.view(heads, -1, embed_dim)

    if AKI_att is not None:
        AKI_reshape_W_q    = AKI_att.self.query.weight.detach().clone().view(heads, -1, embed_dim)
        AKI_reshape_W_k    = AKI_att.self.key.weight.detach().clone().view(heads, -1, embed_dim)
        AKI_reshape_W_v    = AKI_att.self.value.weight.detach().clone().view(heads, -1, embed_dim)
        AKI_reshape_q_bias = AKI_att.self.query.bias.view(heads, -1) if AKI_att.self.query.bias is not None else None
        AKI_reshape_k_bias = AKI_att.self.key.bias.view(heads, -1)   if AKI_att.self.key.bias   is not None else None
        AKI_reshape_v_bias = AKI_att.self.value.bias.view(heads, -1) if AKI_att.self.value.bias is not None else None
        AKI_proj_weight    = AKI_att.output.dense.weight.detach().clone()
        AKI_proj_bias      = AKI_att.output.dense.bias.detach().clone() if AKI_att.output.dense.bias is not None else None
    else:
        AKI_reshape_W_q = AKI_reshape_W_k = AKI_reshape_W_v = None
        AKI_reshape_q_bias = AKI_reshape_k_bias = AKI_reshape_v_bias = None
        AKI_proj_weight = AKI_proj_bias = None

    reshape_W_q_inf = W_q_inf.view(inf_heads, -1, inf_embed_dim)
    reshape_W_k_inf = W_k_inf.view(inf_heads, -1, inf_embed_dim)
    reshape_W_v_inf = W_v_inf.view(inf_heads, -1, inf_embed_dim)
    proj_weight_inf = inf_att.output.dense.weight.detach().clone()

    proj_in_pattern = kqv_heads_pattern

    _shared = dict(
        mode=mode, device=device,
        indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
    )

    inf_reshaped_W_q, inf_reshape_q_bias = inflate_fc_nonint_heads(
        orig_weight=reshape_W_q, orig_bias=reshape_q_bias,
        new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
        heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,
        inf_weight=reshape_W_q_inf, AKI_weight=AKI_reshape_W_q, AKI_bias=AKI_reshape_q_bias,
        orig_param=orig_att.self.query.weight,
        inf_param=inf_att.self.query.weight,
        **_shared)

    inf_reshaped_W_k, inf_reshape_k_bias = inflate_fc_nonint_heads(
        orig_weight=reshape_W_k, orig_bias=reshape_k_bias,
        new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
        heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,
        inf_weight=reshape_W_k_inf, AKI_weight=AKI_reshape_W_k, AKI_bias=AKI_reshape_k_bias,
        orig_param=orig_att.self.key.weight,
        inf_param=inf_att.self.key.weight,
        **_shared)

    inf_reshaped_W_v, inf_reshape_v_bias = inflate_fc_nonint_heads(
        orig_weight=reshape_W_v, orig_bias=reshape_v_bias,
        new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
        heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,
        inf_weight=reshape_W_v_inf, AKI_weight=AKI_reshape_W_v, AKI_bias=AKI_reshape_v_bias,
        orig_param=orig_att.self.value.weight,
        inf_param=inf_att.self.value.weight,
        **_shared)

    _proj_shared = dict(
        mode=out_mode, device=device,
        indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
    )
    inf_proj_weight, inf_proj_bias = inflate_fc_nonint_heads(
        orig_weight=proj_weight, orig_bias=proj_bias,
        new_heads=1, new_out_channels=inf_embed_dim, new_in_channels=inf_embed_dim,
        heads_pattern='circular', out_pattern=proj_out_pattern, in_pattern=proj_in_pattern,
        inf_weight=proj_weight_inf, AKI_weight=AKI_proj_weight, AKI_bias=AKI_proj_bias,
        orig_param=orig_att.output.dense.weight,
        inf_param=inf_att.output.dense.weight,
        **_proj_shared)

    inf_W_q = inf_reshaped_W_q.view(-1, inf_embed_dim)
    inf_W_k = inf_reshaped_W_k.view(-1, inf_embed_dim)
    inf_W_v = inf_reshaped_W_v.view(-1, inf_embed_dim)

    with torch.no_grad():
        inf_att.self.query.weight.data = inf_W_q.data
        inf_att.self.key.weight.data   = inf_W_k.data
        inf_att.self.value.weight.data = inf_W_v.data
        if inf_att.self.query.bias is not None:
            inf_att.self.query.bias.data = inf_reshape_q_bias.view(-1).data
            inf_att.self.key.bias.data   = inf_reshape_k_bias.view(-1).data
            inf_att.self.value.bias.data = inf_reshape_v_bias.view(-1).data
        inf_att.output.dense.weight.data = inf_proj_weight.data
        if inf_att.output.dense.bias is not None:
            inf_att.output.dense.bias.data = inf_proj_bias.data

    # Transfer bias optimizer states (1-D, inflate along the head*out dim)
    _inf_embed = inf_att.self.all_head_size
    _orig_embed = orig_att.self.all_head_size
    if orig_att.self.query.bias is not None and inf_att.self.query.bias is not None:
        for orig_b, inf_b in [
            (orig_att.self.query.bias,  inf_att.self.query.bias),
            (orig_att.self.key.bias,    inf_att.self.key.bias),
            (orig_att.self.value.bias,  inf_att.self.value.bias),
        ]:
            _inflate_optimizer_state_1d(
                orig_b, inf_b, orig_optimizer, inf_optimizer,
                features_new=inf_b.size(0), pattern=kqv_out_pattern,
            )
    if orig_att.output.dense.bias is not None and inf_att.output.dense.bias is not None:
        _inflate_optimizer_state_1d(
            orig_att.output.dense.bias, inf_att.output.dense.bias,
            orig_optimizer, inf_optimizer,
            features_new=inf_att.output.dense.bias.size(0), pattern=proj_out_pattern,
        )

    if mode in ('AKI', 'net2net'):
        print('LN: bert2bert style')
        inflate_ln_bert2bert(orig_att.LayerNorm, inf_att.LayerNorm,
                             pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device,
                             orig_optimizer=orig_optimizer, inf_optimizer=inf_optimizer)
    else:
        print('LN: educated style')
        inflate_ln(orig_att.LayerNorm, inf_att.LayerNorm,
                   pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device,
                   orig_optimizer=orig_optimizer, inf_optimizer=inf_optimizer)


# ---------------------------------------------------------------------------
# Full transformer layer
# ---------------------------------------------------------------------------

def inflate_modBertLayer(
    orig_layer, inf_layer,
    kqv_heads_pattern, kqv_out_pattern, kqv_in_pattern,
    proj_out_pattern,
    mlp_out_pattern, mlp_hidden_pattern, mlp_in_pattern,
    ln_pattern, ln_bias_pattern,
    mode,
    device='cuda', out_mode=None,
    AKI_layer=None, indices=None,
    scalezero=1.0, scalecancel=1.0, scalecirc=1.0,
    circ_mode='projection', noise_sigma: float = 0.0,
    orig_optimizer: Optional[torch.optim.Optimizer] = None,
    inf_optimizer: Optional[torch.optim.Optimizer] = None,
):
    assert isinstance(orig_layer, modBertLayer)
    assert isinstance(inf_layer,  modBertLayer)
    if 'AKI' in mode:
        assert AKI_layer is not None
    if AKI_layer is not None:
        assert isinstance(AKI_layer, modBertLayer)
        AKI_att        = AKI_layer.attention
        AKI_fc1_weight = AKI_layer.intermediate.dense.weight.detach().clone()
        AKI_fc1_bias   = AKI_layer.intermediate.dense.bias.detach().clone()
        AKI_fc2_weight = AKI_layer.output.dense.weight.detach().clone()
        AKI_fc2_bias   = AKI_layer.output.dense.bias.detach().clone()
    else:
        AKI_att = None
        AKI_fc1_weight = AKI_fc1_bias = AKI_fc2_weight = AKI_fc2_bias = None

    if out_mode is None:
        out_mode = mode

    inflate_modBertAttention(
        orig_layer.attention, inf_layer.attention,
        kqv_heads_pattern=kqv_heads_pattern,
        kqv_out_pattern=kqv_out_pattern,
        kqv_in_pattern=kqv_in_pattern,
        proj_out_pattern=proj_out_pattern,
        ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
        mode=mode, device=device, out_mode=out_mode,
        AKI_att=AKI_att, indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
    )

    assert mlp_hidden_pattern in (
        'circular', 'circularcomp', 'symmetric', 'symmetric_scaled'
    ), 'Unsupported mlp_hidden_pattern: {}'.format(mlp_hidden_pattern)

    _mlp_shared = dict(
        new_heads=1, heads_pattern='circular',
        mode=mode, device=device, indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
    )

    inf_fc1_weight, inf_fc1_bias = inflate_fc_nonint_heads(
        orig_weight=orig_layer.intermediate.dense.weight.detach().clone(),
        orig_bias=orig_layer.intermediate.dense.bias.detach().clone(),
        new_out_channels=inf_layer.intermediate.dense.out_features,
        new_in_channels=inf_layer.intermediate.dense.in_features,
        out_pattern=mlp_hidden_pattern,
        in_pattern=mlp_in_pattern,
        inf_weight=inf_layer.intermediate.dense.weight.detach().clone(),
        AKI_weight=AKI_fc1_weight, AKI_bias=AKI_fc1_bias,
        orig_param=orig_layer.intermediate.dense.weight,
        inf_param=inf_layer.intermediate.dense.weight,
        **_mlp_shared)

    _mlp_fc2_shared = dict(
        new_heads=1, heads_pattern='circular',
        mode=out_mode, device=device, indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
    )
    inf_fc2_weight, inf_fc2_bias = inflate_fc_nonint_heads(
        orig_weight=orig_layer.output.dense.weight.detach().clone(),
        orig_bias=orig_layer.output.dense.bias.detach().clone(),
        new_out_channels=inf_layer.output.dense.out_features,
        new_in_channels=inf_layer.output.dense.in_features,
        out_pattern=mlp_out_pattern,
        in_pattern=mlp_hidden_pattern,
        inf_weight=inf_layer.output.dense.weight.detach().clone(),
        AKI_weight=AKI_fc2_weight, AKI_bias=AKI_fc2_bias,
        orig_param=orig_layer.output.dense.weight,
        inf_param=inf_layer.output.dense.weight,
        **_mlp_fc2_shared)

    with torch.no_grad():
        inf_layer.intermediate.dense.weight.data = inf_fc1_weight
        if inf_layer.intermediate.dense.bias is not None:
            inf_layer.intermediate.dense.bias.data = inf_fc1_bias
        inf_layer.output.dense.weight.data = inf_fc2_weight
        if inf_layer.output.dense.bias is not None:
            inf_layer.output.dense.bias.data = inf_fc2_bias

    # Transfer bias optimizer states (1-D, inflate along out dim)
    if orig_layer.intermediate.dense.bias is not None and inf_layer.intermediate.dense.bias is not None:
        _inflate_optimizer_state_1d(
            orig_layer.intermediate.dense.bias, inf_layer.intermediate.dense.bias,
            orig_optimizer, inf_optimizer,
            features_new=inf_layer.intermediate.dense.bias.size(0),
            pattern=mlp_hidden_pattern,
        )
    if orig_layer.output.dense.bias is not None and inf_layer.output.dense.bias is not None:
        _inflate_optimizer_state_1d(
            orig_layer.output.dense.bias, inf_layer.output.dense.bias,
            orig_optimizer, inf_optimizer,
            features_new=inf_layer.output.dense.bias.size(0),
            pattern=mlp_out_pattern,
        )

    if mode in ('AKI', 'net2net'):
        print('LN (MLP): bert2bert style')
        inflate_ln_bert2bert(
            orig_layer.intermediate.LayerNorm, inf_layer.intermediate.LayerNorm,
            pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device,
            orig_optimizer=orig_optimizer, inf_optimizer=inf_optimizer)
    else:
        print('LN (MLP): educated style')
        inflate_ln(
            orig_layer.intermediate.LayerNorm, inf_layer.intermediate.LayerNorm,
            pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device,
            orig_optimizer=orig_optimizer, inf_optimizer=inf_optimizer)


# ---------------------------------------------------------------------------
# MLM head
# ---------------------------------------------------------------------------

def inflate_modBertLMPredictionHead(orig_layer, inf_layer, decoder_out_pattern):
    assert isinstance(orig_layer, modBertLMPredictionHead)
    assert isinstance(inf_layer,  modBertLMPredictionHead)
    with torch.no_grad():
        if inf_layer.decoder.bias is not None:
            inf_layer.decoder.bias.data = inflate(
                orig_layer.decoder.bias.detach().clone(),
                inf_layer.decoder.bias.size(-1),
                dim=0, pattern=decoder_out_pattern,
            )


# ---------------------------------------------------------------------------
# inflate_LEMON — original LEMON recipe (unchanged semantics)
# ---------------------------------------------------------------------------

def inflate_LEMON(
    orig_bert, inf_bert,
    mode='proj', device='cpu', fc_mode=None,
    inflate_new_layers=True,
    orig_circ_mode='comp', depth_circ_mode='comp',
    orig_scalezero=0.1, orig_scalecirc=0.1,
    depth_scalezero=0.1, depth_scalecirc=0.1,
):
    """Original LEMON-style BERT inflation (circular/average patterns)."""
    assert isinstance(orig_bert, modBertForMaskedLM)
    assert isinstance(inf_bert,  modBertForMaskedLM)

    orig_depth = len(orig_bert.bert.encoder.layer)
    inf_depth  = len(inf_bert.bert.encoder.layer)
    assert orig_depth * 2 >= inf_depth

    embedding_pattern  = 'average'
    ln_pattern         = 'unif'
    ln_bias_pattern    = 'zero'

    kqv_heads_pattern  = 'circular'
    kqv_out_pattern    = 'circular'
    kqv_in_pattern     = 'zero'
    proj_out_pattern   = 'average'

    mlp_out_pattern    = 'average'
    mlp_hidden_pattern = 'circular'
    mlp_in_pattern     = 'zero'

    decoder_out_pattern = 'circular'

    inflate_BertEmbeddings(
        orig_bert.bert.embeddings, inf_bert.bert.embeddings,
        pattern=embedding_pattern, ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
    )

    no_inflation_depth = 2 * orig_depth - inf_depth
    orig_layers = orig_bert.bert.encoder.layer
    inf_layers  = inf_bert.bert.encoder.layer

    for i in range(no_inflation_depth):
        print('LEMON: normal inflate layer {}'.format(i))
        inflate_modBertLayer(
            orig_layers[i], inf_layers[i],
            kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
            kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
            mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
            mlp_in_pattern=mlp_in_pattern,
            ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
            mode=mode, device=device,
            scalezero=orig_scalezero, scalecirc=orig_scalecirc,
            circ_mode=orig_circ_mode,
        )

    for i in range(no_inflation_depth, orig_depth):
        ni = no_inflation_depth + (i - no_inflation_depth) * 2
        zi = ni + 1
        print('LEMON: normal inflate Orig {} -> Inf {}'.format(i, ni))
        inflate_modBertLayer(
            orig_layers[i], inf_layers[ni],
            kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
            kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
            mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
            mlp_in_pattern=mlp_in_pattern,
            ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
            mode=mode, device=device,
            scalezero=orig_scalezero, scalecirc=orig_scalecirc,
            circ_mode=orig_circ_mode,
        )
        if inflate_new_layers:
            aki_layer = orig_layers[i if i == orig_depth - 1 else i + 1]
            print('LEMON: zero inflate Orig {} -> Inf {} (AKI from {})'.format(
                i, zi, i if i == orig_depth - 1 else i + 1))
            inflate_modBertLayer(
                orig_layers[i], inf_layers[zi],
                kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
                kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
                mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
                mlp_in_pattern=mlp_in_pattern,
                ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
                mode='AKIproj', out_mode='allzero', device=device, AKI_layer=aki_layer,
                scalezero=depth_scalezero, scalecirc=depth_scalecirc,
                circ_mode=depth_circ_mode,
            )
        else:
            print('LEMON: skipping depth layer {}'.format(zi))

    inflate_ln(
        orig_bert.bert.encoder.ln_f, inf_bert.bert.encoder.ln_f,
        pattern=ln_pattern, bias_pattern=ln_bias_pattern,
        scale_for_bert=True, device=device,
    )
    inflate_modBertLMPredictionHead(
        orig_bert.cls.predictions, inf_bert.cls.predictions,
        decoder_out_pattern=decoder_out_pattern,
    )


# ---------------------------------------------------------------------------
# inflate_LEMON_educated — symmetric / net2net-style split
# ---------------------------------------------------------------------------

def _snapshot(module):
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def _print_snap(snap, indent='  '):
    pass

def _print_diff(before, after, indent='  '):
    pass


def _section(title, width=80):
    bar = '=' * width
    pad = max(0, (width - len(title) - 2) // 2)
    print('\n{}\n{} {} {}\n{}'.format(bar, '=' * pad, title, '=' * pad, bar))


def _subsection(title, width=80):
    print('\n' + '-' * width)
    print('  ' + title)
    print('-' * width)


def inflate_LEMON_educated(
    orig_bert,
    inf_bert,
    mode: str = 'proj',
    device: str = 'cpu',
    inflate_new_layers: bool = True,
    scalecirc: float = 1.0,
    depth_scalezero: float = 0.1,
    depth_scalecirc: float = 0.1,
    noise_sigma: float = 0.0,
    orig_optimizer: Optional[torch.optim.Optimizer] = None,
    inf_optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Educated BERT inflation using symmetric patterns, with optional in-place
    optimizer state transfer from orig_optimizer -> inf_optimizer.

    The caller should:
      1. Train / step orig_bert with orig_optimizer (at least one step).
      2. Instantiate inf_bert (same architecture but larger).
      3. Construct inf_optimizer over inf_bert.parameters() (not yet stepped).
      4. Call inflate_LEMON_educated(..., orig_optimizer=orig_optimizer,
                                        inf_optimizer=inf_optimizer).

    After this call inf_optimizer will have inflated exp_avg / exp_avg_sq for
    every parameter that was copied from orig_bert, and step=0 for all params
    (fresh momentum start as requested).  Parameters belonging to new depth
    layers (the identity copies) will have zeroed optimizer state.
    """
    assert isinstance(orig_bert, modBertForMaskedLM)
    assert isinstance(inf_bert,  modBertForMaskedLM)

    orig_depth = len(orig_bert.bert.encoder.layer)
    inf_depth  = len(inf_bert.bert.encoder.layer)
    assert orig_depth * 2 >= inf_depth, \
        'inf_depth must be <= 2 * orig_depth'

    # ------------------------------------------------------------------ #
    # Pattern table                                                         #
    # ------------------------------------------------------------------ #
    embedding_pattern  = 'average'
    ln_pattern         = 'unif'
    ln_bias_pattern    = 'zero'

    kqv_heads_pattern  = 'symmetric_scaled'
    kqv_out_pattern    = 'symmetric_scaled'
    kqv_in_pattern     = 'symmetric_scaled'
    proj_out_pattern   = 'symmetric'

    mlp_out_pattern    = 'symmetric'
    mlp_hidden_pattern = 'symmetric'
    mlp_in_pattern     = 'symmetric'

    decoder_out_pattern = 'circular'
    circ_mode = 'projection'

    no_inflation_depth = 2 * orig_depth - inf_depth
    orig_layers = orig_bert.bert.encoder.layer
    inf_layers  = inf_bert.bert.encoder.layer

    # Shared kwargs for normal-copy layer calls
    _layer_kw = dict(
        kqv_heads_pattern=kqv_heads_pattern,
        kqv_out_pattern=kqv_out_pattern,
        kqv_in_pattern=kqv_in_pattern,
        proj_out_pattern=proj_out_pattern,
        mlp_out_pattern=mlp_out_pattern,
        mlp_hidden_pattern=mlp_hidden_pattern,
        mlp_in_pattern=mlp_in_pattern,
        ln_pattern=ln_pattern,
        ln_bias_pattern=ln_bias_pattern,
        mode=mode,
        device=device,
        scalecirc=scalecirc,
        circ_mode=circ_mode,
        noise_sigma=noise_sigma,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
    )

    # ================================================================== #
    # EMBEDDINGS
    # (Embedding optimizer state is not inflated here — embeddings use
    #  'average' pattern which is not covered by _inflate_optimizer_state's
    #  symmetric path.  They receive zeroed state, which is fine since
    #  embeddings are sparse and their moments reset quickly.)
    # ================================================================== #
    _section('EMBEDDINGS')

    snap_orig = _snapshot(orig_bert.bert.embeddings)
    inflate_BertEmbeddings(
        orig_bert.bert.embeddings, inf_bert.bert.embeddings,
        pattern=embedding_pattern,
        ln_pattern=ln_pattern,
        ln_bias_pattern=ln_bias_pattern,
        noise_sigma=noise_sigma,
    )

    # Transfer embedding optimizer states (2-D tables, inflate hidden dim)
    for orig_emb, inf_emb in [
        (orig_bert.bert.embeddings.word_embeddings.weight,
         inf_bert.bert.embeddings.word_embeddings.weight),
        (orig_bert.bert.embeddings.position_embeddings.weight,
         inf_bert.bert.embeddings.position_embeddings.weight),
        (orig_bert.bert.embeddings.token_type_embeddings.weight,
         inf_bert.bert.embeddings.token_type_embeddings.weight),
    ]:
        _inflate_optimizer_state_1d(
            orig_emb, inf_emb, orig_optimizer, inf_optimizer,
            features_new=inf_emb.size(1), pattern=embedding_pattern,
        )
    # Transfer embedding LayerNorm state if present
    if hasattr(orig_bert.bert.embeddings, 'LayerNorm'):
        inflate_ln(
            orig_bert.bert.embeddings.LayerNorm,
            inf_bert.bert.embeddings.LayerNorm,
            pattern=ln_pattern, bias_pattern=ln_bias_pattern,
            orig_optimizer=orig_optimizer, inf_optimizer=inf_optimizer,
        )

    _subsection('orig embeddings')
    _print_snap(snap_orig)
    _subsection('inf  embeddings — after')
    _print_snap(_snapshot(inf_bert.bert.embeddings))

    # ================================================================== #
    # WIDTH-ONLY LAYERS
    # ================================================================== #
    _section('WIDTH-ONLY LAYERS  (orig layers 0..{})'.format(no_inflation_depth - 1))

    for i in range(no_inflation_depth):
        _subsection('Layer {}  orig[{}] -> inf[{}]  (width-only)'.format(i, i, i))

        snap_orig = _snapshot(orig_layers[i])
        inflate_modBertLayer(orig_layers[i], inf_layers[i], **_layer_kw)

        print('  [orig layer {}]'.format(i))
        _print_snap(snap_orig)
        print('  [inf  layer {} — after]'.format(i))
        _print_snap(_snapshot(inf_layers[i]))

    # ================================================================== #
    # WIDTH + DEPTH LAYERS
    # ================================================================== #
    _section('WIDTH + DEPTH LAYERS  (orig layers {}..{})'.format(
        no_inflation_depth, orig_depth - 1))

    for i in range(no_inflation_depth, orig_depth):
        ni = no_inflation_depth + (i - no_inflation_depth) * 2
        zi = ni + 1

        # Normal (function-preserving) copy — state transferred
        _subsection('Layer {}  orig[{}] -> inf[{}]  (normal copy)'.format(i, i, ni))

        snap_orig = _snapshot(orig_layers[i])
        inflate_modBertLayer(orig_layers[i], inf_layers[ni], **_layer_kw)

        print('  [orig layer {}]'.format(i))
        _print_snap(snap_orig)
        print('  [inf  layer {} — after]'.format(ni))
        _print_snap(_snapshot(inf_layers[ni]))

        # Identity copy (depth expansion) — no orig state to transfer,
        # so we pass orig_optimizer=None to leave inf state zeroed.
        if inflate_new_layers:
            aki_src = i if i == orig_depth - 1 else i + 1
            aki_layer = orig_layers[aki_src]

            _subsection('Layer {}  orig[{}] -> inf[{}]  (identity copy, AKI from orig[{}])'.format(
                i, i, zi, aki_src))

            inflate_modBertLayer(
                orig_layers[i], inf_layers[zi],
                kqv_heads_pattern=kqv_heads_pattern,
                kqv_out_pattern=kqv_out_pattern,
                kqv_in_pattern=kqv_in_pattern,
                proj_out_pattern=proj_out_pattern,
                mlp_out_pattern=mlp_out_pattern,
                mlp_hidden_pattern=mlp_hidden_pattern,
                mlp_in_pattern=mlp_in_pattern,
                ln_pattern=ln_pattern,
                ln_bias_pattern=ln_bias_pattern,
                mode='AKIproj',
                out_mode='cancelzero',
                device=device,
                AKI_layer=aki_layer,
                scalezero=depth_scalezero,
                scalecirc=depth_scalecirc,
                circ_mode=circ_mode,
                noise_sigma=noise_sigma,
                orig_optimizer=None,       # new depth layer — leave state zeroed
                inf_optimizer=inf_optimizer,
            )

            print('  [orig layer {}]'.format(i))
            _print_snap(snap_orig)
            print('  [inf  layer {} — after]'.format(zi))
            _print_snap(_snapshot(inf_layers[zi]))

        else:
            _subsection('Layer {}  inf[{}]  (depth layer — skipped)'.format(i, zi))

    # ================================================================== #
    # FINAL ENCODER LAYERNORM
    # ================================================================== #
    _section('FINAL ENCODER LAYERNORM')

    snap_orig = _snapshot(orig_bert.bert.encoder.ln_f)
    inflate_ln(
        orig_bert.bert.encoder.ln_f,
        inf_bert.bert.encoder.ln_f,
        pattern=ln_pattern,
        bias_pattern=ln_bias_pattern,
        scale_for_bert=True,
        device=device,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
    )

    _subsection('orig ln_f')
    _print_snap(snap_orig)
    _subsection('inf  ln_f — after')
    _print_snap(_snapshot(inf_bert.bert.encoder.ln_f))

    # ================================================================== #
    # MLM DECODER HEAD
    # ================================================================== #
    _section('MLM DECODER HEAD')

    snap_orig = _snapshot(orig_bert.cls.predictions)
    inflate_modBertLMPredictionHead(
        orig_bert.cls.predictions,
        inf_bert.cls.predictions,
        decoder_out_pattern=decoder_out_pattern,
    )

    # predictions.bias is vocab-sized and unchanged — copy state directly
    if orig_optimizer is not None and inf_optimizer is not None:
        orig_bias_p = orig_bert.cls.predictions.bias
        inf_bias_p  = inf_bert.cls.predictions.bias
        if orig_bias_p is not None and inf_bias_p is not None \
                and orig_bias_p in orig_optimizer.state:
            orig_s = orig_optimizer.state[orig_bias_p]
            if inf_bias_p not in inf_optimizer.state:
                inf_optimizer.state[inf_bias_p] = {}
            inf_s = inf_optimizer.state[inf_bias_p]
            for key in ("exp_avg", "exp_avg_sq"):
                if key in orig_s:
                    inf_s[key] = orig_s[key].detach().clone().to(inf_bias_p.device)
            inf_s["step"] = torch.zeros_like(orig_s["step"]) if "step" in orig_s \
                else torch.tensor(0, dtype=torch.long)

    _subsection('orig predictions')
    _print_snap(snap_orig)
    _subsection('inf  predictions — after')
    _print_snap(_snapshot(inf_bert.cls.predictions))

    _section('INFLATION COMPLETE')


def inflate_AKI(
    orig_bert, inf_bert,
    device='cpu',
    inflate_new_layers=True,
    out_mode='allzero',
    ln_pattern='unif',
    ln_bias_pattern='zero',
    kqv_heads_pattern='circular',
    kqv_out_pattern='circular',
    kqv_in_pattern='zero',
    proj_out_pattern='average',
    mlp_out_pattern='average',
    mlp_hidden_pattern='circular',
    mlp_in_pattern='zero',
    decoder_out_pattern='circular',
    scale=0.1,
):
    """
    AKI-style BERT inflation:
    - normal mapped layers use mode='AKIproj'
    - inserted depth layers also use mode='AKIproj'
    """
    assert isinstance(orig_bert, modBertForMaskedLM)
    assert isinstance(inf_bert,  modBertForMaskedLM)

    orig_depth = len(orig_bert.bert.encoder.layer)
    inf_depth  = len(inf_bert.bert.encoder.layer)
    assert orig_depth * 2 >= inf_depth

    # embeddings
    inflate_BertEmbeddings(
        orig_bert.bert.embeddings, inf_bert.bert.embeddings,
        pattern='average',
        ln_pattern=ln_pattern,
        ln_bias_pattern=ln_bias_pattern,
    )

    no_inflation_depth = 2 * orig_depth - inf_depth
    orig_layers = orig_bert.bert.encoder.layer
    inf_layers  = inf_bert.bert.encoder.layer

    # Part 1: layers that map 1:1
    for i in range(no_inflation_depth):
        print(f'AKI: inflate layer {i}')
        inflate_modBertLayer(
            orig_layers[i], inf_layers[i],
            kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
            kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
            mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
            mlp_in_pattern=mlp_in_pattern,
            ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
            mode='AKIproj', out_mode=out_mode, device=device,
            AKI_layer=orig_layers[i],  # AKI reference
            scalezero=scale, scalecirc=scale,
            circ_mode='comp',
        )

    # Part 2: layers that map to (normal, inserted)
    for i in range(no_inflation_depth, orig_depth):
        ni = no_inflation_depth + (i - no_inflation_depth) * 2
        zi = ni + 1

        # normal mapped layer
        print(f'AKI: Orig {i} -> Inf {ni}')
        inflate_modBertLayer(
            orig_layers[i], inf_layers[ni],
            kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
            kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
            mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
            mlp_in_pattern=mlp_in_pattern,
            ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
            mode='AKIproj', out_mode=out_mode, device=device,
            AKI_layer=orig_layers[i],
            scalezero=scale, scalecirc=scale,
            circ_mode='comp',
        )

        # inserted layer
        if inflate_new_layers:
            aki_layer = orig_layers[i if i == orig_depth - 1 else i + 1]
            print(f'AKI: Orig {i} -> Inf {zi} (AKI from {i if i == orig_depth - 1 else i + 1})')
            inflate_modBertLayer(
                orig_layers[i], inf_layers[zi],
                kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
                kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
                mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
                mlp_in_pattern=mlp_in_pattern,
                ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
                mode='AKIproj', out_mode=out_mode, device=device,
                AKI_layer=aki_layer,
                scalezero=scale, scalecirc=scale,
                circ_mode='comp',
            )
        else:
            print(f'AKI: skipping inserted layer {zi}')

    inflate_ln(
        orig_bert.bert.encoder.ln_f, inf_bert.bert.encoder.ln_f,
        pattern=ln_pattern, bias_pattern=ln_bias_pattern,
        scale_for_bert=True, device=device,
    )
    inflate_modBertLMPredictionHead(
        orig_bert.cls.predictions, inf_bert.cls.predictions,
        decoder_out_pattern=decoder_out_pattern,
    )


def inflate_FPI(orig_bert, inf_bert, mode='proj', device='cpu', inflate_new_layers=True,
                orig_circ_mode='comp', depth_circ_mode='comp',
                orig_scalezero=0.1, orig_scalecirc=0.1,
                depth_scalezero=0.1, depth_scalecirc=0.1):
    """
    BERT depth/width expansion without bert2BERT-AKI neighbor weights (FPI-style).
    Mapped layers use LEMON projection (``mode``, default ``proj``). Inserted depth
    layers use the same projection for Q/K/V and MLP fc1, and ``allzero`` for attention
    output and fc2 — analogous in spirit to ResNet ``inflate_bottleneck_fpi`` (single
    source, no donor layer).
    """
    assert isinstance(orig_bert, modBertForMaskedLM)
    assert isinstance(inf_bert, modBertForMaskedLM)
    orig_depth = len(orig_bert.bert.encoder.layer)
    inf_depth = len(inf_bert.bert.encoder.layer)
    assert orig_depth * 2 >= inf_depth
    embedding_pattern = 'average'
    ln_pattern = 'unif'
    ln_bias_pattern = 'zero'
    kqv_heads_pattern = 'circular'
    kqv_out_pattern = 'circular'
    kqv_in_pattern = 'zero'
    proj_out_pattern = 'average'
    mlp_out_pattern = 'average'
    mlp_hidden_pattern = 'circular'
    mlp_in_pattern = 'zero'
    decoder_out_pattern = 'circular'
    inflate_BertEmbeddings(
        orig_bert.bert.embeddings, inf_bert.bert.embeddings,
        pattern=embedding_pattern, ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
    )
    no_inflation_depth = 2 * orig_depth - inf_depth
    orig_layers = orig_bert.bert.encoder.layer
    inf_layers = inf_bert.bert.encoder.layer
    for i in range(no_inflation_depth):
        print('FPI: normal inflate layer {}'.format(i))
        inflate_modBertLayer(
            orig_layers[i], inf_layers[i],
            kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
            kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
            mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
            mlp_in_pattern=mlp_in_pattern,
            ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
            mode=mode, device=device,
            scalezero=orig_scalezero, scalecirc=orig_scalecirc,
            circ_mode=orig_circ_mode,
        )
    for i in range(no_inflation_depth, orig_depth):
        normal_layer_index = no_inflation_depth + (i - no_inflation_depth) * 2
        zero_layer_index = normal_layer_index + 1
        print('FPI: normal inflate from Orig {} to Inf {}'.format(i, normal_layer_index))
        inflate_modBertLayer(
            orig_layers[i], inf_layers[normal_layer_index],
            kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
            kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
            mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
            mlp_in_pattern=mlp_in_pattern,
            ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
            mode=mode, device=device,
            scalezero=orig_scalezero, scalecirc=orig_scalecirc,
            circ_mode=orig_circ_mode,
        )
        if inflate_new_layers:
            print('FPI: inserted layer from Orig {} to Inf {} (no AKI donor)'.format(
                i, zero_layer_index))
            inflate_modBertLayer(
                orig_layers[i], inf_layers[zero_layer_index],
                kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern,
                kqv_in_pattern=kqv_in_pattern, proj_out_pattern=proj_out_pattern,
                mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern,
                mlp_in_pattern=mlp_in_pattern,
                ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
                mode=mode, device=device, out_mode='allzero',
                scalezero=depth_scalezero, scalecirc=depth_scalecirc,
                circ_mode=depth_circ_mode,
            )
        else:
            print('FPI: skip Inf layer {}'.format(zero_layer_index))
    inflate_ln(
        orig_bert.bert.encoder.ln_f, inf_bert.bert.encoder.ln_f,
        pattern=ln_pattern, bias_pattern=ln_bias_pattern, scale_for_bert=True, device=device,
    )
    inflate_modBertLMPredictionHead(
        orig_bert.cls.predictions, inf_bert.cls.predictions,
        decoder_out_pattern=decoder_out_pattern,
    )
# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_random_input(vocab_size: int, seq_len: int = 16, batch_size: int = 1):
    return {
        "input_ids":      torch.randint(1, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
    }


def _load_configs(config_path: str):
    import json
    from transformers import BertConfig
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    try:
        return BertConfig(**raw["source"]), BertConfig(**raw["destination"])
    except KeyError as e:
        raise ValueError(f"{config_path} must have 'source' and 'destination' keys (missing {e})") from e


def _get_args_parser(add_help=True):
    import argparse
    p = argparse.ArgumentParser(description="BERT educated inflation test", add_help=add_help)
    p.add_argument("--growth-config", default="./config.json", type=str)
    p.add_argument("--seq-len", default=16, type=int)
    p.add_argument("--seed", default=None, type=int)
    return p


def _main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    modconfig, inf_modconfig = _load_configs(os.path.abspath(args.growth_config))
    print("source config:\n", modconfig)
    print("destination config:\n", inf_modconfig)

    # ------------------------------------------------------------------ #
    # Build small model + optimizer
    # ------------------------------------------------------------------ #
    modmodel = modBertForMaskedLM(modconfig)
    modmodel.train()

    orig_optimizer = torch.optim.AdamW(modmodel.parameters(), lr=1e-4)

    rand_input = _make_random_input(modconfig.vocab_size, seq_len=args.seq_len)
    print("\n=== Random input token ids ===")
    print(rand_input["input_ids"])

    # ------------------------------------------------------------------ #
    # Step 1: one forward + backward + optimizer step on small model
    # ------------------------------------------------------------------ #
    _section('SMALL MODEL — forward / backward / step')

    out_before_step = modmodel(**rand_input)
    loss = out_before_step["logits"].sum()   # dummy scalar loss
    loss.backward()
    orig_optimizer.step()
    orig_optimizer.zero_grad()

    modmodel.eval()
    with torch.no_grad():
        small_out = modmodel(**rand_input)

    small_logits = small_out["logits"]
    print("small model logits sum (post-step): {}".format(small_logits.sum().item()))

    # ------------------------------------------------------------------ #
    # Step 2: build large model + fresh optimizer
    # ------------------------------------------------------------------ #
    _section('BUILDING LARGE MODEL + FRESH OPTIMIZER')

    inf_modmodel = modBertForMaskedLM(inf_modconfig)
    inf_optimizer = torch.optim.AdamW(inf_modmodel.parameters(), lr=1e-4)

    # ------------------------------------------------------------------ #
    # Step 3: inflate weights + transfer optimizer state
    # ------------------------------------------------------------------ #
    _section('INFLATING WEIGHTS + TRANSFERRING OPTIMIZER STATE')

    inflate_LEMON_educated(
        modmodel,
        inf_modmodel,
        orig_optimizer=orig_optimizer,
        inf_optimizer=inf_optimizer,
    )

    # ------------------------------------------------------------------ #
    # Step 4: forward pass on large model and compare
    # ------------------------------------------------------------------ #
    _section('LARGE MODEL — forward pass comparison')

    inf_modmodel.eval()
    with torch.no_grad():
        large_out = inf_modmodel(**rand_input)

    large_logits = large_out["logits"]

    print("\n  small logits shape : {}".format(list(small_logits.shape)))
    print("  large logits shape : {}".format(list(large_logits.shape)))
    print("\n  small logits sum   : {:.6f}".format(small_logits.sum().item()))
    print("  large logits sum   : {:.6f}".format(large_logits.sum().item()))

    diff = (large_logits - small_logits).abs().max().item()
    print("\n  max |large - small| (function-preserving error) : {:.3e}".format(diff))

    # ------------------------------------------------------------------ #
    # Step 5: sanity-check optimizer state was transferred
    # ------------------------------------------------------------------ #
    _section('OPTIMIZER STATE SANITY CHECK')

    orig_params = {n: p for n, p in modmodel.named_parameters()}
    inf_params  = {n: p for n, p in inf_modmodel.named_parameters()}

    checked = 0
    for name, orig_p in orig_params.items():
        if orig_p not in orig_optimizer.state:
            continue
        if name not in inf_params:
            continue
        inf_p = inf_params[name]
        if inf_p not in inf_optimizer.state:
            print("  [MISSING] inf optimizer state for {}".format(name))
            continue

        orig_s = orig_optimizer.state[orig_p]
        inf_s  = inf_optimizer.state[inf_p]

        for key in ("exp_avg", "exp_avg_sq"):
            if key not in orig_s:
                continue
            orig_t = orig_s[key]
            inf_t  = inf_s[key]
            print("  [{}] {}  orig shape={}  inf shape={}  inf norm={:.4f}".format(
                key, name,
                list(orig_t.shape), list(inf_t.shape),
                inf_t.norm().item(),
            ))
        checked += 1

    print("\n  Checked {} parameter(s) with optimizer state.".format(checked))
    print("=" * 80)


if __name__ == "__main__":
    import os
    _main(_get_args_parser().parse_args())