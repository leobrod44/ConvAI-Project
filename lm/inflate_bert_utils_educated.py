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
    """
    For each consecutive pair of input blocks of size ``in_features_old``
    produced by the symmetric repeat+scale, inject correlated +-e noise so
    that block b and block b+1 satisfy:

        w_block_b   = w_old / D - e
        w_block_b+1 = w_old / D + e        (sum over all D blocks = w_old)

    This matches the net2net-style identity:
        v_new = w_old/2 - noise  +  w_old/2 + noise

    ``apply_educated_noise`` is called with the mapping that pairs each column
    in block b with the corresponding column in block b+1, keeping the
    function-preserving constraint intact.
    """
    h, o, i_new = weight.shape
    if in_divisor < 2 or i_new != in_divisor * in_features_old:
        return weight
    w2 = weight.reshape(h * o, i_new)
    std_scale = float(w2.std().item()) if w2.numel() > 0 else 0.0
    if std_scale == 0.0:
        return weight
    # For symmetric_scaled the weights have been multiplied by `scale`,
    # so we reference noise relative to that magnitude.
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
            # For symmetric patterns the residue slice is handled identically
            # to circular; scaling is applied in project_linear, not here.
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

    SYMMETRIC PATH (in_pattern in {'symmetric', 'symmetric_scaled'}):
    -----------------------------------------------------------------
    Implements the net2net-style split:

        w_block_k = w_old / D  +/-  e

    where D = in_new // in_old, and the +/-e noise is injected by
    _apply_symmetric_in_split_noise.  This path is completely independent
    of circ_mode and cancel, and requires in_new to be an exact multiple
    of in_old (no residue).

    All other paths (circular / average / zero) behave exactly as before.
    """
    head_old, out_old, in_old = weight.shape
    head_new, out_new, in_new = weight_.shape

    if head_old > head_new or out_old > out_new or in_old > in_new:
        raise ValueError('The expanded model is smaller than the original model.')

    in_divisor = in_new // in_old
    in_residue = in_new % in_old

    # Normalise pattern aliases
    if 'circular' in out_pattern:
        out_pattern = 'circular'
    if 'circular' in head_pattern:
        head_pattern = 'circular'

    # Validation
    # Note: cancel=True with symmetric in_pattern is valid — the zeroed orig_weight
    # passes through the symmetric path and produces cancelling ±e pairs naturally.
    assert out_pattern in ('circular', 'average', 'zero'), \
        'Unsupported out_pattern: {}'.format(out_pattern)
    assert in_pattern in ('circular', 'average', 'zero', 'symmetric', 'symmetric_scaled'), \
        'Unsupported in_pattern: {}'.format(in_pattern)
    assert circ_mode in ('comp', 'projection'), \
        'Unsupported circ_mode: {}'.format(circ_mode)

    # --- Step 1: expand out and head dims ---
    weight = inflate(weight, out_new, dim=1, pattern=out_pattern)
    weight = inflate(weight, head_new, dim=0, pattern=head_pattern)
    # weight is now [head_new, out_new, in_old]

    # ==========================================================================
    # SYMMETRIC PATH — completely independent of circ_mode / cancel
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

        # Replicate along the in-dim: [head_new, out_new, D, in_old]
        w = weight.unsqueeze(2).repeat(1, 1, in_divisor, 1)

        # Scale so the sum over the D blocks equals the original weight:
        #   symmetric        -> each block = w_old / D
        #   symmetric_scaled -> each block = w_old * scalecirc / D
        if in_pattern == 'symmetric':
            w = w / in_divisor
        else:  # symmetric_scaled
            w = w * (scalecirc / in_divisor)

        final_weight = w.reshape(head_new, out_new, in_new)

        # Inject +/-e noise between consecutive blocks (net2net identity preserved
        # because noise cancels across the pair of blocks)
        if noise_sigma > 0.0:
            final_weight = _apply_symmetric_in_split_noise(
                final_weight,
                in_features_old=in_old,
                in_divisor=in_divisor,
                noise_sigma=noise_sigma,
                scale=scalecirc,
                in_pattern=in_pattern,
            )

        # --- Step 2: expand bias ---
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

    else:  # in_residue > 0
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

        else:  # average / zero
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

    # --- Step 2: expand bias ---
    if bias is not None:
        bias = inflate(bias, out_new, dim=1, pattern=out_pattern)
        bias = inflate(bias, head_new, dim=0, pattern=head_pattern)

    return final_weight, bias


# ---------------------------------------------------------------------------
# inflate_fc_nonint_heads — dispatch wrapper
# ---------------------------------------------------------------------------

def inflate_fc_nonint_heads(
    orig_weight, orig_bias,
    new_heads, new_out_channels, new_in_channels,
    heads_pattern, out_pattern, in_pattern,
    mode, device='cuda', inf_weight=None,
    AKI_weight=None, AKI_bias=None, indices=None,
    scalezero=1.0, scalecancel=1.0, scalecirc=1.0,
    circ_mode='projection', noise_sigma: float = 0.0,
):
    """
    Function-preserving expansion of a fully-connected or multi-head-attention
    weight tensor.  in_pattern may now be 'symmetric' or 'symmetric_scaled'
    in addition to the original options.
    """
    flag_with_head = True
    if len(orig_weight.size()) == 2:
        assert new_heads == 1, 'weight only has 2 dims; heads expansion not supported'
        flag_with_head = False
        orig_weight = orig_weight.unsqueeze(0)
        orig_bias   = orig_bias.unsqueeze(0)
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
            orig_bias   = torch.cat([orig_bias,   AKI_bias[:delta].detach().clone()],   dim=0)
        else:
            delta = new_out_channels - out_channels
            orig_weight = torch.cat([orig_weight, AKI_weight[:, :delta].detach().clone()], dim=1)
            orig_bias   = torch.cat([orig_bias,   AKI_bias[:, :delta].detach().clone()],   dim=1)

    elif mode == 'cancelzero':
        print('Use cancelzero')
        # Zero out the small model weights so the symmetric/circular fan-out
        # expansion produces cancelling pairs (0/D ± e = ±e that sum to 0).
        # No new parameters are created — the structure is inherited from orig_weight.
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
    """
    Inject +-e noise into a symmetrically-duplicated embedding matrix.

    After inflate(..., pattern='symmetric'/'average') an embedding table of
    shape [N, D*d_old] has identical column blocks of size d_old.  This adds
    correlated noise between consecutive blocks so that:

        block[b]   -= eps
        block[b+1] += eps

    preserving the column-sum across all D blocks (function-preserving).
    """
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
    inf_ln_layer.eps = orig_ln_layer.eps * scale


def inflate_ln_bert2bert(
    orig_ln_layer, inf_ln_layer,
    pattern='zero', bias_pattern='average',
    scale_for_bert=False, device='cuda',
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

    # proj layer in-pattern must match how heads were expanded
    proj_in_pattern = kqv_heads_pattern

    _shared = dict(
        mode=mode, device=device,
        indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
    )

    inf_reshaped_W_q, inf_reshape_q_bias = inflate_fc_nonint_heads(
        orig_weight=reshape_W_q, orig_bias=reshape_q_bias,
        new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
        heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,
        inf_weight=reshape_W_q_inf, AKI_weight=AKI_reshape_W_q, AKI_bias=AKI_reshape_q_bias,
        **_shared)

    inf_reshaped_W_k, inf_reshape_k_bias = inflate_fc_nonint_heads(
        orig_weight=reshape_W_k, orig_bias=reshape_k_bias,
        new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
        heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,
        inf_weight=reshape_W_k_inf, AKI_weight=AKI_reshape_W_k, AKI_bias=AKI_reshape_k_bias,
        **_shared)

    inf_reshaped_W_v, inf_reshape_v_bias = inflate_fc_nonint_heads(
        orig_weight=reshape_W_v, orig_bias=reshape_v_bias,
        new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
        heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,
        inf_weight=reshape_W_v_inf, AKI_weight=AKI_reshape_W_v, AKI_bias=AKI_reshape_v_bias,
        **_shared)

    _proj_shared = dict(
        mode=out_mode, device=device,
        indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
    )
    inf_proj_weight, inf_proj_bias = inflate_fc_nonint_heads(
        orig_weight=proj_weight, orig_bias=proj_bias,
        new_heads=1, new_out_channels=inf_embed_dim, new_in_channels=inf_embed_dim,
        heads_pattern='circular', out_pattern=proj_out_pattern, in_pattern=proj_in_pattern,
        inf_weight=proj_weight_inf, AKI_weight=AKI_proj_weight, AKI_bias=AKI_proj_bias,
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

    if mode in ('AKI', 'net2net'):
        print('LN: bert2bert style')
        inflate_ln_bert2bert(orig_att.LayerNorm, inf_att.LayerNorm,
                             pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)
    else:
        print('LN: educated style')
        inflate_ln(orig_att.LayerNorm, inf_att.LayerNorm,
                   pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)


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
    )

    assert mlp_hidden_pattern in (
        'circular', 'circularcomp', 'symmetric', 'symmetric_scaled'
    ), 'Unsupported mlp_hidden_pattern: {}'.format(mlp_hidden_pattern)

    _mlp_shared = dict(
        new_heads=1, heads_pattern='circular',
        mode=mode, device=device, indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
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
        **_mlp_shared)

    # fc2 uses out_mode so depth-expansion layers get allzero output
    _mlp_fc2_shared = dict(
        new_heads=1, heads_pattern='circular',
        mode=out_mode, device=device, indices=indices,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        circ_mode=circ_mode, noise_sigma=noise_sigma,
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
        **_mlp_fc2_shared)

    with torch.no_grad():
        inf_layer.intermediate.dense.weight.data = inf_fc1_weight
        if inf_layer.intermediate.dense.bias is not None:
            inf_layer.intermediate.dense.bias.data = inf_fc1_bias
        inf_layer.output.dense.weight.data = inf_fc2_weight
        if inf_layer.output.dense.bias is not None:
            inf_layer.output.dense.bias.data = inf_fc2_bias

    if mode in ('AKI', 'net2net'):
        print('LN (MLP): bert2bert style')
        inflate_ln_bert2bert(
            orig_layer.intermediate.LayerNorm, inf_layer.intermediate.LayerNorm,
            pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)
    else:
        print('LN (MLP): educated style')
        inflate_ln(
            orig_layer.intermediate.LayerNorm, inf_layer.intermediate.LayerNorm,
            pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)


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
    """Return {param_name: tensor.detach().clone()} for every parameter in module."""
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def _print_snap(snap, indent='  '):
    """Print full tensor data for every parameter in a snapshot dict."""
    pass


def _print_diff(before, after, indent='  '):
    """Print element-wise diff (after - before) for every parameter."""
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
    optimizer=None,
):
    """
    Educated BERT inflation using symmetric patterns, mirroring inflate_vit.

    Every weight block is split as:
        w_block_k = w_old / D  +/-  e        (D = expansion factor)

    so the sum over blocks equals w_old (function-preserving) and the +/-e
    noise breaks the exact symmetry to allow gradient-driven differentiation
    after growth.  This is the net2net-style identity: v_new = w_old/2 - e + w_old/2 + e.

    Pattern table (mirrors inflate_vit exactly):
        kqv_heads / kqv_out / kqv_in   ->  'symmetric_scaled'
        proj_out                         ->  'symmetric'
        mlp_out / mlp_hidden / mlp_in   ->  'symmetric'
        ln weight                        ->  'unif'
        ln bias                          ->  'zero'
        embeddings                       ->  'average'

    Args:
        scalecirc:        Scale for symmetric_scaled blocks.
                          1.0 = exact net2net split; <1 shrinks initial magnitude.
        noise_sigma:      Noise e between duplicate blocks.
                          0.0 = exact function-preserving initialisation.
        inflate_new_layers: Whether to initialise depth-doubled identity layers.
        optimizer:        Optional AdamW optimizer already constructed with
                          inf_bert.parameters().  If provided, its state is
                          inflated in-place via grow_optimizer_state_educated
                          immediately after weight inflation completes.
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
    ln_bias_pattern    = 'zero'      # always zero — expected by Pre-LN

    kqv_heads_pattern  = 'symmetric_scaled'
    kqv_out_pattern    = 'symmetric_scaled'
    kqv_in_pattern     = 'symmetric_scaled'
    proj_out_pattern   = 'symmetric'

    mlp_out_pattern    = 'symmetric'
    mlp_hidden_pattern = 'symmetric'
    mlp_in_pattern     = 'symmetric'

    decoder_out_pattern = 'circular'   # vocab dim unchanged

    # circ_mode is irrelevant for the symmetric path but must be valid
    circ_mode = 'projection'

    no_inflation_depth = 2 * orig_depth - inf_depth
    orig_layers = orig_bert.bert.encoder.layer
    inf_layers  = inf_bert.bert.encoder.layer

    # Shared kwargs for normal-copy calls
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
    )

    # ================================================================== #
    # ================================================================== #
    # EMBEDDINGS                                                           #
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

    _subsection('orig embeddings')
    #_print_snap(snap_orig)
    _subsection('inf  embeddings — after')
   # _print_snap(_snapshot(inf_bert.bert.embeddings))

    # ================================================================== #
    # WIDTH-ONLY LAYERS                                                    #
    # ================================================================== #
    _section('WIDTH-ONLY LAYERS  (orig layers 0..{})'.format(no_inflation_depth - 1))

    for i in range(no_inflation_depth):
        _subsection('Layer {}  orig[{}] -> inf[{}]  (width-only)'.format(i, i, i))

        snap_orig = _snapshot(orig_layers[i])
        inflate_modBertLayer(orig_layers[i], inf_layers[i], **_layer_kw)

        # print('  [orig layer {}]'.format(i))
        # _print_snap(snap_orig)
        # print('  [inf  layer {} — after]'.format(i))
        # _print_snap(_snapshot(inf_layers[i]))

    # ================================================================== #
    # WIDTH + DEPTH LAYERS                                                 #
    # ================================================================== #
    _section('WIDTH + DEPTH LAYERS  (orig layers {}..{})'.format(
        no_inflation_depth, orig_depth - 1))

    for i in range(no_inflation_depth, orig_depth):
        ni = no_inflation_depth + (i - no_inflation_depth) * 2   # normal copy
        zi = ni + 1                                               # identity copy

        # -------------------------------------------------------------- #
        # Normal (function-preserving) copy                               #
        # -------------------------------------------------------------- #
        _subsection('Layer {}  orig[{}] -> inf[{}]  (normal copy)'.format(i, i, ni))

        snap_orig = _snapshot(orig_layers[i])
        inflate_modBertLayer(orig_layers[i], inf_layers[ni], **_layer_kw)

        # print('  [orig layer {}]'.format(i))
        # _print_snap(snap_orig)
        # print('  [inf  layer {} — after]'.format(ni))
        # _print_snap(_snapshot(inf_layers[ni]))

        # -------------------------------------------------------------- #
        # Identity copy (depth expansion)                                 #
        # -------------------------------------------------------------- #
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
            )

            # print('  [orig layer {}]'.format(i))
            # _print_snap(snap_orig)   # reuse orig snap from normal copy above
            # print('  [inf  layer {} — after]'.format(zi))
            # _print_snap(_snapshot(inf_layers[zi]))

        else:
            _subsection('Layer {}  inf[{}]  (depth layer — skipped)'.format(i, zi))

    # ================================================================== #
    # FINAL ENCODER LAYERNORM                                             #
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
    )

    # _subsection('orig ln_f')
    # _print_snap(snap_orig)
    # _subsection('inf  ln_f — after')
    # _print_snap(_snapshot(inf_bert.bert.encoder.ln_f))

    # ================================================================== #
    # MLM DECODER HEAD                                                    #
    # ================================================================== #
    _section('MLM DECODER HEAD')

    snap_orig = _snapshot(orig_bert.cls.predictions)
    inflate_modBertLMPredictionHead(
        orig_bert.cls.predictions,
        inf_bert.cls.predictions,
        decoder_out_pattern=decoder_out_pattern,
    )

    # _subsection('orig predictions')
    # _print_snap(snap_orig)
    # _subsection('inf  predictions — after')
    # _print_snap(_snapshot(inf_bert.cls.predictions))

    _section('INFLATION COMPLETE')

