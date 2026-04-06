#!/usr/bin/env python3

from __future__ import annotations

import numpy as np


# LLTF valid subcarriers in this project are ordered as:
#   +6..+31, -31..-6  (52 bins total)
#
# HT-LTF valid subcarriers are ordered as:
#   +2..+58, -58..-2  (114 bins total)
#
# Therefore, the physically overlapping 20 MHz common region is:
#   +6..+31, -31..-6
# and its indices inside the extracted HT-LTF vector are:
#   +6..+31  -> positions 4..29
#   -31..-6  -> positions 84..109
LLTF_COMMON_INDICES = np.arange(52, dtype=np.int64)
HTLTF_COMMON_INDICES = np.concatenate(
    [np.arange(4, 30, dtype=np.int64), np.arange(84, 110, dtype=np.int64)]
)


def compute_lltf_htltf_scale_factor(
    h_l: np.ndarray,
    h_ht: np.ndarray,
    *,
    eps: float = 1e-8,
 ) -> float:
    """Compute a robust scalar amplitude ratio between HT-LTF and LLTF.

    The factor is the median of the element-wise magnitude ratio over the
    physically overlapping 52-bin region.
    """

    h_l = np.asarray(h_l)
    h_ht = np.asarray(h_ht)

    if h_l.shape != (52,):
        raise ValueError(f"h_l must have shape (52,), got {h_l.shape}")
    if h_ht.shape != (114,):
        raise ValueError(f"h_ht must have shape (114,), got {h_ht.shape}")

    amp_l = np.abs(h_l).astype(np.float32, copy=False)
    amp_ht = np.abs(h_ht).astype(np.float32, copy=False)

    common_l = amp_l[LLTF_COMMON_INDICES]
    common_ht = amp_ht[HTLTF_COMMON_INDICES]

    ratio = np.divide(
        common_ht,
        common_l,
        out=np.full(common_ht.shape, np.nan, dtype=np.float32),
        where=common_l > eps,
    )
    valid_ratio = ratio[np.isfinite(ratio)]
    if valid_ratio.size == 0:
        raise ValueError("No valid LLTF bins remained after safe division.")

    scale_factor = float(np.median(valid_ratio))
    if not np.isfinite(scale_factor) or scale_factor <= eps:
        raise ValueError(f"Invalid scale factor computed: {scale_factor}")
    return scale_factor


def normalize_htltf_amplitude_with_lltf(
    h_l: np.ndarray,
    h_ht: np.ndarray,
    *,
    eps: float = 1e-8,
    return_scale: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Normalize HT-LTF amplitude using the overlapping LLTF region."""

    scale_factor = compute_lltf_htltf_scale_factor(h_l, h_ht, eps=eps)
    amp_ht = np.abs(np.asarray(h_ht)).astype(np.float32, copy=False)
    normalized_ht = amp_ht / scale_factor

    if return_scale:
        return normalized_ht, scale_factor
    return normalized_ht


def normalize_htltf_complex_with_lltf(
    h_l: np.ndarray,
    h_ht: np.ndarray,
    *,
    eps: float = 1e-8,
    return_scale: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Normalize complex HT-LTF CSI by a scalar derived from LLTF overlap."""

    h_ht = np.asarray(h_ht)
    if h_ht.shape != (114,):
        raise ValueError(f"h_ht must have shape (114,), got {h_ht.shape}")

    scale_factor = compute_lltf_htltf_scale_factor(h_l, h_ht, eps=eps)
    normalized_ht = (h_ht / scale_factor).astype(np.complex64, copy=False)
    if return_scale:
        return normalized_ht, scale_factor
    return normalized_ht


__all__ = [
    "LLTF_COMMON_INDICES",
    "HTLTF_COMMON_INDICES",
    "compute_lltf_htltf_scale_factor",
    "normalize_htltf_amplitude_with_lltf",
    "normalize_htltf_complex_with_lltf",
]
