"""Dtype-aware absolute/relative tolerances for low-precision numerics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Machine epsilon / practical default atol floors by dtype.
# atol must not sit at float32 defaults when comparing fp16/bf16 near zero.
_DTYPE_DEFAULTS: dict[torch.dtype, tuple[float, float]] = {
    torch.float64: (1e-8, 1e-5),
    torch.float32: (1e-5, 1e-4),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


@dataclass(frozen=True)
class Tolerances:
    atol: float
    rtol: float
    dtype: torch.dtype

    def as_tuple(self) -> tuple[float, float]:
        return self.atol, self.rtol


def _as_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, torch.Tensor):
        return dtype.dtype
    raise TypeError(f"expected torch.dtype or Tensor, got {type(dtype)!r}")


def dtype_tolerances(dtype: Any, *, atol: float | None = None, rtol: float | None = None) -> Tolerances:
    """Return recommended (atol, rtol) for a dtype, with optional overrides."""
    dt = _as_dtype(dtype)
    base_atol, base_rtol = _DTYPE_DEFAULTS.get(dt, (1e-5, 1e-4))
    return Tolerances(
        atol=base_atol if atol is None else atol,
        rtol=base_rtol if rtol is None else rtol,
        dtype=dt,
    )


def suggest_tolerances(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    percentile: float = 99.9,
    safety: float = 1.5,
) -> Tolerances:
    """
    Suggest atol/rtol from observed error distribution.

    Useful when migrating a torch reference to Triton: run once, capture the
    empirical error envelope, then lock tolerances for CI.
    """
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(reference.shape)}")

    dt = actual.dtype if actual.dtype == reference.dtype else torch.promote_types(actual.dtype, reference.dtype)
    a = actual.detach().float().reshape(-1)
    r = reference.detach().float().reshape(-1)

    finite = torch.isfinite(a) & torch.isfinite(r)
    if not bool(finite.any()):
        return dtype_tolerances(dt)

    a, r = a[finite], r[finite]
    abs_err = (a - r).abs()
    denom = r.abs().clamp_min(1e-12)
    rel_err = abs_err / denom

    # High-percentile envelope avoids a single outlier dominating suggestions.
    k = max(1, int(abs_err.numel() * percentile / 100.0))
    abs_p = float(torch.kthvalue(abs_err, k).values)
    rel_p = float(torch.kthvalue(rel_err, k).values)

    base = dtype_tolerances(dt)
    return Tolerances(
        atol=max(base.atol, abs_p * safety),
        rtol=max(base.rtol, rel_p * safety),
        dtype=dt,
    )
