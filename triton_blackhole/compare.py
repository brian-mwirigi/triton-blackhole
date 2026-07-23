"""Rich tensor comparison with multidimensional error localization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from triton_blackhole.tolerances import dtype_tolerances


@dataclass
class ErrorHotspot:
    """A single high-error coordinate with surrounding context."""

    index: tuple[int, ...]
    actual: float
    reference: float
    abs_error: float
    rel_error: float
    neighborhood: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareResult:
    passed: bool
    atol: float
    rtol: float
    dtype_actual: torch.dtype
    dtype_reference: torch.dtype
    shape: tuple[int, ...]
    numel: int
    n_mismatch: int
    mismatch_frac: float
    max_abs_error: float
    max_rel_error: float
    max_abs_index: tuple[int, ...] | None
    max_rel_index: tuple[int, ...] | None
    mean_abs_error: float
    rms_error: float
    n_nan_actual: int
    n_nan_reference: int
    n_inf_actual: int
    n_inf_reference: int
    hotspots: list[ErrorHotspot] = field(default_factory=list)
    axis_max_abs: list[torch.Tensor] = field(default_factory=list)
    mask: torch.Tensor | None = None  # True where elements fail allclose

    @property
    def ok(self) -> bool:
        return self.passed


def _unravel(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx), shape))


def _neighborhood(t: torch.Tensor, index: tuple[int, ...], radius: int = 1) -> dict[str, Any]:
    """Extract a small window around `index` for human-readable context."""
    slices: list[slice] = []
    origin: list[int] = []
    for dim, (idx, size) in enumerate(zip(index, t.shape)):
        lo = max(0, idx - radius)
        hi = min(size, idx + radius + 1)
        slices.append(slice(lo, hi))
        origin.append(lo)
    window = t[tuple(slices)].detach().float().cpu()
    return {
        "origin": tuple(origin),
        "shape": tuple(window.shape),
        "values": window.tolist(),
        "center_index": index,
    }


def localize(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    top_k: int = 5,
    neighborhood_radius: int = 1,
) -> CompareResult:
    """Alias for :func:`compare` — find where tensors diverge."""
    return compare(
        actual,
        reference,
        atol=atol,
        rtol=rtol,
        top_k=top_k,
        neighborhood_radius=neighborhood_radius,
    )


def compare(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    top_k: int = 5,
    neighborhood_radius: int = 1,
    equal_nan: bool = False,
) -> CompareResult:
    """
    Compare Triton output vs torch reference with structured localization.

    Unlike bare ``torch.allclose``, this returns the failing mask, max-error
    coordinates, per-axis error peaks, and neighborhood windows — so you see
    *where* in the multidimensional tensor the drift lives.
    """
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(reference.shape)}")

    # Promote comparison math to float32 so bf16/fp16 diffs are measurable.
    tol = dtype_tolerances(actual.dtype, atol=atol, rtol=rtol)
    a = actual.detach()
    r = reference.detach()
    af = a.float()
    rf = r.float()

    abs_err = (af - rf).abs()
    denom = rf.abs().clamp_min(torch.finfo(torch.float32).tiny)
    rel_err = abs_err / denom

    # allclose predicate: |a-b| <= atol + rtol*|b|
    finite = torch.isfinite(af) & torch.isfinite(rf)
    close = (abs_err <= (tol.atol + tol.rtol * rf.abs())) & finite
    if equal_nan:
        close = close | (torch.isnan(af) & torch.isnan(rf))

    mismatch = ~close
    n_mismatch = int(mismatch.sum().item())
    numel = actual.numel()

    flat_abs = abs_err.reshape(-1)
    flat_rel = rel_err.reshape(-1)
    flat_mis = mismatch.reshape(-1)

    if numel == 0:
        max_abs = 0.0
        max_rel = 0.0
        max_abs_index = None
        max_rel_index = None
    else:
        # Prefer max error among mismatches; fall back to global max.
        if n_mismatch > 0:
            masked_abs = flat_abs.clone()
            masked_abs[~flat_mis] = -1.0
            max_abs_flat = int(masked_abs.argmax().item())
            masked_rel = flat_rel.clone()
            masked_rel[~flat_mis] = -1.0
            max_rel_flat = int(masked_rel.argmax().item())
        else:
            max_abs_flat = int(flat_abs.argmax().item())
            max_rel_flat = int(flat_rel.argmax().item())
        max_abs = float(flat_abs[max_abs_flat].item())
        max_rel = float(flat_rel[max_rel_flat].item())
        max_abs_index = _unravel(max_abs_flat, tuple(actual.shape))
        max_rel_index = _unravel(max_rel_flat, tuple(actual.shape))

    mean_abs = float(abs_err[finite].mean().item()) if bool(finite.any()) else 0.0
    rms = float(torch.sqrt((abs_err[finite] ** 2).mean()).item()) if bool(finite.any()) else 0.0

    hotspots: list[ErrorHotspot] = []
    if n_mismatch > 0:
        # Rank mismatches by absolute error.
        scores = flat_abs.clone()
        scores[~flat_mis] = -1.0
        k = min(top_k, n_mismatch)
        top_vals, top_idx = torch.topk(scores, k)
        for flat_i in top_idx.tolist():
            idx = _unravel(int(flat_i), tuple(actual.shape))
            hotspots.append(
                ErrorHotspot(
                    index=idx,
                    actual=float(af[idx].item()),
                    reference=float(rf[idx].item()),
                    abs_error=float(abs_err[idx].item()),
                    rel_error=float(rel_err[idx].item()),
                    neighborhood={
                        "actual": _neighborhood(af, idx, neighborhood_radius),
                        "reference": _neighborhood(rf, idx, neighborhood_radius),
                    },
                )
            )

    axis_max: list[torch.Tensor] = []
    if actual.ndim > 0:
        for axis in range(actual.ndim):
            reduce_dims = tuple(d for d in range(actual.ndim) if d != axis)
            if reduce_dims:
                profile = abs_err.amax(dim=reduce_dims)
            else:
                profile = abs_err
            axis_max.append(profile.detach().cpu().reshape(-1))

    return CompareResult(
        passed=n_mismatch == 0 and int(torch.isnan(af).sum()) == 0 and int(torch.isnan(rf).sum()) == 0,
        atol=tol.atol,
        rtol=tol.rtol,
        dtype_actual=a.dtype,
        dtype_reference=r.dtype,
        shape=tuple(actual.shape),
        numel=numel,
        n_mismatch=n_mismatch,
        mismatch_frac=n_mismatch / numel if numel else 0.0,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        max_abs_index=max_abs_index,
        max_rel_index=max_rel_index,
        mean_abs_error=mean_abs,
        rms_error=rms,
        n_nan_actual=int(torch.isnan(af).sum().item()),
        n_nan_reference=int(torch.isnan(rf).sum().item()),
        n_inf_actual=int(torch.isinf(af).sum().item()),
        n_inf_reference=int(torch.isinf(rf).sum().item()),
        hotspots=hotspots,
        axis_max_abs=axis_max,
        mask=mismatch.detach().cpu() if n_mismatch > 0 else None,
    )


def assert_close(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    msg: str = "",
) -> CompareResult:
    """Like ``torch.testing.assert_close``, but raises with a triton-blackhole report."""
    from triton_blackhole.report import format_report

    result = compare(actual, reference, atol=atol, rtol=rtol)
    if not result.passed:
        report = format_report(result)
        raise AssertionError((msg + "\n" if msg else "") + report)
    return result
