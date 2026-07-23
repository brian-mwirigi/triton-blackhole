"""Classify numerical drift: benign FP noise vs likely kernel bug."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from triton_blackhole.compare import CompareResult, compare
from triton_blackhole.tolerances import dtype_tolerances


class DriftKind(str, Enum):
    MATCH = "match"
    REDUCTION_ORDER = "reduction_order"  # low-prec accum / nondeterministic reduce
    DTYPE_CAST = "dtype_cast"  # implicit cast / accum dtype mismatch
    LOCALIZED_BUG = "localized_bug"  # tight cluster of large errors → logic bug
    SYSTEMATIC_BIAS = "systematic_bias"  # global shift / scale
    NONFINITE = "nonfinite"  # NaN/Inf
    UNKNOWN = "unknown"


@dataclass
class Classification:
    kind: DriftKind
    confidence: float  # 0..1
    rationale: str
    compare: CompareResult
    hints: list[str]


def classify_drift(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> Classification:
    """
    Heuristically classify why Triton output diverges from a torch reference.

    Not a proof — a triage signal so you know whether to loosen tolerances,
    fix accumulation dtype, or dig into indexing / masking bugs.
    """
    result = compare(actual, reference, atol=atol, rtol=rtol, top_k=8)
    hints: list[str] = []

    if result.passed:
        return Classification(
            kind=DriftKind.MATCH,
            confidence=1.0,
            rationale="outputs agree within tolerances",
            compare=result,
            hints=[],
        )

    if result.n_nan_actual or result.n_nan_reference or result.n_inf_actual or result.n_inf_reference:
        hints.append("Inspect masking on boundary tiles and divide-by-zero in reductions.")
        hints.append("Check that softmax/normalize kernels subtract max in the kernel's working dtype.")
        return Classification(
            kind=DriftKind.NONFINITE,
            confidence=0.95,
            rationale="NaN/Inf present in actual and/or reference",
            compare=result,
            hints=hints,
        )

    # Dtype mismatch between sides is a strong cast signal.
    if actual.dtype != reference.dtype:
        hints.append(
            f"Cast both sides to the same dtype before compare "
            f"(actual={actual.dtype}, reference={reference.dtype})."
        )
        hints.append("Verify tl.dot / tl.sum accumulator dtype (fp32 accum recommended for fp16/bf16).")
        return Classification(
            kind=DriftKind.DTYPE_CAST,
            confidence=0.85,
            rationale="actual and reference dtypes differ",
            compare=result,
            hints=hints,
        )

    tol = dtype_tolerances(actual.dtype, atol=atol, rtol=rtol)
    af = actual.float()
    rf = reference.float()
    diff = af - rf
    abs_err = diff.abs()

    # Systematic bias: mean error large relative to rms of centered error.
    mean_err = float(diff.mean().item())
    centered_rms = float(torch.sqrt(((diff - mean_err) ** 2).mean()).item()) + 1e-12
    if abs(mean_err) > 3 * centered_rms and abs(mean_err) > tol.atol:
        hints.append("Look for missing scale, wrong stride, or an off-by-one in a reduction axis.")
        return Classification(
            kind=DriftKind.SYSTEMATIC_BIAS,
            confidence=0.75,
            rationale=f"mean error {mean_err:.4g} dominates centered rms {centered_rms:.4g}",
            compare=result,
            hints=hints,
        )

    # Localized bug: few elements carry most of the L1 error mass.
    flat = abs_err.reshape(-1)
    total = float(flat.sum().item()) + 1e-12
    top = torch.topk(flat, k=min(32, flat.numel())).values
    top_frac = float(top.sum().item()) / total
    if result.mismatch_frac < 0.01 and top_frac > 0.5 and result.max_abs_error > 10 * tol.atol:
        hints.append("Error mass is concentrated — suspect indexing, masking, or a single bad tile.")
        hints.append("Run triton_blackhole.bisect_axes / bisect_tiles to shrink to the failing region.")
        return Classification(
            kind=DriftKind.LOCALIZED_BUG,
            confidence=0.8,
            rationale=(
                f"{result.mismatch_frac * 100:.3f}% elements mismatch but "
                f"top-32 hold {top_frac * 100:.1f}% of L1 error"
            ),
            compare=result,
            hints=hints,
        )

    # Reduction-order / low-prec drift: widespread small errors within ~few ULPs of dtype.
    if result.mismatch_frac > 0.05 and result.max_abs_error < 50 * tol.atol:
        hints.append("Widespread small errors usually mean reduction-order or fp16/bf16 accum drift.")
        hints.append("Compare against a torch reference that uses the same accum dtype (e.g. fp32).")
        hints.append("Consider torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False.")
        hints.append(f"Suggested CI tolerances for {actual.dtype}: atol≈{tol.atol:g}, rtol≈{tol.rtol:g}.")
        return Classification(
            kind=DriftKind.REDUCTION_ORDER,
            confidence=0.7,
            rationale=(
                f"{result.mismatch_frac * 100:.2f}% mismatches with modest max abs "
                f"{result.max_abs_error:.4g} (dtype {actual.dtype})"
            ),
            compare=result,
            hints=hints,
        )

    # Cast-like: promoting both to fp32 and re-running a pure-fp32 ref shrinks error a lot —
    # we approximate by checking if error correlates with magnitude (relatively flat ULPs).
    rel = abs_err / rf.abs().clamp_min(1e-6)
    rel_std = float(rel.std().item()) if rel.numel() > 1 else 0.0
    if rel_std < 5 * tol.rtol and result.mismatch_frac > 0.01:
        hints.append("Relative error is fairly uniform — check implicit casts inside the kernel.")
        hints.append("Ensure tl.load values are cast to fp32 before long reductions.")
        return Classification(
            kind=DriftKind.DTYPE_CAST,
            confidence=0.55,
            rationale="uniform relative error suggests quantization / cast drift",
            compare=result,
            hints=hints,
        )

    hints.append("Run bisect_axes to localize; compare intermediate probe stages next.")
    return Classification(
        kind=DriftKind.UNKNOWN,
        confidence=0.4,
        rationale="divergence does not match a strong heuristic pattern",
        compare=result,
        hints=hints,
    )


def format_classification(c: Classification) -> str:
    lines = [
        "======== triton-blackhole drift classification ========",
        f"kind        : {c.kind.value}",
        f"confidence  : {c.confidence:.2f}",
        f"rationale   : {c.rationale}",
    ]
    if c.hints:
        lines.append("hints:")
        for h in c.hints:
            lines.append(f"  - {h}")
    return "\n".join(lines)
