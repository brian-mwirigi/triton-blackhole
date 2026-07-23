"""Deterministic spatial and tile bisection to isolate numerical divergence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from triton_blackhole.compare import CompareResult, compare
from triton_blackhole.report import format_report


@dataclass
class SliceSpec:
    """Inclusive-exclusive half-open slices per tensor axis."""

    ranges: tuple[tuple[int, int], ...]  # (start, stop) per dim

    def as_slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, b) for a, b in self.ranges)

    def volume(self) -> int:
        v = 1
        for a, b in self.ranges:
            v *= max(0, b - a)
        return v

    def __str__(self) -> str:
        parts = [f"{a}:{b}" for a, b in self.ranges]
        return "[" + ", ".join(parts) + "]"


@dataclass
class BisectResult:
    found: bool
    minimal_slice: SliceSpec | None
    steps: int
    history: list[dict] = field(default_factory=list)
    final_compare: CompareResult | None = None

    def report(self) -> str:
        lines = ["======== triton-blackhole bisection ========"]
        if not self.found:
            lines.append("no divergence found within tolerances")
            return "\n".join(lines)
        lines.append(f"minimal failing region : {self.minimal_slice}")
        lines.append(f"region volume          : {self.minimal_slice.volume() if self.minimal_slice else 0}")
        lines.append(f"bisection steps        : {self.steps}")
        if self.final_compare is not None:
            lines.append("")
            lines.append(format_report(self.final_compare, title="region compare"))
        return "\n".join(lines)


def _region_fails(
    actual: torch.Tensor,
    reference: torch.Tensor,
    spec: SliceSpec,
    atol: float | None,
    rtol: float | None,
) -> tuple[bool, CompareResult]:
    sl = spec.as_slices()
    result = compare(actual[sl], reference[sl], atol=atol, rtol=rtol, top_k=3)
    return (not result.passed), result


def bisect_axes(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    axes: Sequence[int] | None = None,
    min_volume: int = 1,
    max_steps: int = 64,
) -> BisectResult:
    """
    Binary-search the tensor volume down to a minimal failing sub-tensor.

    This is the primary escape hatch when ``torch.allclose`` fails: instead of
    dumping every thread's ``tl.device_print``, bisect the output space until the
    smallest contiguous region that still diverges remains.
    """
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(reference.shape)}")

    ndim = actual.ndim
    if axes is None:
        axes = tuple(range(ndim))
    axes = tuple(axes)

    full = SliceSpec(tuple((0, s) for s in actual.shape))
    fails, cmp = _region_fails(actual, reference, full, atol, rtol)
    history: list[dict] = [{"slice": str(full), "fails": fails, "n_mismatch": cmp.n_mismatch}]
    if not fails:
        return BisectResult(found=False, minimal_slice=None, steps=0, history=history, final_compare=cmp)

    current = full
    steps = 0

    # Round-robin bisect across requested axes until volume == min_volume or stuck.
    progress = True
    while progress and steps < max_steps and current.volume() > min_volume:
        progress = False
        for axis in axes:
            if steps >= max_steps:
                break
            start, stop = current.ranges[axis]
            width = stop - start
            if width <= 1:
                continue

            mid = start + width // 2
            left_ranges = list(current.ranges)
            right_ranges = list(current.ranges)
            left_ranges[axis] = (start, mid)
            right_ranges[axis] = (mid, stop)
            left = SliceSpec(tuple(left_ranges))
            right = SliceSpec(tuple(right_ranges))

            left_fails, left_cmp = _region_fails(actual, reference, left, atol, rtol)
            steps += 1
            history.append({"slice": str(left), "fails": left_fails, "n_mismatch": left_cmp.n_mismatch})

            if left_fails:
                current = left
                cmp = left_cmp
                progress = True
                continue

            right_fails, right_cmp = _region_fails(actual, reference, right, atol, rtol)
            steps += 1
            history.append({"slice": str(right), "fails": right_fails, "n_mismatch": right_cmp.n_mismatch})

            if right_fails:
                current = right
                cmp = right_cmp
                progress = True
            # If neither half alone fails, the divergence is split — stop on this axis.
            # Keep current (parent) as the minimal contiguous region that fails.

    return BisectResult(
        found=True,
        minimal_slice=current,
        steps=steps,
        history=history,
        final_compare=cmp,
    )


LaunchFn = Callable[..., torch.Tensor]
# Signature for tile launchers:
#   launcher(pid_lo: int, pid_hi: int) -> actual_output_tensor
# Reference is either a full tensor or a callable producing the reference slice.


@dataclass
class TileBisectResult:
    found: bool
    pid_lo: int
    pid_hi: int
    steps: int
    history: list[dict] = field(default_factory=list)
    final_compare: CompareResult | None = None

    def report(self) -> str:
        lines = [
            "======== triton-blackhole tile bisection ========",
            f"failing program_id range : [{self.pid_lo}, {self.pid_hi})",
            f"span                     : {self.pid_hi - self.pid_lo} tile(s)",
            f"bisection steps          : {self.steps}",
        ]
        if self.final_compare is not None:
            lines.append("")
            lines.append(format_report(self.final_compare, title="tile compare"))
        return "\n".join(lines)


def bisect_tiles(
    launch: Callable[[int, int], torch.Tensor],
    reference: torch.Tensor | Callable[[int, int], torch.Tensor],
    *,
    num_programs: int,
    atol: float | None = None,
    rtol: float | None = None,
    max_steps: int = 64,
    compare_fn: Callable[[torch.Tensor, torch.Tensor], CompareResult] | None = None,
) -> TileBisectResult:
    """
    Bisect Triton ``program_id`` space to find which tile(s) diverge.

    ``launch(pid_lo, pid_hi)`` should run only tiles in ``[pid_lo, pid_hi)``
    (mask other stores or shrink the grid) and return the full output tensor
    (uncomputed tiles should match the reference or be zeroed consistently).

    This avoids ``TRITON_INTERPRET`` entirely — the real compiled kernel runs
    on-device with bf16 and indirect loads intact.
    """

    def _ref(lo: int, hi: int) -> torch.Tensor:
        if callable(reference):
            return reference(lo, hi)
        return reference

    def _cmp(lo: int, hi: int) -> CompareResult:
        actual = launch(lo, hi)
        ref = _ref(lo, hi)
        if compare_fn is not None:
            return compare_fn(actual, ref)
        return compare(actual, ref, atol=atol, rtol=rtol, top_k=3)

    full = _cmp(0, num_programs)
    history = [{"pid": [0, num_programs], "fails": not full.passed, "n_mismatch": full.n_mismatch}]
    if full.passed:
        return TileBisectResult(
            found=False, pid_lo=0, pid_hi=num_programs, steps=0, history=history, final_compare=full
        )

    lo, hi = 0, num_programs
    steps = 0
    last = full

    while hi - lo > 1 and steps < max_steps:
        mid = (lo + hi) // 2
        left = _cmp(lo, mid)
        steps += 1
        history.append({"pid": [lo, mid], "fails": not left.passed, "n_mismatch": left.n_mismatch})
        if not left.passed:
            hi = mid
            last = left
            continue
        right = _cmp(mid, hi)
        steps += 1
        history.append({"pid": [mid, hi], "fails": not right.passed, "n_mismatch": right.n_mismatch})
        if not right.passed:
            lo = mid
            last = right
        else:
            # Split across mid — cannot shrink further contiguously.
            break

    return TileBisectResult(
        found=True,
        pid_lo=lo,
        pid_hi=hi,
        steps=steps,
        history=history,
        final_compare=last,
    )
