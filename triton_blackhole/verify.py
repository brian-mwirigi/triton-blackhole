"""
@verify_drift — drop-in verification that runs the TritonDrift core loop:

1. Execute Triton path vs PyTorch reference
2. Precision-aware compare
3. Map hotspot → program_id via block sizes
4. Optional AST instrumentation + single-block probe dump
5. Terminal drift artifact
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from triton_blackhole.bisect import bisect_axes
from triton_blackhole.classify import Classification, classify_drift, format_classification
from triton_blackhole.compare import CompareResult, compare
from triton_blackhole.gridmap import GridMapping, format_grid_mapping, index_to_program_id
from triton_blackhole.report import format_report
@dataclass
class DriftArtifact:
    """Full terminal-facing result of a drift verification run."""

    passed: bool
    compare: CompareResult
    classification: Classification
    grid: GridMapping | None = None
    probe_dump: dict[str, dict[str, float]] | None = None
    instrumented_source: str | None = None
    bisect_report: str | None = None
    extra_notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        return format_drift_artifact(self)


def format_drift_artifact(art: DriftArtifact) -> str:
    parts = [
        "======== triton-blackhole drift artifact ========",
        format_report(art.compare, title="output compare"),
        "",
        format_classification(art.classification),
    ]
    if art.grid is not None:
        parts.extend(["", format_grid_mapping(art.grid)])
    if art.bisect_report:
        parts.extend(["", art.bisect_report])
    if art.probe_dump is not None and art.grid is not None:
        from triton_blackhole.instrument import format_probe_dump

        parts.extend(["", format_probe_dump(art.probe_dump, failing_pid=art.grid.program_id)])
    if art.instrumented_source:
        parts.extend(
            [
                "",
                "======== instrumented kernel (excerpt) ========",
                _excerpt(art.instrumented_source, max_lines=40),
            ]
        )
    if not art.passed:
        sug = suggest_tolerances_safe(art.compare)
        if sug:
            parts.extend(["", sug])
    for note in art.extra_notes:
        parts.extend(["", note])
    parts.append("======== end drift artifact ========")
    return "\n".join(parts)


def suggest_tolerances_safe(cmp: CompareResult) -> str | None:
    try:
        # Recreate tensors not available — just echo mismatch stats.
        return (
            f"tolerances used: atol={cmp.atol:g}  rtol={cmp.rtol:g}  |  "
            f"mismatches={cmp.n_mismatch}/{cmp.numel}"
        )
    except Exception:
        return None


def _excerpt(src: str, max_lines: int = 40) -> str:
    lines = src.splitlines()
    if len(lines) <= max_lines:
        return src
    return "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"])


def run_drift_verify(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    block_sizes: Sequence[int] | None = None,
    axis_map: Sequence[int] | None = None,
    bisect: bool = True,
    # Optional AST instrumentation path:
    kernel: Any | None = None,
    probes: Sequence[str] | None = None,
    relaunch: Callable[..., torch.Tensor] | None = None,
    relaunch_args: tuple[Any, ...] = (),
    relaunch_kwargs: dict[str, Any] | None = None,
) -> DriftArtifact:
    """
    Core TritonDrift loop on concrete tensors (+ optional kernel instrumentation).

    Parameters
    ----------
    block_sizes:
        Constexpr tile sizes along mapped output axes, e.g. ``(BLOCK_M, BLOCK_N)``.
        Enables output-index → ``program_id`` mapping.
    kernel / probes / relaunch:
        If provided on failure, AST-instrument ``kernel`` for ``probes``, then call
        ``relaunch(instrumented_kernel, failing_pid, debug_buf, *relaunch_args, **kwargs)``
        which must invoke the instrumented kernel and return any tensor (output unused
        for probe decode — probes are read from ``debug_buf``).
    """
    cmp = compare(actual, reference, atol=atol, rtol=rtol)
    cls = classify_drift(actual, reference, atol=atol, rtol=rtol)
    notes: list[str] = []
    grid: GridMapping | None = None
    bisect_report = None
    probe_dump = None
    instrumented_source = None

    if cmp.passed:
        return DriftArtifact(passed=True, compare=cmp, classification=cls)

    if bisect:
        bisect_report = bisect_axes(actual, reference, atol=atol, rtol=rtol).report()

    if block_sizes is not None and cmp.max_abs_index is not None:
        grid = index_to_program_id(
            cmp.max_abs_index,
            actual.shape,
            block_sizes,
            axis_map=axis_map,
        )

    if (
        kernel is not None
        and probes
        and relaunch is not None
        and grid is not None
    ):
        try:
            probe_dump, instrumented_source = _run_instrumented(
                kernel=kernel,
                probes=probes,
                failing_pid=grid.program_id,
                relaunch=relaunch,
                relaunch_args=relaunch_args,
                relaunch_kwargs=relaunch_kwargs or {},
                like=actual,
            )
        except Exception as e:  # instrumentation is best-effort
            notes.append(f"AST instrumentation skipped: {type(e).__name__}: {e}")

    return DriftArtifact(
        passed=False,
        compare=cmp,
        classification=cls,
        grid=grid,
        probe_dump=probe_dump,
        instrumented_source=instrumented_source,
        bisect_report=bisect_report,
        extra_notes=notes,
    )


def _run_instrumented(
    *,
    kernel: Any,
    probes: Sequence[str],
    failing_pid: int,
    relaunch: Callable[..., torch.Tensor],
    relaunch_args: tuple[Any, ...],
    relaunch_kwargs: dict[str, Any],
    like: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], str]:
    from triton_blackhole.instrument import (
        PROBE_FLOATS,
        decode_probe_buffer,
        instrument_kernel,
    )

    inst = instrument_kernel(kernel, probes)
    debug = torch.zeros(
        inst.debug_slots,
        device=like.device,
        dtype=torch.float32,
    )
    relaunch(
        inst.kernel,
        failing_pid,
        debug,
        *relaunch_args,
        **relaunch_kwargs,
    )
    decoded = decode_probe_buffer(
        debug,
        inst.probe_names,
        floats_per_probe=PROBE_FLOATS,
    )
    return decoded, inst.source


def verify_drift(
    ref_func: Callable[..., torch.Tensor],
    *,
    atol: float | None = None,
    rtol: float | None = None,
    block_sizes: Sequence[int] | None = None,
    axis_map: Sequence[int] | None = None,
    bisect: bool = True,
    kernel: Any | None = None,
    probes: Sequence[str] | None = None,
    relaunch: Callable[..., torch.Tensor] | None = None,
    raise_on_fail: bool = True,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """
    Decorator: run a Triton launcher vs ``ref_func`` and emit a drift artifact.

    Usage::

        @verify_drift(torch_ref, block_sizes=(32, 32), raise_on_fail=True)
        def run(a, b):
            out = launch_triton(a, b)
            return out

        run(a, b)  # silent if match; raises AssertionError with artifact if not

    The decorated function must accept the same ``*args, **kwargs`` as ``ref_func``
    and return the Triton output tensor.

    For AST probe dumps on failure, also pass ``kernel``, ``probes``, and
    ``relaunch(instrumented, failing_pid, debug_buf, *args, **kwargs)``.
    """

    def decorator(triton_func: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        @functools.wraps(triton_func)
        def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
            actual = triton_func(*args, **kwargs)
            reference = ref_func(*args, **kwargs)
            if not isinstance(actual, torch.Tensor) or not isinstance(reference, torch.Tensor):
                raise TypeError("@verify_drift expects both sides to return torch.Tensor")

            art = run_drift_verify(
                actual,
                reference,
                atol=atol,
                rtol=rtol,
                block_sizes=block_sizes,
                axis_map=axis_map,
                bisect=bisect,
                kernel=kernel,
                probes=probes,
                relaunch=relaunch,
                relaunch_args=args,
                relaunch_kwargs=kwargs,
            )
            # Stash for callers that want structured access.
            wrapper.last_artifact = art  # type: ignore[attr-defined]
            if not art.passed and raise_on_fail:
                raise AssertionError(art.report())
            return actual

        wrapper.last_artifact = None  # type: ignore[attr-defined]
        return wrapper

    return decorator


# Alias matching the product spec name.
verify = verify_drift
