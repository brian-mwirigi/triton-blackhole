"""
Optional Triton integration helpers.

These never require ``TRITON_INTERPRET=1``. Kernels run compiled on-device so
bf16 and indirect loads (``tl.load(tl.load(ptr))``) keep working. Debugging is
done by (1) comparing outputs, (2) bisecting program_id ranges, and (3)
optional debug-buffer stores at probe stages.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from triton_blackhole.bisect import TileBisectResult, bisect_tiles
from triton_blackhole.classify import Classification, classify_drift, format_classification
from triton_blackhole.compare import CompareResult, compare
from triton_blackhole.probe import ProbeBank
from triton_blackhole.report import format_report
from triton_blackhole.tolerances import suggest_tolerances


def triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def diagnose(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    bisect: bool = True,
) -> str:
    """One-shot report: compare → classify → optional spatial bisect."""
    from triton_blackhole.bisect import bisect_axes

    cmp = compare(actual, reference, atol=atol, rtol=rtol)
    cls = classify_drift(actual, reference, atol=atol, rtol=rtol)
    parts = [format_report(cmp), "", format_classification(cls)]

    if not cmp.passed and bisect:
        b = bisect_axes(actual, reference, atol=atol, rtol=rtol)
        parts.extend(["", b.report()])

    if not cmp.passed:
        sug = suggest_tolerances(actual, reference)
        parts.append("")
        parts.append(
            f"empirical tolerance suggestion: atol={sug.atol:.4g}  rtol={sug.rtol:.4g}  ({sug.dtype})"
        )
    return "\n".join(parts)


def debug_kernel(
    triton_launch: Callable[..., torch.Tensor],
    torch_ref: Callable[..., torch.Tensor],
    *args: Any,
    atol: float | None = None,
    rtol: float | None = None,
    num_programs: int | None = None,
    tile_launch: Callable[[int, int], torch.Tensor] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run Triton vs torch reference and return a structured diagnosis dict.

    Parameters
    ----------
    triton_launch
        ``(*args, **kwargs) -> Tensor`` full-grid kernel launcher.
    torch_ref
        Same signature, pure torch reference.
    num_programs / tile_launch
        If both provided, also bisect ``program_id`` space via ``tile_launch(lo, hi)``.
    """
    actual = triton_launch(*args, **kwargs)
    reference = torch_ref(*args, **kwargs)
    cmp = compare(actual, reference, atol=atol, rtol=rtol)
    cls = classify_drift(actual, reference, atol=atol, rtol=rtol)

    out: dict[str, Any] = {
        "passed": cmp.passed,
        "compare": cmp,
        "classification": cls,
        "report": diagnose(actual, reference, atol=atol, rtol=rtol, bisect=True),
    }

    if not cmp.passed and num_programs is not None and tile_launch is not None:
        tiles: TileBisectResult = bisect_tiles(
            tile_launch,
            reference,
            num_programs=num_programs,
            atol=atol,
            rtol=rtol,
        )
        out["tile_bisect"] = tiles
        out["report"] = out["report"] + "\n\n" + tiles.report()

    return out


def attach_probe_buffers(
    bank: ProbeBank,
    buffers: dict[str, torch.Tensor],
    *,
    side: str = "actual",
) -> ProbeBank:
    """Copy named debug buffers (filled by the kernel) into a ProbeBank."""
    for name, tensor in buffers.items():
        bank.capture(name, tensor, side=side)
    return bank


# Template snippet developers can paste into kernels for stage stores.
PROBE_STORE_SNIPPET = '''
# --- triton-blackhole probe store (paste into kernel) ---
# preallocate: probe = torch.empty_like(tile_out)
# pass probe_ptr into the kernel; only store when debugging:
#   if STORE_PROBE:
#       tl.store(probe_ptr + offs, acc, mask=mask)
# Then: bank.capture("stage_name", probe, side="tri")
'''
