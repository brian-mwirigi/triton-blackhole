"""Human-readable and machine-readable drift reports."""

from __future__ import annotations

import json
from typing import Any

from triton_blackhole.compare import CompareResult, ErrorHotspot


def _fmt_index(index: tuple[int, ...] | None) -> str:
    if index is None:
        return "—"
    return "[" + ", ".join(str(i) for i in index) + "]"


def _fmt_hotspot(h: ErrorHotspot, i: int) -> str:
    lines = [
        f"  #{i} index={_fmt_index(h.index)}",
        f"      actual={h.actual:.8g}  reference={h.reference:.8g}",
        f"      abs={h.abs_error:.6g}  rel={h.rel_error:.6g}",
    ]
    neigh = h.neighborhood.get("actual")
    if neigh is not None:
        lines.append(f"      neighborhood origin={neigh['origin']} shape={neigh['shape']}")
    return "\n".join(lines)


def format_report(result: CompareResult, *, title: str = "triton-blackhole numerical report") -> str:
    """Render a structured text report for a comparison."""
    status = "PASS" if result.passed else "FAIL"
    bar = "=" * (len(title) + 8)
    lines = [
        bar,
        f" {title}",
        bar,
        f"status          : {status}",
        f"shape           : {result.shape}",
        f"dtype           : actual={result.dtype_actual}  reference={result.dtype_reference}",
        f"tolerances      : atol={result.atol:g}  rtol={result.rtol:g}",
        f"mismatches      : {result.n_mismatch}/{result.numel} ({100 * result.mismatch_frac:.4f}%)",
        f"max abs error   : {result.max_abs_error:.6g} @ {_fmt_index(result.max_abs_index)}",
        f"max rel error   : {result.max_rel_error:.6g} @ {_fmt_index(result.max_rel_index)}",
        f"mean abs / rms  : {result.mean_abs_error:.6g} / {result.rms_error:.6g}",
        f"nan/inf actual  : nan={result.n_nan_actual}  inf={result.n_inf_actual}",
        f"nan/inf refer.  : nan={result.n_nan_reference}  inf={result.n_inf_reference}",
    ]

    if result.axis_max_abs:
        lines.append("per-axis peak abs error:")
        for axis, peaks in enumerate(result.axis_max_abs):
            # Show argmax along that axis's peak profile.
            if peaks.numel() == 0:
                continue
            am = int(peaks.argmax().item())
            lines.append(f"  axis {axis}: peak={float(peaks[am]):.6g} at index {am} (of {peaks.numel()})")

    if result.hotspots:
        lines.append(f"top-{len(result.hotspots)} hotspots:")
        for i, h in enumerate(result.hotspots, 1):
            lines.append(_fmt_hotspot(h, i))

    lines.append(bar)
    return "\n".join(lines)


def result_to_dict(result: CompareResult) -> dict[str, Any]:
    """Serialize a CompareResult to a JSON-friendly dict."""
    return {
        "passed": result.passed,
        "atol": result.atol,
        "rtol": result.rtol,
        "dtype_actual": str(result.dtype_actual),
        "dtype_reference": str(result.dtype_reference),
        "shape": list(result.shape),
        "numel": result.numel,
        "n_mismatch": result.n_mismatch,
        "mismatch_frac": result.mismatch_frac,
        "max_abs_error": result.max_abs_error,
        "max_rel_error": result.max_rel_error,
        "max_abs_index": list(result.max_abs_index) if result.max_abs_index else None,
        "max_rel_index": list(result.max_rel_index) if result.max_rel_index else None,
        "mean_abs_error": result.mean_abs_error,
        "rms_error": result.rms_error,
        "n_nan_actual": result.n_nan_actual,
        "n_nan_reference": result.n_nan_reference,
        "n_inf_actual": result.n_inf_actual,
        "n_inf_reference": result.n_inf_reference,
        "hotspots": [
            {
                "index": list(h.index),
                "actual": h.actual,
                "reference": h.reference,
                "abs_error": h.abs_error,
                "rel_error": h.rel_error,
            }
            for h in result.hotspots
        ],
    }


def format_json(result: CompareResult, *, indent: int = 2) -> str:
    return json.dumps(result_to_dict(result), indent=indent)
