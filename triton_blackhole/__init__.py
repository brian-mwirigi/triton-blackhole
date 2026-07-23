"""triton-blackhole — deterministic numerical bisection for Triton FP drift."""

from triton_blackhole.bisect import BisectResult, bisect_axes, bisect_tiles
from triton_blackhole.classify import DriftKind, classify_drift
from triton_blackhole.compare import CompareResult, compare, localize
from triton_blackhole.probe import ProbeBank, probe_stage
from triton_blackhole.tolerances import dtype_tolerances, suggest_tolerances
from triton_blackhole.report import format_report

__version__ = "0.1.0"
__all__ = [
    "BisectResult",
    "CompareResult",
    "DriftKind",
    "ProbeBank",
    "bisect_axes",
    "bisect_tiles",
    "classify_drift",
    "compare",
    "dtype_tolerances",
    "format_report",
    "localize",
    "probe_stage",
    "suggest_tolerances",
]
