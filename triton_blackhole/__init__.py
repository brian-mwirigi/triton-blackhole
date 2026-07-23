"""triton-blackhole — deterministic numerical bisection for Triton FP drift."""

from triton_blackhole.bisect import BisectResult, bisect_axes, bisect_tiles
from triton_blackhole.classify import DriftKind, classify_drift
from triton_blackhole.compare import CompareResult, compare, localize
from triton_blackhole.gridmap import GridMapping, index_to_program_id
from triton_blackhole.probe import ProbeBank, probe_stage
from triton_blackhole.report import format_report
from triton_blackhole.tolerances import dtype_tolerances, suggest_tolerances
from triton_blackhole.verify import DriftArtifact, run_drift_verify, verify, verify_drift

__version__ = "0.2.0"
__all__ = [
    "BisectResult",
    "CompareResult",
    "DriftArtifact",
    "DriftKind",
    "GridMapping",
    "ProbeBank",
    "bisect_axes",
    "bisect_tiles",
    "classify_drift",
    "compare",
    "dtype_tolerances",
    "format_report",
    "index_to_program_id",
    "localize",
    "probe_stage",
    "run_drift_verify",
    "suggest_tolerances",
    "verify",
    "verify_drift",
]
