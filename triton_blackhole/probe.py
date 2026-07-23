"""
Intermediate-stage probes — structured alternative to tl.device_print.

Capture named tensors at fusion boundaries (both Triton debug buffers and
torch reference stages), then compare them with full multidimensional context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch

from triton_blackhole.classify import Classification, classify_drift, format_classification
from triton_blackhole.compare import CompareResult, compare
from triton_blackhole.report import format_report


@dataclass
class StageRecord:
    name: str
    tensor: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


class ProbeBank:
    """
    Collect named intermediate tensors from kernel / reference runs.

    Typical workflow::

        bank = ProbeBank()
        # torch reference path
        bank.capture("pre_softmax", scores_ref, side="ref")
        bank.capture("post_softmax", probs_ref, side="ref")
        # triton path writes debug buffers, then:
        bank.capture("pre_softmax", scores_tri, side="tri")
        bank.capture("post_softmax", probs_tri, side="tri")
        report = bank.diff()
    """

    def __init__(self) -> None:
        self._stages: dict[str, dict[str, StageRecord]] = {}

    def capture(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        side: str = "actual",
        **meta: Any,
    ) -> torch.Tensor:
        """Store a clone of ``tensor`` under ``name`` / ``side`` (passthrough)."""
        bucket = self._stages.setdefault(name, {})
        bucket[side] = StageRecord(name=name, tensor=tensor.detach().clone(), meta=dict(meta))
        return tensor

    def stages(self) -> list[str]:
        return list(self._stages.keys())

    def get(self, name: str, side: str) -> torch.Tensor:
        return self._stages[name][side].tensor

    def clear(self) -> None:
        self._stages.clear()

    def diff(
        self,
        *,
        actual_side: str = "actual",
        reference_side: str = "reference",
        atol: float | None = None,
        rtol: float | None = None,
        aliases: tuple[tuple[str, str], ...] = (("tri", "ref"), ("triton", "torch"), ("actual", "reference")),
    ) -> list[tuple[str, CompareResult, Classification]]:
        """
        Compare matching stages across sides.

        Accepts common side-name aliases (tri/ref, actual/reference, ...).
        Returns results in capture order; stops being useful once the first
        stage fails — that is the fusion boundary where drift begins.
        """
        results: list[tuple[str, CompareResult, Classification]] = []
        for name, sides in self._stages.items():
            a_key, r_key = _resolve_sides(sides, actual_side, reference_side, aliases)
            if a_key is None or r_key is None:
                continue
            a = sides[a_key].tensor
            r = sides[r_key].tensor
            cmp = compare(a, r, atol=atol, rtol=rtol)
            cls = classify_drift(a, r, atol=atol, rtol=rtol)
            results.append((name, cmp, cls))
        return results

    def first_divergence(
        self,
        *,
        atol: float | None = None,
        rtol: float | None = None,
        **kwargs: Any,
    ) -> tuple[str, CompareResult, Classification] | None:
        """Return the earliest stage (by insertion order) that fails compare."""
        for name, cmp, cls in self.diff(atol=atol, rtol=rtol, **kwargs):
            if not cmp.passed:
                return name, cmp, cls
        return None

    def report(self, *, atol: float | None = None, rtol: float | None = None) -> str:
        lines = ["======== triton-blackhole probe stage diff ========"]
        diffs = self.diff(atol=atol, rtol=rtol)
        if not diffs:
            lines.append("(no paired stages captured)")
            return "\n".join(lines)

        first_fail: str | None = None
        for name, cmp, cls in diffs:
            status = "PASS" if cmp.passed else "FAIL"
            lines.append(f"[{status}] stage={name!r}  mismatches={cmp.n_mismatch}  max_abs={cmp.max_abs_error:.6g}")
            if not cmp.passed and first_fail is None:
                first_fail = name
                lines.append(format_classification(cls))
                lines.append(format_report(cmp, title=f"stage {name!r}"))

        if first_fail is None:
            lines.append("all probed stages match within tolerances")
        else:
            lines.append(f"first divergence at stage: {first_fail!r}")
        return "\n".join(lines)


def _resolve_sides(
    sides: dict[str, StageRecord],
    actual_side: str,
    reference_side: str,
    aliases: tuple[tuple[str, str], ...],
) -> tuple[str | None, str | None]:
    candidates = [(actual_side, reference_side), *aliases]
    for a, r in candidates:
        if a in sides and r in sides:
            return a, r
    # Fallback: if exactly two sides, treat insertion order as actual/reference.
    if len(sides) == 2:
        keys = list(sides.keys())
        return keys[0], keys[1]
    return None, None


# Module-level bank for decorator convenience.
_DEFAULT_BANK = ProbeBank()


def get_default_bank() -> ProbeBank:
    return _DEFAULT_BANK


def probe_stage(
    name: str,
    *,
    side: str = "actual",
    bank: ProbeBank | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Functional helper: ``out = probe_stage("logits", side="ref")(tensor)``.

    Also usable as a thin decorator around a tensor-returning callable.
    """
    target = bank if bank is not None else _DEFAULT_BANK

    def _capture(t: torch.Tensor) -> torch.Tensor:
        return target.capture(name, t, side=side)

    return _capture


def iter_stage_reports(bank: ProbeBank, **kwargs: Any) -> Iterator[str]:
    for name, cmp, cls in bank.diff(**kwargs):
        yield f"stage={name} passed={cmp.passed} kind={cls.kind.value}"
