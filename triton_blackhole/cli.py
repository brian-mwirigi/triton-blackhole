"""CLI: triton-blackhole diagnose / compare helpers for saved tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from triton_blackhole.bisect import bisect_axes
from triton_blackhole.classify import classify_drift, format_classification
from triton_blackhole.compare import compare
from triton_blackhole.report import format_json, format_report
from triton_blackhole.tolerances import suggest_tolerances


def _load(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("tensor", "data", "out", "output"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
        # single-tensor dict
        tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError(f"cannot extract tensor from {path}")


def cmd_compare(args: argparse.Namespace) -> int:
    actual = _load(Path(args.actual))
    reference = _load(Path(args.reference))
    atol = args.atol
    rtol = args.rtol
    result = compare(actual, reference, atol=atol, rtol=rtol)
    if args.json:
        print(format_json(result))
    else:
        print(format_report(result))
        cls = classify_drift(actual, reference, atol=atol, rtol=rtol)
        print()
        print(format_classification(cls))
        if not result.passed and args.bisect:
            print()
            print(bisect_axes(actual, reference, atol=atol, rtol=rtol).report())
        if not result.passed and args.suggest:
            sug = suggest_tolerances(actual, reference)
            print()
            print(f"suggested tolerances: atol={sug.atol:.6g}  rtol={sug.rtol:.6g}")
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triton-blackhole",
        description="Deterministic numerical bisection debugger for Triton FP drift",
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compare", help="Compare two saved .pt tensors")
    c.add_argument("actual", help="Path to Triton (or candidate) output .pt")
    c.add_argument("reference", help="Path to torch reference output .pt")
    c.add_argument("--atol", type=float, default=None)
    c.add_argument("--rtol", type=float, default=None)
    c.add_argument("--bisect", action="store_true", help="Spatially bisect failing region")
    c.add_argument("--suggest", action="store_true", help="Suggest empirical tolerances")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
