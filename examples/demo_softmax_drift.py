"""
Simulate the classic Triton failure mode: reduction-order / low-prec drift
plus a localized mask bug that torch.allclose cannot explain.
"""

from __future__ import annotations

import torch

from triton_blackhole import bisect_axes, classify_drift, compare, format_report
from triton_blackhole.classify import format_classification
from triton_blackhole.probe import ProbeBank
from triton_blackhole.tolerances import suggest_tolerances


def torch_softmax_ref(x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    x32 = x32 - x32.max(dim=-1, keepdim=True).values
    return torch.exp(x32).div(torch.exp(x32).sum(dim=-1, keepdim=True)).to(x.dtype)


def triton_like_softmax(x: torch.Tensor, *, inject_bug: bool = True) -> torch.Tensor:
    """Stand-in for a Triton kernel with bf16 accum (+ optional localized bug)."""
    x_low = x.to(torch.bfloat16)
    m = x_low.max(dim=-1, keepdim=True).values
    e = torch.exp((x_low - m).float()).to(torch.bfloat16)
    out = (e / e.sum(dim=-1, keepdim=True)).to(x.dtype)

    if inject_bug:
        out = out.clone()
        # Simulate a bad store mask on one tile.
        out[7:9, 100:108] = out[7:9, 100:108] + 0.5
    return out


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 256, dtype=torch.bfloat16)

    ref = torch_softmax_ref(x)
    tri = triton_like_softmax(x, inject_bug=True)

    print("=== bare torch.allclose (the usual dead end) ===")
    print("allclose:", torch.allclose(tri, ref, atol=1e-2, rtol=1e-2))

    print("\n=== triton_blackhole.compare (multidimensional context) ===")
    result = compare(tri, ref)
    print(format_report(result))

    print("\n=== triton_blackhole.classify_drift ===")
    cls = classify_drift(tri, ref)
    print(format_classification(cls))

    print("\n=== triton_blackhole.bisect_axes (minimal failing region) ===")
    # Use tight tolerances so we isolate the injected bug, not bf16 haze.
    b = bisect_axes(tri, ref, atol=1e-1, rtol=1e-1)
    print(b.report())

    print("\n=== probe stages (fusion-boundary bisection) ===")
    bank = ProbeBank()
    # Reference stages
    x32 = x.float()
    x32 = x32 - x32.max(dim=-1, keepdim=True).values
    bank.capture("shifted", x32, side="ref")
    probs = torch.softmax(x32, dim=-1)
    bank.capture("probs", probs, side="ref")

    # "Triton" stages — drift begins at bf16 shift, bug at final probs
    x_low = x.to(torch.bfloat16)
    shifted_tri = (x_low - x_low.max(dim=-1, keepdim=True).values).float()
    bank.capture("shifted", shifted_tri, side="tri")
    bank.capture("probs", tri.float(), side="tri")
    print(bank.report(atol=1e-1, rtol=1e-1))

    print("\n=== empirical tolerance suggestion (bug-free kernel) ===")
    clean = triton_like_softmax(x, inject_bug=False)
    sug = suggest_tolerances(clean, ref)
    print(f"atol={sug.atol:.4g}  rtol={sug.rtol:.4g}  dtype={sug.dtype}")


if __name__ == "__main__":
    main()
