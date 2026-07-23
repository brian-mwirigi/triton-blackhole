"""
Demo: @verify_drift core loop — compare → grid map → artifact.

AST instrumentation of a live @triton.jit kernel is shown when CUDA+triton
are available; otherwise the decorator still localizes the failing program_id.
"""

from __future__ import annotations

import torch

from triton_blackhole import run_drift_verify, verify_drift


def torch_ref(x: torch.Tensor) -> torch.Tensor:
    return x * 2 + 1


def main() -> None:
    torch.manual_seed(0)
    BLOCK = 16
    x = torch.arange(64, dtype=torch.float32)

    # --- decorator path ---
    @verify_drift(torch_ref, block_sizes=(BLOCK,), raise_on_fail=False)
    def buggy_triton_sim(x: torch.Tensor) -> torch.Tensor:
        out = torch_ref(x).clone()
        out[40:48] += 3.0  # corrupts program_id 2 (tiles of 16)
        return out

    out = buggy_triton_sim(x)
    art = buggy_triton_sim.last_artifact
    assert art is not None
    print(art.report())
    assert art.grid is not None
    print(f"\nexpected program_id=2, got {art.grid.program_id}")
    assert art.grid.program_id == 2

    # --- functional path ---
    art2 = run_drift_verify(out, torch_ref(x), block_sizes=(BLOCK,))
    assert not art2.passed
    print("\nverify_drift demo OK")


if __name__ == "__main__":
    main()
