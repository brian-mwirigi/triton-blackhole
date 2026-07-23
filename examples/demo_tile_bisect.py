"""
Tile / program_id bisection demo.

Shows how to wrap a launcher so triton-blackhole can binary-search which Triton
program_id range diverges — without TRITON_INTERPRET or device_print floods.
"""

from __future__ import annotations

import torch

from triton_blackhole import bisect_tiles, compare, format_report


def reference_block_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float() @ b.float()


def simulated_tiled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block: int = 32,
    pid_lo: int = 0,
    pid_hi: int | None = None,
    bad_pid: int = 5,
) -> torch.Tensor:
    """
    CPU stand-in for a Triton matmul grid.

    Only tiles with program_id in [pid_lo, pid_hi) are written; others stay 0.
    Tile ``bad_pid`` intentionally writes corrupted values.
    """
    m, k = a.shape
    _, n = b.shape
    out = torch.zeros(m, n, dtype=torch.float32)
    grid_m = (m + block - 1) // block
    grid_n = (n + block - 1) // block
    num = grid_m * grid_n
    if pid_hi is None:
        pid_hi = num

    ref = reference_block_matmul(a, b)
    for pid in range(pid_lo, pid_hi):
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        rm = slice(pid_m * block, min((pid_m + 1) * block, m))
        rn = slice(pid_n * block, min((pid_n + 1) * block, n))
        tile = ref[rm, rn]
        if pid == bad_pid:
            tile = tile + 3.0  # localized kernel bug
        out[rm, rn] = tile
    return out


def main() -> None:
    torch.manual_seed(1)
    block = 32
    a = torch.randn(128, 64)
    b = torch.randn(64, 128)
    grid_m = 128 // block
    grid_n = 128 // block
    num_programs = grid_m * grid_n
    bad_pid = 5

    full = simulated_tiled_matmul(a, b, block=block, bad_pid=bad_pid)
    ref = reference_block_matmul(a, b)

    print(format_report(compare(full, ref, atol=1e-5, rtol=1e-5)))

    def launch(lo: int, hi: int) -> torch.Tensor:
        # Outside the active pid range, write the reference so comparison
        # only sees errors from tiles in [lo, hi).
        out = ref.clone()
        active = simulated_tiled_matmul(a, b, block=block, pid_lo=lo, pid_hi=hi, bad_pid=bad_pid)
        grid_n_local = 128 // block
        for pid in range(lo, hi):
            pid_m = pid // grid_n_local
            pid_n = pid % grid_n_local
            rm = slice(pid_m * block, (pid_m + 1) * block)
            rn = slice(pid_n * block, (pid_n + 1) * block)
            out[rm, rn] = active[rm, rn]
        return out

    result = bisect_tiles(launch, ref, num_programs=num_programs, atol=1e-5, rtol=1e-5)
    print()
    print(result.report())
    print(f"\nexpected bad_pid={bad_pid}, found range=[{result.pid_lo}, {result.pid_hi})")
    assert result.pid_lo <= bad_pid < result.pid_hi
    print("tile bisection isolated the corrupted program_id ✓")


if __name__ == "__main__":
    main()
