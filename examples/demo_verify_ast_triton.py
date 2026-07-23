"""
Full AST instrumentation path (requires triton + CUDA).

Instruments a @triton.jit kernel, dumps probe stats only for the failing pid.
"""

from __future__ import annotations

import torch

from triton_blackhole.triton_hooks import triton_available
from triton_blackhole.verify import run_drift_verify


def main() -> None:
    if not triton_available() or not torch.cuda.is_available():
        print("skip: need triton + CUDA")
        return

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        acc = x + y
        # intentional bug on pid == 2
        if pid == 2:
            acc = acc + 5.0
        tl.store(out_ptr + offs, acc, mask=mask)

    BLOCK = 128
    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)

    def launch(kernel, x, y):
        out = torch.empty_like(x)
        kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
        return out

    actual = launch(add_kernel, x, y)
    reference = x + y

    def relaunch(ikernel, failing_pid, debug_buf, x, y):
        out = torch.empty_like(x)
        ikernel[(triton.cdiv(n, BLOCK),)](
            x,
            y,
            out,
            n,
            BLOCK=BLOCK,
            _bh_dbg_ptr=debug_buf,
            _BH_FAILING_PID=failing_pid,
        )
        return out

    art = run_drift_verify(
        actual,
        reference,
        block_sizes=(BLOCK,),
        kernel=add_kernel,
        probes=["acc"],
        relaunch=relaunch,
        relaunch_args=(x, y),
    )
    print(art.report())
    assert art.grid is not None
    assert art.grid.program_id == 2
    if art.probe_dump is None:
        print("NOTE: probe dump unavailable (instrumentation note above)")
    else:
        assert art.probe_dump["acc"]["dumped"] != 0.0
        print("AST probe dump OK")


if __name__ == "__main__":
    main()
