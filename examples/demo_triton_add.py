"""Optional live Triton example (skipped if triton/GPU unavailable)."""

from __future__ import annotations

import torch

from triton_blackhole.triton_hooks import diagnose, triton_available


def main() -> None:
    if not triton_available():
        print("triton not installed — skipping live kernel demo")
        print("pip install triton  # Linux + NVIDIA GPU required")
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
        tl.store(out_ptr + offs, x + y, mask=mask)

    def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = 256
        add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
        return out

    if not torch.cuda.is_available():
        print("CUDA not available — skipping")
        return

    x = torch.randn(10_000, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    actual = triton_add(x, y)
    reference = x + y
    print(diagnose(actual, reference))


if __name__ == "__main__":
    main()
