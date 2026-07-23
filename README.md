# triton-blackhole

**Deterministic numerical bisection for Triton kernel floating-point drift.**

When `torch.allclose(triton_out, torch_ref, atol, rtol)` fails, you usually get nothing but a boolean. The fallbacks are worse: `tl.device_print` floods stdout from thousands of unsynchronized threads, and `TRITON_INTERPRET=1` skips the real compiler, lacks bfloat16, and breaks on indirect loads like `tl.load(tl.load(ptr))`. **triton-blackhole** is the missing tool — localize the divergence without leaving the compiled on-device path.

> PyPI/import name: `triton-blackhole` / `triton_blackhole` (the bare name `blackhole` is already taken by an unrelated SMTP MTA).

## What it does

| Pain | triton-blackhole |
|------|------------------|
| `allclose` fails with no location | `compare()` → max-error indices, hotspots, per-axis peaks, neighborhoods |
| Print-debugging floods | `ProbeBank` named stages with tensor context |
| `TRITON_INTERPRET` useless for bf16 / indirect loads | Real compiled kernels + `bisect_tiles(program_id)` |
| Hours hunting reduction-order vs real bugs | `classify_drift()` triage + dtype-aware tolerances |

## Platforms

| Piece | Native Windows | WSL2 + NVIDIA | Native Linux + NVIDIA |
|-------|----------------|---------------|------------------------|
| `compare` / `bisect_axes` / `classify_drift` / `ProbeBank` / CLI | Yes (CPU torch) | Yes | Yes |
| Live Triton kernels | No — no official Windows wheels | Yes | Yes |

**Native Windows** can run the debugger on tensors/demos, but not compile Triton kernels (`pip install triton` only ships manylinux wheels).

**WSL2** is the practical Windows path for the full stack: install Ubuntu in WSL2, enable NVIDIA CUDA on WSL, then `pip install torch triton` and `pip install -e ".[triton]"` inside the distro. Same Linux workflow, on your Windows machine.

```bash
# inside WSL2 (Ubuntu) with NVIDIA CUDA drivers on the Windows host
pip install -e ".[triton,dev]"
python examples/demo_triton_add.py

# or from Windows PowerShell:
wsl -e bash scripts/wsl_verify.sh
```

Verified in WSL2 Ubuntu: `triton` **installs** (manylinux wheel), package tests/demos pass. Live GPU kernels need `nvidia-smi` working inside WSL (CUDA on WSL drivers).

**No local NVIDIA GPU?** Use [Google Colab](https://colab.research.google.com/) (Runtime → GPU), then open [`notebooks/colab_smoke_test.ipynb`](notebooks/colab_smoke_test.ipynb) or:

```python
!pip -q install -U triton "git+https://github.com/brian-mwirigi/triton-blackhole.git"
```

## Quick start

```python
import torch
from triton_blackhole import compare, bisect_axes, classify_drift, format_report
from triton_blackhole.classify import format_classification
from triton_blackhole.triton_hooks import diagnose

triton_out = run_kernel(...)
torch_ref = reference(...)

# 1. Rich compare (bf16/fp16-aware defaults)
print(format_report(compare(triton_out, torch_ref)))

# 2. Triage: reduction_order vs localized_bug vs dtype_cast vs ...
print(format_classification(classify_drift(triton_out, torch_ref)))

# 3. Shrink to the minimal failing sub-tensor
print(bisect_axes(triton_out, torch_ref).report())

# Or one shot:
print(diagnose(triton_out, torch_ref))
```

### Tile / `program_id` bisection

Keep the compiled kernel. Teach your launcher to run only `pid ∈ [lo, hi)`:

```python
from triton_blackhole import bisect_tiles

def launch(pid_lo, pid_hi):
    # mask stores or shrink grid to [pid_lo, pid_hi)
    return run_triton_kernel(..., pid_lo=pid_lo, pid_hi=pid_hi)

result = bisect_tiles(launch, torch_ref, num_programs=grid_size)
print(result.report())  # → failing program_id range
```

### Fusion-boundary probes (not `device_print`)

```python
from triton_blackhole.probe import ProbeBank

bank = ProbeBank()
bank.capture("pre_softmax", scores_ref, side="ref")
bank.capture("pre_softmax", scores_tri, side="tri")  # from a debug buffer store
bank.capture("out", out_ref, side="ref")
bank.capture("out", out_tri, side="tri")
print(bank.report())  # first diverging stage + full tensor context
```

### CLI

```bash
# save tensors from a failing unit test, then:
triton-blackhole compare triton_out.pt torch_ref.pt --bisect --suggest
```

## Why not `TRITON_INTERPRET`?

triton-blackhole never interprets the kernel. Divergence is isolated by:

1. **Output-space bisection** over tensor axes  
2. **Grid-space bisection** over `program_id`  
3. **Stage-space bisection** over named intermediate buffers  

All three run the same binary your production path uses — bf16, tensor cores, and indirect memory access included.

## Examples

```bash
python examples/demo_softmax_drift.py
python examples/demo_tile_bisect.py
python examples/demo_triton_add.py   # requires triton + CUDA
```

## API map

- `triton_blackhole.compare` / `localize` — structured `allclose`
- `triton_blackhole.bisect_axes` — minimal failing sub-tensor
- `triton_blackhole.bisect_tiles` — minimal failing `program_id` range
- `triton_blackhole.classify_drift` — drift taxonomy
- `triton_blackhole.probe.ProbeBank` — intermediate stage diffs
- `triton_blackhole.tolerances` — dtype-aware / empirical atol·rtol
- `triton_blackhole.triton_hooks.diagnose` — one-shot report

## License

MIT
