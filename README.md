# triton-blackhole

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brian-mwirigi/triton-blackhole/blob/main/notebooks/colab_killer_demo.ipynb)
[![PyPI](https://img.shields.io/pypi/v/triton-blackhole)](https://pypi.org/project/triton-blackhole/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**`torch.allclose` failed. Now what?**

triton-blackhole is a deterministic numerical debugger for Triton kernels. It finds *where* your output diverges from a PyTorch reference — and whether it's benign fp16/bf16 drift or a real bug — **without** `tl.device_print` floods or `TRITON_INTERPRET`.

```python
from triton_blackhole import verify_drift

@verify_drift(torch_ref, block_sizes=(BLOCK_M, BLOCK_N))
def run(a, b):
    return launch_triton(a, b)

run(a, b)  # silent if OK; AssertionError + drift artifact if not
```

**[▶ Open the 2‑minute Colab demo](https://colab.research.google.com/github/brian-mwirigi/triton-blackhole/blob/main/notebooks/colab_killer_demo.ipynb)** (Runtime → GPU)

---

## The pain

| You try… | What happens |
|----------|----------------|
| `torch.allclose(...)` | `False`. No index. No cause. |
| `tl.device_print` | Thousands of unsynced lines. No tensor context. |
| `TRITON_INTERPRET=1` | Breaks on bf16 / `tl.load(tl.load(...))`. Not your real kernel. |

## The fix (TritonDrift loop)

| Feature | What you get |
|---------|----------------|
| `@verify_drift` | Drop-in decorator: capture inputs, compare, emit artifact |
| Output → `program_id` | Map hotspot `[i,j]` → tile / `program_id` via `BLOCK_*` |
| AST probe injection | Rewrite kernel AST; dump intermediates **only** on the failing pid |
| Precision-aware diff | bf16/fp16 tolerances + `classify_drift` |
| Terminal artifact | Failing block, expected vs actual, probe stats |

```python
from triton_blackhole import verify_drift, run_drift_verify, index_to_program_id

# Decorator (pytest-friendly)
@verify_drift(torch_ref, block_sizes=(32, 32), raise_on_fail=True)
def run(a, b):
    return triton_launch(a, b)

# Or functional
art = run_drift_verify(tri_out, ref_out, block_sizes=(32, 32))
print(art.report())  # compare + grid map + bisect
```

### Optional: AST dump on the failing block

```python
def relaunch(ikernel, failing_pid, debug_buf, a, b):
    out = torch.empty_like(...)
    ikernel[(grid,)](
        a, b, out, ...,
        _bh_dbg_ptr=debug_buf,
        _BH_FAILING_PID=failing_pid,
    )
    return out

@verify_drift(
    torch_ref,
    block_sizes=(BLOCK_M, BLOCK_N),
    kernel=my_kernel,          # original @triton.jit fn
    probes=["acc"],            # local names assigned in the kernel
    relaunch=relaunch,
)
def run(a, b):
    return launch(a, b)
```

---

## Install

```bash
pip install triton-blackhole
```

Optional (Linux / WSL2 / Colab with NVIDIA):

```bash
pip install triton-blackhole[triton]
# On Colab, pin Triton to whatever torch wants, e.g.:
# pip install "triton==3.6.0"
```

From source:

```bash
pip install -e ".[dev]"
```

---

## Platforms

| Piece | Native Windows | WSL2 + NVIDIA | Linux + NVIDIA | Colab GPU |
|-------|----------------|---------------|----------------|-----------|
| Debugger (compare / bisect / classify) | ✅ | ✅ | ✅ | ✅ |
| Live Triton kernels | ❌ | ✅ | ✅ | ✅ |

No NVIDIA laptop? Use the Colab badge above.

---

## Tile bisection (real kernels)

```python
from triton_blackhole import bisect_tiles

def launch(pid_lo, pid_hi):
    return run_triton_kernel(..., pid_lo=pid_lo, pid_hi=pid_hi)

print(bisect_tiles(launch, torch_ref, num_programs=grid).report())
```

## Stage probes (not `device_print`)

```python
from triton_blackhole.probe import ProbeBank

bank = ProbeBank()
bank.capture("pre_softmax", scores_ref, side="ref")
bank.capture("pre_softmax", scores_tri, side="tri")
print(bank.report())  # first diverging stage
```

## CLI

```bash
triton-blackhole compare triton_out.pt torch_ref.pt --bisect --suggest
```

---

## Why not `TRITON_INTERPRET`?

We never interpret the kernel. We bisect:

1. **Output space** (tensor axes)  
2. **Grid space** (`program_id`)  
3. **Stage space** (named intermediates)  

Same binary as production — bf16, tensor cores, indirect loads included.

---

## Examples

```bash
python examples/demo_verify_drift.py
python examples/demo_verify_ast_triton.py  # needs CUDA + triton
python examples/demo_softmax_drift.py
python examples/demo_tile_bisect.py
python examples/demo_triton_add.py
```

## License

MIT · [brian-mwirigi/triton-blackhole](https://github.com/brian-mwirigi/triton-blackhole)
