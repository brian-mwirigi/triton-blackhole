# Adoption kit

## 1. Publish to PyPI (required for `pip install triton-blackhole`)

### Option A — Trusted Publishing (recommended)

1. Go to https://pypi.org/manage/account/publishing/ (create account if needed)
2. Add a pending publisher:
   - **PyPI project name:** `triton-blackhole`
   - **Owner:** `brian-mwirigi`
   - **Repository:** `triton-blackhole`
   - **Workflow:** `publish.yml`
   - **Environment:** (leave empty) or `pypi`
3. On GitHub: **Releases → Create a new release → tag `v0.1.0`**
4. The `Publish to PyPI` Action uploads the wheel automatically

### Option B — API token

```bash
# WSL or Linux
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TOKEN_HERE
bash scripts/publish.sh
```

---

## 2. Share links (copy-paste)

**Colab (primary CTA):**  
https://colab.research.google.com/github/brian-mwirigi/triton-blackhole/blob/main/notebooks/colab_killer_demo.ipynb

**Repo:**  
https://github.com/brian-mwirigi/triton-blackhole

**PyPI (after publish):**  
https://pypi.org/project/triton-blackhole/

---

## 3. Social posts

### X / Twitter

> `torch.allclose` failed on your Triton kernel. Now what?
>
> triton-blackhole bisects the tensor / program_id / fusion stage and tells you if it's fp16 drift or a real bug — without device_print or TRITON_INTERPRET.
>
> 2‑min Colab: https://colab.research.google.com/github/brian-mwirigi/triton-blackhole/blob/main/notebooks/colab_killer_demo.ipynb
>
> `pip install triton-blackhole`

### LinkedIn

> Shipping custom Triton kernels for LLM work means living in floating-point hell: allclose fails, device_print floods the console, and TRITON_INTERPRET often can't even run modern bf16 kernels.
>
> I open-sourced **triton-blackhole** — a deterministic numerical bisection debugger that localizes divergence (indices, tiles, stages) on the real compiled kernel.
>
> Try the Colab demo (GPU runtime):  
> https://colab.research.google.com/github/brian-mwirigi/triton-blackhole/blob/main/notebooks/colab_killer_demo.ipynb
>
> `pip install triton-blackhole`  
> GitHub: https://github.com/brian-mwirigi/triton-blackhole

### Reddit (r/MachineLearning — “Project” flair)

**Title:** [P] triton-blackhole — localize Triton vs PyTorch numerical drift without device_print / TRITON_INTERPRET

**Body:**

When a fused Triton kernel fails `torch.allclose` against a PyTorch reference, you usually get a boolean and a weekend of pain. `tl.device_print` is unusable at scale; `TRITON_INTERPRET` lacks bf16 and breaks on indirect loads.

**triton-blackhole** treats this as a search problem:

- rich compare (hotspots + bf16 tolerances)
- spatial bisection → minimal failing region
- program_id tile bisection on the *compiled* kernel
- drift classification (localized bug vs reduction-order noise)

Colab demo: https://colab.research.google.com/github/brian-mwirigi/triton-blackhole/blob/main/notebooks/colab_killer_demo.ipynb  
GitHub: https://github.com/brian-mwirigi/triton-blackhole  
`pip install triton-blackhole`

Feedback from people writing attention / GEMM kernels especially welcome.

### Show HN

**Title:** Show HN: triton-blackhole – debug Triton FP drift without device_print  
**URL:** https://github.com/brian-mwirigi/triton-blackhole

---

## 4. Repo polish checklist

- [ ] PyPI 0.1.0 live
- [ ] Colab killer demo runs in a fresh runtime
- [ ] Post X + LinkedIn same day
- [ ] Reddit / Show HN within 48h (don't spam)
- [ ] Pin the Colab link in the GitHub repo About / website field
