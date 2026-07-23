#!/usr/bin/env bash
set -euo pipefail
source ~/tbh-venv/bin/activate
cd /mnt/c/Users/Nesh/Desktop/triton

echo "=== versions ==="
python - <<'PY'
import torch, triton, triton_blackhole as tb
print("torch", torch.__version__)
print("triton", triton.__version__)
print("tb", tb.__version__)
print("cuda", torch.cuda.is_available())
try:
    import subprocess
    print(subprocess.getoutput("nvidia-smi -L") or "nvidia-smi: unavailable")
except Exception as e:
    print("nvidia-smi error", e)
PY

echo ""
echo "=== pytest ==="
python -m pytest tests -q

echo ""
echo "=== demo_softmax_drift ==="
python examples/demo_softmax_drift.py

echo ""
echo "=== demo_tile_bisect ==="
python examples/demo_tile_bisect.py

echo ""
echo "=== demo_triton_add ==="
python examples/demo_triton_add.py

echo ""
echo "=== ALL WSL CHECKS DONE ==="
