#!/usr/bin/env bash
# Build and upload triton-blackhole to PyPI.
#
# One-time setup:
#   1. https://pypi.org/manage/account/token/  → create API token
#   2. export TWINE_USERNAME=__token__
#   3. export TWINE_PASSWORD=pypi-AgEIcHl...   # your token
#
# Or use Trusted Publishing (recommended): configure the GitHub repo on PyPI,
# then create a GitHub Release — .github/workflows/publish.yml uploads automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -U build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
echo "Published. Verify: https://pypi.org/project/triton-blackhole/"
