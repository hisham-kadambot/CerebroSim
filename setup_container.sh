#!/usr/bin/env bash
# One-time set-up inside the PhysicsNeMo container (nvcr.io/nvidia/physicsnemo/physicsnemo:26.08), for a headless
# Linux GPU machine such as an A100. Adds what the image lacks: NVIDIA Warp, uv, a few Python packages and fonts.
# The AI scripts (ai/) then use the container's own PyTorch and PhysicsNeMo (ai/pyproject.toml is for Windows).
#   bash setup_container.sh            (in the repository folder)
set -e
cd "$(dirname "$0")"

if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
    apt-get update -qq && apt-get install -y -qq --no-install-recommends fonts-dejavu-core >/dev/null
fi

PKGS=("warp-lang>=1.18,<1.19" numpy scipy scikit-image matplotlib pillow pyyaml imageio-ffmpeg uv)
python3 -m pip install -q "${PKGS[@]}" || python3 -m pip install -q --break-system-packages "${PKGS[@]}"
python3 -c "import torch_geometric" 2>/dev/null || python3 -m pip install -q torch_geometric \
    || python3 -m pip install -q --break-system-packages torch_geometric

python3 - <<'EOF'
import physicsnemo, torch, warp as wp
wp.config.quiet = True
wp.init()
print("physicsnemo", physicsnemo.__version__, "| torch", torch.__version__, "| warp", wp.__version__)
print("GPU:", torch.cuda.get_device_name(0), "| CUDA available:", torch.cuda.is_available())
EOF
echo "ready: e.g. uv run simulate.py healthy --beats 4, then uv run explore.py (port 8003)"
