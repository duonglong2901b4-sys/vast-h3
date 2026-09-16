#!/usr/bin/env bash
set -Eeuo pipefail

# H3 Ultra Speed Singularity installer v2 for Vast.ai ComfyUI.
# Runs the existing H3 installer, then installs a verified prebuilt
# SageAttention 2.2.0 wheel for Python 3.12 + PyTorch 2.10 + CUDA 12.8.

LOG=/workspace/h3_ultraspeed_provision.log
mkdir -p /workspace
exec > >(tee -a "$LOG") 2>&1
trap 'echo "[ERROR] line $LINENO: $BASH_COMMAND"' ERR

BASE_INSTALLER="https://raw.githubusercontent.com/duonglong2901b4-sys/vast-h3/main/install_h3_ultraspeed.sh"
PY=/venv/main/bin/python
if [ ! -x "$PY" ]; then PY=python3; fi

echo "============================================================"
echo " H3 Ultra Speed v2 provisioning started: $(date -Is)"
echo "============================================================"

echo "[1/3] Running base H3 Ultra Speed installer..."
curl -fsSL "$BASE_INSTALLER" -o /tmp/install_h3_ultraspeed_base.sh
bash /tmp/install_h3_ultraspeed_base.sh

echo "[2/3] Installing verified SageAttention 2.2.0 prebuilt wheel..."
ENV_INFO="$($PY - <<'PY'
import sys, torch
print(f"{sys.version_info.major}.{sys.version_info.minor}|{torch.__version__}|{torch.version.cuda}|{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}")
PY
)"
echo "[env] python|torch|torch_cuda|gpu_arch = $ENV_INFO"

PY_MM="${ENV_INFO%%|*}"
REST="${ENV_INFO#*|}"
TORCH_VER="${REST%%|*}"
REST="${REST#*|}"
TORCH_CUDA="${REST%%|*}"
GPU_ARCH="${REST##*|}"

if [[ "$PY_MM" != "3.12" || "$TORCH_VER" != 2.10* || "$TORCH_CUDA" != "12.8" ]]; then
  echo "[fatal] Prebuilt SageAttention wheel requires Python 3.12 + PyTorch 2.10 + cu128."
  echo "[fatal] Detected: Python $PY_MM | Torch $TORCH_VER | Torch CUDA $TORCH_CUDA | GPU arch $GPU_ARCH"
  exit 20
fi

case "$GPU_ARCH" in
  8.0|8.6|8.9|12.0) ;;
  *)
    echo "[fatal] GPU compute capability $GPU_ARCH is not included in this wheel."
    exit 21
    ;;
esac

WHEEL_URL="https://github.com/thekie/sageattention-wheel/releases/download/2.2.0.post1/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
WHEEL=/tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
EXPECTED_SHA="03573e7c8c9bd338d6a5fe4e7d4bb22a85eefbc12557e1168ba6a4df3052d962"

curl -fL --retry 3 --retry-delay 2 "$WHEEL_URL" -o "$WHEEL"
echo "$EXPECTED_SHA  $WHEEL" | sha256sum -c -

$PY -m pip uninstall -y sageattention || true
$PY -m pip install --no-deps --force-reinstall "$WHEEL"

echo "[smoke-test] Importing SageAttention and launching a tiny CUDA kernel..."
$PY - <<'PY'
import torch
from sageattention import sageattn
q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
y = sageattn(q, q, q, tensor_layout="HND", is_causal=False)
torch.cuda.synchronize()
print("SAGEATTENTION_OK", tuple(y.shape), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

echo "[3/3] Restarting ComfyUI..."
supervisorctl restart comfyui || true

echo "============================================================"
echo " H3 ULTRA SPEED V2: ALL GOOD"
echo " SageAttention 2.2.0 prebuilt wheel installed and GPU-tested."
echo "============================================================"
