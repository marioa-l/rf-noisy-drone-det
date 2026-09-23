#!/usr/bin/env bash
# Set up the conda env for the RF experiments.
#
# The machine is aarch64 (Grace) with a single GH200 and driver 550.90.07 (CUDA 12.4),
# so torch comes from the cu124 index, which publishes linux_aarch64 wheels. The
# scientific stack comes from conda-forge, which has good ARM coverage.
#
# Idempotent: safe to re-run. Installs nothing outside the named env.
#
# Usage:  bash setup_env.sh [env_name]      (default: rf)

set -euo pipefail
ENV="${1:-rf}"

echo "=== target env: ${ENV} ==="
conda run -n "${ENV}" python -c "import sys, platform; \
print('python', sys.version.split()[0], '|', platform.machine())"

# --- archive tools -----------------------------------------------------------
# The dataset ships as RAR5. 7-Zip 23+ reads it; libarchive's bsdtar is the backup.
echo
echo "=== archive tools ==="
conda install -n "${ENV}" -y -c conda-forge 7zip libarchive

# --- scientific stack --------------------------------------------------------
echo
echo "=== scientific stack ==="
conda install -n "${ENV}" -y -c conda-forge \
    numpy scipy pandas matplotlib seaborn scikit-learn \
    h5py pyyaml tqdm pillow einops rich

# --- torch for aarch64 + CUDA 12.4 -------------------------------------------
# Driver 550 = CUDA 12.4. Using the matching index avoids relying on minor-version
# forward compatibility.
echo
echo "=== torch (aarch64, cu124) ==="
conda run -n "${ENV}" pip install --upgrade pip
conda run -n "${ENV}" pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# --- vision model zoo --------------------------------------------------------
# timm covers the ResNet / ViT / Swin family used by the RFUAV baseline.
# mmdet/mmcv are deliberately NOT installed: they need compilation and are painful
# on ARM. Only needed if we go after the detection stage.
echo
echo "=== model zoo ==="
conda run -n "${ENV}" pip install timm transformers safetensors tensorboard

# --- verification ------------------------------------------------------------
echo
echo "=== verification ==="
conda run -n "${ENV}" python - <<'PY'
import shutil, subprocess, sys

print("-- archive extractors --")
for tool in ("7z", "7zz", "bsdtar", "unrar"):
    print(f"  {tool:8s} {shutil.which(tool) or 'not found'}")

print("\n-- packages --")
for mod in ("numpy", "scipy", "pandas", "matplotlib", "sklearn", "h5py",
            "PIL", "timm", "transformers", "torch", "torchvision"):
    try:
        m = __import__(mod)
        print(f"  {mod:14s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  {mod:14s} FAILED: {e}")

print("\n-- gpu --")
try:
    import torch
    print(f"  torch {torch.__version__}  cuda={torch.version.cuda}")
    print(f"  is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"  device: {p.name}  sm_{p.major}{p.minor}  {p.total_memory/1e9:.1f} GB")
        print(f"  memory free now: {free/1e9:.1f} GB of {total/1e9:.1f} GB"
              "   (the rest is another user's job)")
        print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
        # Confirm the GPU actually computes, and report achieved throughput.
        import time
        a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            a @ a
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            a @ a
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 10
        print(f"  bf16 matmul 8192^3: {dt*1e3:.1f} ms  "
              f"-> {2*8192**3/dt/1e12:.1f} TFLOP/s")
    else:
        print("  CUDA NOT AVAILABLE - check that the wheel is an aarch64+cu124 build")
        sys.exit(1)
except Exception as e:                                       # noqa: BLE001
    print(f"  torch check failed: {e}")
    sys.exit(1)
PY

echo
echo "=== done ==="
echo "Activate with:  conda activate ${ENV}"
