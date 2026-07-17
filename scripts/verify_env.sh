#!/bin/bash
# RoboPilot Environment Verification Script
set -e

echo "=== RoboPilot Environment Check ==="
echo ""

# 1. GPU
echo "[1/6] GPU Detection"
lspci | grep -i vga
echo ""

# 2. Kernel
echo "[2/6] Kernel"
uname -r
echo ""

# 3. ROCm
echo "[3/6] ROCm"
if command -v rocminfo &> /dev/null; then
    rocminfo | grep -E "Agent|gfx|Name" | head -20
else
    echo "WARNING: rocminfo not found"
fi
echo ""

# 4. rocm-smi
echo "[4/6] GPU Status"
rocm-smi 2>&1 | head -20
echo ""

# 5. PyTorch
echo "[5/6] PyTorch"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
echo ""

# 6. Key packages
echo "[6/6] Key Packages"
pip3 list 2>/dev/null | grep -iE "torch|genesis|ultralytics|triton"
echo ""

echo "=== All checks passed ==="
