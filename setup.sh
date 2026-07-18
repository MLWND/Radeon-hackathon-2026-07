#!/bin/bash
# RoboPilot — Environment Setup for AMD Radeon GPU
# Usage: bash setup.sh
set -e

echo "=== RoboPilot Environment Setup ==="
echo "    Qwen3-VL + Genesis + Suction Gripper on AMD ROCm"
echo ""

# 1. System dependencies
echo "[1/6] Installing system dependencies..."
sudo apt update -qq && sudo apt install -y -qq git curl wget build-essential > /dev/null 2>&1

# 2. Python virtual environment
echo "[2/6] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 3. Upgrade pip
echo "[3/6] Upgrading pip..."
pip install --upgrade pip -q

# 4. Install PyTorch + ROCm (AMD GPU)
echo "[4/6] Installing PyTorch with ROCm 7.2..."
pip install --index-url http://compute-artifactory.amd.com/artifactory/compute-pytorch-rocm/compute-rocm-rel-7.2/43/torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm7.2 -q 2>/dev/null || {
    echo "  AMD wheel repo unavailable, trying PyPI..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2 -q 2>/dev/null || echo "  PyTorch install skipped (may already be installed)"
}

# 5. Install project dependencies
echo "[5/6] Installing project dependencies..."
pip install -r requirements.txt -q 2>/dev/null || echo "  Some dependencies skipped"

# 6. Pre-download Qwen3-VL model
echo "[6/6] Checking Qwen3-VL model..."
python3 -c "
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print('  Downloading Qwen3-VL-2B-Instruct...')
AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')
Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-2B-Instruct', torch_dtype='auto')
print('  Qwen3-VL model ready')
" 2>/dev/null || echo "  Qwen3-VL download skipped (will download on first run)"

# Verify
echo ""
echo "=== Verifying Environment ==="
python3 -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
try:
    import genesis
    print(f'  Genesis:    {genesis.__version__}')
except: print('  Genesis:    not installed')
try:
    from transformers import Qwen3VLForConditionalGeneration
    print(f'  Qwen3-VL:   native class available')
except: print('  Qwen3-VL:   class not available')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run the full demo:"
echo "  source venv/bin/activate"
echo "  python3 demo/full_demo.py"
echo ""
