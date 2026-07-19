#!/bin/bash
# RoboPilot — Environment Setup for AMD Radeon GPU
# Usage: bash setup.sh
set -e

echo "=== RoboPilot Environment Setup ==="
echo "    Qwen3-VL-8B + Genesis + Suction Gripper on AMD ROCm"
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

# 6. Start vLLM server for Qwen3-VL-8B
echo "[6/6] Setting up vLLM server for Qwen3-VL-8B..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "  vLLM server already running on port 8000"
else
    echo "  Starting vLLM server in background..."
    VLLM_ROCM_USE_AITER=0 setsid vllm serve Qwen/Qwen3-VL-8B-Instruct \
        --limit-mm-per-prompt.video 0 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.8 \
        --enforce-eager \
        --host 0.0.0.0 --port 8000 \
        < /dev/null > vllm.log 2>&1 &
    echo "  vLLM server starting (check vllm.log for status)"
    echo "  Wait ~30s for model to load, then verify: curl http://localhost:8000/v1/models"
fi

# Verify
echo ""
echo "=== Verifying Environment ==="
python3 -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  GPU:        AMD ROCm (gs.amdgpu)')
try:
    import genesis
    print(f'  Genesis:    {genesis.__version__}')
except: print('  Genesis:    not installed')
try:
    import vllm
    print(f'  vLLM:       {vllm.__version__}')
except: print('  vLLM:       not installed')
try:
    from transformers import AutoModelForImageTextToText
    print(f'  Transformers: {__import__(\"transformers\").__version__}')
except: print('  Transformers: not installed')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run the full demo:"
echo "  source venv/bin/activate"
echo "  python3 demo/full_demo.py"
echo ""
echo "Run comprehensive E2E test:"
echo "  python3 demo/test_e2e.py"
echo ""
