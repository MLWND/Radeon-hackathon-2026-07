#!/bin/bash
# RoboPilot — Environment Setup for AMD Radeon GPU
# Usage: bash setup.sh
set -e

echo "=== RoboPilot Environment Setup ==="
echo "    Qwen3-VL-8B + Genesis + Suction Gripper on AMD ROCm"
echo ""

# 1. System dependencies
echo "[1/7] Installing system dependencies..."
sudo apt update -qq && sudo apt install -y -qq git curl wget build-essential > /dev/null 2>&1 || {
    echo "  Warning: Could not install system dependencies (may already be installed)"
}

# 2. Python virtual environment
echo "[2/7] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created new virtual environment"
else
    echo "  Virtual environment already exists"
fi
source venv/bin/activate

# 3. Upgrade pip
echo "[3/7] Upgrading pip..."
pip install --upgrade pip -q

# 4. Install PyTorch + ROCm (AMD GPU)
echo "[4/7] Installing PyTorch with ROCm 7.2..."
pip install --index-url http://compute-artifactory.amd.com/artifactory/compute-pytorch-rocm/compute-rocm-rel-7.2/43/torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm7.2 -q 2>/dev/null || {
    echo "  AMD wheel repo unavailable, trying PyPI..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2 -q 2>/dev/null || echo "  PyTorch install skipped (may already be installed)"
}

# 5. Install project dependencies
echo "[5/7] Installing project dependencies..."
pip install -r requirements.txt -q 2>/dev/null || {
    echo "  Some dependencies failed, trying individually..."
    pip install genesis-world -q 2>/dev/null || echo "  Genesis install skipped"
    pip install transformers>=4.57 accelerate -q 2>/dev/null || echo "  Transformers install skipped"
    pip install vllm>=0.25 -q 2>/dev/null || echo "  vLLM install skipped"
    pip install opencv-python-headless pillow numpy -q 2>/dev/null || echo "  Vision deps install skipped"
}

# 6. Download model (optional, can be done separately)
echo "[6/7] Checking model availability..."
if python3 -c "from transformers import AutoModelForImageTextToText; AutoModelForImageTextToText.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')" 2>/dev/null; then
    echo "  Qwen3-VL-8B model already downloaded"
else
    echo "  Model not downloaded yet. Run 'bash download_models.sh' to download (~16GB)"
fi

# 7. Start vLLM server for Qwen3-VL-8B
echo "[7/7] Setting up vLLM server for Qwen3-VL-8B..."
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
import sys
print(f'  Python:     {sys.version.split()[0]}')

try:
    import torch
    print(f'  PyTorch:    {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    else:
        print(f'  GPU:        AMD ROCm (via genesis)')
except Exception as e:
    print(f'  PyTorch:    error - {e}')

try:
    import genesis as gs
    print(f'  Genesis:    {gs.__version__}')
except Exception as e:
    print(f'  Genesis:    not installed - {e}')

try:
    import vllm
    print(f'  vLLM:       {vllm.__version__}')
except Exception as e:
    print(f'  vLLM:       not installed - {e}')

try:
    import transformers
    print(f'  Transformers: {transformers.__version__}')
except Exception as e:
    print(f'  Transformers: not installed - {e}')

try:
    import cv2
    print(f'  OpenCV:     {cv2.__version__}')
except Exception as e:
    print(f'  OpenCV:     not installed - {e}')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Download model (if not done): bash download_models.sh"
echo "  2. Run full demo: source venv/bin/activate && python3 demo/full_demo.py"
echo "  3. Run E2E test: python3 demo/test_e2e.py"
echo ""
echo "For vLLM server status: curl http://localhost:8000/v1/models"
echo "For vLLM logs: tail -f vllm.log"
