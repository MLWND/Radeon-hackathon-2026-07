#!/bin/bash
# RoboPilot — One-click environment recovery for AMD Radeon Cloud
# Usage: bash setup.sh
set -e

echo "=== RoboPilot Environment Setup ==="

# 1. System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt update -qq && sudo apt install -y -qq git curl wget > /dev/null 2>&1

# 2. Python virtual environment
echo "[2/5] Setting up Python environment..."
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate

# 3. Python packages
echo "[3/5] Installing Python packages (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. Download Qwen3-VL model
echo "[4/5] Checking Qwen3-VL model..."
if [ ! -f /root/.config/Ultralytics/yolov8n.pt ]; then
    curl -k -L -o /root/.config/Ultralytics/yolov8n.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt" 2>/dev/null
    echo "  Downloaded YOLOv8n model"
else
    echo "  YOLOv8n model already present"
fi

# 5. Verify
echo "[5/5] Verifying environment..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
import genesis
print(f'  Genesis: {genesis.__version__}')
"
bash scripts/verify_env.sh 2>/dev/null || true

echo ""
echo "=== Setup Complete ==="
echo "Run: source venv/bin/activate"
echo "Then: python src/system/orchestrator_v2.py"
