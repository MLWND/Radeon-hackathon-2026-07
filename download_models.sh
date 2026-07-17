#!/bin/bash
# Download Qwen3-VL-2B-Instruct model (runs once, ~4GB)
# Usage: bash download_models.sh
set -e

echo "=== Downloading Qwen3-VL-2B-Instruct ==="

python3 -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

model_name = 'Qwen/Qwen3-VL-2B-Instruct'
print(f'Downloading {model_name}...')
AutoProcessor.from_pretrained(model_name)
AutoModelForVision2Seq.from_pretrained(
    model_name, 
    dtype=torch.float16,
    device_map='auto',
)
print('Qwen3-VL downloaded and loaded successfully!')
print(f'Device: {model.device}')
"
echo "=== Done ==="
