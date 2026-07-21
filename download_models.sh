#!/bin/bash
# Download Qwen3-VL-8B-Instruct model (runs once, ~16GB)
# Usage: bash download_models.sh
set -e

echo "=== Downloading Qwen3-VL-8B-Instruct ==="
echo "This will download ~16GB of model weights."
echo ""

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 -c "
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_name = 'Qwen/Qwen3-VL-8B-Instruct'
print(f'Downloading {model_name}...')
print('This may take several minutes...')

# Download processor
print('Downloading processor...')
AutoProcessor.from_pretrained(model_name)

# Download model
print('Downloading model...')
AutoModelForImageTextToText.from_pretrained(
    model_name, 
    dtype='auto',
    device_map='auto',
)

print('Qwen3-VL-8B-Instruct downloaded successfully!')
print('Model is ready for use with vLLM or direct inference.')
"

echo ""
echo "=== Done ==="
echo ""
echo "To use with vLLM server:"
echo "  vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 4096"
echo ""
echo "To use directly:"
echo "  from transformers import AutoModelForImageTextToText, AutoProcessor"
echo "  model = AutoModelForImageTextToText.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')"
