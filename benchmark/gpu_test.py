"""
RoboPilot GPU Benchmark
验证 ROCm + PyTorch + GPU 计算全部正常
"""
import torch
import time

device = "cuda"
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")

a = torch.randn(5000, 5000, device=device)
b = torch.randn(5000, 5000, device=device)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    c = a @ b
torch.cuda.synchronize()

elapsed = time.time() - start
print(f"5000x5000 matrix multiply x100: {elapsed:.2f}s")
print(f"Average per op: {elapsed/100*1000:.1f}ms")
print("GPU Benchmark PASSED")
