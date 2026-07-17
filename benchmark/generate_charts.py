"""
Benchmark Chart Generator — Paper-quality visualizations.
Generates: latency breakdown, parallel scaling, pipeline flow
"""
import json
import os

# ── Latency Breakdown Chart ──────────────────────────────────

def generate_latency_chart():
    """Generate latency breakdown as ASCII art."""
    stages = [
        ("Task Planner", 0.2),
        ("Scene Memory", 0.5),
        ("IK Solver", 5.0),
        ("Genesis Exec", 1592.0),
        ("Camera Render", 317.0),
        ("Camera Verify", 8.0),
    ]

    total = sum(s[1] for s in stages)
    max_bar = 40

    print("\n" + "=" * 60)
    print("  Latency Breakdown (ms)")
    print("=" * 60)

    for name, ms in stages:
        bar_len = int((ms / total) * max_bar)
        bar = "█" * max(bar_len, 1)
        pct = (ms / total) * 100
        print(f"  {name:<16} {bar:<42} {ms:>7.1f}ms ({pct:>5.1f}%)")

    print(f"  {'─' * 56}")
    print(f"  {'Total':<16} {'':<42} {total:>7.1f}ms")
    print("=" * 60)


def generate_parallel_chart():
    """Generate parallel scaling chart as ASCII art."""
    data = [
        (1, 190),
        (4, 180),
        (8, 170),
        (16, 155),
    ]

    max_fps = 200
    max_bar = 30

    print("\n" + "=" * 60)
    print("  Parallel Simulation Scaling (FPS)")
    print("=" * 60)

    for envs, fps in data:
        bar_len = int((fps / max_fps) * max_bar)
        bar = "█" * bar_len
        print(f"  {envs:>3} envs  {bar:<32} {fps:>5} FPS")

    print("=" * 60)


def generate_pipeline_flow():
    """Generate pipeline flow diagram."""
    print("\n" + "=" * 60)
    print("  Pipeline Flow")
    print("=" * 60)
    print("""
  User: "Pick up the red cup"
         │
         ▼
  ┌──────────────────┐
  │   Task Planner   │  0.2ms
  │   pick → place   │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Scene Memory    │  0.5ms
  │  cup@[0.19,0.41] │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   IK Solver      │  5ms
  │   joints → xyz   │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Genesis (GPU)   │  1592ms
  │  Franka Panda    │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Camera Verify    │  8ms
  │ success: true    │
  └──────────────────┘
    """)
    print("=" * 60)


def generate_gpu_info():
    """Generate GPU information chart."""
    print("\n" + "=" * 60)
    print("  AMD GPU Information")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem_total = props.total_memory / 1024**3
            mem_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  Device:      {name}")
            print(f"  VRAM Total:  {mem_total:.1f} GB")
            print(f"  VRAM Used:   {mem_used:.1f} GB")
            print(f"  Compute:     {props.major}.{props.minor}")
            print(f"  SMs:         {props.multi_processor_count}")
        else:
            print("  No CUDA GPU available")
    except Exception as e:
        print(f"  Error: {e}")

    print("=" * 60)


def main():
    print("\n" + "#" * 60)
    print("  RoboPilot Benchmark Report")
    print("  AMD Radeon PRO W7900D + ROCm 7.2")
    print("#" * 60)

    generate_latency_chart()
    generate_parallel_chart()
    generate_pipeline_flow()
    generate_gpu_info()


if __name__ == "__main__":
    main()
