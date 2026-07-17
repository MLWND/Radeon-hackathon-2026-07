"""
Module 14: Benchmark
Performance metrics for all pipeline stages.
"""
import time
import torch
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class LatencyRecord:
    name: str
    times: List[float] = field(default_factory=list)

    def record(self, ms: float):
        self.times.append(ms)

    def avg(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0

    def p95(self) -> float:
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def min_ms(self) -> float:
        return min(self.times) if self.times else 0

    def max_ms(self) -> float:
        return max(self.times) if self.times else 0


class Benchmark:
    def __init__(self):
        self.records = {}
        self.gpu_start = None

    def start(self, name: str):
        self.gpu_start = time.time()
        return name

    def end(self, name: str):
        if self.gpu_start:
            elapsed = (time.time() - self.gpu_start) * 1000
            if name not in self.records:
                self.records[name] = LatencyRecord(name)
            self.records[name].record(elapsed)
            self.gpu_start = None
            return elapsed
        return 0

    def record(self, name: str, ms: float):
        if name not in self.records:
            self.records[name] = LatencyRecord(name)
        self.records[name].record(ms)

    def summary(self) -> Dict:
        return {
            name: {
                "avg_ms": rec.avg(),
                "p95_ms": rec.p95(),
                "min_ms": rec.min_ms(),
                "max_ms": rec.max_ms(),
                "count": len(rec.times),
            }
            for name, rec in self.records.items()
        }

    def gpu_info(self) -> Dict:
        if torch.cuda.is_available():
            return {
                "device": torch.cuda.get_device_name(0),
                "memory_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
                "memory_used_mb": torch.cuda.memory_allocated(0) / 1024**2,
            }
        return {"device": "CPU"}

    def print_report(self):
        print("\n" + "=" * 60)
        print("  Benchmark Report")
        print("=" * 60)

        gpu = self.gpu_info()
        print(f"  Device: {gpu['device']}")
        if "memory_total_mb" in gpu:
            print(f"  VRAM: {gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB")

        print(f"\n  {'Module':<25} {'Avg (ms)':<12} {'P95 (ms)':<12} {'Count':<8}")
        print("  " + "-" * 57)

        for name, rec in self.records.items():
            print(f"  {name:<25} {rec.avg():<12.1f} {rec.p95():<12.1f} {len(rec.times):<8}")

        total = sum(rec.avg() for rec in self.records.values())
        print(f"\n  {'Total Pipeline':<25} {total:<12.1f}")
        print("=" * 60)
