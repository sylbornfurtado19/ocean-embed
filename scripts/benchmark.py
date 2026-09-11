#!/usr/bin/env python3
"""OceanEmbed Phase 6 Performance Benchmark Utility.

Measures:
1. Model checkpoint load time (disk to RAM/CPU).
2. Pure neural forward inference latency (OceanEmbed V2 model pass).
3. End-to-end HTTP API request latency (FastAPI parsing + serialization + model forward pass).
4. Memory footprint (Resident Set Size).

Adheres strictly to scientific accuracy:
- Runs at least 10 trials following a warmup iteration.
- Reports minimum, mean, maximum, and standard deviation in milliseconds.
- Strictly separates internal model inference from external HTTP serving overhead.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any

# Ensure project root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np


def get_process_memory_mb() -> float:
    """Return current process Resident Set Size (RSS) in MB."""
    # resource.getrusage ru_maxrss is in bytes on macOS, kilobytes on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return round(usage.ru_maxrss / (1024 * 1024), 2)
    return round(usage.ru_maxrss / 1024, 2)


def run_benchmark(
    checkpoint_path: str = "checkpoints/oceanembed_v2.pt",
    num_requests: int = 15,
) -> dict[str, Any]:
    """Execute rigorous latency and throughput benchmarks."""
    ckpt_file = WORKSPACE_ROOT / checkpoint_path
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    print("\n" + "=" * 70)
    print("🌊 OCEANEMBED PHASE 6 PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_file}")
    print(f"Number of Evaluation Requests: {num_requests}")
    print(f"Operating System: {sys.platform} (CPU execution)")

    # 1. Benchmark Model Load Time
    print("\n[1/3] Benchmarking Checkpoint Loading...")
    t0_load = time.perf_counter()
    from src.inference import OceanInferenceEngine
    engine = OceanInferenceEngine(ckpt_file)
    load_time_ms = (time.perf_counter() - t0_load) * 1000.0
    mem_after_load = get_process_memory_mb()
    print(f"  → Model load time: {load_time_ms:.2f} ms")
    print(f"  → Process memory after load: {mem_after_load:.2f} MB")

    # Coordinates & Dates for diverse testing
    test_cases = [
        (14.0, 88.0, "2023-02-15"),
        (12.5, 85.0, "2023-03-01"),
        (16.0, 92.0, "2023-04-10"),
        (10.0, 90.0, "2023-05-15"),
        (18.5, 87.5, "2023-06-20"),
    ]

    # Warmup Run
    print("\n[2/3] Benchmarking Pure Model Inference Latency...")
    w_lat, w_lon, w_date = test_cases[0]
    _ = engine.predict_from_location_date(w_lat, w_lon, w_date)

    # 2. Pure Model Inference Benchmarks
    inference_latencies: list[float] = []
    for i in range(num_requests):
        lat, lon, dt_str = test_cases[i % len(test_cases)]
        t0 = time.perf_counter()
        _ = engine.predict_from_location_date(lat, lon, dt_str)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        inference_latencies.append(elapsed_ms)

    inf_min = min(inference_latencies)
    inf_mean = statistics.mean(inference_latencies)
    inf_max = max(inference_latencies)
    inf_std = statistics.stdev(inference_latencies) if len(inference_latencies) > 1 else 0.0

    print(f"  → Pure Model Inference Latency ({num_requests} runs):")
    print(f"     Min:  {inf_min:6.2f} ms")
    print(f"     Mean: {inf_mean:6.2f} ms ± {inf_std:.2f} ms")
    print(f"     Max:  {inf_max:6.2f} ms")

    # 3. HTTP API Latency Benchmarks
    print("\n[3/3] Benchmarking Full HTTP API Request Latency (FastAPI)...")
    from fastapi.testclient import TestClient
    from src.api.main import app

    api_latencies: list[float] = []
    with TestClient(app) as client:
        # API warmup
        _ = client.post("/predict", json={"latitude": w_lat, "longitude": w_lon, "date": w_date})

        for i in range(num_requests):
            lat, lon, dt_str = test_cases[i % len(test_cases)]
            t0 = time.perf_counter()
            resp = client.post(
                "/predict",
                json={
                    "latitude": lat,
                    "longitude": lon,
                    "date": dt_str,
                    "data_mode": "synthetic",
                },
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            api_latencies.append(elapsed_ms)

    api_min = min(api_latencies)
    api_mean = statistics.mean(api_latencies)
    api_max = max(api_latencies)
    api_std = statistics.stdev(api_latencies) if len(api_latencies) > 1 else 0.0

    print(f"  → Full HTTP API Request Latency ({num_requests} runs):")
    print(f"     Min:  {api_min:6.2f} ms")
    print(f"     Mean: {api_mean:6.2f} ms ± {api_std:.2f} ms")
    print(f"     Max:  {api_max:6.2f} ms")

    serving_overhead_ms = max(0.0, api_mean - inf_mean)
    print(f"  → Estimated HTTP & Pydantic Overhead: {serving_overhead_ms:.2f} ms")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Model Checkpoint Load Time:       {load_time_ms:.2f} ms")
    print(f"Pure Model Inference (Mean):      {inf_mean:.2f} ms (Min: {inf_min:.2f} ms, Max: {inf_max:.2f} ms)")
    print(f"Full HTTP Request (Mean):         {api_mean:.2f} ms (Min: {api_min:.2f} ms, Max: {api_max:.2f} ms)")
    print(f"Serving Overhead:                 {serving_overhead_ms:.2f} ms")
    print(f"Memory Footprint (RSS):           {mem_after_load:.2f} MB")
    print("=" * 70 + "\n")

    return {
        "model_load_time_ms": round(load_time_ms, 2),
        "inference_latency_ms": {
            "min": round(inf_min, 2),
            "mean": round(inf_mean, 2),
            "max": round(inf_max, 2),
            "std": round(inf_std, 2),
        },
        "http_api_latency_ms": {
            "min": round(api_min, 2),
            "mean": round(api_mean, 2),
            "max": round(api_max, 2),
            "std": round(api_std, 2),
        },
        "serving_overhead_ms": round(serving_overhead_ms, 2),
        "memory_mb": mem_after_load,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OceanEmbed Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/oceanembed_v2.pt")
    parser.add_argument("--requests", type=int, default=15)
    args = parser.parse_args()

    run_benchmark(checkpoint_path=args.checkpoint, num_requests=args.requests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
