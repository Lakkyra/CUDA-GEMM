#!/usr/bin/env python3
"""
Automated GEMM Benchmark Runner
Executes square, rectangular, and tall/skinny matrix multiplication benchmarks
across all implemented CUDA kernels and NVIDIA cuBLAS.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Automated CUDA GEMM Benchmark Runner")
    parser.add_argument(
        "--executable",
        type=str,
        default="./build/gemm_benchmark.exe" if os.name == "nt" else "./build/gemm_benchmark",
        help="Path to the gemm_benchmark executable",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Square matrix sizes (N=M=K) to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for timing measurement",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmarks",
        help="Directory to save benchmark output JSON and CSV",
    )
    return parser.parse_args()

def run_single_benchmark(executable, M, N, K, iterations, warmup):
    cmd = [
        str(executable),
        "--dim", str(M), str(N), str(K),
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--no-verify",  # Skip verification during large benchmark sweeps
        "--json"
    ]
    print(f">> Running benchmark for {M}x{N}x{K}...", flush=True)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse JSON from stdout (ignoring device info header if any)
        stdout = res.stdout
        json_start = stdout.find("{")
        if json_start == -1:
            raise ValueError(f"No JSON found in output:\n{stdout}")
        json_str = stdout[json_start:]
        return json.loads(json_str)
    except Exception as e:
        print(f"Error running benchmark {cmd}: {e}", file=sys.stderr)
        return None

def main():
    args = parse_args()
    exe_path = Path(args.executable)

    # Check if executable exists or search in build/Release
    if not exe_path.exists():
        fallback_candidates = [
            Path("build/Release/gemm_benchmark.exe"),
            Path("build/gemm_benchmark.exe"),
            Path("build/gemm_benchmark"),
        ]
        found = False
        for cand in fallback_candidates:
            if cand.exists():
                exe_path = cand
                found = True
                break
        if not found:
            print(f"[!] Executable not found at {args.executable} or fallback paths.", file=sys.stderr)
            print("Please build the project first using CMake:\n  cmake -B build && cmake --build build --config Release", file=sys.stderr)
            # Create synthetic mock run for validation/demonstration if binary not built yet
            print("[i] Writing default matrix sizes to run configuration.")

    results = []

    # 1. Square matrices (Powers of 2)
    for sz in args.sizes:
        if exe_path.exists():
            data = run_single_benchmark(exe_path, sz, sz, sz, args.iterations, args.warmup)
            if data:
                results.append(data)

    # 2. Non-power of two test
    for non_pow in [768, 1536]:
        if exe_path.exists():
            data = run_single_benchmark(exe_path, non_pow, non_pow, non_pow, args.iterations, args.warmup)
            if data:
                results.append(data)

    # 3. Tall / Skinny test: [4096 x 512 x 512] and [512 x 4096 x 512]
    tall_skinny_configs = [(4096, 512, 512), (512, 4096, 512)]
    for M, N, K in tall_skinny_configs:
        if exe_path.exists():
            data = run_single_benchmark(exe_path, M, N, K, args.iterations, args.warmup)
            if data:
                results.append(data)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved benchmark JSON to: {json_path}")

    # Generate CSV summary
    csv_path = out_dir / "benchmark_results.csv"
    with open(csv_path, "w") as f:
        f.write("M,N,K,Kernel_ID,Kernel_Tag,Kernel_Name,Time_ms,GFLOPS,Speedup_vs_Naive,Pct_cuBLAS\n")
        for entry in results:
            m, n, k = entry["matrix"]["M"], entry["matrix"]["N"], entry["matrix"]["K"]
            for kern in entry["kernels"]:
                f.write(f"{m},{n},{k},{kern['id']},{kern['tag']},{kern['name']},{kern['time_ms']:.4f},{kern['gflops']:.2f},{kern['speedup_vs_naive']:.2f},{kern['pct_cublas']:.2f}\n")
    print(f"[+] Saved benchmark CSV to: {csv_path}")

if __name__ == "__main__":
    main()
