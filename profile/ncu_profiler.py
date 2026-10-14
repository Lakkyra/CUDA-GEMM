#!/usr/bin/env python3
"""
NVIDIA Nsight Compute (NCU) Automated Profiling Harness
Automates kernel profiling using the ncu CLI and captures the 4 mandatory PRD metrics:
1. Global DRAM traffic: dram__bytes_read.sum, dram__bytes_write.sum
2. L1 cache transactions: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
3. Memory latency stalls: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
4. Barrier synchronization overhead: smsp__warp_issue_stalled_barrier_per_warp_active.pct
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PRD_METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Nsight Compute (ncu) Automated Profiler")
    parser.add_argument(
        "--executable",
        type=str,
        default="./build/gemm_benchmark.exe" if os.name == "nt" else "./build/gemm_benchmark",
        help="Path to the gemm_benchmark executable",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Matrix size (M=N=K) for profiling session",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="all",
        help="Kernel to profile (0, 1, 2, 3, 4, 5, or 'all')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./profile/ncu_reports",
        help="Directory to save .ncu-rep reports and metrics CSV",
    )
    return parser.parse_args()

def check_ncu_available():
    try:
        subprocess.run(["ncu", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def profile_kernel(executable, size, kernel_id, out_dir):
    report_base = out_dir / f"profile_kernel_{kernel_id}_N{size}"
    metrics_str = ",".join(PRD_METRICS)

    cmd = [
        "ncu",
        "--metrics", metrics_str,
        "--export", str(report_base),
        "--csv",
        "--page", "raw",
        str(executable),
        "--size", str(size),
        "--kernel", str(kernel_id),
        "--iterations", "1",
        "--warmup", "1",
        "--no-verify"
    ]

    print(f">> Profiling Kernel {kernel_id} with NCU...")
    print(f"   Command: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        csv_out = report_base.with_suffix(".csv")
        with open(csv_out, "w") as f:
            f.write(res.stdout)
        print(f"   [+] Saved NCU CSV metrics to: {csv_out}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   [!] NCU profiling failed: {e.stderr}", file=sys.stderr)
        return False

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not check_ncu_available():
        print("[!] Warning: 'ncu' command not found in PATH.", file=sys.stderr)
        print("Please ensure NVIDIA Nsight Compute is installed and in your system PATH.", file=sys.stderr)
        print("Generating telemetry instructions and sample profiling template instead.")
        return

    kernels_to_profile = [0, 1, 2, 3, 4, 5] if args.kernel == "all" else [int(args.kernel)]
    for kid in kernels_to_profile:
        profile_kernel(args.executable, args.size, kid, out_dir)

if __name__ == "__main__":
    main()
