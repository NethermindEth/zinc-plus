#!/usr/bin/env python3
"""
Run Zip+ depth-1 IPRS benchmarks over F12289 for 10 and 55 polynomials,
then collect the Criterion results and emit a LaTeX table.

Benchmarks executed:
  - zip_plus_commit_10_f12289  with filter IPRS-1-1/4-F12289
  - zip_plus_commit_10_f12289  with filter IPRS-1-1/2-F12289
  - zip_plus_commit_55_f12289  with filter IPRS-1-1/2-F12289
  - zip_plus_commit_55_f12289  with filter IPRS-1-1/4-F12289

Usage:
    python3 scripts/run_depth2_benchmarks.py
    python3 scripts/run_depth2_benchmarks.py --dry-run      # skip running, just read existing results
    python3 scripts/run_depth2_benchmarks.py --output table.tex
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

WORKSPACE = Path(__file__).resolve().parent.parent
CRITERION_DIR = WORKSPACE / "target" / "criterion"

FEATURES = "asm parallel simd unchecked-butterfly"

# Each entry: (bench_name, filter_pattern)
BENCHMARKS = [
    ("zip_plus_commit_10_f12289", "IPRS-1-1/4-F12289"),
    ("zip_plus_commit_10_f12289", "IPRS-1-1/2-F12289"),
    ("zip_plus_commit_55_f12289", "IPRS-1-1/2-F12289"),
    ("zip_plus_commit_55_f12289", "IPRS-1-1/4-F12289"),
]

# Operations benchmarked in each bench binary
OPERATIONS = ["Commit", "Test", "Evaluate", "Verify"]

# Number of variables (num_vars) in each benchmark
NUM_VARS = [6, 7, 8, 9, 10]


def run_benchmarks(dry_run: bool = False) -> None:
    """Execute all cargo bench commands sequentially."""
    for bench_name, filter_pat in BENCHMARKS:
        cmd = [
            "cargo", "bench",
            "--bench", bench_name,
            "--features", FEATURES,
            "--", filter_pat,
        ]
        print(f"\n{'=' * 72}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'=' * 72}\n")
        if dry_run:
            print("  [dry-run] skipped")
            continue
        result = subprocess.run(cmd, cwd=WORKSPACE)
        if result.returncode != 0:
            print(f"WARNING: benchmark exited with code {result.returncode}", file=sys.stderr)


def iprs_label_to_dir_suffix(iprs_label: str) -> str:
    """Convert 'IPRS-2-1/4-F12289' → 'IPRS-2-1_4-F12289' (filesystem safe)."""
    return iprs_label.replace("/", "_")


def collect_results() -> dict:
    """
    Walk Criterion output directories and gather median timings.

    Returns a nested dict:
        results[n_polys][iprs_label][operation][num_vars] = median_ns
    """
    results: dict = {}

    for bench_name, iprs_label in BENCHMARKS:
        # Derive n_polys from bench name (e.g. zip_plus_commit_10_f12289 → 10)
        m = re.search(r"commit_(\d+)_", bench_name)
        assert m, f"Cannot parse n_polys from {bench_name}"
        n_polys = int(m.group(1))

        iprs_dir_suffix = iprs_label_to_dir_suffix(iprs_label)

        for op in OPERATIONS:
            group_dir_name = f"Zip+ {op} F12289 {n_polys} Polys {iprs_dir_suffix}"
            group_path = CRITERION_DIR / group_dir_name

            if not group_path.is_dir():
                print(f"  [missing] {group_dir_name}", file=sys.stderr)
                continue

            for sub in sorted(group_path.iterdir()):
                if not sub.is_dir() or sub.name == "report":
                    continue

                estimates_path = sub / "new" / "estimates.json"
                if not estimates_path.exists():
                    continue

                with open(estimates_path) as f:
                    estimates = json.load(f)

                median_ns = estimates["median"]["point_estimate"]

                # Extract num_vars from the sub-directory name.
                # Pattern: "... matrix=1x<cols>, Eval=..."
                # cols = 2^num_vars for single-row layout
                col_match = re.search(r"matrix=(\d+)x(\d+)", sub.name)
                if not col_match:
                    continue
                rows = int(col_match.group(1))
                cols = int(col_match.group(2))
                poly_size = rows * cols
                num_vars = poly_size.bit_length() - 1  # log2

                results.setdefault(n_polys, {}) \
                       .setdefault(iprs_label, {}) \
                       .setdefault(op, {})[num_vars] = median_ns

    return results


def fmt_time(ns: float) -> str:
    """Format nanoseconds into a human-friendly string for the LaTeX table."""
    if ns < 1_000:
        return f"{ns:.0f} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.1f} \\textmu{{}}s"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def generate_latex(results: dict) -> str:
    """
    Build a LaTeX table with the collected benchmark data.

    Columns: num_vars | Commit | Test | Evaluate | Verify
    Grouped by (n_polys, iprs_label).
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Zip+ Depth-1 IPRS benchmarks over $\mathbb{F}_{12289}$}")
    lines.append(r"\label{tab:depth1-benchmarks}")
    lines.append(r"\begin{tabular}{c|cccc}")
    lines.append(r"\toprule")
    lines.append(r"$\nu$ & Commit & Test & Evaluate & Verify \\")
    lines.append(r"\midrule")

    for n_polys in sorted(results.keys()):
        for iprs_label in sorted(results[n_polys].keys()):
            op_data = results[n_polys][iprs_label]

            # Section header
            rate = iprs_label.split("-")[2]  # e.g. "1/4" from "IPRS-1-1/4-F12289"
            lines.append(r"\multicolumn{5}{c}{\textbf{" +
                         f"{n_polys} polys, depth-1, rate {rate}" +
                         r"}} \\")
            lines.append(r"\midrule")

            for nv in NUM_VARS:
                cells = []
                for op in OPERATIONS:
                    val = op_data.get(op, {}).get(nv)
                    cells.append(fmt_time(val) if val is not None else "--")
                lines.append(f"  {nv} & " + " & ".join(cells) + r" \\")

            lines.append(r"\midrule")

    # Remove the last \midrule and replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run depth-1 IPRS benchmarks and generate a LaTeX table."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip running benchmarks; only collect existing Criterion results.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write LaTeX table to this file (default: print to stdout).",
    )
    args = parser.parse_args()

    run_benchmarks(dry_run=args.dry_run)

    print("\nCollecting Criterion results …")
    results = collect_results()

    if not results:
        print("No results found. Run benchmarks first (without --dry-run).", file=sys.stderr)
        sys.exit(1)

    latex = generate_latex(results)

    if args.output:
        Path(args.output).write_text(latex + "\n")
        print(f"\nLaTeX table written to {args.output}")
    else:
        print("\n" + latex)


if __name__ == "__main__":
    main()
