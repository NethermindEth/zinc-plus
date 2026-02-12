#!/usr/bin/env python3
"""
Benchmark the full PCS pipeline (Commit, Test, Verify) for Zip+.

This script runs the "PCS Pipeline Suite" criterion benchmarks and collects
the results into a LaTeX table.

The benchmarks use BPoly<31> evaluations with IPRS codes at rate 1/4 over F3329.

    Row length   Config                    poly_size (2^P)
    ──────────   ──────                    ───────────────
    2^4  (16)    F3329 R4B2 D=1 (rate 1/4) 2^8  = 256
    2^4  (16)    F3329 R4B2 D=1 (rate 1/4) 2^9  = 512
    2^5  (32)    F3329 R4B4 D=1 (rate 1/4) 2^10 = 1024
    2^5  (32)    F3329 R4B4 D=1 (rate 1/4) 2^11 = 2048
    2^6  (64)    F3329 R4B8 D=1 (rate 1/4) 2^12 = 4096

Usage:
    python3 scripts/bench_pcs_pipeline.py
    python3 scripts/bench_pcs_pipeline.py --no-run --input bench_output.txt
    python3 scripts/bench_pcs_pipeline.py --output pcs_table.tex
"""

import argparse
import math
import os
import re
import subprocess
import sys

# ── Configuration ──────────────────────────────────────────────────────────────

BENCH_FILTER = "PCS Pipeline Suite"
CARGO_BENCH_CMD = [
    "cargo", "bench",
    "--bench", "zip_plus_benches",
    "--features", "parallel asm simd",
    "--", BENCH_FILTER,
]

# Expected polynomial size exponents (P = 8 through 12, rate 1/4 IPRS codes)
EXPECTED_POLY_EXPS = [8, 9, 10, 11, 12]

# The three phases we benchmark
PHASES = ["Commit", "Test", "Verify"]

# Map poly exponent → (row_len, config description)
POLY_EXP_CONFIG = {
    8:  (16,  "R4B2 D=1"),
    9:  (16,  "R4B2 D=1"),
    10: (32,  "R4B4 D=1"),
    11: (32,  "R4B4 D=1"),
    12: (64,  "R4B8 D=1"),
}


def field_for_poly_exp(p: int) -> str:
    """Map polynomial size exponent to the field label.  All rate-1/4 configs use F3329."""
    return r"$\mathbb{F}_{3329}$"


# ── Parsing ────────────────────────────────────────────────────────────────────

# Matches benchmark name lines like:
#   Commit: Eval=BinaryPoly<32>, Cw=DensePolynomial<i128, 32>, Comb=..., poly_size=2^12
#   Test: Eval=..., poly_size=2^14
#   Verify: Eval=..., poly_size=2^16, modulus=(256 bits)
_RE_PHASE_NAME = re.compile(
    r"(?P<phase>Commit|Test|Verify):\s+Eval=.*?poly_size=2\^(?P<exp>\d+)"
)

# Criterion's time output line
_RE_TIME = re.compile(
    r"time:\s*\[([0-9.]+)\s*(\S+)\s+([0-9.]+)\s*(\S+)\s+([0-9.]+)\s*(\S+)\]"
)


def _parse_time_us(value: str, unit: str) -> float:
    """Convert a criterion time value to microseconds."""
    v = float(value)
    unit = unit.replace("µ", "u")  # normalise µs
    if unit == "ns":
        return v / 1000.0
    elif unit == "us":
        return v
    elif unit == "ms":
        return v * 1000.0
    elif unit == "s":
        return v * 1_000_000.0
    else:
        raise ValueError(f"Unknown time unit: {unit!r}")


def _format_time(us: float) -> str:
    """Format microseconds into a human-friendly string for the table."""
    if us < 1000:
        return f"{us:.1f}\\,\\textmu s"
    elif us < 1_000_000:
        return f"{us / 1000:.2f}\\,ms"
    else:
        return f"{us / 1_000_000:.2f}\\,s"


def parse_criterion_output(
    text: str,
) -> dict[tuple[str, int], tuple[float, float, float]]:
    """
    Parse criterion benchmark output for the PCS pipeline suite.

    Returns a dict mapping (phase, poly_exp) -> (low_us, median_us, high_us).
    phase is one of "Commit", "Test", "Verify".
    poly_exp is the exponent P where poly_size = 2^P.
    """
    results: dict[tuple[str, int], tuple[float, float, float]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m_name = _RE_PHASE_NAME.search(line)
        if m_name:
            phase = m_name.group("phase")
            poly_exp = int(m_name.group("exp"))

            # Look ahead for the time: line
            for j in range(i + 1, min(i + 5, len(lines))):
                m_time = _RE_TIME.search(lines[j])
                if m_time:
                    low = _parse_time_us(m_time.group(1), m_time.group(2))
                    med = _parse_time_us(m_time.group(3), m_time.group(4))
                    high = _parse_time_us(m_time.group(5), m_time.group(6))
                    results[(phase, poly_exp)] = (low, med, high)
                    i = j
                    break
        i += 1
    return results


# ── LaTeX generation ───────────────────────────────────────────────────────────


def generate_latex_table(
    results: dict[tuple[str, int], tuple[float, float, float]],
) -> str:
    """Generate a LaTeX table from the parsed results."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{PCS pipeline benchmark for BPoly\textlangle 31\textrangle{} with IPRS rate 1/4 (parallel+asm+simd).}",
        r"\label{tab:pcs-pipeline-suite}",
        r"\begin{tabular}{r r r l r r r}",
        r"\toprule",
        (
            r"\textbf{Row len} & $\boldsymbol{2^k}$ & $\boldsymbol{2^P}$ & \textbf{Field}"
            r" & \textbf{Commit} & \textbf{Test} & \textbf{Verify} \\"
        ),
        r"\midrule",
    ]

    for poly_exp in EXPECTED_POLY_EXPS:
        row_len, _config = POLY_EXP_CONFIG[poly_exp]
        row_exp = int(math.log2(row_len))
        field = field_for_poly_exp(poly_exp)

        phase_strs = []
        for phase in PHASES:
            key = (phase, poly_exp)
            if key in results:
                _, med, _ = results[key]
                phase_strs.append(_format_time(med))
            else:
                phase_strs.append("---")

        lines.append(
            f"{row_len:>7} & $2^{{{row_exp}}}$ & $2^{{{poly_exp}}}$"
            f" & {field}"
            f" & {phase_strs[0]} & {phase_strs[1]} & {phase_strs[2]} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run PCS pipeline (Commit/Test/Verify) benchmarks and produce a LaTeX table."
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running cargo bench; read from --input instead.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="File containing previous cargo bench output (used with --no-run).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the LaTeX table to this file (default: stdout).",
    )
    args = parser.parse_args()

    # ── Collect benchmark output ──────────────────────────────────────────
    if args.no_run:
        if args.input is None:
            print("Error: --no-run requires --input <file>", file=sys.stderr)
            sys.exit(1)
        with open(args.input) as f:
            bench_output = f.read()
        print(f"Read {len(bench_output)} bytes from {args.input}")
    else:
        # Resolve workspace root (script lives in scripts/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace = os.path.dirname(script_dir)

        print(f"Running: {' '.join(CARGO_BENCH_CMD)}")
        print(f"Working directory: {workspace}")
        print("This may take several minutes...\n")

        proc = subprocess.run(
            CARGO_BENCH_CMD,
            cwd=workspace,
            capture_output=True,
            text=True,
        )

        bench_output = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0:
            print("cargo bench failed:", file=sys.stderr)
            print(bench_output, file=sys.stderr)
            sys.exit(proc.returncode)

        print(bench_output)

    # ── Parse and generate table ──────────────────────────────────────────
    results = parse_criterion_output(bench_output)
    print(f"\nParsed {len(results)} benchmark results.")

    if not results:
        print(
            "Warning: no results parsed. Check the benchmark output.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print summary
    for (phase, poly_exp), (lo, med, hi) in sorted(
        results.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        row_exp = poly_exp // 2
        print(
            f"  {phase:>8}  2^{poly_exp:<2} (row_len=2^{row_exp})  "
            f"{_format_time(lo):>20} .. {_format_time(med):>20} .. {_format_time(hi):>20}"
        )

    # ── Generate LaTeX ────────────────────────────────────────────────────
    table = generate_latex_table(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"\nLaTeX table written to {args.output}")
    else:
        print("\n" + "=" * 72)
        print(table)
        print("=" * 72)


if __name__ == "__main__":
    main()
