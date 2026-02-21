#!/usr/bin/env python3
"""
Benchmark the full PCS pipeline (Encode, Merkle, Commit, Test, Verify) for
**Batched** Zip+ with i32 evaluations and batch_size=27.

This script runs the "Batched PCS Pipeline Suite i32 1row batch27" criterion
benchmarks and collects the results into a LaTeX table.  The benchmarks batch 27
polynomials into a single shared Merkle tree, using rate 1/4 IPRS codes.

    poly_size (2^P)   Config               Field           row_len     num_rows
    ───────────────   ──────               ─────           ───────     ────────
    2^10 = 1024       R4B16 D=2 (rate 1/4) F65537          1024        1
    2^10 = 1024       R4B64 D=1 (rate 1/4) F65537          512         2
    2^11 = 2048       R4B16 D=2 (rate 1/4) F65537          1024        2
    2^12 = 4096       R4B16 D=2 (rate 1/4) F65537          1024        4
    2^13 = 8192       R4B32 D=2 (rate 1/4) F65537          2048        4

Usage:
    python3 scripts/bench_batched_pcs_pipeline_i32_batch27.py
    python3 scripts/bench_batched_pcs_pipeline_i32_batch27.py --no-run --input bench_output.txt
    python3 scripts/bench_batched_pcs_pipeline_i32_batch27.py - -output batched_pcs_table_i32_batch27.tex
"""

import argparse
import os
import re
import subprocess
import sys

# ── Configuration ──────────────────────────────────────────────────────────────

BENCH_FILTER = "Batched PCS Pipeline Suite i32 1row batch27/.* poly_size=2\\^(10|11|12|13) "
CARGO_BENCH_CMD = [
    "cargo", "bench",
    "--bench", "batched_zip_plus_benches",
    "--features", "parallel asm simd",
    "--", BENCH_FILTER,
]

# Expected polynomial size exponents and num_rows for benchmarks.
# Each entry is (poly_exp, num_rows).
# All use rate 1/4 IPRS codes.
EXPECTED_CONFIGS = [
    (10, 1), (10, 2),
    (11, 2), (12, 4), (13, 4),
]

# Number of polynomials in the batch (must match BATCH_SIZE_27 in benchmarks).
BATCH_SIZE = 27

# The five phases we benchmark
PHASES = ["Encode", "Merkle", "Commit", "Test", "Verify"]

# Map (poly exponent, num_rows) → (config description, field)
# All use rate 1/4 IPRS codes
POLY_EXP_CONFIG = {
    (10, 1): ("R4B16 D=2", "F65537"),     # row_len=1024
    (10, 2): ("R4B64 D=1", "F65537"),     # row_len=512
    (11, 2): ("R4B16 D=2", "F65537"),     # row_len=1024
    (12, 4): ("R4B16 D=2", "F65537"),     # row_len=1024
    (13, 4): ("R4B32 D=2", "F65537"),     # row_len=2048
}

# Map field name to LaTeX representation
FIELD_LATEX = {
    "F65537":     r"$\mathbb{F}_{65537}$",
}


def field_for_config(poly_exp: int, num_rows: int) -> str:
    """Map (polynomial size exponent, num_rows) to the field label."""
    _, field = POLY_EXP_CONFIG.get((poly_exp, num_rows), ("", "F65537"))
    return FIELD_LATEX.get(field, r"$\mathbb{F}$")


# ── Parsing ────────────────────────────────────────────────────────────────────

# Matches benchmark name lines like:
#   Encode poly_size=2^10 num_rows=1
#   Merkle poly_size=2^10 num_rows=1
#   Commit poly_size=2^10 num_rows=1
#   Test poly_size=2^11 num_rows=2
#   Verify poly_size=2^13 num_rows=4
_RE_PHASE_NAME = re.compile(
    r"(?P<phase>Encode|Merkle|Commit|Test|Verify)\s+poly_size=2\^(?P<exp>\d+)\s+num_rows=(?P<rows>\d+)"
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
) -> dict[tuple[str, int, int], tuple[float, float, float]]:
    """
    Parse criterion benchmark output for the batched PCS pipeline suite (i32, batch27).

    Returns a dict mapping (phase, poly_exp, num_rows) -> (low_us, median_us, high_us).
    phase is one of "Encode", "Merkle", "Commit", "Test", "Verify".
    poly_exp is the exponent P where poly_size = 2^P.
    num_rows is the number of rows used.

    No scaling is applied — each measurement already covers the full batch of
    BATCH_SIZE polynomials.
    """
    results: dict[tuple[str, int, int], tuple[float, float, float]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m_name = _RE_PHASE_NAME.search(line)
        if m_name:
            phase = m_name.group("phase")
            poly_exp = int(m_name.group("exp"))
            num_rows = int(m_name.group("rows"))

            # Look ahead for the time: line
            for j in range(i + 1, min(i + 5, len(lines))):
                m_time = _RE_TIME.search(lines[j])
                if m_time:
                    low = _parse_time_us(m_time.group(1), m_time.group(2))
                    med = _parse_time_us(m_time.group(3), m_time.group(4))
                    high = _parse_time_us(m_time.group(5), m_time.group(6))
                    results[(phase, poly_exp, num_rows)] = (low, med, high)
                    i = j
                    break
        i += 1
    return results


# ── LaTeX generation ───────────────────────────────────────────────────────────


def generate_latex_table(
    results: dict[tuple[str, int, int], tuple[float, float, float]],
) -> str:
    """Generate a LaTeX table from the parsed results."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Batched PCS pipeline benchmark (batch=" + str(BATCH_SIZE) + r") for i32, rate 1/4 (parallel+asm+simd).}",
        r"\label{tab:batched-pcs-pipeline-suite-i32-1row-r4-batch27}",
        r"\begin{tabular}{r r r r l r r r r r}",
        r"\toprule",
        (
            r"\textbf{Row len} & $\boldsymbol{2^k}$ & \textbf{Rows} & $\boldsymbol{2^P}$ & \textbf{Field}"
            r" & \textbf{Encode} & \textbf{Merkle} & \textbf{Commit} & \textbf{Test} & \textbf{Verify} \\"
        ),
        r"\midrule",
    ]

    for poly_exp, num_rows in EXPECTED_CONFIGS:
        _config, _field = POLY_EXP_CONFIG[(poly_exp, num_rows)]
        poly_size = 2 ** poly_exp
        row_len = poly_size // num_rows
        row_exp = row_len.bit_length() - 1
        field = field_for_config(poly_exp, num_rows)

        phase_strs = []
        for phase in PHASES:
            key = (phase, poly_exp, num_rows)
            if key in results:
                _, med, _ = results[key]
                phase_strs.append(_format_time(med))
            else:
                phase_strs.append("---")

        lines.append(
            f"{row_len:>7} & $2^{{{row_exp}}}$ & {num_rows} & $2^{{{poly_exp}}}$"
            f" & {field}"
            f" & {phase_strs[0]} & {phase_strs[1]} & {phase_strs[2]} & {phase_strs[3]} & {phase_strs[4]} \\\\"
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
        description="Run Batched PCS pipeline (Encode/Merkle/Commit/Test/Verify) benchmarks for i32 (batch=27) and produce a LaTeX table."
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
    for (phase, poly_exp, num_rows), (lo, med, hi) in sorted(
        results.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
    ):
        row_len = (2 ** poly_exp) // num_rows
        row_exp = row_len.bit_length() - 1
        print(
            f"  {phase:>8}  2^{poly_exp:<2} (row_len=2^{row_exp}, num_rows={num_rows})  "
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
