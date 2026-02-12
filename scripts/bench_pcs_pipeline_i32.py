#!/usr/bin/env python3
"""
Benchmark the full PCS pipeline (Commit, Test, Verify) for Zip+ with i32 evaluations and num_rows=1.

This script runs the "PCS Pipeline Suite i32 1row" criterion benchmarks and collects
the results into a LaTeX table.

The benchmarks use i32 evaluations with various IPRS codes,
forcing num_rows=1 (single row layout). With num_rows=1, poly_size must equal row_len.
Different fields are used to support different row lengths.

    poly_size (2^P)   Config              Field           row_len
    ───────────────   ──────              ─────           ───────
    2^4  = 16         R4B2 D=1 (rate 1/4) F3329           16
    2^5  = 32         R4B4 D=1 (rate 1/4) F3329           32
    2^6  = 64         R4B8 D=1 (rate 1/4) F3329           64
    2^7  = 128        B16 D=1 (rate 1/2)  F65537          128
    2^8  = 256        B32 D=1 (rate 1/2)  F65537          256
    2^9  = 512        B64 D=1 (rate 1/2)  F65537          512
    2^10 = 1024       B16 D=2 (rate 1/2)  F65537          1024
    2^11 = 2048       B32 D=2 (rate 1/2)  F65537          2048
    2^12 = 4096       B64 D=2 (rate 1/2)  F65537          4096
    2^13 = 8192       B16 D=3 (rate 1/2)  F65537          8192
    2^14 = 16384      B32 D=3 (rate 1/2)  F65537          16384
    2^16 = 65536      B16 D=4 (rate 1/2)  F1179649        65536
    2^17 = 131072     B32 D=4 (rate 1/2)  F167772161      131072
    2^18 = 262144     B64 D=4 (rate 1/2)  F167772161      262144

Usage:
    python3 scripts/bench_pcs_pipeline_i32.py
    python3 scripts/bench_pcs_pipeline_i32.py --no-run --input bench_output.txt
    python3 scripts/bench_pcs_pipeline_i32.py --output pcs_table_i32.tex
"""

import argparse
import os
import re
import subprocess
import sys

# ── Configuration ──────────────────────────────────────────────────────────────

BENCH_FILTER = "PCS Pipeline Suite i32 1row"
CARGO_BENCH_CMD = [
    "cargo", "bench",
    "--bench", "zip_plus_benches",
    "--features", "parallel asm simd",
    "--", BENCH_FILTER,
]

# Expected polynomial size exponents for num_rows=1 benchmarks.
# With num_rows=1, poly_size must equal row_len.
# Different fields support different row lengths.
EXPECTED_POLY_EXPS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]

# Number of repetitions: report the cost of performing each operation this many
# times (the measured per-operation time is multiplied by this factor).
NUM_REPETITIONS = 24

# The three phases we benchmark
PHASES = ["Commit", "Test", "Verify"]

# Map poly exponent → (num_rows, config description, field)
# With num_rows=1, poly_size = row_len
POLY_EXP_CONFIG = {
    4:  (1, "R4B2 D=1",  "F3329"),      # row_len=16
    5:  (1, "R4B4 D=1",  "F3329"),      # row_len=32
    6:  (1, "R4B8 D=1",  "F3329"),      # row_len=64
    7:  (1, "B16 D=1",   "F65537"),     # row_len=128
    8:  (1, "B32 D=1",   "F65537"),     # row_len=256
    9:  (1, "B64 D=1",   "F65537"),     # row_len=512
    10: (1, "B16 D=2",   "F65537"),     # row_len=1024
    11: (1, "B32 D=2",   "F65537"),     # row_len=2048
    12: (1, "B64 D=2",   "F65537"),     # row_len=4096
    13: (1, "B16 D=3",   "F65537"),     # row_len=8192
    14: (1, "B32 D=3",   "F65537"),     # row_len=16384
    16: (1, "B16 D=4",   "F1179649"),   # row_len=65536
    17: (1, "B32 D=4",   "F167772161"), # row_len=131072
    18: (1, "B64 D=4",   "F167772161"), # row_len=262144
}

# Map field name to LaTeX representation
FIELD_LATEX = {
    "F3329":      r"$\mathbb{F}_{3329}$",
    "F65537":     r"$\mathbb{F}_{65537}$",
    "F1179649":   r"$\mathbb{F}_{1179649}$",
    "F167772161": r"$\mathbb{F}_{167772161}$",
}


def field_for_poly_exp(p: int) -> str:
    """Map polynomial size exponent to the field label."""
    _, _, field = POLY_EXP_CONFIG.get(p, (1, "", "F3329"))
    return FIELD_LATEX.get(field, r"$\mathbb{F}$")


# ── Parsing ────────────────────────────────────────────────────────────────────

# Matches benchmark name lines like:
#   Commit poly_size=2^4 num_rows=1
#   Test poly_size=2^5 num_rows=1
#   Verify poly_size=2^6 num_rows=1
_RE_PHASE_NAME = re.compile(
    r"(?P<phase>Commit|Test|Verify)\s+poly_size=2\^(?P<exp>\d+)"
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
    Parse criterion benchmark output for the PCS pipeline suite (i32).

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
                    # Scale by NUM_REPETITIONS to report the cost of
                    # performing the operation multiple times.
                    results[(phase, poly_exp)] = (
                        low * NUM_REPETITIONS,
                        med * NUM_REPETITIONS,
                        high * NUM_REPETITIONS,
                    )
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
        r"\caption{PCS pipeline benchmark ($\times" + str(NUM_REPETITIONS) + r"$) for i32 with num\_rows=1 (parallel+asm+simd).}",
        r"\label{tab:pcs-pipeline-suite-i32-1row}",
        r"\begin{tabular}{r r r l r r r}",
        r"\toprule",
        (
            r"\textbf{Row len} & $\boldsymbol{2^k}$ & $\boldsymbol{2^P}$ & \textbf{Field}"
            r" & \textbf{Commit} & \textbf{Test} & \textbf{Verify} \\"
        ),
        r"\midrule",
    ]

    for poly_exp in EXPECTED_POLY_EXPS:
        _num_rows, _config, _field = POLY_EXP_CONFIG[poly_exp]
        # With num_rows=1, poly_size = row_len = 2^poly_exp
        row_len = 2 ** poly_exp
        row_exp = poly_exp
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
        description="Run PCS pipeline (Commit/Test/Verify) benchmarks for i32 and produce a LaTeX table."
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
        # With num_rows=1, row_len = poly_size = 2^poly_exp
        print(
            f"  {phase:>8}  2^{poly_exp:<2} (row_len=2^{poly_exp})  "
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
