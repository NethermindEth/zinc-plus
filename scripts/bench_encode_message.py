#!/usr/bin/env python3
"""
Benchmark EncodeMessage for the Zip+ polynomial commitment scheme.

This script runs the "EncodeMessage Suite" criterion benchmarks and collects
the results into a LaTeX table.

EncodeMessage encodes a single row (message vector) of length `row_len` using
an IPRS (Interleaved Pseudorandom Subset) linear code.  The code maps:
    Eval  ->  Cw     (e.g. i32 -> i128, or BinaryPoly<32> -> DensePolynomial<i128,32>)
at rate 1/2 (codeword length = 2 * row_len).

The benchmark is run with batch=1 (single encoding per iteration) for
row_len = 2^6, 2^7, ..., 2^19.  Two element types are benchmarked:
    - i32         (scalar integer evaluations)
    - BPoly<31>   (degree-31 binary polynomials, coefficient-wise encoding)

The following IPRS code configurations are used:

    Row length   Field              Config
    ──────────   ─────              ──────
    2^6  (64)    F3329   (13×2^8+1) B8-D1    (BASE_LEN=8,  DEPTH=1)
    2^7  (128)   F3329              B16-D1   (BASE_LEN=16, DEPTH=1)
    2^8  (256)   F65537  (2^16+1)   B32-D1   (BASE_LEN=32, DEPTH=1)
    2^9  (512)   F65537             B64-D1   (BASE_LEN=64, DEPTH=1)
    2^10 (1024)  F65537             B16-D2   (BASE_LEN=16, DEPTH=2)
    2^11 (2048)  F65537             B32-D2
    2^12 (4096)  F65537             B64-D2
    2^13 (8192)  F65537             B16-D3
    2^14 (16384) F65537             B32-D3
    2^15 (32768) F65537             B64-D3
    2^16 (65536) F1179649 (9×2^17+1) B16-D4
    2^17 (131072) F167772161 (5×2^25+1) B32-D4
    2^18 (262144) F167772161         B64-D4
    2^19 (524288) F167772161         B16-D5

Usage:
    python3 scripts/bench_encode_message.py
    python3 scripts/bench_encode_message.py --no-run        # parse previous output only
    python3 scripts/bench_encode_message.py --output table.tex
"""

import argparse
import math
import os
import re
import subprocess
import sys

# ── Configuration ──────────────────────────────────────────────────────────────

BENCH_FILTER = "EncodeMessage Suite"
CARGO_BENCH_CMD = [
    "cargo", "bench",
    "--bench", "zip_plus_benches",
    "--features", "parallel asm simd",
    "--", BENCH_FILTER,
]

# Expected row_len values (2^6 .. 2^19)
EXPECTED_ROW_LENS = [1 << e for e in range(6, 20)]

# The two element types we benchmark
ELEM_TYPES = ["i32", "BPoly31"]

# Map row_len -> field label for display
def field_for_row_len(row_len: int) -> str:
    exp = int(math.log2(row_len))
    if exp <= 7:
        return r"$\mathbb{F}_{3329}$"
    elif exp <= 15:
        return r"$\mathbb{F}_{65537}$"
    elif exp == 16:
        return r"$\mathbb{F}_{1179649}$"
    else:
        return r"$\mathbb{F}_{167772161}$"


# ── Parsing ────────────────────────────────────────────────────────────────────

# Matches full or truncated row_len at end of line:
#   row_len = 256        -> full: group "full"="256"
#   row_len = 102...     -> partial: group "partial"="102"
#   row_len =...         -> no digits at all (fully truncated)
#   row_len =... #2      -> fully truncated with criterion's dedup suffix
_RE_BENCH_NAME = re.compile(
    r"row_len\s*=\s*(?:(?P<full>\d+)\s*$|(?P<partial>\d+)\.\.\.|\.\.\.)"
)
_RE_TIME = re.compile(
    r"time:\s*\[([0-9.]+)\s*(\S+)\s+([0-9.]+)\s*(\S+)\s+([0-9.]+)\s*(\S+)\]"
)

# Known benchmarks in order per (field, elem_type) group.
# This is used to resolve fully-truncated benchmark names.
_FIELD_ROW_LENS = {
    "F3329":       [64, 128],
    "F65537":      [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    "F1179649":    [65536],
    "F167772161":  [131072, 262144, 524288],
}


def _resolve_row_len(prefix: str) -> int | None:
    """Resolve a truncated row_len prefix to the actual power-of-2 value."""
    for e in range(6, 25):
        val = 1 << e
        if str(val).startswith(prefix):
            return val
    return None


def _extract_field(line: str) -> str | None:
    """Extract the field name (e.g. 'F167772161') from the benchmark line."""
    m = re.search(r"/(F\d+)-", line)
    return m.group(1) if m else None


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


def parse_criterion_output(text: str) -> dict[tuple[str, int], tuple[float, float, float]]:
    """
    Parse criterion benchmark output.

    Returns a dict mapping (elem_type, row_len) -> (low_us, median_us, high_us).
    elem_type is one of "i32" or "BPoly31".
    """
    results: dict[tuple[str, int], tuple[float, float, float]] = {}
    # Track how many benchmarks we've seen per (field, elem_type) to resolve
    # fully-truncated names by position.
    seen_count: dict[tuple[str, str], int] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect the benchmark name line containing row_len (full or truncated)
        m_name = _RE_BENCH_NAME.search(line)
        if m_name:
            if m_name.group("full") is not None:
                # Full row_len value
                row_len = int(m_name.group("full"))
            elif m_name.group("partial") is not None:
                # Truncated with some digits: e.g. "102..." -> resolve to 1024
                row_len_resolved = _resolve_row_len(m_name.group("partial"))
                if row_len_resolved is None:
                    i += 1
                    continue
                row_len = row_len_resolved
            else:
                # Fully truncated: "row_len =..." or "row_len =... #N"
                # Resolve from the field name and sequential position.
                row_len = None
            # Determine elem type from the code_name portion of the benchmark name.
            # Bench names look like:
            #   .../F65537-i32/EncodeMessage batch=1: i32 -> i128, row_len = 256
            #   .../F65537-BPoly31/EncodeMessage batch=1: BPoly<31> -> Poly<i128, 31>, row_len = 256
            if "-BPoly31/" in line or "BPoly<" in line or "BinaryPoly" in line:
                elem_type = "BPoly31"
            else:
                elem_type = "i32"

            # For fully-truncated names, resolve row_len from field + position
            if row_len is None:
                field = _extract_field(line)
                if field is None:
                    i += 1
                    continue
                key = (field, elem_type)
                idx = seen_count.get(key, 0)
                field_lens = _FIELD_ROW_LENS.get(field, [])
                if idx < len(field_lens):
                    row_len = field_lens[idx]
                else:
                    i += 1
                    continue

            # Look ahead for the time: line
            for j in range(i + 1, min(i + 5, len(lines))):
                m_time = _RE_TIME.search(lines[j])
                if m_time:
                    low = _parse_time_us(m_time.group(1), m_time.group(2))
                    med = _parse_time_us(m_time.group(3), m_time.group(4))
                    high = _parse_time_us(m_time.group(5), m_time.group(6))
                    results[(elem_type, row_len)] = (low, med, high)
                    # Track position within each (field, elem_type) group
                    # only when we actually record a result.
                    field = _extract_field(line)
                    if field is not None:
                        key = (field, elem_type)
                        seen_count[key] = seen_count.get(key, 0) + 1
                    i = j
                    break
        i += 1
    return results


# ── LaTeX generation ───────────────────────────────────────────────────────────

def generate_latex_table(results: dict[tuple[str, int], tuple[float, float, float]]) -> str:
    """Generate a LaTeX table from the parsed results."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{EncodeMessage benchmark (batch\,=\,1, rate\,$1/2$, parallel+asm+simd).}",
        r"\label{tab:encode-message-suite}",
        r"\begin{tabular}{r r l r r}",
        r"\toprule",
        r"\textbf{Row len} & $\boldsymbol{2^k}$ & \textbf{Field} & \textbf{i32} & \textbf{BPoly\textlangle 31\textrangle} \\",
        r"\midrule",
    ]

    prev_field = None
    for row_len in EXPECTED_ROW_LENS:
        exp = int(math.log2(row_len))
        field = field_for_row_len(row_len)

        # Add a midrule when the field changes
        if prev_field is not None and field != prev_field:
            lines.append(r"\midrule")
        prev_field = field

        # i32 result
        key_i32 = ("i32", row_len)
        if key_i32 in results:
            _, med, _ = results[key_i32]
            i32_str = _format_time(med)
        else:
            i32_str = "---"

        # BPoly31 result
        key_bp = ("BPoly31", row_len)
        if key_bp in results:
            _, med, _ = results[key_bp]
            bp_str = _format_time(med)
        else:
            bp_str = "---"

        lines.append(
            f"{row_len:>7} & $2^{{{exp}}}$ & {field} & {i32_str} & {bp_str} \\\\"
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
        description="Run EncodeMessage benchmarks and produce a LaTeX table."
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
        print("Warning: no results parsed. Check the benchmark output.", file=sys.stderr)
        sys.exit(1)

    # Print summary
    for (elem, row_len), (lo, med, hi) in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        exp = int(math.log2(row_len))
        print(f"  {elem:>8}  2^{exp:<2} (row_len={row_len:>6})  "
              f"{_format_time(lo):>20} .. {_format_time(med):>20} .. {_format_time(hi):>20}")

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
