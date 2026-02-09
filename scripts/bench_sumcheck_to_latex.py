#!/usr/bin/env python3
"""Run the sumcheck benchmark and output results as a LaTeX table.

Usage:
    python3 scripts/bench_sumcheck_to_latex.py [--output TABLE.tex] [--features "parallel simd"]
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class BenchResult:
    benchmark: str  # e.g. "Sum-of-40 Sumcheck Prover"
    limbs: int
    nvars: int
    npolys: int
    time_low: str   # e.g. "769.09 µs"
    time_mid: str
    time_high: str


# Matches lines like:
#   Sumcheck benchmarks/Sum-of-40 Sumcheck Prover/LIMBS=3/nvars=6/npolys=40
BENCH_NAME_RE = re.compile(
    r"^Sumcheck benchmarks/"
    r"(?P<bench>[^/]+)/"
    r"LIMBS=(?P<limbs>\d+)/"
    r"nvars=(?P<nvars>\d+)"
    r"(?:/npolys=(?P<npolys>\d+))?\s*$"
)

# Matches lines like:
#   time:   [769.09 µs 799.94 µs 849.34 µs]
TIME_RE = re.compile(
    r"time:\s+\[(?P<low>[\d.]+\s*\S+)\s+"
    r"(?P<mid>[\d.]+\s*\S+)\s+"
    r"(?P<high>[\d.]+\s*\S+)\]"
)


def parse_to_ms(time_str: str) -> float:
    """Convert a criterion time string like '769.09 µs' to milliseconds."""
    parts = time_str.strip().split()
    value = float(parts[0])
    unit = parts[1]
    if unit in ("ns",):
        return value / 1_000_000
    if unit in ("µs", "us"):
        return value / 1_000
    if unit in ("ms",):
        return value
    if unit in ("s",):
        return value * 1_000
    raise ValueError(f"Unknown time unit: {unit!r}")


def format_ms(ms: float) -> str:
    """Format milliseconds for display, choosing the best unit."""
    if ms < 1:
        return f"{ms * 1000:.1f}\\,\\micro{{s}}"
    if ms < 1000:
        return f"{ms:.2f}\\,ms"
    return f"{ms / 1000:.2f}\\,s"


def parse_output(output: str) -> list[BenchResult]:
    """Parse criterion benchmark output into structured results."""
    results: list[BenchResult] = []
    current_name: dict | None = None

    for line in output.splitlines():
        m = BENCH_NAME_RE.match(line.strip())
        if m:
            current_name = m.groupdict()
            continue

        m = TIME_RE.search(line)
        if m and current_name is not None:
            results.append(BenchResult(
                benchmark=current_name["bench"],
                limbs=int(current_name["limbs"]),
                nvars=int(current_name["nvars"]),
                npolys=int(current_name.get("npolys") or 0),
                time_low=m.group("low"),
                time_mid=m.group("mid"),
                time_high=m.group("high"),
            ))
            current_name = None

    return results


def results_to_latex(results: list[BenchResult]) -> str:
    """Convert parsed benchmark results into a LaTeX table string."""
    # Group by benchmark type (Prover / Verifier)
    bench_types = sorted(set(r.benchmark for r in results))
    limbs_values = sorted(set(r.limbs for r in results))
    nvars_values = sorted(set(r.nvars for r in results))

    # Build a lookup: (benchmark, limbs, nvars) -> BenchResult
    lookup: dict[tuple[str, int, int], BenchResult] = {}
    for r in results:
        lookup[(r.benchmark, r.limbs, r.nvars)] = r

    # Number of time columns = len(limbs_values) per benchmark type
    n_time_cols = len(limbs_values) * len(bench_types)

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \sisetup{round-mode=places, round-precision=2}")

    # Column spec: nvars | for each bench type: one col per limbs value
    col_spec = "c" + "".join("|" + "c" * len(limbs_values) for _ in bench_types)
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"    \toprule")

    # Header row 1: nvars + benchmark type names spanning columns
    header1_parts = [r"    \multirow{2}{*}{$n$}"]
    for bt in bench_types:
        short_name = bt.replace("Sumcheck ", "")
        header1_parts.append(
            rf"\multicolumn{{{len(limbs_values)}}}{{c}}{{{short_name}}}"
        )
    lines.append(" & ".join(header1_parts) + r" \\")

    # Header row 2: limbs sub-columns
    header2_parts = [""]
    for _ in bench_types:
        for l in limbs_values:
            bits = l * 64
            header2_parts.append(f"{bits}-bit")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"    \midrule")

    # Data rows
    for nv in nvars_values:
        row_parts = [f"${nv}$"]
        for bt in bench_types:
            for l in limbs_values:
                key = (bt, l, nv)
                if key in lookup:
                    ms = parse_to_ms(lookup[key].time_mid)
                    row_parts.append(f"${format_ms(ms)}$")
                else:
                    row_parts.append("--")
        lines.append("    " + " & ".join(row_parts) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")

    # Caption with npolys if available
    npolys_set = set(r.npolys for r in results if r.npolys > 0)
    npolys_str = ", ".join(str(n) for n in sorted(npolys_set)) if npolys_set else ""
    caption = r"Sumcheck benchmark times (median)"
    if npolys_str:
        caption += f", {npolys_str} polynomials"
    lines.append(r"  \caption{" + caption + "}")
    lines.append(r"  \label{tab:sumcheck-bench}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sumcheck benchmarks and produce a LaTeX table."
    )
    parser.add_argument(
        "--output", "-o",
        help="Output .tex file (default: stdout)",
    )
    parser.add_argument(
        "--features",
        default="parallel simd",
        help='Cargo feature flags (default: "parallel simd")',
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Criterion filter regex passed after -- (default: run all)",
    )
    parser.add_argument(
        "--from-stdin",
        action="store_true",
        help="Read benchmark output from stdin instead of running cargo bench",
    )
    args = parser.parse_args()

    if args.from_stdin:
        output = sys.stdin.read()
    else:
        cmd = [
            "cargo", "bench",
            "--bench", "sumcheck",
            "-p", "zinc-piop",
        ]
        if args.features:
            cmd += ["--features", args.features]
        if args.filter:
            cmd += ["--", args.filter]

        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        if proc.returncode != 0:
            print("Benchmark command failed:", file=sys.stderr)
            print(output, file=sys.stderr)
            sys.exit(1)

    results = parse_output(output)
    if not results:
        print("No benchmark results found in output.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(results)} benchmark result(s).", file=sys.stderr)

    latex = results_to_latex(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(latex + "\n")
        print(f"Wrote LaTeX table to {args.output}", file=sys.stderr)
    else:
        print(latex)


if __name__ == "__main__":
    main()
