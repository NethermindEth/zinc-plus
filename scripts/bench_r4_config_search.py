#!/usr/bin/env python3
"""
Benchmark all rate-1/4 IPRS configurations and produce LaTeX tables.

This script runs the "r4_config_search" criterion benchmarks (PCS pipeline
for BPoly<31> and scalar i32) and collects the results into LaTeX tables
grouped by poly_size exponent (P=8..11).

For each poly_size, every viable rate-1/4 IPRS code configuration is tested
by varying:
  - The twiddle field (F3329, F65537, F1179649, F7340033, F167772161)
  - The base-matrix size (BASE_LEN ∈ {2, 4, 8, 16, 32, 64})
  - The recursion depth (DEPTH ∈ {1, 2})

Rate 1/4 means BASE_DIM = 4 × BASE_LEN.
Row length = BASE_LEN × 8^DEPTH.

Usage:
    python3 scripts/bench_r4_config_search.py
    python3 scripts/bench_r4_config_search.py --no-run --input bench_output.txt
    python3 scripts/bench_r4_config_search.py --from-criterion target/criterion
    python3 scripts/bench_r4_config_search.py --output r4_config_search.tex
    python3 scripts/bench_r4_config_search.py --filter "R4 PCS P=8"
    python3 scripts/bench_r4_config_search.py --eval-type bpoly   # BPoly<31> only
    python3 scripts/bench_r4_config_search.py --eval-type scalar  # scalar i32 only
    python3 scripts/bench_r4_config_search.py --eval-type all     # (default) both
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────

BENCH_NAME = "r4_config_search"

# Benchmark group names used in criterion
BPOLY_GROUPS = ["R4 PCS P=8", "R4 PCS P=9", "R4 PCS P=10", "R4 PCS P=11"]
SCALAR_GROUP = "R4 PCS i32"
ENCODE_GROUPS = ["R4 Encode i32 Config Search", "R4 Encode BPoly31 Config Search"]

# The three PCS phases
PHASES = ["Commit", "Test", "Verify"]

# Config label -> (field, base_len_label, row_len_d1, row_len_d2)
CONFIG_INFO = {
    "F3329-R4B2":      ("F3329",      "R4B2",  16,    None),
    "F3329-R4B4":      ("F3329",      "R4B4",  32,    None),
    "F3329-R4B8":      ("F3329",      "R4B8",  64,    None),
    "F65537-R4B16":    ("F65537",     "R4B16", 128,   1024),
    "F65537-R4B32":    ("F65537",     "R4B32", 256,   2048),
    "F65537-R4B64":    ("F65537",     "R4B64", 512,   4096),
    "F1179649-R4B16":  ("F1179649",   "R4B16", 128,   1024),
    "F1179649-R4B32":  ("F1179649",   "R4B32", 256,   2048),
    "F1179649-R4B64":  ("F1179649",   "R4B64", 512,   4096),
    "F7340033-R4B16":  ("F7340033",   "R4B16", 128,   1024),
    "F7340033-R4B32":  ("F7340033",   "R4B32", 256,   2048),
    "F7340033-R4B64":  ("F7340033",   "R4B64", 512,   4096),
    "F167772161-R4B16":("F167772161", "R4B16", 128,   1024),
    "F167772161-R4B32":("F167772161", "R4B32", 256,   2048),
    "F167772161-R4B64":("F167772161", "R4B64", 512,   4096),
}

# Field name → LaTeX
FIELD_LATEX = {
    "F3329":       r"$\mathbb{F}_{3329}$",
    "F65537":      r"$\mathbb{F}_{65537}$",
    "F1179649":    r"$\mathbb{F}_{1179649}$",
    "F7340033":    r"$\mathbb{F}_{7340033}$",
    "F167772161":  r"$\mathbb{F}_{167772161}$",
}


def _config_row_len(config_label: str) -> int:
    """Compute row_len from the config label like F3329-R4B2-D1."""
    parts = config_label.rsplit("-", 1)
    if len(parts) != 2:
        return 0
    base_label = parts[0]  # e.g. "F3329-R4B2"
    depth_str = parts[1]   # e.g. "D1"
    depth = int(depth_str[1:])
    info = CONFIG_INFO.get(base_label)
    if info is None:
        return 0
    # row_len = base_len * 8^depth, but we stored row_len for D=1 and D=2
    if depth == 1:
        return info[2]
    elif depth == 2:
        return info[3] if info[3] is not None else 0
    return 0


def _config_field(config_label: str) -> str:
    """Extract the field name from a config label."""
    base_label = config_label.rsplit("-", 1)[0]
    info = CONFIG_INFO.get(base_label)
    return info[0] if info else ""


def _config_code_name(config_label: str) -> str:
    """Extract a short code name like R4B16 from a config label."""
    base_label = config_label.rsplit("-", 1)[0]
    info = CONFIG_INFO.get(base_label)
    return info[1] if info else ""


def _config_depth(config_label: str) -> int:
    """Extract depth from a config label."""
    parts = config_label.rsplit("-", 1)
    if len(parts) == 2 and parts[1].startswith("D"):
        return int(parts[1][1:])
    return 0


# ── Parsing ────────────────────────────────────────────────────────────────────

# Match the PCS benchmark lines. The _named variants append the label at the end.
# Examples:
#   Commit: Eval=BPoly<31>, Cw=DensePoly<i128, 32>, Comb=DensePoly<Int<5>, 32>, poly_size=2^8 F3329-R4B2-D1
#   Test: Eval=BPoly<31>, Cw=DensePoly<i128, 32>, Comb=DensePoly<Int<5>, 32>, poly_size=2^8 F3329-R4B2-D1
#   Verify poly_size=2^8 num_rows=16 F3329-R4B2-D1
_RE_COMMIT_TEST = re.compile(
    r"(?P<phase>Commit|Test):\s+.*poly_size=2\^(?P<exp>\d+)\s+(?P<config>\S+-R4B\d+-D\d+)"
)
_RE_VERIFY = re.compile(
    r"Verify\s+poly_size=2\^(?P<exp>\d+)\s+num_rows=\d+\s+(?P<config>\S+-R4B\d+-D\d+)"
)

# Criterion's time output line
_RE_TIME = re.compile(
    r"time:\s*\[([0-9.]+)\s*(\S+)\s+([0-9.]+)\s*(\S+)\s+([0-9.]+)\s*(\S+)\]"
)

# Encoding benchmarks:
#   F3329-R4B2-D1/EncodeMessage: BPoly<31> -> DensePoly<i128, 32>, row_len = 16
#   F3329-R4B2-D1/EncodeMessage: i32 -> i128, row_len = 16
_RE_ENCODE = re.compile(
    r"(?P<config>\S+-R4B\d+-D\d+)/EncodeMessage:\s+\S+\s+->\s+\S+,\s+row_len\s*=\s*(?P<row_len>\d+)"
)


def _parse_time_us(value: str, unit: str) -> float:
    """Convert a criterion time value to microseconds."""
    v = float(value)
    unit = unit.replace("µ", "u")
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


def parse_criterion_output(text: str) -> dict:
    """
    Parse criterion benchmark output for the R4 config search benchmarks.

    Returns a dict mapping (phase, poly_exp, config_label) -> (low_us, med_us, high_us).
    phase is one of "Commit", "Test", "Verify", "Encode".
    poly_exp is the exponent P where poly_size = 2^P.
    config_label is e.g. "F3329-R4B2-D1".
    """
    results = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Try PCS Commit/Test
        m = _RE_COMMIT_TEST.search(line)
        if m:
            phase = m.group("phase")
            poly_exp = int(m.group("exp"))
            config = m.group("config")
            timing = _find_timing(lines, i)
            if timing:
                results[(phase, poly_exp, config)] = timing
                i += 1
                continue

        # Try PCS Verify
        m = _RE_VERIFY.search(line)
        if m:
            poly_exp = int(m.group("exp"))
            config = m.group("config")
            timing = _find_timing(lines, i)
            if timing:
                results[("Verify", poly_exp, config)] = timing
                i += 1
                continue

        # Try Encoding
        m = _RE_ENCODE.search(line)
        if m:
            config = m.group("config")
            row_len = int(m.group("row_len"))
            timing = _find_timing(lines, i)
            if timing:
                results[("Encode", row_len, config)] = timing
                i += 1
                continue

        i += 1
    return results


def _find_timing(lines: list, start: int):
    """Look ahead a few lines to find the criterion time: [...] line."""
    for j in range(start + 1, min(start + 6, len(lines))):
        m_time = _RE_TIME.search(lines[j])
        if m_time:
            low = _parse_time_us(m_time.group(1), m_time.group(2))
            med = _parse_time_us(m_time.group(3), m_time.group(4))
            high = _parse_time_us(m_time.group(5), m_time.group(6))
            return (low, med, high)
    return None


def parse_criterion_data(criterion_dir: str) -> tuple[dict, dict]:
    """
    Parse benchmark results from criterion's saved JSON data files.

    Criterion truncates long benchmark IDs in terminal output, which causes
    BPoly Commit/Test entries to be unparseable.  The saved JSON files under
    ``target/criterion/`` always contain the full, untruncated benchmark ID
    in ``benchmark.json``, making this the authoritative source of results.

    Returns ``(bpoly_results, scalar_results)``, each mapping
    ``(phase, poly_exp, config) -> (low_us, med_us, high_us)``.
    """
    bpoly_results: dict = {}
    scalar_results: dict = {}

    # Map group directory names to eval type
    pcs_groups: dict[str, str] = {
        "R4 PCS P=8": "bpoly",
        "R4 PCS P=9": "bpoly",
        "R4 PCS P=10": "bpoly",
        "R4 PCS P=11": "bpoly",
        "R4 PCS i32": "scalar",
    }

    for group_name, eval_type in pcs_groups.items():
        group_dir = os.path.join(criterion_dir, group_name)
        if not os.path.isdir(group_dir):
            continue

        for bench_dir_name in os.listdir(group_dir):
            bench_path = os.path.join(group_dir, bench_dir_name)
            benchmark_json = os.path.join(bench_path, "new", "benchmark.json")
            estimates_json = os.path.join(bench_path, "new", "estimates.json")

            if not os.path.exists(benchmark_json) or not os.path.exists(estimates_json):
                continue

            try:
                with open(benchmark_json) as f:
                    bench_info = json.load(f)
                with open(estimates_json) as f:
                    estimates = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            function_id = bench_info.get("function_id", "")

            # Extract phase, poly_exp, config from the full function_id
            phase = None
            poly_exp = 0
            config = ""

            m = _RE_COMMIT_TEST.search(function_id)
            if m:
                phase = m.group("phase")
                poly_exp = int(m.group("exp"))
                config = m.group("config")
            else:
                m = _RE_VERIFY.search(function_id)
                if m:
                    phase = "Verify"
                    poly_exp = int(m.group("exp"))
                    config = m.group("config")

            if phase is None:
                continue

            # Criterion reports time: [low med high] from slope CI
            slope = estimates.get("slope") or estimates.get("mean")
            if slope is None:
                continue

            ci = slope["confidence_interval"]
            low_us = ci["lower_bound"] / 1000.0   # ns → µs
            med_us = slope["point_estimate"] / 1000.0
            high_us = ci["upper_bound"] / 1000.0

            key = (phase, poly_exp, config)
            if eval_type == "bpoly":
                bpoly_results[key] = (low_us, med_us, high_us)
            else:
                scalar_results[key] = (low_us, med_us, high_us)

    return bpoly_results, scalar_results


# ── LaTeX generation ───────────────────────────────────────────────────────────


def _collect_configs_for_exp(results: dict, poly_exp: int) -> list[str]:
    """Collect all config labels seen for a given poly_exp, sorted consistently."""
    configs = set()
    for (phase, exp, config) in results:
        if phase in PHASES and exp == poly_exp:
            configs.add(config)
    # Sort by: field index, base_len, depth
    field_order = ["F3329", "F65537", "F1179649", "F7340033", "F167772161"]

    def sort_key(c):
        field = _config_field(c)
        fi = field_order.index(field) if field in field_order else 99
        row_len = _config_row_len(c)
        depth = _config_depth(c)
        return (fi, row_len, depth)

    return sorted(configs, key=sort_key)


def generate_pcs_table(results: dict, poly_exp: int, eval_label: str) -> str:
    """Generate a LaTeX table for a single poly_size exponent."""
    configs = _collect_configs_for_exp(results, poly_exp)
    if not configs:
        return f"% No results for P={poly_exp}\n"

    poly_size = 1 << poly_exp

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Rate-1/4 PCS pipeline for {eval_label}, $2^{{{poly_exp}}}={poly_size}$ evaluations (parallel+asm+simd).}}",
        rf"\label{{tab:r4-config-search-p{poly_exp}}}",
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        (
            r"\textbf{Field} & \textbf{Code}"
            r" & \textbf{Depth} & \textbf{row\_len} & \textbf{Commit}"
            r" & \textbf{Test} & \textbf{Verify} \\"
        ),
        r"\midrule",
    ]

    prev_field = None
    for config in configs:
        field_name = _config_field(config)
        code_name = _config_code_name(config)
        depth = _config_depth(config)
        row_len = _config_row_len(config)
        num_rows = poly_size // row_len if row_len > 0 else "?"
        field_latex = FIELD_LATEX.get(field_name, field_name)

        # Add separator between field groups
        if prev_field is not None and prev_field != field_name:
            lines.append(r"\addlinespace")
        prev_field = field_name

        phase_strs = []
        for phase in PHASES:
            key = (phase, poly_exp, config)
            if key in results:
                _, med, _ = results[key]
                phase_strs.append(_format_time(med))
            else:
                phase_strs.append("---")

        lines.append(
            f"{field_latex} & {code_name} & {depth} & {row_len}"
            f" & {phase_strs[0]} & {phase_strs[1]} & {phase_strs[2]} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_combined_table(results: dict, eval_label: str) -> str:
    """Generate a single combined LaTeX table with all poly_sizes."""
    all_configs = set()
    all_exps = set()
    for (phase, exp, config) in results:
        if phase in PHASES:
            all_configs.add(config)
            all_exps.add(exp)

    if not all_configs:
        return "% No PCS results found.\n"

    poly_exps = sorted(all_exps)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Rate-1/4 IPRS config search: PCS pipeline for {eval_label} (parallel+asm+simd).}}",
        r"\label{tab:r4-config-search-combined}",
    ]

    # For each poly_size, generate a subtable
    for poly_exp in poly_exps:
        poly_size = 1 << poly_exp
        configs = _collect_configs_for_exp(results, poly_exp)
        if not configs:
            continue

        lines += [
            "",
            rf"\medskip",
            rf"\textbf{{$2^{{{poly_exp}}} = {poly_size}$ evaluations}}",
            r"\smallskip",
            "",
            r"\begin{tabular}{l l r r r r r}",
            r"\toprule",
            (
                r"\textbf{Field} & \textbf{Code}"
                r" & \textbf{D} & \textbf{row\_len} & \textbf{Commit}"
                r" & \textbf{Test} & \textbf{Verify} \\"
            ),
            r"\midrule",
        ]

        prev_field = None
        for config in configs:
            field_name = _config_field(config)
            code_name = _config_code_name(config)
            depth = _config_depth(config)
            row_len = _config_row_len(config)
            field_latex = FIELD_LATEX.get(field_name, field_name)

            if prev_field is not None and prev_field != field_name:
                lines.append(r"\addlinespace")
            prev_field = field_name

            phase_strs = []
            for phase in PHASES:
                key = (phase, poly_exp, config)
                if key in results:
                    _, med, _ = results[key]
                    phase_strs.append(_format_time(med))
                else:
                    phase_strs.append("---")

            lines.append(
                f"{field_latex} & {code_name} & {depth} & {row_len}"
                f" & {phase_strs[0]} & {phase_strs[1]} & {phase_strs[2]} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
        ]

    lines += [
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_scalar_table(results: dict) -> str:
    """Generate a LaTeX table for the scalar i32 PCS benchmarks.

    The scalar benchmarks put all P values in one group ("R4 PCS i32"),
    so we organise by P then config.
    """
    # Collect all (exp, config) for scalar results
    all_exps = set()
    for (phase, exp, config) in results:
        if phase in PHASES:
            all_exps.add(exp)

    if not all_exps:
        return "% No scalar PCS results found.\n"

    poly_exps = sorted(all_exps)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Rate-1/4 IPRS config search: PCS pipeline for scalar i32 (parallel+asm+simd).}",
        r"\label{tab:r4-config-search-scalar}",
    ]

    for poly_exp in poly_exps:
        poly_size = 1 << poly_exp
        configs = _collect_configs_for_exp(results, poly_exp)
        if not configs:
            continue

        lines += [
            "",
            rf"\medskip",
            rf"\textbf{{$2^{{{poly_exp}}} = {poly_size}$ evaluations}}",
            r"\smallskip",
            "",
            r"\begin{tabular}{l l r r r r r}",
            r"\toprule",
            (
                r"\textbf{Field} & \textbf{Code}"
                r" & \textbf{D} & \textbf{row\_len} & \textbf{Commit}"
                r" & \textbf{Test} & \textbf{Verify} \\"
            ),
            r"\midrule",
        ]

        prev_field = None
        for config in configs:
            field_name = _config_field(config)
            code_name = _config_code_name(config)
            depth = _config_depth(config)
            row_len = _config_row_len(config)
            field_latex = FIELD_LATEX.get(field_name, field_name)

            if prev_field is not None and prev_field != field_name:
                lines.append(r"\addlinespace")
            prev_field = field_name

            phase_strs = []
            for phase in PHASES:
                key = (phase, poly_exp, config)
                if key in results:
                    _, med, _ = results[key]
                    phase_strs.append(_format_time(med))
                else:
                    phase_strs.append("---")

            lines.append(
                f"{field_latex} & {code_name} & {depth} & {row_len}"
                f" & {phase_strs[0]} & {phase_strs[1]} & {phase_strs[2]} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
        ]

    lines += [
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run rate-1/4 IPRS config search benchmarks and produce LaTeX tables."
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
        help="Write the LaTeX tables to this file (default: stdout).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help=(
            "Criterion filter string to run only specific benchmark groups "
            "(e.g. 'R4 PCS P=8'). Default: run all r4_config_search benchmarks."
        ),
    )
    parser.add_argument(
        "--eval-type",
        choices=["bpoly", "scalar", "all"],
        default="all",
        help="Which evaluation types to benchmark: bpoly (BPoly<31>), scalar (i32), or all (default).",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="parallel asm simd",
        help="Cargo features to enable (default: 'parallel asm simd').",
    )
    parser.add_argument(
        "--per-table",
        action="store_true",
        help="Generate one table per poly_size instead of a single combined table.",
    )
    parser.add_argument(
        "--save-raw",
        type=str,
        default=None,
        help="Save raw cargo bench output to this file.",
    )
    parser.add_argument(
        "--from-criterion",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Read results from criterion's saved JSON data directory "
            "(e.g. target/criterion) instead of parsing terminal output.  "
            "This avoids issues with truncated benchmark IDs."
        ),
    )
    args = parser.parse_args()

    # ── Collect benchmark output ──────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(script_dir)
    bench_output = None

    if args.from_criterion:
        # Read directly from criterion JSON data — no terminal output needed.
        criterion_dir = args.from_criterion
        if not os.path.isdir(criterion_dir):
            print(f"Error: criterion directory not found: {criterion_dir}", file=sys.stderr)
            sys.exit(1)
    elif args.no_run:
        if args.input is None:
            print("Error: --no-run requires --input <file>", file=sys.stderr)
            sys.exit(1)
        with open(args.input) as f:
            bench_output = f.read()
        print(f"Read {len(bench_output)} bytes from {args.input}")
    else:
        cmd = [
            "cargo", "bench",
            "--bench", BENCH_NAME,
            "--features", args.features,
        ]
        if args.filter:
            cmd += ["--", args.filter]

        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {workspace}")
        print("This may take a long time...\n")

        # Use a wide pseudo-terminal so criterion does not truncate
        # the benchmark IDs in its output.
        env = os.environ.copy()
        env["COLUMNS"] = "500"

        proc = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            env=env,
        )

        bench_output = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0:
            print("cargo bench failed:", file=sys.stderr)
            print(bench_output, file=sys.stderr)
            sys.exit(proc.returncode)

        print(bench_output)

        if args.save_raw:
            with open(args.save_raw, "w") as f:
                f.write(bench_output)
            print(f"\nRaw output saved to {args.save_raw}")

    # ── Parse results ─────────────────────────────────────────────────────
    # Primary source: criterion's saved JSON data (full, untruncated IDs).
    # Fallback: terminal output parsing (may miss BPoly Commit/Test due to
    # truncation and has Verify key collisions between BPoly and scalar).
    criterion_dir = getattr(args, "from_criterion", None) or os.path.join(
        workspace, "target", "criterion"
    )
    bpoly_results: dict = {}
    scalar_results: dict = {}

    if os.path.isdir(criterion_dir):
        bpoly_results, scalar_results = parse_criterion_data(criterion_dir)
        total = len(bpoly_results) + len(scalar_results)
        print(f"\nParsed {total} benchmark results from criterion data "
              f"({len(bpoly_results)} BPoly, {len(scalar_results)} scalar).")

    # If criterion data is unavailable or incomplete, fall back to terminal
    # output parsing.
    if bench_output and (not bpoly_results or not scalar_results):
        results = parse_criterion_output(bench_output)
        print(f"\nParsed {len(results)} benchmark results from terminal output.")

        bpoly_keys: set = set()
        scalar_keys: set = set()
        _classify_results(bench_output, bpoly_keys, scalar_keys)

        if not bpoly_results:
            bpoly_results = {k: v for k, v in results.items() if k in bpoly_keys}
        if not scalar_results:
            scalar_results = {k: v for k, v in results.items() if k in scalar_keys}

    if not bpoly_results and not scalar_results:
        print(
            "Warning: no results parsed. Check the benchmark output "
            "or pass --from-criterion target/criterion.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print summary
    for label, res in [("BPoly", bpoly_results), ("Scalar", scalar_results)]:
        for (phase, exp_or_rowlen, config), (lo, med, hi) in sorted(
            res.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
        ):
            print(
                f"  {label:>6} {phase:>8}  2^{exp_or_rowlen:<3} {config:<28}  "
                f"{_format_time(med):>20}"
            )

    # ── Generate LaTeX ────────────────────────────────────────────────────
    tables = []

    if args.eval_type in ("bpoly", "all") and bpoly_results:
        if args.per_table:
            for p in sorted({exp for (phase, exp, _) in bpoly_results if phase in PHASES}):
                tables.append(generate_pcs_table(bpoly_results, p, r"BPoly\textlangle 31\textrangle"))
        else:
            tables.append(generate_combined_table(bpoly_results, r"BPoly\textlangle 31\textrangle"))

    if args.eval_type in ("scalar", "all") and scalar_results:
        tables.append(generate_scalar_table(scalar_results))

    if not tables:
        # Fallback: merge all results
        all_results = {**bpoly_results, **scalar_results}
        if args.per_table:
            all_exps = sorted({exp for (phase, exp, _) in all_results if phase in PHASES})
            for p in all_exps:
                tables.append(generate_pcs_table(all_results, p, "all evaluations"))
        else:
            tables.append(generate_combined_table(all_results, "all evaluations"))

    output_text = "\n\n".join(tables)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text + "\n")
        print(f"\nLaTeX table(s) written to {args.output}")
    else:
        print("\n" + "=" * 80)
        print(output_text)
        print("=" * 80)


def _classify_results(text: str, bpoly_keys: set, scalar_keys: set):
    """
    Re-scan the benchmark output to classify results as BPoly or scalar.

    BPoly benchmarks have "BPoly<31>" in the Commit/Test lines, scalar ones
    have "i32". Verify lines don't include the type name, but we can infer
    from the benchmark group header (R4 PCS P=X vs R4 PCS i32).
    """
    lines = text.splitlines()
    current_group_is_scalar = False

    for i, line in enumerate(lines):
        # Detect group headers (on the SAME line as the benchmark ID in
        # criterion output, e.g. "R4 PCS P=8/Commit: ...")
        if "R4 PCS i32" in line:
            current_group_is_scalar = True
        elif re.search(r"R4 PCS P=\d+", line):
            current_group_is_scalar = False

        # Commit/Test lines
        m = _RE_COMMIT_TEST.search(line)
        if m:
            phase = m.group("phase")
            exp = int(m.group("exp"))
            config = m.group("config")
            key = (phase, exp, config)
            if current_group_is_scalar or "i32" in line:
                scalar_keys.add(key)
            else:
                bpoly_keys.add(key)
            continue

        # Verify lines
        m = _RE_VERIFY.search(line)
        if m:
            exp = int(m.group("exp"))
            config = m.group("config")
            key = ("Verify", exp, config)
            if current_group_is_scalar:
                scalar_keys.add(key)
            else:
                bpoly_keys.add(key)
            continue


if __name__ == "__main__":
    main()
