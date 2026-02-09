#!/usr/bin/env python3
"""
Compute combined Zip+ proof size for SHA-256, summing two components:

  A: n_pol=10, flat_vec_norm=133, degree=32
  B: n_pol=5,  flat_vec_norm=155, degree=1

Uses base_field_size=12 and:
  - depth=1 for k=6,7,8 (N = 2^k)
  - depth=2 for k=9,10

The bitbound is computed as: log2(C) + base_field_size + depth*(3 + base_field_size) - 1
where C = num_columns.

Runs for n_queries=100 and n_queries=147.

Usage:
    python3 scripts/sha256_zip_plus_proof_size.py
    python3 scripts/sha256_zip_plus_proof_size.py --min-exp=6 --max-exp=10
"""

import argparse

BASE_FIELD_SIZE = 12


def depth_for_exp(exp: int) -> int:
    """Return IPRS depth: 1 for k<=8, 2 for k>=9."""
    return 1 if exp <= 8 else 2


import math


def compute_bitbound(num_cols: int, base_field_size: int, depth: int) -> int:
    return int(math.log2(num_cols)) + base_field_size + depth * (3 + base_field_size) - 1


def find_optimal(
    total_entries: int,
    n_pol: int, degree: int, n_queries: int,
    base_field_size: int, depth: int,
    col_cost: int,
) -> tuple[int, int, int]:
    """Find the power-of-2 num_rows that minimises proof size."""
    best = None
    num_rows = 1
    while num_rows <= total_entries:
        num_cols = total_entries // num_rows
        if num_cols < 1:
            break
        bitbound = compute_bitbound(num_cols, base_field_size, depth)
        row_cost = n_pol * degree * n_queries * bitbound
        cost = row_cost * num_rows + col_cost * num_cols
        if best is None or cost < best[2]:
            best = (num_rows, num_cols, cost)
        num_rows *= 2
    assert best is not None
    return best


def fmt_size(bits: int) -> str:
    kb = bits / 8 / 1024
    if kb >= 1024:
        return f"{kb / 1024:.2f} MB"
    return f"{kb:.0f} KB"


COMPONENTS = [
    {"label": "A", "n_pol": 10, "flat_vec_norm": 133, "degree": 32},
    {"label": "B", "n_pol": 5,  "flat_vec_norm": 150, "degree": 1},
]

QUERY_COUNTS = [100, 147, 200, 240]


def main():
    parser = argparse.ArgumentParser(description="SHA-256 Zip+ combined proof size")
    parser.add_argument("--min-exp", type=int, default=6, help="Min exponent for total entries (default: 6)")
    parser.add_argument("--max-exp", type=int, default=17, help="Max exponent for total entries (default: 17)")
    args = parser.parse_args()

    for nq in QUERY_COUNTS:
        print(f"\n{'=' * 70}")
        print(f"  n_queries = {nq}, base_field_size = {BASE_FIELD_SIZE}")
        print(f"{'=' * 70}")
        for c in COMPONENTS:
            print(f"  {c['label']}: n_pol={c['n_pol']}, degree={c['degree']}, "
                  f"flat_vec_norm={c['flat_vec_norm']}")
        print(f"  bitbound(C, depth) = log2(C) + {BASE_FIELD_SIZE} + depth*(3+{BASE_FIELD_SIZE}) - 1")

        hdr = (f"{'2^N':>6} | {'depth':>5} | {'A':>12} | {'B':>12} | {'A + B':>12}")
        print(f"\n{hdr}")
        print("-" * len(hdr))

        for exp in range(args.min_exp, args.max_exp + 1):
            total = 1 << exp
            depth = depth_for_exp(exp)
            costs = []
            for c in COMPONENTS:
                cc = c["flat_vec_norm"]
                _, _, cost = find_optimal(
                    total,
                    c["n_pol"], c["degree"], nq,
                    BASE_FIELD_SIZE, depth,
                    cc,
                )
                costs.append(cost)
            total_cost = sum(costs)
            print(f"  2^{exp:<2} | {depth:>5} | {fmt_size(costs[0]):>12} | {fmt_size(costs[1]):>12} | {fmt_size(total_cost):>12}")

        print()


if __name__ == "__main__":
    main()
