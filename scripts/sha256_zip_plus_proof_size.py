#!/usr/bin/env python3
"""
Compute combined Zip+ proof size for SHA-256, summing two components:

  A: n_pol=10, bitbound=43, flat_vec_norm=261, degree=32
  B: n_pol=5,  bitbound=75, flat_vec_norm=293, degree=1

Runs for n_queries=100 and n_queries=147.

Usage:
    python3 scripts/sha256_zip_plus_proof_size.py
    python3 scripts/sha256_zip_plus_proof_size.py --min-exp=10 --max-exp=20
"""

import argparse


def find_optimal(total_entries: int, row_cost: int, col_cost: int) -> tuple[int, int, int]:
    """Find the power-of-2 num_rows that minimises proof size."""
    best = None
    num_rows = 1
    while num_rows <= total_entries:
        num_cols = total_entries // num_rows
        if num_cols < 1:
            break
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
    {"label": "A", "n_pol": 10, "bitbound": 43, "flat_vec_norm": 261, "degree": 32},
    {"label": "B", "n_pol": 5,  "bitbound": 75, "flat_vec_norm": 293, "degree": 1},
]

QUERY_COUNTS = [100, 147]


def main():
    parser = argparse.ArgumentParser(description="SHA-256 Zip+ combined proof size")
    parser.add_argument("--min-exp", type=int, default=6, help="Min exponent for total entries (default: 6)")
    parser.add_argument("--max-exp", type=int, default=17, help="Max exponent for total entries (default: 17)")
    args = parser.parse_args()

    for nq in QUERY_COUNTS:
        print(f"\n{'=' * 70}")
        print(f"  n_queries = {nq}")
        print(f"{'=' * 70}")
        for c in COMPONENTS:
            rc = c["n_pol"] * c["degree"] * nq * c["bitbound"]
            print(f"  {c['label']}: n_pol={c['n_pol']}, bitbound={c['bitbound']}, "
                  f"flat_vec_norm={c['flat_vec_norm']}  =>  row_cost={rc:,}")

        hdr = (f"{'2^N':>6} | {'A':>12} | {'B':>12} | {'A + B':>12}")
        print(f"\n{hdr}")
        print("-" * len(hdr))

        for exp in range(args.min_exp, args.max_exp + 1):
            total = 1 << exp
            costs = []
            for c in COMPONENTS:
                rc = c["n_pol"] * c["degree"] * nq * c["bitbound"]
                cc = c["flat_vec_norm"]
                _, _, cost = find_optimal(total, rc, cc)
                costs.append(cost)
            total_cost = sum(costs)
            print(f"  2^{exp:<2} | {fmt_size(costs[0]):>12} | {fmt_size(costs[1]):>12} | {fmt_size(total_cost):>12}")

        print()


if __name__ == "__main__":
    main()
