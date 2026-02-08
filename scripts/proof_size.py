#!/usr/bin/env python3
"""
Compute optimal matrix dimensions (num_rows x num_columns) for Zip+ proof size.

Formula:
    proof_size = n_pol * degree * n_queries * bitbound * num_rows + flat_vec_norm * num_columns

Subject to:
    num_rows * num_columns = total_entries
    num_rows is a power of 2
    num_columns <= max_columns (optional)


python3 scripts/proof_size.py --n-pol=2 --n-queries=200 --bitbound=60 --flat-vec-norm=256 --min-exp=10 --max-exp=17 --degree=32

"""

import math
import argparse


def proof_size(row_cost: int, col_cost: int, num_rows: int, num_cols: int) -> int:
    return row_cost * num_rows + col_cost * num_cols


def find_optimal(
    total_entries: int,
    row_cost: int,
    col_cost: int,
    max_columns: int | None = None,
) -> tuple[int, int, int]:
    """Find the power-of-2 num_rows that minimises proof size."""
    best = None
    # Try all power-of-2 row counts from 1 up to total_entries
    num_rows = 1
    while num_rows <= total_entries:
        num_cols = total_entries // num_rows
        if num_cols < 1:
            break
        if max_columns is not None and num_cols > max_columns:
            num_rows *= 2
            continue
        cost = proof_size(row_cost, col_cost, num_rows, num_cols)
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


def main():
    parser = argparse.ArgumentParser(description="Zip+ proof size optimizer")
    parser.add_argument("--n-pol", type=int, default=1, help="Number of polynomials (default: 1)")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries (default: 100)")
    parser.add_argument("--bitbound", type=int, default=64, help="Bit bound per element (default: 64)")
    parser.add_argument("--flat-vec-norm", type=int, default=128, help="Bits per column (default: 128)")
    parser.add_argument("--min-exp", type=int, default=13, help="Min exponent for total entries (default: 13)")
    parser.add_argument("--max-exp", type=int, default=17, help="Max exponent for total entries (default: 17)")
    parser.add_argument("--degree", type=int, default=32, help="degree (default: 32)")
    parser.add_argument("--max-columns", type=int, default=None, help="Maximum number of columns (optional)")
    args = parser.parse_args()

    row_cost = args.n_pol * args.degree * args.n_queries * args.bitbound
    col_cost = args.flat_vec_norm

    # Continuous optimum formula
    opt_ratio = col_cost / row_cost
    print(f"Parameters:")
    print(f"  n_pol={args.n_pol}, degree={args.degree}, n_queries={args.n_queries}, bitbound={args.bitbound}, flat_vec_norm={args.flat_vec_norm}")
    print(f"  Row cost = {row_cost:,} bits/row")
    print(f"  Column cost = {col_cost:,} bits/column")
    print(f"  Optimal num_rows (continuous) = sqrt(N * {opt_ratio:.6f}) = sqrt(N / {row_cost / col_cost:.0f})")
    if args.max_columns is not None:
        print(f"  Max columns = {args.max_columns:,}")
    print()

    # Header
    print(f"{'Total Entries':>15} | {'Best rows':>10} | {'Columns':>10} | {'Proof Size':>12} | {'Bits':>15}")
    print("-" * 72)

    for exp in range(args.min_exp, args.max_exp + 1):
        total = 1 << exp
        rows, cols, cost = find_optimal(total, row_cost, col_cost, args.max_columns)
        print(f"  2^{exp} = {total:>7,} | {rows:>10,} | {cols:>10,} | {fmt_size(cost):>12} | {cost:>15,}")


if __name__ == "__main__":
    main()
