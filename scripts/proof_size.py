#!/usr/bin/env python3
"""
Zip+ Proof Size Optimization
=============================

Compute optimal matrix dimensions (num_rows x num_columns) for Zip+ proof size.

Parameters
----------
  k  (--n-pol)          Number of polynomials batched. Let S^w[X] be the set over which the polynomials are defined
  d  (--degree)         w (typically 32)
  q  (--n-queries)      Number of queries (proximity test repetitions)
  b  (--bitbound)       Max bit-length of a coefficient of an entry in a codeword (43 for polynomials typed in {0,1}^{32}[X] with IPRS codes of depth 1 over Mersenne 2^16+1. 75 for polynomials typed in [0,2^32-1] with the same IPRS code)
  f  (--flat-vec-norm)  Max bit-length of an entry of the flat vector sent by the prover during the IOPP opening proof (261  in {0,1}^{32}[X] domains and  293 in [0,2^32-1])
  N = 2^n              Size of each batcehd polynomial
                        (controlled by --min-exp / --max-exp)

Cost Model
----------
The matrix has dimensions R × C with the constraint:

    R · C = N,    R = 2^i  for some i ∈ ℕ.

Each row contributes a cost from the proximity-test openings (the prover sends
k · d field elements of b bits for each of q queries), so the per-row cost is:

    α = k · d · q · b    (bits/row)

Each column contributes a per-column commitment cost:

    β = f                 (bits/column)

The total proof size in bits is therefore:

    S(R, C) = α · R + β · C = k · d · q · b · R + f · C

Continuous Optimum
------------------
Substituting C = N / R turns this into a single-variable problem:

    S(R) = α · R + β · N / R

Setting dS/dR = 0:

    α − β · N / R² = 0  ⟹  R* = √(β / α · N) = √(f / (k · d · q · b) · N)

The minimum proof size at the continuous optimum is:

    S* = 2 · √(α · β · N)

Discrete Search
---------------
Because R must be a power of two, the script enumerates every R = 2^i with
1 ≤ R ≤ N (optionally skipping any R that would make C > C_max), evaluates
S(R, N/R), and returns the (R, C) pair that yields the smallest proof size.

Usage
-----
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
    parser.add_argument("--max-columns", type=int, default=None, help="Maximumπ number of columns (optional)")
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
