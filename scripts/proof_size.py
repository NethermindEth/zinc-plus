#!/usr/bin/env python3
"""
Zip+ Proof Size Optimization
=============================

Compute optimal matrix dimensions (num_rows x num_columns) for Zip+ proof size.

Parameters
----------
  n_pol            (--n-pol)            Number of polynomials batched
  degree           (--degree)           Degree bound (typically 32)
  n_queries        (--n-queries)        Number of queries (proximity test repetitions)
  base_field_size  (--base-field-size)  Bit-length of a base field element (e.g. 16 for Mersenne 2^16+1)
  depth            (--depth)            Depth of the IPRS code (e.g. 1 or 2)
  bitbound         (derived)            Bit-bound per codeword entry coefficient, computed as
                                          log2(num_cols) + base_field_size + depth*(3 + base_field_size) - 1
  flat_vec_norm    (--flat-vec-norm)    Max bit-length of an entry of the flat vector sent by the
                                        prover during the IOPP opening proof
  N = 2^n                               Size of each batched polynomial
                                          (controlled by --min-exp / --max-exp)

Cost Model
----------
The matrix has dimensions num_rows × num_cols with the constraint:

    num_rows · num_cols = N,    num_rows = 2^i  for some i ∈ ℕ.

Each row contributes a cost from the proximity-test openings (the prover sends
n_pol · degree field elements of bitbound bits for each of n_queries queries),
so the per-row cost is:

    row_cost(num_cols) = n_pol · degree · n_queries
                         · (log2(num_cols) + base_field_size + depth·(3 + base_field_size) − 1)

Each column contributes a per-column commitment cost:

    col_cost = flat_vec_norm                              (bits/column)

The total proof size in bits is therefore:

    S(num_rows, num_cols) = row_cost(num_cols) · num_rows + flat_vec_norm · num_cols

Continuous Optimum
------------------
Because row_cost depends on num_cols = N / num_rows, the cost function is no
longer separable in a simple closed form. The script relies on the discrete
search below.

Discrete Search
---------------
Because num_rows must be a power of two, the script enumerates every
num_rows = 2^i with 1 ≤ num_rows ≤ N (optionally skipping any num_rows that
would make num_cols > max_columns), evaluates S(num_rows, N / num_rows), and
returns the (num_rows, num_cols) pair that yields the smallest proof size.

Usage
-----
python3 scripts/proof_size.py --n-pol=2 --n-queries=200 --base-field-size=16 --depth=1 --flat-vec-norm=256 --min-exp=10 --max-exp=17 --degree=32

"""

import math
import argparse


def compute_bitbound(num_cols: int, base_field_size: int, depth: int) -> int:
    return int(math.log2(num_cols)) + base_field_size + depth * (3 + base_field_size) - 1


def proof_size(
    n_pol: int, degree: int, n_queries: int,
    base_field_size: int, depth: int,
    col_cost: int, num_rows: int, num_cols: int,
) -> int:
    bitbound = compute_bitbound(num_cols, base_field_size, depth)
    row_cost = n_pol * degree * n_queries * bitbound
    return row_cost * num_rows + col_cost * num_cols


def find_optimal(
    total_entries: int,
    n_pol: int, degree: int, n_queries: int,
    base_field_size: int, depth: int,
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
        cost = proof_size(
            n_pol, degree, n_queries,
            base_field_size, depth,
            col_cost, num_rows, num_cols,
        )
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
    parser.add_argument("--base-field-size", type=int, default=16, help="Bit-length of a base field element (default: 16)")
    parser.add_argument("--depth", type=int, default=1, help="Depth of the IPRS code (default: 1)")
    parser.add_argument("--flat-vec-norm", type=int, default=128, help="Bits per column (default: 128)")
    parser.add_argument("--min-exp", type=int, default=13, help="Min exponent for total entries (default: 13)")
    parser.add_argument("--max-exp", type=int, default=17, help="Max exponent for total entries (default: 17)")
    parser.add_argument("--degree", type=int, default=32, help="degree (default: 32)")
    parser.add_argument("--max-columns", type=int, default=None, help="Maximumπ number of columns (optional)")
    args = parser.parse_args()

    col_cost = args.flat_vec_norm

    print(f"Parameters:")
    print(f"  n_pol={args.n_pol}, degree={args.degree}, n_queries={args.n_queries}, base_field_size={args.base_field_size}, depth={args.depth}, flat_vec_norm={args.flat_vec_norm}")
    print(f"  bitbound(C) = log2(C) + {args.base_field_size} + {args.depth}*(3+{args.base_field_size}) - 1 = log2(C) + {args.base_field_size + args.depth * (3 + args.base_field_size) - 1}")
    print(f"  Row cost(C) = {args.n_pol} * {args.degree} * {args.n_queries} * bitbound(C) = {args.n_pol * args.degree * args.n_queries} * bitbound(C) bits/row")
    print(f"  Column cost = {col_cost:,} bits/column")
    if args.max_columns is not None:
        print(f"  Max columns = {args.max_columns:,}")
    print()

    # Header
    print(f"{'Total Entries':>15} | {'Best rows':>10} | {'Columns':>10} | {'Proof Size':>12} | {'Bits':>15}")
    print("-" * 72)

    for exp in range(args.min_exp, args.max_exp + 1):
        total = 1 << exp
        rows, cols, cost = find_optimal(
            total,
            args.n_pol, args.degree, args.n_queries,
            args.base_field_size, args.depth,
            col_cost, args.max_columns,
        )
        print(f"  2^{exp} = {total:>7,} | {rows:>10,} | {cols:>10,} | {fmt_size(cost):>12} | {cost:>15,}")


if __name__ == "__main__":
    main()
