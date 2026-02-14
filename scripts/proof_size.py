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

    row_cost(num_cols) = n_pol · degree · n_queries · bitbound(num_cols)

where bitbound(num_cols) is computed using one of two formulas:

  v1: log2(C) + base_field_size + depth*(3 + base_field_size) - 1
  v2: log2(C)/2 + (depth+1)*base_field_size + 0.5*(1+4*depth) - log2(3*pi)*0.5*(1+depth)

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


def compute_bitbound_v1(num_cols: int, base_field_size: int, depth: int) -> float:
    return math.log2(num_cols) + base_field_size + depth * (3 + base_field_size) - 1


def compute_bitbound_v2(num_cols: int, base_field_size: int, depth: int) -> float:
    return (
        math.log2(num_cols) / 2
        + (depth + 1) * base_field_size
        + 0.5 * (1 + 4 * depth)
        - math.log2(3 * math.pi) * 0.5 * (1 + depth)
    )


BITBOUND_FORMULAS = {
    "v1": compute_bitbound_v1,
    "v2": compute_bitbound_v2,
}


def proof_size(
    n_pol: int, degree: int, n_queries: int,
    base_field_size: int, depth: int,
    col_cost: int, num_rows: int, num_cols: int,
    bitbound_fn=compute_bitbound_v1,
) -> float:
    bitbound = bitbound_fn(num_cols, base_field_size, depth)
    row_cost = n_pol * degree * n_queries * bitbound
    return row_cost * num_rows + col_cost * num_cols


def find_optimal(
    total_entries: int,
    n_pol: int, degree: int, n_queries: int,
    base_field_size: int, depth: int,
    col_cost: int,
    max_columns: int | None = None,
    bitbound_fn=compute_bitbound_v1,
) -> tuple[int, int, float]:
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
            bitbound_fn=bitbound_fn,
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
    parser.add_argument("--max-columns", type=int, default=None, help="Maximum number of columns (optional)")
    parser.add_argument("--num-rows", type=int, default=None, help="Fix the number of rows instead of optimising (must be a power of 2)")
    args = parser.parse_args()

    col_cost = args.flat_vec_norm

    print(f"Parameters:")
    print(f"  n_pol={args.n_pol}, degree={args.degree}, n_queries={args.n_queries}, base_field_size={args.base_field_size}, depth={args.depth}, flat_vec_norm={args.flat_vec_norm}")
    print(f"  v1: log2(C) + {args.base_field_size} + {args.depth}*(3+{args.base_field_size}) - 1")
    print(f"  v2: log2(C)/2 + ({args.depth}+1)*{args.base_field_size} + 0.5*(1+4*{args.depth}) - log2(3*pi)*0.5*(1+{args.depth})")
    print(f"  Row cost(C) = {args.n_pol} * {args.degree} * {args.n_queries} * bitbound(C) = {args.n_pol * args.degree * args.n_queries} * bitbound(C) bits/row")
    print(f"  Column cost = {col_cost:,} bits/column")
    if args.max_columns is not None:
        print(f"  Max columns = {args.max_columns:,}")
    if args.num_rows is not None:
        print(f"  Fixed num_rows = {args.num_rows:,}")
    print()

    # Header
    hdr = (f"{'Total Entries':>15} | {'rows(v1)':>10} | {'cols(v1)':>10} | {'v1 Size':>12} | {'v1 Bits':>15}"
           f" | {'rows(v2)':>10} | {'cols(v2)':>10} | {'v2 Size':>12} | {'v2 Bits':>15}")
    print(hdr)
    print("-" * len(hdr))

    for exp in range(args.min_exp, args.max_exp + 1):
        total = 1 << exp
        if args.num_rows is not None:
            nr = args.num_rows
            nc = total // nr
            if nc < 1 or nr * nc != total:
                print(f"  2^{exp} = {total:>7,} | {'N/A':>10} | {'N/A':>10} | {'N/A':>12} | {'N/A':>15}"
                      f" | {'N/A':>10} | {'N/A':>10} | {'N/A':>12} | {'N/A':>15}")
                continue
            cost1 = proof_size(
                args.n_pol, args.degree, args.n_queries,
                args.base_field_size, args.depth,
                col_cost, nr, nc,
                bitbound_fn=compute_bitbound_v1,
            )
            cost2 = proof_size(
                args.n_pol, args.degree, args.n_queries,
                args.base_field_size, args.depth,
                col_cost, nr, nc,
                bitbound_fn=compute_bitbound_v2,
            )
            rows1, cols1, rows2, cols2 = nr, nc, nr, nc
        else:
            rows1, cols1, cost1 = find_optimal(
                total,
                args.n_pol, args.degree, args.n_queries,
                args.base_field_size, args.depth,
                col_cost, args.max_columns,
                bitbound_fn=compute_bitbound_v1,
            )
            rows2, cols2, cost2 = find_optimal(
                total,
                args.n_pol, args.degree, args.n_queries,
                args.base_field_size, args.depth,
                col_cost, args.max_columns,
                bitbound_fn=compute_bitbound_v2,
            )
        print(f"  2^{exp} = {total:>7,} | {rows1:>10,} | {cols1:>10,} | {fmt_size(cost1):>12} | {cost1:>15,.0f}"
              f" | {rows2:>10,} | {cols2:>10,} | {fmt_size(cost2):>12} | {cost2:>15,.0f}")


if __name__ == "__main__":
    main()
