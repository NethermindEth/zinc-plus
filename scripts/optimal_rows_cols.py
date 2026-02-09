#!/usr/bin/env python3
"""Find optimal (num_rows, num_cols) minimizing proof size for 2^k polynomials."""

import argparse


def proof_size_bits(n_queries, num_rows, num_cols, degree):
    return n_queries  * 128 * num_rows + num_cols * 256 + n_queries * degree * 128


def find_optimal(k, n_queries, degree):
    n = 1 << k  # 2^k
    best = None
    for r in range(k + 1):
        num_rows = 1 << r
        num_cols = n >> r  # n / num_rows
        cost = proof_size_bits(n_queries, num_rows, num_cols, degree)
        if best is None or cost < best[0]:
            best = (cost, num_rows, num_cols)
    return best


def main():
    parser = argparse.ArgumentParser(description="Optimal rows/cols for proof size")
    parser.add_argument("--n-queries", type=int, default=110)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--min-exp", type=int, default=10)
    parser.add_argument("--max-exp", type=int, default=17)
    args = parser.parse_args()

    print(f"n_queries={args.n_queries}, degree={args.degree}")
    print(f"{'k':>4} {'num_rows':>10} {'num_cols':>10} {'proof_size (KB)':>16}")
    print("-" * 44)

    for k in range(args.min_exp, args.max_exp + 1):
        cost_bits, num_rows, num_cols = find_optimal(k, args.n_queries, args.degree)
        cost_kb = cost_bits / 8 / 1024
        print(f"{k:>4} {num_rows:>10} {num_cols:>10} {cost_kb:>16.2f}")


if __name__ == "__main__":
    main()
