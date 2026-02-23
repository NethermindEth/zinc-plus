#!/usr/bin/env python3
"""Compare ECDSA PCS proof sizes: old BinaryPoly<32> vs new Int<4>."""
import math

num_queries = 147
row_len = 128
merkle_height = int(math.log2(row_len * 4))  # rate 1/4
merkle_proof = 3 * 8 + (merkle_height - 1) * 32
num_cols = 14

# OLD: BinaryPoly<32> with Cw = DensePolynomial<i64, 32>
old_cw = 8 * 32  # 256 bytes
old_combr = 8 * 6  # Int<6> = 48 bytes
old_f = 32  # MontyField<4> = 256-bit = 32 bytes
old_combined_row = row_len * old_combr
old_col_openings = num_queries * (num_cols * old_cw + merkle_proof)
old_test = old_combined_row + old_col_openings
old_eval = num_cols * (1 + row_len * old_f)
old_total = old_test + old_eval

# NEW: Int<4> with Cw = Int<5>
new_cw = 8 * 5  # 40 bytes
new_combr = 8 * 8  # Int<8> = 64 bytes
new_f = 64  # MontyField<8> = 512-bit = 64 bytes
new_combined_row = row_len * new_combr
new_col_openings = num_queries * (num_cols * new_cw + merkle_proof)
new_test = new_combined_row + new_col_openings
new_eval = num_cols * (1 + row_len * new_f)
new_total = new_test + new_eval

print("ECDSA PCS PROOF SIZE COMPARISON (num_vars=7, 14 cols, 147 queries)")
print()
print("                     OLD (BinaryPoly<32>)    NEW (Int<4>)")
print(f"  Codeword size:     {old_cw:>6} B/elem            {new_cw:>6} B/elem")
print(f"  Combined row:      {old_combined_row:>8} B ({old_combined_row/1024:.1f} KB)    {new_combined_row:>8} B ({new_combined_row/1024:.1f} KB)")
print(f"  Column openings:   {old_col_openings:>8} B ({old_col_openings/1024:.1f} KB)   {new_col_openings:>8} B ({new_col_openings/1024:.1f} KB)")
print(f"  Test phase:        {old_test:>8} B ({old_test/1024:.1f} KB)   {new_test:>8} B ({new_test/1024:.1f} KB)")
print(f"  Eval phase:        {old_eval:>8} B ({old_eval/1024:.1f} KB)   {new_eval:>8} B ({new_eval/1024:.1f} KB)")
print(f"  PCS TOTAL:         {old_total:>8} B ({old_total/1024:.1f} KB)   {new_total:>8} B ({new_total/1024:.1f} KB)")
print()
print(f"  Column opening savings: {old_col_openings - new_col_openings} B ({(old_col_openings-new_col_openings)/1024:.1f} KB) = {old_col_openings/new_col_openings:.1f}x smaller")
print(f"  Overall PCS savings:    {old_total - new_total} B ({(old_total-new_total)/1024:.1f} KB) = {old_total/new_total:.2f}x smaller")
print()
print(f"  Actual measured NEW proof: 145464 B (142.1 KB)")
