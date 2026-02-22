#!/usr/bin/env python3
"""Compute exact proof size breakdown for the 8xSHA256+ECDSA dual-PCS benchmark."""

import math

# Configuration
row_len = 512
codeword_len = 512 * 4      # rate 1/4 = 2048
num_rows = 1                 # DEPTH=1, R4B64, poly_size=512
num_queries = 64

merkle_height = int(math.log2(codeword_len))   # 11
merkle_proof = 3 * 8 + (merkle_height - 1) * 32   # 24 + 320 = 344

# ═══════════════════════════════════════════════════
# SHA PCS batch: 20 columns, Cw = DensePolynomial<i64, 32>
# ═══════════════════════════════════════════════════
sha_batch = 20
sha_cw_bytes = 8 * 32       # 256 bytes per codeword element
sha_combr_bytes = 8 * 6     # Int<6> = 48 bytes
sha_f_bytes = 32             # MontyField<4> = 256-bit

sha_combined_row = row_len * sha_combr_bytes
sha_col_openings = num_queries * (sha_batch * num_rows * sha_cw_bytes + merkle_proof)
sha_test_phase = sha_combined_row + sha_col_openings
sha_eval_phase = sha_batch * (1 + row_len * sha_f_bytes)
sha_total = sha_test_phase + sha_eval_phase

print("SHA PCS PROOF BREAKDOWN")
print(f"  Test phase:")
print(f"    combined_row:    {sha_combined_row:>8} bytes  ({sha_combined_row/1024:.1f} KB)")
print(f"    column openings: {sha_col_openings:>8} bytes  ({sha_col_openings/1024:.1f} KB)")
print(f"      per query: {sha_batch}*{sha_cw_bytes} = {sha_batch*sha_cw_bytes} col + {merkle_proof} merkle = {sha_batch*sha_cw_bytes + merkle_proof}")
print(f"    test total:      {sha_test_phase:>8} bytes  ({sha_test_phase/1024:.1f} KB)")
print(f"  Eval phase:        {sha_eval_phase:>8} bytes  ({sha_eval_phase/1024:.1f} KB)")
print(f"  SHA TOTAL:         {sha_total:>8} bytes  ({sha_total/1024:.1f} KB)")
print()

# ═══════════════════════════════════════════════════
# ECDSA PCS batch: 14 columns, Cw = Int<5>
# ═══════════════════════════════════════════════════
ec_batch = 14
ec_cw_bytes = 8 * 5          # Int<5> = 40 bytes
ec_combr_bytes = 8 * 8       # Int<8> = 64 bytes
ec_f_bytes = 64              # MontyField<8> = 512-bit

ec_combined_row = row_len * ec_combr_bytes
ec_col_openings = num_queries * (ec_batch * num_rows * ec_cw_bytes + merkle_proof)
ec_test_phase = ec_combined_row + ec_col_openings
ec_eval_phase = ec_batch * (1 + row_len * ec_f_bytes)
ec_total = ec_test_phase + ec_eval_phase

print("ECDSA PCS PROOF BREAKDOWN")
print(f"  Test phase:")
print(f"    combined_row:    {ec_combined_row:>8} bytes  ({ec_combined_row/1024:.1f} KB)")
print(f"    column openings: {ec_col_openings:>8} bytes  ({ec_col_openings/1024:.1f} KB)")
print(f"      per query: {ec_batch}*{ec_cw_bytes} = {ec_batch*ec_cw_bytes} col + {merkle_proof} merkle = {ec_batch*ec_cw_bytes + merkle_proof}")
print(f"    test total:      {ec_test_phase:>8} bytes  ({ec_test_phase/1024:.1f} KB)")
print(f"  Eval phase:        {ec_eval_phase:>8} bytes  ({ec_eval_phase/1024:.1f} KB)")
print(f"  ECDSA TOTAL:       {ec_total:>8} bytes  ({ec_total/1024:.1f} KB)")
print()

combined = sha_total + ec_total
print(f"COMBINED TOTAL: {combined} bytes ({combined/1024:.1f} KB)")
print()

# ═══════════════════════════════════════════════════
# ROOT CAUSE: Codeword serialization waste
# ═══════════════════════════════════════════════════
print("ROOT CAUSE: CODEWORD TYPE BYTE SIZES")
print(f"  BinaryPoly<32> has 32 binary coefficients = 32 bits = 4 bytes of info")
print(f"  But Cw = DensePolynomial<i64,32> = 32 x 8 = 256 bytes per element")
print(f"  WASTE: 256 / 4 = 64x")
print()
print(f"  Int<4> = 32 bytes, Cw = Int<5> = 40 bytes")
print(f"  WASTE: 40 / 32 = 1.25x (reasonable)")
print()

# What if BinaryPoly Cw used compact 4-byte serialization?
compact_cw = 4
compact_col = num_queries * (sha_batch * compact_cw + merkle_proof)
compact_test = sha_combined_row + compact_col
compact_sha = compact_test + sha_eval_phase
print(f"IF Cw were packed to 4 bytes per BinaryPoly<32>:")
print(f"  SHA column openings: {compact_col} bytes ({compact_col/1024:.1f} KB) vs {sha_col_openings} ({sha_col_openings/1024:.1f} KB)")
print(f"  SHA total:           {compact_sha} bytes ({compact_sha/1024:.1f} KB)")
print(f"  Combined:            {compact_sha + ec_total} bytes ({(compact_sha + ec_total)/1024:.1f} KB)")
print()

# What does the paper's model assume?
# 14 polynomials, Cw coefficients = 16-bit field elements, 32 coefficients per poly
# So codeword element = 32 * 2 = 64 bytes
print("PAPER'S MODEL (14 polys, Cw = 32 * 2B = 64B, 100 queries)")
nq = 100
bp = 14
pcw = 64
p_col = nq * (bp * pcw + merkle_proof)
p_combined = row_len * sha_combr_bytes
p_test = p_combined + p_col
p_eval = bp * (1 + row_len * 32)
p_total = p_test + p_eval
print(f"  column openings: {p_col} ({p_col/1024:.1f} KB)")
print(f"  test total:      {p_test} ({p_test/1024:.1f} KB)")
print(f"  eval total:      {p_eval} ({p_eval/1024:.1f} KB)")
print(f"  TOTAL:           {p_total} ({p_total/1024:.1f} KB)")
