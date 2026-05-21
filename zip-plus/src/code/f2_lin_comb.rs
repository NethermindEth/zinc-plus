//! `F_2[X]`-coefficient linear combination of `F_2`-RAA codewords.
//!
//! Given:
//!
//! - codeword columns whose cells are `F_2[X]<32>`-typed (each row of
//!   the commit matrix, opened at a fixed column index), and
//! - per-row coefficients drawn from `F_2[X]<128>`,
//!
//! we produce the per-column linear combination
//!
//! ```text
//! out[col] = Σ_j coeffs[j] · cells[j][col]     (over F_2[X])
//! ```
//!
//! where the product `F_2[X]<32> · F_2[X]<128>` lives in `F_2[X]<160>`
//! (degrees add: 31 + 127 = 158, so `< 160` holds with margin) and
//! XOR-summing many such products stays within `F_2[X]<160>`. The
//! result type therefore takes 3 × `u64` words.
//!
//! This is the new linear-combination primitive for the F_2-RAA
//! commit lane. It replaces the integer-arithmetic combined-row
//! computation (`Σ_j β_j · cw_j` over `Z`) used by the legacy
//! RAA path.

use zinc_poly::univariate::{
    F2PackU64,
    binary::BinaryPoly,
    binary_f2_wide::BinaryF2Poly,
};

/// `F_2[X]<32>`. Alias for clarity inside this module.
pub type F2X32 = BinaryPoly<32>;
/// `F_2[X]<128>`. Coefficients of the linear combination.
pub type F2X128 = BinaryF2Poly<2>;
/// `F_2[X]<160>`. Entries of the combined row (`128 + 32 = 160`).
pub type F2X160 = BinaryF2Poly<3>;

/// Compute the F_2[X] linear combination
/// `out[col] = XOR_j coeffs[j] · cells[j][col]`.
///
/// `cells` is a slice of rows; each row has length `row_len`. The
/// flattened layout is row-major: `cells[j * row_len + col]`. The
/// number of rows must equal `coeffs.len()`.
///
/// `out.len() == row_len`. Each entry is in `F_2[X]<160>`.
///
/// Implementation notes:
/// - Cells are bit-packed to `u32` once up front (one Boolean walk
///   per cell amortized across all the row's multiplies), avoiding
///   per-`BinaryPoly` Boolean-array iteration in the inner loop.
/// - The output accumulator is kept as `Vec<[u64; 3]>` raw words for
///   the duration; it's converted back to `F2X160` at the very end.
/// - The 32×128 carryless multiplication is inlined via
///   [`xor_clmul_32x128`], which walks set bits of the 32-bit cell
///   with `trailing_zeros` — `O(popcount(cell))` shifted XORs per
///   multiplication. Zero cells short-circuit before the bit walk.
/// - Loop order is `j` (row) outer, `col` inner. Each pass reads a
///   contiguous slice of `cells_packed` and writes to a contiguous
///   slice of the accumulator — cache-friendly. Inverting the loops
///   to parallelise per-column gave a strided-read pattern on the
///   `cells_packed` array and ~50% slowdown in practice.
#[allow(clippy::arithmetic_side_effects)]
pub fn f2_lin_comb(cells: &[F2X32], coeffs: &[F2X128], row_len: usize) -> Vec<F2X160> {
    assert!(row_len > 0, "f2_lin_comb: row_len must be > 0");
    assert_eq!(
        cells.len(),
        coeffs.len() * row_len,
        "f2_lin_comb: cells.len() ({}) must equal coeffs.len() * row_len ({} * {} = {})",
        cells.len(),
        coeffs.len(),
        row_len,
        coeffs.len() * row_len,
    );

    // Pack each cell to its 32-bit pattern once. The `F2PackU64` impls
    // for `BinaryRefPoly<D ≤ 64>` walk Booleans here; this is the only
    // place that cost is paid.
    let cells_packed: Vec<u32> = cells.iter().map(|c| c.pack_u64() as u32).collect();

    // Accumulator in raw words: 3 × u64 per column, contiguous.
    let mut acc: Vec<[u64; 3]> = vec![[0u64; 3]; row_len];

    for (j, coeff) in coeffs.iter().enumerate() {
        let cw = coeff.words();
        let (lo, hi) = (cw[0], cw[1]);
        let row_start = j * row_len;
        let row = &cells_packed[row_start..row_start + row_len];
        for (col, &cell) in row.iter().enumerate() {
            if cell == 0 {
                continue;
            }
            xor_clmul_32x128(&mut acc[col], cell, lo, hi);
        }
    }

    acc.into_iter().map(F2X160::from_words).collect()
}

/// XOR `cell * (hi:lo)` into the 3-word accumulator `acc`, where:
/// - `cell` is an `F_2[X]<32>` value bit-packed in a `u32` (bit `i`
///   = coefficient of `X^i`),
/// - `(lo, hi)` is an `F_2[X]<128>` value bit-packed in two `u64`s
///   (bit `i` of `lo` = coefficient of `X^i`; bit `i` of `hi` =
///   coefficient of `X^{64+i}`).
///
/// The product has degree ≤ 31 + 127 = 158, so it fits in `acc`
/// (the high 2 bits of `acc[2]` are always zero — bits 159..192).
///
/// Inner loop: O(popcount(cell)) shifted XORs. Each shifted XOR
/// touches all 3 words of `acc`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn xor_clmul_32x128(acc: &mut [u64; 3], cell: u32, lo: u64, hi: u64) {
    let mut bits = cell;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize; // 0..=31
        // XOR (lo, hi, 0) << i into acc.
        if i == 0 {
            acc[0] ^= lo;
            acc[1] ^= hi;
        } else {
            // Bit i in [1, 31]: shift the 128-bit coefficient left by
            // i bits and XOR into acc[0..3]. For 32-bit `cell`, `i`
            // never reaches 64 so there are no zero-shift edge cases.
            acc[0] ^= lo << i;
            acc[1] ^= (lo >> (64 - i)) ^ (hi << i);
            acc[2] ^= hi >> (64 - i);
        }
        // Clear the LSB.
        bits &= bits - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::boolean::Boolean;
    use zinc_poly::univariate::binary_f2_wide::f2_poly_mul;

    fn bp32(bits: u32) -> F2X32 {
        F2X32::from(bits)
    }

    fn bp128(lo: u64, hi: u64) -> F2X128 {
        BinaryF2Poly::from_words([lo, hi])
    }

    #[test]
    fn empty_coeffs_or_zero_cells_yields_zero() {
        let row_len = 4;
        let cells = vec![F2X32::from(0u32); row_len * 3];
        let coeffs = vec![bp128(0xDEAD_BEEF, 0xCAFE_F00D); 3];
        let out = f2_lin_comb(&cells, &coeffs, row_len);
        assert_eq!(out.len(), row_len);
        for v in &out {
            assert!(v.is_zero());
        }
    }

    #[test]
    fn single_row_is_just_scalar_mul() {
        // 1 row, row_len = 2: out[col] = coeff · cell[col].
        let row_len = 2;
        let cells = vec![bp32(0xABCD), bp32(0x1234)];
        let c = bp128(0x9E37_79B1_DEAD_BEEF, 0xCAFE_F00D_F00D_BAAD);
        let out = f2_lin_comb(&cells, &[c], row_len);

        // Reproduce by hand: out[col] = lift32(cells[col]) · c, using
        // the generic `f2_poly_mul` as a reference oracle.
        for (col, &cell) in cells.iter().enumerate() {
            let cell_wide: BinaryF2Poly<1> =
                BinaryF2Poly::from_words([cell.pack_u64()]);
            let expected: F2X160 = f2_poly_mul(&cell_wide, &c);
            assert_eq!(out[col], expected);
        }
    }

    #[test]
    fn matches_reference_generic_mul() {
        // Compare against the generic schoolbook multiplier for a
        // wider workload — catches any drift between the inlined
        // 32×128 path and the reference 32×128 oracle.
        let row_len = 5;
        let num_rows = 7;
        let cells: Vec<F2X32> = (0..(row_len * num_rows) as u32)
            .map(|i| bp32(i.wrapping_mul(0xA5A5_A5A5).wrapping_add(0x12345)))
            .collect();
        let coeffs: Vec<F2X128> = (0..num_rows as u64)
            .map(|j| bp128(j.wrapping_mul(0x9E37_79B1), j.wrapping_mul(0xDEAD_BEEF)))
            .collect();

        let got = f2_lin_comb(&cells, &coeffs, row_len);

        let mut expected = vec![F2X160::zero(); row_len];
        for (j, &coeff) in coeffs.iter().enumerate() {
            for (col, cell) in cells[j * row_len..(j + 1) * row_len].iter().enumerate() {
                let cell_wide: BinaryF2Poly<1> =
                    BinaryF2Poly::from_words([cell.pack_u64()]);
                let prod: F2X160 = f2_poly_mul(&cell_wide, &coeff);
                expected[col] += &prod;
            }
        }

        assert_eq!(got, expected);
    }

    #[test]
    fn linearity_in_cells() {
        let row_len = 3;
        let num_rows = 4;
        let n = row_len * num_rows;
        let a: Vec<F2X32> = (0..n as u32).map(|i| bp32(i.wrapping_mul(0xA5A5_A5A5))).collect();
        let b: Vec<F2X32> = (0..n as u32).map(|i| bp32(i.wrapping_mul(0xDEAD_BEEF))).collect();
        let c: Vec<F2X128> = (0..num_rows as u64).map(|j| bp128(j ^ 0xC0FFEE, j ^ 0xBEEFCAFE)).collect();

        // F_2 XOR of a and b via the cell pack.
        let a_xor_b: Vec<F2X32> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let bits = (x.pack_u64() ^ y.pack_u64()) as u32;
                let coeffs: Vec<Boolean> =
                    (0..32).map(|i| ((bits >> i) & 1 == 1).into()).collect();
                F2X32::new(coeffs)
            })
            .collect();

        let lhs = f2_lin_comb(&a_xor_b, &c, row_len);
        let lc_a = f2_lin_comb(&a, &c, row_len);
        let lc_b = f2_lin_comb(&b, &c, row_len);

        let rhs: Vec<F2X160> = lc_a
            .iter()
            .zip(lc_b.iter())
            .map(|(x, y)| {
                let mut z = *x;
                z += y;
                z
            })
            .collect();

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn linearity_in_coefficients() {
        let row_len = 3;
        let num_rows = 4;
        let n = row_len * num_rows;
        let cells: Vec<F2X32> = (0..n as u32).map(|i| bp32(i.wrapping_mul(0x12345))).collect();
        let c1: Vec<F2X128> = (0..num_rows as u64).map(|j| bp128(j ^ 0x111, j ^ 0x222)).collect();
        let c2: Vec<F2X128> = (0..num_rows as u64).map(|j| bp128(j ^ 0x333, j ^ 0x444)).collect();
        let c_xor: Vec<F2X128> = c1
            .iter()
            .zip(c2.iter())
            .map(|(x, y)| {
                let mut z = *x;
                z += y;
                z
            })
            .collect();

        let lhs = f2_lin_comb(&cells, &c_xor, row_len);
        let l1 = f2_lin_comb(&cells, &c1, row_len);
        let l2 = f2_lin_comb(&cells, &c2, row_len);
        let rhs: Vec<F2X160> = l1
            .iter()
            .zip(l2.iter())
            .map(|(x, y)| {
                let mut z = *x;
                z += y;
                z
            })
            .collect();

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn degree_bound_holds() {
        // `F_2[X]<32>` · `F_2[X]<128>` has degree ≤ 31 + 127 = 158,
        // i.e. bits 0..159 only. Bits 159..192 of the output must be 0.
        let row_len = 2;
        let cells = vec![bp32(u32::MAX); row_len]; // all 32 bits set
        let c = vec![bp128(u64::MAX, u64::MAX)]; // all 128 bits set
        let out = f2_lin_comb(&cells, &c, row_len);
        for entry in &out {
            let high_word = entry.words()[2];
            assert_eq!(
                high_word >> 31,
                0,
                "out-of-range degree bits set: word[2] = {high_word:#x}",
            );
        }
    }
}
