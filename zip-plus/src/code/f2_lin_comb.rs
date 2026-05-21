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
    binary::BinaryPoly,
    binary_f2_wide::{BinaryF2Poly, f2_poly_mul},
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
/// Complexity: `O(num_rows * row_len * mul_cost(32, 128))`, where
/// each multiplication is a 32-bit-by-128-bit carryless product
/// (~32 word ops in the inner loop of `f2_poly_mul`).
#[allow(clippy::arithmetic_side_effects)]
pub fn f2_lin_comb(cells: &[F2X32], coeffs: &[F2X128], row_len: usize) -> Vec<F2X160> {
    assert!(
        row_len > 0,
        "f2_lin_comb: row_len must be > 0"
    );
    assert_eq!(
        cells.len(),
        coeffs.len() * row_len,
        "f2_lin_comb: cells.len() ({}) must equal coeffs.len() * row_len ({} * {} = {})",
        cells.len(),
        coeffs.len(),
        row_len,
        coeffs.len() * row_len,
    );

    let mut out = vec![F2X160::zero(); row_len];
    for (j, &coeff) in coeffs.iter().enumerate() {
        let row_start = j * row_len;
        let row = &cells[row_start..row_start + row_len];
        for (col, &cell) in row.iter().enumerate() {
            // Lift the F_2[X]<32> cell to the wide layout. The
            // `BinaryPoly` alias is feature-gated, so do this via the
            // ConstTranscribable round-trip (write to bytes, read into
            // a u64). Since the cell stores ≤ 32 bits, the low word
            // captures the full coefficient pattern.
            let cell_wide = f2x32_to_wide_low_word(&cell);
            let prod: F2X160 = f2_poly_mul(&cell_wide, &coeff);
            out[col] += &prod;
        }
    }
    out
}

/// Lift an `F_2[X]<32>` value into a `BinaryF2Poly<1>` (single u64
/// word). Works regardless of whether `BinaryPoly` resolves to the
/// `BinaryRefPoly` or `BinaryU64Poly` variant.
fn f2x32_to_wide_low_word(p: &F2X32) -> BinaryF2Poly<1> {
    let mut w: u64 = 0;
    for (i, c) in p.iter().enumerate() {
        if c.inner() {
            // i ≤ 31, never overflows.
            #[allow(clippy::arithmetic_side_effects)]
            {
                w |= 1u64 << i;
            }
        }
    }
    BinaryF2Poly::<1>::from_words([w])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::boolean::Boolean;

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

        // Reproduce by hand: out[col] = lift32(cells[col]) · c.
        for (col, &cell) in cells.iter().enumerate() {
            let cell_wide = f2x32_to_wide_low_word(&cell);
            let expected: F2X160 = f2_poly_mul(&cell_wide, &c);
            assert_eq!(out[col], expected);
        }
    }

    #[test]
    fn linearity_in_cells() {
        // f2_lin_comb is F_2-linear in `cells`. Build A and B and check
        // f2_lin_comb(A XOR B, c) == f2_lin_comb(A, c) XOR f2_lin_comb(B, c).
        let row_len = 3;
        let num_rows = 4;
        let n = row_len * num_rows;
        let a: Vec<F2X32> = (0..n as u32).map(|i| bp32(i.wrapping_mul(0xA5A5_A5A5))).collect();
        let b: Vec<F2X32> = (0..n as u32).map(|i| bp32(i.wrapping_mul(0xDEAD_BEEF))).collect();
        let c: Vec<F2X128> = (0..num_rows as u64).map(|j| bp128(j ^ 0xC0FFEE, j ^ 0xBEEFCAFE)).collect();

        let a_xor_b: Vec<F2X32> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                // XOR via bit-pattern reconstruction.
                let mut bits = 0u32;
                for (i, ci) in x.iter().enumerate() {
                    if ci.inner() {
                        bits |= 1u32 << i;
                    }
                }
                for (i, ci) in y.iter().enumerate() {
                    if ci.inner() {
                        bits ^= 1u32 << i;
                    }
                }
                let coeffs: Vec<Boolean> = (0..32).map(|i| ((bits >> i) & 1 == 1).into()).collect();
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
        // f2_lin_comb is also F_2-linear in `coeffs`:
        // f2_lin_comb(cells, c1 XOR c2) == f2_lin_comb(cells, c1) XOR f2_lin_comb(cells, c2).
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
            // Word 2 of the output should have bit 31..63 clear (since
            // word index 2 starts at bit 128; max set bit is 158, so
            // only bits 0..=30 of word 2 may be set).
            let high_word = entry.words()[2];
            assert_eq!(
                high_word >> 31,
                0,
                "out-of-range degree bits set: word[2] = {high_word:#x}",
            );
        }
    }
}
