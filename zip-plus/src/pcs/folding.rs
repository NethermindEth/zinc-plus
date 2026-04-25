//! Column-splitting utility for the 1× folded Zip+ PCS.
//!
//! # Overview
//!
//! Given a column `v` with entries in `BinaryPoly<D>` (binary polynomials of
//! degree `< D`), each entry can be split as
//!
//! ```text
//! v[i] = u[i] + X^{D/2} * w[i]
//! ```
//!
//! where `u[i]` holds the low `D/2` coefficients and `w[i]` the high `D/2`.
//! Both halves live in `BinaryPoly<{D/2}>`.
//!
//! Instead of committing to the length-`n` column `v` with `BinaryPoly<D>`
//! entries, the folded pipeline commits to `v' = u || w`: a length-`2n`
//! column with `BinaryPoly<D/2>` entries. The first `n` entries of `v'`
//! hold the low halves (`u[0..n]`) and the last `n` entries the high
//! halves (`w[0..n]`). The resulting MLE has `num_vars + 1` variables —
//! the extra variable selects between the low half (0) and high half (1).
//!
//! Because each codeword element is half the size, Zip+ column openings are
//! roughly halved, reducing total proof size. The folding bridge from the
//! PIOP's evaluation claim on `v` to a PCS claim on `v'` at an extended
//! point is implemented by the protocol prover/verifier (Zinc+ PIOP), which
//! samples a Fiat-Shamir challenge `γ` and opens `v'` at `(r₀ ‖ γ)`.

use crypto_primitives::semiring::boolean::Boolean;
use num_traits::Zero;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::binary::BinaryPoly};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Split a column of `BinaryPoly<D>` entries into a concatenated column
/// of `BinaryPoly<HALF_D>` entries.
///
/// Each entry `v[i]` with `D` binary coefficients is split into:
/// - `u[i]` = low `HALF_D` coefficients (indices `0..HALF_D`)
/// - `w[i]` = high `HALF_D` coefficients (indices `HALF_D..D`)
///
/// so that `v[i] = u[i] + X^HALF_D · w[i]`.
///
/// Returns a column of length `2n` where:
/// - `v'[0..n]   = u[0..n]`  (low halves)
/// - `v'[n..2n]  = w[0..n]`  (high halves)
///
/// The returned MLE has `num_vars + 1` variables, with the last variable
/// selecting between the low half (0) and high half (1).
///
/// # Panics
/// Panics if `D != 2 * HALF_D`.
pub fn split_column<const D: usize, const HALF_D: usize>(
    column: &DenseMultilinearExtension<BinaryPoly<D>>,
) -> DenseMultilinearExtension<BinaryPoly<HALF_D>> {
    assert_eq!(
        D,
        2 * HALF_D,
        "split_column: D ({D}) must equal 2 * HALF_D ({HALF_D})"
    );

    let n = column.evaluations.len();
    let mut lo_evals = Vec::with_capacity(n);
    let mut hi_evals = Vec::with_capacity(n);

    for entry in &column.evaluations {
        // `.inner()` takes `&self`, so this works whether `iter()` yields
        // `&Boolean` (BinaryRefPoly) or owned `Boolean` (BinaryU64Poly).
        let bits: Vec<bool> = entry.iter().map(|b| b.inner()).collect();
        let lo_arr: [Boolean; HALF_D] = std::array::from_fn(|i| Boolean::from(bits[i]));
        let hi_arr: [Boolean; HALF_D] = std::array::from_fn(|i| Boolean::from(bits[HALF_D + i]));
        lo_evals.push(BinaryPoly::<HALF_D>::new(lo_arr));
        hi_evals.push(BinaryPoly::<HALF_D>::new(hi_arr));
    }

    // Concatenate: v' = u || w (low halves first, high halves second).
    lo_evals.extend(hi_evals);

    DenseMultilinearExtension::from_evaluations_vec(
        column
            .num_vars
            .checked_add(1)
            .expect("split_column: num_vars overflow"),
        lo_evals,
        BinaryPoly::zero(),
    )
}

/// Split each column of `BinaryPoly<D>` entries into a concatenated column
/// of `BinaryPoly<HALF_D>` entries (see [`split_column`]).
///
/// Parallel across columns when the `parallel` feature is enabled.
pub fn split_columns<const D: usize, const HALF_D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
) -> Vec<DenseMultilinearExtension<BinaryPoly<HALF_D>>> {
    #[cfg(feature = "parallel")]
    {
        return columns
            .par_iter()
            .map(|col| split_column::<D, HALF_D>(col))
            .collect();
    }
    #[cfg(not(feature = "parallel"))]
    {
        columns
            .iter()
            .map(|col| split_column::<D, HALF_D>(col))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zinc_poly::EvaluatablePolynomial;

    /// Helper: create a `BinaryPoly<32>` from a u32.
    fn bp32(val: u32) -> BinaryPoly<32> {
        BinaryPoly::from(val)
    }

    #[test]
    fn split_column_basic() {
        // Column with 4 entries of BinaryPoly<32>.
        let polys: Vec<BinaryPoly<32>> = vec![
            bp32(0x0001_0002),
            bp32(0x0003_0004),
            bp32(0xFFFF_0000),
            bp32(0x0000_FFFF),
        ];
        let col = DenseMultilinearExtension::from_evaluations_vec(2, polys, BinaryPoly::zero());

        let split = split_column::<32, 16>(&col);

        assert_eq!(split.num_vars, 3);
        assert_eq!(split.evaluations.len(), 8);

        // First 4 entries are low halves (bits 0..15); last 4 are high halves
        // (bits 16..31). For 0x0001_0002: lo = 0x0002 (...0010), hi = 0x0001
        // (...0001).
        let lo_0: Vec<bool> = split.evaluations[0].iter().map(|b| b.inner()).collect();
        let hi_0: Vec<bool> = split.evaluations[4].iter().map(|b| b.inner()).collect();

        assert!(lo_0[1]);
        assert!(!lo_0[0]);
        assert!(hi_0[0]);
        assert!(!hi_0[1]);
    }

    #[test]
    fn split_preserves_reconstruction() {
        // v[i](X=2) = u[i](2) + 2^16 * w[i](2)
        let val: u32 = 0xABCD_1234;
        let col = DenseMultilinearExtension::from_evaluations_vec(
            0,
            vec![bp32(val)],
            BinaryPoly::zero(),
        );

        let split = split_column::<32, 16>(&col);
        assert_eq!(split.evaluations.len(), 2);

        let lo_val: u32 = val & 0xFFFF;
        let hi_val: u32 = val >> 16;

        let lo_at_2: i64 = split.evaluations[0].evaluate_at_point(&2i64).unwrap();
        let hi_at_2: i64 = split.evaluations[1].evaluate_at_point(&2i64).unwrap();

        assert_eq!(lo_at_2 as u32, lo_val);
        assert_eq!(hi_at_2 as u32, hi_val);
        assert_eq!(
            lo_at_2 + (1i64 << 16) * hi_at_2,
            bp32(val).evaluate_at_point(&2i64).unwrap()
        );
    }

    #[test]
    fn split_columns_batch() {
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![bp32(0x0001_0002), bp32(0x0003_0004)],
            BinaryPoly::zero(),
        );
        let col2 = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![bp32(0x0005_0006), bp32(0x0007_0008)],
            BinaryPoly::zero(),
        );

        let split = split_columns::<32, 16>(&[col1, col2]);

        assert_eq!(split.len(), 2);
        for s in &split {
            assert_eq!(s.num_vars, 2);
            assert_eq!(s.evaluations.len(), 4);
        }
    }
}
