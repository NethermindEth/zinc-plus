//! Witness generation for the ECDSA verification UAIR.
//!
//! Provides three witness generators:
//! 1. `BinaryPoly<32>`: Random trace for PCS benchmarking.
//! 2. `DensePolynomial<i64, 1>`: Valid Jacobian doubling trace for
//!    IC+CPR testing (toy curve, integer arithmetic, no modular reduction).
//! 3. `Int<4>`: Valid Jacobian doubling trace using 256-bit integers.
//!    This is the target type for the unified ECDSA pipeline where the
//!    same `Int<4>` is used for PCS commitments, PIOP, and constraints.

use super::{EcdsaUair, NUM_COLS};
use super::{COL_B1, COL_B2, COL_K, COL_X, COL_Y, COL_Z};
use super::{COL_X_MID, COL_Y_MID, COL_Z_MID, COL_S, COL_H, COL_RA, COL_U1, COL_U2};
use crypto_primitives::crypto_bigint_int::Int;
use rand::RngCore;
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::binary::BinaryPoly,
    univariate::dense::DensePolynomial,
};
use zinc_utils::from_ref::FromRef;

/// Witness generation trait (matches SHA-256 crate's pattern).
pub trait GenerateWitness<R: crypto_primitives::Semiring + 'static>:
    zinc_uair::Uair<R>
{
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<R>>;
}

impl GenerateWitness<BinaryPoly<32>> for EcdsaUair {
    /// Generate a random ECDSA trace (for PCS benchmarking).
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
        let effective_vars = num_vars.max(9); // 2^9 = 512 ≥ 258
        (0..NUM_COLS)
            .map(|_| DenseMultilinearExtension::rand(effective_vars, rng))
            .collect()
    }
}

impl GenerateWitness<DensePolynomial<i64, 1>> for EcdsaUair {
    /// Generate a valid ECDSA trace (constant-row fixed point, integer arithmetic).
    ///
    /// Uses the Jacobian doubling fixed point **(X, Y, Z) = (1, 1, 0)** with
    /// `b₁ = b₂ = 0` (pure doubling, no addition). Every row is identical:
    ///
    /// - S = Y² = 1
    /// - Z_mid = 2·Y·Z = 0
    /// - X_mid = 9X⁴ − 8XS = 9 − 8 = 1
    /// - Y_mid = 12X³S − 3X²·X_mid − 8S² = 12 − 3 − 8 = 1
    /// - H = T_x·Z_mid² − X_mid = 0 − 1 = −1 (T_x=0 when b₁=b₂=0)
    /// - R_a = T_y·Z_mid³ − Y_mid = 0 − 1 = −1
    ///
    /// Since every row is the same, the wrap-around (last row → row 0) is
    /// automatically consistent and all 11 constraints evaluate to zero at
    /// every hypercube point.
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<i64, 1>>> {
        let dp = |v: i64| DensePolynomial::<i64, 1>::new([v]);
        let num_rows: usize = 1 << num_vars;

        // Pre-allocate columns (14 columns × num_rows), default = 0.
        let mut cols: Vec<Vec<DensePolynomial<i64, 1>>> =
            (0..NUM_COLS).map(|_| vec![dp(0); num_rows]).collect();

        // Fill every row with the fixed-point values.
        for t in 0..num_rows {
            // Scalar bits and quotient (all zero → pure doubling, no addition).
            cols[COL_B1][t] = dp(0);
            cols[COL_B2][t] = dp(0);
            cols[COL_K][t] = dp(0);
            cols[COL_U1][t] = dp(0);
            cols[COL_U2][t] = dp(0);

            // Jacobian accumulator state.
            cols[COL_X][t] = dp(1);
            cols[COL_Y][t] = dp(1);
            cols[COL_Z][t] = dp(0);

            // Doubling intermediates.
            cols[COL_S][t] = dp(1);       // Y² = 1
            cols[COL_Z_MID][t] = dp(0);   // 2·Y·Z = 0
            cols[COL_X_MID][t] = dp(1);   // 9X⁴ - 8XS = 1
            cols[COL_Y_MID][t] = dp(1);   // 12X³S - 3X²X_mid - 8S² = 1

            // Addition intermediates: H = -X_mid, R_a = -Y_mid.
            cols[COL_H][t] = dp(-1);
            cols[COL_RA][t] = dp(-1);
        }

        // Convert to MLEs.
        cols.into_iter()
            .map(|col| {
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    col,
                    dp(0),
                )
            })
            .collect()
    }
}

impl GenerateWitness<Int<4>> for EcdsaUair {
    /// Generate a valid ECDSA trace using `Int<4>` (256-bit integers).
    ///
    /// Same fixed-point witness as the `DensePolynomial<i64, 1>` generator:
    /// **(X, Y, Z) = (1, 1, 0)** with `b₁ = b₂ = 0`. All 11 constraints
    /// evaluate to zero at every hypercube point.
    ///
    /// This is the target witness type for the unified ECDSA pipeline where
    /// `Int<4>` is used for PCS commitments, PIOP constraints, and witness.
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<Int<4>>> {
        let iv = |v: i64| Int::<4>::from_ref(&v);
        let num_rows: usize = 1 << num_vars;

        // Pre-allocate columns (14 columns × num_rows), default = 0.
        let mut cols: Vec<Vec<Int<4>>> =
            (0..NUM_COLS).map(|_| vec![iv(0); num_rows]).collect();

        // Fill every row with the fixed-point values.
        for t in 0..num_rows {
            // Scalar bits and quotient (all zero → pure doubling, no addition).
            cols[COL_B1][t] = iv(0);
            cols[COL_B2][t] = iv(0);
            cols[COL_K][t] = iv(0);
            cols[COL_U1][t] = iv(0);
            cols[COL_U2][t] = iv(0);

            // Jacobian accumulator state.
            cols[COL_X][t] = iv(1);
            cols[COL_Y][t] = iv(1);
            cols[COL_Z][t] = iv(0);

            // Doubling intermediates.
            cols[COL_S][t] = iv(1);       // Y² = 1
            cols[COL_Z_MID][t] = iv(0);   // 2·Y·Z = 0
            cols[COL_X_MID][t] = iv(1);   // 9X⁴ - 8XS = 1
            cols[COL_Y_MID][t] = iv(1);   // 12X³S - 3X²X_mid - 8S² = 1

            // Addition intermediates: H = -X_mid, R_a = -Y_mid.
            cols[COL_H][t] = iv(-1);
            cols[COL_RA][t] = iv(-1);
        }

        // Convert to MLEs.
        cols.into_iter()
            .map(|col| {
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    col,
                    iv(0),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_has_correct_dimensions() {
        let mut rng = rand::rng();
        let trace =
            <EcdsaUair as GenerateWitness<BinaryPoly<32>>>::generate_witness(9, &mut rng);
        assert_eq!(trace.len(), NUM_COLS); // 14 columns
        for col in &trace {
            assert_eq!(col.evaluations.len(), 512); // 2^9 = 512 ≥ 258
        }
    }

    #[test]
    fn i64_witness_fixed_point() {
        let mut rng = rand::rng();
        let num_vars = 4; // 16 rows
        let trace = <EcdsaUair as GenerateWitness<DensePolynomial<i64, 1>>>::generate_witness(
            num_vars, &mut rng,
        );
        assert_eq!(trace.len(), NUM_COLS);
        assert_eq!(trace[0].evaluations.len(), 16);

        // Every row should have the same fixed-point values.
        for t in 0..16 {
            let val = |col: usize| trace[col].evaluations[t].coeffs[0];
            assert_eq!(val(COL_X), 1, "X should be 1 at row {t}");
            assert_eq!(val(COL_Y), 1, "Y should be 1 at row {t}");
            assert_eq!(val(COL_Z), 0, "Z should be 0 at row {t}");
            assert_eq!(val(COL_S), 1, "S should be Y²=1 at row {t}");
            assert_eq!(val(COL_Z_MID), 0, "Z_mid should be 0 at row {t}");
            assert_eq!(val(COL_X_MID), 1, "X_mid should be 1 at row {t}");
            assert_eq!(val(COL_Y_MID), 1, "Y_mid should be 1 at row {t}");
            assert_eq!(val(COL_H), -1, "H should be -X_mid=-1 at row {t}");
            assert_eq!(val(COL_RA), -1, "R_a should be -Y_mid=-1 at row {t}");
        }
    }

    #[test]
    fn int4_witness_fixed_point() {
        let mut rng = rand::rng();
        let num_vars = 4; // 16 rows
        let trace = <EcdsaUair as GenerateWitness<Int<4>>>::generate_witness(
            num_vars, &mut rng,
        );
        assert_eq!(trace.len(), NUM_COLS);
        assert_eq!(trace[0].evaluations.len(), 16);

        let iv = |v: i64| Int::<4>::from_ref(&v);

        // Every row should have the same fixed-point values.
        for t in 0..16 {
            let val = |col: usize| trace[col].evaluations[t];
            assert_eq!(val(COL_X), iv(1), "X should be 1 at row {t}");
            assert_eq!(val(COL_Y), iv(1), "Y should be 1 at row {t}");
            assert_eq!(val(COL_Z), iv(0), "Z should be 0 at row {t}");
            assert_eq!(val(COL_S), iv(1), "S should be Y²=1 at row {t}");
            assert_eq!(val(COL_Z_MID), iv(0), "Z_mid should be 0 at row {t}");
            assert_eq!(val(COL_X_MID), iv(1), "X_mid should be 1 at row {t}");
            assert_eq!(val(COL_Y_MID), iv(1), "Y_mid should be 1 at row {t}");
            assert_eq!(val(COL_H), iv(-1), "H should be -X_mid=-1 at row {t}");
            assert_eq!(val(COL_RA), iv(-1), "R_a should be -Y_mid=-1 at row {t}");
        }
    }
}
