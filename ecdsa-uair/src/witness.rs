//! Witness generation for the ECDSA verification UAIR.
//!
//! Provides three witness generators:
//! 1. `BinaryPoly<32>`: Random trace for PCS benchmarking.
//! 2. `DensePolynomial<i64, 1>`: Valid Jacobian doubling trace for
//!    IC+CPR testing (toy curve over F_101, integer arithmetic, no modular reduction).
//! 3. `Int<4>`: Valid Jacobian doubling trace using 256-bit integers
//!    with secp256k1 curve constants.

use super::{EcdsaUairBp, EcdsaUairDp, EcdsaUairInt, NUM_COLS, NUM_ROWS};
use super::{COL_B1, COL_B2, COL_X, COL_Y, COL_Z};
use super::{COL_X_MID, COL_Y_MID, COL_Z_MID, COL_H};
use super::{COL_SEL_INIT, COL_SEL_FINAL};
use crypto_primitives::crypto_bigint_int::Int;
use rand::RngCore;
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::binary::BinaryPoly,
    univariate::dense::DensePolynomial,
};
use zinc_utils::from_ref::FromRef;

/// Witness generation trait (matches SHA-256 crate's pattern).
pub trait GenerateWitness<R: crypto_primitives::Semiring + 'static> {
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<R>>;
}

impl GenerateWitness<BinaryPoly<32>> for EcdsaUairBp {
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

impl GenerateWitness<DensePolynomial<i64, 1>> for EcdsaUairDp {
    /// Generate a valid ECDSA trace (constant-row fixed point, integer arithmetic).
    ///
    /// Uses the Jacobian doubling fixed point **(X, Y, Z) = (1, 1, 0)** with
    /// `b₁ = b₂ = 0` (pure doubling, no addition). Every row is identical
    /// for the data columns; selector columns `sel_init` and `sel_final`
    /// are 1 at row 0 and row NUM_ROWS−1 respectively.
    ///
    /// - Z_mid = 2·Y·Z = 0
    /// - X_mid = 9X⁴ − 8X·Y² = 9 − 8 = 1
    /// - Y_mid = 12X³·Y² − 3X²·X_mid − 8Y⁴ = 12 − 3 − 8 = 1
    /// - H = T_x·Z_mid² − X_mid = 0 − 1 = −1 (T_x=0 when b₁=b₂=0)
    ///
    /// Since every row is the same (data-wise), the wrap-around (last row → row 0)
    /// is automatically consistent and all 11 constraints evaluate to zero at
    /// every hypercube point.
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<i64, 1>>> {
        let dp = |v: i64| DensePolynomial::<i64, 1>::new([v]);
        let num_rows: usize = 1 << num_vars;

        // Pre-allocate columns (11 columns × num_rows), default = 0.
        let mut cols: Vec<Vec<DensePolynomial<i64, 1>>> =
            (0..NUM_COLS).map(|_| vec![dp(0); num_rows]).collect();

        // Fill every row with the fixed-point values.
        for t in 0..num_rows {
            // Scalar bits (all zero → pure doubling, no addition).
            cols[COL_B1][t] = dp(0);
            cols[COL_B2][t] = dp(0);

            // Jacobian accumulator state.
            cols[COL_X][t] = dp(1);
            cols[COL_Y][t] = dp(1);
            cols[COL_Z][t] = dp(0);

            // Doubling intermediates.
            cols[COL_Z_MID][t] = dp(0);   // 2·Y·Z = 0
            cols[COL_X_MID][t] = dp(1);   // 9X⁴ - 8X·Y² = 1
            cols[COL_Y_MID][t] = dp(1);   // 12X³·Y² - 3X²·X_mid - 8Y⁴ = 1

            // Addition intermediate: H = -X_mid.
            cols[COL_H][t] = dp(-1);

            // Selectors (default 0).
            cols[COL_SEL_INIT][t] = dp(0);
            cols[COL_SEL_FINAL][t] = dp(0);
        }

        // Set boundary selectors.
        cols[COL_SEL_INIT][0] = dp(1);
        if NUM_ROWS.saturating_sub(1) < num_rows {
            cols[COL_SEL_FINAL][NUM_ROWS - 1] = dp(1);
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

impl GenerateWitness<Int<4>> for EcdsaUairInt {
    /// Generate a valid ECDSA trace using `Int<4>` (256-bit integers) with
    /// real secp256k1 F_p arithmetic.
    ///
    /// Uses Shamir's trick for `u₁·G + u₂·Q` where the benchmark case
    /// has `Q = G` and the scalars are hardcoded so that b₁\[0\] = 1.
    ///
    /// **Trace layout:**
    /// - Row 0: init accumulator = G (table point for b₁=1, b₂=0), Z=1.
    ///   C1-C4 compute doubling intermediates.  C5-C7 produce row 1 = 2G + G = 3G.
    /// - Rows 1–256: pure doubling (b₁=b₂=0), exercising full F_p arithmetic.
    /// - Row 257: final accumulator.  B4 checks the affine x-coordinate.
    /// - Rows ≥ 258: padding (all zeros) — C5-C7 are gated by (1−sel_final).
    ///
    /// Selector columns `sel_init` / `sel_final` are 1 at row 0 / 257.
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<Int<4>>> {
        use super::{SECP256K1_P, GX, GY};

        let p = SECP256K1_P;
        let iv = |v: i64| Int::<4>::from_ref(&v);
        let num_rows: usize = 1 << num_vars;

        // Pre-allocate columns (11 columns × num_rows), default = 0.
        let mut cols: Vec<Vec<Int<4>>> =
            (0..NUM_COLS).map(|_| vec![iv(0); num_rows]).collect();

        // ── Row 0: initialization (b₁=1, b₂=0 → table point = G) ──
        cols[COL_B1][0] = iv(1);
        cols[COL_B2][0] = iv(0);
        cols[COL_X][0] = GX;
        cols[COL_Y][0] = GY;
        cols[COL_Z][0] = iv(1);
        cols[COL_SEL_INIT][0] = iv(1);

        // Compute row-0 intermediates and next state
        compute_row_intermediates(&mut cols, 0, p);
        let (nx, ny, nz) = compute_next_state(&cols, 0, p);

        // ── Rows 1–256: pure doubling (b₁=b₂=0) ─────────────────
        for t in 1..NUM_ROWS.min(num_rows) {
            cols[COL_B1][t] = iv(0);
            cols[COL_B2][t] = iv(0);
            if t == 1 {
                cols[COL_X][t] = nx;
                cols[COL_Y][t] = ny;
                cols[COL_Z][t] = nz;
            }
            // Previous row's C5-C7 already set this row's X/Y/Z
            // (except row 1 which we set from row 0's next state above)
            compute_row_intermediates(&mut cols, t, p);
            if t < NUM_ROWS - 1 && t + 1 < num_rows {
                let (nx2, ny2, nz2) = compute_next_state(&cols, t, p);
                cols[COL_X][t + 1] = nx2;
                cols[COL_Y][t + 1] = ny2;
                cols[COL_Z][t + 1] = nz2;
            }
        }

        // ── Boundary selectors ────────────────────────────────────
        if NUM_ROWS.saturating_sub(1) < num_rows {
            cols[COL_SEL_FINAL][NUM_ROWS - 1] = iv(1);
        }

        // ── Padding: rows ≥ NUM_ROWS stay zero (C5-C7 gated by 1−sel_final) ──
        // Already initialized to zero.

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

/// Compute the doubling intermediates (Z_mid, X_mid, Y_mid) and scratch H
/// for row `t`, writing them into the column vectors.  Reads X, Y, Z, b₁, b₂.
fn compute_row_intermediates(cols: &mut [Vec<Int<4>>], t: usize, p: Int<4>) {
    use super::{fp_mul, fp_sub, fp_smul};

    let x = cols[COL_X][t];
    let y = cols[COL_Y][t];
    let z = cols[COL_Z][t];
    let b1 = cols[COL_B1][t];
    let b2 = cols[COL_B2][t];

    // C1: Z_mid = 2·Y·Z
    let z_mid = fp_smul(fp_mul(y, z, p), 2, p);

    // C2: X_mid = 9X⁴ − 8X·Y²
    let x_sq = fp_mul(x, x, p);
    let x_four = fp_mul(x_sq, x_sq, p);
    let y_sq = fp_mul(y, y, p);
    let x_y_sq = fp_mul(x, y_sq, p);
    let x_mid = fp_sub(fp_smul(x_four, 9, p), fp_smul(x_y_sq, 8, p), p);

    // C3: Y_mid = 12X³·Y² − 3X²·X_mid − 8Y⁴
    let x_cubed = fp_mul(x_sq, x, p);
    let x_cubed_y_sq = fp_mul(x_cubed, y_sq, p);
    let x_sq_xmid = fp_mul(x_sq, x_mid, p);
    let y_four = fp_mul(y_sq, y_sq, p);
    let y_mid = fp_sub(
        fp_sub(fp_smul(x_cubed_y_sq, 12, p), fp_smul(x_sq_xmid, 3, p), p),
        fp_smul(y_four, 8, p),
        p,
    );

    // Table point selection (same formula as constraints)
    let (t_x, _t_y) = table_point(b1, b2, p);

    // C4: H = T_x · Z_mid² − X_mid
    let zmid_sq = fp_mul(z_mid, z_mid, p);
    let h = fp_sub(fp_mul(t_x, zmid_sq, p), x_mid, p);

    cols[COL_Z_MID][t] = z_mid;
    cols[COL_X_MID][t] = x_mid;
    cols[COL_Y_MID][t] = y_mid;
    cols[COL_H][t] = h;
}

/// Compute the next state (X[t+1], Y[t+1], Z[t+1]) from row `t`'s data
/// using C5-C7 formulas.
fn compute_next_state(cols: &[Vec<Int<4>>], t: usize, p: Int<4>) -> (Int<4>, Int<4>, Int<4>) {
    use super::{fp_mul, fp_add, fp_sub, fp_smul};

    let iv = |v: i64| Int::<4>::from_ref(&v);
    let b1 = cols[COL_B1][t];
    let b2 = cols[COL_B2][t];
    let x_mid = cols[COL_X_MID][t];
    let y_mid = cols[COL_Y_MID][t];
    let z_mid = cols[COL_Z_MID][t];
    let h = cols[COL_H][t];

    // s = b1 + b2 − b1·b2
    let s_val = {
        let b1b2 = fp_mul(b1, b2, p);
        fp_sub(fp_add(b1, b2, p), b1b2, p)
    };

    // Table point
    let (_t_x, t_y) = table_point(b1, b2, p);

    // Inlined R_a = T_y·Z_mid³ − Y_mid
    let zmid_sq = fp_mul(z_mid, z_mid, p);
    let zmid_cubed = fp_mul(zmid_sq, z_mid, p);
    let r_a = fp_sub(fp_mul(t_y, zmid_cubed, p), y_mid, p);

    let one = iv(1);
    let zero = iv(0);
    let _one_minus_s = fp_sub(one, s_val, p);

    if s_val == zero {
        // Pure doubling: next = (X_mid, Y_mid, Z_mid)
        (x_mid, y_mid, z_mid)
    } else {
        // Mixed addition
        let h_sq = fp_mul(h, h, p);
        let h_cubed = fp_mul(h_sq, h, p);

        // C5: Z[t+1] = Z_mid · H
        let next_z = fp_mul(z_mid, h, p);

        // C6: X[t+1] = R_a² − H³ − 2·X_mid·H²
        let ra_sq = fp_mul(r_a, r_a, p);
        let xmid_h_sq = fp_mul(x_mid, h_sq, p);
        let next_x = fp_sub(
            fp_sub(ra_sq, h_cubed, p),
            fp_smul(xmid_h_sq, 2, p),
            p,
        );

        // C7: Y[t+1] = R_a·(X_mid·H² − X[t+1]) − Y_mid·H³
        let diff = fp_sub(xmid_h_sq, next_x, p);
        let next_y = fp_sub(fp_mul(r_a, diff, p), fp_mul(y_mid, h_cubed, p), p);

        (next_x, next_y, next_z)
    }
}

/// Table point T = (T_x, T_y) selected by bits b₁, b₂.
///   T_x = b₁·(1−b₂)·Gx + (1−b₁)·b₂·Qx + b₁·b₂·PGQx
fn table_point(b1: Int<4>, b2: Int<4>, p: Int<4>) -> (Int<4>, Int<4>) {
    use super::{fp_mul, fp_add, fp_sub, GX, GY, QX, QY, PGQX, PGQY};

    let iv = |v: i64| Int::<4>::from_ref(&v);
    let one = iv(1);
    let one_minus_b2 = fp_sub(one, b2, p);
    let one_minus_b1 = fp_sub(one, b1, p);
    let b1_not_b2 = fp_mul(b1, one_minus_b2, p);
    let not_b1_b2 = fp_mul(one_minus_b1, b2, p);
    let b1b2 = fp_mul(b1, b2, p);

    let t_x = fp_add(
        fp_add(fp_mul(b1_not_b2, GX, p), fp_mul(not_b1_b2, QX, p), p),
        fp_mul(b1b2, PGQX, p),
        p,
    );
    let t_y = fp_add(
        fp_add(fp_mul(b1_not_b2, GY, p), fp_mul(not_b1_b2, QY, p), p),
        fp_mul(b1b2, PGQY, p),
        p,
    );
    (t_x, t_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_has_correct_dimensions() {
        let mut rng = rand::rng();
        let trace =
            <EcdsaUairBp as GenerateWitness<BinaryPoly<32>>>::generate_witness(9, &mut rng);
        assert_eq!(trace.len(), NUM_COLS); // 11 columns
        for col in &trace {
            assert_eq!(col.evaluations.len(), 512); // 2^9 = 512 ≥ 258
        }
    }

    #[test]
    fn i64_witness_fixed_point() {
        let mut rng = rand::rng();
        let num_vars = 9; // 512 rows (≥ 258)
        let trace = <EcdsaUairDp as GenerateWitness<DensePolynomial<i64, 1>>>::generate_witness(
            num_vars, &mut rng,
        );
        assert_eq!(trace.len(), NUM_COLS);
        assert_eq!(trace[0].evaluations.len(), 512);

        // Data columns: every row has the same fixed-point values.
        for t in 0..512 {
            let val = |col: usize| trace[col].evaluations[t].coeffs[0];
            assert_eq!(val(COL_X), 1, "X should be 1 at row {t}");
            assert_eq!(val(COL_Y), 1, "Y should be 1 at row {t}");
            assert_eq!(val(COL_Z), 0, "Z should be 0 at row {t}");
            assert_eq!(val(COL_Z_MID), 0, "Z_mid should be 0 at row {t}");
            assert_eq!(val(COL_X_MID), 1, "X_mid should be 1 at row {t}");
            assert_eq!(val(COL_Y_MID), 1, "Y_mid should be 1 at row {t}");
            assert_eq!(val(COL_H), -1, "H should be -X_mid=-1 at row {t}");
        }

        // Selector columns.
        assert_eq!(trace[COL_SEL_INIT].evaluations[0].coeffs[0], 1, "sel_init should be 1 at row 0");
        assert_eq!(trace[COL_SEL_INIT].evaluations[1].coeffs[0], 0, "sel_init should be 0 at row 1");
        assert_eq!(
            trace[COL_SEL_FINAL].evaluations[NUM_ROWS - 1].coeffs[0], 1,
            "sel_final should be 1 at row {}", NUM_ROWS - 1,
        );
        assert_eq!(trace[COL_SEL_FINAL].evaluations[0].coeffs[0], 0, "sel_final should be 0 at row 0");
    }

    #[test]
    fn int4_witness_real_fp() {
        use crate::{GX, GY, SECP256K1_P, fp_mul, fp_inv, R_SIG};

        let mut rng = rand::rng();
        let num_vars = 9; // 512 rows (≥ 258)
        let trace = <EcdsaUairInt as GenerateWitness<Int<4>>>::generate_witness(
            num_vars, &mut rng,
        );
        assert_eq!(trace.len(), NUM_COLS);
        assert_eq!(trace[0].evaluations.len(), 512);

        let iv = |v: i64| Int::<4>::from_ref(&v);
        let p = SECP256K1_P;

        // Row 0: initialized to G (b₁=1, b₂=0)
        assert_eq!(trace[COL_B1].evaluations[0], iv(1), "b1[0] should be 1");
        assert_eq!(trace[COL_B2].evaluations[0], iv(0), "b2[0] should be 0");
        assert_eq!(trace[COL_X].evaluations[0], GX, "X[0] should be Gx");
        assert_eq!(trace[COL_Y].evaluations[0], GY, "Y[0] should be Gy");
        assert_eq!(trace[COL_Z].evaluations[0], iv(1), "Z[0] should be 1");

        // Selector columns.
        assert_eq!(trace[COL_SEL_INIT].evaluations[0], iv(1));
        assert_eq!(trace[COL_SEL_INIT].evaluations[1], iv(0));
        assert_eq!(trace[COL_SEL_FINAL].evaluations[NUM_ROWS - 1], iv(1));
        assert_eq!(trace[COL_SEL_FINAL].evaluations[0], iv(0));

        // Row 257 (final) should have non-zero Z (real point, not identity).
        let z_final = trace[COL_Z].evaluations[NUM_ROWS - 1];
        assert_ne!(z_final, iv(0), "Final Z should be non-zero");

        // Padding: rows 258+ should be all zeros.
        for t in NUM_ROWS..512 {
            assert_eq!(trace[COL_X].evaluations[t], iv(0), "Padding X[{t}] should be 0");
            assert_eq!(trace[COL_Y].evaluations[t], iv(0), "Padding Y[{t}] should be 0");
            assert_eq!(trace[COL_Z].evaluations[t], iv(0), "Padding Z[{t}] should be 0");
        }

        // Verify B4: affine x-coordinate = R_SIG at the final row.
        let x_final = trace[COL_X].evaluations[NUM_ROWS - 1];
        let z_sq = fp_mul(z_final, z_final, p);
        let z_sq_inv = fp_inv(z_sq, p);
        let affine_x = fp_mul(x_final, z_sq_inv, p);
        assert_eq!(affine_x, *R_SIG, "Affine x at final row should equal R_SIG");
    }
}
