//! ECDSA Verification UAIR (Universal Algebraic Intermediate Representation).
//!
//! This crate defines the ECDSA signature verification arithmetization as
//! described in the Zinc+ paper: **258 rows × 14 columns**.
//!
//! # Architecture
//!
//! ECDSA verification checks: given message hash `e`, signature `(r, s)`,
//! and public key `Q` on secp256k1, verify that `u₁·G + u₂·Q` has
//! x-coordinate equal to `r` (mod `n`), where `u₁ = e·s⁻¹` and `u₂ = r·s⁻¹`.
//!
//! The arithmetization uses Shamir's trick to compute `u₁·G + u₂·Q` in
//! a single 256-step double-and-add loop, processing one bit of each
//! scalar per row.
//!
//! # Structure
//!
//! - Row 1: Precomputation (public-key curve check, G+Q precompute)
//! - Rows 2–257: Shamir's trick double-and-add loop (256 scalar bits)
//! - Row 258: Final verification (inverse, affine conversion, sig check)
//!
//! # Column layout (14 columns)
//!
//! | Index | Name  | Ring       | Description                                |
//! |-------|-------|------------|--------------------------------------------|
//! | 0     | b₁    | Q          | Bit of scalar u₁ (bit-index 257-t)         |
//! | 1     | b₂    | Q          | Bit of scalar u₂ (bit-index 257-t)         |
//! | 2     | k     | Q          | Quotient bit for sig check R_x ≡ r (mod n) |
//! | 3     | X     | F_p        | Accumulator x-coord (Jacobian)             |
//! | 4     | Y     | F_p        | Accumulator y-coord (Jacobian)             |
//! | 5     | Z     | F_p        | Accumulator z-coord (Jacobian)             |
//! | 6     | X_mid | F_p        | Doubled point 2P x-coord (Jacobian)        |
//! | 7     | Y_mid | F_p        | Doubled point 2P y-coord (Jacobian)        |
//! | 8     | Z_mid | F_p        | Doubled point 2P z-coord (Jacobian)        |
//! | 9     | S     | F_p        | Doubling scratch: S = Y²                   |
//! | 10    | H     | F_p        | Addition scratch: chord x-diff             |
//! | 11    | R_a   | F_p        | Addition scratch: chord y-diff             |
//! | 12    | u₁    | F_n        | Scalar accumulator for u₁ = e·s⁻¹          |
//! | 13    | u₂    | F_n        | Scalar accumulator for u₂ = r·s⁻¹          |
//!
//! # Constraints (11 non-boundary)
//!
//! The ECDSA verification constraints operate in F_p and F_n, not in
//! F₂\[X\]. Since the current Zinc+ UAIR framework only supports a
//! single ring per UAIR, and the PCS/PIOP operate on BinaryPoly traces,
//! this crate provides:
//!
//! - Correct trace dimensions for benchmarking (258 rows × 14 columns)
//! - Random witness generation (valid for PCS timing, not constraint-sound)
//! - The UAIR struct with constraint specifications
//!
//! Full constraint implementation requires the multi-ring UAIR extension
//! described in the paper (Section 5.2). See the [`constraints`] module
//! for the mathematical specification of all 11 constraints.
//!
//! # secp256k1 Parameters
//!
//! - Base field F_p: p = 2²⁵⁶ - 2³² - 2⁹ - 2⁸ - 2⁷ - 2⁶ - 2⁴ - 1
//!   = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//! - Group order F_n: n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
//! - Generator G: (Gx, Gy) where
//!   Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
//!   Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
//! - Curve equation: y² = x³ + 7 (a=0, b=7)

#![allow(clippy::arithmetic_side_effects)]

pub mod constraints;
pub mod witness;

use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, IdealCheck, ImpossibleIdeal},
};
use zinc_utils::from_ref::FromRef;

/// Total number of trace columns (from the paper: 14).
pub const NUM_COLS: usize = 14;

/// Number of trace rows: 1 (precomp) + 256 (scalar mul) + 1 (final check) = 258.
pub const NUM_ROWS: usize = 258;

/// Number of non-boundary constraints (from the paper: 11).
pub const NUM_CONSTRAINTS: usize = 0; // Placeholder — constraints require F_p ring

// ─── Column indices ──────────────────────────────────────────────────────────

pub const COL_B1: usize = 0;
pub const COL_B2: usize = 1;
pub const COL_K: usize = 2;
pub const COL_X: usize = 3;
pub const COL_Y: usize = 4;
pub const COL_Z: usize = 5;
pub const COL_X_MID: usize = 6;
pub const COL_Y_MID: usize = 7;
pub const COL_Z_MID: usize = 8;
pub const COL_S: usize = 9;
pub const COL_H: usize = 10;
pub const COL_RA: usize = 11;
pub const COL_U1: usize = 12;
pub const COL_U2: usize = 13;

// ─── Toy curve constants (for testing with DensePolynomial<i64, 1>) ─────────
// Curve: y² = x³ + 7 over F_101 (same equation as secp256k1, small prime)
// G  = (15, 7)   — generator
// Q  = (15, 7)   — public key (= G for simplicity)
// G+Q= (35, 82)  — precomputed table point (= 2G affine)

/// Generator x-coordinate.
pub const GX: i64 = 15;
/// Generator y-coordinate.
pub const GY: i64 = 7;
/// Public key x-coordinate (= G for the toy case).
pub const QX: i64 = 15;
/// Public key y-coordinate.
pub const QY: i64 = 7;
/// Precomputed (G+Q) x-coordinate (= 2G affine).
pub const PGQX: i64 = 35;
/// Precomputed (G+Q) y-coordinate.
pub const PGQY: i64 = 82;

// ─── Ideal types ────────────────────────────────────────────────────────────

/// Placeholder ideal for ECDSA BinaryPoly (no constraints enforced).
#[derive(Clone, Copy, Debug)]
pub struct EcdsaIdeal;

impl Ideal for EcdsaIdeal {}

impl FromRef<EcdsaIdeal> for EcdsaIdeal {
    #[inline(always)]
    fn from_ref(ideal: &EcdsaIdeal) -> Self {
        *ideal
    }
}

impl IdealCheck<BinaryPoly<32>> for EcdsaIdeal {
    fn contains(&self, value: &BinaryPoly<32>) -> bool {
        num_traits::Zero::is_zero(value)
    }
}

// ─── Ideal over the PIOP field ──────────────────────────────────────────────

/// Ideal over the PIOP field for ECDSA IC verification.
///
/// Since all 11 ECDSA constraints use `assert_zero`, the only ideal
/// membership check is "value is zero". This is the field-level analog
/// of `ImpossibleIdeal` (the UAIR-level ideal).
#[derive(Clone, Debug)]
pub struct EcdsaIdealOverF;

impl FromRef<EcdsaIdealOverF> for EcdsaIdealOverF {
    fn from_ref(_: &EcdsaIdealOverF) -> Self {
        EcdsaIdealOverF
    }
}

impl Ideal for EcdsaIdealOverF {}

impl<F: crypto_primitives::PrimeField> IdealCheck<DynamicPolynomialF<F>> for EcdsaIdealOverF {
    fn contains(&self, value: &DynamicPolynomialF<F>) -> bool {
        num_traits::Zero::is_zero(value)
    }
}

// ─── Trace conversion ───────────────────────────────────────────────────────

/// Convert a `BinaryPoly<32>` ECDSA trace to `DensePolynomial<i64, 1>`.
///
/// Each `BinaryPoly<32>` is evaluated at X=2 (giving its integer value)
/// and stored as a degree-0 `DensePolynomial<i64, 1>`.
///
/// **Note:** This only works for non-negative values (0 to 2³²−1).
/// The ECDSA witness with negative values (e.g., H=−1) cannot be roundtripped
/// through `BinaryPoly<32>`. Use the all-zero trace or the direct
/// `DensePolynomial<i64, 1>` witness generator for full constraint testing.
pub fn convert_trace_bp_to_i64(
    trace: &[DenseMultilinearExtension<BinaryPoly<32>>],
) -> Vec<DenseMultilinearExtension<DensePolynomial<i64, 1>>> {
    let zero = DensePolynomial::<i64, 1>::new([0]);
    trace
        .iter()
        .map(|col| {
            let evaluations: Vec<DensePolynomial<i64, 1>> = col
                .evaluations
                .iter()
                .map(|bp| DensePolynomial::<i64, 1>::new([bp.to_u64() as i64]))
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(col.num_vars, evaluations, zero)
        })
        .collect()
}

// ─── ECDSA UAIR ─────────────────────────────────────────────────────────────

/// The ECDSA verification UAIR.
pub struct EcdsaUair;

// ─── BinaryPoly<32> implementation (placeholder, no constraints) ────────────

impl Uair<BinaryPoly<32>> for EcdsaUair {
    type Ideal = EcdsaIdeal;

    fn num_cols() -> usize {
        NUM_COLS
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        _b: &mut B,
        _up: &[B::Expr],
        _down: &[B::Expr],
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &BinaryPoly<32>) -> Option<B::Expr>,
        IFromR: Fn(&EcdsaIdeal) -> B::Ideal,
    {
        // BinaryPoly<32> cannot express F_p arithmetic.
        // Constraints are defined over DensePolynomial<i64, 1> below.
    }
}

// ─── DensePolynomial<i64, 1> implementation (11 constraints) ────────────────
//
// Ring: degree-0 integer polynomials (effectively i64 scalars).
// All constraints are assert_zero (exact integer equality).
// The witness uses integer arithmetic without modular reduction,
// so only works for small values where products don't overflow i64.
//
// For real secp256k1, a proper 256-bit field type would be needed.

/// Number of constraints for the i64 ECDSA UAIR.
pub const NUM_CONSTRAINTS_I64: usize = 11;

impl Uair<DensePolynomial<i64, 1>> for EcdsaUair {
    type Ideal = ImpossibleIdeal;

    fn num_cols() -> usize {
        NUM_COLS
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        down: &[B::Expr],
        from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&DensePolynomial<i64, 1>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<i64, 1>) -> Option<B::Expr>,
        IFromR: Fn(&ImpossibleIdeal) -> B::Ideal,
    {
        // ── Helpers ─────────────────────────────────────────────────
        let dp = |v: i64| DensePolynomial::<i64, 1>::new([v]);
        let cst = |v: i64| from_ref(&dp(v));
        let smul = |e: &B::Expr, v: i64| mbs(e, &dp(v)).unwrap();

        let one = cst(1);

        // ── C1: Scalar accumulation u₁ ─────────────────────────────
        // down[u1] - 2·up[u1] - up[b1] = 0
        b.assert_zero(
            down[COL_U1].clone() - &smul(&up[COL_U1], 2) - &up[COL_B1],
        );

        // ── C2: Scalar accumulation u₂ ─────────────────────────────
        // down[u2] - 2·up[u2] - up[b2] = 0
        b.assert_zero(
            down[COL_U2].clone() - &smul(&up[COL_U2], 2) - &up[COL_B2],
        );

        // ── C3: Doubling scratch S = Y² ────────────────────────────
        b.assert_zero(
            up[COL_S].clone() - &(up[COL_Y].clone() * &up[COL_Y]),
        );

        // ── C4: Doubled Z-coordinate: Z_mid = 2·Y·Z ───────────────
        b.assert_zero(
            up[COL_Z_MID].clone()
                - &smul(&(up[COL_Y].clone() * &up[COL_Z]), 2),
        );

        // ── C5: Doubled X-coordinate ───────────────────────────────
        // X_mid = M² - 2U where M = 3X², U = 4XS
        // Expanded: X_mid = 9X⁴ - 8XS
        let x_sq = up[COL_X].clone() * &up[COL_X];
        let x_four = x_sq.clone() * &x_sq;
        let x_s = up[COL_X].clone() * &up[COL_S];
        b.assert_zero(
            up[COL_X_MID].clone() - &smul(&x_four, 9) + &smul(&x_s, 8),
        );

        // ── C6: Doubled Y-coordinate ───────────────────────────────
        // Y_mid = M·(U - X_mid) - 8S²
        // where M = 3X², U = 4XS
        // Expanded: Y_mid = 12X³S - 3X²·X_mid - 8S²
        let x_cubed_s = x_sq.clone() * &up[COL_X] * &up[COL_S];
        let x_sq_xmid = x_sq * &up[COL_X_MID];
        let s_sq = up[COL_S].clone() * &up[COL_S];
        b.assert_zero(
            up[COL_Y_MID].clone()
                - &smul(&x_cubed_s, 12)
                + &smul(&x_sq_xmid, 3)
                + &smul(&s_sq, 8),
        );

        // ── Shamir selector ────────────────────────────────────────
        // s = b1 + b2 - b1·b2 (= 1 iff any bit is set)
        let b1b2 = up[COL_B1].clone() * &up[COL_B2];
        let s = up[COL_B1].clone() + &up[COL_B2] - &b1b2;

        // Table point selection:
        // T_x = b1·(1-b2)·Gx + (1-b1)·b2·Qx + b1·b2·PGQx
        let one_minus_b2 = one.clone() - &up[COL_B2];
        let one_minus_b1 = one.clone() - &up[COL_B1];
        let b1_not_b2 = up[COL_B1].clone() * &one_minus_b2;
        let not_b1_b2 = one_minus_b1 * &up[COL_B2];
        let t_x = smul(&b1_not_b2, GX)
            + &smul(&not_b1_b2, QX)
            + &smul(&b1b2, PGQX);
        let t_y = smul(&b1_not_b2, GY)
            + &smul(&not_b1_b2, QY)
            + &smul(&b1b2, PGQY);

        // ── C7: Addition scratch H = T_x·Z_mid² - X_mid ───────────
        let zmid_sq = up[COL_Z_MID].clone() * &up[COL_Z_MID];
        b.assert_zero(
            up[COL_H].clone() - &(t_x * &zmid_sq) + &up[COL_X_MID],
        );

        // ── C8: Addition scratch R_a = T_y·Z_mid³ - Y_mid ─────────
        let zmid_cubed = zmid_sq.clone() * &up[COL_Z_MID];
        b.assert_zero(
            up[COL_RA].clone() - &(t_y * &zmid_cubed) + &up[COL_Y_MID],
        );

        // ── C9: Result Z-coordinate ────────────────────────────────
        // Z[t+1] = (1-s)·Z_mid + s·(Z_mid·H)
        let one_minus_s = one - &s;
        let zmid_h = up[COL_Z_MID].clone() * &up[COL_H];
        b.assert_zero(
            down[COL_Z].clone()
                - &(one_minus_s.clone() * &up[COL_Z_MID])
                - &(s.clone() * &zmid_h),
        );

        // ── C10: Result X-coordinate ───────────────────────────────
        // When s=0: X[t+1] = X_mid
        // When s=1: X[t+1] = Ra² - H³ - 2·X_mid·H²
        let ra_sq = up[COL_RA].clone() * &up[COL_RA];
        let h_sq = up[COL_H].clone() * &up[COL_H];
        let h_cubed = h_sq.clone() * &up[COL_H];
        let xmid_h_sq = up[COL_X_MID].clone() * &h_sq;
        let add_x = ra_sq - &h_cubed - &smul(&xmid_h_sq, 2);
        b.assert_zero(
            down[COL_X].clone()
                - &(one_minus_s.clone() * &up[COL_X_MID])
                - &(s.clone() * &add_x),
        );

        // ── C11: Result Y-coordinate ───────────────────────────────
        // When s=0: Y[t+1] = Y_mid
        // When s=1: Y[t+1] = Ra·(X_mid·H² - X[t+1]) - Y_mid·H³
        let xmid_h_sq_2 = up[COL_X_MID].clone() * &h_sq;
        let ra_term = up[COL_RA].clone() * &(xmid_h_sq_2 - &down[COL_X]);
        let ymid_h_cubed = up[COL_Y_MID].clone() * &h_cubed;
        let add_y = ra_term - &ymid_h_cubed;
        b.assert_zero(
            down[COL_Y].clone()
                - &(one_minus_s * &up[COL_Y_MID])
                - &(s * &add_y),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zinc_uair::constraint_counter::count_constraints;
    use zinc_uair::degree_counter::count_max_degree;

    #[test]
    fn correct_dimensions() {
        assert_eq!(NUM_ROWS, 258);
        assert_eq!(NUM_COLS, 14);
    }

    #[test]
    fn i64_constraint_count() {
        let n = count_constraints::<DensePolynomial<i64, 1>, EcdsaUair>();
        assert_eq!(n, NUM_CONSTRAINTS_I64, "Expected 11 ECDSA i64 constraints");
    }

    #[test]
    fn i64_max_degree() {
        let d = count_max_degree::<DensePolynomial<i64, 1>, EcdsaUair>();
        // C11 has degree 6: s(deg2) * Ra(1) * X_mid(1) * H²(2) = 6
        assert!(d <= 6, "Max degree should be at most 6, got {d}");
        assert!(d >= 4, "Max degree should be at least 4 (doubling formulas), got {d}");
    }
}
