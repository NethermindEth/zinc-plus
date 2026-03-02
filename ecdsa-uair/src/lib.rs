//! ECDSA Verification UAIR (Universal Algebraic Intermediate Representation).
//!
//! This crate defines the ECDSA signature verification arithmetization as
//! described in the Zinc+ paper: **258 rows × 9 columns**.
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
//! The scalars u₁, u₂ are public inputs (given to the verifier in the clear).
//! The quotient bit k is handled in boundary constraints only.
//! The auxiliary values S = Y² and R_a = T_y·Z_mid³ − Y_mid are inlined
//! as sub-expressions in the constraints (not separate columns).
//!
//! # Structure
//!
//! - Row 1: Precomputation (public-key curve check, G+Q precompute)
//! - Rows 2–257: Shamir's trick double-and-add loop (256 scalar bits)
//! - Row 258: Final verification (inverse, affine conversion, sig check)
//!
//! # Column layout (11 columns)
//!
//! | Index | Name       | Description                                |
//! |-------|------------|--------------------------------------------|
//! | 0     | b₁         | Bit of scalar u₁ (bit-index 257-t)         |
//! | 1     | b₂         | Bit of scalar u₂ (bit-index 257-t)         |
//! | 2     | X          | Accumulator x-coord (Jacobian)             |
//! | 3     | Y          | Accumulator y-coord (Jacobian)             |
//! | 4     | Z          | Accumulator z-coord (Jacobian)             |
//! | 5     | X_mid      | Doubled point 2P x-coord (Jacobian)        |
//! | 6     | Y_mid      | Doubled point 2P y-coord (Jacobian)        |
//! | 7     | Z_mid      | Doubled point 2P z-coord (Jacobian)        |
//! | 8     | H          | Addition scratch: chord x-diff             |
//! | 9     | sel_init   | Boundary selector: 1 at row 0 (public)     |
//! | 10    | sel_final  | Boundary selector: 1 at row 257 (public)   |
//!
//! # Constraints (7 non-boundary + 4 boundary = 11 total)
//!
//! The ECDSA verification constraints operate in F_p, not in F₂\[X\].
//! Since the current Zinc+ UAIR framework only supports a single ring
//! per UAIR, and the PCS/PIOP operate on BinaryPoly traces, this crate
//! provides:
//!
//! - Correct trace dimensions for benchmarking (258 rows × 11 columns)
//! - Random witness generation (valid for PCS timing, not constraint-sound)
//! - The UAIR struct with constraint specifications
//!
//! Full constraint implementation requires the multi-ring UAIR extension
//! described in the paper (Section 5.2). See the [`constraints`] module
//! for the mathematical specification of all 7 constraints.
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

use crypto_primitives::crypto_bigint_int::Int;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, IdealCheck, ImpossibleIdeal},
};
use zinc_utils::from_ref::FromRef;

/// Total number of trace columns: 9 data + 2 selector = 11.
pub const NUM_COLS: usize = 11;

/// Number of trace rows: 1 (precomp) + 256 (scalar mul) + 1 (final check) = 258.
pub const NUM_ROWS: usize = 258;

/// Number of non-boundary constraints (from the paper: 7).
pub const NUM_CONSTRAINTS: usize = 0; // Placeholder — constraints require F_p ring

/// Number of boundary constraints enforced in the constraint system:
/// 2 (init Z=0, final signature check).
/// Booleanity of b₁/b₂ is checked by the verifier directly on the public
/// column data, so it does not cost constraint-system degrees.
pub const NUM_BOUNDARY_CONSTRAINTS: usize = 2;

// ─── Column indices ──────────────────────────────────────────────────────────

pub const COL_B1: usize = 0;
pub const COL_B2: usize = 1;
pub const COL_X: usize = 2;
pub const COL_Y: usize = 3;
pub const COL_Z: usize = 4;
pub const COL_X_MID: usize = 5;
pub const COL_Y_MID: usize = 6;
pub const COL_Z_MID: usize = 7;

// Down-row indices: with explicit shift specs only the shifted columns
// appear in the down row, so X/Y/Z map to 0/1/2 (not their original column
// indices).
const DOWN_X: usize = 0;
const DOWN_Y: usize = 1;
const DOWN_Z: usize = 2;
pub const COL_H: usize = 8;

// ─── Selector columns (boundary constraints) ───────────────────────────────

/// Initialization selector: 1 at row 0, 0 elsewhere (public).
pub const COL_SEL_INIT: usize = 9;
/// Finalization selector: 1 at row NUM_ROWS−1 (row 257), 0 elsewhere (public).
pub const COL_SEL_FINAL: usize = 10;

// ─── secp256k1 curve constants ──────────────────────────────────────────────
// Curve: y² = x³ + 7 over F_p
// p  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// G  = (Gx, Gy) — secp256k1 generator
// Q  = (Gx, Gy) — public key (= G for benchmark fixed-point witness)
// G+Q= 2G affine = (PGQx, PGQy) — precomputed table point

/// Generator x-coordinate (secp256k1).
/// 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
pub const GX: Int<4> = Int::from_words([
    0x59F2815B16F81798,
    0x029BFCDB2DCE28D9,
    0x55A06295CE870B07,
    0x79BE667EF9DCBBAC,
]);
/// Generator y-coordinate (secp256k1).
/// 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
pub const GY: Int<4> = Int::from_words([
    0x9C47D08FFB10D4B8,
    0xFD17B448A6855419,
    0x5DA4FBFC0E1108A8,
    0x483ADA7726A3C465,
]);
/// Public key x-coordinate (= G for the benchmark case).
pub const QX: Int<4> = GX;
/// Public key y-coordinate.
pub const QY: Int<4> = GY;
/// Precomputed (G+Q) x-coordinate (= 2G affine on secp256k1).
/// 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
pub const PGQX: Int<4> = Int::from_words([
    0xABAC09B95C709EE5,
    0x5C778E4B8CEF3CA7,
    0x3045406E95C07CD8,
    0xC6047F9441ED7D6D,
]);
/// Precomputed (G+Q) y-coordinate (= 2G affine on secp256k1).
/// 0x1AE168FEA63DC339A3C58419466CEAE1032688D15F9C819DE7C68DE6B1BBACD8
pub const PGQY: Int<4> = Int::from_words([
    0xE7C68DE6B1BBACD8,
    0x032688D15F9C819D,
    0xA3C58419466CEAE1,
    0x1AE168FEA63DC339,
]);

/// Expected signature verification result: r_sig = k·n + r.
///
/// For the fixed-point witness (Z=0 everywhere, identity result),
/// this value is unused because the final-check constraint is guarded
/// by Z (vacuously satisfied when Z=0). For a non-trivial witness,
/// set this to the actual k·n + r value.
pub const R_SIG: Int<4> = Int::from_words([0, 0, 0, 0]);

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
/// Since all 7 ECDSA constraints use `assert_zero`, the only ideal
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
                .map(|bp| {
                    let mut val: u64 = 0;
                    for (i, coeff) in bp.iter().enumerate() {
                        if coeff.into_inner() {
                            val |= 1u64 << i;
                        }
                    }
                    DensePolynomial::<i64, 1>::new([val as i64])
                })
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(col.num_vars, evaluations, zero)
        })
        .collect()
}

// ─── ECDSA UAIR ─────────────────────────────────────────────────────────────

/// The ECDSA verification UAIR.
///
/// Compatibility alias for `EcdsaUairBp`.
pub type EcdsaUair = EcdsaUairBp;

/// ECDSA UAIR over `BinaryPoly<32>` (placeholder, no constraints).
pub struct EcdsaUairBp;

/// ECDSA UAIR over `DensePolynomial<i64, 1>` (9 constraints: 7 Jacobian + 2 boundary).
pub struct EcdsaUairDp;

/// ECDSA UAIR over `Int<4>` (9 constraints: 7 Jacobian + 2 boundary, 256-bit integers).
pub struct EcdsaUairInt;

// ─── BinaryPoly<32> implementation (placeholder, no constraints) ────────────

impl Uair for EcdsaUairBp {
    type Ideal = EcdsaIdeal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NUM_COLS,
            arbitrary_poly_cols: 0,
            int_cols: 0,
            shifts: vec![],
            public_columns: vec![COL_B1, COL_B2, COL_SEL_INIT, COL_SEL_FINAL],
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        _b: &mut B,
        _up: zinc_uair::TraceRow<B::Expr>,
        _down: zinc_uair::TraceRow<B::Expr>,
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

// ─── DensePolynomial<i64, 1> implementation (9 constraints) ────────────────
//
// Ring: degree-0 integer polynomials (effectively i64 scalars).
// All constraints are assert_zero (exact integer equality).
// The witness uses integer arithmetic without modular reduction,
// so only works for small values where products don't overflow i64.
//
// This implementation uses a toy curve (y² = x³ + 7 over F_101) because
// secp256k1 coordinates overflow i64.  The Int<4> implementation below
// uses the real secp256k1 constants.
//
// S = Y² and R_a = T_y·Z_mid³ − Y_mid are inlined as sub-expressions
// (not separate columns). u₁, u₂ are public inputs (no trace columns).
// k is boundary-only (no trace column).
//
// Constraints C1–C7 are non-boundary (doubling + addition).
// Constraints B3–B4 are boundary (init, final check).
// Booleanity of b₁/b₂ is verified outside the constraint system.

/// Number of constraints for the i64 ECDSA UAIR: 7 non-boundary + 2 boundary.
pub const NUM_CONSTRAINTS_I64: usize = 9;

impl Uair for EcdsaUairDp {
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<i64, 1>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: NUM_COLS,
            int_cols: 0,
            shifts: vec![
                zinc_uair::ShiftSpec { source_col: COL_X, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_Y, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_Z, shift_amount: 1 },
            ],
            public_columns: vec![COL_B1, COL_B2, COL_SEL_INIT, COL_SEL_FINAL],
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: zinc_uair::TraceRow<B::Expr>,
        down: zinc_uair::TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&DensePolynomial<i64, 1>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<i64, 1>) -> Option<B::Expr>,
        IFromR: Fn(&ImpossibleIdeal) -> B::Ideal,
    {
        // Access columns via TraceRow — all are arbitrary_poly
        let up = up.arbitrary_poly;
        let down = down.arbitrary_poly;
        // ── Helpers ─────────────────────────────────────────────────
        let dp = |v: i64| DensePolynomial::<i64, 1>::new([v]);
        let cst = |v: i64| from_ref(&dp(v));
        let smul = |e: &B::Expr, v: i64| mbs(e, &dp(v)).unwrap();

        let one = cst(1);

        // ── C1: Doubled Z-coordinate: Z_mid = 2·Y·Z ───────────────
        b.assert_zero(
            up[COL_Z_MID].clone()
                - &smul(&(up[COL_Y].clone() * &up[COL_Z]), 2),
        );

        // ── C2: Doubled X-coordinate ───────────────────────────────
        // X_mid = 9X⁴ - 8X·Y² (S = Y² inlined)
        let x_sq = up[COL_X].clone() * &up[COL_X];
        let x_four = x_sq.clone() * &x_sq;
        let y_sq = up[COL_Y].clone() * &up[COL_Y]; // inlined S
        let x_y_sq = up[COL_X].clone() * &y_sq;
        b.assert_zero(
            up[COL_X_MID].clone() - &smul(&x_four, 9) + &smul(&x_y_sq, 8),
        );

        // ── C3: Doubled Y-coordinate ───────────────────────────────
        // Y_mid = 12X³Y² - 3X²·X_mid - 8Y⁴ (S = Y² inlined)
        let x_cubed_y_sq = x_sq.clone() * &up[COL_X] * &y_sq;
        let x_sq_xmid = x_sq * &up[COL_X_MID];
        let y_four = y_sq.clone() * &y_sq;
        b.assert_zero(
            up[COL_Y_MID].clone()
                - &smul(&x_cubed_y_sq, 12)
                + &smul(&x_sq_xmid, 3)
                + &smul(&y_four, 8),
        );

        // ── Shamir selector ────────────────────────────────────────
        // s = b1 + b2 - b1·b2 (= 1 iff any bit is set)
        let b1b2 = up[COL_B1].clone() * &up[COL_B2];
        let s = up[COL_B1].clone() + &up[COL_B2] - &b1b2;

        // Toy curve constants (F_101): y² = x³ + 7
        // G = (15, 7), Q = G, G+Q = 2G = (35, 82)
        let gx: i64 = 15;
        let gy: i64 = 7;
        let qx: i64 = 15;
        let qy: i64 = 7;
        let pgqx: i64 = 35;
        let pgqy: i64 = 82;
        let r_sig: i64 = 0;

        // Table point selection:
        // T_x = b1·(1-b2)·Gx + (1-b1)·b2·Qx + b1·b2·PGQx
        let one_minus_b2 = one.clone() - &up[COL_B2];
        let one_minus_b1 = one.clone() - &up[COL_B1];
        let b1_not_b2 = up[COL_B1].clone() * &one_minus_b2;
        let not_b1_b2 = one_minus_b1 * &up[COL_B2];
        let t_x = smul(&b1_not_b2, gx)
            + &smul(&not_b1_b2, qx)
            + &smul(&b1b2, pgqx);
        let t_y = smul(&b1_not_b2, gy)
            + &smul(&not_b1_b2, qy)
            + &smul(&b1b2, pgqy);

        // ── C4: Addition scratch H = T_x·Z_mid² - X_mid ───────────
        let zmid_sq = up[COL_Z_MID].clone() * &up[COL_Z_MID];
        b.assert_zero(
            up[COL_H].clone() - &(t_x * &zmid_sq) + &up[COL_X_MID],
        );

        // ── Inlined R_a = T_y·Z_mid³ - Y_mid ──────────────────────
        let zmid_cubed = zmid_sq.clone() * &up[COL_Z_MID];
        let r_a = t_y * &zmid_cubed - &up[COL_Y_MID];

        // ── C5: Result Z-coordinate ────────────────────────────────
        // Z[t+1] = (1-s)·Z_mid + s·(Z_mid·H)
        let one_minus_s = one.clone() - &s;
        let zmid_h = up[COL_Z_MID].clone() * &up[COL_H];
        b.assert_zero(
            down[DOWN_Z].clone()
                - &(one_minus_s.clone() * &up[COL_Z_MID])
                - &(s.clone() * &zmid_h),
        );

        // ── C6: Result X-coordinate ───────────────────────────────
        // When s=0: X[t+1] = X_mid
        // When s=1: X[t+1] = R_a² - H³ - 2·X_mid·H²
        let ra_sq = r_a.clone() * &r_a;
        let h_sq = up[COL_H].clone() * &up[COL_H];
        let h_cubed = h_sq.clone() * &up[COL_H];
        let xmid_h_sq = up[COL_X_MID].clone() * &h_sq;
        let add_x = ra_sq - &h_cubed - &smul(&xmid_h_sq, 2);
        b.assert_zero(
            down[DOWN_X].clone()
                - &(one_minus_s.clone() * &up[COL_X_MID])
                - &(s.clone() * &add_x),
        );

        // ── C7: Result Y-coordinate ───────────────────────────────
        // When s=0: Y[t+1] = Y_mid
        // When s=1: Y[t+1] = R_a·(X_mid·H² - X[t+1]) - Y_mid·H³
        let xmid_h_sq_2 = up[COL_X_MID].clone() * &h_sq;
        let ra_term = r_a * &(xmid_h_sq_2 - &down[DOWN_X]);
        let ymid_h_cubed = up[COL_Y_MID].clone() * &h_cubed;
        let add_y = ra_term - &ymid_h_cubed;
        b.assert_zero(
            down[DOWN_Y].clone()
                - &(one_minus_s * &up[COL_Y_MID])
                - &(s * &add_y),
        );

        // ═══════════════════════════════════════════════════════════
        //  Boundary constraints (B3–B4)
        //
        //  NOTE: Booleanity of b₁/b₂ (former B1/B2) is NOT enforced
        //  here. The verifier checks it directly on the public column
        //  data, saving 2 constraint-system degrees.
        // ═══════════════════════════════════════════════════════════

        // ── B3: Initialization: sel_init · Z = 0 ───────────────────
        // At row 0 the accumulator must be the identity (Z=0 in
        // Jacobian coordinates). Elsewhere sel_init=0 → vacuous.
        b.assert_zero(
            up[COL_SEL_INIT].clone() * &up[COL_Z],
        );

        // ── B4: Final signature check (guarded): ───────────────────
        //   sel_final · Z · (X − R_SIG · Z²) = 0
        //
        // When Z≠0 this enforces X = R_SIG·Z² (i.e. the affine
        // x-coordinate X/Z² equals R_SIG = k·n + r).
        // When Z=0 (identity / point at infinity) the constraint is
        // vacuously satisfied — a separate non-degeneracy check
        // would be needed to prove Z≠0 in a production system.
        let z_sq = up[COL_Z].clone() * &up[COL_Z];
        b.assert_zero(
            up[COL_SEL_FINAL].clone()
                * &up[COL_Z]
                * &(up[COL_X].clone() - &smul(&z_sq, r_sig)),
        );
    }
}

// ─── Int<4> implementation (9 constraints, 256-bit integer arithmetic) ──────
//
// Ring: Int<4> (256-bit signed integers). Uses the real secp256k1 curve
// constants (generator G, public key Q = G, precomputed 2G).
//
// This is the target ring for the unified ECDSA pipeline: the same Int<4>
// type is used for PCS commitments, PIOP constraints, and witness values.

/// Number of constraints for the Int<4> ECDSA UAIR: 7 non-boundary + 2 boundary.
pub const NUM_CONSTRAINTS_INT4: usize = 9;

impl Uair for EcdsaUairInt {
    type Ideal = ImpossibleIdeal;
    type Scalar = Int<4>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 0,
            int_cols: NUM_COLS,
            shifts: vec![
                zinc_uair::ShiftSpec { source_col: COL_X, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_Y, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_Z, shift_amount: 1 },
            ],
            public_columns: vec![COL_B1, COL_B2, COL_SEL_INIT, COL_SEL_FINAL],
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: zinc_uair::TraceRow<B::Expr>,
        down: zinc_uair::TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Int<4>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Int<4>) -> Option<B::Expr>,
        IFromR: Fn(&ImpossibleIdeal) -> B::Ideal,
    {
        // Access columns via TraceRow — all are int
        let up = up.int;
        let down = down.int;
        // ── Helpers ─────────────────────────────────────────────────
        let int = |v: i64| Int::<4>::from_ref(&v);
        let cst = |v: i64| from_ref(&int(v));
        let smul = |e: &B::Expr, v: i64| mbs(e, &int(v)).unwrap();
        let smul_c = |e: &B::Expr, c: &Int<4>| mbs(e, c).unwrap();

        let one = cst(1);

        // ── C1: Doubled Z-coordinate: Z_mid = 2·Y·Z ───────────────
        b.assert_zero(
            up[COL_Z_MID].clone()
                - &smul(&(up[COL_Y].clone() * &up[COL_Z]), 2),
        );

        // ── C2: Doubled X-coordinate ───────────────────────────────
        // X_mid = 9X⁴ - 8X·Y² (S = Y² inlined)
        let x_sq = up[COL_X].clone() * &up[COL_X];
        let x_four = x_sq.clone() * &x_sq;
        let y_sq = up[COL_Y].clone() * &up[COL_Y]; // inlined S
        let x_y_sq = up[COL_X].clone() * &y_sq;
        b.assert_zero(
            up[COL_X_MID].clone() - &smul(&x_four, 9) + &smul(&x_y_sq, 8),
        );

        // ── C3: Doubled Y-coordinate ───────────────────────────────
        // Y_mid = 12X³Y² - 3X²·X_mid - 8Y⁴ (S = Y² inlined)
        let x_cubed_y_sq = x_sq.clone() * &up[COL_X] * &y_sq;
        let x_sq_xmid = x_sq * &up[COL_X_MID];
        let y_four = y_sq.clone() * &y_sq;
        b.assert_zero(
            up[COL_Y_MID].clone()
                - &smul(&x_cubed_y_sq, 12)
                + &smul(&x_sq_xmid, 3)
                + &smul(&y_four, 8),
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
        let t_x = smul_c(&b1_not_b2, &GX)
            + &smul_c(&not_b1_b2, &QX)
            + &smul_c(&b1b2, &PGQX);
        let t_y = smul_c(&b1_not_b2, &GY)
            + &smul_c(&not_b1_b2, &QY)
            + &smul_c(&b1b2, &PGQY);

        // ── C4: Addition scratch H = T_x·Z_mid² - X_mid ───────────
        let zmid_sq = up[COL_Z_MID].clone() * &up[COL_Z_MID];
        b.assert_zero(
            up[COL_H].clone() - &(t_x * &zmid_sq) + &up[COL_X_MID],
        );

        // ── Inlined R_a = T_y·Z_mid³ - Y_mid ──────────────────────
        let zmid_cubed = zmid_sq.clone() * &up[COL_Z_MID];
        let r_a = t_y * &zmid_cubed - &up[COL_Y_MID];

        // ── C5: Result Z-coordinate ────────────────────────────────
        // Z[t+1] = (1-s)·Z_mid + s·(Z_mid·H)
        let one_minus_s = one.clone() - &s;
        let zmid_h = up[COL_Z_MID].clone() * &up[COL_H];
        b.assert_zero(
            down[DOWN_Z].clone()
                - &(one_minus_s.clone() * &up[COL_Z_MID])
                - &(s.clone() * &zmid_h),
        );

        // ── C6: Result X-coordinate ───────────────────────────────
        // When s=0: X[t+1] = X_mid
        // When s=1: X[t+1] = R_a² - H³ - 2·X_mid·H²
        let ra_sq = r_a.clone() * &r_a;
        let h_sq = up[COL_H].clone() * &up[COL_H];
        let h_cubed = h_sq.clone() * &up[COL_H];
        let xmid_h_sq = up[COL_X_MID].clone() * &h_sq;
        let add_x = ra_sq - &h_cubed - &smul(&xmid_h_sq, 2);
        b.assert_zero(
            down[DOWN_X].clone()
                - &(one_minus_s.clone() * &up[COL_X_MID])
                - &(s.clone() * &add_x),
        );

        // ── C7: Result Y-coordinate ───────────────────────────────
        // When s=0: Y[t+1] = Y_mid
        // When s=1: Y[t+1] = R_a·(X_mid·H² - X[t+1]) - Y_mid·H³
        let xmid_h_sq_2 = up[COL_X_MID].clone() * &h_sq;
        let ra_term = r_a * &(xmid_h_sq_2 - &down[DOWN_X]);
        let ymid_h_cubed = up[COL_Y_MID].clone() * &h_cubed;
        let add_y = ra_term - &ymid_h_cubed;
        b.assert_zero(
            down[DOWN_Y].clone()
                - &(one_minus_s * &up[COL_Y_MID])
                - &(s * &add_y),
        );

        // ═══════════════════════════════════════════════════════════
        //  Boundary constraints (B3–B4)
        //
        //  NOTE: Booleanity of b₁/b₂ (former B1/B2) is NOT enforced
        //  here. The verifier checks it directly on the public column
        //  data, saving 2 constraint-system degrees.
        // ═══════════════════════════════════════════════════════════

        // ── B3: Initialization: sel_init · Z = 0 ───────────────────
        // At row 0 the accumulator must be the identity (Z=0 in
        // Jacobian coordinates). Elsewhere sel_init=0 → vacuous.
        b.assert_zero(
            up[COL_SEL_INIT].clone() * &up[COL_Z],
        );

        // ── B4: Final signature check (guarded): ───────────────────
        //   sel_final · Z · (X − R_SIG · Z²) = 0
        //
        // When Z≠0 this enforces X = R_SIG·Z² (i.e. the affine
        // x-coordinate X/Z² equals R_SIG = k·n + r).
        // When Z=0 (identity / point at infinity) the constraint is
        // vacuously satisfied — a separate non-degeneracy check
        // would be needed to prove Z≠0 in a production system.
        let z_sq = up[COL_Z].clone() * &up[COL_Z];
        b.assert_zero(
            up[COL_SEL_FINAL].clone()
                * &up[COL_Z]
                * &(up[COL_X].clone() - &smul_c(&z_sq, &R_SIG)),
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
        assert_eq!(NUM_COLS, 11);
    }

    #[test]
    fn i64_constraint_count() {
        let n = count_constraints::<EcdsaUairDp>();
        assert_eq!(n, NUM_CONSTRAINTS_I64, "Expected 9 ECDSA i64 constraints");
    }

    #[test]
    fn i64_max_degree() {
        let d = count_max_degree::<EcdsaUairDp>();
        // C6 has degree 12: s(deg2) * R_a²(deg10) where R_a = T_y·Z_mid³ − Y_mid (deg5)
        assert!(d <= 12, "Max degree should be at most 12, got {d}");
        assert!(d >= 4, "Max degree should be at least 4 (doubling formulas), got {d}");
    }

    #[test]
    fn int4_constraint_count() {
        let n = count_constraints::<EcdsaUairInt>();
        assert_eq!(n, NUM_CONSTRAINTS_INT4, "Expected 9 ECDSA Int<4> constraints");
    }

    #[test]
    fn int4_max_degree() {
        let d = count_max_degree::<EcdsaUairInt>();
        assert!(d <= 12, "Max degree should be at most 12, got {d}");
        assert!(d >= 4, "Max degree should be at least 4 (doubling formulas), got {d}");
    }
}
