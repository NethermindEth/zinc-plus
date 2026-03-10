//! **No-F₂\[X\]** variant of the SHA-256 UAIR⁺.
//!
//! Replaces the 4 F₂\[X\] constraints of the form
//! `Q(...) ∈ (X³² − 1)` over F₂\[X\] with
//! `Q(...) − 2·μ ∈ (X³² − 1)` over Z\[X\],
//! where μ is a *quotient vector*—the part that vanishes when reducing
//! mod 2.
//!
//! Because the coefficients of μ can be 0–3 (not binary), each μ is
//! split into two binary-coefficient polynomials:
//!   `μ = μ_lo + 2·μ_hi`   (μ_lo, μ_hi ∈ {0,1}^{<32}[X])
//!
//! This adds **8 new bit-polynomial columns** to the trace (one μ_lo / μ_hi
//! pair per cyclotomic constraint C1–C4).  The 4 cyclotomic constraints
//! move from the Bp UAIR to a new Qx UAIR that operates over
//! `DensePolynomial<i64, 64>`.
//!
//! # Feature gate
//!
//! Enable with `--features no-f2x` on the `zinc-sha256-uair` crate
//! (or `zinc-snark/no-f2x` which forwards the flag).
//!
//! # Column layout (36 base + 3 integer; 42 with `true-ideal`)
//!
//! Columns 0–24 and int 0–2 are **identical** to the full UAIR.
//! Columns 25–32 are the new μ quotient decompositions:
//!
//! | Index | Name        | Description                              |
//! |-------|-------------|------------------------------------------|
//! | 25    | `mu_c1_lo`  | Σ₀ rotation quotient, low bit            |
//! | 26    | `mu_c1_hi`  | Σ₀ rotation quotient, high bit           |
//! | 27    | `mu_c2_lo`  | Σ₁ rotation quotient, low bit            |
//! | 28    | `mu_c2_hi`  | Σ₁ rotation quotient, high bit           |
//! | 29    | `mu_c3_lo`  | σ₀ rotation+shift quotient, low bit      |
//! | 30    | `mu_c3_hi`  | σ₀ rotation+shift quotient, high bit     |
//! | 31    | `mu_c4_lo`  | σ₁ rotation+shift quotient, low bit      |
//! | 32    | `mu_c4_hi`  | σ₁ rotation+shift quotient, high bit     |
//!
//! With `true-ideal`, 6 correction columns are added (33–38):
//!
//! | Index | Name            | Description                          |
//! |-------|-----------------|--------------------------------------|
//! | 33    | `corr_add_c7`   | Additive correction for C7           |
//! | 34    | `corr_add_c8`   | Additive correction for C8           |
//! | 35    | `corr_add_c9`   | Additive correction for C9           |
//! | 36    | `corr_sub_c7`   | Subtractive correction for C7        |
//! | 37    | `corr_sub_c8`   | Subtractive correction for C8        |
//! | 38    | `corr_sub_c9`   | Subtractive correction for C9        |
//!
//! # Constraint split
//!
//! **Bp UAIR** (`Sha256UairBpNoF2x`, 12 constraints):
//!   C5–C6 (shift decomposition, zero ideal) +
//!   C7–C16 (10 linking constraints, zero ideal).
//!
//! **Qx UAIR** (`Sha256UairQxNoF2x`, 7 constraints):
//!   C1–C4 (rotation / shift, cyclotomic ideal over Z\[X\], with μ correction) +
//!   C7–C9 (carry propagation, trivial or DegreeOne ideal).

use crypto_primitives::PrimeField;
use rand::RngCore;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, IdealCheck},
};
use zinc_utils::from_ref::FromRef;

use crate::{CyclotomicIdeal, witness::GenerateWitness};
use crate::{
    // Original column indices (unchanged 0–24)
    COL_A_HAT, COL_E_HAT, COL_W_HAT,
    COL_SIGMA0_HAT, COL_SIGMA1_HAT, COL_MAJ_HAT,
    COL_CH_EF_HAT, COL_CH_NEG_EG_HAT,
    COL_SIGMA0_W_HAT, COL_SIGMA1_W_HAT,
    COL_S0, COL_S1, COL_R0, COL_R1,
    COL_D_HAT, COL_H_HAT,
    COL_W_TM2, COL_W_TM7, COL_W_TM15, COL_W_TM16,
    COL_K_HAT,
    COL_A_TM1, COL_A_TM2, COL_E_TM1, COL_E_TM2,
    COL_INT_MU_A, COL_INT_MU_E, COL_INT_MU_W,
};

// ─── New column indices (25–32) ─────────────────────────────────────────────

/// Σ₀ rotation quotient, low bit of binary decomposition.
pub const COL_MU_C1_LO: usize = 25;
/// Σ₀ rotation quotient, high bit of binary decomposition.
pub const COL_MU_C1_HI: usize = 26;
/// Σ₁ rotation quotient, low bit of binary decomposition.
pub const COL_MU_C2_LO: usize = 27;
/// Σ₁ rotation quotient, high bit of binary decomposition.
pub const COL_MU_C2_HI: usize = 28;
/// σ₀ rotation+shift quotient, low bit of binary decomposition.
pub const COL_MU_C3_LO: usize = 29;
/// σ₀ rotation+shift quotient, high bit of binary decomposition.
pub const COL_MU_C3_HI: usize = 30;
/// σ₁ rotation+shift quotient, low bit of binary decomposition.
pub const COL_MU_C4_LO: usize = 31;
/// σ₁ rotation+shift quotient, high bit of binary decomposition.
pub const COL_MU_C4_HI: usize = 32;

// ─── Correction column indices (33–38, only with `true-ideal`) ──────────────
//
// Each carry constraint (C7–C9) needs TWO correction columns:
// - `CORR_ADD_*`: added to the constraint (handles negative c_eval)
// - `CORR_SUB_*`: subtracted from the constraint (handles positive c_eval)
//
// A single BinaryPoly<32> column can only represent non-negative values at
// X=2 (in [0, 2^32−1]).  The constraint violation c_eval can be positive or
// negative, so we need:
//     c_inner + corr_add − corr_sub ∈ (X − 2)
// with corr_add = max(−c_eval, 0) and corr_sub = max(c_eval, 0).

/// Additive correction column for C7 (a-update carry).
#[cfg(feature = "true-ideal")]
pub const COL_CORR_ADD_C7: usize = 33;
/// Additive correction column for C8 (e-update carry).
#[cfg(feature = "true-ideal")]
pub const COL_CORR_ADD_C8: usize = 34;
/// Additive correction column for C9 (W-schedule).
#[cfg(feature = "true-ideal")]
pub const COL_CORR_ADD_C9: usize = 35;
/// Subtractive correction column for C7 (a-update carry).
#[cfg(feature = "true-ideal")]
pub const COL_CORR_SUB_C7: usize = 36;
/// Subtractive correction column for C8 (e-update carry).
#[cfg(feature = "true-ideal")]
pub const COL_CORR_SUB_C8: usize = 37;
/// Subtractive correction column for C9 (W-schedule).
#[cfg(feature = "true-ideal")]
pub const COL_CORR_SUB_C9: usize = 38;

// ─── Counts ─────────────────────────────────────────────────────────────────

/// Number of bit-polynomial columns without true-ideal (33 = original 25 + 8 μ).
#[cfg(not(feature = "true-ideal"))]
pub const NO_F2X_NUM_BITPOLY_COLS: usize = 33;

/// Number of bit-polynomial columns with true-ideal (39 = original 25 + 8 μ + 6 correction).
#[cfg(feature = "true-ideal")]
pub const NO_F2X_NUM_BITPOLY_COLS: usize = 39;

/// Number of integer columns (unchanged: 3).
pub const NO_F2X_NUM_INT_COLS: usize = 3;

/// Total number of trace columns.
pub const NO_F2X_NUM_COLS: usize = NO_F2X_NUM_BITPOLY_COLS + NO_F2X_NUM_INT_COLS;

/// Number of constraints in the Bp no-F₂[X] UAIR.
/// C5–C6 (shift decomp) + C7–C16 (10 linking) = 12.
pub const NO_F2X_BP_NUM_CONSTRAINTS: usize = 12;

/// Number of constraints in the Qx no-F₂[X] UAIR.
/// C1–C4 (rotation with μ, cyclotomic) + C7–C9 (carry, trivial) = 7.
pub const NO_F2X_QX_NUM_CONSTRAINTS: usize = 7;

// ─── Qx Ideal for no-F₂[X] ────────────────────────────────────────────────

/// Ideal type for the Q\[X\] no-F₂\[X\] UAIR.
///
/// - `Cyclotomic`: membership in (X³² − 1) over Z\[X\] (or F\_p\[X\]).
/// - `Trivial`: every polynomial is a member (used for carry constraints).
#[derive(Clone, Copy, Debug)]
pub enum Sha256QxNoF2xIdeal {
    /// The cyclotomic ideal (X³² − 1).
    Cyclotomic,
    /// The degree-one ideal (X − 2): p(2) = 0.
    #[cfg(feature = "true-ideal")]
    DegreeOne,
    /// The trivial ideal: contains every polynomial.
    Trivial,
}

impl Ideal for Sha256QxNoF2xIdeal {}

impl FromRef<Sha256QxNoF2xIdeal> for Sha256QxNoF2xIdeal {
    #[inline(always)]
    fn from_ref(ideal: &Sha256QxNoF2xIdeal) -> Self {
        *ideal
    }
}

impl IdealCheck<DensePolynomial<i64, 64>> for Sha256QxNoF2xIdeal {
    fn contains(&self, value: &DensePolynomial<i64, 64>) -> bool {
        match self {
            Sha256QxNoF2xIdeal::Cyclotomic => {
                // Reduce mod (X³² − 1) over Z[X]: X³² ≡ 1, so fold.
                let mut reduced = [0i64; 32];
                for (i, &c) in value.coeffs.iter().enumerate() {
                    reduced[i % 32] = reduced[i % 32].wrapping_add(c);
                }
                reduced.iter().all(|&c| c == 0)
            }
            #[cfg(feature = "true-ideal")]
            Sha256QxNoF2xIdeal::DegreeOne => {
                // Evaluate at X = 2: f(2) = Σ c_i · 2^i
                let mut eval: i64 = 0;
                for (i, &c) in value.coeffs.iter().enumerate() {
                    eval = eval.wrapping_add(c.wrapping_mul(1i64.wrapping_shl(i as u32)));
                }
                eval == 0
            }
            Sha256QxNoF2xIdeal::Trivial => true,
        }
    }
}

// ─── Field-level ideal for verification ─────────────────────────────────────

/// The no-F₂\[X\] ideal lifted to a prime field.
#[derive(Clone, Copy, Debug)]
pub enum Sha256QxNoF2xIdealOverF {
    /// Cyclotomic (X³² − 1) check over F_p\[X\].
    Cyclotomic,
    /// Degree-one ideal (X − 2): evaluate p(2) and check = 0.
    #[cfg(feature = "true-ideal")]
    DegreeOne,
    /// Trivial ideal.
    Trivial,
}

impl Ideal for Sha256QxNoF2xIdealOverF {}

impl FromRef<Sha256QxNoF2xIdealOverF> for Sha256QxNoF2xIdealOverF {
    #[inline(always)]
    fn from_ref(ideal: &Sha256QxNoF2xIdealOverF) -> Self {
        *ideal
    }
}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for Sha256QxNoF2xIdealOverF {
    fn contains(&self, value: &DynamicPolynomialF<F>) -> bool {
        match self {
            Sha256QxNoF2xIdealOverF::Cyclotomic => {
                // Reduce mod (X³² − 1) in F_p[X]: fold coefficients.
                if value.coeffs.is_empty() {
                    return true;
                }
                let cfg = value.coeffs[0].cfg();
                let zero = F::zero_with_cfg(cfg);
                let mut reduced: Vec<F> = vec![zero; 32];
                for (i, coeff) in value.coeffs.iter().enumerate() {
                    let j = i % 32;
                    reduced[j] = reduced[j].clone() + coeff;
                }
                reduced.iter().all(|c| F::is_zero(c))
            }
            #[cfg(feature = "true-ideal")]
            Sha256QxNoF2xIdealOverF::DegreeOne => {
                // Evaluate p(X) at X = 2 in F_p.
                if value.coeffs.is_empty() {
                    return true;
                }
                let cfg = value.coeffs[0].cfg();
                let mut result = F::zero_with_cfg(cfg);
                let mut power = F::one_with_cfg(cfg);
                for coeff in &value.coeffs {
                    result = result + &(coeff.clone() * &power);
                    power = power.clone() + &power; // power *= 2
                }
                F::is_zero(&result)
            }
            Sha256QxNoF2xIdealOverF::Trivial => true,
        }
    }
}

// ─── Bp no-F₂[X] UAIR ─────────────────────────────────────────────────────

/// SHA-256 Bp UAIR variant without F₂\[X\] cyclotomic constraints.
///
/// Emits 12 constraints (all `assert_zero`):
///   C5–C6 (shift decomposition) + C7–C16 (10 linking constraints).
///
/// The rotation / shift constraints C1–C4 are moved to the paired
/// [`Sha256UairQxNoF2x`] UAIR which checks them over Z\[X\] with
/// the μ quotient correction.
pub struct Sha256UairBpNoF2x;

// Down-row indices (same mapping as original Bp UAIR).
const NF2X_DOWN_BP_D: usize = 0;
const NF2X_DOWN_BP_H: usize = 1;
const NF2X_DOWN_BP_W_TM2: usize = 2;
const NF2X_DOWN_BP_W_TM7: usize = 3;
const NF2X_DOWN_BP_W_TM15: usize = 4;
const NF2X_DOWN_BP_W_TM16: usize = 5;
const NF2X_DOWN_BP_A_TM1: usize = 6;
const NF2X_DOWN_BP_A_TM2: usize = 7;
const NF2X_DOWN_BP_E_TM1: usize = 8;
const NF2X_DOWN_BP_E_TM2: usize = 9;

impl Uair for Sha256UairBpNoF2x {
    type Ideal = CyclotomicIdeal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NO_F2X_NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NO_F2X_NUM_INT_COLS,
            shifts: vec![
                zinc_uair::ShiftSpec { source_col: COL_D_HAT,  shift_amount: 3 },
                zinc_uair::ShiftSpec { source_col: COL_H_HAT,  shift_amount: 3 },
                zinc_uair::ShiftSpec { source_col: COL_W_TM2,  shift_amount: 2 },
                zinc_uair::ShiftSpec { source_col: COL_W_TM7,  shift_amount: 7 },
                zinc_uair::ShiftSpec { source_col: COL_W_TM15, shift_amount: 15 },
                zinc_uair::ShiftSpec { source_col: COL_W_TM16, shift_amount: 16 },
                zinc_uair::ShiftSpec { source_col: COL_A_TM1,  shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_A_TM2,  shift_amount: 2 },
                zinc_uair::ShiftSpec { source_col: COL_E_TM1,  shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_E_TM2,  shift_amount: 2 },
            ],
            public_columns: {
                let mut cols = vec![
                    COL_W_HAT, COL_K_HAT,
                    COL_S0, COL_S1, COL_R0, COL_R1,
                    COL_W_TM2, COL_W_TM7, COL_W_TM15, COL_W_TM16,
                ];
                #[cfg(feature = "true-ideal")]
                {
                    cols.push(COL_CORR_ADD_C7);
                    cols.push(COL_CORR_ADD_C8);
                    cols.push(COL_CORR_ADD_C9);
                    cols.push(COL_CORR_SUB_C7);
                    cols.push(COL_CORR_SUB_C8);
                    cols.push(COL_CORR_SUB_C9);
                }
                cols
            },
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: zinc_uair::TraceRow<B::Expr>,
        down: zinc_uair::TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &BinaryPoly<32>) -> Option<B::Expr>,
        IFromR: Fn(&CyclotomicIdeal) -> B::Ideal,
    {
        let up = up.binary_poly;
        let bp_down = down.binary_poly;

        // ── Constraint 5: σ₀ shift decomposition ───────────────────────
        //   W_tm15 = R₀ + X³ · S₀
        let x_cubed = from_ref(&BinaryPoly::<32>::from(1u32 << 3));
        b.assert_zero(
            up[COL_W_TM15].clone()
                - &up[COL_R0]
                - &(up[COL_S0].clone() * &x_cubed),
        );

        // ── Constraint 6: σ₁ shift decomposition ───────────────────────
        //   W_tm2 = R₁ + X¹⁰ · S₁
        let x_10 = from_ref(&BinaryPoly::<32>::from(1u32 << 10));
        b.assert_zero(
            up[COL_W_TM2].clone()
                - &up[COL_R1]
                - &(up[COL_S1].clone() * &x_10),
        );

        // ── Constraint 7: d-link (shift-by-3) ──────────────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_D].clone() - &up[COL_A_HAT]);

        // ── Constraint 8: h-link (shift-by-3) ──────────────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_H].clone() - &up[COL_E_HAT]);

        // ── Constraint 9: W_tm2-link (shift-by-2) ──────────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_W_TM2].clone() - &up[COL_W_HAT]);

        // ── Constraint 10: W_tm7-link (shift-by-7) ─────────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_W_TM7].clone() - &up[COL_W_HAT]);

        // ── Constraint 11: W_tm15-link (shift-by-15) ───────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_W_TM15].clone() - &up[COL_W_HAT]);

        // ── Constraint 12: W_tm16-link (shift-by-16) ───────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_W_TM16].clone() - &up[COL_W_HAT]);

        // ── Constraint 13: a_tm1-link (shift-by-1) ─────────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_A_TM1].clone() - &up[COL_A_HAT]);

        // ── Constraint 14: a_tm2-link (shift-by-2) ─────────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_A_TM2].clone() - &up[COL_A_HAT]);

        // ── Constraint 15: e_tm1-link (shift-by-1) ─────────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_E_TM1].clone() - &up[COL_E_HAT]);

        // ── Constraint 16: e_tm2-link (shift-by-2) ─────────────────────
        b.assert_zero(bp_down[NF2X_DOWN_BP_E_TM2].clone() - &up[COL_E_HAT]);
    }
}

// ─── Qx no-F₂[X] UAIR ─────────────────────────────────────────────────────

/// SHA-256 Qx UAIR variant with cyclotomic rotation constraints moved
/// from F₂\[X\] to Z\[X\] via quotient correction.
///
/// Emits 7 constraints:
///   C1–C4: rotation / shift constraints with μ correction, cyclotomic ideal.
///   C7–C9: carry propagation, trivial ideal.
///
/// All constraints are degree 1 (`max_degree = 1`), enabling MLE-first IC.
pub struct Sha256UairQxNoF2x;

// Down-row indices for the Qx no-F₂[X] UAIR.
// Shifts: â(1), ê(1). Both are binary_poly → down.binary_poly[0..2].
const NF2X_DOWN_QX_A: usize = 0;
const NF2X_DOWN_QX_E: usize = 1;

impl Uair for Sha256UairQxNoF2x {
    type Ideal = Sha256QxNoF2xIdeal;
    type Scalar = DensePolynomial<i64, 64>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NO_F2X_NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NO_F2X_NUM_INT_COLS,
            shifts: vec![
                zinc_uair::ShiftSpec { source_col: COL_A_HAT, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_E_HAT, shift_amount: 1 },
            ],
            public_columns: {
                let mut cols = vec![
                    COL_W_HAT, COL_K_HAT,
                    COL_S0, COL_S1, COL_R0, COL_R1,
                    COL_W_TM2, COL_W_TM7, COL_W_TM15, COL_W_TM16,
                ];
                #[cfg(feature = "true-ideal")]
                {
                    cols.push(COL_CORR_ADD_C7);
                    cols.push(COL_CORR_ADD_C8);
                    cols.push(COL_CORR_ADD_C9);
                    cols.push(COL_CORR_SUB_C7);
                    cols.push(COL_CORR_SUB_C8);
                    cols.push(COL_CORR_SUB_C9);
                }
                cols
            },
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: zinc_uair::TraceRow<B::Expr>,
        down: zinc_uair::TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&DensePolynomial<i64, 64>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<i64, 64>) -> Option<B::Expr>,
        IFromR: Fn(&Sha256QxNoF2xIdeal) -> B::Ideal,
    {
        let bp_up = up.binary_poly;
        let int_up = up.int;
        let bp_down = down.binary_poly;
        let cyclotomic = ideal_from_ref(&Sha256QxNoF2xIdeal::Cyclotomic);
        #[cfg(not(feature = "true-ideal"))]
        let trivial = ideal_from_ref(&Sha256QxNoF2xIdeal::Trivial);

        // ── Helpers: small integer constants as DensePolynomial ──────────

        fn const_poly(val: i64) -> DensePolynomial<i64, 64> {
            let mut coeffs = [0i64; 64];
            coeffs[0] = val;
            DensePolynomial { coeffs }
        }

        fn mono_poly(pos: usize) -> DensePolynomial<i64, 64> {
            let mut coeffs = [0i64; 64];
            coeffs[pos] = 1;
            DensePolynomial { coeffs }
        }

        let two = from_ref(&const_poly(2));
        let four = from_ref(&const_poly(4));

        // ── Rotation polynomials (same as original, lifted to Z[X]) ─────

        // ρ₀ = X³⁰ + X¹⁹ + X¹⁰
        let mut rho0_coeffs = [0i64; 64];
        rho0_coeffs[30] = 1;
        rho0_coeffs[19] = 1;
        rho0_coeffs[10] = 1;
        let rho0 = from_ref(&DensePolynomial { coeffs: rho0_coeffs });

        // ρ₁ = X²⁶ + X²¹ + X⁷
        let mut rho1_coeffs = [0i64; 64];
        rho1_coeffs[26] = 1;
        rho1_coeffs[21] = 1;
        rho1_coeffs[7] = 1;
        let rho1 = from_ref(&DensePolynomial { coeffs: rho1_coeffs });

        // ρ_{σ₀} = X²⁵ + X¹⁴
        let mut rho_s0_coeffs = [0i64; 64];
        rho_s0_coeffs[25] = 1;
        rho_s0_coeffs[14] = 1;
        let rho_sigma0 = from_ref(&DensePolynomial { coeffs: rho_s0_coeffs });

        // ρ_{σ₁} = X¹⁵ + X¹³
        let mut rho_s1_coeffs = [0i64; 64];
        rho_s1_coeffs[15] = 1;
        rho_s1_coeffs[13] = 1;
        let rho_sigma1 = from_ref(&DensePolynomial { coeffs: rho_s1_coeffs });

        // ── Constraint C1: Σ₀ rotation over Z[X] with μ correction ─────
        //   â · ρ₀ − Σ̂₀ − 2·μ₁_lo − 4·μ₁_hi ∈ (X³² − 1)
        b.assert_in_ideal(
            bp_up[COL_A_HAT].clone() * &rho0
                - &bp_up[COL_SIGMA0_HAT]
                - &(bp_up[COL_MU_C1_LO].clone() * &two)
                - &(bp_up[COL_MU_C1_HI].clone() * &four),
            &cyclotomic,
        );

        // ── Constraint C2: Σ₁ rotation over Z[X] with μ correction ─────
        //   ê · ρ₁ − Σ̂₁ − 2·μ₂_lo − 4·μ₂_hi ∈ (X³² − 1)
        b.assert_in_ideal(
            bp_up[COL_E_HAT].clone() * &rho1
                - &bp_up[COL_SIGMA1_HAT]
                - &(bp_up[COL_MU_C2_LO].clone() * &two)
                - &(bp_up[COL_MU_C2_HI].clone() * &four),
            &cyclotomic,
        );

        // ── Constraint C3: σ₀ rotation+shift over Z[X] with μ correction
        //   Ŵ_tm15 · ρ_{σ₀} + S₀ − σ̂₀_w − 2·μ₃_lo − 4·μ₃_hi ∈ (X³² − 1)
        b.assert_in_ideal(
            bp_up[COL_W_TM15].clone() * &rho_sigma0
                + &bp_up[COL_S0]
                - &bp_up[COL_SIGMA0_W_HAT]
                - &(bp_up[COL_MU_C3_LO].clone() * &two)
                - &(bp_up[COL_MU_C3_HI].clone() * &four),
            &cyclotomic,
        );

        // ── Constraint C4: σ₁ rotation+shift over Z[X] with μ correction
        //   Ŵ_tm2 · ρ_{σ₁} + S₁ − σ̂₁_w − 2·μ₄_lo − 4·μ₄_hi ∈ (X³² − 1)
        b.assert_in_ideal(
            bp_up[COL_W_TM2].clone() * &rho_sigma1
                + &bp_up[COL_S1]
                - &bp_up[COL_SIGMA1_W_HAT]
                - &(bp_up[COL_MU_C4_LO].clone() * &two)
                - &(bp_up[COL_MU_C4_HI].clone() * &four),
            &cyclotomic,
        );

        // ── Carry propagation constants ─────────────────────────────────

        // X^32 as a polynomial
        let x32 = from_ref(&mono_poly(32));

        // ── Carry constraints C7–C9 ─────────────────────────────────────
        //
        // Without `true-ideal` (default): trivial ideal, degree 1,
        // enabling the MLE-first IC path.
        //
        // With `true-ideal`: (X − 2) ideal with correction columns,
        // degree 1, enabling the MLE-first IC path.

        let c7_inner =
            bp_down[NF2X_DOWN_QX_A].clone()
                - &bp_up[COL_H_HAT]
                - &bp_up[COL_SIGMA1_HAT]
                - &bp_up[COL_CH_EF_HAT]
                - &bp_up[COL_CH_NEG_EG_HAT]
                - &bp_up[COL_K_HAT]
                - &bp_up[COL_W_HAT]
                - &bp_up[COL_SIGMA0_HAT]
                - &bp_up[COL_MAJ_HAT]
                + &(int_up[COL_INT_MU_A].clone() * &x32);

        let c8_inner =
            bp_down[NF2X_DOWN_QX_E].clone()
                - &bp_up[COL_D_HAT]
                - &bp_up[COL_H_HAT]
                - &bp_up[COL_SIGMA1_HAT]
                - &bp_up[COL_CH_EF_HAT]
                - &bp_up[COL_CH_NEG_EG_HAT]
                - &bp_up[COL_K_HAT]
                - &bp_up[COL_W_HAT]
                + &(int_up[COL_INT_MU_E].clone() * &x32);

        let c9_inner =
            bp_up[COL_W_HAT].clone()
                - &bp_up[COL_W_TM16]
                - &bp_up[COL_SIGMA0_W_HAT]
                - &bp_up[COL_W_TM7]
                - &bp_up[COL_SIGMA1_W_HAT]
                + &(int_up[COL_INT_MU_W].clone() * &x32);

        #[cfg(not(feature = "true-ideal"))]
        {
            // ── C7: a-update carry propagation (trivial ideal) ──────────
            b.assert_in_ideal(c7_inner, &trivial);

            // ── C8: e-update carry propagation (trivial ideal) ──────────
            b.assert_in_ideal(c8_inner, &trivial);

            // ── C9: Message schedule recurrence (trivial ideal) ─────────
            b.assert_in_ideal(c9_inner, &trivial);
        }

        #[cfg(feature = "true-ideal")]
        {
            let degree_one = ideal_from_ref(&Sha256QxNoF2xIdeal::DegreeOne);

            // ── C7: (a-update + corr_add − corr_sub) ∈ (X − 2) ─────────
            b.assert_in_ideal(
                c7_inner + &bp_up[COL_CORR_ADD_C7] - &bp_up[COL_CORR_SUB_C7],
                &degree_one,
            );

            // ── C8: (e-update + corr_add − corr_sub) ∈ (X − 2) ─────────
            b.assert_in_ideal(
                c8_inner + &bp_up[COL_CORR_ADD_C8] - &bp_up[COL_CORR_SUB_C8],
                &degree_one,
            );

            // ── C9: (W-schedule + corr_add − corr_sub) ∈ (X − 2) ───────
            b.assert_in_ideal(
                c9_inner + &bp_up[COL_CORR_ADD_C9] - &bp_up[COL_CORR_SUB_C9],
                &degree_one,
            );
        }
    }
}

// ─── Witness generation ─────────────────────────────────────────────────────

/// Compute the μ quotient for a single constraint row.
///
/// Given:
///   - `poly_a`: first operand (as u32 bit pattern)
///   - `rho_positions`: nonzero monomial positions of the rotation polynomial
///   - `terms`: additional polynomial terms `(bit_pattern, sign)`:
///       `sign = +1` for addition, `sign = -1` for subtraction
///
/// Returns `(mu_lo, mu_hi)` as u32 bit patterns, where the quotient
/// polynomial μ = μ_lo + 2·μ_hi has binary-coefficient parts.
fn compute_mu(
    poly_a: u32,
    rho_positions: &[usize],
    terms: &[(u32, i32)],
) -> (u32, u32) {
    // Step 1: Integer polynomial multiplication
    let mut coeffs = [0i32; 64];
    for bit in 0..32u32 {
        if (poly_a >> bit) & 1 == 1 {
            for &rho_pos in rho_positions {
                coeffs[bit as usize + rho_pos] += 1;
            }
        }
    }

    // Step 2: Add/subtract additional polynomial terms
    for &(poly, sign) in terms {
        for bit in 0..32u32 {
            if (poly >> bit) & 1 == 1 {
                coeffs[bit as usize] += sign;
            }
        }
    }

    // Step 3: Reduce mod (X³² − 1)
    let mut reduced = [0i32; 32];
    for i in 0..64 {
        reduced[i % 32] += coeffs[i];
    }

    // Step 4: μ = reduced / 2, then split into lo/hi
    let mut mu_lo: u32 = 0;
    let mut mu_hi: u32 = 0;
    for j in 0..32 {
        debug_assert!(
            reduced[j] % 2 == 0,
            "Residue coefficient not even at position {j}: {}",
            reduced[j]
        );
        let mu_j = reduced[j] / 2;
        debug_assert!(
            mu_j >= 0 && mu_j < 4,
            "μ coefficient out of range [0,3] at position {j}: {mu_j}"
        );
        if (mu_j & 1) != 0 {
            mu_lo |= 1 << j;
        }
        if (mu_j & 2) != 0 {
            mu_hi |= 1 << j;
        }
    }

    (mu_lo, mu_hi)
}

/// Compute the 6 correction columns and 3 corrected carry columns for `true-ideal`.
///
/// Each carry constraint (C7/C8/C9) needs TWO correction columns:
/// - `corr_add`: added to the constraint expression (handles negative residual)
/// - `corr_sub`: subtracted from the constraint expression (handles positive residual)
///
/// The base trace only sets carry values (μ_a, μ_e, μ_W) at active rows.
/// At inactive boundary rows the carries are 0, making the constraint
/// violation potentially exceed `2^32`, which overflows a `BinaryPoly<32>`.
///
/// This function recomputes the carry at **every** row so that the
/// residual always fits in a `u32`:
///     μ = ⌊max(sum − next, 0) / 2^32⌋
///     residual = next − sum + μ · 2^32   (fits in [−(2^32−1), 2^32−1])
///
/// The corrected constraint is:
///     c_inner + corr_add − corr_sub ∈ (X − 2)
///
/// Returns `(corrections, corrected_carries)`:
/// - corrections: `[corr_add_c7, corr_add_c8, corr_add_c9,
///                  corr_sub_c7, corr_sub_c8, corr_sub_c9]`
/// - corrected_carries: `[mu_a, mu_e, mu_w]` (for ALL rows)
#[cfg(feature = "true-ideal")]
fn compute_corrections_and_carries(
    base: &[DenseMultilinearExtension<BinaryPoly<32>>],
    num_vars: usize,
) -> ([Vec<BinaryPoly<32>>; 6], [Vec<BinaryPoly<32>>; 3]) {
    let num_rows = 1usize << num_vars;

    let bp_val = |col: usize, t: usize| -> u64 {
        let bp = &base[col].evaluations[t];
        let mut val = 0u64;
        for (i, coeff) in bp.iter().enumerate() {
            if coeff.into_inner() {
                val |= 1u64 << i;
            }
        }
        val
    };

    let mut corr_add_c7 = vec![BinaryPoly::<32>::from(0u32); num_rows];
    let mut corr_add_c8 = vec![BinaryPoly::<32>::from(0u32); num_rows];
    let mut corr_add_c9 = vec![BinaryPoly::<32>::from(0u32); num_rows];
    let mut corr_sub_c7 = vec![BinaryPoly::<32>::from(0u32); num_rows];
    let mut corr_sub_c8 = vec![BinaryPoly::<32>::from(0u32); num_rows];
    let mut corr_sub_c9 = vec![BinaryPoly::<32>::from(0u32); num_rows];

    let mut mu_a_col = vec![BinaryPoly::<32>::from(0u32); num_rows];
    let mut mu_e_col = vec![BinaryPoly::<32>::from(0u32); num_rows];
    let mut mu_w_col = vec![BinaryPoly::<32>::from(0u32); num_rows];

    /// Compute the carry and residual for a single constraint.
    ///
    /// Given `next` (the "a_next" or "e_next" value) and `sum` (the sum
    /// of the other terms), returns `(mu, residual)` where:
    ///   mu = ⌊max(sum − next, 0) / 2^32⌋
    ///   residual = next − sum + mu · 2^32
    /// The residual is guaranteed to fit in `i64` with `|residual| < 2^32`.
    fn carry_and_residual(next: u64, sum: u64) -> (u32, i64) {
        if sum > next {
            let diff = sum - next;
            let mu = (diff >> 32) as u32;
            let residual = next as i64 - sum as i64 + (mu as i64) * (1i64 << 32);
            (mu, residual)
        } else {
            // sum <= next: residual = next - sum >= 0, fits in u32
            (0, next as i64 - sum as i64)
        }
    }

    for t in 0..num_rows {
        // The Qx UAIR shift-by-1 gives a_hat[t+1] for C7, e_hat[t+1] for C8.
        let a_next = if t + 1 < num_rows { bp_val(COL_A_HAT, t + 1) } else { 0 };
        let e_next = if t + 1 < num_rows { bp_val(COL_E_HAT, t + 1) } else { 0 };

        let h_val = bp_val(COL_H_HAT, t);
        let sigma1_val = bp_val(COL_SIGMA1_HAT, t);
        let ch_ef_val = bp_val(COL_CH_EF_HAT, t);
        let ch_neg_eg_val = bp_val(COL_CH_NEG_EG_HAT, t);
        let k_val = bp_val(COL_K_HAT, t);
        let w_val = bp_val(COL_W_HAT, t);
        let sigma0_val = bp_val(COL_SIGMA0_HAT, t);
        let maj_val = bp_val(COL_MAJ_HAT, t);
        let d_val = bp_val(COL_D_HAT, t);

        // C7: a[t+1] - (h + Σ₁ + ch_ef + ch_neg_eg + K + W + Σ₀ + Maj) + μ_a · 2^32
        let c7_sum = h_val + sigma1_val + ch_ef_val + ch_neg_eg_val
            + k_val + w_val + sigma0_val + maj_val;
        let (mu_a, c7_residual) = carry_and_residual(a_next, c7_sum);
        mu_a_col[t] = BinaryPoly::from(mu_a);
        if c7_residual < 0 {
            corr_add_c7[t] = BinaryPoly::from((-c7_residual) as u32);
        } else if c7_residual > 0 {
            corr_sub_c7[t] = BinaryPoly::from(c7_residual as u32);
        }

        // C8: e[t+1] - (d + h + Σ₁ + ch_ef + ch_neg_eg + K + W) + μ_e · 2^32
        let c8_sum = d_val + h_val + sigma1_val + ch_ef_val
            + ch_neg_eg_val + k_val + w_val;
        let (mu_e, c8_residual) = carry_and_residual(e_next, c8_sum);
        mu_e_col[t] = BinaryPoly::from(mu_e);
        if c8_residual < 0 {
            corr_add_c8[t] = BinaryPoly::from((-c8_residual) as u32);
        } else if c8_residual > 0 {
            corr_sub_c8[t] = BinaryPoly::from(c8_residual as u32);
        }

        // C9: W - (W_tm16 + σ₀_w + W_tm7 + σ₁_w) + μ_W · 2^32
        let w_tm16_val = bp_val(COL_W_TM16, t);
        let sigma0_w_val = bp_val(COL_SIGMA0_W_HAT, t);
        let w_tm7_val = bp_val(COL_W_TM7, t);
        let sigma1_w_val = bp_val(COL_SIGMA1_W_HAT, t);
        let c9_sum = w_tm16_val + sigma0_w_val + w_tm7_val + sigma1_w_val;
        let (mu_w, c9_residual) = carry_and_residual(w_val, c9_sum);
        mu_w_col[t] = BinaryPoly::from(mu_w);
        if c9_residual < 0 {
            corr_add_c9[t] = BinaryPoly::from((-c9_residual) as u32);
        } else if c9_residual > 0 {
            corr_sub_c9[t] = BinaryPoly::from(c9_residual as u32);
        }
    }

    (
        [corr_add_c7, corr_add_c8, corr_add_c9,
         corr_sub_c7, corr_sub_c8, corr_sub_c9],
        [mu_a_col, mu_e_col, mu_w_col],
    )
}

/// Generate the full witness for the no-F₂\[X\] variant.
///
/// Delegates to the base `Sha256UairBp` witness generator for columns 0–27,
/// then computes the 8 μ quotient decomposition columns (25–32) from
/// the existing trace data.
///
/// The total layout is: `binary_poly[0..33]` || `int[0..3]`.
/// With `true-ideal`, 6 correction columns are appended: `binary_poly[0..39]` || `int[0..3]`.
pub fn generate_no_f2x_witness(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    // Generate the base 28-column trace.
    let base = <crate::Sha256UairBp as GenerateWitness<BinaryPoly<32>>>::generate_witness(
        num_vars, rng,
    );
    assert_eq!(base.len(), crate::NUM_COLS); // 28

    let num_rows = 1usize << num_vars;
    // Helper to extract u32 bit pattern from a BinaryPoly column at row t.
    let bp_val = |col: usize, t: usize| -> u32 {
        let bp = &base[col].evaluations[t];
        let mut val = 0u32;
        for (i, coeff) in bp.iter().enumerate() {
            if coeff.into_inner() {
                val |= 1u32 << i;
            }
        }
        val
    };

    // Rotation polynomial positions.
    let rho0_pos: &[usize] = &[30, 19, 10]; // Σ₀
    let rho1_pos: &[usize] = &[26, 21, 7];  // Σ₁
    let rho_s0_pos: &[usize] = &[25, 14];   // σ₀
    let rho_s1_pos: &[usize] = &[15, 13];   // σ₁

    // Allocate the 8 new μ columns.
    let mut mu_cols: Vec<Vec<BinaryPoly<32>>> =
        (0..8).map(|_| vec![BinaryPoly::<32>::from(0u32); num_rows]).collect();

    for t in 0..num_rows {
        // C1: â · ρ₀ − Σ̂₀
        let a_hat = bp_val(COL_A_HAT, t);
        let sigma0 = bp_val(COL_SIGMA0_HAT, t);
        let (c1_lo, c1_hi) = compute_mu(a_hat, rho0_pos, &[(sigma0, -1)]);
        mu_cols[0][t] = BinaryPoly::from(c1_lo);
        mu_cols[1][t] = BinaryPoly::from(c1_hi);

        // C2: ê · ρ₁ − Σ̂₁
        let e_hat = bp_val(COL_E_HAT, t);
        let sigma1 = bp_val(COL_SIGMA1_HAT, t);
        let (c2_lo, c2_hi) = compute_mu(e_hat, rho1_pos, &[(sigma1, -1)]);
        mu_cols[2][t] = BinaryPoly::from(c2_lo);
        mu_cols[3][t] = BinaryPoly::from(c2_hi);

        // C3: Ŵ_tm15 · ρ_{σ₀} + S₀ − σ̂₀_w
        let w_tm15 = bp_val(COL_W_TM15, t);
        let s0 = bp_val(COL_S0, t);
        let sigma0_w = bp_val(COL_SIGMA0_W_HAT, t);
        let (c3_lo, c3_hi) = compute_mu(w_tm15, rho_s0_pos, &[(s0, 1), (sigma0_w, -1)]);
        mu_cols[4][t] = BinaryPoly::from(c3_lo);
        mu_cols[5][t] = BinaryPoly::from(c3_hi);

        // C4: Ŵ_tm2 · ρ_{σ₁} + S₁ − σ̂₁_w
        let w_tm2 = bp_val(COL_W_TM2, t);
        let s1 = bp_val(COL_S1, t);
        let sigma1_w = bp_val(COL_SIGMA1_W_HAT, t);
        let (c4_lo, c4_hi) = compute_mu(w_tm2, rho_s1_pos, &[(s1, 1), (sigma1_w, -1)]);
        mu_cols[6][t] = BinaryPoly::from(c4_lo);
        mu_cols[7][t] = BinaryPoly::from(c4_hi);
    }

    // Build the trace.
    let mut result: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
        Vec::with_capacity(NO_F2X_NUM_COLS);

    // Original bit-poly columns (0–24).
    for i in 0..crate::NUM_BITPOLY_COLS {
        result.push(base[i].clone());
    }

    // New μ columns (25–32).
    for col in mu_cols {
        result.push(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col,
            BinaryPoly::<32>::from(0u32),
        ));
    }

    // Correction columns (33–38) and corrected int columns when true-ideal is enabled.
    #[cfg(feature = "true-ideal")]
    let corrected_int_cols = {
        let (corr_cols, int_cols) = compute_corrections_and_carries(&base, num_vars);
        for col in corr_cols {
            result.push(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                col,
                BinaryPoly::<32>::from(0u32),
            ));
        }
        Some(int_cols)
    };
    #[cfg(not(feature = "true-ideal"))]
    let corrected_int_cols: Option<[Vec<BinaryPoly<32>>; 3]> = None;

    // Int columns: use corrected carries when true-ideal is enabled,
    // otherwise clone from the base trace.
    if let Some(int_cols) = corrected_int_cols {
        for col in int_cols {
            result.push(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                col,
                BinaryPoly::<32>::from(0u32),
            ));
        }
    } else {
        for i in crate::NUM_BITPOLY_COLS..crate::NUM_COLS {
            result.push(base[i].clone());
        }
    }

    assert_eq!(result.len(), NO_F2X_NUM_COLS);
    result
}

/// Generate only the BinaryPoly columns (indices 0 to NO_F2X_NUM_BITPOLY_COLS-1).
pub fn generate_no_f2x_poly_witness(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let full = generate_no_f2x_witness(num_vars, rng);
    full[..NO_F2X_NUM_BITPOLY_COLS].to_vec()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zinc_uair::{
        collect_scalars::collect_scalars,
        constraint_counter::count_constraints,
        degree_counter::count_max_degree,
    };

    #[test]
    fn bp_correct_column_count() {
        let sig = Sha256UairBpNoF2x::signature();
        assert_eq!(sig.binary_poly_cols, NO_F2X_NUM_BITPOLY_COLS);
        assert_eq!(sig.int_cols, NO_F2X_NUM_INT_COLS);
        assert_eq!(sig.total_cols(), NO_F2X_NUM_COLS);
    }

    #[test]
    fn bp_correct_constraint_count() {
        assert_eq!(
            count_constraints::<Sha256UairBpNoF2x>(),
            NO_F2X_BP_NUM_CONSTRAINTS,
        );
    }

    #[test]
    fn bp_max_degree_is_one() {
        assert_eq!(count_max_degree::<Sha256UairBpNoF2x>(), 1);
    }

    #[test]
    fn qx_correct_column_count() {
        let sig = Sha256UairQxNoF2x::signature();
        assert_eq!(sig.binary_poly_cols, NO_F2X_NUM_BITPOLY_COLS);
        assert_eq!(sig.int_cols, NO_F2X_NUM_INT_COLS);
        assert_eq!(sig.total_cols(), NO_F2X_NUM_COLS);
    }

    #[test]
    fn qx_correct_constraint_count() {
        assert_eq!(
            count_constraints::<Sha256UairQxNoF2x>(),
            NO_F2X_QX_NUM_CONSTRAINTS,
        );
    }

    #[test]
    #[cfg(not(feature = "selector"))]
    fn qx_max_degree_is_one() {
        assert_eq!(count_max_degree::<Sha256UairQxNoF2x>(), 1);
    }

    #[test]
    #[cfg(feature = "selector")]
    fn qx_max_degree_is_two_with_selector() {
        assert_eq!(count_max_degree::<Sha256UairQxNoF2x>(), 2);
    }

    #[test]
    fn no_f2x_witness_has_correct_column_count() {
        let mut rng = rand::rng();
        let trace = generate_no_f2x_witness(7, &mut rng);
        assert_eq!(trace.len(), NO_F2X_NUM_COLS);
    }

    #[test]
    fn qx_scalars_contain_rotation_polynomials() {
        let scalars = collect_scalars::<Sha256UairQxNoF2x>();

        let mut rho0 = DensePolynomial { coeffs: [0i64; 64] };
        rho0.coeffs[30] = 1;
        rho0.coeffs[19] = 1;
        rho0.coeffs[10] = 1;

        let mut rho1 = DensePolynomial { coeffs: [0i64; 64] };
        rho1.coeffs[26] = 1;
        rho1.coeffs[21] = 1;
        rho1.coeffs[7] = 1;

        assert!(scalars.contains(&rho0), "ρ₀ not found in collected scalars");
        assert!(scalars.contains(&rho1), "ρ₁ not found in collected scalars");
    }

    #[test]
    fn mu_computation_sanity() {
        // Simple test: a = 0, so product = 0, Σ₀ = 0.
        // μ should be all zeros.
        let (lo, hi) = compute_mu(0, &[30, 19, 10], &[(0, -1)]);
        assert_eq!(lo, 0);
        assert_eq!(hi, 0);

        // a = 1 (only coefficient 0 is set), ρ₀ = X³⁰ + X¹⁹ + X¹⁰.
        // Product = X³⁰ + X¹⁹ + X¹⁰ (all coefficients 0 or 1).
        // Σ₀(a=1) = ROTR²(1) ⊕ ROTR¹³(1) ⊕ ROTR²²(1)
        //         = X³⁰ ⊕ X¹⁹ ⊕ X¹⁰ (same as product mod 2!)
        // So Q = product - Σ₀ = 0, μ = 0.
        let sigma0_of_1 = (1u32.rotate_right(2)) ^ (1u32.rotate_right(13)) ^ (1u32.rotate_right(22));
        let (lo, hi) = compute_mu(1, &[30, 19, 10], &[(sigma0_of_1, -1)]);
        assert_eq!(lo, 0);
        assert_eq!(hi, 0);
    }
}
