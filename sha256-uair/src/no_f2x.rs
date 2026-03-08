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
//! # Column layout (38 total: 35 bit-poly + 3 integer)
//!
//! Columns 0–26 and int 0–2 are **identical** to the full UAIR.
//! Columns 27–34 are the new μ quotient decompositions:
//!
//! | Index | Name        | Description                              |
//! |-------|-------------|------------------------------------------|
//! | 27    | `mu_c1_lo`  | Σ₀ rotation quotient, low bit            |
//! | 28    | `mu_c1_hi`  | Σ₀ rotation quotient, high bit           |
//! | 29    | `mu_c2_lo`  | Σ₁ rotation quotient, low bit            |
//! | 30    | `mu_c2_hi`  | Σ₁ rotation quotient, high bit           |
//! | 31    | `mu_c3_lo`  | σ₀ rotation+shift quotient, low bit      |
//! | 32    | `mu_c3_hi`  | σ₀ rotation+shift quotient, high bit     |
//! | 33    | `mu_c4_lo`  | σ₁ rotation+shift quotient, low bit      |
//! | 34    | `mu_c4_hi`  | σ₁ rotation+shift quotient, high bit     |
//!
//! # Constraint split
//!
//! **Bp UAIR** (`Sha256UairBpNoF2x`, 12 constraints):
//!   C5–C6 (shift decomposition, zero ideal) +
//!   C7–C16 (10 linking constraints, zero ideal).
//!
//! **Qx UAIR** (`Sha256UairQxNoF2x`, 7 constraints):
//!   C1–C4 (rotation / shift, cyclotomic ideal over Z\[X\], with μ correction) +
//!   C7–C9 (carry propagation, trivial ideal).

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
    // Original column indices (unchanged 0–26)
    COL_A_HAT, COL_E_HAT, COL_W_HAT,
    COL_SIGMA0_HAT, COL_SIGMA1_HAT, COL_MAJ_HAT,
    COL_CH_EF_HAT, COL_CH_NEG_EG_HAT,
    COL_SIGMA0_W_HAT, COL_SIGMA1_W_HAT,
    COL_S0, COL_S1, COL_R0, COL_R1,
    COL_D_HAT, COL_H_HAT,
    COL_W_TM2, COL_W_TM7, COL_W_TM15, COL_W_TM16,
    COL_K_HAT,
    COL_A_TM1, COL_A_TM2, COL_E_TM1, COL_E_TM2,
    COL_SEL_ROUND, COL_SEL_SCHED,
    COL_INT_MU_A, COL_INT_MU_E, COL_INT_MU_W,
};

// ─── New column indices (27–34) ─────────────────────────────────────────────

/// Σ₀ rotation quotient, low bit of binary decomposition.
pub const COL_MU_C1_LO: usize = 27;
/// Σ₀ rotation quotient, high bit of binary decomposition.
pub const COL_MU_C1_HI: usize = 28;
/// Σ₁ rotation quotient, low bit of binary decomposition.
pub const COL_MU_C2_LO: usize = 29;
/// Σ₁ rotation quotient, high bit of binary decomposition.
pub const COL_MU_C2_HI: usize = 30;
/// σ₀ rotation+shift quotient, low bit of binary decomposition.
pub const COL_MU_C3_LO: usize = 31;
/// σ₀ rotation+shift quotient, high bit of binary decomposition.
pub const COL_MU_C3_HI: usize = 32;
/// σ₁ rotation+shift quotient, low bit of binary decomposition.
pub const COL_MU_C4_LO: usize = 33;
/// σ₁ rotation+shift quotient, high bit of binary decomposition.
pub const COL_MU_C4_HI: usize = 34;

// ─── Counts ─────────────────────────────────────────────────────────────────

/// Number of bit-polynomial columns (35 = original 27 + 8 μ).
pub const NO_F2X_NUM_BITPOLY_COLS: usize = 35;

/// Number of integer columns (unchanged: 3).
pub const NO_F2X_NUM_INT_COLS: usize = 3;

/// Total number of trace columns (35 + 3 = 38).
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
            public_columns: vec![
                COL_W_HAT, COL_K_HAT,
                COL_S0, COL_S1, COL_R0, COL_R1,
                COL_W_TM2, COL_W_TM7, COL_W_TM15, COL_W_TM16,
                COL_SEL_ROUND, COL_SEL_SCHED,
            ],
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
            public_columns: vec![
                COL_W_HAT, COL_K_HAT,
                COL_S0, COL_S1, COL_R0, COL_R1,
                COL_W_TM2, COL_W_TM7, COL_W_TM15, COL_W_TM16,
                COL_SEL_ROUND, COL_SEL_SCHED,
            ],
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

        // ── Constraint C7: a-update carry propagation (trivial ideal) ───
        b.assert_in_ideal(
            bp_down[NF2X_DOWN_QX_A].clone()
                - &bp_up[COL_H_HAT]
                - &bp_up[COL_SIGMA1_HAT]
                - &bp_up[COL_CH_EF_HAT]
                - &bp_up[COL_CH_NEG_EG_HAT]
                - &bp_up[COL_K_HAT]
                - &bp_up[COL_W_HAT]
                - &bp_up[COL_SIGMA0_HAT]
                - &bp_up[COL_MAJ_HAT]
                + &(int_up[COL_INT_MU_A].clone() * &x32),
            &trivial,
        );

        // ── Constraint C8: e-update carry propagation (trivial ideal) ───
        b.assert_in_ideal(
            bp_down[NF2X_DOWN_QX_E].clone()
                - &bp_up[COL_D_HAT]
                - &bp_up[COL_H_HAT]
                - &bp_up[COL_SIGMA1_HAT]
                - &bp_up[COL_CH_EF_HAT]
                - &bp_up[COL_CH_NEG_EG_HAT]
                - &bp_up[COL_K_HAT]
                - &bp_up[COL_W_HAT]
                + &(int_up[COL_INT_MU_E].clone() * &x32),
            &trivial,
        );

        // ── Constraint C9: Message schedule recurrence (trivial ideal) ──
        b.assert_in_ideal(
            bp_up[COL_W_HAT].clone()
                - &bp_up[COL_W_TM16]
                - &bp_up[COL_SIGMA0_W_HAT]
                - &bp_up[COL_W_TM7]
                - &bp_up[COL_SIGMA1_W_HAT]
                + &(int_up[COL_INT_MU_W].clone() * &x32),
            &trivial,
        );
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

/// Generate the full 38-column witness for the no-F₂\[X\] variant.
///
/// Delegates to the base `Sha256UairBp` witness generator for columns 0–29,
/// then computes the 8 μ quotient decomposition columns (27–34) from
/// the existing trace data.
///
/// Note: columns 27–29 in the base trace are the integer carry columns
/// μ_a, μ_e, μ_W (encoded as BinaryPoly).  After extending, the base
/// columns 27–29 shift to positions 30–34.  Actually no—we INSERT the
/// 8 new columns at positions 27–34 and move the int columns to 35–37.
///
/// Wait—that would change the int column indices.  Instead, we keep
/// the original 30 columns (0–29) intact and **append** the 8 new μ
/// columns at positions 30–37.
///
/// Hmm, but the spec says columns 27–34 for the μ and the int columns
/// stay at their original indices (27–29 in the int sub-slice).
/// The total layout is: `binary_poly[0..35]` || `int[0..3]`.
/// In the flattened trace vector: `bp[0..35]` then `int[0..3]` = 38 entries.
///
/// The existing trace has `bp[0..27]` then `int[0..3]` encoded as
/// BinaryPoly = 30 entries.  For no-f2x we produce:
///   bp[0..27] (original) + bp[27..35] (new μ cols) + int[0..3] = 38 entries.
pub fn generate_no_f2x_witness(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    // Generate the base 30-column trace.
    let base = <crate::Sha256UairBp as GenerateWitness<BinaryPoly<32>>>::generate_witness(
        num_vars, rng,
    );
    assert_eq!(base.len(), crate::NUM_COLS); // 30

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

    // Build the 38-column trace:
    //   positions 0–26: original bit-poly columns
    //   positions 27–34: new μ columns
    //   positions 35–37: original int columns (were at 27–29)
    let mut result: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
        Vec::with_capacity(NO_F2X_NUM_COLS);

    // Original bit-poly columns (0–26).
    for i in 0..crate::NUM_BITPOLY_COLS {
        result.push(base[i].clone());
    }

    // New μ columns (27–34).
    for col in mu_cols {
        result.push(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col,
            BinaryPoly::<32>::from(0u32),
        ));
    }

    // Original int columns (moved to 35–37).
    for i in crate::NUM_BITPOLY_COLS..crate::NUM_COLS {
        result.push(base[i].clone());
    }

    assert_eq!(result.len(), NO_F2X_NUM_COLS);
    result
}

/// Generate only the 35 BinaryPoly columns (indices 0–34).
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
    fn qx_max_degree_is_one() {
        assert_eq!(count_max_degree::<Sha256UairQxNoF2x>(), 1);
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
