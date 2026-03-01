//! SHA-256 UAIR⁺ (Universal Algebraic Intermediate Representation).
//!
//! This crate defines the SHA-256 arithmetization as a UAIR⁺ with two trace
//! components following the paper's specification:
//!
//! - **Q\[X\]-trace** (23 columns): 10 bit-polynomial columns in
//!   {0,1}^{<32}\[X\], 4 F₂\[X\] columns, 7 auxiliary lookback columns,
//!   and 2 selector columns.
//! - **Integer columns** (3 columns): carry values μ_a, μ_e, μ_W.
//!
//! Total: 26 witness columns.
//!
//! # Column layout
//!
//! ## Bit-polynomial columns — {0,1}^{<32}\[X\] (indices 0–9)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 0     | `a_hat`          | Working variable a (bit-poly representative)    |
//! | 1     | `e_hat`          | Working variable e (bit-poly representative)    |
//! | 2     | `W_hat`          | Message schedule word W_t                       |
//! | 3     | `Sigma0_hat`     | Σ₀(a) = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a)     |
//! | 4     | `Sigma1_hat`     | Σ₁(e) = ROTR⁶(e) ⊕ ROTR¹¹(e) ⊕ ROTR²⁵(e)     |
//! | 5     | `Maj_hat`        | Maj(a,b,c) = (a∧b) ⊕ (a∧c) ⊕ (b∧c)           |
//! | 6     | `ch_ef_hat`      | e ∧ f (part of Ch decomposition)                |
//! | 7     | `ch_neg_eg_hat`  | (¬e) ∧ g (part of Ch decomposition)             |
//! | 8     | `sigma0_w_hat`   | σ₀(W_{t−15}) for message schedule               |
//! | 9     | `sigma1_w_hat`   | σ₁(W_{t−2}) for message schedule                |
//!
//! ## F₂\[X\] columns — kept as {0,1}^{<32}\[X\] (indices 10–13)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 10    | `S0`             | Shift quotient for σ₀ (deg < 29)                |
//! | 11    | `S1`             | Shift quotient for σ₁ (deg < 22)                |
//! | 12    | `R0`             | Shift remainder for σ₀ (deg < 3)                |
//! | 13    | `R1`             | Shift remainder for σ₁ (deg < 10)               |
//!
//! ## Auxiliary lookback columns — {0,1}^{<32}\[X\] (indices 14–20)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 14    | `d_hat`          | d_t = a_{t−3} (initial H values for t < 3)     |
//! | 15    | `h_hat`          | h_t = e_{t−3} (initial H values for t < 3)     |
//! | 16    | `W_tm2`          | W[t−2] (0 for t < 2)                            |
//! | 17    | `W_tm7`          | W[t−7] (0 for t < 7)                            |
//! | 18    | `W_tm15`         | W[t−15] (0 for t < 15)                          |
//! | 19    | `W_tm16`         | W[t−16] (0 for t < 16)                          |
//! | 20    | `K_hat`          | Round constant K_t (0 for t ≥ 64)               |
//!
//! ## Selector columns — {0,1}^{<32}\[X\] (indices 21–22)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 21    | `sel_round`      | 1 for t ∈ [0, 63], 0 otherwise                  |
//! | 22    | `sel_sched`      | 1 for t ∈ [16, 63], 0 otherwise                 |
//!
//! ## Integer columns — Z (indices 23–25 in flattened trace)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 23    | `mu_a`           | Carry for a-update (∈ {0,…,6})                  |
//! | 24    | `mu_e`           | Carry for e-update (∈ {0,…,5})                  |
//! | 25    | `mu_W`           | Carry for W schedule (∈ {0,…,3})                |
//!
//! The auxiliary columns store shifted copies of committed columns so
//! that cross-row references (lookbacks) can be expressed as same-row
//! constraints. Forward shifts in the Bp UAIR provide **linking
//! constraints** that verify each auxiliary column equals the correct
//! shifted source.
//!
//! # Constraints
//!
//! ## F₂\[X\] constraints (rotation, shift & linking)
//!
//! 1.  **Σ₀ rotation**: `â · ρ₀ − Σ̂₀ ∈ (X³² − 1)`
//! 2.  **Σ₁ rotation**: `ê · ρ₁ − Σ̂₁ ∈ (X³² − 1)`
//! 3.  **σ₀ rotation+shift**: `Ŵ_tm15·ρ_{σ₀} + S₀ − σ̂₀_w ∈ (X³² − 1)`
//! 4.  **σ₁ rotation+shift**: `Ŵ_tm2·ρ_{σ₁} + S₁ − σ̂₁_w ∈ (X³² − 1)`
//! 5.  **σ₀ shift decomp**: `Ŵ_tm15 = R₀ + X³·S₀`
//! 6.  **σ₁ shift decomp**: `Ŵ_tm2 = R₁ + X¹⁰·S₁`
//! 7.  **d-link**: `d̂[t+3] = â[t]`  (shift-by-3 linking)
//! 8.  **h-link**: `ĥ[t+3] = ê[t]`  (shift-by-3 linking)
//! 9.  **W_tm2-link**: `Ŵ_tm2[t+2] = Ŵ[t]`  (shift-by-2 linking)
//! 10. **W_tm7-link**: `Ŵ_tm7[t+7] = Ŵ[t]`  (shift-by-7 linking)
//! 11. **W_tm15-link**: `Ŵ_tm15[t+15] = Ŵ[t]`  (shift-by-15 linking)
//! 12. **W_tm16-link**: `Ŵ_tm16[t+16] = Ŵ[t]`  (shift-by-16 linking)
//!
//! ## Q\[X\] constraints (carry propagation, selector-gated)
//!
//! 7. **a-update**: `sel_round · (â[t+1] − ĥ − Σ̂₁ − Ĉh − K̂ − Ŵ − Σ̂₀ − M̂aj + μ_a·X^w) ∈ (X−2)`
//! 8. **e-update**: `sel_round · (ê[t+1] − d̂ − ĥ − Σ̂₁ − Ĉh − K̂ − Ŵ + μ_e·X^w) ∈ (X−2)`
//! 9. **W schedule**: `sel_sched · (Ŵ − Ŵ_tm16 − σ̂₀_w − Ŵ_tm7 − σ̂₁_w + μ_W·X^w) ∈ (X−2)`

#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows

pub mod constants;
pub mod witness;

use crypto_primitives::PrimeField;
use num_traits::Zero;
use zinc_poly::EvaluatablePolynomial;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_poly::univariate::ideal::DegreeOneIdeal;
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};
use zinc_utils::from_ref::FromRef;

// ─── Trace conversion ───────────────────────────────────────────────────────

/// Convert a `BinaryPoly<32>` trace to `DensePolynomial<i64, 64>` for Q\[X\]
/// constraints.
///
/// Each `BinaryPoly<32>` element (32 binary coefficients) is mapped to a
/// `DensePolynomial<i64, 64>` with the same 0/1 values in the first 32
/// positions and zeros in positions 32–63. The polynomials evaluate to
/// the same integer at X = 2.
pub fn convert_trace_to_qx(
    trace: &[DenseMultilinearExtension<BinaryPoly<32>>],
) -> Vec<DenseMultilinearExtension<DensePolynomial<i64, 64>>> {
    let zero_qx = DensePolynomial { coeffs: [0i64; 64] };
    trace
        .iter()
        .map(|col| {
            let evaluations: Vec<DensePolynomial<i64, 64>> = col
                .evaluations
                .iter()
                .map(|bp| {
                    let mut val: u64 = 0;
                    for (i, coeff) in bp.iter().enumerate() {
                        if coeff.into_inner() {
                            val |= 1u64 << i;
                        }
                    }
                    let mut coeffs = [0i64; 64];
                    for i in 0..32 {
                        coeffs[i] = ((val >> i) & 1) as i64;
                    }
                    DensePolynomial { coeffs }
                })
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(
                col.num_vars,
                evaluations,
                zero_qx,
            )
        })
        .collect()
}

// ─── Column indices ──────────────────────────────────────────────────────────

/// Total number of trace columns (23 bit-poly + 3 integer).
pub const NUM_COLS: usize = 26;

/// Number of bit-polynomial columns ({0,1}^{<32}[X]).
/// Includes the 10 Q[X] bit-poly columns, 4 F₂[X] columns, 7 auxiliary
/// lookback columns, and 2 selector columns.
pub const NUM_BITPOLY_COLS: usize = 23;

/// Number of integer columns (Z).
pub const NUM_INT_COLS: usize = 3;

// ── Bit-polynomial columns (indices 0–9) ────────────────────────────────────

/// Working variable *a* (bit-poly representative â_t).
pub const COL_A_HAT: usize = 0;
/// Working variable *e* (bit-poly representative ê_t).
pub const COL_E_HAT: usize = 1;
/// Message schedule word Ŵ_t.
pub const COL_W_HAT: usize = 2;
/// Σ₀(a) = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a).
pub const COL_SIGMA0_HAT: usize = 3;
/// Σ₁(e) = ROTR⁶(e) ⊕ ROTR¹¹(e) ⊕ ROTR²⁵(e).
pub const COL_SIGMA1_HAT: usize = 4;
/// Maj(a,b,c).
pub const COL_MAJ_HAT: usize = 5;
/// e ∧ f (first term of Ch).
pub const COL_CH_EF_HAT: usize = 6;
/// (¬e) ∧ g (second term of Ch).
pub const COL_CH_NEG_EG_HAT: usize = 7;
/// σ₀(W_{t−15}) for the message schedule.
pub const COL_SIGMA0_W_HAT: usize = 8;
/// σ₁(W_{t−2}) for the message schedule.
pub const COL_SIGMA1_W_HAT: usize = 9;

// ── F₂[X] columns kept as {0,1}^{<32}[X] (indices 10–13) ───────────────────

/// Shift quotient for σ₀ (deg < w−3 = 29).
pub const COL_S0: usize = 10;
/// Shift quotient for σ₁ (deg < w−10 = 22).
pub const COL_S1: usize = 11;
/// Shift remainder for σ₀ (= W_{t−15} mod X³, deg < 3).
pub const COL_R0: usize = 12;
/// Shift remainder for σ₁ (= W_{t−2} mod X¹⁰, deg < 10).
pub const COL_R1: usize = 13;

// ── Auxiliary lookback columns (indices 14–20) ──────────────────────────────

/// d_t = a_{t−3} (inlined register d via shift-register identity).
pub const COL_D_HAT: usize = 14;
/// h_t = e_{t−3} (inlined register h via shift-register identity).
pub const COL_H_HAT: usize = 15;
/// W[t−2] for the σ₁ constraint and message schedule.
pub const COL_W_TM2: usize = 16;
/// W[t−7] for the message schedule recurrence.
pub const COL_W_TM7: usize = 17;
/// W[t−15] for the σ₀ constraint and message schedule.
pub const COL_W_TM15: usize = 18;
/// W[t−16] for the message schedule recurrence.
pub const COL_W_TM16: usize = 19;
/// Round constant K_t as a bit-polynomial.
pub const COL_K_HAT: usize = 20;

// ── Selector columns (indices 21–22) ────────────────────────────────────────

/// Round selector: 1 for t ∈ [0, 63], 0 otherwise.
/// Gates carry propagation constraints C7/C8.
pub const COL_SEL_ROUND: usize = 21;
/// Schedule selector: 1 for t ∈ [16, 63], 0 otherwise.
/// Gates the message schedule recurrence C9.
pub const COL_SEL_SCHED: usize = 22;

// ── Integer columns (indices 0–2 within the int sub-slice) ──────────────────
// NOTE: These are accessed via `up.int[COL_INT_MU_*]`, not `up.binary_poly[..]`.

/// Carry for the *a* state update (∈ {0,…,6}).
pub const COL_INT_MU_A: usize = 0;
/// Carry for the *e* state update (∈ {0,…,5}).
pub const COL_INT_MU_E: usize = 1;
/// Carry for the W_t message schedule update (∈ {0,…,3}).
pub const COL_INT_MU_W: usize = 2;

// ─── Number of constraints ──────────────────────────────────────────────────

/// Number of F₂[X] polynomial constraints emitted by the Bp UAIR.
/// C1–C6 (rotation + shift) + C7–C12 (6 linking constraints).
pub const NUM_CONSTRAINTS: usize = 12;

// ─── Ideal types ────────────────────────────────────────────────────────────

/// The cyclotomic ideal (X³² − 1) in F₂\[X\].
///
/// An element p ∈ F₂\[X\] belongs to this ideal iff it is divisible by
/// X³² − 1 = X³² + 1 (over F₂). For BinaryPoly<32> (degree ≤ 31),
/// the only member is 0, but the ideal is meaningful for products that
/// have degree > 31 before reduction, such as `a · ρ₀` (degree ≤ 61).
///
/// In the PIOP pipeline the constraint
/// `expr ∈ (X³² − 1)` is verified by checking that `expr(α)` is
/// divisible by `α³² − 1` after projection to a prime field at
/// evaluation point α.
#[derive(Clone, Copy, Debug)]
pub struct CyclotomicIdeal;

impl Ideal for CyclotomicIdeal {}

impl FromRef<CyclotomicIdeal> for CyclotomicIdeal {
    #[inline(always)]
    fn from_ref(ideal: &CyclotomicIdeal) -> Self {
        *ideal
    }
}

impl IdealCheck<BinaryPoly<32>> for CyclotomicIdeal {
    /// For BinaryPoly<32> values (degree ≤ 31) only 0 is in (X³² − 1).
    fn contains(&self, value: &BinaryPoly<32>) -> bool {
        num_traits::Zero::is_zero(value)
    }
}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for CyclotomicIdeal {
    /// A polynomial g(X) ∈ F[X] belongs to the ideal (X³² − 1) iff
    /// g(X) mod (X³² − 1) = 0.
    ///
    /// Since X³² ≡ 1 mod (X³² − 1), the reduction is:
    ///   g mod (X³² − 1) = Σⱼ₌₀³¹ (Σₖ g_{j+32k}) Xʲ
    ///
    /// The polynomial is in the ideal iff all 32 reduced coefficients are zero.
    fn contains(&self, value: &DynamicPolynomialF<F>) -> bool {
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
}

/// Converts an `IdealOrZero<CyclotomicIdeal>` to a field-level ideal check
/// suitable for the IdealCheck verifier. Returns `CyclotomicIdeal` for
/// `Ideal` variants, and for `Zero` variants returns a `CyclotomicIdeal`
/// as well (the zero polynomial is in every ideal, so this is sound —
/// the batched ideal check only calls `contains` on the actual constraint
/// values, and for `assert_zero` constraints those values are already zero).
pub fn cyclotomic_ideal_over_f(
    ideal: &IdealOrZero<CyclotomicIdeal>,
) -> IdealOrZero<CyclotomicIdeal> {
    ideal.clone()
}

// ─── SHA-256 UAIR ───────────────────────────────────────────────────────────

/// Compatibility alias — refers to `Sha256UairBp` for code that was
/// written before the trait split.
pub type Sha256Uair = Sha256UairBp;

/// The SHA-256 UAIR over `BinaryPoly<32>` (F₂[X] rotation/shift constraints).
///
/// Describes a trace with [`NUM_BITPOLY_COLS`] bit-polynomial columns and
/// [`NUM_INT_COLS`] integer columns (one row per SHA-256 round, 65 rows
/// total → `num_vars = 7`). Emits [`NUM_CONSTRAINTS`] F₂[X] constraints:
/// C1–C6 (rotation + shift) and C7–C12 (6 linking constraints).
///
/// The linking constraints use forward shifts (of 2, 3, 7, 15, 16 steps)
/// to verify that each auxiliary lookback column equals the correct
/// shifted source column.
pub struct Sha256UairBp;

// Down-row indices for the Bp UAIR's shifted columns.
// The shifts are: d(3), h(3), W_tm2(2), W_tm7(7), W_tm15(15), W_tm16(16).
// All source columns are binary_poly, so they map to down.binary_poly[0..6].
const DOWN_BP_D: usize = 0;
const DOWN_BP_H: usize = 1;
const DOWN_BP_W_TM2: usize = 2;
const DOWN_BP_W_TM7: usize = 3;
const DOWN_BP_W_TM15: usize = 4;
const DOWN_BP_W_TM16: usize = 5;

impl Uair for Sha256UairBp {
    type Ideal = CyclotomicIdeal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NUM_INT_COLS,
            shifts: vec![
                // Linking shifts for auxiliary lookback columns.
                zinc_uair::ShiftSpec { source_col: COL_D_HAT,  shift_amount: 3 },
                zinc_uair::ShiftSpec { source_col: COL_H_HAT,  shift_amount: 3 },
                zinc_uair::ShiftSpec { source_col: COL_W_TM2,  shift_amount: 2 },
                zinc_uair::ShiftSpec { source_col: COL_W_TM7,  shift_amount: 7 },
                zinc_uair::ShiftSpec { source_col: COL_W_TM15, shift_amount: 15 },
                zinc_uair::ShiftSpec { source_col: COL_W_TM16, shift_amount: 16 },
            ],
            public_columns: vec![COL_W_HAT, COL_K_HAT],
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
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &BinaryPoly<32>) -> Option<B::Expr>,
        IFromR: Fn(&CyclotomicIdeal) -> B::Ideal,
    {
        let up = up.binary_poly;
        let bp_down = down.binary_poly;
        let cyclotomic = ideal_from_ref(&CyclotomicIdeal);

        // ── Rotation polynomials ────────────────────────────────────────

        // ρ₀ = X³⁰ + X¹⁹ + X¹⁰  →  encodes ROTR(2,13,22) for Σ₀
        let rho0 = from_ref(&BinaryPoly::<32>::from(
            (1u32 << 30) | (1u32 << 19) | (1u32 << 10),
        ));

        // ρ₁ = X²⁶ + X²¹ + X⁷  →  encodes ROTR(6,11,25) for Σ₁
        let rho1 = from_ref(&BinaryPoly::<32>::from(
            (1u32 << 26) | (1u32 << 21) | (1u32 << 7),
        ));

        // ρ_{σ₀} = X²⁵ + X¹⁴  →  encodes ROTR(7,18) for σ₀
        let rho_sigma0 = from_ref(&BinaryPoly::<32>::from(
            (1u32 << 25) | (1u32 << 14),
        ));

        // ρ_{σ₁} = X¹⁵ + X¹³  →  encodes ROTR(17,19) for σ₁
        let rho_sigma1 = from_ref(&BinaryPoly::<32>::from(
            (1u32 << 15) | (1u32 << 13),
        ));

        // ── Constraint 1: Σ₀ rotation ──────────────────────────────────
        //   a_hat · ρ₀ − Sigma0_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[COL_A_HAT].clone() * &rho0 - &up[COL_SIGMA0_HAT],
            &cyclotomic,
        );

        // ── Constraint 2: Σ₁ rotation ──────────────────────────────────
        //   e_hat · ρ₁ − Sigma1_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[COL_E_HAT].clone() * &rho1 - &up[COL_SIGMA1_HAT],
            &cyclotomic,
        );

        // ── Constraint 3: σ₀ rotation + shift ──────────────────────────
        //   W_tm15 · ρ_{σ₀} + S₀ − sigma0_w_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[COL_W_TM15].clone() * &rho_sigma0
                + &up[COL_S0]
                - &up[COL_SIGMA0_W_HAT],
            &cyclotomic,
        );

        // ── Constraint 4: σ₁ rotation + shift ──────────────────────────
        //   W_tm2 · ρ_{σ₁} + S₁ − sigma1_w_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[COL_W_TM2].clone() * &rho_sigma1
                + &up[COL_S1]
                - &up[COL_SIGMA1_W_HAT],
            &cyclotomic,
        );

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
        //   d[t+3] = a[t]
        b.assert_zero(
            bp_down[DOWN_BP_D].clone() - &up[COL_A_HAT],
        );

        // ── Constraint 8: h-link (shift-by-3) ──────────────────────────
        //   h[t+3] = e[t]
        b.assert_zero(
            bp_down[DOWN_BP_H].clone() - &up[COL_E_HAT],
        );

        // ── Constraint 9: W_tm2-link (shift-by-2) ──────────────────────
        //   W_tm2[t+2] = W[t]
        b.assert_zero(
            bp_down[DOWN_BP_W_TM2].clone() - &up[COL_W_HAT],
        );

        // ── Constraint 10: W_tm7-link (shift-by-7) ─────────────────────
        //   W_tm7[t+7] = W[t]
        b.assert_zero(
            bp_down[DOWN_BP_W_TM7].clone() - &up[COL_W_HAT],
        );

        // ── Constraint 11: W_tm15-link (shift-by-15) ───────────────────
        //   W_tm15[t+15] = W[t]
        b.assert_zero(
            bp_down[DOWN_BP_W_TM15].clone() - &up[COL_W_HAT],
        );

        // ── Constraint 12: W_tm16-link (shift-by-16) ───────────────────
        //   W_tm16[t+16] = W[t]
        b.assert_zero(
            bp_down[DOWN_BP_W_TM16].clone() - &up[COL_W_HAT],
        );
    }
}

// ─── Number of Q[X] (integer polynomial) constraints ────────────────────────

/// Number of Q[X] constraints: 3 carry propagation checks (selector-gated).
///
/// - C7: a-update carry via (X−2) ideal, gated by sel_round.
/// - C8: e-update carry via (X−2) ideal, gated by sel_round.
/// - C9: W schedule recurrence via (X−2) ideal, gated by sel_sched.
///
/// All cross-row references are resolved via auxiliary lookback columns
/// (d_hat, h_hat, W_tm2/7/15/16, K_hat) verified by the Bp UAIR's
/// linking constraints.
pub const NUM_QX_CONSTRAINTS: usize = 3;

// ─── Q[X] ideal type enum ──────────────────────────────────────────────────

/// Ideal type for the Q[X] SHA-256 UAIR.
///
/// Constraints use:
/// - `DegreeOne(2)`: evaluation at X = 2 gives zero (carry propagation)
///
/// BitPoly membership (binary coefficient checks) is now enforced by
/// lookups rather than ideal checks.
#[derive(Clone, Debug)]
pub enum Sha256QxIdeal {
    DegreeOne(DegreeOneIdeal<i64>),
}

impl Ideal for Sha256QxIdeal {}

impl FromRef<Sha256QxIdeal> for Sha256QxIdeal {
    fn from_ref(ideal: &Sha256QxIdeal) -> Self {
        ideal.clone()
    }
}

impl IdealCheck<DensePolynomial<i64, 64>> for Sha256QxIdeal {
    fn contains(&self, value: &DensePolynomial<i64, 64>) -> bool {
        match self {
            Sha256QxIdeal::DegreeOne(_ideal) => {
                // Evaluate at X = 2: f(2) = Σ c_i * 2^i
                let mut eval: i64 = 0;
                for (i, &c) in value.coeffs.iter().enumerate() {
                    eval = eval.wrapping_add(c.wrapping_mul(1i64.wrapping_shl(i as u32)));
                }
                eval == 0
            }
        }
    }
}

// ─── Q[X] ideal lifted to F_p ──────────────────────────────────────────────

/// The Q\[X\] ideal lifted to a prime field for IdealCheck verification.
///
/// This enum maps `Sha256QxIdeal` variants to their field-level equivalents:
/// - `DegreeOne(root)`: evaluation at `root` (= 2) in F_p gives zero.
///   This IS a real ideal ((X−2) ⊂ F_p\[X\]) and lifts correctly from Z\[X\].
/// - `Zero`: exact zero polynomial.
#[derive(Clone, Debug)]
pub enum Sha256QxIdealOverF<F: PrimeField> {
    /// Carry propagation: evaluate at root and check = 0.
    DegreeOne(F),
    /// Exact zero.
    Zero,
}

impl<F: PrimeField> Ideal for Sha256QxIdealOverF<F> {}

impl<F: PrimeField> FromRef<Sha256QxIdealOverF<F>> for Sha256QxIdealOverF<F> {
    fn from_ref(ideal: &Sha256QxIdealOverF<F>) -> Self {
        ideal.clone()
    }
}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for Sha256QxIdealOverF<F> {
    fn contains(&self, value: &DynamicPolynomialF<F>) -> bool {
        match self {
            Sha256QxIdealOverF::DegreeOne(root) => {
                if value.coeffs.is_empty() {
                    return true;
                }
                value
                    .evaluate_at_point(root)
                    .map_or(true, |v| F::is_zero(&v))
            }
            Sha256QxIdealOverF::Zero => value.is_zero(),
        }
    }
}

// ─── Q[X] SHA-256 UAIR ─────────────────────────────────────────────────────

/// The SHA-256 UAIR over `DensePolynomial<i64, 64>` (Z[X] with degree < 64).
///
/// Defines the integer-polynomial carry propagation constraints that cannot
/// be expressed in F₂[X]. All cross-row references are resolved via the
/// auxiliary lookback columns (d_hat, h_hat, W_tm2/7/15/16, K_hat)
/// verified by the Bp UAIR's linking constraints.
///
/// The constraints are gated by selector columns to handle boundary rows:
///
/// - C7: `sel_round · (â[t+1] − ĥ − Σ̂₁ − Ĉh − K̂ − Ŵ − Σ̂₀ − M̂aj + μ_a·X^w) ∈ (X−2)`
/// - C8: `sel_round · (ê[t+1] − d̂ − ĥ − Σ̂₁ − Ĉh − K̂ − Ŵ + μ_e·X^w) ∈ (X−2)`
/// - C9: `sel_sched · (Ŵ − Ŵ_tm16 − σ̂₀_w − Ŵ_tm7 − σ̂₁_w + μ_W·X^w) ∈ (X−2)`
pub struct Sha256UairQx;

// Down-row indices for the Qx UAIR.
// Shifts: â(1), ê(1). Both are binary_poly → down.binary_poly[0..2].
const DOWN_QX_A: usize = 0;
const DOWN_QX_E: usize = 1;

impl Uair for Sha256UairQx {
    type Ideal = Sha256QxIdeal;
    type Scalar = DensePolynomial<i64, 64>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NUM_INT_COLS,
            // Forward shifts for â[t+1] and ê[t+1], used in C7/C8.
            shifts: vec![
                zinc_uair::ShiftSpec { source_col: COL_A_HAT, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_E_HAT, shift_amount: 1 },
            ],
            public_columns: vec![COL_W_HAT, COL_K_HAT],
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
        IFromR: Fn(&Sha256QxIdeal) -> B::Ideal,
    {
        let bp_up = up.binary_poly;
        let int_up = up.int;
        let bp_down = down.binary_poly;
        let carry_ideal = ideal_from_ref(&Sha256QxIdeal::DegreeOne(DegreeOneIdeal::new(2_i64)));

        // ── Constant polynomials ────────────────────────────────────────

        // X^w (= X³²) as a polynomial: coefficient 1 at index 32
        let x32: DensePolynomial<i64, 64> = {
            let mut coeffs = [0i64; 64];
            coeffs[32] = 1;
            DensePolynomial { coeffs }
        };
        let x32_expr = from_ref(&x32);

        // ── Constraint 7: a-update carry propagation (selector-gated) ───
        //
        //   sel_round · (â[t+1] − h_hat − Σ̂₁ − ch_ef − ch_neg_eg
        //                − K_hat − Ŵ − Σ̂₀ − Maj + μ_a·X^w) ∈ (X−2)
        //
        // h_hat stores e[t−3] (= h_t), K_hat stores K_t as a
        // bit-polynomial. The selector is 1 for t∈[0,63], ensuring
        // the constraint is only active during valid rounds.
        let c7_inner =
            bp_down[DOWN_QX_A].clone()
                - &bp_up[COL_H_HAT]
                - &bp_up[COL_SIGMA1_HAT]
                - &bp_up[COL_CH_EF_HAT]
                - &bp_up[COL_CH_NEG_EG_HAT]
                - &bp_up[COL_K_HAT]
                - &bp_up[COL_W_HAT]
                - &bp_up[COL_SIGMA0_HAT]
                - &bp_up[COL_MAJ_HAT]
                + &(int_up[COL_INT_MU_A].clone() * &x32_expr);
        b.assert_in_ideal(
            bp_up[COL_SEL_ROUND].clone() * &c7_inner,
            &carry_ideal,
        );

        // ── Constraint 8: e-update carry propagation (selector-gated) ───
        //
        //   sel_round · (ê[t+1] − d_hat − h_hat − Σ̂₁ − ch_ef
        //                − ch_neg_eg − K_hat − Ŵ + μ_e·X^w) ∈ (X−2)
        let c8_inner =
            bp_down[DOWN_QX_E].clone()
                - &bp_up[COL_D_HAT]
                - &bp_up[COL_H_HAT]
                - &bp_up[COL_SIGMA1_HAT]
                - &bp_up[COL_CH_EF_HAT]
                - &bp_up[COL_CH_NEG_EG_HAT]
                - &bp_up[COL_K_HAT]
                - &bp_up[COL_W_HAT]
                + &(int_up[COL_INT_MU_E].clone() * &x32_expr);
        b.assert_in_ideal(
            bp_up[COL_SEL_ROUND].clone() * &c8_inner,
            &carry_ideal,
        );

        // ── Constraint 9: Message schedule recurrence (selector-gated) ──
        //
        //   sel_sched · (Ŵ − W_tm16 − σ̂₀_w − W_tm7 − σ̂₁_w
        //                + μ_W·X^w) ∈ (X−2)
        let c9_inner =
            bp_up[COL_W_HAT].clone()
                - &bp_up[COL_W_TM16]
                - &bp_up[COL_SIGMA0_W_HAT]
                - &bp_up[COL_W_TM7]
                - &bp_up[COL_SIGMA1_W_HAT]
                + &(int_up[COL_INT_MU_W].clone() * &x32_expr);
        b.assert_in_ideal(
            bp_up[COL_SEL_SCHED].clone() * &c9_inner,
            &carry_ideal,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use zinc_uair::{
        collect_scalars::collect_scalars, constraint_counter::count_constraints,
        degree_counter::count_max_degree,
    };

    #[test]
    fn correct_number_of_columns() {
        let sig = Sha256UairBp::signature();
        assert_eq!(sig.binary_poly_cols, NUM_BITPOLY_COLS);
        assert_eq!(sig.int_cols, NUM_INT_COLS);
        assert_eq!(sig.total_cols(), NUM_COLS);
    }

    #[test]
    fn correct_number_of_constraints() {
        assert_eq!(
            count_constraints::<Sha256UairBp>(),
            NUM_CONSTRAINTS  // 12
        );
    }

    #[test]
    fn correct_number_of_qx_constraints() {
        assert_eq!(
            count_constraints::<Sha256UairQx>(),
            NUM_QX_CONSTRAINTS  // 3
        );
    }

    #[test]
    fn max_constraint_degree_is_one() {
        // The rotation constraints have degree 1 (variable * constant).
        // The shift decomposition constraints:
        //   W - R0 - S0 * X³  has degree 1 (variable * constant).
        // So overall max degree is 1.
        assert_eq!(count_max_degree::<Sha256UairBp>(), 1);
    }

    #[test]
    fn qx_max_constraint_degree() {
        // C7–C9 have degree 2 (selector variable * degree-1 carry expression)
        assert_eq!(count_max_degree::<Sha256UairQx>(), 2);
    }

    #[test]
    fn scalars_contain_rotation_polynomials() {
        let scalars = collect_scalars::<Sha256UairBp>();

        let rho0 = BinaryPoly::<32>::from((1u32 << 30) | (1u32 << 19) | (1u32 << 10));
        let rho1 = BinaryPoly::<32>::from((1u32 << 26) | (1u32 << 21) | (1u32 << 7));

        assert!(scalars.contains(&rho0), "ρ₀ not found in collected scalars");
        assert!(scalars.contains(&rho1), "ρ₁ not found in collected scalars");
    }

    #[test]
    fn qx_scalars_contain_x32() {
        let scalars = collect_scalars::<Sha256UairQx>();

        let x32: DensePolynomial<i64, 64> = {
            let mut coeffs = [0i64; 64];
            coeffs[32] = 1;
            DensePolynomial { coeffs }
        };

        assert!(scalars.contains(&x32), "X³² not found in Q[X] collected scalars");
    }
}
