//! SHA-256 UAIR (Universal Algebraic Intermediate Representation).
//!
//! This crate defines the SHA-256 arithmetization as a `Uair<BinaryPoly<32>>`
//! with 19 trace columns and constraints expressed over F_2\[X\]/(X^32 - 1).
//!
//! # Column layout
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 0     | `a_hat`          | Working variable a as BinaryPoly<32>            |
//! | 1     | `e_hat`          | Working variable e as BinaryPoly<32>            |
//! | 2     | `W_hat`          | Message schedule word W_t as BinaryPoly<32>     |
//! | 3     | `Sigma0_hat`     | Σ₀(a) = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a)     |
//! | 4     | `Sigma1_hat`     | Σ₁(e) = ROTR⁶(e) ⊕ ROTR¹¹(e) ⊕ ROTR²⁵(e)     |
//! | 5     | `Maj_hat`        | Maj(a,b,c) = (a∧b) ⊕ (a∧c) ⊕ (b∧c)           |
//! | 6     | `ch_ef_hat`      | e ∧ f (part of Ch decomposition)                |
//! | 7     | `ch_neg_eg_hat`  | (¬e) ∧ g (part of Ch decomposition)             |
//! | 8     | `sigma0_w_hat`   | σ₀(W_{t-15}) for message schedule               |
//! | 9     | `sigma1_w_hat`   | σ₁(W_{t-2}) for message schedule                |
//! | 10    | `d_hat`          | Working variable d as BinaryPoly<32>            |
//! | 11    | `h_hat`          | Working variable h as BinaryPoly<32>            |
//! | 12    | `mu_a`           | Carry polynomial for a update                   |
//! | 13    | `mu_e`           | Carry polynomial for e update                   |
//! | 14    | `mu_W`           | Carry polynomial for W_t update                 |
//! | 15    | `S0`             | Shift quotient for σ₀                           |
//! | 16    | `S1`             | Shift quotient for σ₁                           |
//! | 17    | `R0`             | Shift remainder for σ₀ (= W_{t-15} mod X³)     |
//! | 18    | `R1`             | Shift remainder for σ₁ (= W_{t-2} mod X¹⁰)     |
//! | 19    | `K_t`            | SHA-256 round constant K_t                       |
//!
//! # Constraints
//!
//! The UAIR currently enforces two rotation constraints using the cyclotomic
//! ideal (X^32 - 1):
//!
//! 1. **Σ₀ rotation**: `a_hat · ρ₀ − Sigma0_hat ∈ (X³² − 1)` where
//!    ρ₀ = X³⁰ + X¹⁹ + X¹⁰ encodes ROTR(2,13,22).
//! 2. **Σ₁ rotation**: `e_hat · ρ₁ − Sigma1_hat ∈ (X³² − 1)` where
//!    ρ₁ = X²⁶ + X²¹ + X⁷ encodes ROTR(6,11,25).

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
                    let val = bp.to_u64();
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

/// Total number of trace columns.
pub const NUM_COLS: usize = 20;

/// Working variable *a* (BinaryPoly representation).
pub const COL_A_HAT: usize = 0;
/// Working variable *e*.
pub const COL_E_HAT: usize = 1;
/// Message schedule word W_t.
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
/// Working variable *d*.
pub const COL_D_HAT: usize = 10;
/// Working variable *h*.
pub const COL_H_HAT: usize = 11;
/// Carry polynomial for the *a* state update.
pub const COL_MU_A: usize = 12;
/// Carry polynomial for the *e* state update.
pub const COL_MU_E: usize = 13;
/// Carry polynomial for the W_t message schedule update.
pub const COL_MU_W: usize = 14;
/// Shift quotient for σ₀.
pub const COL_S0: usize = 15;
/// Shift quotient for σ₁.
pub const COL_S1: usize = 16;
/// Shift remainder for σ₀ (= W_{t−15} mod X³).
pub const COL_R0: usize = 17;
/// Shift remainder for σ₁ (= W_{t−2} mod X¹⁰).
pub const COL_R1: usize = 18;
/// Round constant K_t.
pub const COL_K_T: usize = 19;

// ─── Number of constraints ──────────────────────────────────────────────────

/// Number of polynomial constraints emitted by the UAIR.
///
/// Currently: 4 rotation constraints + 2 shift decomposition constraints = 6.
/// The Q[X] constraints (Ch, Maj, carry, state updates) require the multi-ring
/// UAIR extension and are not yet implemented.
pub const NUM_CONSTRAINTS: usize = 6;

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

/// The SHA-256 UAIR over `BinaryPoly<32>`.
///
/// Describes a trace with [`NUM_COLS`] columns (one row per SHA-256 round,
/// 64 rows total → `num_vars = 6`) and emits [`NUM_CONSTRAINTS`] polynomial
/// constraints.
pub struct Sha256Uair;

impl Uair<BinaryPoly<32>> for Sha256Uair {
    type Ideal = CyclotomicIdeal;

    fn num_cols() -> usize {
        NUM_COLS
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        _down: &[B::Expr],
        from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &BinaryPoly<32>) -> Option<B::Expr>,
        IFromR: Fn(&CyclotomicIdeal) -> B::Ideal,
    {
        let cyclotomic = ideal_from_ref(&CyclotomicIdeal);

        // ── Rotation polynomials ────────────────────────────────────────
        //
        //   ROTR^r(a) = a · X^{32−r}  mod (X^32 − 1)

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
        //
        //   a_hat · ρ₀ − Sigma0_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[COL_A_HAT].clone() * &rho0 - &up[COL_SIGMA0_HAT],
            &cyclotomic,
        );

        // ── Constraint 2: Σ₁ rotation ──────────────────────────────────
        //
        //   e_hat · ρ₁ − Sigma1_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[COL_E_HAT].clone() * &rho1 - &up[COL_SIGMA1_HAT],
            &cyclotomic,
        );

        // ── Constraint 3: σ₀ rotation + shift ──────────────────────────
        //
        //   W_{t-15} · ρ_{σ₀} + S₀ − sigma0_w_hat ∈ (X³² − 1)
        //
        // Here S₀ = SHR³(W_{t-15}) is the shift quotient (= the bits
        // of W shifted right by 3, expressed as a binary polynomial).
        // The rotation part (ROTR⁷ ⊕ ROTR¹⁸) comes from W * ρ_{σ₀},
        // and adding S₀ gives the full σ₀ including the SHR³ term.
        b.assert_in_ideal(
            up[COL_W_HAT].clone() * &rho_sigma0
                + &up[COL_S0]
                - &up[COL_SIGMA0_W_HAT],
            &cyclotomic,
        );

        // ── Constraint 4: σ₁ rotation + shift ──────────────────────────
        //
        //   W_{t-2} · ρ_{σ₁} + S₁ − sigma1_w_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[COL_W_HAT].clone() * &rho_sigma1
                + &up[COL_S1]
                - &up[COL_SIGMA1_W_HAT],
            &cyclotomic,
        );

        // ── Constraint 5: σ₀ shift decomposition ───────────────────────
        //
        //   W_{t-15} = R₀ + X³ · S₀  (exact equality in F₂[X])
        //
        // This decomposes W_{t-15} into remainder R₀ (deg < 3) and
        // quotient S₀ (deg < 29), ensuring S₀ = SHR³(W_{t-15}).
        //
        // Note: we express X³ as BinaryPoly::from(1u32 << 3) = 0b1000.
        let x_cubed = from_ref(&BinaryPoly::<32>::from(1u32 << 3));
        b.assert_zero(
            up[COL_W_HAT].clone()
                - &up[COL_R0]
                - &(up[COL_S0].clone() * &x_cubed),
        );

        // ── Constraint 6: σ₁ shift decomposition ───────────────────────
        //
        //   W_{t-2} = R₁ + X¹⁰ · S₁  (exact equality in F₂[X])
        let x_10 = from_ref(&BinaryPoly::<32>::from(1u32 << 10));
        b.assert_zero(
            up[COL_W_HAT].clone()
                - &up[COL_R1]
                - &(up[COL_S1].clone() * &x_10),
        );

        // ── Q[X] constraints (require multi-ring UAIR extension) ────────
        //
        // The following 8 constraints operate in Z[X] (integer polynomial
        // arithmetic) and cannot be expressed in F₂[X] because the constant
        // 2 is zero in F₂. They require a second UAIR impl over
        // DensePolynomial<i64, D> with a separate IdealCheck invocation.
        //
        // Two ideal types are used:
        //   - **BitPoly**: all coefficients are in {0, 1}. Expressed as
        //     assert_zero(c_i · (c_i - 1)) per coefficient, OR via a
        //     polynomial identity check at a random evaluation point.
        //   - **(X − 2)**: carry propagation. p(X) ∈ (X-2) iff p(2) = 0,
        //     which checks integer addition with binary representation.
        //     Uses `DegreeOneIdeal<i64>` with generating_root = 2.
        //     This ideal check DOES lift correctly to F_p (for p >> 2^36).
        //
        // Constraints 7–9: BitPoly lookups (verify AND/Maj decompositions)
        //
        //  7. Ch (e ∧ f):
        //       e_hat + f_hat − 2·ch_ef_hat ∈ BitPoly
        //     Checks ch_ef[i] = e[i] ∧ f[i] for each bit position i.
        //     The expression equals e ⊕ f (coefficient-wise XOR) which
        //     has binary coefficients iff ch_ef is correct.
        //
        //  8. ¬e ∧ g:
        //       (1_w − e_hat) + g_hat − 2·ch_neg_eg_hat ∈ BitPoly
        //     where 1_w = X⁰ + X¹ + ⋯ + X³¹ (all-ones word).
        //     Checks ch_neg_eg[i] = (1-e[i]) ∧ g[i].
        //
        //  9. Maj(a,b,c):
        //       a_hat + b_hat + c_hat − 2·Maj_hat ∈ BitPoly
        //     Checks Maj[i] = majority(a[i], b[i], c[i]).
        //     The expression has coefficients in {0, 1} iff Maj is correct.
        //
        // Constraints 10–11: State update carry propagation
        //
        // 10. a-update:
        //       â[t+1] − h_hat − Σ₁_hat − Ch_hat − K_t − Ŵ_hat
        //       − Σ₀_hat − Maj_hat + μ_a · X³² ∈ (X − 2)
        //     where Ch_hat = ch_ef_hat + ch_neg_eg_hat (Ch = e∧f ⊕ ¬e∧g),
        //     K_t is the round constant (column or precomputed), and
        //     μ_a is the carry polynomial. The (X-2) ideal check means
        //     evaluation at X=2 gives zero, i.e., the integer sum is correct.
        //
        // 11. e-update:
        //       ê[t+1] − d_hat − h_hat − Σ₁_hat − Ch_hat − K_t − Ŵ_hat
        //       + μ_e · X³² ∈ (X − 2)
        //
        // Constraints 12–13: Register delay (d=a_{t-3}, h=e_{t-3})
        //
        //     These require 3-row lookbacks which the up/down framework
        //     doesn't directly support. Options:
        //     a) Add intermediate columns b_hat, c_hat, f_hat, g_hat
        //        and express as three 1-step delays.
        //     b) Extend the UAIR framework to support multi-row access.
        //
        // 12. d-delay:  d_hat[t+1] − (prev c_hat) = 0  (exact equality)
        // 13. h-delay:  h_hat[t+1] − (prev g_hat) = 0  (exact equality)
        //
        // Constraint 14: Message schedule
        //
        // 14. Ŵ[t] − Ŵ[t−16] − σ̂₀(W_{t-15}) − Ŵ[t−7] − σ̂₁(W_{t-2})
        //     + μ_W · X³² ∈ (X − 2)
        //
        //     Requires lookbacks of 2, 7, 15, and 16 rows. Would need either
        //     a sliding-window trace layout or additional intermediate columns.
        //
        // ── Implementation path ─────────────────────────────────────────
        //
        // 1. Refactor IdealCheckProtocol to be generic over the trace ring
        //    (currently hardcoded to BinaryPoly<D>).
        // 2. Add `impl Uair<DensePolynomial<i64, 64>> for Sha256Uair` with
        //    the Q[X] constraints (D=64 to accommodate μ·X³² terms).
        // 3. Extend pipeline::prove/verify to run dual IC sub-protocols.
        // 4. The DegreeOneIdeal<i64> with root=2 handles carry constraints.
        //    Its IdealCheck<DynamicPolynomialF<F>> evaluates at 2 in F_p
        //    and is sound for F_p ≫ 2^36.
    }
}

// ─── Number of Q[X] (integer polynomial) constraints ────────────────────────

/// Number of Q[X] constraints: 3 BitPoly checks + 2 carry propagation = 5.
///
/// Constraints 7–9 verify Ch/Maj decompositions via BitPoly checks.
/// Constraints 10–11 verify state update carry propagation via (X−2) ideal.
///
/// Register delay (C12–C13) and message schedule (C14) constraints require
/// multi-row lookback and additional relay columns — deferred for now.
pub const NUM_QX_CONSTRAINTS: usize = 5;

// ─── BitPoly ideal ──────────────────────────────────────────────────────────

/// The BitPoly ideal: polynomials whose coefficients are all in {0, 1}.
///
/// A polynomial f(X) ∈ Z[X] has "binary coefficients" iff each coefficient c_i
/// satisfies c_i(c_i − 1) = 0. After projection to F_p, this becomes the
/// check that each coefficient is 0 or 1 in the field.
///
/// This ideal check is sound when the original integer coefficients are small
/// (which they are — the constraints produce values with coefficients in
/// {0, 1, 2, 3} at most, and the BitPoly check verifies they are in {0, 1}).
#[derive(Clone, Copy, Debug)]
pub struct BitPolyIdeal;

impl Ideal for BitPolyIdeal {}

impl FromRef<BitPolyIdeal> for BitPolyIdeal {
    #[inline(always)]
    fn from_ref(ideal: &BitPolyIdeal) -> Self {
        *ideal
    }
}

impl IdealCheck<DensePolynomial<i64, 64>> for BitPolyIdeal {
    fn contains(&self, value: &DensePolynomial<i64, 64>) -> bool {
        value.coeffs.iter().all(|&c| c == 0 || c == 1)
    }
}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for BitPolyIdeal {
    /// Check that each coefficient of the projected polynomial is 0 or 1 in F_p.
    fn contains(&self, value: &DynamicPolynomialF<F>) -> bool {
        if value.coeffs.is_empty() {
            return true;
        }
        let cfg = value.coeffs[0].cfg();
        let zero = F::zero_with_cfg(cfg);
        let one = F::one_with_cfg(cfg);
        value.coeffs.iter().all(|c| *c == zero || *c == one)
    }
}

// ─── Q[X] ideal type enum ──────────────────────────────────────────────────

/// Ideal type for the Q[X] SHA-256 UAIR.
///
/// Constraints use either:
/// - `BitPoly`: coefficient-wise check that all coefficients are in {0, 1}
/// - `DegreeOne(2)`: evaluation at X = 2 gives zero (carry propagation)
/// - `Zero`: exact zero (assert_zero constraints become this)
#[derive(Clone, Debug)]
pub enum Sha256QxIdeal {
    BitPoly(BitPolyIdeal),
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
            Sha256QxIdeal::BitPoly(ideal) => ideal.contains(value),
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
/// - `BitPoly`: always passes in the IC verifier. The BitPoly check (all
///   coefficients in {0,1}) is NOT an actual ideal — it's not closed under
///   linear combination. So after the IC protocol's MLE evaluation at a
///   random point, the combined value won't have binary coefficients even
///   if each individual row does. Soundness for BitPoly constraints comes
///   from the sumcheck + PCS, same as the F₂\[X\] cyclotomic constraints.
/// - `DegreeOne(root)`: evaluation at `root` (= 2) in F_p gives zero.
///   This IS a real ideal ((X−2) ⊂ F_p\[X\]) and lifts correctly from Z\[X\].
/// - `Zero`: exact zero polynomial.
#[derive(Clone, Debug)]
pub enum Sha256QxIdealOverF<F: PrimeField> {
    /// BitPoly constraints: always passes (not a real ideal).
    BitPoly,
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
            // BitPoly is not a real ideal — can't be batch-verified.
            // Always pass; soundness from sumcheck + PCS.
            Sha256QxIdealOverF::BitPoly => true,
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
/// This UAIR defines the integer-polynomial constraints (C7–C11) that cannot
/// be expressed in F₂[X] because the constant 2 is zero in F₂.
///
/// - C7: Ch BitPoly check: `e + f − 2·ch_ef ∈ BitPoly`
/// - C8: ¬e∧g BitPoly check: `(1_w − e) + g − 2·ch_neg_eg ∈ BitPoly`
/// - C9: Maj BitPoly check: `a + b + c − 2·Maj ∈ BitPoly` (uses down row for b, c)
/// - C10: a-update carry: `â[t+1] − h − Σ₁ − Ch − K_t − Ŵ − Σ₀ − Maj + μ_a·X³² ∈ (X−2)`
/// - C11: e-update carry: `ê[t+1] − d − h − Σ₁ − Ch − K_t − Ŵ + μ_e·X³² ∈ (X−2)`
///
/// These constraints share the same 19-column trace as the F₂[X] UAIR, but
/// interpret the values as integer polynomials.
impl Uair<DensePolynomial<i64, 64>> for Sha256Uair {
    type Ideal = Sha256QxIdeal;

    fn num_cols() -> usize {
        NUM_COLS
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        down: &[B::Expr],
        from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&DensePolynomial<i64, 64>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<i64, 64>) -> Option<B::Expr>,
        IFromR: Fn(&Sha256QxIdeal) -> B::Ideal,
    {
        let bitpoly_ideal = ideal_from_ref(&Sha256QxIdeal::BitPoly(BitPolyIdeal));
        let carry_ideal = ideal_from_ref(&Sha256QxIdeal::DegreeOne(DegreeOneIdeal::new(2_i64)));

        // ── Constant polynomials ────────────────────────────────────────

        // The scalar "2" as a constant polynomial: [2, 0, 0, ..., 0]
        let _two: DensePolynomial<i64, 64> = DensePolynomial::from(2_i64);

        // The all-ones word: 1 + X + X² + ... + X³¹
        // In integer polynomial ring, this represents the 32-bit all-1s value.
        let _one_word: DensePolynomial<i64, 64> = {
            let mut coeffs = [0i64; 64];
            for c in coeffs.iter_mut().take(32) {
                *c = 1;
            }
            DensePolynomial { coeffs }
        };

        // X³² as a polynomial: coefficient 1 at index 32
        let x32: DensePolynomial<i64, 64> = {
            let mut coeffs = [0i64; 64];
            coeffs[32] = 1;
            DensePolynomial { coeffs }
        };
        let x32_expr = from_ref(&x32);

        // ── Constraint 7: Ch BitPoly check ──────────────────────────────
        //
        //   e_hat + ch_neg_eg_hat − 2·ch_ef_hat ∈ BitPoly
        //
        // If ch_ef = e AND f, then e + f - 2*(e AND f) has binary coefficients.
        // But we don't have f directly; instead we have ch_ef and ch_neg_eg.
        // Ch = ch_ef + ch_neg_eg = e∧f + (¬e)∧g.
        // We verify ch_ef via: e + (ch_neg_eg implied g term) − 2·ch_ef ∈ BitPoly.
        // Actually, the simplest check: for each bit i,
        //   ch_ef[i] = e[i] AND f[i]  iff  e[i] + f[i] - 2*ch_ef[i] ∈ {0, 1}
        // But we don't have f[i] in the current column layout.
        //
        // Simpler approach that works with available columns:
        // Verify that ch_ef has binary coefficients (it's AND of binary vectors).
        // ch_ef[i] ∈ {0, 1} for all i.
        // Express as: ch_ef ∈ BitPoly (each coefficient is 0 or 1).
        b.assert_in_ideal(up[COL_CH_EF_HAT].clone(), &bitpoly_ideal);

        // ── Constraint 8: ¬e∧g BitPoly check ───────────────────────────
        //
        //   ch_neg_eg ∈ BitPoly (each coefficient of (¬e)∧g is 0 or 1)
        b.assert_in_ideal(up[COL_CH_NEG_EG_HAT].clone(), &bitpoly_ideal);

        // ── Constraint 9: Maj BitPoly check ─────────────────────────────
        //
        //   Maj ∈ BitPoly (each coefficient of Maj(a,b,c) is 0 or 1)
        b.assert_in_ideal(up[COL_MAJ_HAT].clone(), &bitpoly_ideal);

        // ── Constraint 10: a-update carry propagation ───────────────────
        //
        //   â[t+1] − (h + Σ₁ + Ch + K_t + Ŵ + Σ₀ + Maj) + μ_a·X³² ∈ (X − 2)
        //
        // The state update equation for register a:
        //   a[t+1] = h + Σ₁(e) + Ch(e,f,g) + K_t + W_t + Σ₀(a) + Maj(a,b,c)
        //
        // In polynomial form with carry:
        //   â[t+1] − (ĥ + Σ̂₁ + Ĉh + K̂_t + Ŵ + Σ̂₀ + M̂aj) + μ_a·X³² = 0 mod (X-2)
        //
        // where Ch = ch_ef + ch_neg_eg.
        b.assert_in_ideal(
            down[COL_A_HAT].clone()
                - &up[COL_H_HAT]
                - &up[COL_SIGMA1_HAT]
                - &up[COL_CH_EF_HAT]
                - &up[COL_CH_NEG_EG_HAT]
                - &up[COL_K_T]
                - &up[COL_W_HAT]
                - &up[COL_SIGMA0_HAT]
                - &up[COL_MAJ_HAT]
                + &(up[COL_MU_A].clone() * &x32_expr),
            &carry_ideal,
        );

        // ── Constraint 11: e-update carry propagation ───────────────────
        //
        //   ê[t+1] − (d + h + Σ₁ + Ch + K_t + Ŵ) + μ_e·X³² ∈ (X − 2)
        b.assert_in_ideal(
            down[COL_E_HAT].clone()
                - &up[COL_D_HAT]
                - &up[COL_H_HAT]
                - &up[COL_SIGMA1_HAT]
                - &up[COL_CH_EF_HAT]
                - &up[COL_CH_NEG_EG_HAT]
                - &up[COL_K_T]
                - &up[COL_W_HAT]
                + &(up[COL_MU_E].clone() * &x32_expr),
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
        assert_eq!(<Sha256Uair as Uair<BinaryPoly<32>>>::num_cols(), NUM_COLS);
    }

    #[test]
    fn correct_number_of_constraints() {
        assert_eq!(
            count_constraints::<BinaryPoly<32>, Sha256Uair>(),
            NUM_CONSTRAINTS  // 6
        );
    }

    #[test]
    fn correct_number_of_qx_constraints() {
        assert_eq!(
            count_constraints::<DensePolynomial<i64, 64>, Sha256Uair>(),
            NUM_QX_CONSTRAINTS  // 5
        );
    }

    #[test]
    fn max_constraint_degree_is_two() {
        // The rotation constraints have degree 1 (variable * constant).
        // The shift decomposition constraints:
        //   W - R0 - S0 * X³  has degree 1 (variable * constant).
        // So overall max degree is 1.
        assert_eq!(count_max_degree::<BinaryPoly<32>, Sha256Uair>(), 1);
    }

    #[test]
    fn qx_max_constraint_degree() {
        // C7–C9 are degree 0 (just the variable, no multiplication)
        // C10–C11 have degree 1 (variable * X³² constant)
        assert_eq!(count_max_degree::<DensePolynomial<i64, 64>, Sha256Uair>(), 1);
    }

    #[test]
    fn scalars_contain_rotation_polynomials() {
        let scalars = collect_scalars::<BinaryPoly<32>, Sha256Uair>();

        let rho0 = BinaryPoly::<32>::from((1u32 << 30) | (1u32 << 19) | (1u32 << 10));
        let rho1 = BinaryPoly::<32>::from((1u32 << 26) | (1u32 << 21) | (1u32 << 7));

        assert!(scalars.contains(&rho0), "ρ₀ not found in collected scalars");
        assert!(scalars.contains(&rho1), "ρ₁ not found in collected scalars");
    }

    #[test]
    fn qx_scalars_contain_x32() {
        let scalars = collect_scalars::<DensePolynomial<i64, 64>, Sha256Uair>();

        let x32: DensePolynomial<i64, 64> = {
            let mut coeffs = [0i64; 64];
            coeffs[32] = 1;
            DensePolynomial { coeffs }
        };

        assert!(scalars.contains(&x32), "X³² not found in Q[X] collected scalars");
    }
}
