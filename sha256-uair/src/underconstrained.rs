//! Underconstrained SHA-256 UAIR⁺ — F₂\[X\] columns removed.
//!
//! This is a variant of the full SHA-256 UAIR that **omits** the 4 F₂\[X\]
//! columns (S₀, S₁, R₀, R₁) and the 4 constraints that reference them
//! (σ₀ rotation+shift, σ₁ rotation+shift, σ₀ shift decomposition,
//! σ₁ shift decomposition).
//!
//! The remaining constraints are:
//! - C1: Σ₀ rotation  (`â · ρ₀ − Σ̂₀ ∈ (X³² − 1)`)
//! - C2: Σ₁ rotation  (`ê · ρ₁ − Σ̂₁ ∈ (X³² − 1)`)
//! - C3–C8:  linking constraints (d, h, W_tm2, W_tm7, W_tm15, W_tm16)
//! - C9–C12: Ch/Maj lookback links (a_tm1, a_tm2, e_tm1, e_tm2)
//!
//! # Column layout (24 total: 21 bit-poly + 3 integer)
//!
//! ## Bit-polynomial columns — {0,1}^{<32}\[X\] (indices 0–9)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 0     | `a_hat`          | Working variable a (bit-poly representative)    |
//! | 1     | `e_hat`          | Working variable e (bit-poly representative)    |
//! | 2     | `W_hat`          | Message schedule word W_t                       |
//! | 3     | `Sigma0_hat`     | Σ₀(a)                                           |
//! | 4     | `Sigma1_hat`     | Σ₁(e)                                           |
//! | 5     | `Maj_hat`        | Maj(a,b,c)                                      |
//! | 6     | `ch_ef_hat`      | e ∧ f (part of Ch decomposition)                |
//! | 7     | `ch_neg_eg_hat`  | (¬e) ∧ g (part of Ch decomposition)             |
//! | 8     | `sigma0_w_hat`   | σ₀(W_{t−15}) for message schedule               |
//! | 9     | `sigma1_w_hat`   | σ₁(W_{t−2}) for message schedule                |
//!
//! ## Auxiliary lookback columns — {0,1}^{<32}\[X\] (indices 10–16)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 10    | `d_hat`          | d_t = a_{t−3}                                   |
//! | 11    | `h_hat`          | h_t = e_{t−3}                                   |
//! | 12    | `W_tm2`          | W[t−2]                                           |
//! | 13    | `W_tm7`          | W[t−7]                                           |
//! | 14    | `W_tm15`         | W[t−15]                                          |
//! | 15    | `W_tm16`         | W[t−16]                                          |
//! | 16    | `K_hat`          | Round constant K_t                               |
//!
//! ## Ch/Maj lookback columns (indices 17–20)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 17    | `a_tm1`          | a[t−1] = b_t                                     |
//! | 18    | `a_tm2`          | a[t−2] = c_t                                     |
//! | 19    | `e_tm1`          | e[t−1] = f_t                                     |
//! | 20    | `e_tm2`          | e[t−2] = g_t                                     |
//!
//! ## Integer columns — Z (indices 21–23 in flattened trace)
//!
//! | Index | Name             | Description                                     |
//! |-------|------------------|-------------------------------------------------|
//! | 21    | `mu_a`           | Carry for a-update (∈ {0,…,6})                  |
//! | 22    | `mu_e`           | Carry for e-update (∈ {0,…,5})                  |
//! | 23    | `mu_W`           | Carry for W schedule (∈ {0,…,3})                |

use crypto_primitives::crypto_bigint_int::Int;
use rand::RngCore;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_uair::{
    ConstraintBuilder, Uair,
};

use crate::Sha256Ideal;
use crate::witness::GenerateWitness;

// ─── Column counts ──────────────────────────────────────────────────────────

/// Total number of trace columns (21 bit-poly + 3 integer).
pub const UC_NUM_COLS: usize = 24;

/// Number of bit-polynomial columns.
pub const UC_NUM_BITPOLY_COLS: usize = 21;

/// Number of integer columns (Z).
pub const UC_NUM_INT_COLS: usize = 3;

// ── Bit-polynomial columns (indices 0–9) — same as full UAIR ───────────────

/// Working variable *a* (bit-poly representative â_t).
pub const UC_COL_A_HAT: usize = 0;
/// Working variable *e* (bit-poly representative ê_t).
pub const UC_COL_E_HAT: usize = 1;
/// Message schedule word Ŵ_t.
pub const UC_COL_W_HAT: usize = 2;
/// Σ₀(a) = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a).
pub const UC_COL_SIGMA0_HAT: usize = 3;
/// Σ₁(e) = ROTR⁶(e) ⊕ ROTR¹¹(e) ⊕ ROTR²⁵(e).
pub const UC_COL_SIGMA1_HAT: usize = 4;
/// Maj(a,b,c).
pub const UC_COL_MAJ_HAT: usize = 5;
/// e ∧ f (first term of Ch).
pub const UC_COL_CH_EF_HAT: usize = 6;
/// (¬e) ∧ g (second term of Ch).
pub const UC_COL_CH_NEG_EG_HAT: usize = 7;
/// σ₀(W_{t−15}) for the message schedule.
pub const UC_COL_SIGMA0_W_HAT: usize = 8;
/// σ₁(W_{t−2}) for the message schedule.
pub const UC_COL_SIGMA1_W_HAT: usize = 9;

// ── Auxiliary lookback columns (indices 10–16) ──────────────────────────────
// (Shifted down by 4 from the full UAIR because F₂[X] cols 10–13 removed.)

/// d_t = a_{t−3}.
pub const UC_COL_D_HAT: usize = 10;
/// h_t = e_{t−3}.
pub const UC_COL_H_HAT: usize = 11;
/// W[t−2].
pub const UC_COL_W_TM2: usize = 12;
/// W[t−7].
pub const UC_COL_W_TM7: usize = 13;
/// W[t−15].
pub const UC_COL_W_TM15: usize = 14;
/// W[t−16].
pub const UC_COL_W_TM16: usize = 15;
/// Round constant K_t.
pub const UC_COL_K_HAT: usize = 16;

// ── Ch/Maj lookback columns (indices 17–20) ─────────────────────────────────

/// a[t−1] = b_t.
pub const UC_COL_A_TM1: usize = 17;
/// a[t−2] = c_t.
pub const UC_COL_A_TM2: usize = 18;
/// e[t−1] = f_t.
pub const UC_COL_E_TM1: usize = 19;
/// e[t−2] = g_t.
pub const UC_COL_E_TM2: usize = 20;


// ── Integer columns (indices 0–2 within the int sub-slice) ──────────────────

/// Carry for the *a* state update (∈ {0,…,6}).
pub const UC_COL_INT_MU_A: usize = 0;
/// Carry for the *e* state update (∈ {0,…,5}).
pub const UC_COL_INT_MU_E: usize = 1;
/// Carry for the W_t message schedule update (∈ {0,…,3}).
pub const UC_COL_INT_MU_W: usize = 2;

// ── Constraint count ────────────────────────────────────────────────────────

/// Number of polynomial constraints emitted by the underconstrained Bp UAIR.
/// C1–C2 (rotation) + C3–C12 (10 linking constraints) + C13–C15 (3 carry constraints).
/// Constraints 3–6 from the full UAIR (σ₀/σ₁ shift constraints that
/// referenced S₀, S₁, R₀, R₁) are removed.
pub const UC_NUM_CONSTRAINTS: usize = 15;

// ─── Underconstrained SHA-256 Bp UAIR ───────────────────────────────────────

/// Underconstrained SHA-256 UAIR over `BinaryPoly<32>`.
///
/// Same as [`crate::Sha256UairBp`] but with F₂\[X\] columns (S₀, S₁, R₀, R₁)
/// and their associated constraints removed.
pub struct Sha256UairBpUnderconstrained;

// Down-row indices — same shift structure as full UAIR, but column indices
// adjusted for the removed F₂[X] columns.
const UC_DOWN_BP_D: usize = 0;
const UC_DOWN_BP_H: usize = 1;
const UC_DOWN_BP_W_TM2: usize = 2;
const UC_DOWN_BP_W_TM7: usize = 3;
const UC_DOWN_BP_W_TM15: usize = 4;
const UC_DOWN_BP_W_TM16: usize = 5;
const UC_DOWN_BP_A_TM1: usize = 6;
const UC_DOWN_BP_A_TM2: usize = 7;
const UC_DOWN_BP_E_TM1: usize = 8;
const UC_DOWN_BP_E_TM2: usize = 9;
const UC_DOWN_BP_A_NEXT: usize = 10;
const UC_DOWN_BP_E_NEXT: usize = 11;

impl Uair for Sha256UairBpUnderconstrained {
    type Ideal = Sha256Ideal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: UC_NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: UC_NUM_INT_COLS,
            shifts: vec![
                // Linking shifts for auxiliary lookback columns.
                zinc_uair::ShiftSpec { source_col: UC_COL_D_HAT,  shift_amount: 3 },
                zinc_uair::ShiftSpec { source_col: UC_COL_H_HAT,  shift_amount: 3 },
                zinc_uair::ShiftSpec { source_col: UC_COL_W_TM2,  shift_amount: 2 },
                zinc_uair::ShiftSpec { source_col: UC_COL_W_TM7,  shift_amount: 7 },
                zinc_uair::ShiftSpec { source_col: UC_COL_W_TM15, shift_amount: 15 },
                zinc_uair::ShiftSpec { source_col: UC_COL_W_TM16, shift_amount: 16 },
                // Linking shifts for Ch/Maj affine-combination lookback columns.
                zinc_uair::ShiftSpec { source_col: UC_COL_A_TM1,  shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: UC_COL_A_TM2,  shift_amount: 2 },
                zinc_uair::ShiftSpec { source_col: UC_COL_E_TM1,  shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: UC_COL_E_TM2,  shift_amount: 2 },
                // Forward shifts for carry constraints (â[t+1], ê[t+1]).
                zinc_uair::ShiftSpec { source_col: UC_COL_A_HAT,  shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: UC_COL_E_HAT,  shift_amount: 1 },
            ],
            public_columns: vec![
                UC_COL_W_HAT, UC_COL_K_HAT,
                UC_COL_W_TM2, UC_COL_W_TM7, UC_COL_W_TM15, UC_COL_W_TM16,
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
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &BinaryPoly<32>) -> Option<B::Expr>,
        IFromR: Fn(&Sha256Ideal) -> B::Ideal,
    {
        let int_up = up.int;
        let up = up.binary_poly;
        let bp_down = down.binary_poly;
        let cyclotomic = ideal_from_ref(&Sha256Ideal::Cyclotomic);

        // ── Rotation polynomials ────────────────────────────────────────

        // ρ₀ = X³⁰ + X¹⁹ + X¹⁰  →  encodes ROTR(2,13,22) for Σ₀
        let rho0 = from_ref(&BinaryPoly::<32>::from(
            (1u32 << 30) | (1u32 << 19) | (1u32 << 10),
        ));

        // ρ₁ = X²⁶ + X²¹ + X⁷  →  encodes ROTR(6,11,25) for Σ₁
        let rho1 = from_ref(&BinaryPoly::<32>::from(
            (1u32 << 26) | (1u32 << 21) | (1u32 << 7),
        ));

        // ── Constraint 1: Σ₀ rotation ──────────────────────────────────
        //   a_hat · ρ₀ − Sigma0_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[UC_COL_A_HAT].clone() * &rho0 - &up[UC_COL_SIGMA0_HAT],
            &cyclotomic,
        );

        // ── Constraint 2: Σ₁ rotation ──────────────────────────────────
        //   e_hat · ρ₁ − Sigma1_hat ∈ (X³² − 1)
        b.assert_in_ideal(
            up[UC_COL_E_HAT].clone() * &rho1 - &up[UC_COL_SIGMA1_HAT],
            &cyclotomic,
        );

        // NOTE: Constraints 3–6 from the full UAIR (σ₀/σ₁ rotation+shift
        // and shift decomposition) are intentionally removed. They
        // referenced the F₂[X] columns S₀, S₁, R₀, R₁ which are not
        // present in this underconstrained version.

        // ── Constraint 3: d-link (shift-by-3) ──────────────────────────
        //   d[t+3] = a[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_D].clone() - &up[UC_COL_A_HAT],
        );

        // ── Constraint 4: h-link (shift-by-3) ──────────────────────────
        //   h[t+3] = e[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_H].clone() - &up[UC_COL_E_HAT],
        );

        // ── Constraint 5: W_tm2-link (shift-by-2) ──────────────────────
        //   W_tm2[t+2] = W[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_W_TM2].clone() - &up[UC_COL_W_HAT],
        );

        // ── Constraint 6: W_tm7-link (shift-by-7) ──────────────────────
        //   W_tm7[t+7] = W[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_W_TM7].clone() - &up[UC_COL_W_HAT],
        );

        // ── Constraint 7: W_tm15-link (shift-by-15) ────────────────────
        //   W_tm15[t+15] = W[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_W_TM15].clone() - &up[UC_COL_W_HAT],
        );

        // ── Constraint 8: W_tm16-link (shift-by-16) ────────────────────
        //   W_tm16[t+16] = W[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_W_TM16].clone() - &up[UC_COL_W_HAT],
        );

        // ── Constraint 9: a_tm1-link (shift-by-1) ──────────────────────
        //   a_tm1[t+1] = a[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_A_TM1].clone() - &up[UC_COL_A_HAT],
        );

        // ── Constraint 10: a_tm2-link (shift-by-2) ─────────────────────
        //   a_tm2[t+2] = a[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_A_TM2].clone() - &up[UC_COL_A_HAT],
        );

        // ── Constraint 11: e_tm1-link (shift-by-1) ─────────────────────
        //   e_tm1[t+1] = e[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_E_TM1].clone() - &up[UC_COL_E_HAT],
        );

        // ── Constraint 12: e_tm2-link (shift-by-2) ─────────────────────
        //   e_tm2[t+2] = e[t]
        b.assert_zero(
            bp_down[UC_DOWN_BP_E_TM2].clone() - &up[UC_COL_E_HAT],
        );

        // ── Carry constraints (C13–C15) ─────────────────────────────────
        //
        // Carry propagation constraints over Q[X], using the (X−2) ideal.
        // X^32 is expressed as X^16 * X^16 since BinaryPoly<32> cannot
        // represent degree 32 directly.

        let trivial_ideal = ideal_from_ref(&Sha256Ideal::Trivial);

        let x16 = from_ref(&BinaryPoly::from(1u32 << 16));
        let x32_expr = x16.clone() * &x16;

        // ── Constraint 13: a-update carry propagation (trivial) ─────────
        //   â[t+1] − ĥ − Σ̂₁ − ch_ef − ch_neg_eg
        //   − K̂ − Ŵ − Σ̂₀ − Maj + μ_a·X^32 ∈ Trivial
        b.assert_in_ideal(
            bp_down[UC_DOWN_BP_A_NEXT].clone()
                - &up[UC_COL_H_HAT]
                - &up[UC_COL_SIGMA1_HAT]
                - &up[UC_COL_CH_EF_HAT]
                - &up[UC_COL_CH_NEG_EG_HAT]
                - &up[UC_COL_K_HAT]
                - &up[UC_COL_W_HAT]
                - &up[UC_COL_SIGMA0_HAT]
                - &up[UC_COL_MAJ_HAT]
                + &(int_up[UC_COL_INT_MU_A].clone() * &x32_expr),
            &trivial_ideal,
        );

        // ── Constraint 14: e-update carry propagation (trivial) ─────────
        //   ê[t+1] − d̂ − ĥ − Σ̂₁ − ch_ef − ch_neg_eg
        //   − K̂ − Ŵ + μ_e·X^32 ∈ Trivial
        b.assert_in_ideal(
            bp_down[UC_DOWN_BP_E_NEXT].clone()
                - &up[UC_COL_D_HAT]
                - &up[UC_COL_H_HAT]
                - &up[UC_COL_SIGMA1_HAT]
                - &up[UC_COL_CH_EF_HAT]
                - &up[UC_COL_CH_NEG_EG_HAT]
                - &up[UC_COL_K_HAT]
                - &up[UC_COL_W_HAT]
                + &(int_up[UC_COL_INT_MU_E].clone() * &x32_expr),
            &trivial_ideal,
        );

        // ── Constraint 15: message schedule carry propagation (trivial) ─
        //   Ŵ − Ŵ_tm16 − σ̂₀_w − Ŵ_tm7 − σ̂₁_w
        //   + μ_W·X^32 ∈ Trivial
        b.assert_in_ideal(
            up[UC_COL_W_HAT].clone()
                - &up[UC_COL_W_TM16]
                - &up[UC_COL_SIGMA0_W_HAT]
                - &up[UC_COL_W_TM7]
                - &up[UC_COL_SIGMA1_W_HAT]
                + &(int_up[UC_COL_INT_MU_W].clone() * &x32_expr),
            &trivial_ideal,
        );
    }
}

// ─── Witness generation ─────────────────────────────────────────────────────



/// Mapping from underconstrained column indices to full UAIR column indices.
///
/// Full UAIR columns 10–13 (S0, S1, R0, R1) are skipped.
/// Full column:  0  1  2  3  4  5  6  7  8  9  14 15 16 17 18 19 20 21 22 23 24 25 26 27
/// UC column:    0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23
const UC_TO_FULL_COL: [usize; UC_NUM_COLS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,      // bit-poly (same)
    14, 15, 16, 17, 18, 19, 20,          // aux lookback (was 14–20)
    21, 22, 23, 24,                       // Ch/Maj lookback (was 21–24)
    25, 26, 27,                           // integer (was 25–27)
];

impl GenerateWitness<BinaryPoly<32>> for Sha256UairBpUnderconstrained {
    /// Generate the underconstrained SHA-256 witness trace.
    ///
    /// Delegates to the full witness generator and strips out the 4 F₂[X]
    /// columns (S₀, S₁, R₀, R₁ at full indices 10–13).
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
        let full =
            <crate::Sha256UairBp as GenerateWitness<BinaryPoly<32>>>::generate_witness(
                num_vars, rng,
            );
        UC_TO_FULL_COL
            .iter()
            .map(|&i| full[i].clone())
            .collect()
    }
}

/// Generate only the 21 BinaryPoly columns (indices 0–20) used in the
/// underconstrained UAIR.
pub fn generate_uc_poly_witness(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let full =
        <Sha256UairBpUnderconstrained as GenerateWitness<BinaryPoly<32>>>::generate_witness(
            num_vars, rng,
        );
    UC_POLY_COLUMN_INDICES
        .iter()
        .map(|&i| full[i].clone())
        .collect()
}

/// Generate the 3 integer columns for the underconstrained UAIR,
/// encoded as `Int<1>` (same as the full UAIR).
pub fn generate_uc_int_witness(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> Vec<DenseMultilinearExtension<Int<1>>> {
    // Reuse the full UAIR's int witness generator (same columns).
    crate::witness::generate_int_witness(num_vars, rng)
}

// ─── Column classification for split PCS batches ────────────────────────────

/// Bit-polynomial column indices (0–20): the 10 Q[X] bit-poly columns,
/// 7 auxiliary lookback columns, and 4 Ch/Maj lookback columns.
/// (No F₂[X] columns.)
pub const UC_POLY_COLUMN_INDICES: [usize; 21] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
];

/// Integer column indices (21–23): the 3 carry columns μ_a, μ_e, μ_W.
pub const UC_INT_COLUMN_INDICES: [usize; 3] = [21, 22, 23];

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
    fn correct_number_of_columns() {
        let sig = Sha256UairBpUnderconstrained::signature();
        assert_eq!(sig.binary_poly_cols, UC_NUM_BITPOLY_COLS);
        assert_eq!(sig.int_cols, UC_NUM_INT_COLS);
        assert_eq!(sig.total_cols(), UC_NUM_COLS);
    }

    #[test]
    fn correct_number_of_constraints() {
        assert_eq!(
            count_constraints::<Sha256UairBpUnderconstrained>(),
            UC_NUM_CONSTRAINTS,
        );
    }

    #[test]
    fn max_constraint_degree_is_one() {
        // Rotation constraints use mul_by_scalar (degree 1); carry
        // constraints are now degree 1 (trivial ideal, no selectors).
        assert_eq!(count_max_degree::<Sha256UairBpUnderconstrained>(), 1);
    }

    #[test]
    fn scalars_contain_rotation_polynomials() {
        let scalars = collect_scalars::<Sha256UairBpUnderconstrained>();

        let rho0 = BinaryPoly::<32>::from((1u32 << 30) | (1u32 << 19) | (1u32 << 10));
        let rho1 = BinaryPoly::<32>::from((1u32 << 26) | (1u32 << 21) | (1u32 << 7));

        assert!(scalars.contains(&rho0), "ρ₀ not found in collected scalars");
        assert!(scalars.contains(&rho1), "ρ₁ not found in collected scalars");
    }

    #[test]
    fn scalars_do_not_contain_sigma_shift_polynomials() {
        let scalars = collect_scalars::<Sha256UairBpUnderconstrained>();

        // These were used in the removed σ₀/σ₁ shift constraints.
        let rho_sigma0 = BinaryPoly::<32>::from((1u32 << 25) | (1u32 << 14));
        let rho_sigma1 = BinaryPoly::<32>::from((1u32 << 15) | (1u32 << 13));
        let x_cubed = BinaryPoly::<32>::from(1u32 << 3);
        let x_10 = BinaryPoly::<32>::from(1u32 << 10);

        assert!(!scalars.contains(&rho_sigma0), "ρ_σ₀ should not be present");
        assert!(!scalars.contains(&rho_sigma1), "ρ_σ₁ should not be present");
        assert!(!scalars.contains(&x_cubed), "X³ should not be present");
        assert!(!scalars.contains(&x_10), "X¹⁰ should not be present");
    }

    #[test]
    fn scalars_contain_x16() {
        let scalars = collect_scalars::<Sha256UairBpUnderconstrained>();

        let x16 = BinaryPoly::<32>::from(1u32 << 16);

        assert!(scalars.contains(&x16), "X¹⁶ not found in collected scalars");
    }
}
