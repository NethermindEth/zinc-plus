//! SHA-256 compression UAIR over `F_2[X]`.
//!
//! Spec source: `documentation/sha-f2x-doc/`. This is the binary-native
//! companion to the integer-pipeline [`Sha256CompressionSliceUair`].
//! Every committed witness cell is a bit-polynomial in
//! `bitpoly_{32} ⊂ F_2[X]` and gets wired through the `F_2` prove path
//! ([`zinc_protocol::f2_prove::ZincPlusPiopF2`]). Booleanity is
//! structural (the ambient ring is `F_2`); the `Q[X]`-pipeline's
//! booleanity sumcheck disappears.
//!
//! ## Column layout (binary_poly only — the `int` lane is empty for F_2 UAIRs)
//!
//! Public bit-poly prefix:
//!   - `PA_A`, `PA_E`              boundary `H_i` for `a`/`e` (init prefix + junction)
//!   - `PA_K`                      round constants `K_t`
//!   - `PA_M`                      message-block words `M_i[0..16]`
//!   - `PA_C_W`, `PA_C_T1`, `PA_C_T2`, `PA_C_A`, `PA_C_E`,
//!     `PA_C_FF_A`, `PA_C_FF_E`    7 compensators absorbing the inner
//!                                 expressions of each modular sum on
//!                                 inactive rows (zero on active rows)
//!   - `S_INIT_PREFIX`, `S_FF`,
//!     `S_MSG_INIT`                3 `{0,1}`-valued selectors (carried
//!                                 as bit-polys, value 1 ↔ poly `1`)
//!
//! Committed witnesses (27):
//!   - state (3):                  `W_A`, `W_E`, `W_W`
//!   - rotation/shift outputs (4): `W_SIGMA0`, `W_SIGMA1`, `W_SIG0`,
//!                                 `W_SIG1` (pinned by C1–C4)
//!   - SHR auxiliaries (2):        `W_SHR3_W`, `W_SHR10_W` — committed
//!                                 here because the `SHR^r` access
//!                                 mechanism is left as a design
//!                                 parameter (cf. doc §3.3)
//!   - AND-result (3):             `W_UEF`, `W_UNEG_E_G`, `W_MAJ`
//!                                 (pinned by external Hadamard checks,
//!                                 not by in-circuit constraints —
//!                                 mechanism deferred per doc §3.4)
//!   - round intermediates (2):    `W_T1`, `W_T2`
//!   - CSA majorities (6):         `W_M_W1`, `W_M_W2`, `W_M_T1_{1..4}`
//!   - Binius carries (7):         `W_C_W`, `W_C_T1`, `W_C_T2`,
//!                                 `W_C_A`, `W_C_E`, `W_C_FF_A`,
//!                                 `W_C_FF_E`
//!
//! ## Constraint families
//!
//! A — Rotation/shift identities (C1–C4), all `(X^32 − 1)`-ideal in
//!     `F_2[X]`: bind the committed Σ/σ-output columns to the state
//!     columns via the `rho` polynomials of doc §2.2 / §2.3.
//!
//! B — Modular-addition LinSum constraints (C5–C11), all
//!     `(X^32 − 1)`-ideal. Each is the closed-form `F_2[X]`-linear
//!     identity from doc §3.6 / §3.7: sum of inputs − sum of
//!     `X·m_j` (CSA majorities) − `X·c_*` (final Binius carry) +
//!     compensator. The Hadamard half of the modular-add gadget is
//!     external (deferred per doc §3.4).
//!
//! C — Boundary pinning (C16–C18), `assert_zero` gated by selectors:
//!     pin init-prefix `W_A`/`W_E` to `PA_A`/`PA_E`, pin
//!     message-seed `W_W` to `PA_M`.
//!
//! The Hadamard `AND`-relations and CSA Hadamard constraints (C12–C15
//! in the doc) are NOT enforced at the AIR layer. They would be
//! discharged by a separate Hadamard-product check — see doc §3.4 for
//! the discharge-mechanism gap. This is the same "soundness gap by
//! design" that the doc explicitly calls out.

use core::marker::PhantomData;

use crypto_primitives::{ConstSemiring, PrimeField, Semiring};
use rand::RngCore;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial,
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, ShiftSpec, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace,
    ideal::{Ideal, IdealCheck, IdealCheckError, rotation::RotationIdeal},
    ideal_collector::IdealOrZero,
};
use zinc_utils::from_ref::FromRef;

use crate::GenerateRandomTrace;

// ---------------------------------------------------------------------------
// Sha256F2Ideal — the two F_2[X] ideals the UAIR's constraints live in.
// ---------------------------------------------------------------------------

/// Ideals used by the F_2[X] SHA-256 UAIR. Two variants:
///
/// - [`Cyclotomic`](Self::Cyclotomic) = `(X^32 − 1)`. Used by the
///   rotation/shift constraints C1–C4: cyclic shifts on 32-bit cells
///   are exactly `X·_ mod (X^32 − 1)`.
/// - [`Power`](Self::Power) = `(X^32)`. Used by the modular-addition
///   constraints C5–C11. Under `(X^32)` the addition equations cleanly
///   express `target_i = a_i ⊕ b_i ⊕ k_{i-1}` for `i = 0..31` with the
///   overflow bit `k_31` absorbed at bit 32 — no compensator column
///   needed, and the carry word `k` becomes the simple shift
///   `k = ShiftR(1)(target ⊕ Σ(other-summands))` (no cyclic wrap).
///   See `protocol/src/f2_gkr_plan.md` for the derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sha256F2Ideal {
    /// `(X^32 − 1)` — for cyclic-shift / rotation constraints.
    Cyclotomic,
    /// `(X^32)` — for modular-addition constraints (no overflow wrap).
    Power,
}

impl FromRef<Sha256F2Ideal> for Sha256F2Ideal {
    fn from_ref(value: &Sha256F2Ideal) -> Self {
        *value
    }
}

impl Ideal for Sha256F2Ideal {}

impl<F> IdealCheck<DynamicPolynomialF<F>> for Sha256F2Ideal
where
    F: PrimeField,
{
    fn contains(&self, value: &DynamicPolynomialF<F>) -> Result<bool, IdealCheckError> {
        if value.coeffs.is_empty() {
            return Ok(true);
        }
        let cfg = value.coeffs[0].cfg();
        let root = match self {
            Sha256F2Ideal::Cyclotomic => F::one_with_cfg(cfg),
            Sha256F2Ideal::Power => F::zero_with_cfg(cfg),
        };
        IdealOrZero::NonZero(RotationIdeal::<F, 32>::new(root)).contains(value)
    }
}

/// Project an `IdealOrZero<Sha256F2Ideal>` into the verifier field —
/// preserves the variant tag (the only data the ideal carries) and
/// drops the `Zero` arm, which is filtered upstream by the IC's
/// per-constraint loop before any projection runs.
pub fn sha256_f2_project_ideal(ideal: &IdealOrZero<Sha256F2Ideal>) -> Sha256F2Ideal {
    match ideal {
        IdealOrZero::NonZero(i) => *i,
        // The `Zero` arm is filtered upstream; fall back to the
        // cyclotomic ideal so this function is total.
        IdealOrZero::Zero => Sha256F2Ideal::Cyclotomic,
    }
}

// ---------------------------------------------------------------------------
// Column indices.
// ---------------------------------------------------------------------------

pub mod cols {
    // === Public bit-poly prefix (14) ===
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    pub const PA_K: usize = 2;
    pub const PA_M: usize = 3;
    pub const PA_C_W: usize = 4;
    pub const PA_C_T1: usize = 5;
    pub const PA_C_T2: usize = 6;
    pub const PA_C_A: usize = 7;
    pub const PA_C_E: usize = 8;
    pub const PA_C_FF_A: usize = 9;
    pub const PA_C_FF_E: usize = 10;
    pub const S_INIT_PREFIX: usize = 11;
    pub const S_FF: usize = 12;
    pub const S_MSG_INIT: usize = 13;
    pub const NUM_BIN_PUB: usize = 14;

    // === Witness bit-poly suffix (27) ===
    // State (3)
    pub const W_A: usize = NUM_BIN_PUB;
    pub const W_E: usize = NUM_BIN_PUB + 1;
    pub const W_W: usize = NUM_BIN_PUB + 2;
    // Rotation/shift outputs (4) — pinned by C1–C4
    pub const W_SIGMA0: usize = NUM_BIN_PUB + 3;
    pub const W_SIGMA1: usize = NUM_BIN_PUB + 4;
    pub const W_SIG0: usize = NUM_BIN_PUB + 5;
    pub const W_SIG1: usize = NUM_BIN_PUB + 6;
    // SHR auxiliaries (2) — committed because the SHR^r access
    // mechanism is left as a design parameter (cf. doc §3.3).
    pub const W_SHR3_W: usize = NUM_BIN_PUB + 7;
    pub const W_SHR10_W: usize = NUM_BIN_PUB + 8;
    // AND results (3) — pinned by external Hadamard checks (deferred)
    pub const W_UEF: usize = NUM_BIN_PUB + 9;
    pub const W_UNEG_E_G: usize = NUM_BIN_PUB + 10;
    pub const W_MAJ: usize = NUM_BIN_PUB + 11;
    // Round intermediates (2)
    pub const W_T1: usize = NUM_BIN_PUB + 12;
    pub const W_T2: usize = NUM_BIN_PUB + 13;
    // CSA majorities (6)
    pub const W_M_W1: usize = NUM_BIN_PUB + 14;
    pub const W_M_W2: usize = NUM_BIN_PUB + 15;
    pub const W_M_T1_1: usize = NUM_BIN_PUB + 16;
    pub const W_M_T1_2: usize = NUM_BIN_PUB + 17;
    pub const W_M_T1_3: usize = NUM_BIN_PUB + 18;
    pub const W_M_T1_4: usize = NUM_BIN_PUB + 19;
    // Binius carries (7)
    pub const W_C_W: usize = NUM_BIN_PUB + 20;
    pub const W_C_T1: usize = NUM_BIN_PUB + 21;
    pub const W_C_T2: usize = NUM_BIN_PUB + 22;
    pub const W_C_A: usize = NUM_BIN_PUB + 23;
    pub const W_C_E: usize = NUM_BIN_PUB + 24;
    pub const W_C_FF_A: usize = NUM_BIN_PUB + 25;
    pub const W_C_FF_E: usize = NUM_BIN_PUB + 26;

    pub const NUM_BIN_WIT: usize = 27;
    pub const NUM_BIN: usize = NUM_BIN_PUB + NUM_BIN_WIT; // 41

    // === Chained-compression layout ===
    pub const NUM_COMPRESSIONS: usize = 7;
    pub const ROWS_PER_COMP: usize = 68;
    pub const ROUNDS_PER_COMP: usize = 64;
    pub const ACTIVE_ROWS: usize = NUM_COMPRESSIONS * ROWS_PER_COMP + 4;
    /// `2^MIN_NUM_VARS >= ACTIVE_ROWS`: 7·68+4 = 480 ≤ 512 = 2^9.
    pub const MIN_NUM_VARS: usize = 9;
}

// ---------------------------------------------------------------------------
// The UAIR.
// ---------------------------------------------------------------------------

/// SHA-256 compression UAIR over `F_2[X]`. See the module-level doc
/// for the column layout, constraint families, and the scope of
/// in-circuit checks vs. deferred external Hadamard discharge.
#[derive(Clone, Debug)]
pub struct Sha256F2Uair<R>(PhantomData<R>);

impl<R> Uair for Sha256F2Uair<R>
where
    R: ConstSemiring + 'static,
{
    type Ideal = Sha256F2Ideal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        // All public cols are bit-polys (the F_2 prove path requires
        // the int lane to be empty).
        let total = TotalColumnLayout::new(cols::NUM_BIN, 0, 0);
        let public = PublicColumnLayout::new(cols::NUM_BIN_PUB, 0, 0);

        // Shifts — listed in (source_col, shift_amount) ascending
        // order so the `down.binary_poly` slot indices are stable.
        let shifts: Vec<ShiftSpec> = vec![
            // W_A: shift 4 for a[t+1] in C8, C10.
            ShiftSpec::new(cols::W_A, 4),
            // W_E: shift 4 for e[t+1] in C9, C11.
            ShiftSpec::new(cols::W_E, 4),
            // W_W: 1, 3, 9, 14, 16 across C3, C4, C5, C6.
            ShiftSpec::new(cols::W_W, 1),
            ShiftSpec::new(cols::W_W, 3),
            ShiftSpec::new(cols::W_W, 9),
            ShiftSpec::new(cols::W_W, 14),
            ShiftSpec::new(cols::W_W, 16),
            // W_SIGMA1: shift 3 for Σ_1(e[t]) in C6.
            ShiftSpec::new(cols::W_SIGMA1, 3),
            // W_SIG0: shift 1 for σ_0(W[t-15]) in C5; shift 1 for
            // its C3 LHS pin.
            ShiftSpec::new(cols::W_SIG0, 1),
            // W_SIG1: shift 14 for σ_1(W[t-2]) in C5; shift 14 for
            // its C4 LHS pin.
            ShiftSpec::new(cols::W_SIG1, 14),
            // W_SHR3_W: shift 1 for SHR^3(W[t-15]) on C3 RHS.
            ShiftSpec::new(cols::W_SHR3_W, 1),
            // W_SHR10_W: shift 14 for SHR^10(W[t-2]) on C4 RHS.
            ShiftSpec::new(cols::W_SHR10_W, 14),
            // W_UEF, W_UNEG_E_G: shift 3 for Ch[t] = u_ef + u_neg in C6.
            ShiftSpec::new(cols::W_UEF, 3),
            ShiftSpec::new(cols::W_UNEG_E_G, 3),
            // W_T1: shift 3 for T_1[t] in C8, C9.
            ShiftSpec::new(cols::W_T1, 3),
            // W_T2: shift 3 for T_2[t] in C8.
            ShiftSpec::new(cols::W_T2, 3),
            // PA_K: shift 3 for K[t] in C6.
            ShiftSpec::new(cols::PA_K, 3),
        ];

        UairSignature::new(total, public, shifts, vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let bp = up.binary_poly;

        // Up-row columns (public + witness).
        let pa_a = &bp[cols::PA_A];
        let pa_e = &bp[cols::PA_E];
        let pa_m = &bp[cols::PA_M];
        let pa_c_w = &bp[cols::PA_C_W];
        let pa_c_t1 = &bp[cols::PA_C_T1];
        let pa_c_t2 = &bp[cols::PA_C_T2];
        let pa_c_a = &bp[cols::PA_C_A];
        let pa_c_e = &bp[cols::PA_C_E];
        let pa_c_ff_a = &bp[cols::PA_C_FF_A];
        let pa_c_ff_e = &bp[cols::PA_C_FF_E];
        let s_init_prefix = &bp[cols::S_INIT_PREFIX];
        let s_msg_init = &bp[cols::S_MSG_INIT];

        let w_a = &bp[cols::W_A];
        let w_e = &bp[cols::W_E];
        let w_w = &bp[cols::W_W];
        let w_sigma0 = &bp[cols::W_SIGMA0];
        let w_sigma1 = &bp[cols::W_SIGMA1];
        let w_t2 = &bp[cols::W_T2];
        let w_maj = &bp[cols::W_MAJ];
        let w_m_w1 = &bp[cols::W_M_W1];
        let w_m_w2 = &bp[cols::W_M_W2];
        let w_m_t1_1 = &bp[cols::W_M_T1_1];
        let w_m_t1_2 = &bp[cols::W_M_T1_2];
        let w_m_t1_3 = &bp[cols::W_M_T1_3];
        let w_m_t1_4 = &bp[cols::W_M_T1_4];
        let w_c_w = &bp[cols::W_C_W];
        let w_c_t1 = &bp[cols::W_C_T1];
        let w_c_t2 = &bp[cols::W_C_T2];
        let w_c_a = &bp[cols::W_C_A];
        let w_c_e = &bp[cols::W_C_E];
        let w_c_ff_a = &bp[cols::W_C_FF_A];
        let w_c_ff_e = &bp[cols::W_C_FF_E];

        // Down-row shifts. Slot order matches the ShiftSpec list in
        // `signature()` after sort by (source_col, shift_amount).
        // Source-col order: PA_K(2), W_A(14), W_E(15), W_W(16),
        // W_SIGMA1(18), W_SIG0(19), W_SIG1(20), W_SHR3_W(21),
        // W_SHR10_W(22), W_UEF(23), W_UNEG_E_G(24), W_T1(25), W_T2(26).
        let down_pa_k_3 = &down.binary_poly[0];
        let down_w_a_4 = &down.binary_poly[1];
        let down_w_e_4 = &down.binary_poly[2];
        let down_w_w_1 = &down.binary_poly[3];
        let down_w_w_3 = &down.binary_poly[4];
        let down_w_w_9 = &down.binary_poly[5];
        let down_w_w_14 = &down.binary_poly[6];
        let down_w_w_16 = &down.binary_poly[7];
        let down_w_sigma1_3 = &down.binary_poly[8];
        let down_w_sig0_1 = &down.binary_poly[9];
        let down_w_sig1_14 = &down.binary_poly[10];
        let down_w_shr3_w_1 = &down.binary_poly[11];
        let down_w_shr10_w_14 = &down.binary_poly[12];
        let down_w_uef_3 = &down.binary_poly[13];
        let down_w_uneg_e_g_3 = &down.binary_poly[14];
        let down_w_t1_3 = &down.binary_poly[15];
        let down_w_t2_3 = &down.binary_poly[16];

        // Ideals. `ideal_rot = (X^32 − 1)` for the rotation/shift
        // constraints (C1–C4). `ideal_pow = (X^32)` for the modular-
        // addition constraints (C5–C11): under this ideal the
        // compensator columns are honest-zero and the carry word
        // becomes a non-cyclic `ShiftR(1)` of the inputs — see
        // `protocol/src/f2_gkr_plan.md`. Phase 2a switches the
        // compensator-less feed-forward additions C10/C11 first;
        // C5–C9 still use `ideal_rot` until the rest of the Phase 2a
        // rollout swaps them too.
        let ideal_rot = ideal_from_ref(&Sha256F2Ideal::Cyclotomic);
        let ideal_pow = ideal_from_ref(&Sha256F2Ideal::Power);

        // Scalars.
        // rho polynomials — bit positions per Lemma 2.2 / 2.3.
        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]); // X^30 + X^19 + X^10
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]); //  X^26 + X^21 + X^7
        let rho_lsig0 = rho_poly::<R>(&[14, 25]); //    X^25 + X^14
        let rho_lsig1 = rho_poly::<R>(&[13, 15]); //    X^15 + X^13
        let x_scalar = monomial_x::<R>(); //            X (for X·c_*, X·m_*)

        // -----------------------------------------------------------
        // (A) Rotation/shift identities — C1, C2, C3, C4.
        // -----------------------------------------------------------

        // C1:  W_a · rho_sig0 − W_SIGMA0 ∈ (X^32 − 1).
        b.assert_in_ideal(
            mbs(w_a, &rho_sig0).expect("W_a · rho_sig0 overflow") - w_sigma0,
            &ideal_rot,
        );

        // C2:  W_e · rho_sig1 − W_SIGMA1 ∈ (X^32 − 1).
        b.assert_in_ideal(
            mbs(w_e, &rho_sig1).expect("W_e · rho_sig1 overflow") - w_sigma1,
            &ideal_rot,
        );

        // C3 (anchor t = t' − 1):
        //   W_W^{↓1} · rho_lsig0 + SHR^3(W_W)^{↓1} − W_SIG0^{↓1} ∈ (X^32 − 1).
        b.assert_in_ideal(
            mbs(down_w_w_1, &rho_lsig0).expect("W_W^↓1 · rho_lsig0 overflow")
                + down_w_shr3_w_1
                - down_w_sig0_1,
            &ideal_rot,
        );

        // C4 (anchor t = t' − 14):
        //   W_W^{↓14} · rho_lsig1 + SHR^10(W_W)^{↓14} − W_SIG1^{↓14} ∈ (X^32 − 1).
        b.assert_in_ideal(
            mbs(down_w_w_14, &rho_lsig1).expect("W_W^↓14 · rho_lsig1 overflow")
                + down_w_shr10_w_14
                - down_w_sig1_14,
            &ideal_rot,
        );

        // -----------------------------------------------------------
        // (B) Modular-addition LinSum constraints — C5..C11.
        //
        // Each is the closed-form F_2[X]-linear identity from doc
        // §3.6: target − inputs − Σ X·m_j − X·c_* + compensator ∈
        // (X^32 − 1). All signs collapse to + over F_2 anyway; we keep
        // the `−` form for parity with the integer-pipeline UAIR.
        // -----------------------------------------------------------

        // C5 (4-input message schedule, anchor t = t' − 16):
        //   W_W^{↓16} − W_W − W_SIG0^{↓1} − W_W^{↓9} − W_SIG1^{↓14}
        //              − X·m_W1 − X·m_W2 − X·c_W + PA_C_W ∈ (X^32−1).
        let lin_w = down_w_w_16.clone()
            - w_w
            - down_w_sig0_1
            - down_w_w_9
            - down_w_sig1_14
            - &mbs(w_m_w1, &x_scalar).expect("X · m_W1 overflow")
            - &mbs(w_m_w2, &x_scalar).expect("X · m_W2 overflow")
            - &mbs(w_c_w, &x_scalar).expect("X · c_W overflow")
            + pa_c_w;
        b.assert_in_ideal(lin_w, &ideal_pow);

        // C6 (6-input T_1, anchor t = t' − 3): inputs are h[t'] (= e[t]
        // via the shift-register), Σ_1(e[t]), u_ef[t'], u_{¬e,g}[t'],
        // K[t'], W[t']; carries are m_T1_{1..4} and c_T1.
        let lin_t1 = down_w_t1_3.clone()
            - w_e
            - down_w_sigma1_3
            - down_w_uef_3
            - down_w_uneg_e_g_3
            - down_pa_k_3
            - down_w_w_3
            - &mbs(w_m_t1_1, &x_scalar).expect("X · m_T1_1 overflow")
            - &mbs(w_m_t1_2, &x_scalar).expect("X · m_T1_2 overflow")
            - &mbs(w_m_t1_3, &x_scalar).expect("X · m_T1_3 overflow")
            - &mbs(w_m_t1_4, &x_scalar).expect("X · m_T1_4 overflow")
            - &mbs(w_c_t1, &x_scalar).expect("X · c_T1 overflow")
            + pa_c_t1;
        b.assert_in_ideal(lin_t1, &ideal_pow);

        // C7 (2-input T_2, row-local):
        //   W_T2 − W_SIGMA0 − W_MAJ − X·c_T2 + PA_C_T2 ∈ (X^32 − 1).
        let lin_t2 = w_t2.clone()
            - w_sigma0
            - w_maj
            - &mbs(w_c_t2, &x_scalar).expect("X · c_T2 overflow")
            + pa_c_t2;
        b.assert_in_ideal(lin_t2, &ideal_pow);

        // C8 (a' = T_1 + T_2, anchor t = t' − 3, target a[t'+1] at
        // up.W_a^{↓4}):
        //   W_A^{↓4} − W_T1^{↓3} − W_T2^{↓3} − X·c_A + PA_C_A ∈ (X^32 − 1).
        let lin_a = down_w_a_4.clone()
            - down_w_t1_3
            - down_w_t2_3
            - &mbs(w_c_a, &x_scalar).expect("X · c_A overflow")
            + pa_c_a;
        b.assert_in_ideal(lin_a, &ideal_pow);

        // C9 (e' = d + T_1, anchor t = t' − 3): d[t'] = a[t'−4] is
        // up.W_a; T_1[t'] is down.W_T1^{↓3}.
        //   W_E^{↓4} − W_A − W_T1^{↓3} − X·c_E + PA_C_E ∈ (X^32 − 1).
        let lin_e = down_w_e_4.clone()
            - w_a
            - down_w_t1_3
            - &mbs(w_c_e, &x_scalar).expect("X · c_E overflow")
            + pa_c_e;
        b.assert_in_ideal(lin_e, &ideal_pow);

        // C10 (feed-forward a, junction-window): H_{i+1}[a] = H_i[a] +
        // internal_final[a]. Phase-2a: uses `(X^32)` so the overflow
        // bit `c_FF_A_31` falls off cleanly at bit 32 and PA_C_FF_A is
        // honest-zero on every row. The `+ pa_c_ff_a` term stays in
        // the expression for now (it XORs in zero, hence a no-op);
        // removing the public column entirely happens once all C5–C11
        // share the `Power` ideal.
        //   W_A^{↓4} − W_A − PA_A − X·c_FF_A + PA_C_FF_A ∈ (X^32).
        let lin_ff_a = down_w_a_4.clone()
            - w_a
            - pa_a
            - &mbs(w_c_ff_a, &x_scalar).expect("X · c_FF_A overflow")
            + pa_c_ff_a;
        b.assert_in_ideal(lin_ff_a, &ideal_pow);

        // C11 (feed-forward e):
        //   W_E^{↓4} − W_E − PA_E − X·c_FF_E + PA_C_FF_E ∈ (X^32 − 1).
        let lin_ff_e = down_w_e_4.clone()
            - w_e
            - pa_e
            - &mbs(w_c_ff_e, &x_scalar).expect("X · c_FF_E overflow")
            + pa_c_ff_e;
        b.assert_in_ideal(lin_ff_e, &ideal_pow);

        // -----------------------------------------------------------
        // (D) Boundary pinning — C16, C17, C18.
        // -----------------------------------------------------------

        // C16: S_INIT_PREFIX · (W_A − PA_A) = 0.
        b.assert_zero(s_init_prefix.clone() * &(w_a.clone() - pa_a));

        // C17: S_INIT_PREFIX · (W_E − PA_E) = 0.
        b.assert_zero(s_init_prefix.clone() * &(w_e.clone() - pa_e));

        // C18: S_MSG_INIT · (W_W − PA_M) = 0.
        b.assert_zero(s_msg_init.clone() * &(w_w.clone() - pa_m));
    }
}

// ---------------------------------------------------------------------------
// Scalar projection: lift a `Self::Scalar = DensePolynomial<R, 32>` to
// `DynamicPolynomialF<BinaryFieldGF192>`.
//
// We treat each `R` coefficient as `0` or `1` in F_2 (using `is_zero`
// since the trace generator only ever puts 0/1 there for rho_sig*,
// rho_lsig*, X), then embed F_2 ⊂ GF(2^192) trivially.
// ---------------------------------------------------------------------------

/// Project a scalar polynomial in `R[X]` whose coefficients are 0 or 1
/// (the only values the F_2[X] UAIR uses) to a
/// `DynamicPolynomialF<BinaryFieldGF192>` for the F_2 IC pipeline.
pub fn sha256_f2_project_scalar<R>(
    scalar: &DensePolynomial<R, 32>,
) -> DynamicPolynomialF<zinc_poly::univariate::binary_gf192::BinaryFieldGF192>
where
    R: Semiring + num_traits::Zero,
{
    use zinc_poly::univariate::binary_gf192::BinaryFieldGF192;
    let cfg = ();
    let coeffs: Vec<BinaryFieldGF192> = scalar
        .coeffs
        .iter()
        .map(|c| {
            if num_traits::Zero::is_zero(c) {
                BinaryFieldGF192::zero_with_cfg(&cfg)
            } else {
                BinaryFieldGF192::one_with_cfg(&cfg)
            }
        })
        .collect();
    // Use `new_trimmed` to strip trailing zero coefficients. This
    // shrinks the IC's per-row schoolbook by 16× for X-mults
    // (`x_scalar = [0, 1, 0, ..., 0]` of length 32 trims to length 2)
    // and by 2% for ρ_σ0/ρ_σ1 (top bit is zero). Without the trim,
    // `DynamicPolynomialF`'s `mul_schoolbook` runs the full O(len_a ·
    // len_b) GF(2^192) multiplication loop, paying for the trailing
    // zero coefficients with full-cost field mults each iteration.
    DynamicPolynomialF::new_trimmed(coeffs)
}

// ---------------------------------------------------------------------------
// Scalar helpers.
// ---------------------------------------------------------------------------

fn rho_poly<R: ConstSemiring>(positions: &[usize]) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    for &p in positions {
        debug_assert!(p < 32);
        coeffs[p] = R::ONE;
    }
    DensePolynomial::<R, 32>::new(coeffs)
}

fn monomial_x<R: ConstSemiring>() -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[1] = R::ONE;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// SHA-256 reference helpers and round-constant table.
// ---------------------------------------------------------------------------

/// FIPS 180-4 §4.2.2 round constants `K[0..64]`.
pub const K_CANONICAL: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[inline]
fn rotr(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}
#[inline]
fn big_sigma0(x: u32) -> u32 {
    rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
}
#[inline]
fn big_sigma1(x: u32) -> u32 {
    rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
}
#[inline]
fn small_sigma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
}
#[inline]
fn small_sigma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
}
#[inline]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ ((!x) & z)
}
#[inline]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

// ---------------------------------------------------------------------------
// CSA + Binius-adder honest-witness derivations.
// ---------------------------------------------------------------------------

/// Compute the Binius full-adder carry word for two 32-bit summands.
/// `c[i] = (a[i] ∧ b[i]) ⊕ (c[i−1] ∧ (a[i] ⊕ b[i]))` for `i ∈ [0, 32)`,
/// with `c[−1] = 0`. The returned 32-bit word has the carry at
/// position `i` in bit `i`.
fn binius_carry(a: u32, b: u32) -> u32 {
    let mut c = 0u32;
    for i in 0..32 {
        let ai = (a >> i) & 1;
        let bi = (b >> i) & 1;
        let cprev = if i == 0 { 0 } else { (c >> (i - 1)) & 1 };
        let ci = (ai & bi) ^ (cprev & (ai ^ bi));
        c |= ci << i;
    }
    c
}

/// CSA layer: from `(a, b, c)` derive the next layer's `(sum, carry)`
/// pair where `sum_i = a_i ⊕ b_i ⊕ c_i` and `carry_i = maj(a_i, b_i,
/// c_i)`. The maj word `carry` is the CSA majority witness `m_j`; the
/// sum is the running XOR fold of the inputs.
fn csa_step(a: u32, b: u32, c: u32) -> (u32, u32) {
    let sum = a ^ b ^ c;
    let maj_word = maj(a, b, c);
    (sum, maj_word)
}

// ---------------------------------------------------------------------------
// GenerateRandomTrace.
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for Sha256F2Uair<R>
where
    R: ConstSemiring + 'static,
{
    type PolyCoeff = BinaryPoly<32>;
    type Int = BinaryPoly<32>;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, BinaryPoly<32>, BinaryPoly<32>, 32> {
        let n = 1usize << num_vars;
        assert!(
            num_vars >= cols::MIN_NUM_VARS,
            "trace too small for {} chained compressions: need num_vars ≥ {}, got {}",
            cols::NUM_COMPRESSIONS,
            cols::MIN_NUM_VARS,
            num_vars,
        );

        let big_n = cols::NUM_COMPRESSIONS;
        let rpc = cols::ROWS_PER_COMP;
        let rounds = cols::ROUNDS_PER_COMP;

        // ---- u32-typed trace buffers ----------------------------------
        let mut a_vals = vec![0u32; n];
        let mut e_vals = vec![0u32; n];
        let mut w_vals = vec![0u32; n];
        let mut k_vals = vec![0u32; n];
        let mut pa_a_vals = vec![0u32; n];
        let mut pa_e_vals = vec![0u32; n];
        let mut pa_m_vals = vec![0u32; n];

        // Per-constraint CSA majorities + final Binius carries.
        let mut m_w1_vals = vec![0u32; n];
        let mut m_w2_vals = vec![0u32; n];
        let mut c_w_vals = vec![0u32; n];
        let mut m_t1_1_vals = vec![0u32; n];
        let mut m_t1_2_vals = vec![0u32; n];
        let mut m_t1_3_vals = vec![0u32; n];
        let mut m_t1_4_vals = vec![0u32; n];
        let mut c_t1_vals = vec![0u32; n];
        let mut c_t2_vals = vec![0u32; n];
        let mut c_a_vals = vec![0u32; n];
        let mut c_e_vals = vec![0u32; n];
        let mut c_ff_a_vals = vec![0u32; n];
        let mut c_ff_e_vals = vec![0u32; n];

        // T_1[t'], T_2[t'] at trace anchor row k = t' − 3.
        let mut t1_vals = vec![0u32; n];
        let mut t2_vals = vec![0u32; n];

        // Initial state H_0 — random 8-word digest. Stored under the
        // shift-register layout (d, c, b, a) / (h, g, f, e) at row
        // offsets 0..4 of each init prefix.
        let mut h_a: [u32; 4] = [
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
        ];
        let mut h_e: [u32; 4] = [
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
        ];

        for i in 0..big_n {
            let start = i * rpc;

            // 1) Init prefix.
            for j in 0..4 {
                a_vals[start + j] = h_a[j];
                e_vals[start + j] = h_e[j];
                pa_a_vals[start + j] = h_a[j];
                pa_e_vals[start + j] = h_e[j];
            }

            // 2) Message block (16 seeds, then 48 schedule-derived).
            for j in 0..16 {
                let m = rng.next_u32();
                w_vals[start + j] = m;
                pa_m_vals[start + j] = m;
            }
            for j in 16..rpc {
                let t = start + j;
                let w_15 = w_vals[t - 15];
                let w_2 = w_vals[t - 2];
                let lsig0 = small_sigma0(w_15);
                let lsig1 = small_sigma1(w_2);
                let sum = (w_vals[t - 16] as u64)
                    + (lsig0 as u64)
                    + (w_vals[t - 7] as u64)
                    + (lsig1 as u64);
                w_vals[t] = sum as u32;
                // CSA reduction of 4 inputs to 2, then Binius adder.
                // The 4 inputs at anchor k = t − 16 (= j − 16 within
                // this compression):
                let k = t - 16;
                let x1 = w_vals[t - 16];
                let x2 = lsig0;
                let x3 = w_vals[t - 7];
                let x4 = lsig1;
                let (s1, m1) = csa_step(x1, x2, x3);
                // After layer 1: x1+x2+x3 = s1 + 2·m1 (integer).
                // Layer 2: fold (s1, 2·m1, x4) → (s2, m2). The "2·m1"
                // contribution to the integer sum is `m1 << 1` in u32
                // (Fshift's X·m1; bit 31 of m1 is dropped). At the
                // F_2[X]/(X^32-1) (Frot) level the constraint absorbs
                // either convention.
                let (s2, m2) = csa_step(s1, m1 << 1, x4);
                // After layer 2: total = s2 + 2·m2 (integer).
                // Final Binius adder on (s2, 2·m2):
                //   s2 + (m2<<1) = (s2 ^ (m2<<1)) ^ (c_W << 1)   in u32
                let c_w = binius_carry(s2, m2 << 1);
                let s_final = s2 ^ (m2 << 1) ^ (c_w << 1);
                debug_assert_eq!(s_final, w_vals[t]);
                m_w1_vals[k] = m1;
                m_w2_vals[k] = m2;
                // Mask bit 31: under `(X^32)` the carry word's top
                // bit is unconstrained (it falls off at bit 32 of the
                // LHS, absorbed by the ideal). Zeroing it makes
                // `c_w = ShiftR(1)(target ⊕ a ⊕ b ⊕ comp)` an exact
                // equality — required for Phase-2a virtualization.
                c_w_vals[k] = c_w & 0x7FFF_FFFF;
            }

            // 3) Round constants.
            for j in 0..rounds {
                k_vals[start + 3 + j] = K_CANONICAL[j];
            }

            // 4) Round updates (64 rounds, anchor k = start..start+64,
            //    target a[k+4]/e[k+4]).
            for j in 0..rounds {
                let k = start + j;
                let t = k + 3;

                let a_t = a_vals[k + 3];
                let a_t1 = a_vals[k + 2];
                let a_t2 = a_vals[k + 1];
                let e_t = e_vals[k + 3];
                let e_t1 = e_vals[k + 2];
                let e_t2 = e_vals[k + 1];

                let sig0_a = big_sigma0(a_t);
                let sig1_e = big_sigma1(e_t);
                let ch_t = ch(e_t, e_t1, e_t2);
                let maj_t = maj(a_t, a_t1, a_t2);

                // T_1 = h + Σ_1(e) + Ch + K + W (6 inputs).
                // Ch decomposes as u_ef + u_neg_e_g (disjoint bit support).
                let h_val = e_vals[k]; // h = e[t-3]
                let u_ef = e_t & e_t1;
                let u_neg = (!e_t) & e_t2;
                debug_assert_eq!(u_ef ^ u_neg, ch_t);

                // CSA tree for T_1: fold (h, Σ_1, u_ef, u_neg, K, W) →
                // (sum, carry) via 4 CSA layers, then one Binius adder.
                let x = [h_val, sig1_e, u_ef, u_neg, k_vals[t], w_vals[t]];
                // Layer 1: (x0, x1, x2) → (s, m1).
                let (s, m1) = csa_step(x[0], x[1], x[2]);
                // Layer 2: (s, m1<<1, x3) → (s, m2).
                let (s, m2) = csa_step(s, m1 << 1, x[3]);
                // Layer 3: (s, m2<<1, x4) → (s, m3).
                let (s, m3) = csa_step(s, m2 << 1, x[4]);
                // Layer 4: (s, m3<<1, x5) → (s, m4).
                let (s, m4) = csa_step(s, m3 << 1, x[5]);
                // Final Binius adder on (s, m4<<1):
                //   t1 = s + (m4<<1) = (s ^ (m4<<1)) ^ (c_t1 << 1)   in u32
                let c_t1 = binius_carry(s, m4 << 1);
                let t1 = s ^ (m4 << 1) ^ (c_t1 << 1);
                let t1_sum: u64 = (h_val as u64)
                    + (sig1_e as u64)
                    + (u_ef as u64)
                    + (u_neg as u64)
                    + (k_vals[t] as u64)
                    + (w_vals[t] as u64);
                debug_assert_eq!(t1 as u64, t1_sum & 0xFFFF_FFFF);

                // W_T1 is row-local on the SHA round index (= t = k+3),
                // so C8 / C9 (anchored at t' − 3 = k) can read
                // W_T1^↓3[k] = W_T1[k+3] = T_1[round t' = k+3]. The
                // CSA majorities and final carry stay at the C6 anchor
                // row k (C6 reads them without shifts).
                t1_vals[t] = t1;
                m_t1_1_vals[k] = m1;
                m_t1_2_vals[k] = m2;
                m_t1_3_vals[k] = m3;
                m_t1_4_vals[k] = m4;
                c_t1_vals[k] = c_t1 & 0x7FFF_FFFF;

                // T_2 = Σ_0(a) + Maj (2 inputs). W_T2 and W_C_T2 are
                // both row-local at the SHA round t (= k+3): C7 is the
                // "every update-active row t" form, with no shifts on
                // any operand, so the entire row at t = k+3 must
                // contain T_2's components.
                let t2 = sig0_a.wrapping_add(maj_t);
                let c_t2_val = binius_carry(sig0_a, maj_t);
                debug_assert_eq!(t2, sig0_a ^ maj_t ^ (c_t2_val << 1));
                t2_vals[t] = t2;
                c_t2_vals[t] = c_t2_val & 0x7FFF_FFFF;

                // a' = T_1 + T_2.
                let a_next = t1.wrapping_add(t2);
                let c_a_val = binius_carry(t1, t2);
                debug_assert_eq!(a_next, t1 ^ t2 ^ (c_a_val << 1));
                a_vals[k + 4] = a_next;
                c_a_vals[k] = c_a_val & 0x7FFF_FFFF;

                // e' = d + T_1 = a[t-3] + T_1.
                let d_val = a_vals[k];
                let e_next = d_val.wrapping_add(t1);
                let c_e_val = binius_carry(d_val, t1);
                debug_assert_eq!(e_next, d_val ^ t1 ^ (c_e_val << 1));
                e_vals[k + 4] = e_next;
                c_e_vals[k] = c_e_val & 0x7FFF_FFFF;
            }

            // 5) Feed-forward: H_{i+1} = internal_final + H_i.
            let mut h_a_next = [0u32; 4];
            let mut h_e_next = [0u32; 4];
            for j in 0..4 {
                let internal_a = a_vals[start + 64 + j];
                let internal_e = e_vals[start + 64 + j];
                let prior_a = h_a[j];
                let prior_e = h_e[j];
                pa_a_vals[start + 64 + j] = prior_a;
                pa_e_vals[start + 64 + j] = prior_e;

                let next_a = internal_a.wrapping_add(prior_a);
                let next_e = internal_e.wrapping_add(prior_e);
                let c_ff_a_val = binius_carry(internal_a, prior_a);
                let c_ff_e_val = binius_carry(internal_e, prior_e);
                debug_assert_eq!(
                    next_a,
                    internal_a ^ prior_a ^ (c_ff_a_val << 1)
                );
                debug_assert_eq!(
                    next_e,
                    internal_e ^ prior_e ^ (c_ff_e_val << 1)
                );
                h_a_next[j] = next_a;
                h_e_next[j] = next_e;
                c_ff_a_vals[start + 64 + j] = c_ff_a_val & 0x7FFF_FFFF;
                c_ff_e_vals[start + 64 + j] = c_ff_e_val & 0x7FFF_FFFF;
            }
            h_a = h_a_next;
            h_e = h_e_next;
        }

        // 6) H_N output prefix.
        let out_start = big_n * rpc;
        for j in 0..4 {
            a_vals[out_start + j] = h_a[j];
            e_vals[out_start + j] = h_e[j];
            pa_a_vals[out_start + j] = h_a[j];
            pa_e_vals[out_start + j] = h_e[j];
        }

        // ---- Per-row derived columns ---------------------------------
        let sigma0_vals: Vec<u32> = a_vals.iter().copied().map(big_sigma0).collect();
        let sigma1_vals: Vec<u32> = e_vals.iter().copied().map(big_sigma1).collect();
        let sig0_vals: Vec<u32> = w_vals.iter().copied().map(small_sigma0).collect();
        let sig1_vals: Vec<u32> = w_vals.iter().copied().map(small_sigma1).collect();
        let shr3_w_vals: Vec<u32> = w_vals.iter().copied().map(|w| w >> 3).collect();
        let shr10_w_vals: Vec<u32> = w_vals.iter().copied().map(|w| w >> 10).collect();

        // Per-row Ch / Maj operands. We populate on every row (not
        // just SHA-active ones) to keep the trace honest; off-active
        // rows don't participate in any constraint.
        let u_ef_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 1 { e_vals[t] & e_vals[t - 1] } else { 0 })
            .collect();
        let u_neg_e_g_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { (!e_vals[t]) & e_vals[t - 2] } else { 0 })
            .collect();
        let maj_vals: Vec<u32> = (0..n)
            .map(|t| {
                if t >= 2 {
                    maj(a_vals[t], a_vals[t - 1], a_vals[t - 2])
                } else {
                    0
                }
            })
            .collect();

        // ---- Compensator columns (PA_C_*) ----------------------------
        // For each linear constraint Cℓ, the compensator carries the
        // F_2[X] residue of the constraint's inner expression mod the
        // constraint's ideal on every row. On active rows with honest
        // witnesses the residue is zero; on inactive boundary rows it
        // absorbs whatever the row's contents happen to imply, making
        // the constraint hold unconditionally.
        //
        // Phase 2a: C5–C11 use `Sha256F2Ideal::Power = (X^32)`, so the
        // `X · _` terms (carries, majorities) shift NON-cyclically:
        // `shiftl1(p) = p << 1`, with `p_31` falling off into bit 32
        // (absorbed by the ideal's slack). The cyclic `rotl1` is no
        // longer used here; kept-and-unused only to flag the change in
        // diffs.
        let _rotl1_legacy = |x: u32| -> u32 { (x << 1) | (x >> 31) };
        let shiftl1 = |x: u32| -> u32 { x << 1 };
        let xor4 = |a: u32, b: u32, c: u32, d: u32| -> u32 { a ^ b ^ c ^ d };

        // C5 inner at row k (= 0 on schedule-active rows when honest):
        //   inner_W = W[k+16] ^ W[k] ^ σ_0(W)[k+1] ^ W[k+9] ^
        //             σ_1(W)[k+14] ^ rotl1(m_W1) ^ rotl1(m_W2) ^
        //             rotl1(c_W).
        // PA_C_W = inner_W on every row (zero on active rows under the
        // honest CSA derivation, nonzero on inactive rows where the
        // free witness columns determine the residue).
        let load = |arr: &[u32], idx: usize| -> u32 { if idx < n { arr[idx] } else { 0 } };
        let pa_c_w_vals: Vec<u32> = (0..n)
            .map(|k| {
                let inner = xor4(
                    load(&w_vals, k + 16),
                    load(&w_vals, k),
                    load(&sig0_vals, k + 1),
                    load(&w_vals, k + 9),
                ) ^ load(&sig1_vals, k + 14)
                    ^ shiftl1(load(&m_w1_vals, k))
                    ^ shiftl1(load(&m_w2_vals, k))
                    ^ shiftl1(load(&c_w_vals, k));
                inner
            })
            .collect();

        // C6 inner at row k:
        //   inner_T1 = W_T1[k+3] ^ W_e[k] ^ Σ_1(e)[k+3] ^ u_ef[k+3] ^
        //              u_neg[k+3] ^ K[k+3] ^ W[k+3] ^
        //              rotl1(m_T1_1) ^ rotl1(m_T1_2) ^
        //              rotl1(m_T1_3) ^ rotl1(m_T1_4) ^ rotl1(c_T1).
        let pa_c_t1_vals: Vec<u32> = (0..n)
            .map(|k| {
                load(&t1_vals, k + 3)
                    ^ load(&e_vals, k)
                    ^ load(&sigma1_vals, k + 3)
                    ^ load(&u_ef_vals, k + 3)
                    ^ load(&u_neg_e_g_vals, k + 3)
                    ^ load(&k_vals, k + 3)
                    ^ load(&w_vals, k + 3)
                    ^ shiftl1(load(&m_t1_1_vals, k))
                    ^ shiftl1(load(&m_t1_2_vals, k))
                    ^ shiftl1(load(&m_t1_3_vals, k))
                    ^ shiftl1(load(&m_t1_4_vals, k))
                    ^ shiftl1(load(&c_t1_vals, k))
            })
            .collect();

        // C7 inner at row k (row-local, no shifts on inputs):
        //   inner_T2 = W_T2[k] ^ Σ_0(a)[k] ^ Maj[k] ^ rotl1(c_T2).
        let pa_c_t2_vals: Vec<u32> = (0..n)
            .map(|k| {
                load(&t2_vals, k)
                    ^ load(&sigma0_vals, k)
                    ^ load(&maj_vals, k)
                    ^ shiftl1(load(&c_t2_vals, k))
            })
            .collect();

        // C8 inner at row k:
        //   inner_A = W_a[k+4] ^ W_T1[k+3] ^ W_T2[k+3] ^ rotl1(c_A).
        let pa_c_a_vals: Vec<u32> = (0..n)
            .map(|k| {
                load(&a_vals, k + 4)
                    ^ load(&t1_vals, k + 3)
                    ^ load(&t2_vals, k + 3)
                    ^ shiftl1(load(&c_a_vals, k))
            })
            .collect();

        // C9 inner at row k:
        //   inner_E = W_e[k+4] ^ W_a[k] ^ W_T1[k+3] ^ rotl1(c_E).
        let pa_c_e_vals: Vec<u32> = (0..n)
            .map(|k| {
                load(&e_vals, k + 4)
                    ^ load(&a_vals, k)
                    ^ load(&t1_vals, k + 3)
                    ^ shiftl1(load(&c_e_vals, k))
            })
            .collect();

        // C10 under `Sha256F2Ideal::Power` = `(X^32)`.
        let pa_c_ff_a_vals: Vec<u32> = (0..n)
            .map(|k| {
                load(&a_vals, k + 4)
                    ^ load(&a_vals, k)
                    ^ load(&pa_a_vals, k)
                    ^ shiftl1(load(&c_ff_a_vals, k))
            })
            .collect();

        // C11 inner at row k.
        let pa_c_ff_e_vals: Vec<u32> = (0..n)
            .map(|k| {
                load(&e_vals, k + 4)
                    ^ load(&e_vals, k)
                    ^ load(&pa_e_vals, k)
                    ^ shiftl1(load(&c_ff_e_vals, k))
            })
            .collect();

        // ---- Selectors ----------------------------------------------
        // Each selector cell is a bit-poly: 1 ↔ BinaryPoly with bit 0
        // set, 0 ↔ BinaryPoly::zero().
        let mut s_init_prefix = vec![0u32; n];
        for i in 0..=big_n {
            for j in 0..4 {
                s_init_prefix[i * rpc + j] = 1;
            }
        }
        let mut s_ff = vec![0u32; n];
        for i in 0..big_n {
            for j in 0..4 {
                s_ff[i * rpc + 64 + j] = 1;
            }
        }
        let mut s_msg_init = vec![0u32; n];
        for i in 0..big_n {
            for j in 0..16 {
                s_msg_init[i * rpc + j] = 1;
            }
        }

        // ---- Pack u32 → BinaryPoly<32> → MLE ------------------------
        let to_bits = |v: &[u32]| -> Vec<BinaryPoly<32>> {
            v.iter().copied().map(BinaryPoly::<32>::from).collect()
        };
        let to_mle = |col: Vec<BinaryPoly<32>>| -> DenseMultilinearExtension<BinaryPoly<32>> {
            col.into_iter().collect()
        };

        // Order must match `cols::*` indices exactly.
        let binary_poly = vec![
            // -- Public prefix (14) --
            to_mle(to_bits(&pa_a_vals)),     // PA_A
            to_mle(to_bits(&pa_e_vals)),     // PA_E
            to_mle(to_bits(&k_vals)),        // PA_K
            to_mle(to_bits(&pa_m_vals)),     // PA_M
            to_mle(to_bits(&pa_c_w_vals)),   // PA_C_W
            to_mle(to_bits(&pa_c_t1_vals)),  // PA_C_T1
            to_mle(to_bits(&pa_c_t2_vals)),  // PA_C_T2
            to_mle(to_bits(&pa_c_a_vals)),   // PA_C_A
            to_mle(to_bits(&pa_c_e_vals)),   // PA_C_E
            to_mle(to_bits(&pa_c_ff_a_vals)),// PA_C_FF_A
            to_mle(to_bits(&pa_c_ff_e_vals)),// PA_C_FF_E
            to_mle(to_bits(&s_init_prefix)), // S_INIT_PREFIX
            to_mle(to_bits(&s_ff)),          // S_FF
            to_mle(to_bits(&s_msg_init)),    // S_MSG_INIT
            // -- Witness suffix (27) --
            to_mle(to_bits(&a_vals)),        // W_A
            to_mle(to_bits(&e_vals)),        // W_E
            to_mle(to_bits(&w_vals)),        // W_W
            to_mle(to_bits(&sigma0_vals)),   // W_SIGMA0
            to_mle(to_bits(&sigma1_vals)),   // W_SIGMA1
            to_mle(to_bits(&sig0_vals)),     // W_SIG0
            to_mle(to_bits(&sig1_vals)),     // W_SIG1
            to_mle(to_bits(&shr3_w_vals)),   // W_SHR3_W
            to_mle(to_bits(&shr10_w_vals)),  // W_SHR10_W
            to_mle(to_bits(&u_ef_vals)),     // W_UEF
            to_mle(to_bits(&u_neg_e_g_vals)),// W_UNEG_E_G
            to_mle(to_bits(&maj_vals)),      // W_MAJ
            to_mle(to_bits(&t1_vals)),       // W_T1
            to_mle(to_bits(&t2_vals)),       // W_T2
            to_mle(to_bits(&m_w1_vals)),     // W_M_W1
            to_mle(to_bits(&m_w2_vals)),     // W_M_W2
            to_mle(to_bits(&m_t1_1_vals)),   // W_M_T1_1
            to_mle(to_bits(&m_t1_2_vals)),   // W_M_T1_2
            to_mle(to_bits(&m_t1_3_vals)),   // W_M_T1_3
            to_mle(to_bits(&m_t1_4_vals)),   // W_M_T1_4
            to_mle(to_bits(&c_w_vals)),      // W_C_W
            to_mle(to_bits(&c_t1_vals)),     // W_C_T1
            to_mle(to_bits(&c_t2_vals)),     // W_C_T2
            to_mle(to_bits(&c_a_vals)),      // W_C_A
            to_mle(to_bits(&c_e_vals)),      // W_C_E
            to_mle(to_bits(&c_ff_a_vals)),   // W_C_FF_A
            to_mle(to_bits(&c_ff_e_vals)),   // W_C_FF_E
        ];

        debug_assert_eq!(binary_poly.len(), cols::NUM_BIN);

        UairTrace {
            binary_poly: binary_poly.into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_uair::degree_counter::{count_effective_max_degree, count_max_degree};

    #[test]
    fn k_canonical_sha256_empty_string_digest() {
        // SHA-256 H_0 (FIPS 180-4 §5.3.3).
        let h_in: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ];
        let mut m = [0u32; 16];
        m[0] = 0x80000000;

        let mut w = [0u32; 64];
        w[..16].copy_from_slice(&m);
        for t in 16..64 {
            w[t] = w[t - 16]
                .wrapping_add(small_sigma0(w[t - 15]))
                .wrapping_add(w[t - 7])
                .wrapping_add(small_sigma1(w[t - 2]));
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = h_in;
        for t in 0..64 {
            let t1 = h
                .wrapping_add(big_sigma1(e))
                .wrapping_add(ch(e, f, g))
                .wrapping_add(K_CANONICAL[t])
                .wrapping_add(w[t]);
            let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
            h = g; g = f; f = e;
            e = d.wrapping_add(t1);
            d = c; c = b; b = a;
            a = t1.wrapping_add(t2);
        }

        let h_out: [u32; 8] = [
            a.wrapping_add(h_in[0]),
            b.wrapping_add(h_in[1]),
            c.wrapping_add(h_in[2]),
            d.wrapping_add(h_in[3]),
            e.wrapping_add(h_in[4]),
            f.wrapping_add(h_in[5]),
            g.wrapping_add(h_in[6]),
            h.wrapping_add(h_in[7]),
        ];
        let expected: [u32; 8] = [
            0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924,
            0x27ae41e4, 0x649b934c, 0xa495991b, 0x7852b855,
        ];
        assert_eq!(h_out, expected);
    }

    #[test]
    fn sha_f2_uair_signature_shape() {
        type U = Sha256F2Uair<Int<4>>;
        let sig = U::signature();
        // 41 total bin-poly cols (14 public + 27 witness), 0 arb-poly,
        // 0 int (the F_2 prove path requires an empty int lane).
        assert_eq!(sig.witness_cols().num_binary_poly_cols(), cols::NUM_BIN_WIT);
        assert_eq!(sig.public_cols().num_binary_poly_cols(), cols::NUM_BIN_PUB);
        assert_eq!(sig.public_cols().num_int_cols(), 0);
    }

    /// Sanity: all non-zero-ideal constraints must be degree-1 in the
    /// trace MLEs so the F_2 IC can dispatch through its MLE-first
    /// path. The boundary `assert_zero` constraints can have higher
    /// raw degree.
    #[test]
    fn sha_f2_uair_is_mle_first_eligible() {
        type U = Sha256F2Uair<Int<4>>;
        assert_eq!(count_effective_max_degree::<U>(), 1);
        assert!(count_max_degree::<U>() >= 2);
    }

    /// Trace generator runs without panicking and yields a column
    /// count matching the layout.
    #[test]
    fn sha_f2_trace_gen_shape() {
        type U = Sha256F2Uair<Int<4>>;
        let mut rng = rand::rng();
        let trace = U::generate_random_trace(cols::MIN_NUM_VARS, &mut rng);
        assert_eq!(trace.binary_poly.len(), cols::NUM_BIN);
        assert_eq!(trace.arbitrary_poly.len(), 0);
        assert_eq!(trace.int.len(), 0);
    }
}
