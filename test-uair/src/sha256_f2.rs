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
//! Public bit-poly prefix (8):
//!   - `PA_A`, `PA_E`              boundary `H_i` for `a`/`e` (init prefix + junction)
//!   - `PA_K`                      round constants `K_t`
//!   - `PA_M`                      message-block words `M_i[0..16]`
//!   - `PA_C`                      single packed LSB-compensator
//!                                 column. Bit `j` is the per-step
//!                                 compensator `κ_j` for the binary
//!                                 step at bit-position `j` (see
//!                                 `KAPPA_BIT_*` in the `cols`
//!                                 submodule); positions 0..12 used,
//!                                 13..31 are verifier-checked zero.
//!   - `S_INIT_PREFIX`, `S_FF`,
//!     `S_MSG_INIT`                3 `{0,1}`-valued selectors (carried
//!                                 as bit-polys, value 1 ↔ poly `1`)
//!
//! Committed witnesses (20):
//!   - state (3):                  `W_A`, `W_E`, `W_W`
//!   - rotation/shift outputs (4): `W_SIGMA0`, `W_SIGMA1`, `W_SIG0`,
//!                                 `W_SIG1` (pinned by C1–C4)
//!   - SHR auxiliaries (2):        `W_SHR3_W`, `W_SHR10_W` — committed
//!                                 here because the row-shifted
//!                                 `SHR^r` access (`SHR^r(W_W)^{↓Δ}`)
//!                                 is a design-parameter compose of
//!                                 row-shift + per-cell bit-op that
//!                                 isn't shipped (cf. doc §3.3 and
//!                                 `f2x-sha-todo.md`)
//!   - AND-result (3):             `W_UEF`, `W_UNEG_E_G`, `W_MAJ`
//!                                 (pinned by external Hadamard checks,
//!                                 not by in-circuit constraints —
//!                                 mechanism deferred per doc §3.4)
//!   - round intermediates (2):    `W_T1`, `W_T2`
//!   - chained-Binius intermediate-sums (6):
//!                                 `W_W_S1`, `W_W_S2` (W chain)
//!                                 `W_T1_S1..4` (T_1 chain)
//!                                 — per-row partial sums materialised
//!                                 per doc §3.7 Definition 1.
//!
//! Bit-op virtual columns (12, declared in `signature()`):
//!   `SHR^j(PA_C)` for `j ∈ {1..12}`, used to extract bit `j` of
//!   `PA_C` as the per-step compensator `κ_j` in the LSB check.
//!   `κ_0` is bit 0 of `PA_C` itself (no shift needed, no spec).
//!
//! ## Constraint families
//!
//! A — Rotation/shift identities (C1–C4), all `(X^32 − 1)`-ideal in
//!     `F_2[X]` (the doc's `F_rot`): bind the committed Σ/σ-output
//!     columns to the state columns via the `rho` polynomials of
//!     doc §2.2 / §2.3.
//!
//! B — Per-step `BiniusAdd` LSB constraints (C5–C11 unfolded into 13
//!     binary steps), all `(X)`-ideal in `F_2[X]`. Each is the
//!     per-step scalar LSB check from doc §3.6 / §3.7:
//!     `(target + input_x + input_y + κ_k)[0] = 0`,
//!     where `κ_k` is bit `j_k` of `PA_C` (accessed via the
//!     bit-op virtual `SHR^{j_k}(PA_C)`). Bits 1..31 of the addition
//!     are pinned by the column-level Hadamard
//!     `(x + X·c) ⊙ (y + X·c) = c + X·c` in `F_shift`, which is
//!     external/deferred per doc §3.4.
//!
//! C — Boundary pinning (C16–C18), `assert_zero` gated by selectors:
//!     pin init-prefix `W_A`/`W_E` to `PA_A`/`PA_E`, pin
//!     message-seed `W_W` to `PA_M`.
//!
//! Soundness gap (carried over from the prior CSA-flattened impl): the
//! Hadamard halves of the 13 modular-add `BiniusAdd`s and the 3 AND
//! relations (C12–C14) are NOT enforced at the AIR layer. They would
//! be discharged by a separate Hadamard-product check — see doc §3.4.
//! The AIR-level LSB checks pin only bit 0 of each addition; bits
//! 1..31 are honest-prover-only.

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
    BitOp, BitOpSpec, ConstraintBuilder, PublicColumnLayout, ShiftSpec, TotalColumnLayout,
    TraceRow, Uair, UairSignature, UairTrace,
    ideal::{Ideal, IdealCheck, IdealCheckError, rotation::RotationIdeal},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{cfg_into_iter, cfg_iter};
use zinc_utils::from_ref::FromRef;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::GenerateRandomTrace;

// ---------------------------------------------------------------------------
// Sha256F2Ideal — ideals used by the F_2[X] SHA-256 UAIR.
// ---------------------------------------------------------------------------

/// Ideals used by this UAIR.
///
/// Spec doc `documentation/sha-f2x-doc/sections/arithmetization.tex` uses
/// three quotients:
///   - `F_rot = F_2[X]/(X^32 − 1)` for rotation identities (Σ, σ).
///   - `F_shift = F_2[X]/(X^32)` for modular-addition identities (where
///     `X·c` drops bit 31 of `c` via the quotient — the natural home for
///     the Binius identity with virtual carry).
///   - `(X)` for the per-step `BiniusAdd` LSB scalar check `(t + x + y +
///     κ)[0] = 0` — only bit 0 of the expression is constrained; bits
///     1..31 are pinned by the (deferred) external Hadamard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sha256F2Ideal {
    /// `(X^32 − 1)` in `F_2[X]`. Rotation identities (C1–C4).
    RotXw1,
    /// `(X^32)` in `F_2[X]`. Currently unused at the constraint layer;
    /// would be the natural ideal for any polynomial-valued modular-
    /// add LinSum constraint (the pre-step-2 form). Kept as a variant
    /// so we don't have to re-add it when wiring up the deferred
    /// Hadamard half of `BiniusAdd`.
    ShiftX32,
    /// `(X)` in `F_2[X]`. The doc's per-step `BiniusAdd` LSB ideal: a
    /// polynomial lies in `(X)` iff its coefficient at position 0 is
    /// zero. Used by C5–C11 (13 per-step LSB checks total).
    LsbX,
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
        match self {
            Sha256F2Ideal::RotXw1 => {
                let one = F::one_with_cfg(value.coeffs[0].cfg());
                IdealOrZero::NonZero(RotationIdeal::<F, 32>::new(one)).contains(value)
            }
            Sha256F2Ideal::ShiftX32 => {
                // Membership in `(X^32)`: coefficients at positions
                // 0..32 must all be zero (the polynomial is divisible
                // by `X^32`). Higher-degree coefficients are
                // unconstrained.
                let limit = value.coeffs.len().min(32);
                for c in &value.coeffs[..limit] {
                    if !F::is_zero(c) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Sha256F2Ideal::LsbX => {
                // Membership in `(X)`: coefficient at position 0 must
                // be zero. Positions ≥ 1 are unconstrained — they're
                // pinned by the (deferred) external Hadamard half of
                // `BiniusAdd`, not by this scalar LSB check.
                if value.coeffs.is_empty() {
                    return Ok(true);
                }
                Ok(F::is_zero(&value.coeffs[0]))
            }
        }
    }
}

/// Project an `IdealOrZero<Sha256F2Ideal>` into the verifier field.
/// The `Zero` arm is filtered upstream by the IC's per-constraint loop
/// before any projection runs; we return `RotXw1` as a harmless
/// default in that branch.
pub fn sha256_f2_project_ideal(ideal: &IdealOrZero<Sha256F2Ideal>) -> Sha256F2Ideal {
    match ideal {
        IdealOrZero::NonZero(i) => *i,
        IdealOrZero::Zero => Sha256F2Ideal::RotXw1,
    }
}

// ---------------------------------------------------------------------------
// Column indices.
// ---------------------------------------------------------------------------

pub mod cols {
    // === Public bit-poly prefix (8) ===
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    pub const PA_K: usize = 2;
    pub const PA_M: usize = 3;
    /// Packed LSB compensator: bit `j` of `PA_C[t]` is `κ_j`, the
    /// per-step `BiniusAdd` compensator for the binary step at bit
    /// position `j`. Positions 0..12 are used (one per binary step;
    /// see `BETA_BIT_*` constants below for the allocation); bits
    /// 13..31 must be zero.
    pub const PA_C: usize = 4;
    pub const S_INIT_PREFIX: usize = 5;
    pub const S_FF: usize = 6;
    pub const S_MSG_INIT: usize = 7;
    pub const NUM_BIN_PUB: usize = 8;

    // === Witness bit-poly suffix (20) ===
    // State (3)
    pub const W_A: usize = NUM_BIN_PUB;
    pub const W_E: usize = NUM_BIN_PUB + 1;
    pub const W_W: usize = NUM_BIN_PUB + 2;
    // Rotation/shift outputs (4) — pinned by C1–C4
    pub const W_SIGMA0: usize = NUM_BIN_PUB + 3;
    pub const W_SIGMA1: usize = NUM_BIN_PUB + 4;
    pub const W_SIG0: usize = NUM_BIN_PUB + 5;
    pub const W_SIG1: usize = NUM_BIN_PUB + 6;
    // SHR auxiliaries (2) — committed because the SHR^r row-shifted
    // access mechanism is left as a design parameter (cf. doc §3.3).
    // Note: the per-cell `BitOp::ShiftR` machinery is now wired in
    // (used for `PA_C` bit extraction below), but row-shifted
    // composition with `BitOp` isn't, so these two columns can't yet
    // be virtualised: the constraint reads `SHR^3(W_W)^{↓1}` which is
    // shift-then-bit-op, not bit-op-then-shift.
    pub const W_SHR3_W: usize = NUM_BIN_PUB + 7;
    pub const W_SHR10_W: usize = NUM_BIN_PUB + 8;
    // AND results (3) — pinned by external Hadamard checks (deferred)
    pub const W_UEF: usize = NUM_BIN_PUB + 9;
    pub const W_UNEG_E_G: usize = NUM_BIN_PUB + 10;
    pub const W_MAJ: usize = NUM_BIN_PUB + 11;
    // Round intermediates (2)
    pub const W_T1: usize = NUM_BIN_PUB + 12;
    pub const W_T2: usize = NUM_BIN_PUB + 13;
    // Chained-Binius intermediate-sum columns (6) — per doc §3.7
    // Definition 1. Materialised at the chain anchor row k.
    pub const W_W_S1: usize = NUM_BIN_PUB + 14; // s_2 of W chain (after 1 add)
    pub const W_W_S2: usize = NUM_BIN_PUB + 15; // s_3 of W chain (after 2 adds)
    pub const W_T1_S1: usize = NUM_BIN_PUB + 16; // s_2 of T_1 chain
    pub const W_T1_S2: usize = NUM_BIN_PUB + 17; // s_3 of T_1 chain
    pub const W_T1_S3: usize = NUM_BIN_PUB + 18; // s_4 of T_1 chain
    pub const W_T1_S4: usize = NUM_BIN_PUB + 19; // s_5 of T_1 chain

    pub const NUM_BIN_WIT: usize = 20;
    pub const NUM_BIN: usize = NUM_BIN_PUB + NUM_BIN_WIT; // 28

    // === PA_C bit-allocation (per doc §4.2 binius-packing remark) ===
    //
    // Each binary step `k ∈ [0, 13)` claims bit `KAPPA_BIT_*` of
    // `PA_C[t]` as its LSB compensator `κ_k`. The constraint
    // expression accesses bit `j` of `PA_C` via the bit-op virtual
    // `SHR^j(PA_C)` (whose bit 0 = bit j of `PA_C`).
    pub const KAPPA_BIT_W_1: u32 = 0;     // W chain step 1
    pub const KAPPA_BIT_W_2: u32 = 1;     // W chain step 2
    pub const KAPPA_BIT_W_3: u32 = 2;     // W chain step 3
    pub const KAPPA_BIT_T1_1: u32 = 3;    // T_1 chain step 1
    pub const KAPPA_BIT_T1_2: u32 = 4;    // T_1 chain step 2
    pub const KAPPA_BIT_T1_3: u32 = 5;    // T_1 chain step 3
    pub const KAPPA_BIT_T1_4: u32 = 6;    // T_1 chain step 4
    pub const KAPPA_BIT_T1_5: u32 = 7;    // T_1 chain step 5
    pub const KAPPA_BIT_T2: u32 = 8;      // T_2 (single binary step)
    pub const KAPPA_BIT_A: u32 = 9;       // a' (single binary step)
    pub const KAPPA_BIT_E: u32 = 10;      // e' (single binary step)
    pub const KAPPA_BIT_FF_A: u32 = 11;   // feed-forward-a
    pub const KAPPA_BIT_FF_E: u32 = 12;   // feed-forward-e

    /// Total number of per-step compensator bits packed into `PA_C`.
    pub const NUM_KAPPA: usize = 13;

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

        // Shifts — `UairSignature::new` sorts by (source_col,
        // shift_amount); listing in source-col-ascending order makes
        // the `down.binary_poly` slot indices below match this list.
        let shifts: Vec<ShiftSpec> = vec![
            // PA_K: shift 3 for K[t] in C6.
            ShiftSpec::new(cols::PA_K, 3),
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
        ];

        // Bit-op virtuals: `SHR^j(PA_C)` for j ∈ {1..12}, used to
        // extract bit `j` of `PA_C` as the per-step κ compensator in
        // the LSB checks (C5–C11). `UairSignature::new` sorts by
        // (source_col, op_kind, c). PA_C = 4 < every witness col, so
        // these bit-ops sort to `down.bit_op[0..12]` in shift-amount
        // ascending order (since they all share source PA_C and op
        // kind ShiftR).
        let bit_op_specs: Vec<BitOpSpec> = (1u32..=12u32)
            .map(|c| BitOpSpec::new(cols::PA_C, BitOp::shift_r(c)))
            .collect();

        UairSignature::new(total, public, shifts, vec![], bit_op_specs)
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
        let pa_c = &bp[cols::PA_C];
        let s_init_prefix = &bp[cols::S_INIT_PREFIX];
        let s_msg_init = &bp[cols::S_MSG_INIT];

        let w_a = &bp[cols::W_A];
        let w_e = &bp[cols::W_E];
        let w_w = &bp[cols::W_W];
        let w_sigma0 = &bp[cols::W_SIGMA0];
        let w_sigma1 = &bp[cols::W_SIGMA1];
        // W_UEF, W_UNEG_E_G are pinned by external Hadamards C12/C13
        // (deferred); their up-row values don't enter any AIR-level
        // constraint. The down-row shifted versions (`down_w_uef_3`,
        // `down_w_uneg_e_g_3`) DO enter C6 as inputs.
        let w_maj = &bp[cols::W_MAJ];
        let w_t2 = &bp[cols::W_T2];
        let w_w_s1 = &bp[cols::W_W_S1];
        let w_w_s2 = &bp[cols::W_W_S2];
        let w_t1_s1 = &bp[cols::W_T1_S1];
        let w_t1_s2 = &bp[cols::W_T1_S2];
        let w_t1_s3 = &bp[cols::W_T1_S3];
        let w_t1_s4 = &bp[cols::W_T1_S4];

        // Down-row shifts. Slot order = `signature()`'s ShiftSpec list
        // sorted by (source_col, shift_amount). Source-col order:
        // PA_K(2), W_A(8), W_E(9), W_W(10), W_SIGMA1(12), W_SIG0(13),
        // W_SIG1(14), W_SHR3_W(15), W_SHR10_W(16), W_UEF(17),
        // W_UNEG_E_G(18), W_T1(19), W_T2(20).
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

        // Bit-op virtuals: `SHR^j(PA_C)` for j ∈ {1..12}. Sort order
        // (source_col=PA_C, op_kind=ShiftR, c ascending) ⇒
        // `down.bit_op[j-1] = SHR^j(PA_C)`. Bit 0 of `SHR^j(PA_C)` is
        // bit `j` of `PA_C` — the per-step κ_j compensator. For j=0
        // we use `pa_c` directly (no shift needed).
        let _ = &up.bit_op; // up.bit_op is always empty
        let kappa = |j: u32| -> &B::Expr {
            // j ∈ {1..12} → down.bit_op[j-1]; j=0 unused (use pa_c
            // directly at the call site).
            debug_assert!((1..=12).contains(&j), "kappa(j): j out of range");
            &down.bit_op[(j as usize) - 1]
        };

        // Ideals.
        let ideal_rot = ideal_from_ref(&Sha256F2Ideal::RotXw1);
        let ideal_lsb = ideal_from_ref(&Sha256F2Ideal::LsbX);

        // Scalars.
        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]); // X^30 + X^19 + X^10
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]); //  X^26 + X^21 + X^7
        let rho_lsig0 = rho_poly::<R>(&[14, 25]); //    X^25 + X^14
        let rho_lsig1 = rho_poly::<R>(&[13, 15]); //    X^15 + X^13

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
        // (B) Per-step `BiniusAdd` LSB constraints — 13 binary steps
        //     unfolded from C5–C11. Each enforces only the LSB scalar
        //     equation `(target + x + y + κ_k)[0] = 0`. Bits 1..31
        //     are pinned by the column-level Hadamard (deferred per
        //     doc §3.4).
        //
        // `κ_0` is bit 0 of `PA_C`, accessed directly via `pa_c`. For
        // `j ∈ {1..12}` we use the bit-op virtual `SHR^j(PA_C)` (its
        // bit 0 = bit j of `PA_C`).
        // -----------------------------------------------------------

        // C5 — W chain (4-input, anchor t = t' − 16, 3 binary steps).
        //
        // C5a: W_W^{(1)} ≡ W_W + W_SIG0^{↓1}. LSB:
        //   (W_W_S1 + W_W + W_SIG0^{↓1} + κ_{W,1})[0] = 0,  κ_{W,1} = pa_c[0]
        b.assert_in_ideal(
            w_w_s1.clone() + w_w + down_w_sig0_1 + pa_c,
            &ideal_lsb,
        );
        // C5b: W_W^{(2)} ≡ W_W^{(1)} + W_W^{↓9}.
        b.assert_in_ideal(
            w_w_s2.clone() + w_w_s1 + down_w_w_9 + kappa(cols::KAPPA_BIT_W_2),
            &ideal_lsb,
        );
        // C5c: W_W^{↓16} ≡ W_W^{(2)} + W_SIG1^{↓14}.
        b.assert_in_ideal(
            down_w_w_16.clone() + w_w_s2 + down_w_sig1_14 + kappa(cols::KAPPA_BIT_W_3),
            &ideal_lsb,
        );

        // C6 — T_1 chain (6-input, anchor t = t' − 3, 5 binary steps).
        // Inputs in chained order: h[t'] (= e[t] via shift-register),
        // Σ_1(e[t]), u_ef[t'], u_{¬e,g}[t'], K[t'], W[t'].
        //
        // C6a: W_{T_1}^{(1)} ≡ W_E + W_SIGMA1^{↓3}.
        b.assert_in_ideal(
            w_t1_s1.clone() + w_e + down_w_sigma1_3 + kappa(cols::KAPPA_BIT_T1_1),
            &ideal_lsb,
        );
        // C6b: W_{T_1}^{(2)} ≡ W_{T_1}^{(1)} + W_UEF^{↓3}.
        b.assert_in_ideal(
            w_t1_s2.clone() + w_t1_s1 + down_w_uef_3 + kappa(cols::KAPPA_BIT_T1_2),
            &ideal_lsb,
        );
        // C6c: W_{T_1}^{(3)} ≡ W_{T_1}^{(2)} + W_UNEG_E_G^{↓3}.
        b.assert_in_ideal(
            w_t1_s3.clone() + w_t1_s2 + down_w_uneg_e_g_3 + kappa(cols::KAPPA_BIT_T1_3),
            &ideal_lsb,
        );
        // C6d: W_{T_1}^{(4)} ≡ W_{T_1}^{(3)} + PA_K^{↓3}.
        b.assert_in_ideal(
            w_t1_s4.clone() + w_t1_s3 + down_pa_k_3 + kappa(cols::KAPPA_BIT_T1_4),
            &ideal_lsb,
        );
        // C6e: W_T1^{↓3} ≡ W_{T_1}^{(4)} + W_W^{↓3}.
        b.assert_in_ideal(
            down_w_t1_3.clone() + w_t1_s4 + down_w_w_3 + kappa(cols::KAPPA_BIT_T1_5),
            &ideal_lsb,
        );

        // C7 — T_2 (2-input, row-local, 1 binary step):
        //   W_T2 ≡ W_SIGMA0 + W_MAJ.
        b.assert_in_ideal(
            w_t2.clone() + w_sigma0 + w_maj + kappa(cols::KAPPA_BIT_T2),
            &ideal_lsb,
        );

        // C8 — a' = T_1 + T_2 (anchor t = t' − 3):
        //   W_A^{↓4} ≡ W_T1^{↓3} + W_T2^{↓3}.
        b.assert_in_ideal(
            down_w_a_4.clone() + down_w_t1_3 + down_w_t2_3 + kappa(cols::KAPPA_BIT_A),
            &ideal_lsb,
        );

        // C9 — e' = d + T_1 (d = a[t'−4] = up.W_A, T_1 = down.W_T1^{↓3}):
        //   W_E^{↓4} ≡ W_A + W_T1^{↓3}.
        b.assert_in_ideal(
            down_w_e_4.clone() + w_a + down_w_t1_3 + kappa(cols::KAPPA_BIT_E),
            &ideal_lsb,
        );

        // C10 — feed-forward a (junction-window):
        //   W_A^{↓4} ≡ W_A + PA_A.
        b.assert_in_ideal(
            down_w_a_4.clone() + w_a + pa_a + kappa(cols::KAPPA_BIT_FF_A),
            &ideal_lsb,
        );

        // C11 — feed-forward e (junction-window):
        //   W_E^{↓4} ≡ W_E + PA_E.
        b.assert_in_ideal(
            down_w_e_4.clone() + w_e + pa_e + kappa(cols::KAPPA_BIT_FF_E),
            &ideal_lsb,
        );

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

        // Chained-Binius intermediate sums (one column per partial
        // sum). Each `s_k_vals[r]` is materialised at chain anchor row
        // `r`; outside the chain's row set the column carries zeros.
        let mut w_w_s1_vals = vec![0u32; n]; // s_2 of W chain
        let mut w_w_s2_vals = vec![0u32; n]; // s_3 of W chain
        let mut w_t1_s1_vals = vec![0u32; n]; // s_2 of T_1 chain
        let mut w_t1_s2_vals = vec![0u32; n]; // s_3 of T_1 chain
        let mut w_t1_s3_vals = vec![0u32; n]; // s_4 of T_1 chain
        let mut w_t1_s4_vals = vec![0u32; n]; // s_5 of T_1 chain

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
                // Chained-Binius intermediates at anchor k = t − 16
                // (= j − 16 within this compression). Per doc §3.7:
                //   s_2 = W[k] + σ_0(W[k+1]) mod 2^32
                //   s_3 = s_2 + W[k+9]      mod 2^32
                //   target W[k+16] = s_3 + σ_1(W[k+14]) mod 2^32
                let k = t - 16;
                let s2 = w_vals[t - 16].wrapping_add(lsig0);
                let s3 = s2.wrapping_add(w_vals[t - 7]);
                debug_assert_eq!(s3.wrapping_add(lsig1), w_vals[t]);
                w_w_s1_vals[k] = s2;
                w_w_s2_vals[k] = s3;
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

                // Chained-Binius intermediates for T_1. Inputs in
                // chain order: h, Σ_1(e), u_ef, u_neg, K, W. Per
                // doc §3.7:
                //   s_2 = h + Σ_1(e)
                //   s_3 = s_2 + u_ef
                //   s_4 = s_3 + u_neg
                //   s_5 = s_4 + K
                //   t1  = s_5 + W
                let s2 = h_val.wrapping_add(sig1_e);
                let s3 = s2.wrapping_add(u_ef);
                let s4 = s3.wrapping_add(u_neg);
                let s5 = s4.wrapping_add(k_vals[t]);
                let t1 = s5.wrapping_add(w_vals[t]);
                let t1_sum: u64 = (h_val as u64)
                    + (sig1_e as u64)
                    + (u_ef as u64)
                    + (u_neg as u64)
                    + (k_vals[t] as u64)
                    + (w_vals[t] as u64);
                debug_assert_eq!(t1 as u64, t1_sum & 0xFFFF_FFFF);

                // W_T1 is row-local on the SHA round index (= t = k+3),
                // so C8 / C9 (anchored at t' − 3 = k) can read
                // W_T1^↓3[k] = W_T1[k+3]. The chained-intermediate
                // columns sit at the C6 anchor row k (C6 reads them
                // without shifts).
                t1_vals[t] = t1;
                w_t1_s1_vals[k] = s2;
                w_t1_s2_vals[k] = s3;
                w_t1_s3_vals[k] = s4;
                w_t1_s4_vals[k] = s5;

                // T_2 = Σ_0(a) + Maj (single binary step, row-local at
                // SHA round t = k+3).
                let t2 = sig0_a.wrapping_add(maj_t);
                t2_vals[t] = t2;

                // a' = T_1 + T_2.
                a_vals[k + 4] = t1.wrapping_add(t2);

                // e' = d + T_1, with d = a[t' − 4] = a_vals[k].
                e_vals[k + 4] = a_vals[k].wrapping_add(t1);
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

                h_a_next[j] = internal_a.wrapping_add(prior_a);
                h_e_next[j] = internal_e.wrapping_add(prior_e);
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
        // Each of these is `O(n)` work over `n = 2^num_vars` rows with
        // no inter-row dependencies (Σ/σ/SHR are per-cell; Ch/Maj use
        // a small fixed row window). The `cfg_*_iter!(_, MIN_LEN)`
        // form caps rayon's split granularity at `MIN_LEN` rows per
        // task — at `n = 2^9 = 512` (test fixture) the whole job is
        // smaller than MIN_LEN and runs on a single thread, avoiding
        // rayon's task-spawn overhead; at `n ≥ 2^16` chunks divide
        // cleanly across cores.
        const PAR_MIN_LEN: usize = 1 << 12; // 4096 rows per parallel chunk
        let sigma0_vals: Vec<u32> = cfg_iter!(a_vals, PAR_MIN_LEN)
            .copied()
            .map(big_sigma0)
            .collect();
        let sigma1_vals: Vec<u32> = cfg_iter!(e_vals, PAR_MIN_LEN)
            .copied()
            .map(big_sigma1)
            .collect();
        let sig0_vals: Vec<u32> = cfg_iter!(w_vals, PAR_MIN_LEN)
            .copied()
            .map(small_sigma0)
            .collect();
        let sig1_vals: Vec<u32> = cfg_iter!(w_vals, PAR_MIN_LEN)
            .copied()
            .map(small_sigma1)
            .collect();
        let shr3_w_vals: Vec<u32> = cfg_iter!(w_vals, PAR_MIN_LEN)
            .copied()
            .map(|w| w >> 3)
            .collect();
        let shr10_w_vals: Vec<u32> = cfg_iter!(w_vals, PAR_MIN_LEN)
            .copied()
            .map(|w| w >> 10)
            .collect();

        // Per-row Ch / Maj operands. We populate on every row (not
        // just SHA-active ones) to keep the trace honest; off-active
        // rows don't participate in any constraint.
        let u_ef_vals: Vec<u32> = cfg_into_iter!(0..n, PAR_MIN_LEN)
            .map(|t| if t >= 1 { e_vals[t] & e_vals[t - 1] } else { 0 })
            .collect();
        let u_neg_e_g_vals: Vec<u32> = cfg_into_iter!(0..n, PAR_MIN_LEN)
            .map(|t| if t >= 2 { (!e_vals[t]) & e_vals[t - 2] } else { 0 })
            .collect();
        let maj_vals: Vec<u32> = cfg_into_iter!(0..n, PAR_MIN_LEN)
            .map(|t| {
                if t >= 2 {
                    maj(a_vals[t], a_vals[t - 1], a_vals[t - 2])
                } else {
                    0
                }
            })
            .collect();

        // ---- Packed LSB-compensator column (PA_C) --------------------
        // PA_C[k] is a 32-bit bit-poly with 13 used bit positions
        // (`KAPPA_BIT_*` in the `cols` submodule). Bit `j_step` is
        // the per-step LSB compensator `κ_step` for the binary step
        // at that bit position. Each `κ_step` at row k equals the
        // LSB of `target[k] + x[k] + y[k]` (in F_2): zero on rows
        // where the witness is consistent with the binary add, the
        // residue otherwise. Bits 13..31 are kept at zero.
        let load = |arr: &[u32], idx: usize| -> u32 { if idx < n { arr[idx] } else { 0 } };
        let lsb = |x: u32| -> u32 { x & 1 };

        let pa_c_vals: Vec<u32> = cfg_into_iter!(0..n, PAR_MIN_LEN)
            .map(|k| {
                let mut v = 0u32;
                // C5 — W chain (3 steps; LSB checks (target + x + y)[0] = 0).
                v |= lsb(load(&w_w_s1_vals, k) ^ load(&w_vals, k) ^ load(&sig0_vals, k + 1))
                    << cols::KAPPA_BIT_W_1;
                v |= lsb(load(&w_w_s2_vals, k) ^ load(&w_w_s1_vals, k) ^ load(&w_vals, k + 9))
                    << cols::KAPPA_BIT_W_2;
                v |= lsb(load(&w_vals, k + 16) ^ load(&w_w_s2_vals, k) ^ load(&sig1_vals, k + 14))
                    << cols::KAPPA_BIT_W_3;
                // C6 — T_1 chain (5 steps).
                v |= lsb(load(&w_t1_s1_vals, k) ^ load(&e_vals, k) ^ load(&sigma1_vals, k + 3))
                    << cols::KAPPA_BIT_T1_1;
                v |= lsb(load(&w_t1_s2_vals, k) ^ load(&w_t1_s1_vals, k) ^ load(&u_ef_vals, k + 3))
                    << cols::KAPPA_BIT_T1_2;
                v |= lsb(
                    load(&w_t1_s3_vals, k) ^ load(&w_t1_s2_vals, k) ^ load(&u_neg_e_g_vals, k + 3),
                ) << cols::KAPPA_BIT_T1_3;
                v |= lsb(load(&w_t1_s4_vals, k) ^ load(&w_t1_s3_vals, k) ^ load(&k_vals, k + 3))
                    << cols::KAPPA_BIT_T1_4;
                v |= lsb(load(&t1_vals, k + 3) ^ load(&w_t1_s4_vals, k) ^ load(&w_vals, k + 3))
                    << cols::KAPPA_BIT_T1_5;
                // C7 — T_2 (row-local).
                v |= lsb(load(&t2_vals, k) ^ load(&sigma0_vals, k) ^ load(&maj_vals, k))
                    << cols::KAPPA_BIT_T2;
                // C8 — a'.
                v |= lsb(load(&a_vals, k + 4) ^ load(&t1_vals, k + 3) ^ load(&t2_vals, k + 3))
                    << cols::KAPPA_BIT_A;
                // C9 — e' (= d + T_1, d = a[t-3] = a_vals[k]).
                v |= lsb(load(&e_vals, k + 4) ^ load(&a_vals, k) ^ load(&t1_vals, k + 3))
                    << cols::KAPPA_BIT_E;
                // C10 — feed-forward a.
                v |= lsb(load(&a_vals, k + 4) ^ load(&a_vals, k) ^ load(&pa_a_vals, k))
                    << cols::KAPPA_BIT_FF_A;
                // C11 — feed-forward e.
                v |= lsb(load(&e_vals, k + 4) ^ load(&e_vals, k) ^ load(&pa_e_vals, k))
                    << cols::KAPPA_BIT_FF_E;
                v
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
            // -- Public prefix (8) --
            to_mle(to_bits(&pa_a_vals)),     // PA_A
            to_mle(to_bits(&pa_e_vals)),     // PA_E
            to_mle(to_bits(&k_vals)),        // PA_K
            to_mle(to_bits(&pa_m_vals)),     // PA_M
            to_mle(to_bits(&pa_c_vals)),     // PA_C
            to_mle(to_bits(&s_init_prefix)), // S_INIT_PREFIX
            to_mle(to_bits(&s_ff)),          // S_FF
            to_mle(to_bits(&s_msg_init)),    // S_MSG_INIT
            // -- Witness suffix (20) --
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
            to_mle(to_bits(&w_w_s1_vals)),   // W_W_S1
            to_mle(to_bits(&w_w_s2_vals)),   // W_W_S2
            to_mle(to_bits(&w_t1_s1_vals)),  // W_T1_S1
            to_mle(to_bits(&w_t1_s2_vals)),  // W_T1_S2
            to_mle(to_bits(&w_t1_s3_vals)),  // W_T1_S3
            to_mle(to_bits(&w_t1_s4_vals)),  // W_T1_S4
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
