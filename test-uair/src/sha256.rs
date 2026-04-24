//! SHA-256 compression UAIR (slice).
//!
//! Spec source: `arithmetization_standalone/hybrid_arithmetics/sha256/`.
//!
//! Implements the full SHA-256 compression round-function and message
//! schedule as a UAIR, with the `F_2[X]` rotation constraints lifted to
//! `Q[X]` via per-coefficient overflow witnesses (see below). The UAIR
//! ties both ends of the compression via public inputs:
//!
//! - init: `a_hat[0] = y_a_public` (one boundary row for `a` only)
//! - final: `a_hat[row] = y_a_public[row]` and `e_hat[row] = y_e_public[row]`
//!   at each of the last four rows, encoding `(d, c, b, a)` / `(h, g, f, e)`
//!   under the SHA-256 shift-register convention.
//!
//! Covered constraint families, for every active row `t`:
//!
//! 1. `Sigma_0` rotation:  `a_hat · rho_Sig0 − Sig0_hat − 2·ov_Sig0 ∈ (X^32 − 1)`
//! 2. `Sigma_1` rotation:  `e_hat · rho_Sig1 − Sig1_hat − 2·ov_Sig1 ∈ (X^32 − 1)`
//! 3. `sigma_0` right-shift decomposition:
//!    `W_hat − T_0 − X^3 · S_0 == 0`  (exact Z[X] equality; both sides are
//!    bit-polynomials by construction).
//! 4. `sigma_0` rotation:
//!    `W_hat · rho_lsig0 + S_0 − lsig0_hat − 2·ov_lsig0 ∈ (X^32 − 1)`
//! 5. `sigma_1` right-shift decomposition:
//!    `W_hat − T_1 − X^10 · S_1 == 0`
//! 6. `sigma_1` rotation:
//!    `W_hat · rho_lsig1 + S_1 − lsig1_hat − 2·ov_lsig1 ∈ (X^32 − 1)`
//! 7. Message-schedule modular sum:
//!    `s_sched · (W[t] − W[t-16] − sigma_0(W[t-15]) − W[t-7] − sigma_1(W[t-2])
//!      + 2·X^31 · mu_W[t]) ∈ (X − 2)`,
//!    expressed at anchor row `k = t − 16` with forward shifts. Because
//!    `DensePolynomial<R, 32>` can't hold `X^32`, we use `2·X^31` — same
//!    value `2^32` when evaluated at `X = 2`.
//! 8. Register-update `a`:
//!    `s_upd · (a[t+1] − (h[t] + Sigma_1(e[t]) + Ch[t] + K[t] + W[t]
//!                        + Sigma_0(a[t]) + Maj[t]) + 2·X^31 · mu_a[t]) ∈ (X − 2)`
//!    where `h[t] = e[t-3]` under the SHA-256 shift-register trick. Anchored
//!    at row `k = t − 3`; shifts 0..4 on `w_a`, `w_e`, plus shift 3 on all
//!    per-row quantities.
//! 9. Register-update `e`:
//!    `s_upd · (e[t+1] − (d[t] + h[t] + Sigma_1(e[t]) + Ch[t] + K[t] + W[t])
//!              + 2·X^31 · mu_e[t]) ∈ (X − 2)`
//!    where `d[t] = a[t-3]`.
//! 10. Init boundary:  `s_init  · (a_hat − y_a_public) == 0`  (row 0).
//! 11. Final boundary (a-family): `s_final · (a_hat − y_a_public) == 0`
//!     applied at rows `n−4 .. n−1`. `y_a_public` is a dual-purpose column:
//!     it carries the initial `a` at row 0 and the final `(d, c, b, a)` values
//!     at rows `n−4, n−3, n−2, n−1`, with zeros everywhere in between. Under
//!     the SHA-256 shift-register convention, `a[t]` at the last round *is*
//!     the final `a`; `a[t−1]` is the final `b`; etc., so placing the
//!     values in the `a` column at consecutive rows at the end of the trace
//!     correctly encodes `(y_a, y_b, y_c, y_d)`.
//!
//! Ch and Maj are left as **free witness columns** (bit-polys, unenforced
//! against the true boolean function). A spec-faithful implementation would
//! add lookup or degree-2 constraints; lookups are stubbed upstream so this
//! slice adopts the same "lookup-gap" stance we already have for other
//! bit-valued cells.
//!
//! The `rho_*` scalars encode the rotation parts of the SHA-256 Σ and σ
//! functions:
//!
//! - `rho_Sig0(X) = X^30 + X^19 + X^10`     (= `X^{32-2} + X^{32-13} + X^{32-22}`)
//! - `rho_Sig1(X) = X^26 + X^21 + X^7`      (= `X^{32-6} + X^{32-11} + X^{32-25}`)
//! - `rho_lsig0(X) = X^25 + X^14`            (= `X^{32-7} + X^{32-18}`)
//! - `rho_lsig1(X) = X^15 + X^13`            (= `X^{32-17} + X^{32-19}`)
//!
//! The right-shift parts of
//! `sigma_0(W) = ROTR(W, 7) ⊕ ROTR(W, 18) ⊕ SHR(W, 3)` and
//! `sigma_1(W) = ROTR(W, 17) ⊕ ROTR(W, 19) ⊕ SHR(W, 10)`
//! are carried by the `S_i` / `T_i` splits: `W = T_0 + X^3 · S_0` and
//! `W = T_1 + X^10 · S_1` (both exact over bit-polynomials), and
//! `SHR(W, k) = S_i` as polynomials.
//!
//! ## F_2[X] → Q[X] lifting
//!
//! The spec places the rotation-family constraints in `F_2[X]/(X^32 − 1)`.
//! The zinc-plus verifier operates over a random prime field `F` (char ≠ 2),
//! so `F_2[X]` columns cannot be stored natively — we store everything as
//! `binary_poly` (structurally `{0, 1}`-valued) and lift each `F_2[X]`
//! identity to a `Q[X]` identity by adding a per-coefficient overflow
//! witness:
//!
//! ```text
//!   a_hat · rho_Sig0 ≡ Sig0_hat (mod X^32 − 1, mod 2)               [spec, F_2[X]]
//!   a_hat · rho_Sig0 ≡ Sig0_hat + 2 · ov_Sig0 (mod X^32 − 1)        [Q[X]]
//! ```
//!
//! After the lift the constraint is a pure `(X^W − 1)` check over `R[X]`,
//! carried by [`Sha256Ideal::RotXw1`]. The mod-2 accounting is enforced by
//! the prover-side overflow witness, not by the verifier.
//!
//! **Overflow witnesses are bit-polynomials.** For `rho_Sig0` / `rho_Sig1`
//! (3 terms each), per-position contribution counting caps the reduced
//! coefficient at 3, giving overflow `∈ {0, 1}`. For `rho_lsig0` (2 terms)
//! the reduced coefficient is in `{0, 1, 2}`, plus adding `S_0` (bit) gives
//! `{0, 1, 2, 3}`, again with overflow `∈ {0, 1}`. See
//! [`sigma0_overflow`] for the full Sig0 derivation; the others are analogous.
//!
//! ### Soundness caveat
//!
//! Having the verifier literally check `y ∈ ideal mod 2` pre-projection is
//! not realizable in the current `IdealCheck` subprotocol, because that
//! protocol combines per-row constraint values via a random `F`-linear
//! combination before handing them to `IdealCheck::contains`. After the
//! combination, coefficients are uniform in `F`, not small-integer lifts,
//! so any coefficient-parity test rejects honest provers with overwhelming
//! probability. Closing the soundness gap requires a per-row ideal check
//! placed before the random linear combination, or a protocol rewrite;
//! both are out of scope.
//!
//! ## Lookup-enforced range checks
//!
//! The integer-carry columns are range-checked via logup-GKR
//! (declared in `signature()`'s `lookup_specs`):
//!
//! - `mu_W ∈ [0, 3]`  → `Word { width: 2 }` (4-term modular-sum carry)
//! - `mu_a ∈ [0, 6]`  → `Word { width: 3 }` (7-term register-update carry)
//! - `mu_e ∈ [0, 5]`  → `Word { width: 3 }` (6-term register-update carry)
//!
//! `mu_a` and `mu_e` share the Word{width:3} table, so
//! `group_lookup_specs` collapses them into one lookup group with a
//! shared multiplicity column (`W_M_W3`). `mu_W` is alone in its
//! group (`W_M_W2`). The multiplicity columns are auto-generated by
//! the trace generator and sit at the end of the int section per
//! the "last N int cols = multiplicities" convention.
//!
//! ## Out of scope
//! - **Ch and Maj enforcement** — stored as free bit-poly witness columns
//!   that the prover populates honestly; not constrained to match the true
//!   Boolean functions. A spec-faithful arithmetization would add a
//!   `BitPolyset_32` lookup on `e + f − 2·u_ef` / `a + b + c − 2·Maj` (both
//!   currently unavailable) or coefficient-wise degree-2 constraints.
//! - **K column** is populated with random integers rather than the
//!   SHA-256-specified round constants. Both prover and verifier see the
//!   same values (it's a public column), so the round-trip succeeds; a
//!   full SHA-256 implementation would pin these.
//! - **Initial-state public inputs for `e`, `d = a[t-3]`, `h = e[t-3]`** —
//!   these would let a verifier pin the initial compression state. The
//!   init boundary currently only constrains `a[0] = pa_a[0]`.

use core::marker::PhantomData;

use crypto_primitives::{ConstSemiring, PrimeField, Semiring};
use num_traits::Zero;
use rand::RngCore;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial,
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_uair::{
    AffineExpr, ConstraintBuilder, LookupColumnSpec, LookupTableType, PublicColumnLayout, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace,
    ideal::{Ideal, IdealCheck, IdealCheckError, rotation::RotationIdeal},
};
use zinc_utils::from_ref::FromRef;

use crate::GenerateRandomTrace;

// ---------------------------------------------------------------------------
// Sha256Ideal: enum of ideals used by the SHA-256 slice UAIR.
// ---------------------------------------------------------------------------

/// Ideals used by the SHA-256 arithmetization.
///
/// See the module-level doc for how each variant is used.
#[derive(Clone, Debug)]
pub enum Sha256Ideal<R: Semiring> {
    /// `(X − 2)` in `R[X]`. Used for Q[X] modular-sum constraints.
    RotX2(RotationIdeal<R, 1>),
    /// `(X^32 − 1)` in `R[X]`. Used for rotation constraints (Sigma_0,
    /// Sigma_1, sigma_0, sigma_1) after lifting the spec's `F_2[X]/(X^32 − 1)`
    /// identity to `R[X]` via an overflow witness. See the module doc.
    RotXw1,
}

impl<R: Semiring> FromRef<Sha256Ideal<R>> for Sha256Ideal<R> {
    fn from_ref(value: &Sha256Ideal<R>) -> Self {
        value.clone()
    }
}

impl<R: Semiring> Ideal for Sha256Ideal<R> {}

// ---------------------------------------------------------------------------
// IdealCheck<DynamicPolynomialF<F>> impl for Sha256Ideal<F>.
// ---------------------------------------------------------------------------

// We implement `IdealCheck` directly on `Sha256Ideal<F>` (the local type)
// rather than on `IdealOrZero<Sha256Ideal<F>>` to respect the orphan rule.
// The projection closure in `protocol/src/lib.rs` converts `IdealOrZero::NonZero`
// variants into a `Sha256Ideal<F>`; zero ideals are filtered out before the
// closure runs (see `piop/src/ideal_check.rs:287`), so this path never needs
// to represent `IdealOrZero::Zero` as an `IdealOverF`.
impl<F> IdealCheck<DynamicPolynomialF<F>> for Sha256Ideal<F>
where
    F: PrimeField,
{
    fn contains(&self, value: &DynamicPolynomialF<F>) -> Result<bool, IdealCheckError> {
        use zinc_uair::ideal_collector::IdealOrZero;
        match self {
            Sha256Ideal::RotX2(ideal) => IdealOrZero::NonZero(ideal.clone()).contains(value),
            Sha256Ideal::RotXw1 => {
                if value.coeffs.is_empty() {
                    return Ok(true);
                }
                let one = F::one_with_cfg(value.coeffs[0].cfg());
                IdealOrZero::NonZero(RotationIdeal::<F, 32>::new(one)).contains(value)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sha256CompressionSliceUair: the UAIR itself.
// ---------------------------------------------------------------------------

/// SHA-256 compression UAIR (slice). See module docs for the full list of
/// constraint families, the `F_2[X] → Q[X]` lifting convention, and the
/// in-scope public-input wiring.
#[derive(Clone, Debug)]
pub struct Sha256CompressionSliceUair<R>(PhantomData<R>);

/// Column indices within the flat trace (binary || arbitrary || int).
///
/// All polynomial columns are bit-polynomials (stored as `binary_poly`);
/// the int slot carries 0/1-valued selectors.
pub mod cols {
    // binary_poly, public prefix of length 2:
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    // binary_poly, witness suffix. Grouped by constraint family for clarity.
    // Sigma_0:
    pub const W_A: usize = 2;
    pub const W_SIG0: usize = 3;
    pub const W_OV_SIG0: usize = 4;
    // Sigma_1:
    pub const W_E: usize = 5;
    pub const W_SIG1: usize = 6;
    pub const W_OV_SIG1: usize = 7;
    // sigma_0 (with right-shift decomposition) — shares W_W with sigma_1:
    pub const W_W: usize = 8;
    pub const W_LSIG0: usize = 9;
    pub const W_S0: usize = 10;
    pub const W_T0: usize = 11;
    pub const W_OV_LSIG0: usize = 12;
    // sigma_1 (with right-shift decomposition):
    pub const W_LSIG1: usize = 13;
    pub const W_S1: usize = 14;
    pub const W_T1: usize = 15;
    pub const W_OV_LSIG1: usize = 16;
    // Register update (free-witness Ch and Maj — see module doc):
    pub const W_CH: usize = 17;
    pub const W_MAJ: usize = 18;

    /// Total number of binary_poly columns.
    pub const NUM_BIN: usize = 19;
    /// Number of public binary_poly columns (prefix).
    pub const NUM_BIN_PUB: usize = 2;

    // int columns. Public selectors come first (required by PublicColumnLayout
    // prefix convention), witness columns last.
    pub const S_INIT: usize = 0; // public: 1 at trace row 0, else 0
    pub const S_FINAL: usize = 1; // public: 1 on the final 4 rows (n−4 .. n−1), else 0
    pub const PA_K: usize = 2; // public: round constants column (free for this slice)
    // Public compensator columns: zero on rows where the corresponding
    // constraint is honestly satisfied (the "active" range), non-zero on
    // inactive rows so that `inner + compensator ∈ (X − 2)` everywhere.
    // Replaces the old `s_sched_anch` / `s_upd_anch` selector gates,
    // dropping C7-C9 from degree 2 to degree 1.
    //
    // TODO(verifier): the verifier must check `pa_c_c{7,8,9}` is zero on the
    // active row range (the rows where the corresponding constraint is
    // intended to bind). Without that check, a malicious prover could put a
    // nonzero compensator on active rows and absorb arbitrary `inner`,
    // breaking the SHA round binding. Tracked as a follow-up.
    pub const PA_C_C7: usize = 3; // compensator for C7 (sched_anch)
    pub const PA_C_C8: usize = 4; // compensator for C8 (upd_anch a)
    pub const PA_C_C9: usize = 5; // compensator for C9 (upd_anch e)
    pub const W_MU_W: usize = 6; // witness: integer carry for the modular-sum constraint
    pub const W_MU_A: usize = 7; // witness: integer carry for the a-update
    pub const W_MU_E: usize = 8; // witness: integer carry for the e-update
    // Lookup multiplicity columns (one per group, placed at the end per the
    // protocol's "last N int cols = multiplicities" convention). Group order
    // is the BTreeMap sort over `LookupTableType`:
    //   group 0 = Word{ width: 2 }  covers mu_W     ∈ [0, 3]
    //   group 1 = Word{ width: 3 }  covers mu_a/mu_e ∈ [0, 6]/[0, 5]
    pub const W_M_W2: usize = 9; // multiplicity column for group 0
    pub const W_M_W3: usize = 10; // multiplicity column for group 1
    /// Total number of int columns.
    pub const NUM_INT: usize = 11;
    /// Number of public int columns (prefix).
    pub const NUM_INT_PUB: usize = 6;

    /// Flat trace indices for ShiftSpec (binary_poly || arbitrary_poly || int).
    pub const FLAT_W_A: usize = W_A;
    pub const FLAT_W_SIG0: usize = W_SIG0;
    pub const FLAT_W_E: usize = W_E;
    pub const FLAT_W_SIG1: usize = W_SIG1;
    pub const FLAT_W_W: usize = W_W;
    pub const FLAT_W_LSIG0: usize = W_LSIG0;
    pub const FLAT_W_LSIG1: usize = W_LSIG1;
    pub const FLAT_W_CH: usize = W_CH;
    pub const FLAT_W_MAJ: usize = W_MAJ;
    pub const FLAT_PA_K: usize = NUM_BIN + PA_K;
    pub const FLAT_W_MU_W: usize = NUM_BIN + W_MU_W;
    pub const FLAT_W_MU_A: usize = NUM_BIN + W_MU_A;
    pub const FLAT_W_MU_E: usize = NUM_BIN + W_MU_E;
    pub const FLAT_W_M_W2: usize = NUM_BIN + W_M_W2;
    pub const FLAT_W_M_W3: usize = NUM_BIN + W_M_W3;
}

impl<R> Uair for Sha256CompressionSliceUair<R>
where
    R: ConstSemiring + 'static,
{
    type Ideal = Sha256Ideal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(cols::NUM_BIN, 0, cols::NUM_INT);
        // Public: pa_a, pa_e (2 binary_poly) + selectors + pa_K.
        let public = PublicColumnLayout::new(cols::NUM_BIN_PUB, 0, cols::NUM_INT_PUB);
        // Shifts. Listed in the exact slot order that `down.binary_poly` /
        // `down.int` will have after UairSignature::new's stable sort-by-
        // source_col — within the same source_col, insertion order here
        // determines the slot. The `DownSlots` constants below mirror this.
        let shifts: Vec<ShiftSpec> = vec![
            // === binary_poly shifts (in source_col ascending order) ===
            // w_a (flat 1): target a[t+1] at anchor t-3. Ch/Maj are free
            // witnesses, so intermediate shifts on a/e aren't needed.
            ShiftSpec::new(cols::FLAT_W_A, 4),
            // w_sig0 (flat 2): Sigma_0(a[t]) at anchor t-3.
            ShiftSpec::new(cols::FLAT_W_SIG0, 3),
            // w_e (flat 4): target e[t+1] at anchor t-3.
            ShiftSpec::new(cols::FLAT_W_E, 4),
            // w_sig1 (flat 5): Sigma_1(e[t]) at anchor t-3.
            ShiftSpec::new(cols::FLAT_W_SIG1, 3),
            // w_W (flat 7): message-schedule 9, 16 AND register-update 3.
            ShiftSpec::new(cols::FLAT_W_W, 3),
            ShiftSpec::new(cols::FLAT_W_W, 9),
            ShiftSpec::new(cols::FLAT_W_W, 16),
            // w_lsig0 (flat 8): message-schedule sigma_0(W[t-15]).
            ShiftSpec::new(cols::FLAT_W_LSIG0, 1),
            // w_lsig1 (flat 12): message-schedule sigma_1(W[t-2]).
            ShiftSpec::new(cols::FLAT_W_LSIG1, 14),
            // w_ch (flat 16): Ch[t] at anchor t-3.
            ShiftSpec::new(cols::FLAT_W_CH, 3),
            // w_maj (flat 17): Maj[t] at anchor t-3.
            ShiftSpec::new(cols::FLAT_W_MAJ, 3),
            // === int shifts (in source_col ascending order) ===
            ShiftSpec::new(cols::FLAT_PA_K, 3),
            ShiftSpec::new(cols::FLAT_W_MU_W, 16),
            ShiftSpec::new(cols::FLAT_W_MU_A, 3),
            ShiftSpec::new(cols::FLAT_W_MU_E, 3),
        ];
        // Range-check the integer carries via logup-GKR. The logup table
        // types determine the group layout: Word{width:2} and Word{width:3}
        // yield two groups, with Word{width:2} first (BTreeMap order over
        // `LookupTableType`). Multiplicity columns `W_M_W2` / `W_M_W3`
        // sit at the end of the int section in the corresponding order.
        //
        // Per-column ranges:
        //   mu_W ∈ [0, 3] — 4-term modular sum carry           → Word{width:2}
        //   mu_a ∈ [0, 6] — 7-term register-update carry       → Word{width:3}
        //   mu_e ∈ [0, 5] — 6-term register-update carry       → Word{width:3}
        let lookup_specs = vec![
            LookupColumnSpec {
                expression: AffineExpr::single(cols::FLAT_W_MU_W),
                table_type: LookupTableType::Word {
                    width: 2,
                    chunk_width: None,
                },
            },
            LookupColumnSpec {
                expression: AffineExpr::single(cols::FLAT_W_MU_A),
                table_type: LookupTableType::Word {
                    width: 3,
                    chunk_width: None,
                },
            },
            LookupColumnSpec {
                expression: AffineExpr::single(cols::FLAT_W_MU_E),
                table_type: LookupTableType::Word {
                    width: 3,
                    chunk_width: None,
                },
            },
        ];
        UairSignature::new(total, public, shifts, lookup_specs)
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
        let sel = up.int;

        // Columns.
        let pa_a = &bp[cols::PA_A];
        let pa_e = &bp[cols::PA_E];
        let w_a = &bp[cols::W_A];
        let w_sig0 = &bp[cols::W_SIG0];
        let w_ov_sig0 = &bp[cols::W_OV_SIG0];
        let w_e = &bp[cols::W_E];
        let w_sig1 = &bp[cols::W_SIG1];
        let w_ov_sig1 = &bp[cols::W_OV_SIG1];
        let w_big_w = &bp[cols::W_W];
        let w_lsig0 = &bp[cols::W_LSIG0];
        let w_s0 = &bp[cols::W_S0];
        let w_t0 = &bp[cols::W_T0];
        let w_ov_lsig0 = &bp[cols::W_OV_LSIG0];
        let w_lsig1 = &bp[cols::W_LSIG1];
        let w_s1 = &bp[cols::W_S1];
        let w_t1 = &bp[cols::W_T1];
        let w_ov_lsig1 = &bp[cols::W_OV_LSIG1];

        let s_init = &sel[cols::S_INIT];
        let s_final = &sel[cols::S_FINAL];
        let pa_c_c7 = &sel[cols::PA_C_C7];
        let pa_c_c8 = &sel[cols::PA_C_C8];
        let pa_c_c9 = &sel[cols::PA_C_C9];

        // `down` slot layout (mirrors the ShiftSpec order in signature()).
        // bin slots:
        let down_w_a_sh4 = &down.binary_poly[0];
        let down_w_sig0_sh3 = &down.binary_poly[1];
        let down_w_e_sh4 = &down.binary_poly[2];
        let down_w_sig1_sh3 = &down.binary_poly[3];
        let down_w_w_sh3 = &down.binary_poly[4];
        let down_w_w_sh9 = &down.binary_poly[5];
        let down_w_w_sh16 = &down.binary_poly[6];
        let down_w_lsig0_sh1 = &down.binary_poly[7];
        let down_w_lsig1_sh14 = &down.binary_poly[8];
        let down_w_ch_sh3 = &down.binary_poly[9];
        let down_w_maj_sh3 = &down.binary_poly[10];
        // int slots:
        let down_pa_k_sh3 = &down.int[0];
        let down_w_mu_w_sh16 = &down.int[1];
        let down_w_mu_a_sh3 = &down.int[2];
        let down_w_mu_e_sh3 = &down.int[3];

        // Ideals.
        let ideal_rot_xw1 = ideal_from_ref(&Sha256Ideal::<R>::RotXw1);
        let ideal_rot_x2 = ideal_from_ref(&Sha256Ideal::<R>::RotX2(
            RotationIdeal::new(R::ONE + R::ONE), // (X − 2)
        ));

        // Scalars.
        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]); // X^30 + X^19 + X^10
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]); //  X^26 + X^21 + X^7
        let rho_lsig0 = rho_poly::<R>(&[14, 25]); //    X^25 + X^14
        let rho_lsig1 = rho_poly::<R>(&[13, 15]); //    X^15 + X^13
        let two_scalar = const_scalar::<R>(R::ONE + R::ONE);
        let x_pow_3 = mono_x_pow::<R>(3);
        let x_pow_10 = mono_x_pow::<R>(10);
        // `two_times_x31` evaluates to 2 · 2^31 = 2^32 at X = 2 — a
        // representable proxy for the `X^32 · mu_W` carry term from the spec.
        let two_times_x31 = {
            let mut coeffs = [R::ZERO; 32];
            coeffs[31] = R::ONE + R::ONE;
            DensePolynomial::<R, 32>::new(coeffs)
        };

        // C1–C6 apply on every row unconditionally — no selector needed, the
        // rotation identities are row-local and honest values satisfy them on
        // every row. Dropping the `s_active` gate (identically 1 in the
        // trace) takes these constraints from degree 2 to degree 1 in the
        // trace MLEs.

        // Constraint 1: Sigma_0 rotation, Q[X]-lifted.
        //   (a_hat · rho_sig0 − sig0_hat − 2 · ov_sig0) ∈ (X^32 − 1)
        b.assert_in_ideal(
            mbs(w_a, &rho_sig0).expect("a · rho_sig0 overflow") - w_sig0
                - &mbs(w_ov_sig0, &two_scalar).expect("2 · ov_sig0 overflow"),
            &ideal_rot_xw1,
        );

        // Constraint 2: Sigma_1 rotation, Q[X]-lifted.
        //   (e_hat · rho_sig1 − sig1_hat − 2 · ov_sig1) ∈ (X^32 − 1)
        b.assert_in_ideal(
            mbs(w_e, &rho_sig1).expect("e · rho_sig1 overflow") - w_sig1
                - &mbs(w_ov_sig1, &two_scalar).expect("2 · ov_sig1 overflow"),
            &ideal_rot_xw1,
        );

        // Constraint 3: sigma_0 right-shift decomposition (exact Z[X] equality,
        // not an ideal check — both sides are bit-polynomials).
        //   W_hat − T_0 − X^3 · S_0 == 0
        b.assert_zero(
            w_big_w.clone() - w_t0 - &mbs(w_s0, &x_pow_3).expect("X^3 · S_0 overflow"),
        );

        // Constraint 4: sigma_0 rotation (with the shift piece pre-split into
        // S_0), Q[X]-lifted.
        //   (W_hat · rho_lsig0 + S_0 − lsig0_hat − 2 · ov_lsig0) ∈ (X^32 − 1)
        b.assert_in_ideal(
            mbs(w_big_w, &rho_lsig0).expect("W · rho_lsig0 overflow") + w_s0 - w_lsig0
                - &mbs(w_ov_lsig0, &two_scalar).expect("2 · ov_lsig0 overflow"),
            &ideal_rot_xw1,
        );

        // Constraint 5: sigma_1 right-shift decomposition.
        //   W_hat − T_1 − X^10 · S_1 == 0
        b.assert_zero(
            w_big_w.clone() - w_t1 - &mbs(w_s1, &x_pow_10).expect("X^10 · S_1 overflow"),
        );

        // Constraint 6: sigma_1 rotation, Q[X]-lifted.
        //   (W_hat · rho_lsig1 + S_1 − lsig1_hat − 2 · ov_lsig1) ∈ (X^32 − 1)
        b.assert_in_ideal(
            mbs(w_big_w, &rho_lsig1).expect("W · rho_lsig1 overflow") + w_s1 - w_lsig1
                - &mbs(w_ov_lsig1, &two_scalar).expect("2 · ov_lsig1 overflow"),
            &ideal_rot_xw1,
        );

        // Constraint 7: Message-schedule modular sum, anchored at k = t − 16.
        //   (W[k+16] − W[k] − sigma_0(W[k+1]_row) − W[k+9] − sigma_1(W[k+14]_row)
        //                   + 2·X^31 · mu_W[k+16] + pa_c_c7) ∈ (X − 2)
        //
        // Notes:
        // - Our σ_0/σ_1 columns are row-local (lsig0[r] = σ_0(W[r])), so the
        //   spec's "sigma_0(W[t-15])" is our lsig0 at row k+1, and
        //   "sigma_1(W[t-2])" is our lsig1 at row k+14.
        // - `2·X^31` stands in for `X^32` — they coincide when evaluated at
        //   X = 2, which is where the (X − 2) ideal check lives.
        // - `pa_c_c7` is a public compensator column: zero on the active
        //   range (rows 0..=n−17), and equal to `−inner(2)` mod p on inactive
        //   rows so the sum lies in (X − 2) everywhere. See the
        //   `cols::PA_C_C7` doc for the verifier-side TODO.
        let two_x31_mu_w = mbs(down_w_mu_w_sh16, &two_times_x31)
            .expect("2·X^31 · mu_W overflow");
        let sched_inner = down_w_w_sh16.clone()
            - w_big_w
            - down_w_lsig0_sh1
            - down_w_w_sh9
            - down_w_lsig1_sh14
            + &two_x31_mu_w;
        b.assert_in_ideal(sched_inner + pa_c_c7, &ideal_rot_x2);

        // Constraint 8: Register-update for `a`, anchored at k = t − 3.
        //   (a[t+1] − (h[t] + Sigma_1(e[t]) + Ch[t] + K[t] + W[t]
        //              + Sigma_0(a[t]) + Maj[t])
        //    + 2·X^31 · mu_a[t] + pa_c_c8) ∈ (X − 2)
        //
        // With the shift-register aliasing h[t] = e[t-3] = up.w_e.
        // References at anchor k:
        //   a[t+1]       = down.w_a^↓4     e[t-3]       = up.w_e
        //   a[t]         = down.w_a^↓3     Sigma_1(e[t]) = down.w_sig1^↓3
        //   Sigma_0(a[t]) = down.w_sig0^↓3 Ch[t]         = down.w_ch^↓3
        //   Maj[t]       = down.w_maj^↓3   W[t]         = down.w_W^↓3
        //   K[t]         = down.pa_K^↓3   mu_a[t]       = down.w_mu_a^↓3
        // pa_c_c8 is the public compensator (see C7 note).
        let two_x31_mu_a = mbs(down_w_mu_a_sh3, &two_times_x31)
            .expect("2·X^31 · mu_a overflow");
        let a_update_inner = down_w_a_sh4.clone()
            - w_e                         // h[t] = e[t-3]
            - down_w_sig1_sh3             // Sigma_1(e[t])
            - down_w_ch_sh3               // Ch[t]
            - down_pa_k_sh3               // K[t]
            - down_w_w_sh3                // W[t]
            - down_w_sig0_sh3             // Sigma_0(a[t])
            - down_w_maj_sh3              // Maj[t]
            + &two_x31_mu_a;
        b.assert_in_ideal(a_update_inner + pa_c_c8, &ideal_rot_x2);

        // Constraint 9: Register-update for `e`, anchored at k = t − 3.
        //   (e[t+1] − (d[t] + h[t] + Sigma_1(e[t]) + Ch[t] + K[t] + W[t])
        //    + 2·X^31 · mu_e[t] + pa_c_c9) ∈ (X − 2)
        //
        // With d[t] = a[t-3] = up.w_a and h[t] = e[t-3] = up.w_e.
        let two_x31_mu_e = mbs(down_w_mu_e_sh3, &two_times_x31)
            .expect("2·X^31 · mu_e overflow");
        let e_update_inner = down_w_e_sh4.clone()
            - w_a                         // d[t] = a[t-3]
            - w_e                         // h[t] = e[t-3]
            - down_w_sig1_sh3
            - down_w_ch_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            + &two_x31_mu_e;
        b.assert_in_ideal(e_update_inner + pa_c_c9, &ideal_rot_x2);

        // Constraint 10: Init boundary — at trace row 0, a_hat equals y_a_public.
        //   s_init · (a_hat − y_a) == 0
        b.assert_zero(s_init.clone() * &(w_a.clone() - pa_a));

        // Constraint 11: Final boundary (a-family) — applied at rows n−4..n−1.
        //   s_final · (a_hat − y_a) == 0
        b.assert_zero(s_final.clone() * &(w_a.clone() - pa_a));

        // Constraint 12: Final boundary (e-family) — applied at rows n−4..n−1.
        //   s_final · (e_hat − y_e) == 0
        // Encodes (h, g, f, e) at rows n−4..n−1 via the same shift-register
        // convention used for the a-family.
        b.assert_zero(s_final.clone() * &(w_e.clone() - pa_e));
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers (rho polynomials, monomials, constants).
// ---------------------------------------------------------------------------

/// Build a rho polynomial from a list of "1" coefficient positions.
fn rho_poly<R: ConstSemiring>(positions: &[usize]) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    for &p in positions {
        debug_assert!(p < 32);
        coeffs[p] = R::ONE;
    }
    DensePolynomial::<R, 32>::new(coeffs)
}

/// Build the monomial `X^k` as a `DensePolynomial<R, 32>`.
fn mono_x_pow<R: ConstSemiring>(k: usize) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[k] = R::ONE;
    DensePolynomial::<R, 32>::new(coeffs)
}

/// Build the constant-polynomial `c` as a `DensePolynomial<R, 32>`.
fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// SHA-256 reference helpers (for witness generation).
// ---------------------------------------------------------------------------

#[inline]
fn rotr(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

/// `Sigma_0(x) = ROTR(x, 2) ⊕ ROTR(x, 13) ⊕ ROTR(x, 22)`.
#[inline]
fn big_sigma0(x: u32) -> u32 {
    rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
}

/// `Sigma_1(x) = ROTR(x, 6) ⊕ ROTR(x, 11) ⊕ ROTR(x, 25)`.
#[inline]
fn big_sigma1(x: u32) -> u32 {
    rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
}

/// `sigma_0(x) = ROTR(x, 7) ⊕ ROTR(x, 18) ⊕ SHR(x, 3)`.
#[inline]
fn small_sigma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
}

/// `sigma_1(x) = ROTR(x, 17) ⊕ ROTR(x, 19) ⊕ SHR(x, 10)`.
#[inline]
fn small_sigma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
}

/// `Ch(x, y, z) = (x ∧ y) ⊕ (¬x ∧ z)`.
#[inline]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ ((!x) & z)
}

/// `Maj(x, y, z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)`.
#[inline]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

/// Generic overflow computer for a Q[X]-lifted rotation constraint:
/// given the pre-reduction product of two bit-polynomials (`input_bits` times
/// the rho pattern encoded by `rho_positions`), reduce modulo `(X^32 − 1)`,
/// optionally add another bit-polynomial `s0_bits` (pass 0 to skip), and
/// return the per-coefficient quotient `floor((reduced + s0 − out_bits) / 2)`
/// packed as a 32-bit word.
///
/// Per the module doc, for the rho patterns we use (3 or 2 nonzero terms)
/// each per-position quotient fits in `{0, 1}`; the returned word is
/// therefore a valid 32-bit bit-polynomial.
fn rotation_overflow(
    input_bits: u32,
    rho_positions: &[usize],
    s0_bits: u32,
    out_bits: u32,
) -> u32 {
    // Compute the Z[X] product coefficients of `input · rho`.
    let mut prod = [0u32; 64];
    for i in 0..32 {
        if (input_bits >> i) & 1 == 1 {
            for &p in rho_positions {
                prod[i + p] += 1;
            }
        }
    }
    // Reduce modulo (X^32 − 1): fold the high half back into the low half.
    let mut reduced = [0u32; 32];
    reduced.copy_from_slice(&prod[..32]);
    for k in 32..64 {
        reduced[k - 32] += prod[k];
    }
    // Add the optional S_0 contribution.
    for k in 0..32 {
        reduced[k] += (s0_bits >> k) & 1;
    }
    let mut overflow: u32 = 0;
    for k in 0..32 {
        let out_k = (out_bits >> k) & 1;
        debug_assert!(
            reduced[k] >= out_k,
            "reduced coeff < output bit at k={k}: reduced={}, out={}",
            reduced[k],
            out_k,
        );
        debug_assert_eq!(
            (reduced[k] - out_k) % 2,
            0,
            "parity mismatch at k={k}: reduced={}, out={}",
            reduced[k],
            out_k,
        );
        let ov_k = (reduced[k] - out_k) / 2;
        debug_assert!(
            ov_k <= 1,
            "overflow coeff out of {{0,1}} at k={k}: got {ov_k}"
        );
        overflow |= ov_k << k;
    }
    overflow
}

/// `Sigma_0` overflow. Kept as a named helper to make the derivation
/// referenced in the module doc easy to find.
fn sigma0_overflow(a_val: u32, sigma_val: u32) -> u32 {
    rotation_overflow(a_val, &[10, 19, 30], 0, sigma_val)
}

fn sigma1_overflow(e_val: u32, sigma_val: u32) -> u32 {
    rotation_overflow(e_val, &[7, 21, 26], 0, sigma_val)
}

fn lsig0_overflow(w_val: u32, lsig0_val: u32) -> u32 {
    // S_0 = W >> 3 (high 29 bits shifted down). T_0 is unused here (it
    // only appears in the decomposition constraint).
    let s0 = w_val >> 3;
    rotation_overflow(w_val, &[14, 25], s0, lsig0_val)
}

fn lsig1_overflow(w_val: u32, lsig1_val: u32) -> u32 {
    // S_1 = W >> 10 (high 22 bits shifted down).
    let s1 = w_val >> 10;
    rotation_overflow(w_val, &[13, 15], s1, lsig1_val)
}

// ---------------------------------------------------------------------------
// GenerateRandomTrace for the slice.
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for Sha256CompressionSliceUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n = 1usize << num_vars;
        assert!(n >= 16, "message schedule requires at least 16 rows");

        // Random round constants `K` (one per row). The slice does not pin
        // these to SHA-256's canonical K table because we don't enforce it;
        // both prover and verifier see the same public column, which is
        // sufficient for the round-trip.
        let k_vals: Vec<u32> = (0..n).map(|_| rng.next_u32()).collect();

        // W follows the SHA-256 message-schedule recurrence starting at row 16.
        let mut w_vals: Vec<u32> = (0..16).map(|_| rng.next_u32()).collect();
        let mut mu_w_vals: Vec<u32> = vec![0u32; 16];
        for t in 16..n {
            let sum_u64: u64 = (w_vals[t - 16] as u64)
                + (small_sigma0(w_vals[t - 15]) as u64)
                + (w_vals[t - 7] as u64)
                + (small_sigma1(w_vals[t - 2]) as u64);
            w_vals.push(sum_u64 as u32);
            let carry = (sum_u64 >> 32) as u32;
            debug_assert!(carry <= 3, "message-schedule carry out of [0,3]: {carry}");
            mu_w_vals.push(carry);
        }

        // `a` and `e` evolve according to the SHA-256 round function for
        // rows where the register-update constraint is active. The first
        // four rows (0..=3) are the initial state — think of them as
        // (d, c, b, a) at round t=3 under the shift-register trick.
        //
        // At spec row `t ∈ [3..n-2]` (anchor k = t − 3 ∈ [0..n-5]) we set:
        //   a[t+1] = (h + Sigma_1(e[t]) + Ch + K[t] + W[t] + Sigma_0(a[t]) + Maj) mod 2^32
        //   e[t+1] = (d + h + Sigma_1(e[t]) + Ch + K[t] + W[t]) mod 2^32
        // with h = e[t-3], d = a[t-3], b = a[t-1], c = a[t-2],
        //      f = e[t-1], g = e[t-2].
        //
        // The integer sums are bounded:
        //   T1 = h + Sigma_1(e) + Ch + K + W: 5 terms × (2^32 − 1) ≤ 5·(2^32 − 1).
        //   T2 = Sigma_0(a) + Maj: 2 terms.
        //   a_sum = T1 + T2: 7 terms ⇒ carry mu_a ∈ {0, .., 6}.
        //   e_sum = d + T1: 6 terms ⇒ carry mu_e ∈ {0, .., 5}.
        let mut a_vals: Vec<u32> = Vec::with_capacity(n);
        let mut e_vals: Vec<u32> = Vec::with_capacity(n);
        // Rows 0..=3 are free initial state.
        for _ in 0..4 {
            a_vals.push(rng.next_u32());
            e_vals.push(rng.next_u32());
        }
        // mu_a[t], mu_e[t] live at spec row t. The recurrence loop starts at
        // t = 3 and pushes mu_a_vals[t]; so we pre-pad indices 0..=2 with
        // zeros (the register-update constraint is inactive there regardless,
        // because shifts {0..=4} can't form a full 5-row window at anchor k < 0).
        let mut mu_a_vals: Vec<u32> = vec![0u32; 3];
        let mut mu_e_vals: Vec<u32> = vec![0u32; 3];

        for t in 3..(n - 1) {
            let sig0_a_t = big_sigma0(a_vals[t]);
            let sig1_e_t = big_sigma1(e_vals[t]);
            let ch_t = ch(e_vals[t], e_vals[t - 1], e_vals[t - 2]);
            let maj_t = maj(a_vals[t], a_vals[t - 1], a_vals[t - 2]);
            let h_t = e_vals[t - 3];
            let d_t = a_vals[t - 3];
            let w_t = w_vals[t];
            let k_t = k_vals[t];

            let t1: u64 = (h_t as u64)
                + (sig1_e_t as u64)
                + (ch_t as u64)
                + (k_t as u64)
                + (w_t as u64);
            let t2: u64 = (sig0_a_t as u64) + (maj_t as u64);
            let a_sum: u64 = t1 + t2;
            let e_sum: u64 = (d_t as u64) + t1;

            let a_next = a_sum as u32;
            let e_next = e_sum as u32;
            let mu_a_t = (a_sum >> 32) as u32;
            let mu_e_t = (e_sum >> 32) as u32;

            debug_assert!(mu_a_t <= 6, "mu_a out of [0,6]: {mu_a_t}");
            debug_assert!(mu_e_t <= 5, "mu_e out of [0,5]: {mu_e_t}");

            a_vals.push(a_next);
            e_vals.push(e_next);
            // mu_a, mu_e at spec row t (not t+1).
            mu_a_vals.push(mu_a_t);
            mu_e_vals.push(mu_e_t);
        }
        // Pad mu vectors to length n (last row has no constraint, set to 0).
        assert_eq!(a_vals.len(), n);
        assert_eq!(e_vals.len(), n);
        while mu_a_vals.len() < n {
            mu_a_vals.push(0);
        }
        while mu_e_vals.len() < n {
            mu_e_vals.push(0);
        }

        // Pre-compute Ch and Maj per row (honest, matching the round function).
        // At spec row t, w_ch[t] uses e[t], e[t-1], e[t-2]; w_maj[t] uses
        // a[t], a[t-1], a[t-2]. For t < 2 the references fall off-trace;
        // for those rows we write zeros (the constraint is inactive there).
        let ch_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { ch(e_vals[t], e_vals[t - 1], e_vals[t - 2]) } else { 0 })
            .collect();
        let maj_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { maj(a_vals[t], a_vals[t - 1], a_vals[t - 2]) } else { 0 })
            .collect();

        // Derived values.
        let sig0_vals: Vec<u32> = a_vals.iter().copied().map(big_sigma0).collect();
        let sig1_vals: Vec<u32> = e_vals.iter().copied().map(big_sigma1).collect();
        let lsig0_vals: Vec<u32> = w_vals.iter().copied().map(small_sigma0).collect();
        let lsig1_vals: Vec<u32> = w_vals.iter().copied().map(small_sigma1).collect();

        let ov_sig0_vals: Vec<u32> = a_vals
            .iter()
            .zip(&sig0_vals)
            .map(|(&a, &s)| sigma0_overflow(a, s))
            .collect();
        let ov_sig1_vals: Vec<u32> = e_vals
            .iter()
            .zip(&sig1_vals)
            .map(|(&e, &s)| sigma1_overflow(e, s))
            .collect();
        let ov_lsig0_vals: Vec<u32> = w_vals
            .iter()
            .zip(&lsig0_vals)
            .map(|(&w, &l)| lsig0_overflow(w, l))
            .collect();
        let ov_lsig1_vals: Vec<u32> = w_vals
            .iter()
            .zip(&lsig1_vals)
            .map(|(&w, &l)| lsig1_overflow(w, l))
            .collect();

        // Right-shift decomposition: S_i = W >> k, T_i = W & ((1<<k) - 1).
        let s0_vals: Vec<u32> = w_vals.iter().map(|&w| w >> 3).collect();
        let t0_vals: Vec<u32> = w_vals.iter().map(|&w| w & 0b111).collect();
        let s1_vals: Vec<u32> = w_vals.iter().map(|&w| w >> 10).collect();
        let t1_vals: Vec<u32> = w_vals.iter().map(|&w| w & 0x3FF).collect();

        // Public: pa_a is dual-purpose — it carries the initial `a` at row 0
        // (gated by s_init) and the final a-family values (d, c, b, a) at
        // rows n−4, n−3, n−2, n−1 (gated by s_final), with zeros in between.
        let mut pa_a_col: Vec<BinaryPoly<32>> =
            (0..n).map(|_| BinaryPoly::<32>::zero()).collect();
        pa_a_col[0] = BinaryPoly::<32>::from(a_vals[0]);
        for i in (n - 4)..n {
            pa_a_col[i] = BinaryPoly::<32>::from(a_vals[i]);
        }

        // pa_e carries the final e-family values (h, g, f, e) at rows
        // n−4..n−1 and is zero elsewhere. No init boundary on e in this slice.
        let mut pa_e_col: Vec<BinaryPoly<32>> =
            (0..n).map(|_| BinaryPoly::<32>::zero()).collect();
        for i in (n - 4)..n {
            pa_e_col[i] = BinaryPoly::<32>::from(e_vals[i]);
        }

        let to_bits = |v: &[u32]| -> Vec<BinaryPoly<32>> {
            v.iter().copied().map(BinaryPoly::<32>::from).collect()
        };

        let to_bin_mle = |col: Vec<BinaryPoly<32>>| -> DenseMultilinearExtension<
            BinaryPoly<32>,
        > { col.into_iter().collect() };

        let binary_poly = vec![
            to_bin_mle(pa_a_col),
            to_bin_mle(pa_e_col),
            to_bin_mle(to_bits(&a_vals)),
            to_bin_mle(to_bits(&sig0_vals)),
            to_bin_mle(to_bits(&ov_sig0_vals)),
            to_bin_mle(to_bits(&e_vals)),
            to_bin_mle(to_bits(&sig1_vals)),
            to_bin_mle(to_bits(&ov_sig1_vals)),
            to_bin_mle(to_bits(&w_vals)),
            to_bin_mle(to_bits(&lsig0_vals)),
            to_bin_mle(to_bits(&s0_vals)),
            to_bin_mle(to_bits(&t0_vals)),
            to_bin_mle(to_bits(&ov_lsig0_vals)),
            to_bin_mle(to_bits(&lsig1_vals)),
            to_bin_mle(to_bits(&s1_vals)),
            to_bin_mle(to_bits(&t1_vals)),
            to_bin_mle(to_bits(&ov_lsig1_vals)),
            to_bin_mle(to_bits(&ch_vals)),
            to_bin_mle(to_bits(&maj_vals)),
        ];

        // Selectors (kept: s_init, s_final).
        let mut s_init_col: Vec<R> = (0..n).map(|_| R::ZERO).collect();
        s_init_col[0] = R::ONE;
        // Final-boundary selector: 1 on the final 4 rows (rows n−4 through n−1).
        let s_final_col: Vec<R> = (0..n)
            .map(|i| if i + 4 >= n { R::ONE } else { R::ZERO })
            .collect();
        let k_col: Vec<R> = k_vals.iter().copied().map(R::from).collect();
        let mu_w_col: Vec<R> = mu_w_vals.iter().copied().map(R::from).collect();
        let mu_a_col: Vec<R> = mu_a_vals.iter().copied().map(R::from).collect();
        let mu_e_col: Vec<R> = mu_e_vals.iter().copied().map(R::from).collect();

        // ----- Compensator columns (replace s_sched_anch / s_upd_anch). -----
        //
        // For each constraint Cᵢ ∈ {C7, C8, C9}, we publish a public column
        // `pa_c_cᵢ[k]` with the property that (innerᵢ + pa_c_cᵢ) ∈ (X − 2)
        // on every row k. Concretely we pick `pa_c_cᵢ[k] = −innerᵢ(2)` mod
        // R's modulus; the protocol projects R into the random field, so
        // the negation lands as `−innerᵢ(2) mod p` — exactly the value
        // needed for the constraint to lie in (X − 2).
        //
        // On the corresponding active range (where the original selector
        // was 1), the SHA recurrence makes `innerᵢ(2) = 0` for an honest
        // prover, so `pa_c_cᵢ[k] = 0` automatically. On inactive rows the
        // compensator absorbs whatever `innerᵢ(2)` happens to be.
        let two_to_32: R = R::from(0x10000u32) * &R::from(0x10000u32);
        let load = |arr: &[u32], idx: usize| -> R {
            if idx < n { R::from(arr[idx]) } else { R::ZERO }
        };

        // C7: inner(2) = w_W[k+16] − w_W[k] − lsig0[k+1] − w_W[k+9]
        //               − lsig1[k+14] + 2^32 · mu_W[k+16]
        let pa_c_c7_col: Vec<R> = (0..n)
            .map(|k| {
                let w_k16 = load(&w_vals, k + 16);
                let w_k = load(&w_vals, k);
                let lsig0_k1 = load(&lsig0_vals, k + 1);
                let w_k9 = load(&w_vals, k + 9);
                let lsig1_k14 = load(&lsig1_vals, k + 14);
                let mu_k16 = load(&mu_w_vals, k + 16);
                let two32_mu = two_to_32.clone() * &mu_k16;
                // comp = −inner(2) = w_k + lsig0_k1 + w_k9 + lsig1_k14
                //                    − 2^32·mu_k16 − w_k16
                w_k + &lsig0_k1 + &w_k9 + &lsig1_k14 - &two32_mu - &w_k16
            })
            .collect();

        // C8: inner(2) = w_a[k+4] − w_e[k] − sig1[k+3] − ch[k+3] − K[k+3]
        //               − W[k+3] − sig0[k+3] − maj[k+3] + 2^32 · mu_a[k+3]
        let pa_c_c8_col: Vec<R> = (0..n)
            .map(|k| {
                let w_a_k4 = load(&a_vals, k + 4);
                let w_e_k = load(&e_vals, k);
                let sig1_k3 = load(&sig1_vals, k + 3);
                let ch_k3 = load(&ch_vals, k + 3);
                let k_k3 = load(&k_vals, k + 3);
                let w_k3 = load(&w_vals, k + 3);
                let sig0_k3 = load(&sig0_vals, k + 3);
                let maj_k3 = load(&maj_vals, k + 3);
                let mu_a_k3 = load(&mu_a_vals, k + 3);
                let two32_mu = two_to_32.clone() * &mu_a_k3;
                w_e_k
                    + &sig1_k3
                    + &ch_k3
                    + &k_k3
                    + &w_k3
                    + &sig0_k3
                    + &maj_k3
                    - &two32_mu
                    - &w_a_k4
            })
            .collect();

        // C9: inner(2) = w_e[k+4] − w_a[k] − w_e[k] − sig1[k+3] − ch[k+3]
        //               − K[k+3] − W[k+3] + 2^32 · mu_e[k+3]
        let pa_c_c9_col: Vec<R> = (0..n)
            .map(|k| {
                let w_e_k4 = load(&e_vals, k + 4);
                let w_a_k = load(&a_vals, k);
                let w_e_k = load(&e_vals, k);
                let sig1_k3 = load(&sig1_vals, k + 3);
                let ch_k3 = load(&ch_vals, k + 3);
                let k_k3 = load(&k_vals, k + 3);
                let w_k3 = load(&w_vals, k + 3);
                let mu_e_k3 = load(&mu_e_vals, k + 3);
                let two32_mu = two_to_32.clone() * &mu_e_k3;
                w_a_k
                    + &w_e_k
                    + &sig1_k3
                    + &ch_k3
                    + &k_k3
                    + &w_k3
                    - &two32_mu
                    - &w_e_k4
            })
            .collect();

        // Lookup-argument multiplicity columns. Length = n (trace length);
        // entries at table indices j < 2^width carry counts, indices past
        // the table (j >= 2^width) are zero-padded (they contribute
        // 0/(α − padded_T_j) to the logup cumulative sum).
        //
        // Group 0 (Word{width:2}): counts of each value in mu_w_vals.
        // Group 1 (Word{width:3}): shared counts across mu_a and mu_e.
        let mut m_w2_raw = vec![0u32; n];
        for &v in &mu_w_vals {
            debug_assert!(v < 4, "mu_W out of [0, 3]: {v}");
            m_w2_raw[v as usize] += 1;
        }
        let mut m_w3_raw = vec![0u32; n];
        for &v in mu_a_vals.iter().chain(mu_e_vals.iter()) {
            debug_assert!(v < 8, "mu_a/mu_e out of [0, 7]: {v}");
            m_w3_raw[v as usize] += 1;
        }
        let m_w2_col: Vec<R> = m_w2_raw.into_iter().map(R::from).collect();
        let m_w3_col: Vec<R> = m_w3_raw.into_iter().map(R::from).collect();

        let to_int_mle = |col: Vec<R>| -> DenseMultilinearExtension<R> {
            col.into_iter().collect()
        };
        let int = vec![
            to_int_mle(s_init_col),
            to_int_mle(s_final_col),
            to_int_mle(k_col),
            to_int_mle(pa_c_c7_col),
            to_int_mle(pa_c_c8_col),
            to_int_mle(pa_c_c9_col),
            to_int_mle(mu_w_col),
            to_int_mle(mu_a_col),
            to_int_mle(mu_e_col),
            to_int_mle(m_w2_col),
            to_int_mle(m_w3_col),
        ];

        UairTrace {
            binary_poly: binary_poly.into(),
            int: int.into(),
            ..Default::default()
        }
    }
}

// Correctness is validated by `tests::test_e2e_sha256_slice` in
// `protocol/src/lib.rs`, which runs the full prove/verify round-trip.
