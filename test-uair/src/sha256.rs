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
//! ## Range-check status (NYI)
//!
//! The integer-carry columns conceptually need range checks
//! (`mu_W ∈ [0, 3]`, `mu_a ∈ [0, 6]`, `mu_e ∈ [0, 5]`). The
//! protocol no longer wires logup-GKR, so no lookup is enforced
//! today and `signature()` returns an empty `lookup_specs`.
//!
//! ## Ch / Maj enforcement (Table 9 affine-combination lookups)
//!
//! `Ch(e, f, g) = (e ∧ f) ⊕ (¬e ∧ g)` is split into two operand bit-polys
//! `u_ef = e ∧ f` and `u_{¬e,g} = ¬e ∧ g`. Since the two AND terms can never
//! both have a bit set at the same position, `Ch = u_ef + u_{¬e,g}`
//! coefficient-wise — so the C8/C9 register-update constraints simply
//! replace the old `Ch[t]` reference with `u_ef[t] + u_{¬e,g}[t]`.
//!
//! Per Table 9 of the spec, three affine combinations must lie in
//! `{0, 1}^32` (lookup-checked): the two Ch operand witnesses and the
//! `Maj` witness. Direct affine-combination lookups aren't supported, so
//! we materialize each combination in a dedicated bit-poly column
//! (`B_1`, `B_2`, `B_3`) and lookup-check each column against the
//! `BitPoly { width: 32 }` table:
//!
//! - `B_1[t] = e[t]      + e[t-1] − 2 · u_ef[t]`           (Ch eq 62)
//! - `B_2[t] = (1₃₂ − e[t]) + e[t-2] − 2 · u_{¬e,g}[t]`     (Ch eq 63)
//! - `B_3[t] = a[t] + a[t-1] + a[t-2] − 2 · Maj[t]`        (Maj eq 64)
//!
//! Each materialization constraint is gated by the public selector
//! `S_B_ACTIVE`, which is 1 on rows where the constraint window stays in
//! trace and 0 on the last two rows (where the forward shifts of length
//! 2 land in zero-padded territory). The active rows cover the entire
//! range required by C8/C9.
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
    ConstraintBuilder, LookupColumnSpec, LookupTableType, PublicColumnLayout, ShiftSpec,
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
    // binary_poly, public prefix of length 6.
    //
    // The 4 PA_OV_* columns are the per-coefficient mod-2 overflow
    // witnesses for the rotation-ideal constraints (C1, C2, C4, C6).
    // They're "quotient witnesses" for the F_2[X] → Q[X] lift —
    // verifier-supplied, since the verifier can derive them from the
    // bit-poly values when shadow-running the SHA computation.
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    pub const PA_OV_SIG0: usize = 2;
    pub const PA_OV_SIG1: usize = 3;
    pub const PA_OV_LSIG0: usize = 4;
    pub const PA_OV_LSIG1: usize = 5;
    // binary_poly, witness suffix. Grouped by constraint family for clarity.
    // Sigma_0:
    pub const W_A: usize = 6;
    pub const W_SIG0: usize = 7;
    // Sigma_1:
    pub const W_E: usize = 8;
    pub const W_SIG1: usize = 9;
    // sigma_0 (with right-shift decomposition) — shares W_W with sigma_1:
    pub const W_W: usize = 10;
    pub const W_LSIG0: usize = 11;
    pub const W_S0: usize = 12;
    pub const W_T0: usize = 13;
    // sigma_1 (with right-shift decomposition):
    pub const W_LSIG1: usize = 14;
    pub const W_S1: usize = 15;
    pub const W_T1: usize = 16;
    // Register update — Ch is replaced by two AND-operand bit-polys
    // (`u_ef = e ∧ f` and `u_{¬e,g} = ¬e ∧ g`). Maj is still a free
    // witness on this column. See the module doc for the Table 9 split.
    pub const W_U_EF: usize = 17;
    pub const W_U_NEG_E_G: usize = 18;
    pub const W_MAJ: usize = 19;
    // Affine-combination materializations for the Table 9 lookups.
    // Each is constrained to equal the corresponding affine expression
    // (gated by S_B_ACTIVE) and is lookup-checked against the
    // `BitPoly { width: 32 }` table.
    pub const W_B1: usize = 20;
    pub const W_B2: usize = 21;
    pub const W_B3: usize = 22;

    /// Total number of binary_poly columns.
    pub const NUM_BIN: usize = 23;
    /// Number of public binary_poly columns (prefix).
    pub const NUM_BIN_PUB: usize = 6;

    // int columns. Public selectors come first (required by PublicColumnLayout
    // prefix convention), witness columns last.
    //
    // ## Layout for chained compressions
    //
    // The trace runs `NUM_COMPRESSIONS` chained SHA-256 compressions packed
    // contiguously: compression `i ∈ [0, NUM_COMPRESSIONS)` occupies rows
    // `[ROWS_PER_COMP·i, ROWS_PER_COMP·(i+1))`. Plus a 4-row H_N output
    // prefix at rows `[ROWS_PER_COMP·N, ROWS_PER_COMP·N + 4)`. PA_A / PA_E
    // carry H_i values at TWO row-blocks per compression: the init prefix
    // `[ROWS_PER_COMP·i, ROWS_PER_COMP·i + 4)` (pinned to w_a / w_e by
    // S_INIT_PREFIX, also at the H_N output block) and the junction window
    // `[ROWS_PER_COMP·i + 64, ROWS_PER_COMP·(i+1))` (read by S_FEEDFORWARD
    // as the "prior init" addend in the SHA-256 feed-forward addition).
    pub const S_INIT_PREFIX: usize = 0; // public: 1 on the (NUM_COMPRESSIONS+1) init-prefix blocks
    pub const S_FEEDFORWARD: usize = 1; // public: 1 on the NUM_COMPRESSIONS junction windows
    pub const PA_K: usize = 2; // public: round constants column (free for this slice)
    // Public compensator columns: zero on rows where the corresponding
    // constraint is honestly satisfied (the "active" range, now a union
    // of NUM_COMPRESSIONS disjoint per-compression windows), non-zero on
    // inactive rows so that `inner + compensator ∈ (X − 2)` everywhere.
    // Keeps C7-C9 at degree 1 in the trace MLEs.
    //
    // TODO(verifier): the verifier must check `pa_c_c{7,8,9}` is zero on
    // each compression's active range. Without that check, a malicious
    // prover could put a nonzero compensator on active rows and absorb
    // arbitrary `inner`, breaking the SHA round binding. Tracked as a
    // follow-up — harder to discharge now since the active range is a
    // union of NUM_COMPRESSIONS per-compression windows rather than one
    // contiguous block.
    pub const PA_C_C7: usize = 3; // compensator for C7 (sched_anch)
    pub const PA_C_C8: usize = 4; // compensator for C8 (upd_anch a)
    pub const PA_C_C9: usize = 5; // compensator for C9 (upd_anch e)
    // Compensators for the per-junction feed-forward addition
    // constraints (a-half and e-half). Same compensator pattern as
    // pa_c_c7/c8/c9: zero on the junction-window rows where the
    // SHA-256 feed-forward holds honestly, nonzero elsewhere to keep
    // `(ff_inner + pa_c_ff) ∈ (X − 2)` everywhere. Lets us drop the
    // `s_feedforward` selector multiplier and keep C12/C13 at degree 1
    // in the trace MLEs (so the UAIR remains MLE-first eligible —
    // `count_effective_max_degree::<U>() <= 1` in the bench gate).
    pub const PA_C_FF_A: usize = 6; // compensator for C12 (feed-forward a-half)
    pub const PA_C_FF_E: usize = 7; // compensator for C13 (feed-forward e-half)
    // Selector gating the three Table 9 materialization constraints
    // (B_1/B_2/B_3). 1 on per-compression anchor rows k where the
    // length-2 forward shifts stay within the same compression's
    // 68-row window; 0 elsewhere.
    pub const S_B_ACTIVE: usize = 8;
    pub const W_MU_W: usize = 9; // witness: integer carry for the modular-sum constraint
    pub const W_MU_A: usize = 10; // witness: integer carry for the a-update
    pub const W_MU_E: usize = 11; // witness: integer carry for the e-update
    // Witnesses for the SHA-256 feed-forward addition at each junction
    // between consecutive compressions: `H_{i+1} = internal_final_i + H_i
    // mod 2^32` componentwise. Each carry is in {0, 1} since both
    // summands are < 2^32. Range check is NYI (same status as
    // mu_W/mu_a/mu_e). Nonzero only on the junction-window rows.
    pub const W_MU_JUNCTION_A: usize = 12; // witness: feed-forward carry for a-half
    pub const W_MU_JUNCTION_E: usize = 13; // witness: feed-forward carry for e-half
    /// Total number of int columns.
    pub const NUM_INT: usize = 14;
    /// Number of public int columns (prefix).
    pub const NUM_INT_PUB: usize = 9;

    // ---------------------------------------------------------------------
    // Chained-compression layout constants.
    // ---------------------------------------------------------------------

    /// Number of chained SHA-256 compressions in the trace.
    pub const NUM_COMPRESSIONS: usize = 7;
    /// Rows per compression: 4 init-prefix rows + 64 round-update outputs.
    pub const ROWS_PER_COMP: usize = 68;
    /// Number of round-update steps per compression.
    pub const ROUNDS_PER_COMP: usize = 64;
    /// Trace rows used by all compressions plus the H_N output prefix.
    pub const ACTIVE_ROWS: usize = NUM_COMPRESSIONS * ROWS_PER_COMP + 4;
    /// Required `num_vars`: smallest power-of-two `n ≥ ACTIVE_ROWS`.
    /// 7·68+4 = 480 ≤ 512 = 2^9, so num_vars must be ≥ 9.
    pub const MIN_NUM_VARS: usize = 9;

    /// Flat trace indices for ShiftSpec (binary_poly || arbitrary_poly || int).
    pub const FLAT_W_A: usize = W_A;
    pub const FLAT_W_SIG0: usize = W_SIG0;
    pub const FLAT_W_E: usize = W_E;
    pub const FLAT_W_SIG1: usize = W_SIG1;
    pub const FLAT_W_W: usize = W_W;
    pub const FLAT_W_LSIG0: usize = W_LSIG0;
    pub const FLAT_W_LSIG1: usize = W_LSIG1;
    pub const FLAT_W_U_EF: usize = W_U_EF;
    pub const FLAT_W_U_NEG_E_G: usize = W_U_NEG_E_G;
    pub const FLAT_W_MAJ: usize = W_MAJ;
    pub const FLAT_W_B1: usize = W_B1;
    pub const FLAT_W_B2: usize = W_B2;
    pub const FLAT_W_B3: usize = W_B3;
    pub const FLAT_PA_K: usize = NUM_BIN + PA_K;
    pub const FLAT_W_MU_W: usize = NUM_BIN + W_MU_W;
    pub const FLAT_W_MU_A: usize = NUM_BIN + W_MU_A;
    pub const FLAT_W_MU_E: usize = NUM_BIN + W_MU_E;
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
            // === binary_poly shifts (in source_col ascending order;
            //     within the same source_col, in shift_amount order) ===
            // w_a: shifts 1, 2 for the Maj affine combination (B_3
            // anchored at t-2); shift 4 for target a[t+1] in C8.
            ShiftSpec::new(cols::FLAT_W_A, 1),
            ShiftSpec::new(cols::FLAT_W_A, 2),
            ShiftSpec::new(cols::FLAT_W_A, 4),
            // w_sig0: Sigma_0(a[t]) at C8 anchor t-3.
            ShiftSpec::new(cols::FLAT_W_SIG0, 3),
            // w_e: shifts 1, 2 for the Ch affine combinations (B_1/B_2
            // anchored at t-2); shift 4 for target e[t+1] in C9.
            ShiftSpec::new(cols::FLAT_W_E, 1),
            ShiftSpec::new(cols::FLAT_W_E, 2),
            ShiftSpec::new(cols::FLAT_W_E, 4),
            // w_sig1: Sigma_1(e[t]) at anchor t-3.
            ShiftSpec::new(cols::FLAT_W_SIG1, 3),
            // w_W: message-schedule 9, 16 AND register-update 3.
            ShiftSpec::new(cols::FLAT_W_W, 3),
            ShiftSpec::new(cols::FLAT_W_W, 9),
            ShiftSpec::new(cols::FLAT_W_W, 16),
            // w_lsig0: message-schedule sigma_0(W[t-15]).
            ShiftSpec::new(cols::FLAT_W_LSIG0, 1),
            // w_lsig1: message-schedule sigma_1(W[t-2]).
            ShiftSpec::new(cols::FLAT_W_LSIG1, 14),
            // w_u_ef: shift 2 for B_1 (anchor t-2); shift 3 for the
            // Ch[t] reference in C8/C9 (anchor t-3), where Ch is the
            // sum u_ef + u_{¬e,g} coefficient-wise.
            ShiftSpec::new(cols::FLAT_W_U_EF, 2),
            ShiftSpec::new(cols::FLAT_W_U_EF, 3),
            // w_u_neg_e_g: shift 2 for B_2; shift 3 for C8/C9.
            ShiftSpec::new(cols::FLAT_W_U_NEG_E_G, 2),
            ShiftSpec::new(cols::FLAT_W_U_NEG_E_G, 3),
            // w_maj: shift 2 for B_3; shift 3 for Maj[t] in C8.
            ShiftSpec::new(cols::FLAT_W_MAJ, 2),
            ShiftSpec::new(cols::FLAT_W_MAJ, 3),
            // w_b1/w_b2/w_b3: shift 2 for the materialization
            // constraints anchored at t-2.
            ShiftSpec::new(cols::FLAT_W_B1, 2),
            ShiftSpec::new(cols::FLAT_W_B2, 2),
            ShiftSpec::new(cols::FLAT_W_B3, 2),
            // === int shifts (in source_col ascending order) ===
            ShiftSpec::new(cols::FLAT_PA_K, 3),
            ShiftSpec::new(cols::FLAT_W_MU_W, 16),
            ShiftSpec::new(cols::FLAT_W_MU_A, 3),
            ShiftSpec::new(cols::FLAT_W_MU_E, 3),
        ];
        // BitPoly { width: 32 } lookups for the Ch operand witnesses
        // and the three Table 9 affine-combination materializations.
        // Mu_{W,a,e} are int range-check candidates (Word{width:2/3});
        // not declared here because the protocol's lookup pipeline
        // currently only handles BitPoly tables.
        let bitpoly32 = LookupTableType::BitPoly {
            width: 32,
            chunk_width: None,
        };
        let lookup_specs: Vec<LookupColumnSpec> = vec![
            LookupColumnSpec {
                column_index: cols::W_U_EF,
                table_type: bitpoly32.clone(),
            },
            LookupColumnSpec {
                column_index: cols::W_U_NEG_E_G,
                table_type: bitpoly32.clone(),
            },
            LookupColumnSpec {
                column_index: cols::W_B1,
                table_type: bitpoly32.clone(),
            },
            LookupColumnSpec {
                column_index: cols::W_B2,
                table_type: bitpoly32.clone(),
            },
            LookupColumnSpec {
                column_index: cols::W_B3,
                table_type: bitpoly32,
            },
        ];
        UairSignature::new(total, public, shifts, lookup_specs, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
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
        let pa_ov_sig0 = &bp[cols::PA_OV_SIG0];
        let pa_ov_sig1 = &bp[cols::PA_OV_SIG1];
        let pa_ov_lsig0 = &bp[cols::PA_OV_LSIG0];
        let pa_ov_lsig1 = &bp[cols::PA_OV_LSIG1];
        let w_a = &bp[cols::W_A];
        let w_sig0 = &bp[cols::W_SIG0];
        let w_e = &bp[cols::W_E];
        let w_sig1 = &bp[cols::W_SIG1];
        let w_big_w = &bp[cols::W_W];
        let w_lsig0 = &bp[cols::W_LSIG0];
        let w_s0 = &bp[cols::W_S0];
        let w_t0 = &bp[cols::W_T0];
        let w_lsig1 = &bp[cols::W_LSIG1];
        let w_s1 = &bp[cols::W_S1];
        let w_t1 = &bp[cols::W_T1];

        let s_init_prefix = &sel[cols::S_INIT_PREFIX];
        // s_feedforward is retained in the public layout for the future
        // verifier-side check that pa_c_ff_{a,e} are zero on junction
        // rows (same status as the C7/C8/C9 compensator-zero TODO);
        // the constraint expression itself doesn't reference it.
        let _s_feedforward = &sel[cols::S_FEEDFORWARD];
        let pa_c_c7 = &sel[cols::PA_C_C7];
        let pa_c_c8 = &sel[cols::PA_C_C8];
        let pa_c_c9 = &sel[cols::PA_C_C9];
        let pa_c_ff_a = &sel[cols::PA_C_FF_A];
        let pa_c_ff_e = &sel[cols::PA_C_FF_E];
        let s_b_active = &sel[cols::S_B_ACTIVE];
        let w_mu_junction_a = &sel[cols::W_MU_JUNCTION_A];
        let w_mu_junction_e = &sel[cols::W_MU_JUNCTION_E];

        // `down` slot layout (mirrors the ShiftSpec order in signature()).
        // bin slots:
        let down_w_a_sh1 = &down.binary_poly[0];
        let down_w_a_sh2 = &down.binary_poly[1];
        let down_w_a_sh4 = &down.binary_poly[2];
        let down_w_sig0_sh3 = &down.binary_poly[3];
        let down_w_e_sh1 = &down.binary_poly[4];
        let down_w_e_sh2 = &down.binary_poly[5];
        let down_w_e_sh4 = &down.binary_poly[6];
        let down_w_sig1_sh3 = &down.binary_poly[7];
        let down_w_w_sh3 = &down.binary_poly[8];
        let down_w_w_sh9 = &down.binary_poly[9];
        let down_w_w_sh16 = &down.binary_poly[10];
        let down_w_lsig0_sh1 = &down.binary_poly[11];
        let down_w_lsig1_sh14 = &down.binary_poly[12];
        let down_w_u_ef_sh2 = &down.binary_poly[13];
        let down_w_u_ef_sh3 = &down.binary_poly[14];
        let down_w_u_neg_e_g_sh2 = &down.binary_poly[15];
        let down_w_u_neg_e_g_sh3 = &down.binary_poly[16];
        let down_w_maj_sh2 = &down.binary_poly[17];
        let down_w_maj_sh3 = &down.binary_poly[18];
        let down_w_b1_sh2 = &down.binary_poly[19];
        let down_w_b2_sh2 = &down.binary_poly[20];
        let down_w_b3_sh2 = &down.binary_poly[21];
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
                - &mbs(pa_ov_sig0, &two_scalar).expect("2 · ov_sig0 overflow"),
            &ideal_rot_xw1,
        );

        // Constraint 2: Sigma_1 rotation, Q[X]-lifted.
        //   (e_hat · rho_sig1 − sig1_hat − 2 · ov_sig1) ∈ (X^32 − 1)
        b.assert_in_ideal(
            mbs(w_e, &rho_sig1).expect("e · rho_sig1 overflow") - w_sig1
                - &mbs(pa_ov_sig1, &two_scalar).expect("2 · ov_sig1 overflow"),
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
                - &mbs(pa_ov_lsig0, &two_scalar).expect("2 · ov_lsig0 overflow"),
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
                - &mbs(pa_ov_lsig1, &two_scalar).expect("2 · ov_lsig1 overflow"),
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
        // With the shift-register aliasing h[t] = e[t-3] = up.w_e, and
        // Ch[t] = u_ef[t] + u_{¬e,g}[t] (the two AND-operand bit-polys
        // never share a set bit, so addition equals XOR coefficient-wise).
        // References at anchor k:
        //   a[t+1]       = down.w_a^↓4     e[t-3]       = up.w_e
        //   a[t]         = down.w_a^↓3     Sigma_1(e[t]) = down.w_sig1^↓3
        //   Sigma_0(a[t]) = down.w_sig0^↓3 u_ef[t]       = down.w_u_ef^↓3
        //   u_{¬e,g}[t]  = down.w_u_neg_e_g^↓3
        //   Maj[t]       = down.w_maj^↓3   W[t]         = down.w_W^↓3
        //   K[t]         = down.pa_K^↓3   mu_a[t]       = down.w_mu_a^↓3
        // pa_c_c8 is the public compensator (see C7 note).
        let two_x31_mu_a = mbs(down_w_mu_a_sh3, &two_times_x31)
            .expect("2·X^31 · mu_a overflow");
        let a_update_inner = down_w_a_sh4.clone()
            - w_e                         // h[t] = e[t-3]
            - down_w_sig1_sh3             // Sigma_1(e[t])
            - down_w_u_ef_sh3             // Ch[t] = u_ef + u_{¬e,g}
            - down_w_u_neg_e_g_sh3
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
        // With d[t] = a[t-3] = up.w_a and h[t] = e[t-3] = up.w_e, and
        // Ch[t] = u_ef[t] + u_{¬e,g}[t] as in C8.
        let two_x31_mu_e = mbs(down_w_mu_e_sh3, &two_times_x31)
            .expect("2·X^31 · mu_e overflow");
        let e_update_inner = down_w_e_sh4.clone()
            - w_a                         // d[t] = a[t-3]
            - w_e                         // h[t] = e[t-3]
            - down_w_sig1_sh3
            - down_w_u_ef_sh3             // Ch[t] = u_ef + u_{¬e,g}
            - down_w_u_neg_e_g_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            + &two_x31_mu_e;
        b.assert_in_ideal(e_update_inner + pa_c_c9, &ideal_rot_x2);

        // Constraints 13–15: Table 9 affine-combination materializations,
        // each anchored at k = t − 2 and gated by `s_b_active` (1 on
        // rows where the length-2 forward shifts stay in trace). Each
        // B_i is independently lookup-checked to lie in {0,1}^32, so the
        // materialization equality forces the operand witnesses (u_ef,
        // u_{¬e,g}, Maj) to match the spec's bit-wise definitions.
        let two_scalar_poly = const_scalar::<R>(R::ONE + R::ONE);
        let one_32_poly = {
            let mut coeffs = [R::ZERO; 32];
            for c in coeffs.iter_mut() {
                *c = R::ONE;
            }
            DensePolynomial::<R, 32>::new(coeffs)
        };

        // C13: B_1[t] = e[t] + e[t-1] − 2 · u_ef[t]   (Ch eq 62)
        // At anchor k = t − 2:
        //   B_1[t]       = down.w_b1^↓2
        //   e[t]         = down.w_e^↓2     e[t-1] = down.w_e^↓1
        //   u_ef[t]      = down.w_u_ef^↓2
        let two_u_ef_sh2 = mbs(down_w_u_ef_sh2, &two_scalar_poly)
            .expect("2 · u_ef overflow");
        let b1_inner = down_w_b1_sh2.clone()
            - down_w_e_sh2
            - down_w_e_sh1
            + &two_u_ef_sh2;
        b.assert_zero(s_b_active.clone() * &b1_inner);

        // C14: B_2[t] = (1₃₂ − e[t]) + e[t-2] − 2 · u_{¬e,g}[t]   (Ch eq 63)
        // At anchor k = t − 2:
        //   B_2[t]         = down.w_b2^↓2
        //   e[t]           = down.w_e^↓2  e[t-2] = up.w_e
        //   u_{¬e,g}[t]    = down.w_u_neg_e_g^↓2
        // Inner = B_2 − 1₃₂ + e[t] − e[t-2] + 2 · u_{¬e,g}[t].
        let two_u_neg_e_g_sh2 = mbs(down_w_u_neg_e_g_sh2, &two_scalar_poly)
            .expect("2 · u_{¬e,g} overflow");
        let b2_inner = down_w_b2_sh2.clone()
            - &from_ref(&one_32_poly)
            + down_w_e_sh2
            - w_e
            + &two_u_neg_e_g_sh2;
        b.assert_zero(s_b_active.clone() * &b2_inner);

        // C15: B_3[t] = a[t] + a[t-1] + a[t-2] − 2 · Maj[t]   (Maj eq 64)
        // At anchor k = t − 2:
        //   B_3[t] = down.w_b3^↓2
        //   a[t]   = down.w_a^↓2  a[t-1] = down.w_a^↓1  a[t-2] = up.w_a
        //   Maj[t] = down.w_maj^↓2
        let two_maj_sh2 = mbs(down_w_maj_sh2, &two_scalar_poly)
            .expect("2 · Maj overflow");
        let b3_inner = down_w_b3_sh2.clone()
            - down_w_a_sh2
            - down_w_a_sh1
            - w_a
            + &two_maj_sh2;
        b.assert_zero(s_b_active.clone() * &b3_inner);

        // Constraint 10 (init-prefix pinning, a-family). For each
        // compression `i ∈ [0, NUM_COMPRESSIONS)`, pin the 4 init-prefix
        // rows `[ROWS_PER_COMP·i, ROWS_PER_COMP·i + 4)` to the public H_i
        // values. Also pins the H_N output prefix at the trailing 4 rows
        // [ROWS_PER_COMP·N, ROWS_PER_COMP·N + 4). Subsumes the spec's
        // Table-9 init- and final-state boundary constraints: at small
        // intra-compression anchors, C8/C9/C13–C15 read these pinned
        // prefix values via shifts and the boundary cases drop out for
        // free.
        //   s_init_prefix · (w_a − pa_a) == 0
        b.assert_zero(s_init_prefix.clone() * &(w_a.clone() - pa_a));

        // Constraint 11 (init-prefix pinning, e-family).
        //   s_init_prefix · (w_e − pa_e) == 0
        b.assert_zero(s_init_prefix.clone() * &(w_e.clone() - pa_e));

        // Constraint 12 (feed-forward, a-family). Anchored at junction
        // rows `[ROWS_PER_COMP·i + 64, ROWS_PER_COMP·(i+1))` for each
        // i ∈ [0, NUM_COMPRESSIONS). Enforces the SHA-256 inter-
        // compression addition `H_{i+1} = internal_final_i + H_i mod 2^32`
        // componentwise. References at junction anchor k = 68i+64+j
        // (j ∈ [0, 4)):
        //   up.w_a       = internal_final_i,  j-th component (= a[k]).
        //   up.pa_a      = H_i, j-th component (placed at junction rows
        //                  for this purpose; same value as pa_a at the
        //                  init-prefix rows of compression i).
        //   down.w_a^↓4  = w_a[k+4] = w_a[68(i+1)+j] = H_{i+1}, j-th
        //                  component (pinned by C10 to pa_a there).
        //   up.w_mu_junction_a = carry ∈ {0, 1} for this addition.
        //
        // Gated by the public compensator `pa_c_ff_a`, mirroring the
        // C7/C8/C9 compensator pattern: zero on the junction-window rows
        // where the addition holds honestly, nonzero elsewhere so that
        // `(ff_inner + pa_c_ff_a) ∈ (X − 2)` everywhere. Keeps C12 at
        // degree 1 in the trace MLEs (preserving MLE-first eligibility)
        // and avoids a multiplicative selector that would push the
        // effective max degree to 2.
        let two_x31_mu_ff_a = mbs(w_mu_junction_a, &two_times_x31)
            .expect("2·X^31 · mu_junction_a overflow");
        let ff_a_inner = down_w_a_sh4.clone()
            - w_a
            - pa_a
            + &two_x31_mu_ff_a;
        b.assert_in_ideal(ff_a_inner + pa_c_ff_a, &ideal_rot_x2);

        // Constraint 13 (feed-forward, e-family). Mirrors C12 on the
        // e-half via `pa_c_ff_e`.
        let two_x31_mu_ff_e = mbs(w_mu_junction_e, &two_times_x31)
            .expect("2·X^31 · mu_junction_e overflow");
        let ff_e_inner = down_w_e_sh4.clone()
            - w_e
            - pa_e
            + &two_x31_mu_ff_e;
        b.assert_in_ideal(ff_e_inner + pa_c_ff_e, &ideal_rot_x2);
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
        assert!(
            num_vars >= cols::MIN_NUM_VARS,
            "trace too small for {} chained compressions: need num_vars ≥ {}, got {num_vars}",
            cols::NUM_COMPRESSIONS,
            cols::MIN_NUM_VARS,
        );

        // ===== Chained-compression layout =====
        //
        // Run NUM_COMPRESSIONS independent SHA-256 compressions chained
        // via the spec's feed-forward addition `H_{i+1} = compress(H_i,
        // M_i) + H_i mod 2^32` componentwise. Compression i ∈ [0, N) uses
        // rows [i·RPC, (i+1)·RPC) where RPC = ROWS_PER_COMP = 68:
        //   - rows [start, start+4):    init prefix (= H_i, pinned to pa_a/pa_e
        //                               by S_INIT_PREFIX). Under the shift-
        //                               register convention, w_a[start+j] holds
        //                               H_i's (d, c, b, a) for j=0..3, w_e[start+j]
        //                               holds H_i's (h, g, f, e).
        //   - rows [start+4, start+68): 64 round-update outputs.
        //   - rows [start+64, start+68): "junction window" — w_a/w_e hold
        //                                internal_final_i; pa_a/pa_e hold a SECOND
        //                                copy of H_i so the feed-forward constraint
        //                                can read the prior init via `up.pa_a`.
        // After the last compression, rows [N·RPC, N·RPC+4) hold the H_N output
        // prefix, pinned by S_INIT_PREFIX in the same way.
        //
        // Slack rows [N·RPC + 4, n) are zero-padded; all SHA constraints
        // are inactive there (compensators absorb C7/C8/C9; selectors
        // gate off C13–C15 and the boundary/junction families).
        let big_n = cols::NUM_COMPRESSIONS;
        let rpc = cols::ROWS_PER_COMP;
        let rounds = cols::ROUNDS_PER_COMP;

        // Trace-row buffers, all length n, zero-initialized.
        let mut a_vals = vec![0u32; n];
        let mut e_vals = vec![0u32; n];
        let mut w_vals = vec![0u32; n];
        let mut k_vals = vec![0u32; n];
        let mut mu_w_vals = vec![0u32; n];
        let mut mu_a_vals = vec![0u32; n];
        let mut mu_e_vals = vec![0u32; n];
        let mut mu_junction_a_vals = vec![0u32; n];
        let mut mu_junction_e_vals = vec![0u32; n];

        // pa_a / pa_e: H_i values at init-prefix rows (gated by
        // S_INIT_PREFIX) AND at junction rows (read by the feed-forward
        // constraint). Both copies hold the same H_i values; they live
        // at different rows for different constraint uses.
        let mut pa_a_vals = vec![0u32; n];
        let mut pa_e_vals = vec![0u32; n];

        // H_0: random initial state for testing. Stored as two 4-arrays
        // (d, c, b, a) for the a-half and (h, g, f, e) for the e-half, in
        // the order they appear at the init prefix rows (so index j → row
        // `start + j` directly).
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

            // 1) Init prefix [start, start+4): pin to H_i.
            for j in 0..4 {
                a_vals[start + j] = h_a[j];
                e_vals[start + j] = h_e[j];
                pa_a_vals[start + j] = h_a[j];
                pa_e_vals[start + j] = h_e[j];
            }

            // 2) Per-compression message block. 16 random seeds, then 48
            //    derived via the SHA-256 message-schedule recurrence —
            //    contained entirely within compression i's window.
            for j in 0..16 {
                w_vals[start + j] = rng.next_u32();
            }
            for j in 16..rpc {
                let t = start + j;
                let sum_u64: u64 = (w_vals[t - 16] as u64)
                    + (small_sigma0(w_vals[t - 15]) as u64)
                    + (w_vals[t - 7] as u64)
                    + (small_sigma1(w_vals[t - 2]) as u64);
                w_vals[t] = sum_u64 as u32;
                let carry = (sum_u64 >> 32) as u32;
                debug_assert!(carry <= 3, "message-schedule carry out of [0,3]: {carry}");
                mu_w_vals[t] = carry;
            }

            // 3) Per-compression round constants. Random per-row (the
            //    slice does not pin to SHA-256's canonical K table —
            //    both prover and verifier see the same public column,
            //    sufficient for the round-trip). A "real SHA" upgrade
            //    would cycle the canonical 64-entry K table per
            //    compression.
            for j in 0..rpc {
                k_vals[start + j] = rng.next_u32();
            }

            // 4) Round-update: 64 rounds, anchor k = start+0..=start+63
            //    produces a[k+4]/e[k+4] from the 4-row window a[k..=k+3]
            //    / e[k..=k+3]. All back-references stay within
            //    compression i (the first round's reads land on the init
            //    prefix at [start, start+4); the last round's reads land
            //    on rows [start+60, start+64)).
            //
            //    Bounds: T1 = h + Σ_1(e) + Ch + K + W (5 terms of <2^32).
            //            T2 = Σ_0(a) + Maj (2 terms).
            //            a_sum = T1 + T2 (7 terms ⇒ mu_a ∈ {0..=6}).
            //            e_sum = d + T1   (6 terms ⇒ mu_e ∈ {0..=5}).
            for j in 0..rounds {
                let k = start + j;
                let t = k + 3; // spec round number under the t = k+3 anchor convention

                let a_t = a_vals[k + 3]; // a[t]
                let a_t1 = a_vals[k + 2]; // a[t-1] = b
                let a_t2 = a_vals[k + 1]; // a[t-2] = c
                let e_t = e_vals[k + 3]; // e[t]
                let e_t1 = e_vals[k + 2]; // e[t-1] = f
                let e_t2 = e_vals[k + 1]; // e[t-2] = g

                let sig0_a_t = big_sigma0(a_t);
                let sig1_e_t = big_sigma1(e_t);
                let ch_t = ch(e_t, e_t1, e_t2);
                let maj_t = maj(a_t, a_t1, a_t2);

                let t1: u64 = (e_vals[k] as u64) // h = e[t-3]
                    + (sig1_e_t as u64)
                    + (ch_t as u64)
                    + (k_vals[t] as u64)
                    + (w_vals[t] as u64);
                let t2: u64 = (sig0_a_t as u64) + (maj_t as u64);
                let a_sum: u64 = t1 + t2;
                let e_sum: u64 = (a_vals[k] as u64) + t1; // d + T1, d = a[t-3]

                a_vals[k + 4] = a_sum as u32;
                e_vals[k + 4] = e_sum as u32;
                let mu_a_t = (a_sum >> 32) as u32;
                let mu_e_t = (e_sum >> 32) as u32;
                debug_assert!(mu_a_t <= 6, "mu_a out of [0,6]: {mu_a_t}");
                debug_assert!(mu_e_t <= 5, "mu_e out of [0,5]: {mu_e_t}");
                // Convention: mu_a/mu_e at "spec row t = k+3" (not t+1),
                // matching the existing C8/C9 read pattern down.w_mu^↓3.
                mu_a_vals[t] = mu_a_t;
                mu_e_vals[t] = mu_e_t;
            }

            // 5) Feed-forward: H_{i+1} = internal_final_i + H_i mod 2^32
            //    componentwise. internal_final_i lives at rows
            //    [start+64, start+68); we place a second copy of H_i in
            //    pa_a/pa_e at the same rows (so the feed-forward
            //    constraint can read the prior init via `up.pa_a`), and
            //    record the per-component carry in w_mu_junction_{a,e}.
            //    Each carry is in {0, 1} since both summands are < 2^32.
            let mut h_a_next: [u32; 4] = [0; 4];
            let mut h_e_next: [u32; 4] = [0; 4];
            for j in 0..4 {
                let internal_a = a_vals[start + 64 + j];
                let internal_e = e_vals[start + 64 + j];
                let prior_a = h_a[j];
                let prior_e = h_e[j];
                let sum_a: u64 = (internal_a as u64) + (prior_a as u64);
                let sum_e: u64 = (internal_e as u64) + (prior_e as u64);
                h_a_next[j] = sum_a as u32;
                h_e_next[j] = sum_e as u32;
                let carry_a = (sum_a >> 32) as u32;
                let carry_e = (sum_e >> 32) as u32;
                debug_assert!(carry_a <= 1, "feed-forward a-carry out of {{0,1}}: {carry_a}");
                debug_assert!(carry_e <= 1, "feed-forward e-carry out of {{0,1}}: {carry_e}");

                pa_a_vals[start + 64 + j] = prior_a;
                pa_e_vals[start + 64 + j] = prior_e;
                mu_junction_a_vals[start + 64 + j] = carry_a;
                mu_junction_e_vals[start + 64 + j] = carry_e;
            }
            h_a = h_a_next;
            h_e = h_e_next;
        }

        // 6) H_N output prefix at rows [big_n·rpc, big_n·rpc + 4): pin
        //    to H_N (the final compression's output) so the verifier can
        //    read the digest from the public columns.
        let h_out_start = big_n * rpc;
        for j in 0..4 {
            a_vals[h_out_start + j] = h_a[j];
            e_vals[h_out_start + j] = h_e[j];
            pa_a_vals[h_out_start + j] = h_a[j];
            pa_e_vals[h_out_start + j] = h_e[j];
        }

        // ===== Per-row Ch / Maj operand witnesses & B_i materializations =====
        //
        // Computed honestly on every row from a_vals / e_vals contents.
        // At rows where C13–C15 are gated off (s_b_active = 0, including
        // junction windows and slack), the values still need to be valid
        // bit-polynomials (booleanity), which `& / | / ^` on u32 ensures.
        let u_ef_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 1 { e_vals[t] & e_vals[t - 1] } else { 0 })
            .collect();
        let u_neg_e_g_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { (!e_vals[t]) & e_vals[t - 2] } else { 0 })
            .collect();
        let maj_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { maj(a_vals[t], a_vals[t - 1], a_vals[t - 2]) } else { 0 })
            .collect();

        let b1_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 1 { e_vals[t] ^ e_vals[t - 1] } else { 0 })
            .collect();
        let b2_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { (!e_vals[t]) ^ e_vals[t - 2] } else { 0 })
            .collect();
        let b3_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { a_vals[t] ^ a_vals[t - 1] ^ a_vals[t - 2] } else { 0 })
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

        let to_bits = |v: &[u32]| -> Vec<BinaryPoly<32>> {
            v.iter().copied().map(BinaryPoly::<32>::from).collect()
        };

        let to_bin_mle = |col: Vec<BinaryPoly<32>>| -> DenseMultilinearExtension<
            BinaryPoly<32>,
        > { col.into_iter().collect() };

        // Layout: 6 public bin_poly cols (PA_A, PA_E, PA_OV_SIG0,
        // PA_OV_SIG1, PA_OV_LSIG0, PA_OV_LSIG1) + 17 witness cols.
        // pa_a / pa_e were populated above with H_i values at init-prefix
        // rows (for compression i and the H_N output block) AND at junction
        // rows (where the feed-forward constraint reads the prior H_i).
        let binary_poly = vec![
            to_bin_mle(to_bits(&pa_a_vals)),
            to_bin_mle(to_bits(&pa_e_vals)),
            to_bin_mle(to_bits(&ov_sig0_vals)),
            to_bin_mle(to_bits(&ov_sig1_vals)),
            to_bin_mle(to_bits(&ov_lsig0_vals)),
            to_bin_mle(to_bits(&ov_lsig1_vals)),
            to_bin_mle(to_bits(&a_vals)),
            to_bin_mle(to_bits(&sig0_vals)),
            to_bin_mle(to_bits(&e_vals)),
            to_bin_mle(to_bits(&sig1_vals)),
            to_bin_mle(to_bits(&w_vals)),
            to_bin_mle(to_bits(&lsig0_vals)),
            to_bin_mle(to_bits(&s0_vals)),
            to_bin_mle(to_bits(&t0_vals)),
            to_bin_mle(to_bits(&lsig1_vals)),
            to_bin_mle(to_bits(&s1_vals)),
            to_bin_mle(to_bits(&t1_vals)),
            to_bin_mle(to_bits(&u_ef_vals)),
            to_bin_mle(to_bits(&u_neg_e_g_vals)),
            to_bin_mle(to_bits(&maj_vals)),
            to_bin_mle(to_bits(&b1_vals)),
            to_bin_mle(to_bits(&b2_vals)),
            to_bin_mle(to_bits(&b3_vals)),
        ];

        // ===== Selectors =====
        //
        // s_init_prefix: 1 on the init-prefix windows for every compression
        //                (4 rows × NUM_COMPRESSIONS) plus the H_N output
        //                block (4 more rows). Pins w_a / w_e to pa_a / pa_e.
        // s_feedforward: 1 on the junction windows [start+64, start+68) for
        //                every compression. Gates the SHA-256 inter-
        //                compression addition constraint.
        let mut s_init_prefix_col: Vec<R> = (0..n).map(|_| R::ZERO).collect();
        for i in 0..=big_n {
            // i = big_n: the H_N output block.
            for j in 0..4 {
                s_init_prefix_col[i * rpc + j] = R::ONE;
            }
        }
        let mut s_feedforward_col: Vec<R> = (0..n).map(|_| R::ZERO).collect();
        for i in 0..big_n {
            for j in 0..4 {
                s_feedforward_col[i * rpc + 64 + j] = R::ONE;
            }
        }

        let k_col: Vec<R> = k_vals.iter().copied().map(R::from).collect();
        let mu_w_col: Vec<R> = mu_w_vals.iter().copied().map(R::from).collect();
        let mu_a_col: Vec<R> = mu_a_vals.iter().copied().map(R::from).collect();
        let mu_e_col: Vec<R> = mu_e_vals.iter().copied().map(R::from).collect();
        let mu_junction_a_col: Vec<R> =
            mu_junction_a_vals.iter().copied().map(R::from).collect();
        let mu_junction_e_col: Vec<R> =
            mu_junction_e_vals.iter().copied().map(R::from).collect();

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

        // C8: inner(2) = w_a[k+4] − w_e[k] − sig1[k+3] − Ch[k+3] − K[k+3]
        //               − W[k+3] − sig0[k+3] − maj[k+3] + 2^32 · mu_a[k+3]
        // with Ch[k+3] = u_ef[k+3] + u_{¬e,g}[k+3].
        let pa_c_c8_col: Vec<R> = (0..n)
            .map(|k| {
                let w_a_k4 = load(&a_vals, k + 4);
                let w_e_k = load(&e_vals, k);
                let sig1_k3 = load(&sig1_vals, k + 3);
                let u_ef_k3 = load(&u_ef_vals, k + 3);
                let u_neg_e_g_k3 = load(&u_neg_e_g_vals, k + 3);
                let k_k3 = load(&k_vals, k + 3);
                let w_k3 = load(&w_vals, k + 3);
                let sig0_k3 = load(&sig0_vals, k + 3);
                let maj_k3 = load(&maj_vals, k + 3);
                let mu_a_k3 = load(&mu_a_vals, k + 3);
                let two32_mu = two_to_32.clone() * &mu_a_k3;
                w_e_k
                    + &sig1_k3
                    + &u_ef_k3
                    + &u_neg_e_g_k3
                    + &k_k3
                    + &w_k3
                    + &sig0_k3
                    + &maj_k3
                    - &two32_mu
                    - &w_a_k4
            })
            .collect();

        // C9: inner(2) = w_e[k+4] − w_a[k] − w_e[k] − sig1[k+3] − Ch[k+3]
        //               − K[k+3] − W[k+3] + 2^32 · mu_e[k+3]
        // with Ch[k+3] = u_ef[k+3] + u_{¬e,g}[k+3].
        let pa_c_c9_col: Vec<R> = (0..n)
            .map(|k| {
                let w_e_k4 = load(&e_vals, k + 4);
                let w_a_k = load(&a_vals, k);
                let w_e_k = load(&e_vals, k);
                let sig1_k3 = load(&sig1_vals, k + 3);
                let u_ef_k3 = load(&u_ef_vals, k + 3);
                let u_neg_e_g_k3 = load(&u_neg_e_g_vals, k + 3);
                let k_k3 = load(&k_vals, k + 3);
                let w_k3 = load(&w_vals, k + 3);
                let mu_e_k3 = load(&mu_e_vals, k + 3);
                let two32_mu = two_to_32.clone() * &mu_e_k3;
                w_a_k
                    + &w_e_k
                    + &sig1_k3
                    + &u_ef_k3
                    + &u_neg_e_g_k3
                    + &k_k3
                    + &w_k3
                    - &two32_mu
                    - &w_e_k4
            })
            .collect();

        // C12/C13 feed-forward compensators. inner_a(2) at row k =
        //   w_a[k+4] − w_a[k] − pa_a[k] + 2^32 · mu_junction_a[k]
        // (e-half symmetric). On junction rows the SHA-256 feed-forward
        // makes inner = 0 honestly, so the compensator is 0. Off-
        // junction (init prefix straddle, round-update windows, slack)
        // it absorbs whatever inner happens to be so that
        // `(inner + pa_c_ff) ∈ (X − 2)` everywhere.
        let pa_c_ff_a_col: Vec<R> = (0..n)
            .map(|k| {
                let w_a_k4 = load(&a_vals, k + 4);
                let w_a_k = load(&a_vals, k);
                let pa_a_k = load(&pa_a_vals, k);
                let mu_ff_k = load(&mu_junction_a_vals, k);
                let two32_mu = two_to_32.clone() * &mu_ff_k;
                // comp = −inner(2) = w_a_k + pa_a_k − 2^32·mu_ff_k − w_a_k4
                w_a_k + &pa_a_k - &two32_mu - &w_a_k4
            })
            .collect();
        let pa_c_ff_e_col: Vec<R> = (0..n)
            .map(|k| {
                let w_e_k4 = load(&e_vals, k + 4);
                let w_e_k = load(&e_vals, k);
                let pa_e_k = load(&pa_e_vals, k);
                let mu_ff_k = load(&mu_junction_e_vals, k);
                let two32_mu = two_to_32.clone() * &mu_ff_k;
                w_e_k + &pa_e_k - &two32_mu - &w_e_k4
            })
            .collect();

        // s_b_active: 1 on per-compression anchor rows k where the
        // length-2 forward shifts of C13–C15 stay within the same
        // compression's 68-row window: k ∈ [start, start + 66) for each
        // compression i (so reads at rows k+1, k+2 ≤ start + 67 land
        // inside compression i). 0 elsewhere — the last two rows of
        // every compression, the H_N output prefix, and slack.
        let mut s_b_active_col: Vec<R> = (0..n).map(|_| R::ZERO).collect();
        for i in 0..big_n {
            let start = i * rpc;
            for k in start..(start + rpc - 2) {
                s_b_active_col[k] = R::ONE;
            }
        }

        let to_int_mle = |col: Vec<R>| -> DenseMultilinearExtension<R> {
            col.into_iter().collect()
        };
        // Layout matches cols::S_INIT_PREFIX..NUM_INT order.
        let int = vec![
            to_int_mle(s_init_prefix_col),
            to_int_mle(s_feedforward_col),
            to_int_mle(k_col),
            to_int_mle(pa_c_c7_col),
            to_int_mle(pa_c_c8_col),
            to_int_mle(pa_c_c9_col),
            to_int_mle(pa_c_ff_a_col),
            to_int_mle(pa_c_ff_e_col),
            to_int_mle(s_b_active_col),
            to_int_mle(mu_w_col),
            to_int_mle(mu_a_col),
            to_int_mle(mu_e_col),
            to_int_mle(mu_junction_a_col),
            to_int_mle(mu_junction_e_col),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_uair::degree_counter::{count_effective_max_degree, count_max_degree};

    /// All non-zero-ideal SHA constraints (C1, C2, C4, C6, C7, C8, C9,
    /// and the new feed-forward C12/C13) must stay degree-1 in the
    /// trace MLEs so the bench's MLE-first dispatch gate
    /// (`count_effective_max_degree::<U>() <= 1`) keeps firing for the
    /// standalone SHA UAIR. Asserts on the chained-compression layout
    /// with public-compensator gating for the feed-forward.
    #[test]
    fn sha_uair_is_mle_first_eligible() {
        type U = Sha256CompressionSliceUair<Int<5>>;
        assert_eq!(count_effective_max_degree::<U>(), 1);
        // assert_zero constraints (init-prefix pinning, B_i materializations)
        // can have higher degree without disqualifying MLE-first.
        assert!(count_max_degree::<U>() >= 2);
    }
}
