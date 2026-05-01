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
//! 4. `sigma_0` row-local Q[X] equality (with bit-XOR overflow):
//!    `ROT^25(W) + ROT^14(W) + SHIFTR^3(W) − lsig0_hat − 2·ov_lsig0 == 0`
//!    where `ROT^c(W)` and `SHIFTR^c(W)` are CPR bit-op virtual columns
//!    over `W` (declared in `signature()`'s `bit_op_specs`). Replaces
//!    the previous `(X^32 − 1)` ideal lift and the C3 right-shift
//!    decomposition constraint: ROT/SHIFTR virtual columns are mod
//!    `X^32` by construction.
//! 6. `sigma_1` row-local Q[X] equality:
//!    `ROT^15(W) + ROT^13(W) + SHIFTR^10(W) − lsig1_hat − 2·ov_lsig1 == 0`
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
//! The `rho_*` scalars encode the rotation parts of the SHA-256 Σ
//! functions (Σ_0 and Σ_1 still use the F_2[X] → Q[X] rotation lift;
//! σ_0/σ_1 instead route through CPR bit-op virtual columns — see
//! below):
//!
//! - `rho_Sig0(X) = X^30 + X^19 + X^10`     (= `X^{32-2} + X^{32-13} + X^{32-22}`)
//! - `rho_Sig1(X) = X^26 + X^21 + X^7`      (= `X^{32-6} + X^{32-11} + X^{32-25}`)
//!
//! The full σ_0/σ_1 expressions
//! `sigma_0(W) = ROTR(W, 7) ⊕ ROTR(W, 18) ⊕ SHR(W, 3)` and
//! `sigma_1(W) = ROTR(W, 17) ⊕ ROTR(W, 19) ⊕ SHR(W, 10)`
//! are materialized via CPR bit-op virtual columns over `W`: each
//! `ROTR^c(W)` is `BitOp::Rot(32 − c)` (multiplication by `X^{32-c}
//! mod (X^32 − 1)` in F_2[X]), and each `SHR(W, k)` is
//! `BitOp::ShiftR(k)`. Six bit-op virtual columns total; no committed
//! `S_i` / `T_i` operands.
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
//! coefficient at 3, giving overflow `∈ {0, 1}`. For σ_0 / σ_1
//! (`ROT^a(W) + ROT^b(W) + SHIFTR^c(W)`, 3 bit-poly terms), the per-
//! coefficient sum is in `{0, 1, 2, 3}`, again with overflow `∈ {0, 1}`.
//! See [`sigma0_overflow`] for the full Sig0 derivation; the others are
//! analogous.
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
//! ## Ch / Maj enforcement (Table 9 virtual binary_poly residuals)
//!
//! `Ch(e, f, g) = (e ∧ f) ⊕ (¬e ∧ g)` is split into two operand bit-polys
//! `u_ef = e ∧ f` and `u_{¬e,g} = ¬e ∧ g`. Since the two AND terms can never
//! both have a bit set at the same position, `Ch = u_ef + u_{¬e,g}`
//! coefficient-wise — so the C8/C9 register-update constraints simply
//! replace the old `Ch[t]` reference with `u_ef[t] + u_{¬e,g}[t]`.
//!
//! Per Table 9 of the spec, three affine combinations must lie in
//! `{0, 1}^32` per coefficient. We declare these as **virtual binary_poly
//! columns** anchored at `k = t − 2` (so all references are forward shifts
//! of `e, a, u_ef, u_{¬e,g}, Maj`), and pin them by the same booleanity
//! sumcheck that handles genuine witness binary_poly cols. No committed
//! `B_i` columns, no materialization identity — the virtual MLE
//! definition is itself the affine combination, and booleanity on its
//! per-bit slices forces it into `{0, 1}^32` per coefficient:
//!
//! - `r_ch1[k] = e[k+2] + e[k+1] − 2·u_ef[k+2]`                   (Ch eq 62)
//! - `r_ch2[k] = e[k+2] − e[k]   + 2·u_{¬e,g}[k+2] + 2·comp_ch2[k]`
//!     — alt complement form of `(1₃₂ − e[t]) + e[t−2] − 2·u_{¬e,g}[t]`;
//!     per coefficient `r_orig + r_alt = 1`, so both are bit-valid
//!     simultaneously and the `1₃₂` constant is dropped from the
//!     virtual-source enum                                        (Ch eq 63)
//! - `r_maj[k] = a[k] + a[k+1] + a[k+2] − 2·Maj[k+2] − 2·comp_maj[k]`
//!                                                                (Maj eq 64)
//!
//! With `u_ef`, `u_{¬e,g}`, `Maj` set on every row by their truth tables
//! (the trace builder does this unconditionally), per-row residuals
//! collapse to `XOR` / `MAJ-XOR` patterns that lie in `{0, 1}` per
//! coefficient — including across compression-junction boundaries where
//! the length-2 forward shifts read into the next compression. The only
//! rows that fall outside `{0, 1}` are the last two rows of the trace
//! (`k ∈ {n−2, n−1}`), where the forward shifts pull zero-padded values
//! and `−e[k] / a[k]+a[k+1]` slip negative or to 2; the public
//! compensators `PA_R_CH2_COMP` / `PA_R_MAJ_COMP` carry the bit pattern
//! that absorbs the off-trace zero-padding (zero on every other row).
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
    BitOp, BitOpSpec, ConstraintBuilder, LookupColumnSpec, PublicColumnLayout,
    PublicStructureError, ShiftSpec, ShiftedBitSliceSpec, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace, VirtualBinaryPolySource, VirtualBinaryPolySpec,
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
    // binary_poly, public prefix of length 8.
    //
    // The 4 PA_OV_* columns are the per-coefficient mod-2 overflow
    // witnesses for the rotation-ideal constraints (C1, C2, C4, C6).
    // They're "quotient witnesses" for the F_2[X] → Q[X] lift —
    // verifier-supplied, since the verifier can derive them from the
    // bit-poly values when shadow-running the SHA computation.
    //
    // The two PA_R_*_COMP columns are public compensators for the Ch (63)
    // and Maj (64) virtual binary_poly residuals (see module doc): zero
    // on all rows except the last two (`k ∈ {n−2, n−1}`), where the
    // length-2 forward shifts pull zero-padding and the residual
    // arithmetic would otherwise leave `{0,1}` per coefficient. The
    // verifier can derive both compensators from the public e/a values
    // when shadow-running.
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    pub const PA_OV_SIG0: usize = 2;
    pub const PA_OV_SIG1: usize = 3;
    pub const PA_OV_LSIG0: usize = 4;
    pub const PA_OV_LSIG1: usize = 5;
    pub const PA_R_CH2_COMP: usize = 6;
    pub const PA_R_MAJ_COMP: usize = 7;
    // Public message-block words. Holds the 16 message words M_i[0..16]
    // of compression i ∈ [0, NUM_COMPRESSIONS) at rows
    // [ROWS_PER_COMP·i, ROWS_PER_COMP·i + 16); zero elsewhere. Pinned
    // to w_W at those rows by the C16 message-init constraint, gated
    // by S_MSG_INIT. Implements Table 9 row (77).
    pub const PA_M: usize = 8;
    // binary_poly, witness suffix. Grouped by constraint family for clarity.
    // Sigma_0:
    pub const W_A: usize = 9;
    pub const W_SIG0: usize = 10;
    // Sigma_1:
    pub const W_E: usize = 11;
    pub const W_SIG1: usize = 12;
    // sigma_0 / sigma_1 — shares W_W. The σ_0/σ_1 right-shift
    // decomposition columns (S_0/T_0/S_1/T_1) are gone: ROT^c(W) and
    // SHIFTR^c(W) are now CPR bit-op virtual columns over W (declared in
    // signature()'s `bit_op_specs`), so σ_0/σ_1 reduce to row-local
    // Q[X] equalities with the `pa_ov_lsig{0,1}` XOR-overflow witnesses
    // retained.
    pub const W_W: usize = 13;
    pub const W_LSIG0: usize = 14;
    pub const W_LSIG1: usize = 15;
    // Register update — Ch is replaced by two AND-operand bit-polys
    // (`u_ef = e ∧ f` and `u_{¬e,g} = ¬e ∧ g`). Maj is still a free
    // witness on this column. See the module doc for the Table 9 split.
    pub const W_U_EF: usize = 16;
    pub const W_U_NEG_E_G: usize = 17;
    pub const W_MAJ: usize = 18;
    // Packed integer-carry witness column. Replaces the five prior int
    // carry columns (W_MU_W/A/E/JUNCTION_A/E) with a single
    // booleanity-checked binary_poly column. Bit layout per row k
    // (= constraint anchor row):
    //
    //     bits 0-1: mu_W   (carry from C7's message-schedule recurrence
    //                       at spec round t = k+16; honest in [0, 3])
    //     bits 2-4: mu_a   (carry from C8's a-update at spec round
    //                       t = k+3; honest in [0, 6])
    //     bits 5-7: mu_e   (carry from C9's e-update at spec round
    //                       t = k+3; honest in [0, 5])
    //     bit 8:    mu_ff_a (junction-window carry; honest in {0, 1})
    //     bit 9:    mu_ff_e (junction-window carry; honest in {0, 1})
    //     bits 10-31: must be 0 (pinned by C22)
    //
    // Range-check soundness: booleanity gives each bit ∈ {0, 1} for
    // free. For non-power-of-2 honest ranges (mu_a ≤ 6, mu_e ≤ 5),
    // booleanity allows mu_a = 7 / mu_e ∈ {6, 7}, but those would
    // force a[k+4]/e[k+4] = honest_sum − k·2^32 < 0, wrapping in F_p
    // to a non-32-bit value — caught by w_a/w_e booleanity. Pure
    // bit-width is sufficient soundness.
    pub const W_MU_PACKED: usize = 19;
    // The Table 9 affine combinations B_1 / B_2 / B_3 are no longer
    // committed — they're declared as packed virtual binary_poly
    // columns in signature() via `with_virtual_binary_poly_cols`,
    // pinned by the booleanity sumcheck (per-bit closing override).

    /// Total number of binary_poly columns.
    pub const NUM_BIN: usize = 20;
    /// Number of public binary_poly columns (prefix).
    pub const NUM_BIN_PUB: usize = 9;

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
    // Selector for the message-init constraint (C16). 1 on rows
    // [ROWS_PER_COMP·i, ROWS_PER_COMP·i + 16) for each compression
    // i ∈ [0, NUM_COMPRESSIONS) — the 16 trace rows where w_W must
    // equal the public message word PA_M[k]; 0 elsewhere.
    pub const S_MSG_INIT: usize = 2;
    pub const PA_K: usize = 3; // public: round constants column (free for this slice)
    // Public compensator columns: zero on rows where the corresponding
    // constraint is honestly satisfied (the "active" range, a union of
    // NUM_COMPRESSIONS disjoint per-compression windows), non-zero on
    // inactive rows so that `inner + compensator ∈ (X − 2)` everywhere.
    // Keeps C7-C9 at degree 1 in the trace MLEs.
    //
    // Compensator-zero on active rows is enforced by **direct
    // verifier inspection** of public_trace via
    // `Uair::verify_public_structure`, not by an in-circuit
    // selector-multiplied constraint. The compensators are public
    // columns — the verifier holds every cell value before the proof
    // begins, so a row-wise check costs no additional column and no
    // algebraic constraint. See the SHA UAIR document for the full
    // list of structural checks.
    //
    // Per-compression active windows (relative to start = ROWS_PER_COMP·i):
    //   - SCHED (C7):     k ∈ [start, start + 48), i.e., the 48 anchors
    //                     where the message-schedule recurrence
    //                     produces W[k+16] inside compression i's
    //                     16..=63 derived range.
    //   - UPD (C8/C9):    k ∈ [start, start + 64), i.e., the 64
    //                     round-update anchors per compression.
    //   - JUNCTION (C12/C13): k ∈ [start + 64, start + 68), the 4
    //                     junction-window rows.
    pub const PA_C_C7: usize = 4; // compensator for C7 (sched_anch)
    pub const PA_C_C8: usize = 5; // compensator for C8 (upd_anch a)
    pub const PA_C_C9: usize = 6; // compensator for C9 (upd_anch e)
    pub const PA_C_FF_A: usize = 7; // compensator for C12 (feed-forward a-half)
    pub const PA_C_FF_E: usize = 8; // compensator for C13 (feed-forward e-half)
    /// Total number of int columns.
    ///
    /// The 5 prior witness int carry columns (W_MU_W/A/E/JUNCTION_A/E)
    /// are gone — replaced by the W_MU_PACKED binary_poly column at
    /// index 19 in the binary_poly section. Booleanity provides the
    /// range checks for free (see W_MU_PACKED's doc).
    ///
    /// The 2 prior compensator-zero selector columns
    /// (S_ACTIVE_SCHED, S_ACTIVE_UPD) are gone — compensator-zero is
    /// enforced by direct verifier inspection of public_trace.
    pub const NUM_INT: usize = 9;
    /// Number of public int columns (prefix). All ints are public.
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
    pub const FLAT_PA_K: usize = NUM_BIN + PA_K;
    pub const FLAT_W_MU_PACKED: usize = W_MU_PACKED;
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
            // w_a: shifts 1, 2 for the Maj virtual residual (r_maj
            // anchored at k = t-2); shift 4 for target a[t+1] in C8.
            ShiftSpec::new(cols::FLAT_W_A, 1),
            ShiftSpec::new(cols::FLAT_W_A, 2),
            ShiftSpec::new(cols::FLAT_W_A, 4),
            // w_sig0: Sigma_0(a[t]) at C8 anchor t-3.
            ShiftSpec::new(cols::FLAT_W_SIG0, 3),
            // w_e: shifts 1, 2 for the Ch virtual residuals (r_ch1 /
            // r_ch2 anchored at k = t-2); shift 4 for target e[t+1] in C9.
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
            // w_u_ef: shift 2 for r_ch1 (anchor k = t-2); shift 3 for
            // the Ch[t] reference in C8/C9 (anchor t-3), where Ch is
            // the sum u_ef + u_{¬e,g} coefficient-wise.
            ShiftSpec::new(cols::FLAT_W_U_EF, 2),
            ShiftSpec::new(cols::FLAT_W_U_EF, 3),
            // w_u_neg_e_g: shift 2 for r_ch2; shift 3 for C8/C9.
            ShiftSpec::new(cols::FLAT_W_U_NEG_E_G, 2),
            ShiftSpec::new(cols::FLAT_W_U_NEG_E_G, 3),
            // w_maj: shift 2 for r_maj; shift 3 for Maj[t] in C8.
            ShiftSpec::new(cols::FLAT_W_MAJ, 2),
            ShiftSpec::new(cols::FLAT_W_MAJ, 3),
            // === int shifts (in source_col ascending order) ===
            ShiftSpec::new(cols::FLAT_PA_K, 3),
            // The 3 prior int shifts on W_MU_W/A/E are gone alongside
            // the dropped int carry columns — C7/C8/C9 read all 5
            // carries from W_MU_PACKED at the up row via the
            // BitOp::ShiftR virtuals declared below.
        ];
        // No explicit `LookupColumnSpec`s. Booleanity already enforces
        // the bit-poly property on every witness binary_poly column for
        // free, which is precisely what a `BitPoly { width: 32 }`
        // lookup would assert — making explicit lookup specs redundant.
        // Mu_{W,a,e} are int range-check candidates (Word{width:2/3})
        // and would need a different table type, but the protocol's
        // lookup pipeline currently only handles BitPoly.
        let lookup_specs: Vec<LookupColumnSpec> = Vec::new();
        // Bit-op virtual columns over W for σ_0/σ_1. `Rot(c)` on the
        // 32-coefficient F_2[X] cell is multiplication by `X^c mod
        // (X^32 − 1)`, which equals `rotate_left(c)` on the underlying
        // u32 — equivalently `ROTR^{32-c}`. So:
        //   ROTR^7  on W = Rot(25)        ROTR^17 on W = Rot(15)
        //   ROTR^18 on W = Rot(14)        ROTR^19 on W = Rot(13)
        //   SHR^3   on W = ShiftR(3)      SHR^10  on W = ShiftR(10)
        // The σ_0/σ_1 constraints (C4/C6) consume these via
        // `down.bit_op` and the (X^32 − 1) modular lift goes away:
        // ROT/SHIFTR virtual columns are mod X^32 by construction.
        let bit_op_specs: Vec<BitOpSpec> = vec![
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(25)),    // σ_0: ROTR^7
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(14)),    // σ_0: ROTR^18
            BitOpSpec::new(cols::FLAT_W_W, BitOp::ShiftR(3)),  // σ_0: SHR^3
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(15)),    // σ_1: ROTR^17
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(13)),    // σ_1: ROTR^19
            BitOpSpec::new(cols::FLAT_W_W, BitOp::ShiftR(10)), // σ_1: SHR^10
            // Bit-op virtuals over W_MU_PACKED for extracting each
            // carry from its bit slice. The C7/C8/C9/C12/C13
            // contributions of `2^32 · mu_X` are reconstructed via
            // `2^32 · ShiftR(k_low) − 2^{32+w} · ShiftR(k_low+w)`.
            // ShiftR(10) is also assert_zero'd by C22 to pin
            // positions 10..31 to 0.
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(2)),  // skips mu_W
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(5)),  // skips mu_W + mu_a
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(8)),  // skips through mu_e
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(9)),  // keeps mu_ff_e + (high)
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(10)), // (high) — must be 0
        ];
        // Witness-relative col indices (post-public) for virtual specs.
        const W_A_WIT_IDX: usize = cols::W_A - cols::NUM_BIN_PUB; // 0
        const W_E_WIT_IDX: usize = cols::W_E - cols::NUM_BIN_PUB; // 2
        const W_U_EF_WIT_IDX: usize = cols::W_U_EF - cols::NUM_BIN_PUB; // 7
        const W_U_NEG_E_G_WIT_IDX: usize = cols::W_U_NEG_E_G - cols::NUM_BIN_PUB; // 8
        const W_MAJ_WIT_IDX: usize = cols::W_MAJ - cols::NUM_BIN_PUB; // 9
        // Order here is the spec_idx that
        // `VirtualBinaryPolySource::ShiftedWitnessCol` references — must
        // match the corresponding `ShiftSpec` ordering above (sorted by
        // (source_col, shift_amount) inside `UairSignature::new`).
        const SBS_W_A_SH1: usize = 0;
        const SBS_W_A_SH2: usize = 1;
        const SBS_W_E_SH1: usize = 2;
        const SBS_W_E_SH2: usize = 3;
        const SBS_W_U_EF_SH2: usize = 4;
        const SBS_W_U_NEG_E_G_SH2: usize = 5;
        const SBS_W_MAJ_SH2: usize = 6;
        let shifted_bit_slice_specs = vec![
            ShiftedBitSliceSpec::new(W_A_WIT_IDX, 1),
            ShiftedBitSliceSpec::new(W_A_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_E_WIT_IDX, 1),
            ShiftedBitSliceSpec::new(W_E_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_U_EF_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_U_NEG_E_G_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_MAJ_WIT_IDX, 2),
        ];
        // Virtual binary_poly cols — Table 9 (62)/(63)/(64) anchored at
        // k = t-2. See module doc for the residual definitions and the
        // alt-complement form for r_ch2 (drops the `1₃₂` constant).
        let virtual_binary_poly_cols = vec![
            // r_ch1[k] = e[k+2] + e[k+1] − 2·u_ef[k+2]   (Ch eq 62)
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH2,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH1,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_U_EF_SH2,
                        },
                    ),
                ],
            },
            // r_ch2[k] (alt) = e[k+2] − e[k] + 2·u_{¬e,g}[k+2] + 2·comp_ch2[k]
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH2,
                        },
                    ),
                    (
                        -1,
                        VirtualBinaryPolySource::SelfWitnessCol {
                            witness_col_idx: W_E_WIT_IDX,
                        },
                    ),
                    (
                        2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_U_NEG_E_G_SH2,
                        },
                    ),
                    (
                        2,
                        VirtualBinaryPolySource::PublicCol {
                            public_col_idx: cols::PA_R_CH2_COMP,
                        },
                    ),
                ],
            },
            // r_maj[k] = a[k] + a[k+1] + a[k+2] − 2·Maj[k+2] − 2·comp_maj[k]
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::SelfWitnessCol {
                            witness_col_idx: W_A_WIT_IDX,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_A_SH1,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_A_SH2,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_MAJ_SH2,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::PublicCol {
                            public_col_idx: cols::PA_R_MAJ_COMP,
                        },
                    ),
                ],
            },
        ];
        UairSignature::new(total, public, shifts, lookup_specs, bit_op_specs)
            .with_shifted_bit_slice_specs(shifted_bit_slice_specs)
            .with_virtual_binary_poly_cols(virtual_binary_poly_cols)
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
        let pa_ov_sig0 = &bp[cols::PA_OV_SIG0];
        let pa_ov_sig1 = &bp[cols::PA_OV_SIG1];
        let pa_ov_lsig0 = &bp[cols::PA_OV_LSIG0];
        let pa_ov_lsig1 = &bp[cols::PA_OV_LSIG1];
        let pa_m = &bp[cols::PA_M];
        let w_a = &bp[cols::W_A];
        let w_sig0 = &bp[cols::W_SIG0];
        let w_e = &bp[cols::W_E];
        let w_sig1 = &bp[cols::W_SIG1];
        let w_big_w = &bp[cols::W_W];
        let w_lsig0 = &bp[cols::W_LSIG0];
        let w_lsig1 = &bp[cols::W_LSIG1];
        // Packed-carry binary_poly column. Bit layout per row:
        //   0-1: mu_W,  2-4: mu_a,  5-7: mu_e,  8: mu_ff_a,  9: mu_ff_e,
        //   10-31: must be zero (pinned by C22). Each carry contribution
        //   `2^32 · mu_X` to C7/C8/C9/C12/C13 is reconstructed from the
        //   ShiftR virtuals on this column.
        let w_mu_packed = &bp[cols::W_MU_PACKED];

        let s_init_prefix = &sel[cols::S_INIT_PREFIX];
        // S_FEEDFORWARD remains as a public selector documenting the
        // junction-window row pattern (1 on junction windows, 0
        // elsewhere); it is no longer multiplied into any in-circuit
        // constraint since the FF compensator-zero pins (formerly
        // C20/C21) moved to verifier-side `verify_public_structure`.
        let _s_feedforward = &sel[cols::S_FEEDFORWARD];
        let s_msg_init = &sel[cols::S_MSG_INIT];
        let pa_c_c7 = &sel[cols::PA_C_C7];
        let pa_c_c8 = &sel[cols::PA_C_C8];
        let pa_c_c9 = &sel[cols::PA_C_C9];
        let pa_c_ff_a = &sel[cols::PA_C_FF_A];
        let pa_c_ff_e = &sel[cols::PA_C_FF_E];
        // The 5 prior int carry columns (W_MU_W/A/E/JUNCTION_A/E) are
        // gone — replaced by W_MU_PACKED (binary_poly) read above. The
        // `mu_*_contrib` polynomial expressions below extract each
        // carry's `2^32 · mu_X` contribution from the packed bits.

        // `down` slot layout (mirrors the ShiftSpec order in signature()).
        // bin slots:
        // sh1 / sh2 entries are kept in `signature()`'s shift list for
        // the virtual binary_poly residuals (declared via
        // `with_shifted_bit_slice_specs`); they're consumed by the
        // booleanity batch's shifted bit-slice consistency check, not
        // by `constrain_general` directly. Hence the `_` prefix.
        let _down_w_a_sh1 = &down.binary_poly[0];
        let _down_w_a_sh2 = &down.binary_poly[1];
        let down_w_a_sh4 = &down.binary_poly[2];
        let down_w_sig0_sh3 = &down.binary_poly[3];
        let _down_w_e_sh1 = &down.binary_poly[4];
        let _down_w_e_sh2 = &down.binary_poly[5];
        let down_w_e_sh4 = &down.binary_poly[6];
        let down_w_sig1_sh3 = &down.binary_poly[7];
        let down_w_w_sh3 = &down.binary_poly[8];
        let down_w_w_sh9 = &down.binary_poly[9];
        let down_w_w_sh16 = &down.binary_poly[10];
        let down_w_lsig0_sh1 = &down.binary_poly[11];
        let down_w_lsig1_sh14 = &down.binary_poly[12];
        let _down_w_u_ef_sh2 = &down.binary_poly[13];
        let down_w_u_ef_sh3 = &down.binary_poly[14];
        let _down_w_u_neg_e_g_sh2 = &down.binary_poly[15];
        let down_w_u_neg_e_g_sh3 = &down.binary_poly[16];
        let _down_w_maj_sh2 = &down.binary_poly[17];
        let down_w_maj_sh3 = &down.binary_poly[18];
        // int slots: only PA_K survives (the 3 mu_* int shift specs
        // are gone alongside the dropped int carry columns).
        let down_pa_k_sh3 = &down.int[0];

        // Bit-op virtual columns sorted by (source_col, op_kind, c)
        // inside `UairSignature::new`. With FLAT_W_W < FLAT_W_MU_PACKED,
        // W's 6 bit-ops occupy slots 0-5, then W_MU_PACKED's 5
        // bit-ops occupy slots 6-10.
        let down_w_rot13 = &down.bit_op[0]; // σ_1: ROTR^19
        let down_w_rot14 = &down.bit_op[1]; // σ_0: ROTR^18
        let down_w_rot15 = &down.bit_op[2]; // σ_1: ROTR^17
        let down_w_rot25 = &down.bit_op[3]; // σ_0: ROTR^7
        let down_w_shr3 = &down.bit_op[4]; //  σ_0: SHR^3
        let down_w_shr10 = &down.bit_op[5]; // σ_1: SHR^10
        // Bit-extraction shifts on W_MU_PACKED.
        let down_w_mu_packed_shr2 = &down.bit_op[6];   // skips mu_W
        let down_w_mu_packed_shr5 = &down.bit_op[7];   // skips mu_W + mu_a
        let down_w_mu_packed_shr8 = &down.bit_op[8];   // skips through mu_e
        let down_w_mu_packed_shr9 = &down.bit_op[9];   // keeps only mu_ff_e + (high)
        let down_w_mu_packed_shr10 = &down.bit_op[10]; // (high) — must be 0

        // Ideals.
        let ideal_rot_xw1 = ideal_from_ref(&Sha256Ideal::<R>::RotXw1);
        let ideal_rot_x2 = ideal_from_ref(&Sha256Ideal::<R>::RotX2(
            RotationIdeal::new(R::ONE + R::ONE), // (X − 2)
        ));

        // Scalars.
        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]); // X^30 + X^19 + X^10
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]); //  X^26 + X^21 + X^7
        let two_scalar = const_scalar::<R>(R::ONE + R::ONE);
        // Carry-extraction multipliers (constant scalars, degree 0).
        // For mu_X packed at bits [k_low, k_low+w) of w_mu_packed, the
        // contribution `2^32 · mu_X` to C7/C8/C9/C12/C13 is built as:
        //
        //     2^32 · ShiftR(k_low)(w)  −  2^{32+w} · ShiftR(k_low+w)(w)
        //
        // Each term: `binary_poly × scalar` (degree 31 + 0 = 31).
        let const_2_to_32 = const_scalar::<R>(pow_two::<R>(32));
        let const_2_to_33 = const_scalar::<R>(pow_two::<R>(33));
        let const_2_to_34 = const_scalar::<R>(pow_two::<R>(34));
        let const_2_to_35 = const_scalar::<R>(pow_two::<R>(35));

        // mu_X contributions (each evaluates to `2^32 · mu_X` at X=2).
        let mu_w_contrib = mbs(w_mu_packed, &const_2_to_32)
            .expect("2^32 · w_mu_packed overflow")
            - &mbs(down_w_mu_packed_shr2, &const_2_to_34)
                .expect("2^34 · ShiftR(2)(w_mu_packed) overflow");
        let mu_a_contrib = mbs(down_w_mu_packed_shr2, &const_2_to_32)
            .expect("2^32 · ShiftR(2)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr5, &const_2_to_35)
                .expect("2^35 · ShiftR(5)(w_mu_packed) overflow");
        let mu_e_contrib = mbs(down_w_mu_packed_shr5, &const_2_to_32)
            .expect("2^32 · ShiftR(5)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr8, &const_2_to_35)
                .expect("2^35 · ShiftR(8)(w_mu_packed) overflow");
        let mu_ff_a_contrib = mbs(down_w_mu_packed_shr8, &const_2_to_32)
            .expect("2^32 · ShiftR(8)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr9, &const_2_to_33)
                .expect("2^33 · ShiftR(9)(w_mu_packed) overflow");
        let mu_ff_e_contrib = mbs(down_w_mu_packed_shr9, &const_2_to_32)
            .expect("2^32 · ShiftR(9)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr10, &const_2_to_33)
                .expect("2^33 · ShiftR(10)(w_mu_packed) overflow");

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

        // Constraint 4 (was σ_0 (X^32 − 1) ideal-lift): row-local Q[X]
        // equality with bit-XOR overflow correction. The σ_0/σ_1 right-
        // shift decomposition columns S_0/T_0 are gone — ROT/SHIFTR
        // virtual columns are mod X^32 by construction so the modular
        // lift goes away. `pa_ov_lsig0` still absorbs the F_2[X] → Q[X]
        // coefficient sum {0..3} → bit XOR.
        //   ROT^25(W) + ROT^14(W) + SHIFTR^3(W) − lsig0 − 2 · pa_ov_lsig0 == 0
        b.assert_zero(
            down_w_rot25.clone() + down_w_rot14 + down_w_shr3 - w_lsig0
                - &mbs(pa_ov_lsig0, &two_scalar).expect("2 · ov_lsig0 overflow"),
        );

        // Constraint 6 (was σ_1 (X^32 − 1) ideal-lift): σ_1 analogue of C4.
        //   ROT^15(W) + ROT^13(W) + SHIFTR^10(W) − lsig1 − 2 · pa_ov_lsig1 == 0
        b.assert_zero(
            down_w_rot15.clone() + down_w_rot13 + down_w_shr10 - w_lsig1
                - &mbs(pa_ov_lsig1, &two_scalar).expect("2 · ov_lsig1 overflow"),
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
        // mu_W is now read from the up row (chained-comp re-anchoring
        // stores each carry at its constraint's anchor row, not at
        // spec-row t). `mu_w_contrib` evaluates to `2^32 · mu_W` at X=2.
        let sched_inner = down_w_w_sh16.clone()
            - w_big_w
            - down_w_lsig0_sh1
            - down_w_w_sh9
            - down_w_lsig1_sh14
            + &mu_w_contrib;
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
        let a_update_inner = down_w_a_sh4.clone()
            - w_e                         // h[t] = e[t-3]
            - down_w_sig1_sh3             // Sigma_1(e[t])
            - down_w_u_ef_sh3             // Ch[t] = u_ef + u_{¬e,g}
            - down_w_u_neg_e_g_sh3
            - down_pa_k_sh3               // K[t]
            - down_w_w_sh3                // W[t]
            - down_w_sig0_sh3             // Sigma_0(a[t])
            - down_w_maj_sh3              // Maj[t]
            + &mu_a_contrib;              // = 2^32 · mu_a (bits 2-4 of W_MU_PACKED)
        b.assert_in_ideal(a_update_inner + pa_c_c8, &ideal_rot_x2);

        // Constraint 9: Register-update for `e`, anchored at k = t − 3.
        //   (e[t+1] − (d[t] + h[t] + Sigma_1(e[t]) + Ch[t] + K[t] + W[t])
        //    + 2·X^31 · mu_e[t] + pa_c_c9) ∈ (X − 2)
        //
        // With d[t] = a[t-3] = up.w_a and h[t] = e[t-3] = up.w_e, and
        // Ch[t] = u_ef[t] + u_{¬e,g}[t] as in C8.
        let e_update_inner = down_w_e_sh4.clone()
            - w_a                         // d[t] = a[t-3]
            - w_e                         // h[t] = e[t-3]
            - down_w_sig1_sh3
            - down_w_u_ef_sh3             // Ch[t] = u_ef + u_{¬e,g}
            - down_w_u_neg_e_g_sh3
            - down_pa_k_sh3
            - down_w_w_sh3
            + &mu_e_contrib;              // = 2^32 · mu_e (bits 5-7 of W_MU_PACKED)
        b.assert_in_ideal(e_update_inner + pa_c_c9, &ideal_rot_x2);

        // C13–C15 (B_1/B_2/B_3 materialization identities) are gone:
        // the residuals are now packed virtual binary_poly columns
        // declared in `signature()` via `with_virtual_binary_poly_cols`.
        // The booleanity sumcheck pins each per-bit residual MLE into
        // `{0,1}` per coefficient via the closing-override mechanism —
        // jointly with the `u_ef` / `u_{¬e,g}` / `Maj` truth-table
        // values populated on every row by the trace builder, this
        // forces the spec's AND/MAJ semantics on every active row.

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
        let ff_a_inner = down_w_a_sh4.clone()
            - w_a
            - pa_a
            + &mu_ff_a_contrib;           // = 2^32 · mu_ff_a (bit 8 of W_MU_PACKED)
        b.assert_in_ideal(ff_a_inner + pa_c_ff_a, &ideal_rot_x2);

        // Constraint 13 (feed-forward, e-family). Mirrors C12 on the
        // e-half via `pa_c_ff_e`. mu_ff_e from bit 9 of W_MU_PACKED.
        let ff_e_inner = down_w_e_sh4.clone()
            - w_e
            - pa_e
            + &mu_ff_e_contrib;
        b.assert_in_ideal(ff_e_inner + pa_c_ff_e, &ideal_rot_x2);

        // Constraint 16: message init (Table 9 row 77). For each
        // compression `i ∈ [0, NUM_COMPRESSIONS)`, pin the 16 message-
        // schedule seed rows `[ROWS_PER_COMP·i, ROWS_PER_COMP·i + 16)`
        // to the public message-block words `pa_m`. Lets a verifier
        // specify *which* message is being hashed; without this, w_W
        // seeds are prover-chosen and the proof attests only to "some"
        // SHA-256 round-trip.
        //
        //   s_msg_init · (w_W − pa_m) == 0
        //
        // Zero-ideal (assert_zero), so it doesn't count toward
        // `count_effective_max_degree` — MLE-first eligibility intact.
        b.assert_zero(s_msg_init.clone() * &(w_big_w.clone() - pa_m));

        // Compensator-zero pinning (formerly C17–C21) is now enforced
        // by direct verifier inspection of public_trace via
        // `verify_public_structure`, not as in-circuit constraints —
        // see this UAIR's `verify_public_structure` impl below.
        //
        // Suppress unused-variable warnings on the compensator binders;
        // they appear in C7/C8/C9 (W) and C12/C13 (FF), already used
        // earlier in this function.
        let _ = (pa_c_c7, pa_c_c8, pa_c_c9, pa_c_ff_a, pa_c_ff_e);
    }

    /// Verify the public-column structural properties that the
    /// in-circuit constraints do not capture (formerly C17–C21).
    ///
    /// The five compensator columns must be zero on each constraint's
    /// active row range; the two tail-compensator columns must be zero
    /// on every inner row. The verifier discharges these by direct
    /// inspection of `public_trace`.
    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + num_traits::Zero,
    {
        let n = 1usize << num_vars;
        debug_assert_eq!(public_trace.int.len(), cols::NUM_INT_PUB);
        debug_assert!(public_trace.binary_poly.len() >= cols::NUM_BIN_PUB);

        let pa_c_c7 = &public_trace.int[cols::PA_C_C7].evaluations;
        let pa_c_c8 = &public_trace.int[cols::PA_C_C8].evaluations;
        let pa_c_c9 = &public_trace.int[cols::PA_C_C9].evaluations;
        let pa_c_ff_a = &public_trace.int[cols::PA_C_FF_A].evaluations;
        let pa_c_ff_e = &public_trace.int[cols::PA_C_FF_E].evaluations;
        let pa_r_ch2_comp = &public_trace.binary_poly[cols::PA_R_CH2_COMP].evaluations;
        let pa_r_maj_comp = &public_trace.binary_poly[cols::PA_R_MAJ_COMP].evaluations;

        // Per-compression active windows. start = ROWS_PER_COMP·i.
        for i in 0..cols::NUM_COMPRESSIONS {
            let start = i * cols::ROWS_PER_COMP;

            // Schedule-active anchors: k ∈ [start, start + ROUNDS_PER_COMP - 16),
            // i.e. the 48 anchors where C7 produces W[k+16] inside the
            // 16..=63 derived range.
            let sched_end = start + (cols::ROUNDS_PER_COMP - 16);
            for k in start..sched_end.min(n) {
                if !pa_c_c7[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_C7",
                        row: k,
                    });
                }
            }

            // Update-active anchors: k ∈ [start, start + ROUNDS_PER_COMP),
            // i.e. the 64 round-update anchors per compression.
            let upd_end = start + cols::ROUNDS_PER_COMP;
            for k in start..upd_end.min(n) {
                if !pa_c_c8[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_C8",
                        row: k,
                    });
                }
                if !pa_c_c9[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_C9",
                        row: k,
                    });
                }
            }

            // Junction-window rows: [start + ROUNDS_PER_COMP, start + ROWS_PER_COMP)
            // i.e. the 4 rows per compression where the SHA-256
            // feed-forward H_{i+1} = H_i + (...) is checked.
            let junc_start = start + cols::ROUNDS_PER_COMP;
            let junc_end = start + cols::ROWS_PER_COMP;
            for k in junc_start.min(n)..junc_end.min(n) {
                if !pa_c_ff_a[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_FF_A",
                        row: k,
                    });
                }
                if !pa_c_ff_e[k].is_zero() {
                    return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                        column: "PA_C_FF_E",
                        row: k,
                    });
                }
            }
        }

        // Tail compensators: zero on every inner row k with k+2 < n.
        // (At k ∈ {n-2, n-1} the compensator takes the closed-form
        // boundary value; we don't re-derive that here, since the
        // residuals C19/C20 booleanity already captures any cell
        // value at those rows that violates {0,1} per coefficient.)
        let inner_end = n.saturating_sub(2);
        for k in 0..inner_end {
            if !pa_r_ch2_comp[k].iter().all(|c| !c.into_inner()) {
                return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                    column: "PA_R_CH2_COMP",
                    row: k,
                });
            }
            if !pa_r_maj_comp[k].iter().all(|c| !c.into_inner()) {
                return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                    column: "PA_R_MAJ_COMP",
                    row: k,
                });
            }
        }

        Ok(())
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

/// Build the constant-polynomial `c` as a `DensePolynomial<R, 32>`.
fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

/// Compute `2^k` as an `R` value via repeated doubling. Used by the
/// carry-extraction multipliers in C7/C8/C9/C12/C13 (see
/// `constrain_general` for how `2^32` / `2^33` / `2^34` / `2^35`
/// scalars combine `BitOp::ShiftR` virtuals on `W_MU_PACKED` to
/// reconstruct each carry's `2^32 · mu_X` contribution).
fn pow_two<R: ConstSemiring>(k: u32) -> R {
    let mut result = R::ONE;
    for _ in 0..k {
        let copy = result.clone();
        result += &copy;
    }
    result
}

// ---------------------------------------------------------------------------
// SHA-256 reference helpers (for witness generation).
// ---------------------------------------------------------------------------

/// Canonical SHA-256 round constants K[0..64] from FIPS 180-4 §4.2.2 —
/// the first 32 bits of the fractional parts of the cube roots of the
/// first 64 primes. Cycled per compression by the trace generator: at
/// each compression starting at row `start = 68·i`, the trace gen
/// writes `K_CANONICAL[j]` to `pa_K[start + 3 + j]` for `j ∈ [0, 64)`,
/// so the C8/C9 read at anchor `k = start + j` (which references
/// `down.pa_K^↓3 = pa_K[k+3]`) lands on the right round constant.
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
    // Add the optional bit-poly addend (e.g. SHR^3(W) for lsig0).
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
    // The σ_0 constraint is now `ROT^25(W) + ROT^14(W) + SHR^3(W) −
    // lsig0 − 2·ov_lsig0 == 0` (row-local Q[X], no committed S_0/T_0).
    // `rotation_overflow` reduces `W·(X^25 + X^14)` modulo (X^32 − 1)
    // — i.e. ROT^25(W) + ROT^14(W) — and adds `W >> 3` (= SHR^3(W)),
    // matching the new constraint coefficient-wise.
    let shr3 = w_val >> 3;
    rotation_overflow(w_val, &[14, 25], shr3, lsig0_val)
}

fn lsig1_overflow(w_val: u32, lsig1_val: u32) -> u32 {
    // σ_1 analogue: ROT^15(W) + ROT^13(W) + SHR^10(W) − lsig1 −
    // 2·ov_lsig1 == 0.
    let shr10 = w_val >> 10;
    rotation_overflow(w_val, &[13, 15], shr10, lsig1_val)
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
        // pa_m: per-compression message-block words. Holds M_i[0..16]
        // at rows [start, start+16) for compression i; zero elsewhere.
        // Pinned to w_W at those rows by C16 (s_msg_init).
        let mut pa_m_vals = vec![0u32; n];

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

            // 2) Per-compression message block. 16 random seeds (which
            //    also populate the public pa_m column so C16 pins them),
            //    then 48 derived via the SHA-256 message-schedule
            //    recurrence — contained entirely within compression i's
            //    window.
            for j in 0..16 {
                let m_word = rng.next_u32();
                w_vals[start + j] = m_word;
                pa_m_vals[start + j] = m_word;
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
                // Store mu_W at the C7 anchor row k = t − 16 (not at
                // spec row t) so C7 reads it via `up.w_mu_packed` bits
                // 0-1 with no shift.
                mu_w_vals[t - 16] = carry;
            }

            // 3) Per-compression round constants. Cycle the canonical
            //    SHA-256 K table per compression at rows
            //    `[start + 3, start + 67)` so that C8/C9 at active
            //    anchors `k ∈ [start, start + 64)` (which read
            //    `down.pa_K^↓3 = pa_K[k+3]`) see `K_CANONICAL[k - start]`.
            //    Rows `start..start+3` and `start+67` are not read by
            //    any active anchor of compression i, so they're left
            //    as zero. (The compensator pa_c_c8/c9 absorbs whatever
            //    those rows contain.)
            for j in 0..cols::ROUNDS_PER_COMP {
                k_vals[start + 3 + j] = K_CANONICAL[j];
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
                // Store mu_a/mu_e at the C8/C9 anchor row k (not at
                // spec row t = k+3) so C8/C9 read via `up.w_mu_packed`
                // bits 2-4 / 5-7 with no shift.
                mu_a_vals[k] = mu_a_t;
                mu_e_vals[k] = mu_e_t;
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

        // ===== Per-row Ch / Maj operand witnesses =====
        //
        // Computed honestly on every row from a_vals / e_vals contents.
        // The truth-table values must hold on every row (not only
        // SHA-active ones) to keep the Ch/Maj virtual residuals
        // (`r_ch1` / `r_ch2` / `r_maj`, declared in `signature()`'s
        // `with_virtual_binary_poly_cols`) bit-valid per coefficient
        // across compression-junction boundaries: the booleanity
        // sumcheck checks every row, including ones the spec doesn't
        // care about.
        let u_ef_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 1 { e_vals[t] & e_vals[t - 1] } else { 0 })
            .collect();
        let u_neg_e_g_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { (!e_vals[t]) & e_vals[t - 2] } else { 0 })
            .collect();
        let maj_vals: Vec<u32> = (0..n)
            .map(|t| if t >= 2 { maj(a_vals[t], a_vals[t - 1], a_vals[t - 2]) } else { 0 })
            .collect();

        // ===== Tail compensators for the Ch (63) / Maj (64) virtual residuals =====
        //
        // Zero on every row except `k ∈ {n−2, n−1}` where the length-2
        // forward shifts in r_ch2 / r_maj read into off-trace zero-
        // padding and the residual would slip outside `{0,1}` per
        // coefficient. Match the compensator logic in option-a-virtual-
        // residuals (8787cbd):
        //   r_ch2 (alt complement form) at boundary k = n-2 / n-1:
        //     u_{¬e,g}[k+2] = 0 (off-trace), e[k+2] = 0, e[k] real.
        //     residual = -e[k] + 2·comp_ch2 ∈ {0,1} ⇒ comp_ch2[k] = e[k].
        //   r_maj at boundary k = n-2:
        //     a[k+2] = Maj[k+2] = 0 (off-trace), a[k+1] real.
        //     residual = a[k] + a[k+1] − 2·comp_maj ∈ {0,1}
        //     ⇒ comp_maj[k] = AND(a[k], a[k+1]).
        //   r_maj at k = n-1: a[k+1] = a[k+2] = 0, residual = a[k] ∈
        //     {0,1} already; comp_maj = 0.
        let mut pa_r_ch2_comp_vals: Vec<u32> = vec![0; n];
        let mut pa_r_maj_comp_vals: Vec<u32> = vec![0; n];
        for k in 0..n {
            let off_kp1 = k + 1 >= n;
            let off_kp2 = k + 2 >= n;
            if off_kp2 {
                pa_r_ch2_comp_vals[k] = e_vals[k];
            }
            if off_kp2 && !off_kp1 {
                pa_r_maj_comp_vals[k] = a_vals[k] & a_vals[k + 1];
            }
        }

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

        // The σ_0/σ_1 right-shift decomposition columns S_i / T_i are
        // gone — their role (carrying SHR(W, k) for the F_2[X] sum) is
        // taken over by the `BitOp::ShiftR(k)` virtual columns over W.
        // `lsig0_overflow` / `lsig1_overflow` already compute the
        // matching `pa_ov_lsig{0,1}` per-bit values for the new
        // constraint (the algebraic identity is unchanged).

        // Pack all 5 carries per row into the W_MU_PACKED binary_poly
        // column. Each carry was stored at its constraint's anchor row
        // (mu_W at C7-anchor k = t-16, mu_a/mu_e at C8/C9-anchor k =
        // t-3, mu_ff_a/e at junction-anchor k). Bit layout:
        //   bits 0-1: mu_W,  2-4: mu_a,  5-7: mu_e,  8: mu_ff_a,  9: mu_ff_e.
        // Positions 10..31 stay 0 (pinned by C22's high-bits-zero
        // assert_zero on ShiftR(10)(W_MU_PACKED)).
        let w_mu_packed_vals: Vec<u32> = (0..n)
            .map(|k| {
                (mu_w_vals[k] & 0b11)
                    | ((mu_a_vals[k] & 0b111) << 2)
                    | ((mu_e_vals[k] & 0b111) << 5)
                    | ((mu_junction_a_vals[k] & 0b1) << 8)
                    | ((mu_junction_e_vals[k] & 0b1) << 9)
            })
            .collect();

        let to_bits = |v: &[u32]| -> Vec<BinaryPoly<32>> {
            v.iter().copied().map(BinaryPoly::<32>::from).collect()
        };

        let to_bin_mle = |col: Vec<BinaryPoly<32>>| -> DenseMultilinearExtension<
            BinaryPoly<32>,
        > { col.into_iter().collect() };

        // Layout: 8 public bin_poly cols (PA_A, PA_E, PA_OV_SIG0,
        // PA_OV_SIG1, PA_OV_LSIG0, PA_OV_LSIG1, PA_R_CH2_COMP,
        // PA_R_MAJ_COMP) + 10 witness cols. pa_a / pa_e were populated
        // above with H_i values at init-prefix rows (for compression i
        // and the H_N output block) AND at junction rows (where the
        // feed-forward constraint reads the prior H_i). The two
        // PA_R_*_COMP columns are zero except on the trace tail.
        let binary_poly = vec![
            to_bin_mle(to_bits(&pa_a_vals)),
            to_bin_mle(to_bits(&pa_e_vals)),
            to_bin_mle(to_bits(&ov_sig0_vals)),
            to_bin_mle(to_bits(&ov_sig1_vals)),
            to_bin_mle(to_bits(&ov_lsig0_vals)),
            to_bin_mle(to_bits(&ov_lsig1_vals)),
            to_bin_mle(to_bits(&pa_r_ch2_comp_vals)),
            to_bin_mle(to_bits(&pa_r_maj_comp_vals)),
            to_bin_mle(to_bits(&pa_m_vals)),
            to_bin_mle(to_bits(&a_vals)),
            to_bin_mle(to_bits(&sig0_vals)),
            to_bin_mle(to_bits(&e_vals)),
            to_bin_mle(to_bits(&sig1_vals)),
            to_bin_mle(to_bits(&w_vals)),
            to_bin_mle(to_bits(&lsig0_vals)),
            to_bin_mle(to_bits(&lsig1_vals)),
            to_bin_mle(to_bits(&u_ef_vals)),
            to_bin_mle(to_bits(&u_neg_e_g_vals)),
            to_bin_mle(to_bits(&maj_vals)),
            to_bin_mle(to_bits(&w_mu_packed_vals)),
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
        // s_msg_init: 1 on the 16 message-block-seed rows of every
        // compression, 0 elsewhere. Gates C16 (`w_W − pa_m == 0`).
        let mut s_msg_init_col: Vec<R> = (0..n).map(|_| R::ZERO).collect();
        for i in 0..big_n {
            for j in 0..16 {
                s_msg_init_col[i * rpc + j] = R::ONE;
            }
        }
        // The s_active_sched / s_active_upd selectors are gone — the
        // verifier enforces the compensator-zero pinning by direct
        // public_trace inspection, see `verify_public_structure`.

        let k_col: Vec<R> = k_vals.iter().copied().map(R::from).collect();
        // mu_w_vals / mu_a_vals / mu_e_vals / mu_junction_{a,e}_vals
        // are no longer materialized as separate int columns — they're
        // packed into the W_MU_PACKED binary_poly column above.

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
                // mu_W stored at C7-anchor row k (= round t = k+16 was
                // formerly stored at k+16; with chained-comp re-anchoring
                // it's now at row k).
                let mu_k = load(&mu_w_vals, k);
                let two32_mu = two_to_32.clone() * &mu_k;
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
                // mu_a stored at C8-anchor row k (= round t = k+3 was
                // formerly stored at k+3; now at row k).
                let mu_a_k = load(&mu_a_vals, k);
                let two32_mu = two_to_32.clone() * &mu_a_k;
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
                // mu_e stored at C9-anchor row k (analogous to mu_a).
                let mu_e_k = load(&mu_e_vals, k);
                let two32_mu = two_to_32.clone() * &mu_e_k;
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

        let to_int_mle = |col: Vec<R>| -> DenseMultilinearExtension<R> {
            col.into_iter().collect()
        };
        // Layout matches cols::S_INIT_PREFIX..NUM_INT order. S_B_ACTIVE
        // is gone alongside the dropped C13–C15 materialization
        // constraints (residuals are now virtual binary_poly cols).
        let int = vec![
            to_int_mle(s_init_prefix_col),
            to_int_mle(s_feedforward_col),
            to_int_mle(s_msg_init_col),
            to_int_mle(k_col),
            to_int_mle(pa_c_c7_col),
            to_int_mle(pa_c_c8_col),
            to_int_mle(pa_c_c9_col),
            to_int_mle(pa_c_ff_a_col),
            to_int_mle(pa_c_ff_e_col),
            // The 2 prior compensator-zero selectors
            // (s_active_sched/upd) are gone — compensator-zero is
            // enforced by direct verifier inspection of public_trace.
            // The 5 prior int carry columns (mu_W/a/e/junction_a/e)
            // are gone — packed into W_MU_PACKED above.
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

    /// Cross-check the K_CANONICAL table against the canonical SHA-256
    /// initial hash values H_0 — running one full compression of the
    /// empty-padding block (with H_0 as input) must produce the
    /// SHA-256 digest of the empty string,
    /// `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
    /// Catches any drift in the K constants (or in the round-update
    /// logic itself).
    #[test]
    fn k_canonical_matches_sha256_empty_string_digest() {
        // SHA-256 H_0 (FIPS 180-4 §5.3.3).
        let h_in: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ];
        // Empty-string padded block: single 0x80 byte then 63 zero bytes.
        let mut m = [0u32; 16];
        m[0] = 0x80000000;
        // Length (in bits) at the end: 0.

        // Run the message schedule.
        let mut w = [0u32; 64];
        w[..16].copy_from_slice(&m);
        for t in 16..64 {
            w[t] = w[t - 16]
                .wrapping_add(small_sigma0(w[t - 15]))
                .wrapping_add(w[t - 7])
                .wrapping_add(small_sigma1(w[t - 2]));
        }

        // Run the 64 round updates.
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

        // Feed-forward.
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
        assert_eq!(h_out, expected, "SHA-256(\"\") digest mismatch — K table or round logic drift");
    }
}
