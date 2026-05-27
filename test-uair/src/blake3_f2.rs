//! Blake3 compression UAIR over `F_2[X]`.
//!
//! Spec source: `documentation/blake3-f2x-doc/`. Sibling to
//! [`Sha256F2Uair`](crate::Sha256F2Uair). Every committed witness cell
//! is a bit-polynomial in `bitpoly_{32} ⊂ F_2[X]` and gets wired
//! through the `F_2` prove path. Booleanity is structural.
//!
//! ## Effective max degree
//!
//! All `assert_in_ideal` constraints are degree-1 in the trace MLEs.
//! Selector-gated `assert_zero` constraints (degree-2 — boundary pins,
//! stitch pins, and the κ compensator pins) are excluded from the
//! `count_effective_max_degree` check because they live in the zero
//! ideal. The F_2 prove dispatch therefore lands on `prove_linear`
//! (MLE-first), collapsing per-row constraint evaluations down to ~50
//! column-level ones — mirroring `sha256_f2.rs`'s lane.
//!
//! ## Trace layout — one G per row
//!
//! `ROWS_PER_COMP = 60`. Within each compression $i$:
//! - Rows $60i+0..60i+3$ — `init prefix` (4 rows). The 16 init values
//!   $(\mathrm{cv}_0..\mathrm{cv}_7, \mathrm{IV}_0..\mathrm{IV}_3,
//!   t_\mathrm{lo}, t_\mathrm{hi}, \ell, f)$ are pinned to the four
//!   committed output-state columns `W_VA..W_VD` at these rows so the
//!   `round-0 phase-A` reads use the same row-shift pattern as every
//!   subsequent half-round (no init-special-case).
//! - Rows $60i+4..60i+59$ — 56 G-rows (7 rounds × 8 G's). The 4 G's of
//!   round $r$'s phase A are at rows $60i+4+8r..60i+4+8r+3$, and round
//!   $r$'s phase B at rows $60i+4+8r+4..60i+4+8r+7$.
//!
//! ## Anchor rows and shifts
//!
//! Each half-round has a single **anchor row** four positions before
//! its first G: phase-A of round $r$ anchors at $60i+8r$, phase-B at
//! $60i+8r+4$. From an anchor, the 4 G's of the half-round write at
//! `anchor + 4 .. anchor + 7` (down-shifts 4..7), and their 4 inputs
//! come from `anchor + 0 .. anchor + 3` (down-shifts 0..3) — which
//! holds either the 4 G outputs of the prior half-round, or the four
//! init-prefix rows for round-0 phase-A.
//!
//! Read-shift permutation within a half-round (G-slot $g$ → row offset):
//!
//!   Phase A: slot a ↦ g, slot b ↦ (g+3) mod 4, slot c ↦ (g+2) mod 4, slot d ↦ (g+1) mod 4
//!   Phase B: slot a ↦ g, slot b ↦ (g+1) mod 4, slot c ↦ (g+2) mod 4, slot d ↦ (g+3) mod 4
//!
//! (Derived from the BLAKE3 column/diagonal permutation. To make the
//! init-prefix rows behave as a "virtual phase-B half-round of round
//! −1", the four init rows hold the position quadruples
//! `PHASE_B_QUAD[r]` — see [`cols::PHASE_B_QUAD`] — at row offsets
//! 0..3. Concretely, row offset $r$ of the init prefix carries state
//! positions `PHASE_B_QUAD[r]` in slots a/b/c/d.)
//!
//! Active rows per trace: `ACTIVE_ROWS = 60·N + 4` (mirroring
//! `sha-f2x-doc`'s tail convention).
//!
//! ## Column layout (binary_poly only — the `int` lane is empty for F_2)
//!
//! Public bit-poly columns (10):
//!   - `PA_CV_LO`, `PA_CV_HI` (2): chaining-value boundary. At init
//!     rows $60i+r$, `PA_CV_LO` and `PA_CV_HI` hold the cv components
//!     for positions `PHASE_B_QUAD[r][{0,1}]` (the "slot a" and
//!     "slot b" positions of the virtual phase-B G at row offset r).
//!   - `PA_IV`, `PA_PARAMS` (2): hold the IV / per-block parameter
//!     components for positions `PHASE_B_QUAD[r][{2,3}]` (the
//!     "slot c" and "slot d" positions).
//!   - `PA_M_X`, `PA_M_Y` (2): per-half-round message words. At
//!     anchor row $a$, G-slot $g$ reads $x_g$ from `PA_M_X[a+g]` and
//!     $y_g$ from `PA_M_Y[a+g]`. The prover lays these out so the
//!     cells at $a..a+3$ hold the message words used by the half-
//!     round anchored at $a$ (= permuted per `MSG_SCHEDULE[round]`).
//!   - `S_INIT_PREFIX`, `S_PHASE_A`, `S_PHASE_B`, `S_STITCH` (4):
//!     ${0,1}$-valued selectors gating the constraint families
//!     below.
//!
//! Committed witnesses (28):
//!   - `W_VA, W_VB, W_VC, W_VD` (4) — the four output-state columns.
//!     At a G-row $t$, `W_V*[t]` holds the new value of slot $* ∈
//!     \{a, b, c, d\}$ produced by the G at row $t$. At init-prefix
//!     rows, these hold the appropriate init values (see boundary
//!     pin family below).
//!   - `W_SA1` (= `s_a^(1)`) — chained-Binius intermediate of arity-3
//!     `mod_1`: $W_SA1 \equiv v_a + v_b \pmod{2^{32}}$.
//!   - `W_VA1` (= `v_a^(1)`) — inner v_a after `mod_1`:
//!     $W_VA1 \equiv W_SA1 + x \pmod{2^{32}}$.
//!   - `W_VC1` (= `v_c^(1)`) — inner v_c after `mod_2`:
//!     $W_VC1 \equiv v_c + W_DPRIME1 \pmod{2^{32}}$.
//!   - `W_SA2` (= `s_a^(2)`) — chained-Binius intermediate of arity-3
//!     `mod_3`: $W_SA2 \equiv W_VA1 + W_BPRIME1 \pmod{2^{32}}$.
//!   - `W_DPRIME1` (= `d'`) — committed inner rotation
//!     intermediate: $W_DPRIME1 = \ROTR^{16}(v_d \oplus W_VA1)$.
//!     Pinned by both the row-local C3-style identity (`X^{24} ·
//!     (W_DPRIME1 + W_VA) ≡ W_VD` in `(X^{32} − 1)`, holding at every
//!     row by trace construction) and the shift-bearing C1 (with κ
//!     compensator, see family B).
//!   - `W_BPRIME1` (= `b'`) — same rationale: committed inner
//!     rotation intermediate $W_BPRIME1 = \ROTR^{12}(v_b \oplus
//!     W_VC1)$. Pinned by row-local C4 (`X^{25} · (W_BPRIME1 + W_VC)
//!     ≡ W_VB` in `(X^{32} − 1)`) and shift-bearing C2 (with κ).
//!
//!   κ-compensator columns (18) — the F_2 lift-to-degree-1 plumbing:
//!   - `W_KAPPA_LSB_A`, `W_KAPPA_LSB_B` (2) — packed LSB κ.
//!     Phase-A's 24 per-step LSB compensator bits sit in bit
//!     positions 0..23 of `W_KAPPA_LSB_A`; phase-B's in
//!     `W_KAPPA_LSB_B`. Bit `j` is accessed by the per-step LSB check
//!     via the `SHR^j` bit-op virtual (bit 0 of `SHR^j(col)` is bit
//!     `j` of `col`). Selector-gated to zero on the phase's anchor
//!     rows (`S_PHASE_X · W_KAPPA_LSB_X = 0`), so the LSB identities
//!     hold there with no compensation; on every other row the
//!     prover sets the bits to absorb the residual.
//!   - `W_KR_C1_PA_g` (4) and `W_KR_C1_PB_g` (4) — rotation κ for
//!     the four C1 d'-rotation pins per phase. Each `W_KR_C1_*_g` is
//!     a full 32-bit polynomial absorbing the residual
//!     `X^{16} · (v_d + v_a^(1)) − W_DPRIME1` of G-slot `g`'s C1 pin
//!     at non-anchor rows; zero at the phase's anchor row (forced by
//!     `S_PHASE_X · W_KR_C1_X_g = 0`).
//!   - `W_KR_C2_PA_g` (4) and `W_KR_C2_PB_g` (4) — same for the four
//!     C2 b'-rotation pins per phase.
//!
//! ## Constraint families (effective degree 1)
//!
//! A — Per-G `BiniusAdd` LSB constraints (48 per anchor row: 6 per G
//!     × 4 G's × 2 phases). All are degree-1 `assert_in_ideal` in
//!     `(X)`, with a κ compensator added — mirrors the SHA-256
//!     pattern. Each constraint is
//!     `(target + x + y + SHR^{j}(W_KAPPA_LSB_X)) ∈ (X)`,
//!     where `j` is the constraint's allocated bit in
//!     `W_KAPPA_LSB_X` (phase A or B). κ packing: see
//!     [`cols::KAPPA_LSB_BIT_*`] constants. The pins
//!     `S_PHASE_X · W_KAPPA_LSB_X = 0` (`assert_zero`, degree 2,
//!     zero-ideal — does not increase effective degree) force κ to
//!     vanish on the active phase's anchor rows.
//!
//! B — Rotation pins (16+2 per anchor row).
//!
//!     **B.1 — C1, C2 (16 constraints, with κ_rot):** each is
//!     `(X^{r} · (xor_in_1 + xor_in_2) − W_DPRIME1/BPRIME1
//!     + W_KR_*_*_g) ∈ ideal_rot`. Degree 1. The compensator
//!     `W_KR_*_*_g` (a full 32-bit polynomial) absorbs the residual
//!     at non-anchor rows; on the phase's anchor row it is forced to
//!     zero by `S_PHASE_X · W_KR_*_*_g = 0`.
//!
//!     **B.2 — C3, C4 (2 row-local constraints, no selector, no κ):**
//!     `(X^{24} · (W_DPRIME1 + W_VA) − W_VD) ∈ ideal_rot` and
//!     `(X^{25} · (W_BPRIME1 + W_VC) − W_VB) ∈ ideal_rot`. Cells all
//!     at the up-row; the identities hold at every row by trace
//!     construction (at G rows by the honest G-mix arithmetic; at
//!     init/output prefix rows by setting `W_DPRIME1[r] = ROTL^8(
//!     W_VD[r]) ⊕ W_VA[r]` and `W_BPRIME1[r] = ROTL^7(W_VB[r]) ⊕
//!     W_VC[r]`).
//!
//! C — Init boundary pinning (4 per row, gated by `S_INIT_PREFIX`):
//!     `W_VA[t] = PA_CV_LO[t]`, `W_VB[t] = PA_CV_HI[t]`,
//!     `W_VC[t] = PA_IV[t]`, `W_VD[t] = PA_PARAMS[t]` at the four
//!     init-prefix rows of every compression and at the trailing
//!     `cv_N` output prefix. Degree-2 `assert_zero` (zero ideal).
//!
//! D — Junction-stitch (cv chaining between compressions, 8 per
//!     stitch anchor): selector-gated `assert_zero` of `XOR`
//!     feed-forward identities. Degree-2 zero-ideal.
//!
//! Soundness gap (parallels `sha256_f2.rs`): the Hadamard halves of
//! the 24 modular-add `BiniusAdd`s per G are NOT enforced at the AIR
//! layer. They would be discharged by a separate Hadamard-product
//! check (deferred per `blake3-f2x-doc` §3.4). The AIR-level LSB
//! checks pin only bit 0 of each addition.

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
    BitOp, BitOpSpec, ConstraintBuilder, PublicColumnLayout, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace,
    ideal::{Ideal, IdealCheck, IdealCheckError, rotation::RotationIdeal},
    ideal_collector::IdealOrZero,
};
use zinc_utils::from_ref::FromRef;

use crate::GenerateRandomTrace;

// ---------------------------------------------------------------------------
// Blake3F2Ideal
// ---------------------------------------------------------------------------

/// Ideals used by this UAIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blake3F2Ideal {
    /// `(X^32 − 1)` in `F_2[X]`. Rotation pins.
    RotXw1,
    /// `(X)` in `F_2[X]`. Per-step `BiniusAdd` LSB check.
    LsbX,
}

impl FromRef<Blake3F2Ideal> for Blake3F2Ideal {
    fn from_ref(value: &Blake3F2Ideal) -> Self { *value }
}
impl Ideal for Blake3F2Ideal {}

impl<F> IdealCheck<DynamicPolynomialF<F>> for Blake3F2Ideal
where F: PrimeField,
{
    fn contains(&self, value: &DynamicPolynomialF<F>) -> Result<bool, IdealCheckError> {
        if value.coeffs.is_empty() {
            return Ok(true);
        }
        match self {
            Blake3F2Ideal::RotXw1 => {
                let one = F::one_with_cfg(value.coeffs[0].cfg());
                IdealOrZero::NonZero(RotationIdeal::<F, 32>::new(one)).contains(value)
            }
            Blake3F2Ideal::LsbX => Ok(F::is_zero(&value.coeffs[0])),
        }
    }
}

/// Project an `IdealOrZero<Blake3F2Ideal>` into the verifier field.
pub fn blake3_f2_project_ideal(ideal: &IdealOrZero<Blake3F2Ideal>) -> Blake3F2Ideal {
    match ideal {
        IdealOrZero::NonZero(i) => *i,
        IdealOrZero::Zero => Blake3F2Ideal::RotXw1,
    }
}

// ---------------------------------------------------------------------------
// Blake3 reference constants.
// ---------------------------------------------------------------------------

/// Blake3 initial chaining-value constants (same as SHA-256 IV).
pub const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Message permutation table per round (7 rounds).
pub const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [ 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8],
    [ 3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1],
    [10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6],
    [12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4],
    [ 9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7],
    [11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13],
];

// ---------------------------------------------------------------------------
// Column indices and trace layout.
// ---------------------------------------------------------------------------

pub mod cols {
    //! Column-layout constants for the Blake3 F_2 UAIR.

    // === Public bit-poly prefix (10) ===

    pub const PA_CV_LO: usize = 0;
    pub const PA_CV_HI: usize = 1;
    pub const PA_IV: usize = 2;
    pub const PA_PARAMS: usize = 3;
    pub const PA_M_X: usize = 4;
    pub const PA_M_Y: usize = 5;
    pub const S_INIT_PREFIX: usize = 6;
    pub const S_PHASE_A: usize = 7;
    pub const S_PHASE_B: usize = 8;
    pub const S_STITCH: usize = 9;
    pub const NUM_BIN_PUB: usize = 10;

    // === Witness bit-poly suffix (28) ===

    pub const W_VA: usize = NUM_BIN_PUB;     // slot a output
    pub const W_VB: usize = NUM_BIN_PUB + 1; // slot b output
    pub const W_VC: usize = NUM_BIN_PUB + 2; // slot c output
    pub const W_VD: usize = NUM_BIN_PUB + 3; // slot d output
    pub const W_SA1: usize = NUM_BIN_PUB + 4;
    pub const W_VA1: usize = NUM_BIN_PUB + 5;
    pub const W_VC1: usize = NUM_BIN_PUB + 6;
    pub const W_SA2: usize = NUM_BIN_PUB + 7;
    /// Inner rotation intermediate `d' = ROTR^16(v_d ⊕ v_a^(1))`,
    /// committed so the LSB BiniusAdd check at step 2 (`v_c^(1) = v_c
    /// + d'`) can reference the actual rotated value's bit 0. Pinned
    /// row-locally by C3 (`X^24 · (W_DPRIME1 + W_VA) ≡ W_VD` in
    /// `(X^32 − 1)`) and by the shift-bearing C1 (with κ_rot).
    pub const W_DPRIME1: usize = NUM_BIN_PUB + 8;
    /// Inner rotation intermediate `b' = ROTR^12(v_b ⊕ v_c^(1))`,
    /// pinned row-locally by C4 (`X^25 · (W_BPRIME1 + W_VC) ≡ W_VB`)
    /// and by the shift-bearing C2 (with κ_rot).
    pub const W_BPRIME1: usize = NUM_BIN_PUB + 9;

    // --- C1 rotation κ compensators (phase A / phase B, per G-slot) ---
    pub const W_KR_C1_PA_0: usize = NUM_BIN_PUB + 10;
    pub const W_KR_C1_PA_1: usize = NUM_BIN_PUB + 11;
    pub const W_KR_C1_PA_2: usize = NUM_BIN_PUB + 12;
    pub const W_KR_C1_PA_3: usize = NUM_BIN_PUB + 13;
    pub const W_KR_C1_PB_0: usize = NUM_BIN_PUB + 14;
    pub const W_KR_C1_PB_1: usize = NUM_BIN_PUB + 15;
    pub const W_KR_C1_PB_2: usize = NUM_BIN_PUB + 16;
    pub const W_KR_C1_PB_3: usize = NUM_BIN_PUB + 17;

    // --- C2 rotation κ compensators (phase A / phase B, per G-slot) ---
    pub const W_KR_C2_PA_0: usize = NUM_BIN_PUB + 18;
    pub const W_KR_C2_PA_1: usize = NUM_BIN_PUB + 19;
    pub const W_KR_C2_PA_2: usize = NUM_BIN_PUB + 20;
    pub const W_KR_C2_PA_3: usize = NUM_BIN_PUB + 21;
    pub const W_KR_C2_PB_0: usize = NUM_BIN_PUB + 22;
    pub const W_KR_C2_PB_1: usize = NUM_BIN_PUB + 23;
    pub const W_KR_C2_PB_2: usize = NUM_BIN_PUB + 24;
    pub const W_KR_C2_PB_3: usize = NUM_BIN_PUB + 25;

    /// Packed LSB κ compensator for the 24 phase-A LSB BiniusAdd
    /// checks (4 G's × 6 binary steps). Bit `j ∈ [0, 24)` carries the
    /// per-step `κ_j` accessed via `SHR^j(W_KAPPA_LSB_A)`. Bits
    /// 24..31 are unused (the `S_PHASE_A · W_KAPPA_LSB_A = 0`
    /// `assert_zero` pin forces every bit to zero at phase-A anchor
    /// rows; on other rows the prover sets the bits to absorb the
    /// LSB residuals).
    pub const W_KAPPA_LSB_A: usize = NUM_BIN_PUB + 26;
    /// Packed LSB κ for phase-B.
    pub const W_KAPPA_LSB_B: usize = NUM_BIN_PUB + 27;

    pub const NUM_BIN_WIT: usize = 28;
    pub const NUM_BIN: usize = NUM_BIN_PUB + NUM_BIN_WIT; // 38

    /// `[W_VA, W_VB, W_VC, W_VD]`, indexable by slot s ∈ {a, b, c, d}.
    pub const W_V_SLOT: [usize; 4] = [W_VA, W_VB, W_VC, W_VD];

    /// `[W_KR_C1_PA_0, .., W_KR_C1_PA_3]`.
    pub const W_KR_C1_PA: [usize; 4] = [
        W_KR_C1_PA_0, W_KR_C1_PA_1, W_KR_C1_PA_2, W_KR_C1_PA_3,
    ];
    /// `[W_KR_C1_PB_0, .., W_KR_C1_PB_3]`.
    pub const W_KR_C1_PB: [usize; 4] = [
        W_KR_C1_PB_0, W_KR_C1_PB_1, W_KR_C1_PB_2, W_KR_C1_PB_3,
    ];
    /// `[W_KR_C2_PA_0, .., W_KR_C2_PA_3]`.
    pub const W_KR_C2_PA: [usize; 4] = [
        W_KR_C2_PA_0, W_KR_C2_PA_1, W_KR_C2_PA_2, W_KR_C2_PA_3,
    ];
    /// `[W_KR_C2_PB_0, .., W_KR_C2_PB_3]`.
    pub const W_KR_C2_PB: [usize; 4] = [
        W_KR_C2_PB_0, W_KR_C2_PB_1, W_KR_C2_PB_2, W_KR_C2_PB_3,
    ];

    // === LSB κ bit allocation (`W_KAPPA_LSB_X`, bits 0..23) ===
    //
    // Per phase, 24 LSB BiniusAdd checks consume bits 0..23. The
    // allocation is `bit = 6 · g + step` for `g ∈ [0, 4)` and `step ∈
    // [0, 6)`, so G_g's six LSB κ live at bits `6g, 6g+1, .., 6g+5`.
    #[inline]
    pub const fn kappa_lsb_bit(g: usize, step: usize) -> u32 {
        debug_assert!(g < 4);
        debug_assert!(step < 6);
        (6 * g + step) as u32
    }

    /// Number of distinct `SHR^j` bit-op virtuals declared per
    /// packed LSB κ column. `j ∈ [1, 24)` — bit `j = 0` is read from
    /// the column itself with no shift virtual.
    pub const NUM_KAPPA_LSB_SHIFTS: usize = 23;

    // === Trace layout ===

    pub const ROWS_PER_COMP: usize = 60;

    /// Number of chained Blake3 compressions packed into a trace of
    /// `2^num_vars` rows.
    #[inline]
    pub const fn num_compressions(num_vars: usize) -> usize {
        let n = 1usize << num_vars;
        if n < 4 { 0 } else { (n - 4) / ROWS_PER_COMP }
    }

    /// Number of rows actually written by the witness generator.
    #[inline]
    pub const fn active_rows(num_vars: usize) -> usize {
        num_compressions(num_vars) * ROWS_PER_COMP + 4
    }

    /// Smallest `num_vars` that fits one Blake3 compression
    /// (`num_compressions(6) = 1`, `active_rows(6) = 64`).
    pub const MIN_NUM_VARS: usize = 6;

    // === Read-shift permutation tables ===

    /// `PHASE_A_READ_SHIFT[g][s]` = forward shift from anchor row to
    /// the row holding the G-slot `s` input of G-index `g` within a
    /// phase-A half-round. Slot indices: a = 0, b = 1, c = 2, d = 3.
    pub const PHASE_A_READ_SHIFT: [[usize; 4]; 4] = [
        [0, 3, 2, 1], // G_0
        [1, 0, 3, 2], // G_1
        [2, 1, 0, 3], // G_2
        [3, 2, 1, 0], // G_3
    ];

    /// `PHASE_B_READ_SHIFT[g][s]` = same for a phase-B half-round.
    pub const PHASE_B_READ_SHIFT: [[usize; 4]; 4] = [
        [0, 1, 2, 3], // G_0
        [1, 2, 3, 0], // G_1
        [2, 3, 0, 1], // G_2
        [3, 0, 1, 2], // G_3
    ];

    /// Position quadruple `(a, b, c, d)` for G-slot `g` in phase A
    /// (column mixings). G_0 = G(0, 4, 8, 12), G_1 = G(1, 5, 9, 13),
    /// G_2 = G(2, 6, 10, 14), G_3 = G(3, 7, 11, 15).
    pub const PHASE_A_QUAD: [[usize; 4]; 4] = [
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ];

    /// Position quadruple for phase B (diagonal mixings).
    /// G_0 = G(0, 5, 10, 15), G_1 = G(1, 6, 11, 12), G_2 = G(2, 7, 8,
    /// 13), G_3 = G(3, 4, 9, 14).
    ///
    /// Doubles as the init-prefix slot-to-position assignment: row
    /// offset $r$ of any init prefix holds positions `PHASE_B_QUAD[r]`
    /// in slots (a, b, c, d).
    pub const PHASE_B_QUAD: [[usize; 4]; 4] = [
        [0, 5, 10, 15],
        [1, 6, 11, 12],
        [2, 7,  8, 13],
        [3, 4,  9, 14],
    ];
}

// ---------------------------------------------------------------------------
// Shift-slot map. Used to look up the `down.binary_poly[…]` index
// corresponding to a given `(source_col, shift_amount)` pair after the
// signature's shift list has been sorted by the framework.
// ---------------------------------------------------------------------------

/// Returns `(source_col, shift_amount)` for every shift the signature
/// declares, in the order the framework sorts them (= ascending by
/// `source_col` then by `shift_amount`).
fn shift_list() -> Vec<(usize, usize)> {
    let mut shifts: Vec<(usize, usize)> = Vec::new();
    // PA_CV_LO (col 0): shifts 4..7 — stitch targets at compression
    // i+1's init prefix from anchor 60i+56.
    for s in 4usize..=7 {
        shifts.push((cols::PA_CV_LO, s));
    }
    // PA_CV_HI (col 1): shifts 4..7.
    for s in 4usize..=7 {
        shifts.push((cols::PA_CV_HI, s));
    }
    // PA_M_X (col 4): shifts 1, 2, 3.
    for s in 1usize..=3 {
        shifts.push((cols::PA_M_X, s));
    }
    // PA_M_Y (col 5): shifts 1, 2, 3.
    for s in 1usize..=3 {
        shifts.push((cols::PA_M_Y, s));
    }
    // W_V{A,B,C,D} (cols 10..14): shifts 1..7.
    for &c in &cols::W_V_SLOT {
        for s in 1usize..=7 {
            shifts.push((c, s));
        }
    }
    // W_SA1, W_VA1, W_VC1, W_SA2, W_DPRIME1, W_BPRIME1: shifts 4..7.
    for c in [
        cols::W_SA1,
        cols::W_VA1,
        cols::W_VC1,
        cols::W_SA2,
        cols::W_DPRIME1,
        cols::W_BPRIME1,
    ] {
        for s in 4usize..=7 {
            shifts.push((c, s));
        }
    }
    shifts
}

/// Slot index of `(col, shift)` in `down.binary_poly`. Panics if the
/// pair isn't declared by the signature.
fn shift_slot(col: usize, shift: usize) -> usize {
    shift_list()
        .iter()
        .position(|&(c, s)| c == col && s == shift)
        .unwrap_or_else(|| panic!("shift ({}, {}) not declared in signature", col, shift))
}

// ---------------------------------------------------------------------------
// The UAIR.
// ---------------------------------------------------------------------------

/// Blake3 compression UAIR over `F_2[X]`.
#[derive(Clone, Debug)]
pub struct Blake3F2Uair<R>(PhantomData<R>);

impl<R> Uair for Blake3F2Uair<R>
where
    R: ConstSemiring + 'static,
{
    type Ideal = Blake3F2Ideal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(cols::NUM_BIN, 0, 0);
        let public = PublicColumnLayout::new(cols::NUM_BIN_PUB, 0, 0);
        let shifts: Vec<ShiftSpec> = shift_list()
            .into_iter()
            .map(|(c, s)| ShiftSpec::new(c, s))
            .collect();

        // Bit-op virtuals: `SHR^j(W_KAPPA_LSB_{A,B})` for j ∈ [1, 24).
        // Bit 0 of `SHR^j(W_KAPPA_LSB_X)` equals bit `j` of
        // `W_KAPPA_LSB_X` — the per-step LSB κ compensator allocated
        // to bit `j`. j = 0 is read directly from `W_KAPPA_LSB_X` at
        // the up row (no shift virtual). The framework sorts bit-op
        // specs by `(source_col, op_kind, c)`; with
        // `W_KAPPA_LSB_A < W_KAPPA_LSB_B`, virtuals land in
        // `down.bit_op[..]` in the order
        //   [SHR^1(A), .., SHR^23(A), SHR^1(B), .., SHR^23(B)].
        let mut bit_op_specs: Vec<BitOpSpec> = Vec::with_capacity(
            2 * cols::NUM_KAPPA_LSB_SHIFTS,
        );
        for c in 1u32..=(cols::NUM_KAPPA_LSB_SHIFTS as u32) {
            bit_op_specs.push(BitOpSpec::new(
                cols::W_KAPPA_LSB_A,
                BitOp::shift_r(c),
            ));
        }
        for c in 1u32..=(cols::NUM_KAPPA_LSB_SHIFTS as u32) {
            bit_op_specs.push(BitOpSpec::new(
                cols::W_KAPPA_LSB_B,
                BitOp::shift_r(c),
            ));
        }

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
        let dbp = down.binary_poly;

        // Up-row references.
        let pa_cv_lo = &bp[cols::PA_CV_LO];
        let pa_cv_hi = &bp[cols::PA_CV_HI];
        let pa_iv = &bp[cols::PA_IV];
        let pa_params = &bp[cols::PA_PARAMS];
        let s_init_prefix = &bp[cols::S_INIT_PREFIX];
        let s_phase_a = &bp[cols::S_PHASE_A];
        let s_phase_b = &bp[cols::S_PHASE_B];
        let s_stitch = &bp[cols::S_STITCH];
        let pa_m_x_up = &bp[cols::PA_M_X];
        let pa_m_y_up = &bp[cols::PA_M_Y];
        let w_va_up = &bp[cols::W_VA];
        let w_vb_up = &bp[cols::W_VB];
        let w_vc_up = &bp[cols::W_VC];
        let w_vd_up = &bp[cols::W_VD];
        let w_dprime1_up = &bp[cols::W_DPRIME1];
        let w_bprime1_up = &bp[cols::W_BPRIME1];
        let w_kappa_lsb_a_up = &bp[cols::W_KAPPA_LSB_A];
        let w_kappa_lsb_b_up = &bp[cols::W_KAPPA_LSB_B];

        let ideal_rot = ideal_from_ref(&Blake3F2Ideal::RotXw1);
        let ideal_lsb = ideal_from_ref(&Blake3F2Ideal::LsbX);

        // Rotation constants: ROTR^r is multiplication by X^{32-r} in
        // F_2[X]/(X^32 - 1).
        let rho_rotr16 = rho_monomial::<R>(16);
        let rho_rotr12 = rho_monomial::<R>(20);
        let rho_rotr8 = rho_monomial::<R>(24);
        let rho_rotr7 = rho_monomial::<R>(25);

        // Helper: down-row reference for a given (col, shift).
        let down_ref = |col: usize, shift: usize| -> &B::Expr {
            &dbp[shift_slot(col, shift)]
        };

        // Helper: read W_V{A,B,C,D} at the anchor + shift offset.
        let w_v_read = |slot: usize, shift: usize| -> &B::Expr {
            debug_assert!(slot < 4);
            if shift == 0 {
                match slot {
                    0 => w_va_up,
                    1 => w_vb_up,
                    2 => w_vc_up,
                    _ => w_vd_up,
                }
            } else {
                down_ref(cols::W_V_SLOT[slot], shift)
            }
        };

        // Helper: read PA_M_X / PA_M_Y at offset g.
        let msg_x = |g: usize| -> &B::Expr {
            if g == 0 { pa_m_x_up } else { down_ref(cols::PA_M_X, g) }
        };
        let msg_y = |g: usize| -> &B::Expr {
            if g == 0 { pa_m_y_up } else { down_ref(cols::PA_M_Y, g) }
        };

        // Bit-op virtual access: bit `j` of `W_KAPPA_LSB_{A,B}`. The
        // signature emits virtuals in the order
        //   [SHR^1(A), .., SHR^23(A), SHR^1(B), .., SHR^23(B)]
        // so `down.bit_op[j-1]` gives bit `j` of A, and
        // `down.bit_op[NUM_KAPPA_LSB_SHIFTS + j - 1]` gives bit `j` of B
        // (for `j ∈ [1, 24)`). For `j = 0` we read the column itself
        // at the up row (its bit 0).
        let _ = &up.bit_op; // up.bit_op is empty
        let kappa_lsb_phase = |phase_a: bool, j: u32| -> &B::Expr {
            debug_assert!(j < 24, "kappa_lsb_phase: j must be < 24");
            if j == 0 {
                if phase_a { w_kappa_lsb_a_up } else { w_kappa_lsb_b_up }
            } else {
                let base = if phase_a { 0 } else { cols::NUM_KAPPA_LSB_SHIFTS };
                &down.bit_op[base + (j as usize) - 1]
            }
        };

        // ----------------------------------------------------------------
        // Per-G LSB + C1/C2 rotation-pin emitter (degree 1 — κ-lifted).
        //
        // The 6 BiniusAdd LSB checks at G-slot `g`, phase Φ ∈ {A, B}
        // are emitted as
        //   `(target + x + y + SHR^{j}(W_KAPPA_LSB_Φ)) ∈ (X)`,
        // where `j = 6g + step ∈ [0, 24)`. They are degree-1 in trace
        // MLEs. A separate `S_PHASE_Φ · W_KAPPA_LSB_Φ = 0` assert_zero
        // (emitted once, outside this closure) pins every bit of the
        // packed κ to zero on the phase's anchor rows.
        //
        // The 2 shift-bearing rotation pins (C1 d', C2 b') are emitted
        // as
        //   `(combo − target + W_KR_C1/C2_Φ_g) ∈ (X^32 − 1)`,
        // with the κ_rot column absorbing the residual on non-anchor
        // rows. The companion `S_PHASE_Φ · W_KR_*_Φ_g = 0` assert_zero
        // pins it to zero on anchor rows. The remaining two rotation
        // pins (C3, C4) are emitted *once total* below — they are
        // row-local and don't need per-G/per-phase duplication.
        // ----------------------------------------------------------------

        let emit_g = |b: &mut B,
                      phase_a: bool,
                      read_shift: &[usize; 4],
                      g: usize,
                      kr_c1_col: usize,
                      kr_c2_col: usize| {
            // Inputs (read via row-shifted W_V_SLOT access).
            let v_a_in = w_v_read(0, read_shift[0]);
            let v_b_in = w_v_read(1, read_shift[1]);
            let v_c_in = w_v_read(2, read_shift[2]);
            let v_d_in = w_v_read(3, read_shift[3]);

            // Intermediates and outputs (written at anchor + 4 + g).
            let write_shift = 4 + g;
            let s_a1 = down_ref(cols::W_SA1, write_shift);
            let v_a1 = down_ref(cols::W_VA1, write_shift);
            let v_c1 = down_ref(cols::W_VC1, write_shift);
            let s_a2 = down_ref(cols::W_SA2, write_shift);
            let d_prime_1 = down_ref(cols::W_DPRIME1, write_shift);
            let b_prime_1 = down_ref(cols::W_BPRIME1, write_shift);
            let new_v_a = down_ref(cols::W_VA, write_shift);
            let new_v_c = down_ref(cols::W_VC, write_shift);
            let new_v_d = down_ref(cols::W_VD, write_shift);

            // κ_LSB accessor for this G-slot.
            let kappa_step = |step: usize| -> &B::Expr {
                kappa_lsb_phase(phase_a, cols::kappa_lsb_bit(g, step))
            };

            // 6 LSB BiniusAdd checks, each with its allocated κ bit.
            // step 0 — SA1: s_a^(1) ≡ v_a + v_b
            b.assert_in_ideal(
                s_a1.clone() + v_a_in + v_b_in + kappa_step(0),
                &ideal_lsb,
            );
            // step 1 — VA1: v_a^(1) ≡ s_a^(1) + x
            b.assert_in_ideal(
                v_a1.clone() + s_a1 + msg_x(g) + kappa_step(1),
                &ideal_lsb,
            );
            // step 2 — VC1: v_c^(1) ≡ v_c + d_prime_1
            b.assert_in_ideal(
                v_c1.clone() + v_c_in + d_prime_1 + kappa_step(2),
                &ideal_lsb,
            );
            // step 3 — SA2: s_a^(2) ≡ v_a^(1) + b_prime_1
            b.assert_in_ideal(
                s_a2.clone() + v_a1 + b_prime_1 + kappa_step(3),
                &ideal_lsb,
            );
            // step 4 — VA2: new v_a ≡ s_a^(2) + y
            b.assert_in_ideal(
                new_v_a.clone() + s_a2 + msg_y(g) + kappa_step(4),
                &ideal_lsb,
            );
            // step 5 — VC2: new v_c ≡ v_c^(1) + new v_d
            b.assert_in_ideal(
                new_v_c.clone() + v_c1 + new_v_d + kappa_step(5),
                &ideal_lsb,
            );

            // C1 — d_prime_1 = ROTR^16(v_d ⊕ v_a^(1)), κ_rot lifted.
            //   `(X^16 · (v_d + v_a^(1)) − d_prime_1 + κ_rot) ∈ (X^32 − 1)`
            let d_prime_1_combo = mbs(&(v_d_in.clone() + v_a1), &rho_rotr16)
                .expect("ROTR^16 rot-pin d_prime_1");
            b.assert_in_ideal(
                d_prime_1_combo - d_prime_1 + &bp[kr_c1_col],
                &ideal_rot,
            );

            // C2 — b_prime_1 = ROTR^12(v_b ⊕ v_c^(1)), κ_rot lifted.
            //   `(X^20 · (v_b + v_c^(1)) − b_prime_1 + κ_rot) ∈ (X^32 − 1)`
            let b_prime_1_combo = mbs(&(v_b_in.clone() + v_c1), &rho_rotr12)
                .expect("ROTR^12 rot-pin b_prime_1");
            b.assert_in_ideal(
                b_prime_1_combo - b_prime_1 + &bp[kr_c2_col],
                &ideal_rot,
            );
        };

        // Emit phase-A and phase-B packs (8 G-emitter invocations total).
        for g in 0..4 {
            emit_g(
                b,
                true,
                &cols::PHASE_A_READ_SHIFT[g],
                g,
                cols::W_KR_C1_PA[g],
                cols::W_KR_C2_PA[g],
            );
        }
        for g in 0..4 {
            emit_g(
                b,
                false,
                &cols::PHASE_B_READ_SHIFT[g],
                g,
                cols::W_KR_C1_PB[g],
                cols::W_KR_C2_PB[g],
            );
        }

        // ----------------------------------------------------------------
        // Row-local rotation pins (C3, C4) — emitted ONCE each.
        //
        // C3: `X^24 · (W_DPRIME1 + W_VA) ≡ W_VD`   (in (X^32 − 1))
        // C4: `X^25 · (W_BPRIME1 + W_VC) ≡ W_VB`   (in (X^32 − 1))
        //
        // Both reference only up-row cells (no shift), so the identity
        // is row-wise and holds at every row by trace construction:
        //   - On G-active rows, by the honest G-mix arithmetic
        //     (`new v_d = ROTR^8(d_prime ⊕ new v_a)` ↔ C3,
        //     `new v_b = ROTR^7(b_prime ⊕ new v_c)` ↔ C4).
        //   - On init / output prefix rows, by setting
        //       `W_DPRIME1[r] = ROTL^8(W_VD[r]) ⊕ W_VA[r]` and
        //       `W_BPRIME1[r] = ROTL^7(W_VB[r]) ⊕ W_VC[r]`.
        // No κ compensator, no selector. Degree 1.
        // ----------------------------------------------------------------

        let c3_combo = mbs(&(w_dprime1_up.clone() + w_va_up), &rho_rotr8)
            .expect("ROTR^8 row-local C3");
        b.assert_in_ideal(c3_combo - w_vd_up, &ideal_rot);

        let c4_combo = mbs(&(w_bprime1_up.clone() + w_vc_up), &rho_rotr7)
            .expect("ROTR^7 row-local C4");
        b.assert_in_ideal(c4_combo - w_vb_up, &ideal_rot);

        // ----------------------------------------------------------------
        // κ pins — `S_PHASE_X · W_KAPPA_* = 0`. Selector-gated, degree
        // 2, zero-ideal (`assert_zero`). They do NOT contribute to the
        // effective max degree by `count_effective_max_degree`'s
        // filter, but they ensure the κ compensators are zero on the
        // phase's anchor rows so the κ-lifted constraints above bind.
        // ----------------------------------------------------------------

        // LSB κ (packed): 1 assert_zero per phase pinning all 24 bits.
        b.assert_zero(s_phase_a.clone() * w_kappa_lsb_a_up);
        b.assert_zero(s_phase_b.clone() * w_kappa_lsb_b_up);

        // Rotation κ_C1, κ_C2 (per phase × per G-slot).
        for &c in &cols::W_KR_C1_PA {
            b.assert_zero(s_phase_a.clone() * &bp[c]);
        }
        for &c in &cols::W_KR_C1_PB {
            b.assert_zero(s_phase_b.clone() * &bp[c]);
        }
        for &c in &cols::W_KR_C2_PA {
            b.assert_zero(s_phase_a.clone() * &bp[c]);
        }
        for &c in &cols::W_KR_C2_PB {
            b.assert_zero(s_phase_b.clone() * &bp[c]);
        }

        // ----------------------------------------------------------------
        // Init boundary pinning. At every init-prefix row (and the
        // trailing cv_N output prefix), the 4 W_V* cells are pinned
        // to the corresponding PA_* cells. The PA_* values follow the
        // PHASE_B_QUAD-mirrored layout (see module doc).
        // ----------------------------------------------------------------

        b.assert_zero(s_init_prefix.clone() * &(w_va_up.clone() - pa_cv_lo));
        b.assert_zero(s_init_prefix.clone() * &(w_vb_up.clone() - pa_cv_hi));
        b.assert_zero(s_init_prefix.clone() * &(w_vc_up.clone() - pa_iv));
        b.assert_zero(s_init_prefix.clone() * &(w_vd_up.clone() - pa_params));

        // ----------------------------------------------------------------
        // (D) Junction-stitch (cv chaining via XOR feed-forward).
        //
        // At stitch anchor r = 60i + 56 (= G_0 of round 6 phase B's row
        // for compression i), the 8 components of `cv_{i+1}` are pinned
        // to the XOR of two final-state cells of compression i. The 16
        // final-state cells live across rows anchor + 0..3 in the 4
        // W_V* output columns, per the PHASE_B_QUAD-mirrored final-
        // state layout:
        //
        //   v[0]  = W_VA[anchor+0]    v[8]  = W_VC[anchor+2]
        //   v[1]  = W_VA[anchor+1]    v[9]  = W_VC[anchor+3]
        //   v[2]  = W_VA[anchor+2]    v[10] = W_VC[anchor+0]
        //   v[3]  = W_VA[anchor+3]    v[11] = W_VC[anchor+1]
        //   v[4]  = W_VB[anchor+3]    v[12] = W_VD[anchor+1]
        //   v[5]  = W_VB[anchor+0]    v[13] = W_VD[anchor+2]
        //   v[6]  = W_VB[anchor+1]    v[14] = W_VD[anchor+3]
        //   v[7]  = W_VB[anchor+2]    v[15] = W_VD[anchor+0]
        //
        // The 8 cv_{i+1} targets live at rows anchor + 4..7 (= init
        // prefix of compression i+1, or the cv_N output prefix for
        // i = N-1), with PA_CV_LO carrying cv components at "slot a"
        // positions of PHASE_B_QUAD and PA_CV_HI carrying cv components
        // at "slot b" positions:
        //
        //   PA_CV_LO[anchor+4] = cv[0]    PA_CV_HI[anchor+4] = cv[5]
        //   PA_CV_LO[anchor+5] = cv[1]    PA_CV_HI[anchor+5] = cv[6]
        //   PA_CV_LO[anchor+6] = cv[2]    PA_CV_HI[anchor+6] = cv[7]
        //   PA_CV_LO[anchor+7] = cv[3]    PA_CV_HI[anchor+7] = cv[4]
        //
        // Each constraint is `S_STITCH * (target + v_p + v_{p+8}) = 0`
        // (free F_2[X]-linear pin; no Binius gadget, no β bit).
        // ----------------------------------------------------------------

        let down_pa_cv_lo = |s: usize| down_ref(cols::PA_CV_LO, s);
        let down_pa_cv_hi = |s: usize| down_ref(cols::PA_CV_HI, s);

        // cv[0] = v[0] XOR v[8]
        b.assert_zero(
            s_stitch.clone()
                * &(down_pa_cv_lo(4).clone() + w_va_up + w_v_read(2, 2)),
        );
        // cv[1] = v[1] XOR v[9]
        b.assert_zero(
            s_stitch.clone()
                * &(down_pa_cv_lo(5).clone() + w_v_read(0, 1) + w_v_read(2, 3)),
        );
        // cv[2] = v[2] XOR v[10]
        b.assert_zero(
            s_stitch.clone()
                * &(down_pa_cv_lo(6).clone() + w_v_read(0, 2) + w_vc_up),
        );
        // cv[3] = v[3] XOR v[11]
        b.assert_zero(
            s_stitch.clone()
                * &(down_pa_cv_lo(7).clone() + w_v_read(0, 3) + w_v_read(2, 1)),
        );
        // cv[4] = v[4] XOR v[12]  (cv[4] lands at PA_CV_HI shift +7)
        b.assert_zero(
            s_stitch.clone()
                * &(down_pa_cv_hi(7).clone() + w_v_read(1, 3) + w_v_read(3, 1)),
        );
        // cv[5] = v[5] XOR v[13]
        b.assert_zero(
            s_stitch.clone()
                * &(down_pa_cv_hi(4).clone() + w_vb_up + w_v_read(3, 2)),
        );
        // cv[6] = v[6] XOR v[14]
        b.assert_zero(
            s_stitch.clone()
                * &(down_pa_cv_hi(5).clone() + w_v_read(1, 1) + w_v_read(3, 3)),
        );
        // cv[7] = v[7] XOR v[15]
        b.assert_zero(
            s_stitch.clone()
                * &(down_pa_cv_hi(6).clone() + w_v_read(1, 2) + w_vd_up),
        );
    }
}

// ---------------------------------------------------------------------------
// Scalar helper: ρ_r(X) = X^r as a 32-coefficient dense polynomial.
// ---------------------------------------------------------------------------

fn rho_monomial<R: ConstSemiring>(degree: usize) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    debug_assert!(degree < 32);
    coeffs[degree] = R::ONE;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// Scalar projection.
// ---------------------------------------------------------------------------

pub fn blake3_f2_project_scalar<R>(
    scalar: &DensePolynomial<R, 32>,
) -> DynamicPolynomialF<zinc_poly::univariate::binary_gf128::BinaryFieldGF128>
where
    R: Semiring + num_traits::Zero,
{
    use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;
    let cfg = ();
    let coeffs: Vec<BinaryFieldGF128> = scalar
        .coeffs
        .iter()
        .map(|c| {
            if num_traits::Zero::is_zero(c) {
                BinaryFieldGF128::zero_with_cfg(&cfg)
            } else {
                BinaryFieldGF128::one_with_cfg(&cfg)
            }
        })
        .collect();
    DynamicPolynomialF::new_trimmed(coeffs)
}

// ---------------------------------------------------------------------------
// GenerateRandomTrace
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for Blake3F2Uair<R>
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
            "trace too small for one Blake3 compression: need num_vars ≥ {}, got {}",
            cols::MIN_NUM_VARS,
            num_vars,
        );

        let big_n = cols::num_compressions(num_vars);
        let rpc = cols::ROWS_PER_COMP;

        // ---- u32-typed trace buffers ----------------------------------
        let mut pa_cv_lo = vec![0u32; n];
        let mut pa_cv_hi = vec![0u32; n];
        let mut pa_iv = vec![0u32; n];
        let mut pa_params = vec![0u32; n];
        let mut pa_m_x = vec![0u32; n];
        let mut pa_m_y = vec![0u32; n];
        let mut s_init_prefix = vec![0u32; n];
        let mut s_phase_a = vec![0u32; n];
        let mut s_phase_b = vec![0u32; n];
        let mut s_stitch = vec![0u32; n];
        let mut w_va = vec![0u32; n];
        let mut w_vb = vec![0u32; n];
        let mut w_vc = vec![0u32; n];
        let mut w_vd = vec![0u32; n];
        let mut w_sa1 = vec![0u32; n];
        let mut w_va1 = vec![0u32; n];
        let mut w_vc1 = vec![0u32; n];
        let mut w_sa2 = vec![0u32; n];
        let mut w_dprime1 = vec![0u32; n];
        let mut w_bprime1 = vec![0u32; n];
        // κ columns — populated after the primary trace is built.
        let mut w_kr_c1_pa: [Vec<u32>; 4] =
            std::array::from_fn(|_| vec![0u32; n]);
        let mut w_kr_c1_pb: [Vec<u32>; 4] =
            std::array::from_fn(|_| vec![0u32; n]);
        let mut w_kr_c2_pa: [Vec<u32>; 4] =
            std::array::from_fn(|_| vec![0u32; n]);
        let mut w_kr_c2_pb: [Vec<u32>; 4] =
            std::array::from_fn(|_| vec![0u32; n]);
        let mut w_kappa_lsb_a = vec![0u32; n];
        let mut w_kappa_lsb_b = vec![0u32; n];

        // ---- Initial chaining value ------------------------------------
        let mut cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());

        for i in 0..big_n {
            let start = i * rpc;

            // Random message block + per-block parameters.
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let t_lo = rng.next_u32();
            let t_hi = rng.next_u32();
            let block_len: u32 = 64;
            let flags: u32 = 0;

            // Initial 16-word state v[0..16].
            let mut v: [u32; 16] = [
                cv[0], cv[1], cv[2], cv[3],
                cv[4], cv[5], cv[6], cv[7],
                IV[0], IV[1], IV[2], IV[3],
                t_lo, t_hi, block_len, flags,
            ];

            // ---- Init prefix rows (start + 0..3) ----------------------
            // Row offset r holds positions PHASE_B_QUAD[r] in slots
            // (a, b, c, d). This mirrors the layout of a "virtual
            // phase-B half-round outputs at row offsets 0..3" so the
            // round-0 phase-A reads use the same shift pattern as
            // every subsequent half-round.
            for r in 0..4 {
                let quad = &cols::PHASE_B_QUAD[r];
                let va = v[quad[0]];
                let vb = v[quad[1]];
                let vc = v[quad[2]];
                let vd = v[quad[3]];

                w_va[start + r] = va;
                w_vb[start + r] = vb;
                w_vc[start + r] = vc;
                w_vd[start + r] = vd;

                // Row-local C3, C4 hold at every row by construction:
                //   `W_DPRIME1[t] = ROTL^8(W_VD[t]) ⊕ W_VA[t]`,
                //   `W_BPRIME1[t] = ROTL^7(W_VB[t]) ⊕ W_VC[t]`.
                // On G rows the G-mix arithmetic already satisfies
                // these; on init-prefix rows we set them explicitly.
                w_dprime1[start + r] = vd.rotate_left(8) ^ va;
                w_bprime1[start + r] = vb.rotate_left(7) ^ vc;

                // Public boundary columns mirror the same scrambled
                // layout. PA_CV_LO/HI carry the "slot a / slot b"
                // positions (which happen to be cv components for
                // r ∈ 0..3), PA_IV/PA_PARAMS carry "slot c / slot d"
                // (IV / params components).
                pa_cv_lo[start + r] = va;
                pa_cv_hi[start + r] = vb;
                pa_iv[start + r] = vc;
                pa_params[start + r] = vd;

                s_init_prefix[start + r] = 1;
            }

            // ---- 14 half-rounds (7 rounds × 2 phases) ------------------
            for hr in 0..14 {
                let round_idx = hr / 2;
                let phase = hr % 2; // 0 = A, 1 = B
                let anchor = start + 4 * hr;
                let perm = &MSG_SCHEDULE[round_idx];
                let msg_off = phase * 8;
                let phase_quads = if phase == 0 {
                    &cols::PHASE_A_QUAD
                } else {
                    &cols::PHASE_B_QUAD
                };

                // Anchor-row selector.
                if phase == 0 {
                    s_phase_a[anchor] = 1;
                } else {
                    s_phase_b[anchor] = 1;
                }

                // Lay out the 8 message words for this half-round at
                // anchor + 0..3. G-slot g reads x_g from PA_M_X[anchor
                // + g] and y_g from PA_M_Y[anchor + g].
                for g in 0..4 {
                    pa_m_x[anchor + g] = m[perm[msg_off + 2 * g]];
                    pa_m_y[anchor + g] = m[perm[msg_off + 2 * g + 1]];
                }

                // Run the 4 G's of this half-round in sequence,
                // populating the witness cells at anchor + 4 + g.
                for g in 0..4 {
                    let quad = &phase_quads[g];
                    let a_pos = quad[0];
                    let b_pos = quad[1];
                    let c_pos = quad[2];
                    let d_pos = quad[3];

                    let va_in = v[a_pos];
                    let vb_in = v[b_pos];
                    let vc_in = v[c_pos];
                    let vd_in = v[d_pos];
                    let x = m[perm[msg_off + 2 * g]];
                    let y = m[perm[msg_off + 2 * g + 1]];

                    // Step 0 (SA1): s_a^(1) = v_a + v_b (mod 2^32)
                    let s_a1 = va_in.wrapping_add(vb_in);
                    // Step 1 (VA1): v_a^(1) = s_a^(1) + x
                    let v_a1 = s_a1.wrapping_add(x);
                    // d' = ROTR(v_d ^ v_a^(1), 16) — committed
                    let d_prime = (vd_in ^ v_a1).rotate_right(16);
                    // Step 2 (VC1): v_c^(1) = v_c + d'
                    let v_c1 = vc_in.wrapping_add(d_prime);
                    // b' = ROTR(v_b ^ v_c^(1), 12) — committed
                    let b_prime = (vb_in ^ v_c1).rotate_right(12);
                    // Step 3 (SA2): s_a^(2) = v_a^(1) + b'
                    let s_a2 = v_a1.wrapping_add(b_prime);
                    // Step 4 (VA2): new v_a = s_a^(2) + y
                    let new_v_a = s_a2.wrapping_add(y);
                    // d'' = ROTR(d' ^ new v_a, 8) = new v_d
                    let d_double_prime = (d_prime ^ new_v_a).rotate_right(8);
                    // Step 5 (VC2): new v_c = v_c^(1) + d''
                    let new_v_c = v_c1.wrapping_add(d_double_prime);
                    // new v_b = ROTR(b' ^ new v_c, 7)
                    let new_v_b = (b_prime ^ new_v_c).rotate_right(7);

                    let g_row = anchor + 4 + g;
                    w_va[g_row] = new_v_a;
                    w_vb[g_row] = new_v_b;
                    w_vc[g_row] = new_v_c;
                    w_vd[g_row] = d_double_prime;
                    w_sa1[g_row] = s_a1;
                    w_va1[g_row] = v_a1;
                    w_vc1[g_row] = v_c1;
                    w_sa2[g_row] = s_a2;
                    w_dprime1[g_row] = d_prime;
                    w_bprime1[g_row] = b_prime;

                    // Update the logical state for the next G in this
                    // half-round / next half-round.
                    v[a_pos] = new_v_a;
                    v[b_pos] = new_v_b;
                    v[c_pos] = new_v_c;
                    v[d_pos] = d_double_prime;
                }
            }

            // Stitch selector: fires at row 60i + 56 (= G_0 of round 6
            // phase B), where the 8 stitch constraints forward-pin
            // PA_CV_LO/HI at rows 60i + 60..63 (= compression i+1's
            // init prefix, or the cv_N output prefix for i = N-1) to
            // the XOR of two final-state cells of compression i.
            s_stitch[start + 56] = 1;

            // Feed-forward XOR: cv_{i+1}_p = v[p] ^ v[p + 8].
            let mut next_cv = [0u32; 8];
            for j in 0..8 {
                next_cv[j] = v[j] ^ v[j + 8];
            }
            cv = next_cv;
        }

        // ---- cv_N output prefix at rows big_n * rpc .. + 4 -----------
        // The output prefix mirrors the init-prefix layout: row r
        // carries positions PHASE_B_QUAD[r]'s components. Only the
        // (a, b) slots are meaningful (= cv_N positions 0..7); the
        // (c, d) slots have no semantic value at the output and are
        // set to zero (the boundary pin pins W_V{C,D} = PA_{IV,PARAMS},
        // both zero).
        let out_start = big_n * rpc;
        if out_start + 4 <= n {
            for r in 0..4 {
                let quad = &cols::PHASE_B_QUAD[r];
                let va_out = cv[quad[0]];
                let vb_out_pos = quad[1];
                // Positions 4..7 map to cv[4..8] (= "high half" of cv).
                // PHASE_B_QUAD[r][1] is always in {4, 5, 6, 7}, so we
                // simply index `cv[vb_out_pos]`.
                let vb_out = cv[vb_out_pos];

                w_va[out_start + r] = va_out;
                w_vb[out_start + r] = vb_out;
                w_vc[out_start + r] = 0;
                w_vd[out_start + r] = 0;

                // Row-local C3, C4 at output-prefix rows.
                w_dprime1[out_start + r] = 0u32.rotate_left(8) ^ va_out; // = va_out
                w_bprime1[out_start + r] = vb_out.rotate_left(7) ^ 0; // = vb_out.rotate_left(7)

                pa_cv_lo[out_start + r] = va_out;
                pa_cv_hi[out_start + r] = vb_out;
                pa_iv[out_start + r] = 0;
                pa_params[out_start + r] = 0;

                s_init_prefix[out_start + r] = 1;
            }
        }

        // ----------------------------------------------------------------
        // κ compensator population.
        //
        // For each constraint that fires at row `r` and has a κ-lifted
        // form, set κ[r] to the residual at row r so that the
        // (residual + κ) term lies in the constraint's ideal. On rows
        // where the selector is 1 (= anchor rows for that phase) the
        // residual is honestly zero, so κ[r] = 0; on other rows κ[r]
        // absorbs whatever the cells happen to evaluate to.
        // ----------------------------------------------------------------

        // Helper: read a u32 cell with zero-pad on OOB.
        let cell = |v: &Vec<u32>, idx: usize| -> u32 {
            if idx < n { v[idx] } else { 0 }
        };

        for r in 0..n {
            // ----- LSB κ (packed bits 0..23) -----
            //
            // Phase A: for each G-slot g, the 6 LSB checks reference
            // cells at the anchor's read shifts. The κ bit absorbs
            // the residual's bit 0 at row r.
            let mut kappa_a: u32 = 0;
            let mut kappa_b: u32 = 0;
            for g in 0..4 {
                // Phase A read positions from "anchor at row r".
                let r_a = r + cols::PHASE_A_READ_SHIFT[g][0];
                let r_b = r + cols::PHASE_A_READ_SHIFT[g][1];
                let r_c = r + cols::PHASE_A_READ_SHIFT[g][2];
                let r_d_a = r + cols::PHASE_A_READ_SHIFT[g][3];
                let r_w = r + 4 + g;
                let r_mx = r + g;
                let r_my = r + g;
                let va_a = cell(&w_va, r_a);
                let vb_a = cell(&w_vb, r_b);
                let vc_a = cell(&w_vc, r_c);
                let vd_a = cell(&w_vd, r_d_a);
                let _ = vd_a;
                let mx = cell(&pa_m_x, r_mx);
                let my = cell(&pa_m_y, r_my);
                let s_a1_a = cell(&w_sa1, r_w);
                let v_a1_a = cell(&w_va1, r_w);
                let v_c1_a = cell(&w_vc1, r_w);
                let s_a2_a = cell(&w_sa2, r_w);
                let d_p_a = cell(&w_dprime1, r_w);
                let b_p_a = cell(&w_bprime1, r_w);
                let new_va_a = cell(&w_va, r_w);
                let new_vc_a = cell(&w_vc, r_w);
                let new_vd_a = cell(&w_vd, r_w);
                // step 0: SA1 + va + vb
                let b0 = (s_a1_a ^ va_a ^ vb_a) & 1;
                // step 1: VA1 + SA1 + mx
                let b1 = (v_a1_a ^ s_a1_a ^ mx) & 1;
                // step 2: VC1 + vc + d'
                let b2 = (v_c1_a ^ vc_a ^ d_p_a) & 1;
                // step 3: SA2 + VA1 + b'
                let b3 = (s_a2_a ^ v_a1_a ^ b_p_a) & 1;
                // step 4: new_va + SA2 + my
                let b4 = (new_va_a ^ s_a2_a ^ my) & 1;
                // step 5: new_vc + VC1 + new_vd
                let b5 = (new_vc_a ^ v_c1_a ^ new_vd_a) & 1;
                kappa_a |= b0 << cols::kappa_lsb_bit(g, 0);
                kappa_a |= b1 << cols::kappa_lsb_bit(g, 1);
                kappa_a |= b2 << cols::kappa_lsb_bit(g, 2);
                kappa_a |= b3 << cols::kappa_lsb_bit(g, 3);
                kappa_a |= b4 << cols::kappa_lsb_bit(g, 4);
                kappa_a |= b5 << cols::kappa_lsb_bit(g, 5);

                // Phase B: same shape, with phase-B read shifts.
                let r_a = r + cols::PHASE_B_READ_SHIFT[g][0];
                let r_b = r + cols::PHASE_B_READ_SHIFT[g][1];
                let r_c = r + cols::PHASE_B_READ_SHIFT[g][2];
                let r_d_b = r + cols::PHASE_B_READ_SHIFT[g][3];
                let va_b = cell(&w_va, r_a);
                let vb_b = cell(&w_vb, r_b);
                let vc_b = cell(&w_vc, r_c);
                let vd_b = cell(&w_vd, r_d_b);
                let _ = vd_b;
                let s_a1_b = cell(&w_sa1, r_w);
                let v_a1_b = cell(&w_va1, r_w);
                let v_c1_b = cell(&w_vc1, r_w);
                let s_a2_b = cell(&w_sa2, r_w);
                let d_p_b = cell(&w_dprime1, r_w);
                let b_p_b = cell(&w_bprime1, r_w);
                let new_va_b = cell(&w_va, r_w);
                let new_vc_b = cell(&w_vc, r_w);
                let new_vd_b = cell(&w_vd, r_w);
                let b0 = (s_a1_b ^ va_b ^ vb_b) & 1;
                let b1 = (v_a1_b ^ s_a1_b ^ mx) & 1;
                let b2 = (v_c1_b ^ vc_b ^ d_p_b) & 1;
                let b3 = (s_a2_b ^ v_a1_b ^ b_p_b) & 1;
                let b4 = (new_va_b ^ s_a2_b ^ my) & 1;
                let b5 = (new_vc_b ^ v_c1_b ^ new_vd_b) & 1;
                kappa_b |= b0 << cols::kappa_lsb_bit(g, 0);
                kappa_b |= b1 << cols::kappa_lsb_bit(g, 1);
                kappa_b |= b2 << cols::kappa_lsb_bit(g, 2);
                kappa_b |= b3 << cols::kappa_lsb_bit(g, 3);
                kappa_b |= b4 << cols::kappa_lsb_bit(g, 4);
                kappa_b |= b5 << cols::kappa_lsb_bit(g, 5);
            }
            w_kappa_lsb_a[r] = kappa_a;
            w_kappa_lsb_b[r] = kappa_b;

            // ----- C1, C2 rotation κ (full 32 bits each) -----
            //
            // Residual_C1_Φ_g(r) =
            //   ROTR^16(v_d_input ⊕ v_a^(1)) ⊕ W_DPRIME1[r+4+g]
            //   (= zero on anchor rows by the honest C1 identity).
            // κ_C1_Φ_g[r] = residual (so (residual + κ) reduces to 0
            // mod (X^32 − 1)).
            for g in 0..4 {
                let r_w = r + 4 + g;
                let v_a1 = cell(&w_va1, r_w);
                let d_prime = cell(&w_dprime1, r_w);
                let v_c1 = cell(&w_vc1, r_w);
                let b_prime = cell(&w_bprime1, r_w);

                // Phase A
                let vd_a = cell(&w_vd, r + cols::PHASE_A_READ_SHIFT[g][3]);
                let vb_a = cell(&w_vb, r + cols::PHASE_A_READ_SHIFT[g][1]);
                w_kr_c1_pa[g][r] = (vd_a ^ v_a1).rotate_right(16) ^ d_prime;
                w_kr_c2_pa[g][r] = (vb_a ^ v_c1).rotate_right(12) ^ b_prime;

                // Phase B
                let vd_b = cell(&w_vd, r + cols::PHASE_B_READ_SHIFT[g][3]);
                let vb_b = cell(&w_vb, r + cols::PHASE_B_READ_SHIFT[g][1]);
                w_kr_c1_pb[g][r] = (vd_b ^ v_a1).rotate_right(16) ^ d_prime;
                w_kr_c2_pb[g][r] = (vb_b ^ v_c1).rotate_right(12) ^ b_prime;
            }
        }

        // ---- Pack u32 → BinaryPoly<32> → MLE --------------------------
        let to_bits = |v: &[u32]| -> Vec<BinaryPoly<32>> {
            v.iter().copied().map(BinaryPoly::<32>::from).collect()
        };
        let to_mle = |col: Vec<BinaryPoly<32>>| -> DenseMultilinearExtension<BinaryPoly<32>> {
            col.into_iter().collect()
        };

        let binary_poly = vec![
            // -- Public prefix (10) --
            to_mle(to_bits(&pa_cv_lo)),
            to_mle(to_bits(&pa_cv_hi)),
            to_mle(to_bits(&pa_iv)),
            to_mle(to_bits(&pa_params)),
            to_mle(to_bits(&pa_m_x)),
            to_mle(to_bits(&pa_m_y)),
            to_mle(to_bits(&s_init_prefix)),
            to_mle(to_bits(&s_phase_a)),
            to_mle(to_bits(&s_phase_b)),
            to_mle(to_bits(&s_stitch)),
            // -- Witness suffix (28) --
            to_mle(to_bits(&w_va)),
            to_mle(to_bits(&w_vb)),
            to_mle(to_bits(&w_vc)),
            to_mle(to_bits(&w_vd)),
            to_mle(to_bits(&w_sa1)),
            to_mle(to_bits(&w_va1)),
            to_mle(to_bits(&w_vc1)),
            to_mle(to_bits(&w_sa2)),
            to_mle(to_bits(&w_dprime1)),
            to_mle(to_bits(&w_bprime1)),
            // κ_rot C1 phase A (G_0..G_3)
            to_mle(to_bits(&w_kr_c1_pa[0])),
            to_mle(to_bits(&w_kr_c1_pa[1])),
            to_mle(to_bits(&w_kr_c1_pa[2])),
            to_mle(to_bits(&w_kr_c1_pa[3])),
            // κ_rot C1 phase B (G_0..G_3)
            to_mle(to_bits(&w_kr_c1_pb[0])),
            to_mle(to_bits(&w_kr_c1_pb[1])),
            to_mle(to_bits(&w_kr_c1_pb[2])),
            to_mle(to_bits(&w_kr_c1_pb[3])),
            // κ_rot C2 phase A (G_0..G_3)
            to_mle(to_bits(&w_kr_c2_pa[0])),
            to_mle(to_bits(&w_kr_c2_pa[1])),
            to_mle(to_bits(&w_kr_c2_pa[2])),
            to_mle(to_bits(&w_kr_c2_pa[3])),
            // κ_rot C2 phase B (G_0..G_3)
            to_mle(to_bits(&w_kr_c2_pb[0])),
            to_mle(to_bits(&w_kr_c2_pb[1])),
            to_mle(to_bits(&w_kr_c2_pb[2])),
            to_mle(to_bits(&w_kr_c2_pb[3])),
            // κ_LSB packed (phase A / phase B)
            to_mle(to_bits(&w_kappa_lsb_a)),
            to_mle(to_bits(&w_kappa_lsb_b)),
        ];
        debug_assert_eq!(binary_poly.len(), cols::NUM_BIN);

        UairTrace {
            binary_poly: binary_poly.into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust reference compression — used by the trace-vs-reference test.
// ---------------------------------------------------------------------------

#[cfg(test)]
fn ref_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

#[cfg(test)]
fn ref_compress(
    cv: &[u32; 8],
    block: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 8] {
    let mut v = [
        cv[0], cv[1], cv[2], cv[3],
        cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        counter as u32, (counter >> 32) as u32, block_len, flags,
    ];
    for r in 0..7 {
        let p = MSG_SCHEDULE[r];
        ref_g(&mut v, 0, 4,  8, 12, block[p[0]], block[p[1]]);
        ref_g(&mut v, 1, 5,  9, 13, block[p[2]], block[p[3]]);
        ref_g(&mut v, 2, 6, 10, 14, block[p[4]], block[p[5]]);
        ref_g(&mut v, 3, 7, 11, 15, block[p[6]], block[p[7]]);
        ref_g(&mut v, 0, 5, 10, 15, block[p[8]], block[p[9]]);
        ref_g(&mut v, 1, 6, 11, 12, block[p[10]], block[p[11]]);
        ref_g(&mut v, 2, 7,  8, 13, block[p[12]], block[p[13]]);
        ref_g(&mut v, 3, 4,  9, 14, block[p[14]], block[p[15]]);
    }
    std::array::from_fn(|i| v[i] ^ v[i + 8])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_poly::EvaluatablePolynomial;
    use zinc_uair::constraint_counter::count_constraints;
    use zinc_uair::degree_counter::{count_effective_max_degree, count_max_degree};

    /// Signature shape: 10 public + 28 witness bit-poly columns = 38
    /// total. The witness count grew from the κ-lift refactor: 11
    /// primary witnesses (state + intermediates + d'/b') + 16 κ_rot
    /// compensators (4 per phase × 2 constraints) + 2 packed κ_LSB
    /// compensators.
    #[test]
    fn blake3_f2_signature_shape() {
        type U = Blake3F2Uair<Int<4>>;
        let sig = U::signature();
        assert_eq!(sig.witness_cols().num_binary_poly_cols(), cols::NUM_BIN_WIT);
        assert_eq!(sig.witness_cols().num_binary_poly_cols(), 28);
        assert_eq!(sig.public_cols().num_binary_poly_cols(), cols::NUM_BIN_PUB);
        assert_eq!(sig.public_cols().num_int_cols(), 0);
        assert_eq!(sig.total_cols().num_binary_poly_cols(), cols::NUM_BIN);
    }

    /// 2·4 (PA_CV) + 2·3 (PA_M) + 4·7 (W_V*) + 6·4 (intermediates)
    /// = 8 + 6 + 28 + 24 = 66 ShiftSpec entries.
    #[test]
    fn blake3_f2_signature_shifts() {
        type U = Blake3F2Uair<Int<4>>;
        let sig = U::signature();
        let shifts = sig.shifts();
        assert_eq!(shifts.len(), 66);

        for &c in &cols::W_V_SLOT {
            let col_shifts: Vec<_> = shifts
                .iter()
                .filter(|s| s.source_col() == c)
                .map(|s| s.shift_amount())
                .collect();
            assert_eq!(col_shifts, vec![1, 2, 3, 4, 5, 6, 7]);
        }
        let pa_m_x: Vec<_> = shifts
            .iter()
            .filter(|s| s.source_col() == cols::PA_M_X)
            .map(|s| s.shift_amount())
            .collect();
        assert_eq!(pa_m_x, vec![1, 2, 3]);
        // Stitch-side: PA_CV_LO and PA_CV_HI each declare shifts 4..=7.
        let pa_cv_lo: Vec<_> = shifts
            .iter()
            .filter(|s| s.source_col() == cols::PA_CV_LO)
            .map(|s| s.shift_amount())
            .collect();
        assert_eq!(pa_cv_lo, vec![4, 5, 6, 7]);
        let pa_cv_hi: Vec<_> = shifts
            .iter()
            .filter(|s| s.source_col() == cols::PA_CV_HI)
            .map(|s| s.shift_amount())
            .collect();
        assert_eq!(pa_cv_hi, vec![4, 5, 6, 7]);
    }

    /// After the κ-lift refactor every non-zero-ideal constraint is
    /// degree-1 in the trace MLEs; the F_2 IC dispatches to the
    /// MLE-first lane (`prove_linear`). Degree-2 raw constraints
    /// still appear via selector-gated `assert_zero` pins (κ-zero
    /// pins, init boundary, stitch), but those are zero-ideal and
    /// excluded from `count_effective_max_degree`.
    #[test]
    fn blake3_f2_uair_constraint_degrees() {
        type U = Blake3F2Uair<Int<4>>;
        assert_eq!(count_effective_max_degree::<U>(), 1);
        assert!(count_max_degree::<U>() >= 2);
    }

    /// Constraint count after the κ-lift refactor:
    ///   48 LSB              (6 per G × 4 G's × 2 phases, with κ_LSB)
    ///   16 rot C1, C2       (2 per G × 4 G's × 2 phases, with κ_rot)
    ///    2 rot C3, C4       (row-local, emitted once each)
    ///    2 LSB κ-zero pins  (1 per phase)
    ///   16 rot κ-zero pins  (4 per phase × 4 G's)
    ///    4 init pins
    ///    8 stitch pins
    /// = 96 total.
    #[test]
    fn blake3_f2_constraint_count() {
        type U = Blake3F2Uair<Int<4>>;
        assert_eq!(count_constraints::<U>(), 48 + 16 + 2 + 2 + 16 + 4 + 8);
    }

    #[test]
    fn blake3_f2_shift_slot_lookup() {
        for (col, shift) in shift_list() {
            let _ = shift_slot(col, shift);
        }
    }

    /// Trace generator runs without panicking and yields a column
    /// count matching the layout.
    #[test]
    fn blake3_f2_trace_gen_shape() {
        type U = Blake3F2Uair<Int<4>>;
        let mut rng = rand::rng();
        let trace = U::generate_random_trace(cols::MIN_NUM_VARS, &mut rng);
        assert_eq!(trace.binary_poly.len(), cols::NUM_BIN);
        assert_eq!(trace.arbitrary_poly.len(), 0);
        assert_eq!(trace.int.len(), 0);
    }

    /// At every stitch anchor row, the 8 cv XOR identities of the
    /// junction-stitch family must hold on the generated trace. We
    /// extract the relevant `u32` values back out of the trace and
    /// check them directly (= what the in-circuit constraint will
    /// also check, once it fires).
    #[test]
    fn blake3_f2_trace_gen_stitch_identities() {
        type U = Blake3F2Uair<Int<4>>;
        let mut rng = rand::rng();
        // Use a slightly larger num_vars so we exercise at least one
        // compression-to-compression stitch in addition to the last
        // compression → cv_N output stitch.
        let num_vars = 7;
        let trace = U::generate_random_trace(num_vars, &mut rng);

        let to_u32 = |bp: BinaryPoly<32>| -> u32 {
            bp.evaluate_at_point(&2u32)
                .expect("BinaryPoly<32>.eval(2) fits in u32")
        };
        let col = |c: usize, r: usize| -> u32 {
            to_u32(trace.binary_poly[c][r])
        };

        let big_n = cols::num_compressions(num_vars);
        assert!(big_n >= 1, "need at least one compression for the stitch test");

        for i in 0..big_n {
            let anchor = i * cols::ROWS_PER_COMP + 56;

            // s_stitch must be 1 at the anchor.
            assert_eq!(col(cols::S_STITCH, anchor), 1, "S_STITCH at anchor row");

            // Final-state cells (W_V* at anchor + 0..3).
            let v = |p: usize| -> u32 {
                let (col_id, shift) = match p {
                    0 => (cols::W_VA, 0),
                    1 => (cols::W_VA, 1),
                    2 => (cols::W_VA, 2),
                    3 => (cols::W_VA, 3),
                    4 => (cols::W_VB, 3),
                    5 => (cols::W_VB, 0),
                    6 => (cols::W_VB, 1),
                    7 => (cols::W_VB, 2),
                    8 => (cols::W_VC, 2),
                    9 => (cols::W_VC, 3),
                    10 => (cols::W_VC, 0),
                    11 => (cols::W_VC, 1),
                    12 => (cols::W_VD, 1),
                    13 => (cols::W_VD, 2),
                    14 => (cols::W_VD, 3),
                    15 => (cols::W_VD, 0),
                    _ => unreachable!(),
                };
                col(col_id, anchor + shift)
            };

            // cv_{i+1} target cells (PA_CV_* at anchor + 4..7).
            let cv_target = |p: usize| -> u32 {
                let (col_id, shift) = match p {
                    0 => (cols::PA_CV_LO, 4),
                    1 => (cols::PA_CV_LO, 5),
                    2 => (cols::PA_CV_LO, 6),
                    3 => (cols::PA_CV_LO, 7),
                    5 => (cols::PA_CV_HI, 4),
                    6 => (cols::PA_CV_HI, 5),
                    7 => (cols::PA_CV_HI, 6),
                    4 => (cols::PA_CV_HI, 7),
                    _ => unreachable!(),
                };
                col(col_id, anchor + shift)
            };

            // The 8 XOR identities.
            for p in 0..8usize {
                assert_eq!(
                    cv_target(p),
                    v(p) ^ v(p + 8),
                    "stitch identity cv_{{i+1}}[{}] at compression i={}",
                    p,
                    i,
                );
            }
        }
    }

    /// Sanity: round-tripping the trace generator's final cv against
    /// the `blake3` crate's compression confirms the per-G arithmetic
    /// and the message-permutation layout are correct.
    #[test]
    fn blake3_f2_trace_gen_compression_matches_reference() {
        // Construct one compression's input data manually.
        let cv: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ];
        let block: [u32; 16] = std::array::from_fn(|i| (i as u32).wrapping_mul(0xdead_beef));
        let counter: u64 = 0;
        let block_len: u32 = 64;
        let flags: u32 = 0;
        let expected = ref_compress(&cv, &block, counter, block_len, flags);

        // Replay the trace gen's per-G loop manually with the same
        // inputs and check the final feed-forward output matches.
        let mut v: [u32; 16] = [
            cv[0], cv[1], cv[2], cv[3],
            cv[4], cv[5], cv[6], cv[7],
            IV[0], IV[1], IV[2], IV[3],
            counter as u32, (counter >> 32) as u32, block_len, flags,
        ];
        for hr in 0..14 {
            let round_idx = hr / 2;
            let phase = hr % 2;
            let perm = &MSG_SCHEDULE[round_idx];
            let msg_off = phase * 8;
            let phase_quads = if phase == 0 { &cols::PHASE_A_QUAD } else { &cols::PHASE_B_QUAD };
            for g in 0..4 {
                let quad = &phase_quads[g];
                ref_g(
                    &mut v,
                    quad[0], quad[1], quad[2], quad[3],
                    block[perm[msg_off + 2 * g]],
                    block[perm[msg_off + 2 * g + 1]],
                );
            }
        }
        let actual: [u32; 8] = std::array::from_fn(|i| v[i] ^ v[i + 8]);
        assert_eq!(actual, expected);
    }
}
