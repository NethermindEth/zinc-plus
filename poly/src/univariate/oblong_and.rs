//! Phase-1 of the Binius64-style **oblong univariate AND zerocheck** for
//! the F_2 Hadamard discharge: the additive-NTT word extension and the
//! univariate-skip round message `R₀(Z)`.
//!
//! Port plan: `documentation/f2-hadamard-oblong-port-plan.md` (§1 protocol,
//! P4/P5). Reference: binius64
//! `crates/prover/src/and_reduction/sumcheck_round_messages.rs`. This module
//! is the **naive `GF(2^128)` correctness path** — it runs the NTT and the
//! per-point product in the full field, with no `GF(2^8)` byte-lookup table
//! yet (that is the prover-side speed lever, a follow-up). Its purpose is to
//! de-risk the core math: that the oblong round message is correct over our
//! [`BinaryFieldGF128`].
//!
//! ## The constraint, restated
//!
//! We argue `A ⊙ B = C` (bitwise AND of `WORD_BITS`-bit words) for a column
//! of `2ⁿ` words, *without* splitting into `WORD_BITS` `GF(2^128)` bit-slices
//! (the memory-bound representation the discharge uses today). The operands
//! are **oblong** multilinears `A(Z, X₀…Xₙ₋₁)`: `Z ∈ {0,1}^{SKIPPED_VARS}` is
//! the bit-index within a word, `X` is the word/row index. For a fixed word
//! `X`, `A(·, X)` is the degree-`< WORD_BITS` univariate whose evaluations on
//! the `WORD_BITS`-point base subspace are exactly that word's bits.
//!
//! ## Phase 1 (this module)
//!
//! The prover sends the univariate
//! ```text
//!   R₀(Z) = Σ_{X ∈ {0,1}ⁿ} (A·B − C)(Z, X) · eq(X; r)
//! ```
//! over a `(SKIPPED_VARS+1)`-dim subspace (`2·WORD_BITS` points). Because the
//! AND holds, `A·B − C = bit·bit − bit = 0` on every **base** subspace point,
//! so `R₀` vanishes there and the prover sends only its `WORD_BITS`
//! **extension**-domain evaluations. `deg R₀ ≤ 2(WORD_BITS − 1)`, so the
//! base zeros plus the extension evals over-determine it.
//!
//! The verifier prepends `WORD_BITS` zeros, samples `z`, extrapolates `R₀(z)`
//! (via [`super::binary_subspace::extrapolate_over_subspace`]), and proceeds
//! to a standard degree-2 sumcheck over the `n` row variables (Phase 2, not
//! in this module). The cross-check test below is exactly binius64's
//! `test_first_round_message_matches_next_round_sum_claim`: `R₀(z)` recovered
//! from the round message must equal the brute-force folded sum claim.

use super::binary_gf128::BinaryFieldGF128;
use super::binary_subspace::{BinarySubspace, lagrange_evals};

type F = BinaryFieldGF128;

/// `log2(WORD_BITS)` — the number of bit-index variables fused into the
/// univariate-skip round. `5` for the 32-bit SHA-256 words.
pub const SKIPPED_VARS: usize = 5;

/// Bits per word = size of the base/extension univariate domain (`|D|`).
/// Binius calls this `ROWS_PER_HYPERCUBE_VERTEX`. `32` for SHA-256.
pub const WORD_BITS: usize = 1 << SKIPPED_VARS;

/// Precomputed additive-NTT that extends a packed word (its `WORD_BITS`
/// bits = base-domain evaluations) to its `WORD_BITS` extension-domain
/// evaluations.
///
/// For a word `w`, `A(·)` is the degree-`< WORD_BITS` univariate with
/// `A(base.get(b)) = bit_b(w)`. Its value at extension point `j` is
/// `A(ext_j) = Σ_b bit_b(w) · L_b(ext_j)`, where `L_b` is the Lagrange
/// basis of the base subspace. Since `bit_b ∈ {0,1}`, this is a pure
/// select-and-XOR over the precomputed rows `ext_lagrange[j][b] = L_b(ext_j)`
/// — **no multiplications**. (The fast path replaces the per-bit XOR with an
/// 8-bit byte lookup; same values, fewer ops.)
#[derive(Debug, Clone)]
pub struct AdditiveNtt {
    /// `ext_lagrange[j][b] = L_b(extension_point_j)`, `j, b ∈ 0..WORD_BITS`.
    ext_lagrange: Vec<[F; WORD_BITS]>,
}

impl Default for AdditiveNtt {
    fn default() -> Self {
        Self::new()
    }
}

impl AdditiveNtt {
    /// Precompute the extension-domain Lagrange rows for `WORD_BITS`-bit
    /// words over the default monomial basis.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new() -> Self {
        let base = BinarySubspace::with_dim(SKIPPED_VARS); // dim 5 → 32 base points
        let full = BinarySubspace::with_dim(SKIPPED_VARS + 1); // dim 6 → 64 points
        let ext_lagrange = (0..WORD_BITS)
            .map(|j| {
                // Extension point j is full.get(WORD_BITS + j); for the
                // monomial basis that is the field element (WORD_BITS + j).
                let ext_point = full.get(WORD_BITS + j);
                let lag = lagrange_evals(&base, ext_point);
                let mut row = [F::zero(); WORD_BITS];
                row.copy_from_slice(&lag);
                row
            })
            .collect();
        Self { ext_lagrange }
    }

    /// Extension-domain evaluations of the word `w` (low `WORD_BITS` bits):
    /// `out[j] = A(extension_point_j)`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn extend_word(&self, w: u32) -> [F; WORD_BITS] {
        let mut out = [F::zero(); WORD_BITS];
        let mut bits = w;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let row = &self.ext_lagrange[b.min(WORD_BITS - 1)];
            for (o, &r) in out.iter_mut().zip(row.iter()) {
                *o += r;
            }
        }
        out
    }

    /// Fold a word at a univariate challenge `z`: `A(z) = Σ_b bit_b(w)·L_b(z)`
    /// over the **base** subspace. This is the Phase-1→Phase-2 transition
    /// (binius `fold_words_with_transform`), exposed here for the cross-check.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn fold_word_at(base_lagrange_z: &[F; WORD_BITS], w: u32) -> F {
        let mut acc = F::zero();
        let mut bits = w;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            acc += base_lagrange_z[b.min(WORD_BITS - 1)];
        }
        acc
    }
}

/// The base-subspace Lagrange weights at `z` (`L_b(z)`, `b ∈ 0..WORD_BITS`),
/// used to fold words at the univariate challenge.
pub fn base_lagrange_at(z: F) -> [F; WORD_BITS] {
    let base = BinarySubspace::with_dim(SKIPPED_VARS);
    let lag = lagrange_evals(&base, z);
    let mut row = [F::zero(); WORD_BITS];
    row.copy_from_slice(&lag);
    row
}

/// The Phase-1 oblong round message: the `WORD_BITS` **extension-domain**
/// evaluations of
/// `R₀(Z) = Σ_X (A·B − C)(Z, X) · eq(X; r)`.
///
/// `a_words`/`b_words`/`c_words` are the packed operand columns (one word per
/// row `X`, `2ⁿ` rows). `eq` is the equality indicator over the `n` row
/// variables (`eq[X] = eq(X; r)`, length `2ⁿ`). The result is `R₀` on the
/// extension domain; its base-domain values are zero when `c = a & b`.
#[allow(clippy::arithmetic_side_effects)]
pub fn univariate_round_message(
    a_words: &[u32],
    b_words: &[u32],
    c_words: &[u32],
    eq: &[F],
    ntt: &AdditiveNtt,
) -> [F; WORD_BITS] {
    let n = a_words.len();
    assert_eq!(b_words.len(), n, "operand columns must have equal length");
    assert_eq!(c_words.len(), n, "operand columns must have equal length");
    assert_eq!(eq.len(), n, "eq indicator must cover every row");

    let mut acc = [F::zero(); WORD_BITS];
    for x in 0..n {
        let ae = ntt.extend_word(a_words[x]);
        let be = ntt.extend_word(b_words[x]);
        let ce = ntt.extend_word(c_words[x]);
        let w = eq[x];
        for j in 0..WORD_BITS {
            acc[j] += (ae[j] * be[j] - ce[j]) * w;
        }
    }
    acc
}

/// Tensor-product equality indicator `eq[X] = ∏_i (bit_i(X)? r_i : 1−r_i)`,
/// **little-endian**: bit `i` of the row index `X` is the variable bound to
/// `r[i]`. Length `2^|r|`. Building it little-endian (new variable `k` placed
/// at value `2^k`, i.e. the high half of the current table) is what makes it
/// agree with [`fold_low`], which binds bit 0 first to `gammas[0]`, and with
/// the verifier's `eq_star = ∏ eq1(gammas[k]; r[k])`.
#[allow(clippy::arithmetic_side_effects)]
pub fn eq_indicator(r: &[F]) -> Vec<F> {
    let mut ev = vec![F::one()];
    for &ri in r {
        let half = ev.len();
        let mut next = vec![F::zero(); half * 2];
        for j in 0..half {
            next[j] = ev[j] * (F::one() - ri); // bit k = 0 (low half)
            next[j + half] = ev[j] * ri; // bit k = 1 (high half ⇒ value 2^k)
        }
        ev = next;
    }
    ev
}

/// Output of the oblong AND zerocheck: the claimed operand evaluations at the
/// oblong point `(z, γ₀…γₙ₋₁)`, plus that evaluation point (binius
/// `AndCheckOutput`). These are what the ψ_α integration seam (port plan §4)
/// must tie back to the committed columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AndCheckOutput {
    pub a_eval: F,
    pub b_eval: F,
    pub c_eval: F,
    /// `[z, γ₀, …, γₙ₋₁]` — the univariate (bit) challenge prepended to the
    /// row-variable sumcheck challenges.
    pub eval_point: Vec<F>,
}

/// Proof produced by [`prove_oblong_and`]: the Phase-1 extension-domain round
/// message, the Phase-2 degree-≤3 round polynomials (4 evals each, over the
/// dim-2 subspace `{0,1,2,3}`), and the closing operand evals.
#[derive(Clone, Debug)]
pub struct OblongAndProof {
    pub round_message: [F; WORD_BITS],
    pub round_polys: Vec<[F; 4]>,
    pub a_eval: F,
    pub b_eval: F,
    pub c_eval: F,
}

/// Why an oblong AND proof was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OblongError {
    #[error("round {0}: g(0)+g(1) ≠ claim")]
    RoundConsistency(usize),
    #[error("closing check (a·b−c)·eq ≠ final claim")]
    FinalCheck,
    #[error("malformed proof: {0} round polynomials, expected {1}")]
    Shape(usize, usize),
}

/// The four base points `{0, 1, X, X+1}` of the dim-2 subspace, on which each
/// Phase-2 round polynomial (degree ≤ 3) is reported.
#[inline]
fn round_poly_points() -> [F; 4] {
    [
        F::from_words([0, 0]),
        F::from_words([1, 0]),
        F::from_words([2, 0]),
        F::from_words([3, 0]),
    ]
}

/// Phase-2 round polynomial `g(t) = Σ_rest (a_t·b_t − c_t)·eq_t`, where the
/// **low** bound variable's pair `(2i, 2i+1)` is linearly extrapolated to each
/// of the four points. Degree ≤ 3 (a·b degree 2 times eq degree 1).
#[allow(clippy::arithmetic_side_effects)]
fn round_poly(a: &[F], b: &[F], c: &[F], eq: &[F]) -> [F; 4] {
    let half = a.len() / 2;
    let pts = round_poly_points();
    let mut g = [F::zero(); 4];
    for i in 0..half {
        let (alo, ahi) = (a[2 * i], a[2 * i + 1]);
        let (blo, bhi) = (b[2 * i], b[2 * i + 1]);
        let (clo, chi) = (c[2 * i], c[2 * i + 1]);
        let (elo, ehi) = (eq[2 * i], eq[2 * i + 1]);
        let (ad, bd, cd, ed) = (ahi - alo, bhi - blo, chi - clo, ehi - elo);
        for (k, &t) in pts.iter().enumerate() {
            let at = alo + t * ad;
            let bt = blo + t * bd;
            let ct = clo + t * cd;
            let et = elo + t * ed;
            g[k] += (at * bt - ct) * et;
        }
    }
    g
}

/// Fold a multilinear table at a challenge by binding its low variable:
/// `out[i] = tbl[2i] + γ·(tbl[2i+1] − tbl[2i])`.
#[allow(clippy::arithmetic_side_effects)]
fn fold_low(tbl: &[F], gamma: F) -> Vec<F> {
    let half = tbl.len() / 2;
    (0..half)
        .map(|i| tbl[2 * i] + gamma * (tbl[2 * i + 1] - tbl[2 * i]))
        .collect()
}

/// A Fiat–Shamir channel for the oblong AND zerocheck: absorb prover messages
/// and sample `GF(2^128)` challenges. Kept transcript-agnostic so the `poly`
/// crate has no transcript dependency — the `protocol` crate supplies a
/// `Blake3Transcript` adapter, and [`ReplayChannel`] gives the
/// explicit-challenge path used by the math tests.
pub trait OblongChannel {
    /// Absorb a prover message (the round message, or a round polynomial).
    fn absorb(&mut self, scalars: &[F]);
    /// Sample the next challenge.
    fn sample(&mut self) -> F;
}

/// An [`OblongChannel`] that replays a fixed challenge sequence and ignores
/// absorbs — the explicit-challenge path. The sequence is queried in sampling
/// order: the `n` zerocheck challenges `r`, then the univariate-skip `z`, then
/// the `n` Phase-2 sumcheck challenges `γ`.
pub struct ReplayChannel {
    seq: Vec<F>,
    idx: usize,
}

impl ReplayChannel {
    pub fn new(seq: Vec<F>) -> Self {
        Self { seq, idx: 0 }
    }

    /// Build the replay sequence `[r…, z, γ…]` from explicit challenges.
    fn from_challenges(r: &[F], z: F, gammas: &[F]) -> Self {
        let mut seq = Vec::with_capacity(r.len().saturating_add(1).saturating_add(gammas.len()));
        seq.extend_from_slice(r);
        seq.push(z);
        seq.extend_from_slice(gammas);
        Self::new(seq)
    }
}

impl OblongChannel for ReplayChannel {
    fn absorb(&mut self, _scalars: &[F]) {}
    #[allow(clippy::arithmetic_side_effects)]
    fn sample(&mut self) -> F {
        let v = self.seq[self.idx];
        self.idx += 1;
        v
    }
}

/// Prover for the oblong AND zerocheck `A ⊙ B = C` (one relation), driven by a
/// Fiat–Shamir [`OblongChannel`]. The channel supplies the `n` zerocheck
/// challenges `r` (first), then `z` (after the round message is absorbed), then
/// each `γ` (after its round polynomial is absorbed) — so the challenges are not
/// part of the returned proof.
///
/// `*_words` are the packed operand columns (`2ⁿ` rows, `n = log2(len)`).
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_oblong_and_channel<C: OblongChannel>(
    ch: &mut C,
    a_words: &[u32],
    b_words: &[u32],
    c_words: &[u32],
    ntt: &AdditiveNtt,
) -> OblongAndProof {
    let len = a_words.len();
    let n = len.trailing_zeros() as usize;
    assert_eq!(1usize << n, len, "columns must have a power-of-two row count");
    assert_eq!(b_words.len(), len);
    assert_eq!(c_words.len(), len);

    // Zerocheck eq-randomness, sampled first.
    let r: Vec<F> = (0..n).map(|_| ch.sample()).collect();
    let eq = eq_indicator(&r);

    // Phase 1: round message, then the univariate-skip challenge z.
    let round_message = univariate_round_message(a_words, b_words, c_words, &eq, ntt);
    ch.absorb(&round_message);
    let z = ch.sample();

    // Phase-1 → Phase-2 transition: fold each word at z into an MLE over rows.
    let lag_z = base_lagrange_at(z);
    let mut a: Vec<F> = a_words.iter().map(|&w| AdditiveNtt::fold_word_at(&lag_z, w)).collect();
    let mut b: Vec<F> = b_words.iter().map(|&w| AdditiveNtt::fold_word_at(&lag_z, w)).collect();
    let mut c: Vec<F> = c_words.iter().map(|&w| AdditiveNtt::fold_word_at(&lag_z, w)).collect();
    let mut eqt = eq;

    // Phase 2: degree-≤3 sumcheck over the n row variables.
    let mut round_polys = Vec::with_capacity(n);
    for _ in 0..n {
        let g = round_poly(&a, &b, &c, &eqt);
        ch.absorb(&g);
        let gamma = ch.sample();
        round_polys.push(g);
        a = fold_low(&a, gamma);
        b = fold_low(&b, gamma);
        c = fold_low(&c, gamma);
        eqt = fold_low(&eqt, gamma);
    }

    OblongAndProof {
        round_message,
        round_polys,
        a_eval: a[0],
        b_eval: b[0],
        c_eval: c[0],
    }
}

/// Explicit-challenge prover: a thin wrapper over [`prove_oblong_and_channel`]
/// with a [`ReplayChannel`]. `r`/`z`/`gammas` are the `n`+1+`n` challenges in
/// sampling order. Used by the math tests; the `protocol` crate uses the
/// channel form with a real transcript.
pub fn prove_oblong_and(
    a_words: &[u32],
    b_words: &[u32],
    c_words: &[u32],
    r: &[F],
    z: F,
    gammas: &[F],
    ntt: &AdditiveNtt,
) -> OblongAndProof {
    let mut ch = ReplayChannel::from_challenges(r, z, gammas);
    prove_oblong_and_channel(&mut ch, a_words, b_words, c_words, ntt)
}

/// Verifier for the oblong AND zerocheck, driven by a Fiat–Shamir
/// [`OblongChannel`]. Samples `r`, reconstructs `R₀(z)` from the round message
/// (base zeros ++ extension evals), runs the Phase-2 sumcheck-consistency checks
/// (absorbing each round polynomial before sampling its `γ`, mirroring the
/// prover), and verifies the closing `(a·b − c)·eq(γ; r) = claim`. `n` is the
/// number of row variables.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_oblong_and_channel<C: OblongChannel>(
    ch: &mut C,
    proof: &OblongAndProof,
    n: usize,
) -> Result<AndCheckOutput, OblongError> {
    if proof.round_polys.len() != n {
        return Err(OblongError::Shape(proof.round_polys.len(), n));
    }

    let r: Vec<F> = (0..n).map(|_| ch.sample()).collect();

    // Reconstruct R₀(z): the prover sends only the extension half; the base
    // half is zero iff the AND holds, so a corrupt C is caught right here at
    // round 0 (the reconstructed claim won't match g(0)+g(1)).
    ch.absorb(&proof.round_message);
    let z = ch.sample();
    let full = BinarySubspace::with_dim(SKIPPED_VARS + 1);
    let mut coeffs = vec![F::zero(); 2 * WORD_BITS];
    coeffs[WORD_BITS..].copy_from_slice(&proof.round_message);
    let mut claim = super::binary_subspace::extrapolate_over_subspace(&full, &coeffs, z);

    let dim2 = BinarySubspace::with_dim(2);
    let mut gammas = Vec::with_capacity(n);
    for (k, g) in proof.round_polys.iter().enumerate() {
        if g[0] + g[1] != claim {
            return Err(OblongError::RoundConsistency(k));
        }
        ch.absorb(g);
        let gamma = ch.sample();
        claim = super::binary_subspace::extrapolate_over_subspace(&dim2, g, gamma);
        gammas.push(gamma);
    }

    // Closing check: eq(γ; r) recomputed independently.
    let mut eq_star = F::one();
    for (&g, &ri) in gammas.iter().zip(&r) {
        eq_star *= g * ri + (F::one() - g) * (F::one() - ri);
    }
    if (proof.a_eval * proof.b_eval - proof.c_eval) * eq_star != claim {
        return Err(OblongError::FinalCheck);
    }

    let mut eval_point = Vec::with_capacity(n + 1);
    eval_point.push(z);
    eval_point.extend_from_slice(&gammas);
    Ok(AndCheckOutput {
        a_eval: proof.a_eval,
        b_eval: proof.b_eval,
        c_eval: proof.c_eval,
        eval_point,
    })
}

/// Explicit-challenge verifier: a thin wrapper over
/// [`verify_oblong_and_channel`] with a [`ReplayChannel`].
pub fn verify_oblong_and(
    proof: &OblongAndProof,
    r: &[F],
    z: F,
    gammas: &[F],
) -> Result<AndCheckOutput, OblongError> {
    let mut ch = ReplayChannel::from_challenges(r, z, gammas);
    verify_oblong_and_channel(&mut ch, proof, r.len())
}

#[cfg(test)]
mod tests {
    use super::super::binary_subspace::{evaluate_univariate, extrapolate_over_subspace};
    use super::*;

    /// Deterministic GF(2^128) sample from a u64 seed.
    fn sample(seed: u64) -> F {
        let hi = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ 0x1234_5678_9ABC_DEF0;
        F::from_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    /// Deterministic 32-bit word from a seed.
    fn sample32(seed: u64) -> u32 {
        (seed.wrapping_mul(0xD1B5_4A32_D192_ED03) >> 17) as u32
    }

    #[test]
    fn extend_word_matches_extrapolation() {
        // extend_word(w)[j] must equal the degree-<32 univariate (base evals
        // = bits of w) extrapolated to extension point j.
        let ntt = AdditiveNtt::new();
        let base = BinarySubspace::with_dim(SKIPPED_VARS);
        let full = BinarySubspace::with_dim(SKIPPED_VARS + 1);
        for &w in &[0u32, 1, 0xFFFF_FFFF, 0x8000_0001, sample32(7), sample32(99)] {
            let base_evals: Vec<F> = (0..WORD_BITS)
                .map(|b| if (w >> b) & 1 == 1 { F::one() } else { F::zero() })
                .collect();
            let ext = ntt.extend_word(w);
            for j in 0..WORD_BITS {
                let want = extrapolate_over_subspace(&base, &base_evals, full.get(WORD_BITS + j));
                assert_eq!(ext[j], want, "word={w:#x} ext point {j}");
            }
        }
    }

    #[test]
    fn round_message_vanishes_on_base_domain_when_and_holds() {
        // With c = a & b, R₀ must be zero on every BASE point — the property
        // that lets the prover send only the extension half.
        let n = 5usize;
        let num = 1usize << n;
        let a: Vec<u32> = (0..num).map(|i| sample32(i as u64)).collect();
        let b: Vec<u32> = (0..num).map(|i| sample32(i as u64 + 500)).collect();
        let c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        let r: Vec<F> = (0..n).map(|i| sample(i as u64 + 1)).collect();
        let eq = eq_indicator(&r);

        // R₀ at base point p = Σ_x (A(p)B(p) − C(p)) eq[x], A(p)=bit, etc.
        let base = BinarySubspace::with_dim(SKIPPED_VARS);
        for bp in 0..WORD_BITS {
            let p = base.get(bp);
            let lag_p = base_lagrange_at(p); // selects bit bp (δ at a base point)
            let mut r0 = F::zero();
            for x in 0..num {
                let av = AdditiveNtt::fold_word_at(&lag_p, a[x]);
                let bv = AdditiveNtt::fold_word_at(&lag_p, b[x]);
                let cv = AdditiveNtt::fold_word_at(&lag_p, c[x]);
                r0 += (av * bv - cv) * eq[x];
            }
            assert_eq!(r0, F::zero(), "R₀ nonzero at base point {bp}");
        }
    }

    #[test]
    fn round_message_matches_folded_sum_claim() {
        // Binius64's gate: R₀(z) recovered from the round message (base
        // zeros ++ extension evals, extrapolated at z) equals the direct
        // folded sum claim Σ_x (A(z,x)B(z,x) − C(z,x)) eq[x].
        for n in [4usize, 6, 8] {
            let num = 1usize << n;
            let a: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 3 + 1)).collect();
            let b: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 5 + 2)).collect();
            let c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let r: Vec<F> = (0..n).map(|i| sample(i as u64 * 7 + 3)).collect();
            let eq = eq_indicator(&r);

            let ntt = AdditiveNtt::new();
            let rmsg = univariate_round_message(&a, &b, &c, &eq, &ntt);

            // Verifier-side reconstruction: [0;WB] ++ rmsg over the dim-6 domain.
            let full = BinarySubspace::with_dim(SKIPPED_VARS + 1);
            let mut coeffs = vec![F::zero(); 2 * WORD_BITS];
            coeffs[WORD_BITS..].copy_from_slice(&rmsg);

            let z = sample(0xABCD_1234 ^ n as u64);
            let expected = extrapolate_over_subspace(&full, &coeffs, z);

            // Direct: fold words at z, brute-force the sum claim.
            let lag_z = base_lagrange_at(z);
            let mut actual = F::zero();
            for x in 0..num {
                let av = AdditiveNtt::fold_word_at(&lag_z, a[x]);
                let bv = AdditiveNtt::fold_word_at(&lag_z, b[x]);
                let cv = AdditiveNtt::fold_word_at(&lag_z, c[x]);
                actual = actual + (av * bv - cv) * eq[x];
            }
            assert_eq!(expected, actual, "round-message cross-check failed at n={n}");
        }
    }

    #[test]
    fn base_lagrange_at_base_point_is_delta() {
        // Folding at a base point selects exactly that bit — sanity for the
        // base-domain machinery and the fold transition.
        let base = BinarySubspace::with_dim(SKIPPED_VARS);
        for bp in 0..WORD_BITS {
            let lag = base_lagrange_at(base.get(bp));
            for (b, &l) in lag.iter().enumerate() {
                let want = if b == bp { F::one() } else { F::zero() };
                assert_eq!(l, want, "L_{b}(base[{bp}]) wrong");
            }
        }
        // …and a word folded at a base point yields that single bit.
        let w = 0b1011_0110u32;
        let lag2 = base_lagrange_at(base.get(2));
        assert_eq!(AdditiveNtt::fold_word_at(&lag2, w), F::one()); // bit 2 set
        let lag3 = base_lagrange_at(base.get(3));
        assert_eq!(AdditiveNtt::fold_word_at(&lag3, w), F::zero()); // bit 3 clear
        let _ = evaluate_univariate(&[F::zero()], F::zero()); // keep import used
    }

    /// Independent MLE evaluation of the operand `A(z, ·)` at point `gammas`:
    /// `Σ_X foldword(words, z, X) · eq(X; gammas)`. A different computation
    /// path than the prover's repeated fold, so it genuinely checks `a_eval`.
    #[allow(clippy::arithmetic_side_effects)]
    fn independent_eval(words: &[u32], z: F, gammas: &[F]) -> F {
        let lag_z = base_lagrange_at(z);
        let folded: Vec<F> = words.iter().map(|&w| AdditiveNtt::fold_word_at(&lag_z, w)).collect();
        let eqg = eq_indicator(gammas);
        folded
            .iter()
            .zip(&eqg)
            .fold(F::zero(), |acc, (&v, &e)| acc + v * e)
    }

    #[test]
    fn full_round_trip_accepts_honest() {
        // End-to-end Phase-1 + Phase-2: honest AND must verify, and the
        // closing evals must match an independent MLE evaluation.
        for n in [4usize, 6, 8] {
            let num = 1usize << n;
            let a: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 9 + 1)).collect();
            let b: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 13 + 4)).collect();
            let c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let r: Vec<F> = (0..n).map(|i| sample(i as u64 * 11 + 7)).collect();
            let z = sample(0xBEEF ^ n as u64);
            let gammas: Vec<F> = (0..n).map(|k| sample(0xCAFE + k as u64 * 131 + n as u64)).collect();

            let ntt = AdditiveNtt::new();
            let proof = prove_oblong_and(&a, &b, &c, &r, z, &gammas, &ntt);
            let out = verify_oblong_and(&proof, &r, z, &gammas).expect("honest must verify");

            assert_eq!(out.a_eval, independent_eval(&a, z, &gammas), "a_eval n={n}");
            assert_eq!(out.b_eval, independent_eval(&b, z, &gammas), "b_eval n={n}");
            assert_eq!(out.c_eval, independent_eval(&c, z, &gammas), "c_eval n={n}");
            assert_eq!(out.eval_point.len(), n + 1);
            assert_eq!(out.eval_point[0], z);
        }
    }

    #[test]
    fn corrupted_c_is_rejected() {
        // Flipping a bit of C breaks base-domain vanishing, so the verifier's
        // reconstructed R₀(z) ≠ the prover's true sum ⇒ round-0 rejection.
        let n = 6usize;
        let num = 1usize << n;
        let a: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 3 + 2)).collect();
        let b: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 7 + 5)).collect();
        let mut c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        c[5] ^= 1 << 4; // corrupt one bit so the AND fails at row 5

        let r: Vec<F> = (0..n).map(|i| sample(i as u64 + 100)).collect();
        let z = sample(0x1357);
        let gammas: Vec<F> = (0..n).map(|k| sample(0x2468 + k as u64 * 97)).collect();

        let ntt = AdditiveNtt::new();
        let proof = prove_oblong_and(&a, &b, &c, &r, z, &gammas, &ntt);
        let res = verify_oblong_and(&proof, &r, z, &gammas);
        assert!(
            matches!(res, Err(OblongError::RoundConsistency(_))),
            "corrupted C must be rejected, got {res:?}"
        );
    }

    #[test]
    fn tampered_eval_is_rejected() {
        // An honest proof with a tampered closing eval passes the round
        // consistency chain but fails the final (a·b−c)·eq check.
        let n = 5usize;
        let num = 1usize << n;
        let a: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 17 + 3)).collect();
        let b: Vec<u32> = (0..num).map(|i| sample32(i as u64 * 19 + 8)).collect();
        let c: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        let r: Vec<F> = (0..n).map(|i| sample(i as u64 + 42)).collect();
        let z = sample(0x9999);
        let gammas: Vec<F> = (0..n).map(|k| sample(0x3030 + k as u64)).collect();

        let ntt = AdditiveNtt::new();
        let mut proof = prove_oblong_and(&a, &b, &c, &r, z, &gammas, &ntt);
        verify_oblong_and(&proof, &r, z, &gammas).expect("baseline honest must verify");

        proof.a_eval = proof.a_eval + F::one(); // tamper
        assert!(matches!(
            verify_oblong_and(&proof, &r, z, &gammas),
            Err(OblongError::FinalCheck)
        ));
    }
}
