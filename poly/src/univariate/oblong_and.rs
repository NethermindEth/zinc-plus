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
//! ## Word width is a const parameter `WB`
//!
//! The whole path is generic over the word width `WB` (= bits per word =
//! `|D|`, the base/extension univariate domain size). SHA-256 instantiates
//! `WB = 32`; Keccak's 64-bit lanes instantiate `WB = 64`. Words are carried
//! as `u64` (a 32-bit word zero-extends, and only its low `WB` bits are read).
//! `WB` must be a power of two with `WB ≤ 64` (so `WB` base + `WB` extension
//! points fit the `GF(2^8)` accel domain for the byte-lookup scheme, and the
//! bit index fits a `u64`). [`WORD_BITS`]/[`SKIPPED_VARS`] remain the SHA-256
//! (`WB = 32`) defaults the `GF(2^8)` scheme and the discharge still reference.
//!
//! ## The constraint, restated
//!
//! We argue `A ⊙ B = C` (bitwise AND of `WB`-bit words) for a column of `2ⁿ`
//! words, *without* splitting into `WB` `GF(2^128)` bit-slices. The operands
//! are **oblong** multilinears `A(Z, X₀…Xₙ₋₁)`: `Z ∈ {0,1}^{log₂WB}` is the
//! bit-index within a word, `X` is the word/row index. For a fixed word `X`,
//! `A(·, X)` is the degree-`< WB` univariate whose evaluations on the
//! `WB`-point base subspace are exactly that word's bits.
//!
//! ## Phase 1 (this module)
//!
//! The prover sends the univariate
//! ```text
//!   R₀(Z) = Σ_{X ∈ {0,1}ⁿ} (A·B − C)(Z, X) · eq(X; r)
//! ```
//! over a `(log₂WB+1)`-dim subspace (`2·WB` points). Because the AND holds,
//! `A·B − C = bit·bit − bit = 0` on every **base** subspace point, so `R₀`
//! vanishes there and the prover sends only its `WB` **extension**-domain
//! evaluations. `deg R₀ ≤ 2(WB − 1)`, so the base zeros plus the extension
//! evals over-determine it.
//!
//! The verifier prepends `WB` zeros, samples `z`, extrapolates `R₀(z)`
//! (via [`super::binary_subspace::extrapolate_over_subspace`]), and proceeds
//! to a degree-2 **MLE-check** over the `n` row variables (Phase 2, below in
//! this module — Gruen's eq-factored sumcheck).

use super::binary_gf128::BinaryFieldGF128;
use super::binary_subspace::{BinarySubspace, lagrange_evals};

type F = BinaryFieldGF128;

/// `log2(WORD_BITS)` — the number of bit-index variables fused into the
/// univariate-skip round. `5` for the 32-bit SHA-256 words (the default that
/// the `GF(2^8)` scheme and the discharge still reference).
pub const SKIPPED_VARS: usize = 5;

/// Bits per word for the SHA-256 default (`WB = 32`). Generic code uses the
/// `WB` const parameter; this constant is the `GF(2^8)` scheme / discharge
/// default only.
pub const WORD_BITS: usize = 1 << SKIPPED_VARS;

/// `log2(word_bits)` — the bit-index dimension for a given word width.
#[inline]
pub const fn skipped_vars(word_bits: usize) -> usize {
    word_bits.trailing_zeros() as usize
}

/// Minimum items per rayon job for the parallel hot loops (`with_min_len`), so
/// small workloads (e.g. the nvars=9 production default, ~8K stacked words) stay
/// effectively serial and don't pay task-spawn overhead, while large ones (nvars
/// ≥ 16, ~1M words) split across cores.
pub(crate) const PAR_MIN_LEN: usize = 1 << 14;

/// Precomputed additive-NTT that extends a packed `WB`-bit word (its `WB`
/// bits = base-domain evaluations) to its `WB` extension-domain evaluations.
///
/// For a word `w`, `A(·)` is the degree-`< WB` univariate with
/// `A(base.get(b)) = bit_b(w)`. Its value at extension point `j` is
/// `A(ext_j) = Σ_b bit_b(w) · L_b(ext_j)`, where `L_b` is the Lagrange
/// basis of the base subspace. Since `bit_b ∈ {0,1}`, this is a pure
/// select-and-XOR over the precomputed rows `ext_lagrange[j][b] = L_b(ext_j)`
/// — **no multiplications**. (The fast path replaces the per-bit XOR with an
/// 8-bit byte lookup; same values, fewer ops.)
#[derive(Debug, Clone)]
pub struct AdditiveNtt<const WB: usize> {
    /// `ext_lagrange[j][b] = L_b(extension_point_j)`, `j, b ∈ 0..WB`.
    ext_lagrange: Vec<[F; WB]>,
}

impl<const WB: usize> Default for AdditiveNtt<WB> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WB: usize> AdditiveNtt<WB> {
    /// Precompute the extension-domain Lagrange rows for `WB`-bit words over
    /// the default monomial basis.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new() -> Self {
        let sv = skipped_vars(WB);
        let base = BinarySubspace::with_dim(sv); // dim log₂WB → WB base points
        let full = BinarySubspace::with_dim(sv + 1); // dim log₂WB+1 → 2·WB points
        let ext_lagrange = (0..WB)
            .map(|j| {
                // Extension point j is full.get(WB + j); for the monomial basis
                // that is the field element (WB + j).
                let ext_point = full.get(WB + j);
                let lag = lagrange_evals(&base, ext_point);
                let mut row = [F::zero(); WB];
                row.copy_from_slice(&lag);
                row
            })
            .collect();
        Self { ext_lagrange }
    }

    /// Extension-domain evaluations of the word `w` (low `WB` bits):
    /// `out[j] = A(extension_point_j)`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn extend_word(&self, w: u64) -> [F; WB] {
        let mut out = [F::zero(); WB];
        let mut bits = w & word_mask(WB);
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let row = &self.ext_lagrange[b.min(WB - 1)];
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
    pub fn fold_word_at(base_lagrange_z: &[F; WB], w: u64) -> F {
        let mut acc = F::zero();
        let mut bits = w & word_mask(WB);
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            acc += base_lagrange_z[b.min(WB - 1)];
        }
        acc
    }
}

/// Low-`wb`-bit mask as a `u64` (`wb < 64` ⇒ `(1<<wb)−1`; `wb == 64` ⇒ all
/// ones). Reads only the low `WB` bits of a `u64`-carried word.
#[inline]
pub const fn word_mask(wb: usize) -> u64 {
    if wb >= 64 { u64::MAX } else { (1u64 << wb) - 1 }
}

/// The base-subspace Lagrange weights at `z` (`L_b(z)`, `b ∈ 0..WB`), used to
/// fold words at the univariate challenge.
pub fn base_lagrange_at<const WB: usize>(z: F) -> [F; WB] {
    let base = BinarySubspace::with_dim(skipped_vars(WB));
    let lag = lagrange_evals(&base, z);
    let mut row = [F::zero(); WB];
    row.copy_from_slice(&lag);
    row
}

/// The Phase-1 oblong round message: the `WB` **extension-domain** evaluations
/// of `R₀(Z) = Σ_X (A·B − C)(Z, X) · eq(X; r)`.
///
/// `a_words`/`b_words`/`c_words` are the packed operand columns (one word per
/// row `X`, `2ⁿ` rows). `eq` is the equality indicator over the `n` row
/// variables (`eq[X] = eq(X; r)`, length `2ⁿ`). The result is `R₀` on the
/// extension domain; its base-domain values are zero when `c = a & b`.
#[allow(clippy::arithmetic_side_effects)]
pub fn univariate_round_message<const WB: usize>(
    a_words: &[u64],
    b_words: &[u64],
    c_words: &[u64],
    eq: &[F],
    ntt: &AdditiveNtt<WB>,
) -> [F; WB] {
    let n = a_words.len();
    assert_eq!(b_words.len(), n, "operand columns must have equal length");
    assert_eq!(c_words.len(), n, "operand columns must have equal length");
    assert_eq!(eq.len(), n, "eq indicator must cover every row");

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || [F::zero(); WB],
                |mut acc, x| {
                    accumulate_naive_word(&mut acc, ntt, a_words[x], b_words[x], c_words[x], eq[x]);
                    acc
                },
            )
            .reduce(
                || [F::zero(); WB],
                |mut a, b| {
                    for j in 0..WB {
                        a[j] += b[j];
                    }
                    a
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = [F::zero(); WB];
        for x in 0..n {
            accumulate_naive_word(&mut acc, ntt, a_words[x], b_words[x], c_words[x], eq[x]);
        }
        acc
    }
}

/// Accumulate one word-triple's contribution to the naive `GF(2^128)` round
/// message: NTT-extend, `(A·B − C)` per extension point, weight by `eq`.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn accumulate_naive_word<const WB: usize>(
    acc: &mut [F; WB],
    ntt: &AdditiveNtt<WB>,
    a: u64,
    b: u64,
    c: u64,
    w: F,
) {
    let ae = ntt.extend_word(a);
    let be = ntt.extend_word(b);
    let ce = ntt.extend_word(c);
    for j in 0..WB {
        acc[j] += (ae[j] * be[j] - ce[j]) * w;
    }
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
/// message (`WB` evals), the Phase-2 MLE-check round polynomials, and the
/// closing operand evals.
///
/// Each Phase-2 round poly is the **degree-2 prime polynomial** `h` of Gruen's
/// eq-factored sumcheck ([Gruen24] §3; binius64 `quadratic_mle.rs`), stored in
/// **truncated monomial form** `[c₁, c₂]`. With the equality indicator factored
/// out of the round message the round poly drops from degree 3 to degree 2, and
/// the constant `c₀` is recovered by the verifier from the MLE-check eq-relation
/// `(1−r_i)·h(0) + r_i·h(1) = claim` — so only two coefficients ship per round
/// (vs the four evals of the naive eq-folded form).
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
#[derive(Clone, Debug)]
pub struct OblongAndProof<const WB: usize> {
    pub round_message: [F; WB],
    pub round_polys: Vec<[F; 2]>,
    pub a_eval: F,
    pub b_eval: F,
    pub c_eval: F,
}

/// Why an oblong AND proof was rejected. The eq-factored MLE-check has **no
/// per-round consistency rejection** (the verifier's recovered constant always
/// satisfies the round relation), so a corrupt witness — a non-vanishing `R₀`,
/// or tampered closing evals — is caught at the closing [`FinalCheck`].
///
/// [`FinalCheck`]: OblongError::FinalCheck
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OblongError {
    #[error("closing check a·b−c ≠ final claim")]
    FinalCheck,
    #[error("malformed proof: {0} round polynomials, expected {1}")]
    Shape(usize, usize),
}

/// Pair `(2i, 2i+1)`'s contribution to the **prime polynomial** `h` of the
/// eq-factored Phase-2 round (Gruen's trick), accumulated as the three evals
/// `(h(0), h(1), h(∞))`.
///
/// With `eq` factored out of the round message, the round poly is the degree-2
/// `h(t) = Σ_rest (a(t)·b(t) − c(t)) · eq_rest[rest]`, where `a(t)` linearly
/// interpolates the low-variable pair `(a[2i], a[2i+1])` and `eq_rest` weights
/// the *remaining* row variables. `h(∞)` is the leading (t²) coefficient — the
/// Karatsuba "infinity" eval — which for `a·b − c` is `a_d·b_d` with
/// `a_d = a[2i+1] − a[2i]` (the linear-difference of `a`, etc.).
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn prime_poly_pair(acc: &mut [F; 3], a: &[F], b: &[F], c: &[F], eq_rest: &[F], i: usize) {
    let (alo, ahi) = (a[2 * i], a[2 * i + 1]);
    let (blo, bhi) = (b[2 * i], b[2 * i + 1]);
    let (clo, chi) = (c[2 * i], c[2 * i + 1]);
    let w = eq_rest[i];
    let (ad, bd) = (ahi - alo, bhi - blo);
    acc[0] += (alo * blo - clo) * w; // h(0)
    acc[1] += (ahi * bhi - chi) * w; // h(1)
    acc[2] += (ad * bd) * w; // h(∞): leading coeff of (a·b − c)
}

/// The Phase-2 prime polynomial `h` as its three evals `(h(0), h(1), h(∞))`.
/// `eq_rest` (length `a.len()/2`) is the equality indicator over the row
/// variables *not yet* bound and *not* the current one. Degree 2 in the bound
/// variable; ports binius64 `QuadraticMleCheckProver::execute`.
#[allow(clippy::arithmetic_side_effects)]
fn prime_poly(a: &[F], b: &[F], c: &[F], eq_rest: &[F]) -> [F; 3] {
    let half = a.len() / 2;
    debug_assert_eq!(eq_rest.len(), half, "eq_rest must cover every remaining-variable cube point");
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..half)
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || [F::zero(); 3],
                |mut acc, i| {
                    prime_poly_pair(&mut acc, a, b, c, eq_rest, i);
                    acc
                },
            )
            .reduce(
                || [F::zero(); 3],
                |mut x, y| {
                    for k in 0..3 {
                        x[k] += y[k];
                    }
                    x
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = [F::zero(); 3];
        for i in 0..half {
            prime_poly_pair(&mut acc, a, b, c, eq_rest, i);
        }
        acc
    }
}

/// Shrink the `eq_rest` indicator one variable by collapsing its low variable:
/// `out[i] = tbl[2i] + tbl[2i+1]`. This is the eq-factored sumcheck's cheap
/// **XOR-only** eq update (binius `eq_ind_truncate_low_inplace`): summing the
/// pair drops the lowest-variable `eq` factor, leaving `eq` over one fewer
/// variable — no challenge multiply, unlike [`fold_low`], because that factor is
/// reintroduced by the verifier rather than folded into the table.
#[allow(clippy::arithmetic_side_effects)]
fn sum_fold_low(tbl: &[F]) -> Vec<F> {
    let half = tbl.len() / 2;
    let f = |i: usize| tbl[2 * i] + tbl[2 * i + 1];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..half).into_par_iter().with_min_len(PAR_MIN_LEN).map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..half).map(f).collect()
    }
}

/// Fold a multilinear table at a challenge by binding its low variable:
/// `out[i] = tbl[2i] + γ·(tbl[2i+1] − tbl[2i])`.
#[allow(clippy::arithmetic_side_effects)]
fn fold_low(tbl: &[F], gamma: F) -> Vec<F> {
    let half = tbl.len() / 2;
    let f = |i: usize| tbl[2 * i] + gamma * (tbl[2 * i + 1] - tbl[2 * i]);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..half).into_par_iter().with_min_len(PAR_MIN_LEN).map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..half).map(f).collect()
    }
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

/// The subspace-dependent **prover** operations of the oblong zerocheck: the
/// Phase-1 round message and the fold-at-`z` Lagrange weights. Swapping the
/// scheme switches the additive-NTT basis — [`MonomialScheme`] (naive
/// `GF(2^128)`, the monomial subspace) vs the `GF(2^8)` byte-lookup scheme over
/// `embed(H₈)` ([`super::oblong_and_gf8::Gf8Scheme`]). The Phase-2 sumcheck and
/// the verifier's round-consistency checks are subspace-independent; only the
/// round message, the fold weights, and the verifier's `R₀` reconstruction
/// subspace depend on the scheme (the verifier takes that subspace directly).
///
/// Generic over the word width `WB` (the round message / fold weights are
/// `WB`-wide).
pub trait OblongScheme<const WB: usize> {
    /// Deterministic small-field skip challenges (already embedded to
    /// `GF(2^128)`); the first `len()` row-variables use these instead of
    /// channel-sampled ones, enabling the **eq-split**. Empty ⇒ no eq-split
    /// (the monomial scheme). For `Gf8Scheme` these are `{α, α², α⁴}` embedded,
    /// whose tensor product is `F_2`-independent.
    fn small_challenges(&self) -> &[F] {
        &[]
    }
    /// The Phase-1 round message `R₀` on the extension domain (`WB` evals).
    /// `big_challenges` are the **big-field** row-variable challenges (the
    /// channel-sampled ones); the scheme builds `eq` over them and, if it has
    /// small challenges, combines with its cheap small-field `eq`.
    fn round_message(&self, a: &[u64], b: &[u64], c: &[u64], big_challenges: &[F]) -> [F; WB];
    /// The base-domain Lagrange weights `L_b(z)` for folding a word at `z`.
    fn base_lagrange(&self, z: F) -> [F; WB];
}

/// Naive `GF(2^128)` scheme over the monomial subspace (borrows the additive
/// NTT so the explicit-challenge wrappers keep their `&AdditiveNtt` signature).
/// Its `R₀` reconstruction subspace is `BinarySubspace::with_dim(log₂WB+1)`.
pub struct MonomialScheme<'a, const WB: usize> {
    ntt: &'a AdditiveNtt<WB>,
}

impl<'a, const WB: usize> MonomialScheme<'a, WB> {
    pub fn new(ntt: &'a AdditiveNtt<WB>) -> Self {
        Self { ntt }
    }
}

impl<const WB: usize> OblongScheme<WB> for MonomialScheme<'_, WB> {
    fn round_message(&self, a: &[u64], b: &[u64], c: &[u64], big_challenges: &[F]) -> [F; WB] {
        // No eq-split: big_challenges = all row challenges; eq is the full table.
        let eq = eq_indicator(big_challenges);
        univariate_round_message(a, b, c, &eq, self.ntt)
    }
    fn base_lagrange(&self, z: F) -> [F; WB] {
        base_lagrange_at::<WB>(z)
    }
}

/// Prover for the oblong AND zerocheck `A ⊙ B = C` (one relation), driven by a
/// Fiat–Shamir [`OblongChannel`] and a [`OblongScheme`] (the additive-NTT basis).
/// The channel supplies the `n` zerocheck challenges `r` (first), then `z` (after
/// the round message is absorbed), then each `γ` (after its round polynomial is
/// absorbed) — so the challenges are not part of the returned proof.
///
/// Returns the proof **and** the evaluation point `[z, γ₀, …, γₙ₋₁]` (the same
/// `AndCheckOutput.eval_point` the verifier re-derives). Callers that bind the
/// operand evals to a commitment need `z`/`γ` to recombine the projected columns;
/// callers that don't (the math tests, the standalone discharge) drop the point.
///
/// `*_words` are the packed operand columns (`2ⁿ` rows, `n = log2(len)`).
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_oblong_and_channel<const WB: usize, C: OblongChannel>(
    ch: &mut C,
    a_words: &[u64],
    b_words: &[u64],
    c_words: &[u64],
    scheme: &impl OblongScheme<WB>,
) -> (OblongAndProof<WB>, Vec<F>) {
    let len = a_words.len();
    let n = len.trailing_zeros() as usize;
    assert_eq!(1usize << n, len, "columns must have a power-of-two row count");
    assert_eq!(b_words.len(), len);
    assert_eq!(c_words.len(), len);

    // The first `small.len()` row challenges are deterministic (the scheme's
    // eq-split skip challenges); the rest are channel-sampled (big field).
    let small = scheme.small_challenges();
    assert!(small.len() <= n, "more small challenges than row variables");
    let n_big = n - small.len();
    let big: Vec<F> = (0..n_big).map(|_| ch.sample()).collect();

    // Phase 1: round message over the big challenges (the scheme builds eq),
    // then the univariate-skip challenge z.
    let round_message = {
        let _g = zinc_utils::prof::scope("round_message");
        scheme.round_message(a_words, b_words, c_words, &big)
    };
    ch.absorb(&round_message);
    let z = ch.sample();

    // Gruen eq-factoring (binius `QuadraticMleCheckProver`): maintain only the
    // equality indicator over the *remaining* row variables `r[1..]`, never the
    // full `eq(·; r)` table. Each Phase-2 round the current variable's `eq`
    // factor is reintroduced by the verifier and the prefix product of bound
    // variables threads out of the claim, so the prover never folds `eq` as a
    // fourth (challenge-fold) table — it sum-folds this half-size table instead.
    let r: Vec<F> = small.iter().copied().chain(big.iter().copied()).collect();
    let mut eq_rest = eq_indicator(r.get(1..).unwrap_or(&[]));

    // Phase-1 → Phase-2 transition: fold each word at z into an MLE over rows.
    let lag_z = scheme.base_lagrange(z);
    // Fold every stacked word at `z` — embarrassingly parallel (each word folds
    // independently). This dominates the discharge (~62% at nvars=16: ~1M words
    // ×3), so run it across cores like the round-message / Phase-2 loops.
    let fold_at = |words: &[u64]| -> Vec<F> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            words
                .par_iter()
                .with_min_len(PAR_MIN_LEN)
                .map(|&w| AdditiveNtt::<WB>::fold_word_at(&lag_z, w))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            words.iter().map(|&w| AdditiveNtt::<WB>::fold_word_at(&lag_z, w)).collect()
        }
    };
    let (mut a, mut b, mut c) = {
        let _g = zinc_utils::prof::scope("fold_at_z");
        (fold_at(a_words), fold_at(b_words), fold_at(c_words))
    };

    // Phase 2: degree-2 MLE-check over the n row variables. Each round ships the
    // prime polynomial `h` truncated to its monomial `[c₁, c₂]` (`c₀` recovered
    // by the verifier).
    let mut round_polys = Vec::with_capacity(n);
    let mut gammas = Vec::with_capacity(n);
    {
        let _g = zinc_utils::prof::scope("phase2_mlecheck");
        for i in 0..n {
            let [h0, h1, hinf] = prime_poly(&a, &b, &c, &eq_rest);
            let trunc = [h1 - h0 - hinf, hinf]; // [c₁, c₂]; c₀ = h(0) recovered by verifier
            ch.absorb(&trunc);
            let gamma = ch.sample();
            round_polys.push(trunc);
            gammas.push(gamma);
            a = fold_low(&a, gamma);
            b = fold_low(&b, gamma);
            c = fold_low(&c, gamma);
            if i + 1 < n {
                eq_rest = sum_fold_low(&eq_rest);
            }
        }
    }

    // The evaluation point `[z, γ…]`, identical to what the verifier re-derives
    // (`verify_oblong_and_channel`'s `AndCheckOutput.eval_point`). Exposed so the
    // prover can recombine the projected operand columns at `γ` without re-running
    // the transcript.
    let mut eval_point = Vec::with_capacity(n + 1);
    eval_point.push(z);
    eval_point.extend_from_slice(&gammas);

    (
        OblongAndProof {
            round_message,
            round_polys,
            a_eval: a[0],
            b_eval: b[0],
            c_eval: c[0],
        },
        eval_point,
    )
}

/// Explicit-challenge prover: a thin wrapper over [`prove_oblong_and_channel`]
/// with a [`ReplayChannel`]. `r`/`z`/`gammas` are the `n`+1+`n` challenges in
/// sampling order. Used by the math tests; the `protocol` crate uses the
/// channel form with a real transcript.
pub fn prove_oblong_and<const WB: usize>(
    a_words: &[u64],
    b_words: &[u64],
    c_words: &[u64],
    r: &[F],
    z: F,
    gammas: &[F],
    ntt: &AdditiveNtt<WB>,
) -> OblongAndProof<WB> {
    let mut ch = ReplayChannel::from_challenges(r, z, gammas);
    let scheme = MonomialScheme::new(ntt);
    // Explicit-challenge path (math tests): the caller already has `z`/`γ`, so the
    // returned eval-point is redundant — drop it.
    prove_oblong_and_channel(&mut ch, a_words, b_words, c_words, &scheme).0
}

/// Verifier for the oblong AND zerocheck, driven by a Fiat–Shamir
/// [`OblongChannel`]. Samples `r`, reconstructs `R₀(z)` from the round message
/// (base zeros ++ extension evals) as the initial MLE-check claim, runs the
/// eq-factored Phase-2 MLE-check (absorbing each truncated round polynomial
/// before sampling its `γ`, mirroring the prover; recovering each `c₀` from the
/// eq-relation and threading the claim by `h(γ)`), and verifies the closing
/// `a·b − c = claim`. `n` is the number of row variables.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_oblong_and_channel<const WB: usize, C: OblongChannel>(
    ch: &mut C,
    proof: &OblongAndProof<WB>,
    n: usize,
    full_subspace: &BinarySubspace,
    small_challenges: &[F],
) -> Result<AndCheckOutput, OblongError> {
    if proof.round_polys.len() != n {
        return Err(OblongError::Shape(proof.round_polys.len(), n));
    }

    // Mirror the prover: deterministic small-field prefix, then sampled big.
    let n_big = n.saturating_sub(small_challenges.len());
    let big: Vec<F> = (0..n_big).map(|_| ch.sample()).collect();
    let r: Vec<F> = small_challenges.iter().copied().chain(big).collect();

    // Reconstruct R₀(z) → the initial MLE-check claim: the prover sends only the
    // extension half; the base half is zero iff the AND holds, so a corrupt C
    // makes this claim disagree with the folded tables — which surfaces at the
    // closing check once the rounds have threaded it through. `full_subspace` is
    // the dim-(log₂WB+1) univariate domain — monomial or `embed(H₈)`, matching
    // the prover's scheme.
    ch.absorb(&proof.round_message);
    let z = ch.sample();
    let mut coeffs = vec![F::zero(); 2 * WB];
    coeffs[WB..].copy_from_slice(&proof.round_message);
    let mut claim = super::binary_subspace::extrapolate_over_subspace(full_subspace, &coeffs, z);

    // Phase-2 MLE-check (Gruen). Each round the prover sent the degree-2 prime
    // polynomial `h` as a truncated monomial `[c₁, c₂]`. Recover the dropped
    // constant from the eq-relation `claim = (1−α)·h(0) + α·h(1)` (α = r_i, so
    // `c₀ = claim − α·(c₁ + c₂)`), then thread the claim by `h(γ_i)`. There is no
    // per-round consistency rejection — the recovered `c₀` always satisfies the
    // relation — so a corrupt witness surfaces only at the closing check below
    // (mirrors binius64 `mlecheck::verify`).
    let mut gammas = Vec::with_capacity(n);
    for (i, trunc) in proof.round_polys.iter().enumerate() {
        let [c1, c2] = *trunc;
        ch.absorb(trunc);
        let gamma = ch.sample();
        let c0 = claim - r[i] * (c1 + c2);
        claim = c0 + (c1 + c2 * gamma) * gamma; // h(γ_i), Horner
        gammas.push(gamma);
    }

    // Closing check: the per-variable `eq` factors have been threaded out of the
    // claim round by round, so the final claim is exactly the composition
    // `a·b − c` at the challenge point (no separate `eq(γ; r)` factor).
    if proof.a_eval * proof.b_eval - proof.c_eval != claim {
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
pub fn verify_oblong_and<const WB: usize>(
    proof: &OblongAndProof<WB>,
    r: &[F],
    z: F,
    gammas: &[F],
) -> Result<AndCheckOutput, OblongError> {
    let mut ch = ReplayChannel::from_challenges(r, z, gammas);
    let full = BinarySubspace::with_dim(skipped_vars(WB) + 1);
    verify_oblong_and_channel(&mut ch, proof, r.len(), &full, &[])
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

    /// Deterministic 32-bit word from a seed (carried as u64).
    fn sample32(seed: u64) -> u64 {
        ((seed.wrapping_mul(0xD1B5_4A32_D192_ED03) >> 17) as u32) as u64
    }

    /// Deterministic 64-bit word from a seed.
    fn sample64(seed: u64) -> u64 {
        seed.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ (seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 7)
    }

    #[test]
    fn extend_word_matches_extrapolation() {
        // extend_word(w)[j] must equal the degree-<32 univariate (base evals
        // = bits of w) extrapolated to extension point j.
        let ntt = AdditiveNtt::<WORD_BITS>::new();
        let base = BinarySubspace::with_dim(SKIPPED_VARS);
        let full = BinarySubspace::with_dim(SKIPPED_VARS + 1);
        for &w in &[0u64, 1, 0xFFFF_FFFF, 0x8000_0001, sample32(7), sample32(99)] {
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
        let a: Vec<u64> = (0..num).map(|i| sample32(i as u64)).collect();
        let b: Vec<u64> = (0..num).map(|i| sample32(i as u64 + 500)).collect();
        let c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        let r: Vec<F> = (0..n).map(|i| sample(i as u64 + 1)).collect();
        let eq = eq_indicator(&r);

        // R₀ at base point p = Σ_x (A(p)B(p) − C(p)) eq[x], A(p)=bit, etc.
        let base = BinarySubspace::with_dim(SKIPPED_VARS);
        for bp in 0..WORD_BITS {
            let p = base.get(bp);
            let lag_p = base_lagrange_at::<WORD_BITS>(p); // selects bit bp (δ at a base point)
            let mut r0 = F::zero();
            for x in 0..num {
                let av = AdditiveNtt::<WORD_BITS>::fold_word_at(&lag_p, a[x]);
                let bv = AdditiveNtt::<WORD_BITS>::fold_word_at(&lag_p, b[x]);
                let cv = AdditiveNtt::<WORD_BITS>::fold_word_at(&lag_p, c[x]);
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
            let a: Vec<u64> = (0..num).map(|i| sample32(i as u64 * 3 + 1)).collect();
            let b: Vec<u64> = (0..num).map(|i| sample32(i as u64 * 5 + 2)).collect();
            let c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let r: Vec<F> = (0..n).map(|i| sample(i as u64 * 7 + 3)).collect();
            let eq = eq_indicator(&r);

            let ntt = AdditiveNtt::<WORD_BITS>::new();
            let rmsg = univariate_round_message(&a, &b, &c, &eq, &ntt);

            // Verifier-side reconstruction: [0;WB] ++ rmsg over the dim-6 domain.
            let full = BinarySubspace::with_dim(SKIPPED_VARS + 1);
            let mut coeffs = vec![F::zero(); 2 * WORD_BITS];
            coeffs[WORD_BITS..].copy_from_slice(&rmsg);

            let z = sample(0xABCD_1234 ^ n as u64);
            let expected = extrapolate_over_subspace(&full, &coeffs, z);

            // Direct: fold words at z, brute-force the sum claim.
            let lag_z = base_lagrange_at::<WORD_BITS>(z);
            let mut actual = F::zero();
            for x in 0..num {
                let av = AdditiveNtt::<WORD_BITS>::fold_word_at(&lag_z, a[x]);
                let bv = AdditiveNtt::<WORD_BITS>::fold_word_at(&lag_z, b[x]);
                let cv = AdditiveNtt::<WORD_BITS>::fold_word_at(&lag_z, c[x]);
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
            let lag = base_lagrange_at::<WORD_BITS>(base.get(bp));
            for (b, &l) in lag.iter().enumerate() {
                let want = if b == bp { F::one() } else { F::zero() };
                assert_eq!(l, want, "L_{b}(base[{bp}]) wrong");
            }
        }
        // …and a word folded at a base point yields that single bit.
        let w = 0b1011_0110u64;
        let lag2 = base_lagrange_at::<WORD_BITS>(base.get(2));
        assert_eq!(AdditiveNtt::<WORD_BITS>::fold_word_at(&lag2, w), F::one()); // bit 2 set
        let lag3 = base_lagrange_at::<WORD_BITS>(base.get(3));
        assert_eq!(AdditiveNtt::<WORD_BITS>::fold_word_at(&lag3, w), F::zero()); // bit 3 clear
        let _ = evaluate_univariate(&[F::zero()], F::zero()); // keep import used
    }

    /// Independent MLE evaluation of the operand `A(z, ·)` at point `gammas`:
    /// `Σ_X foldword(words, z, X) · eq(X; gammas)`. A different computation
    /// path than the prover's repeated fold, so it genuinely checks `a_eval`.
    #[allow(clippy::arithmetic_side_effects)]
    fn independent_eval<const WB: usize>(words: &[u64], z: F, gammas: &[F]) -> F {
        let lag_z = base_lagrange_at::<WB>(z);
        let folded: Vec<F> = words.iter().map(|&w| AdditiveNtt::<WB>::fold_word_at(&lag_z, w)).collect();
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
            let a: Vec<u64> = (0..num).map(|i| sample32(i as u64 * 9 + 1)).collect();
            let b: Vec<u64> = (0..num).map(|i| sample32(i as u64 * 13 + 4)).collect();
            let c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let r: Vec<F> = (0..n).map(|i| sample(i as u64 * 11 + 7)).collect();
            let z = sample(0xBEEF ^ n as u64);
            let gammas: Vec<F> = (0..n).map(|k| sample(0xCAFE + k as u64 * 131 + n as u64)).collect();

            let ntt = AdditiveNtt::<WORD_BITS>::new();
            let proof = prove_oblong_and(&a, &b, &c, &r, z, &gammas, &ntt);
            let out = verify_oblong_and(&proof, &r, z, &gammas).expect("honest must verify");

            assert_eq!(out.a_eval, independent_eval::<WORD_BITS>(&a, z, &gammas), "a_eval n={n}");
            assert_eq!(out.b_eval, independent_eval::<WORD_BITS>(&b, z, &gammas), "b_eval n={n}");
            assert_eq!(out.c_eval, independent_eval::<WORD_BITS>(&c, z, &gammas), "c_eval n={n}");
            assert_eq!(out.eval_point.len(), n + 1);
            assert_eq!(out.eval_point[0], z);
        }
    }

    #[test]
    fn full_round_trip_accepts_honest_64bit() {
        // The Keccak word width: WB = 64 (lanes). Honest AND over 64-bit words
        // must verify, and the closing evals must match an independent eval.
        const WB: usize = 64;
        for n in [4usize, 6] {
            let num = 1usize << n;
            let a: Vec<u64> = (0..num).map(|i| sample64(i as u64 * 9 + 1)).collect();
            let b: Vec<u64> = (0..num).map(|i| sample64(i as u64 * 13 + 4)).collect();
            let c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let r: Vec<F> = (0..n).map(|i| sample(i as u64 * 11 + 7)).collect();
            let z = sample(0xBEEF ^ n as u64);
            let gammas: Vec<F> = (0..n).map(|k| sample(0xCAFE + k as u64 * 131 + n as u64)).collect();

            let ntt = AdditiveNtt::<WB>::new();
            let proof = prove_oblong_and(&a, &b, &c, &r, z, &gammas, &ntt);
            let out = verify_oblong_and(&proof, &r, z, &gammas).expect("honest 64-bit must verify");

            assert_eq!(out.a_eval, independent_eval::<WB>(&a, z, &gammas), "a_eval n={n}");
            assert_eq!(out.b_eval, independent_eval::<WB>(&b, z, &gammas), "b_eval n={n}");
            assert_eq!(out.c_eval, independent_eval::<WB>(&c, z, &gammas), "c_eval n={n}");
        }
    }

    #[test]
    fn corrupted_c_is_rejected() {
        // Flipping a bit of C breaks base-domain vanishing, so the verifier's
        // reconstructed R₀(z) ≠ the true folded sum. The eq-factored MLE-check
        // has no per-round rejection (the recovered constant always satisfies
        // the round relation), so the bad claim threads through every round and
        // is caught at the closing check.
        let n = 6usize;
        let num = 1usize << n;
        let a: Vec<u64> = (0..num).map(|i| sample32(i as u64 * 3 + 2)).collect();
        let b: Vec<u64> = (0..num).map(|i| sample32(i as u64 * 7 + 5)).collect();
        let mut c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        c[5] ^= 1 << 4; // corrupt one bit so the AND fails at row 5

        let r: Vec<F> = (0..n).map(|i| sample(i as u64 + 100)).collect();
        let z = sample(0x1357);
        let gammas: Vec<F> = (0..n).map(|k| sample(0x2468 + k as u64 * 97)).collect();

        let ntt = AdditiveNtt::<WORD_BITS>::new();
        let proof = prove_oblong_and(&a, &b, &c, &r, z, &gammas, &ntt);
        let res = verify_oblong_and(&proof, &r, z, &gammas);
        assert!(
            matches!(res, Err(OblongError::FinalCheck)),
            "corrupted C must be rejected, got {res:?}"
        );
    }

    #[test]
    fn tampered_eval_is_rejected() {
        // An honest proof with a tampered closing eval passes the round
        // consistency chain but fails the final (a·b−c)·eq check.
        let n = 5usize;
        let num = 1usize << n;
        let a: Vec<u64> = (0..num).map(|i| sample32(i as u64 * 17 + 3)).collect();
        let b: Vec<u64> = (0..num).map(|i| sample32(i as u64 * 19 + 8)).collect();
        let c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        let r: Vec<F> = (0..n).map(|i| sample(i as u64 + 42)).collect();
        let z = sample(0x9999);
        let gammas: Vec<F> = (0..n).map(|k| sample(0x3030 + k as u64)).collect();

        let ntt = AdditiveNtt::<WORD_BITS>::new();
        let mut proof = prove_oblong_and(&a, &b, &c, &r, z, &gammas, &ntt);
        verify_oblong_and(&proof, &r, z, &gammas).expect("baseline honest must verify");

        proof.a_eval = proof.a_eval + F::one(); // tamper
        assert!(matches!(
            verify_oblong_and(&proof, &r, z, &gammas),
            Err(OblongError::FinalCheck)
        ));
    }
}
