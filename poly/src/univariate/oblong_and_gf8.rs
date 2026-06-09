//! `GF(2^8)`-accelerated additive-NTT + oblong round message — the prover
//! **speed lever** for the oblong AND zerocheck (port plan
//! `documentation/f2-hadamard-oblong-port-plan.md`, prerequisite P4).
//!
//! The naive path ([`super::oblong_and`]) runs the additive-NTT word extension
//! and the per-extension-point products in `GF(2^128)` (CLMUL + 16-byte XORs).
//! Here we run them in [`Gf8`] (`GF(2^8)`: log/antilog mult, 1-byte XORs) via a
//! precomputed **byte-lookup table** (binius `ntt_lookup.rs`), and lift to
//! `GF(2^128)` only for the final `eq`-weighted accumulation.
//!
//! ## Generic over the word width `WB`
//!
//! Everything here is const-generic over the lane width `WB` (`= 32` for SHA-256,
//! `= 64` for Keccak). The NTT runs over the `GF(2^8)` subspace
//! `H = {Gf8(0), …, Gf8(WB−1)}` (the `Gf8` monomial basis `{1,α,…,α^{log₂WB−1}}`)
//! and reconstructs over the full domain `{Gf8(0), …, Gf8(2·WB−1)}`. The only
//! constraint is `2·WB ≤ 256` (the full domain must fit `GF(2^8)`): `WB ≤ 128`,
//! so `WB ∈ {32, 64}` both work (`2·64 = 128 < 256`). `WB = 32` is numerically
//! identical to the original fixed-width implementation.
//!
//! ## Why this is sound: the embed subspace
//!
//! `H`'s image `embed(H) ⊂ GF(2^128)` is the `(log₂WB)`-dim subspace the
//! **verifier** extrapolates over. Since [`embed`](super::binary_gf8::embed) is a
//! field homomorphism, every `Gf8` NTT value / product equals (after `embed`)
//! the corresponding `GF(2^128)` computation over `embed(H)` — so the `Gf8` path
//! produces exactly the round message a `GF(2^128)` prover would over that
//! subspace, only faster. This is a **different subspace** than the naive path's
//! monomial `{X⁰…X^{log₂WB−1}}`; the two are not meant to produce equal round
//! messages, only each internally consistent with its own verifier.
//!
//! Current status: the NTT + products run in `GF(2^8)`; the `eq`-weighting
//! still runs per-word in `GF(2^128)`. Binius's further optimisation — split
//! `eq` into ≤3 deterministic small-field vars (weighted in `GF(2^8)`,
//! accumulated per chunk) + the big-field remainder (weighted once per chunk) —
//! is applied in [`gf8_round_message_split`].

use super::binary_gf128::BinaryFieldGF128;
use super::binary_gf8::{Gf8, embed, gf8_mul_embed_tables};
use super::binary_subspace::{BinarySubspace, lagrange_evals};
use super::oblong_and::{OblongScheme, PAR_MIN_LEN, eq_indicator, skipped_vars};

type F128 = BinaryFieldGF128;

/// Lagrange basis evaluations `L_b(z)` over the `GF(2^8)` base subspace
/// `{Gf8(0), …, Gf8(WB−1)}` (an `F_2`-subgroup ⇒ single barycentric weight).
/// The `Gf8` analogue of [`super::binary_subspace::lagrange_evals`], generic
/// over the lane width `WB`.
#[allow(clippy::arithmetic_side_effects)]
fn gf8_base_lagrange<const WB: usize>(z: Gf8) -> [Gf8; WB] {
    let domain: [Gf8; WB] = std::array::from_fn(|i| Gf8(i as u8));

    // w = (∏_{j=1}^{WB−1} domain[j])^{-1}.
    let prod = domain[1..].iter().fold(Gf8::ONE, |acc, &d| acc.mul(d));
    let w = prod.inverse();

    let mut prefixes = [Gf8::ONE; WB];
    for i in 1..WB {
        prefixes[i] = prefixes[i - 1].mul(z - domain[i - 1]);
    }
    let mut suffixes = [Gf8::ONE; WB];
    for i in (0..WB - 1).rev() {
        suffixes[i] = suffixes[i + 1].mul(z - domain[i + 1]);
    }
    std::array::from_fn(|i| prefixes[i].mul(suffixes[i]).mul(w))
}

/// Precomputed byte-lookup additive NTT over `GF(2^8)`, generic over `WB`.
///
/// `lut[byte_idx][byte_val][j]` is the contribution to extension-point `j` of
/// the 8 bits of byte `byte_idx` taking value `byte_val`, i.e.
/// `Σ_{bit set in byte_val} L_{byte_idx·8 + bit}(ext_j)`. Then
/// `extend_word(w)[j] = Σ_{byte_idx} lut[byte_idx][byte(w, byte_idx)][j]` —
/// `WB/8` lookups + adds, vs the naive `popcount(w)` Lagrange-row XORs.
#[derive(Clone)]
pub struct Gf8Ntt<const WB: usize> {
    lut: Vec<[[Gf8; WB]; 256]>, // length WB/8
}

impl<const WB: usize> Default for Gf8Ntt<WB> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WB: usize> Gf8Ntt<WB> {
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new() -> Self {
        // The full domain {Gf8(0..2·WB−1)} must fit GF(2^8) (256 elements), and
        // WB must be a power of two (so {Gf8(0..WB−1)} is an F_2-subgroup).
        assert!(WB.is_power_of_two(), "WB must be a power of two");
        assert!(2 * WB <= 256, "the 2·WB-point domain must fit GF(2^8)");
        let word_bytes = WB / 8;

        // ext_lagrange_by_point[j][b] = L_b(ext_j), ext_j = Gf8(WB + j).
        let ext_lagrange_by_point: Vec<[Gf8; WB]> = (0..WB)
            .map(|j| gf8_base_lagrange::<WB>(Gf8((WB + j) as u8)))
            .collect();

        let mut lut = vec![[[Gf8::ZERO; WB]; 256]; word_bytes];
        for (byte_idx, table) in lut.iter_mut().enumerate() {
            for (val, row) in table.iter_mut().enumerate() {
                for (j, slot) in row.iter_mut().enumerate() {
                    let mut acc = Gf8::ZERO;
                    for bit in 0..8 {
                        if (val >> bit) & 1 == 1 {
                            acc = acc + ext_lagrange_by_point[j][byte_idx * 8 + bit];
                        }
                    }
                    *slot = acc;
                }
            }
        }
        Self { lut }
    }

    /// Extension-domain evaluations of the word `w` in `GF(2^8)`:
    /// `out[j] = A(ext_j)` over the `Gf8` subspace. Words are carried as `u64`;
    /// the low `WB/8` bytes (the `WB` bits) are read.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn extend_word(&self, w: u64) -> [Gf8; WB] {
        let bytes = w.to_le_bytes();
        let mut out = [Gf8::ZERO; WB];
        for (byte_idx, table) in self.lut.iter().enumerate() {
            let row = &table[bytes[byte_idx] as usize];
            for (o, &r) in out.iter_mut().zip(row.iter()) {
                *o = *o + r;
            }
        }
        out
    }
}

/// Accumulate one word-triple's contribution into the round-message
/// accumulator: NTT-extend in `GF(2^8)`, `(A·B − C)` per extension point in
/// `GF(2^8)`, then `embed` + weight by the `GF(2^128)` eq value.
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn accumulate_gf8_word<const WB: usize>(
    acc: &mut [F128; WB],
    ntt: &Gf8Ntt<WB>,
    a: u64,
    b: u64,
    c: u64,
    w: F128,
    mt: &[u8; 65536],
    et: &[F128; 256],
) {
    let ae = ntt.extend_word(a);
    let be = ntt.extend_word(b);
    let ce = ntt.extend_word(c);
    for j in 0..WB {
        // ae·be in GF(2^8) (table lookup), − ce (XOR), embed (table), · w (GF128).
        let prod = mt[((ae[j].0 as usize) << 8) | be[j].0 as usize];
        acc[j] += et[(prod ^ ce[j].0) as usize] * w;
    }
}

/// The Phase-1 oblong round message computed with the `GF(2^8)` NTT: the `WB`
/// extension-domain evaluations of `R₀(Z)=Σ_X (A·B−C)(Z,X)·eq(X;r)`, returned in
/// `GF(2^128)` (the NTT + product run in `GF(2^8)`, then `embed`). Output matches
/// [`super::oblong_and::univariate_round_message`] run over the `embed(H)`
/// subspace (not the monomial one). `eq` is the `GF(2^128)` row-var equality
/// indicator (length `2ⁿ`).
#[allow(clippy::arithmetic_side_effects)]
pub fn gf8_round_message<const WB: usize>(
    a_words: &[u64],
    b_words: &[u64],
    c_words: &[u64],
    eq: &[F128],
    ntt: &Gf8Ntt<WB>,
) -> [F128; WB] {
    let n = a_words.len();
    assert_eq!(b_words.len(), n);
    assert_eq!(c_words.len(), n);
    assert_eq!(eq.len(), n);

    // Fetch the GF(2^8) product + embed tables once (not per multiply).
    let (mt, et) = gf8_mul_embed_tables();

    // Parallel map-reduce over the words (each contributes to the WB-element
    // accumulator). The fused discharge this replaces is parallel too.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || [F128::zero(); WB],
                |mut acc, x| {
                    accumulate_gf8_word::<WB>(&mut acc, ntt, a_words[x], b_words[x], c_words[x], eq[x], mt, et);
                    acc
                },
            )
            .reduce(
                || [F128::zero(); WB],
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
        let mut acc = [F128::zero(); WB];
        for x in 0..n {
            accumulate_gf8_word::<WB>(&mut acc, ntt, a_words[x], b_words[x], c_words[x], eq[x], mt, et);
        }
        acc
    }
}

/// `GF(2^8)` equality indicator `eq[s] = ∏_i (bit_i(s)? r_i : 1−r_i)` over the
/// small-field skip challenges (little-endian, matching [`eq_indicator`]).
#[allow(clippy::arithmetic_side_effects)]
fn eq_indicator_gf8(r: &[Gf8]) -> Vec<Gf8> {
    let mut ev = vec![Gf8::ONE];
    for &ri in r {
        let half = ev.len();
        let mut next = vec![Gf8::ZERO; half * 2];
        for j in 0..half {
            next[j] = ev[j].mul(Gf8::ONE - ri); // bit = 0 (low half)
            next[j + half] = ev[j].mul(ri); // bit = 1 (high half)
        }
        ev = next;
    }
    ev
}

/// Accumulate one chunk of `eq_small.len()` words into the eq-split round
/// message: weight each word's `GF(2^8)` product by the small-field eq in
/// `GF(2^8)`, sum into `chunk_ntt`, then `embed` + weight by the big-field eq
/// **once** for the whole chunk (the eq-split's GF(2^128)-mul saving).
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn process_gf8_chunk<const WB: usize>(
    acc: &mut [F128; WB],
    chunk: usize,
    a_words: &[u64],
    b_words: &[u64],
    c_words: &[u64],
    eq_big: &[F128],
    eq_small: &[u8],
    ntt: &Gf8Ntt<WB>,
    mt: &[u8; 65536],
    et: &[F128; 256],
) {
    let chunk_size = eq_small.len();
    let mut chunk_ntt = [0u8; WB]; // GF(2^8) bytes, accumulated by XOR
    for s in 0..chunk_size {
        let x = chunk * chunk_size + s;
        let ae = ntt.extend_word(a_words[x]);
        let be = ntt.extend_word(b_words[x]);
        let ce = ntt.extend_word(c_words[x]);
        let ws = eq_small[s] as usize;
        for j in 0..WB {
            let prod = mt[((ae[j].0 as usize) << 8) | be[j].0 as usize]; // ae·be
            let pmc = (prod ^ ce[j].0) as usize; // − ce
            chunk_ntt[j] ^= mt[(pmc << 8) | ws]; // ·eq_small, GF(2^8) add
        }
    }
    // embed + big-field eq weight, once per chunk (the eq-split's GF128 saving).
    let big = eq_big[chunk];
    for j in 0..WB {
        acc[j] += et[chunk_ntt[j] as usize] * big;
    }
}

/// The **eq-split** GF(2^8) round message: the row variables split into the
/// small-field skip vars (`eq_small`, weighted in GF(2^8) per word) and the
/// big-field vars (`eq_big`, weighted in GF(2^128) once per `eq_small.len()`-word
/// chunk). Cuts the GF(2^128) eq-mults by the chunk size vs [`gf8_round_message`].
#[allow(clippy::arithmetic_side_effects)]
pub fn gf8_round_message_split<const WB: usize>(
    a_words: &[u64],
    b_words: &[u64],
    c_words: &[u64],
    eq_big: &[F128],
    ntt: &Gf8Ntt<WB>,
    eq_small: &[Gf8],
) -> [F128; WB] {
    let chunk_size = eq_small.len();
    let num_chunks = a_words.len() / chunk_size;
    assert_eq!(eq_big.len(), num_chunks, "eq_big must cover every chunk");

    let (mt, et) = gf8_mul_embed_tables();
    let eq_small: Vec<u8> = eq_small.iter().map(|g| g.0).collect();
    let eq_small = eq_small.as_slice();

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..num_chunks)
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN / chunk_size + 1)
            .fold(
                || [F128::zero(); WB],
                |mut acc, chunk| {
                    process_gf8_chunk::<WB>(&mut acc, chunk, a_words, b_words, c_words, eq_big, eq_small, ntt, mt, et);
                    acc
                },
            )
            .reduce(
                || [F128::zero(); WB],
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
        let mut acc = [F128::zero(); WB];
        for chunk in 0..num_chunks {
            process_gf8_chunk::<WB>(&mut acc, chunk, a_words, b_words, c_words, eq_big, eq_small, ntt, mt, et);
        }
        acc
    }
}

/// The `embed(H)` subspaces the verifier must use with the `GF(2^8)` path:
/// `(base, full)` of dims `log₂WB` and `log₂WB+1`, as `GF(2^128)`
/// `BinarySubspace`s. (The naive path uses the monomial subspace instead.)
pub fn embed_subspaces<const WB: usize>() -> (BinarySubspace, BinarySubspace) {
    // embed is F_2-linear, so embed(H) is spanned by embed of the Gf8 monomial
    // basis {α⁰…α^{dim-1}} = bytes {1,2,4,…}. The basis (not the points!) goes to
    // new_unchecked: dim-many elements, with get(i) = embed(Gf8(i)).
    let sv = skipped_vars(WB);
    let base_basis: Vec<F128> = (0..sv).map(|i| embed(Gf8(1u8 << i))).collect();
    let full_basis: Vec<F128> = (0..=sv).map(|i| embed(Gf8(1u8 << i))).collect();
    (
        BinarySubspace::new_unchecked(base_basis),
        BinarySubspace::new_unchecked(full_basis),
    )
}

/// The `GF(2^8)` [`OblongScheme`]: the byte-lookup additive NTT over the
/// `embed(H)` subspace, generic over the lane width `WB`. Drop-in for
/// [`super::oblong_and::MonomialScheme`] in `prove_oblong_and_channel`; the
/// verifier must pair it with [`Gf8Scheme::full_subspace`]. Holds the
/// precomputed NTT table + the `embed(H)` base/full subspaces.
pub struct Gf8Scheme<const WB: usize> {
    ntt: Gf8Ntt<WB>,
    base: BinarySubspace,
    full: BinarySubspace,
    /// `{α, α², α⁴}` embedded to `GF(2^128)` — the deterministic skip challenges
    /// for the first 3 row-variables (their tensor product is `F_2`-independent).
    small_embedded: Vec<F128>,
    /// `eq` over `{α, α², α⁴}` in `GF(2^8)` (8 weights) for the eq-split.
    eq_small_gf8: Vec<Gf8>,
}

impl<const WB: usize> Default for Gf8Scheme<WB> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WB: usize> Gf8Scheme<WB> {
    pub fn new() -> Self {
        let (base, full) = embed_subspaces::<WB>();
        let g = Gf8::generator();
        let small_gf8 = [g, g.pow(2), g.pow(4)];
        let small_embedded = small_gf8.iter().map(|&c| embed(c)).collect();
        let eq_small_gf8 = eq_indicator_gf8(&small_gf8);
        Self {
            ntt: Gf8Ntt::<WB>::new(),
            base,
            full,
            small_embedded,
            eq_small_gf8,
        }
    }

    /// The full univariate domain (`embed(H)`, dim `log₂WB+1`) the verifier
    /// extrapolates `R₀` over with this scheme.
    pub fn full_subspace(&self) -> &BinarySubspace {
        &self.full
    }
}

impl<const WB: usize> OblongScheme<WB> for Gf8Scheme<WB> {
    fn small_challenges(&self) -> &[F128] {
        &self.small_embedded
    }

    fn round_message(&self, a: &[u64], b: &[u64], c: &[u64], big_challenges: &[F128]) -> [F128; WB] {
        // eq-split: eq_big (GF128) over the big vars; eq_small (GF8) baked in.
        let eq_big = eq_indicator(big_challenges);
        gf8_round_message_split::<WB>(a, b, c, &eq_big, &self.ntt, &self.eq_small_gf8)
    }

    fn base_lagrange(&self, z: F128) -> [F128; WB] {
        // L_b(z) over the embed(H) base subspace — the fold weights matching
        // the GF(2^8) round message's basis.
        let lag = lagrange_evals(&self.base, z);
        let mut out = [F128::zero(); WB];
        out.copy_from_slice(&lag);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::binary_subspace::{BinarySubspace, extrapolate_over_subspace};
    use super::*;

    fn sample_word(seed: u64, wb: usize) -> u64 {
        let raw = seed.wrapping_mul(0xD1B5_4A32_D192_ED03).rotate_left(17)
            ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        if wb >= 64 { raw } else { raw & ((1u64 << wb) - 1) }
    }
    fn sample128(seed: u64) -> F128 {
        let hi = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ 0x1234_5678_9ABC_DEF0;
        F128::from_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    /// GF(2^128) reference: extend a word over the embed(H) base subspace by
    /// interpolating its bits and extrapolating to embed(ext_j).
    fn gf128_extend_over_embed<const WB: usize>(
        w: u64,
        base: &BinarySubspace,
        ext_points: &[F128],
    ) -> Vec<F128> {
        let bits: Vec<F128> = (0..WB)
            .map(|b| if (w >> b) & 1 == 1 { F128::one() } else { F128::zero() })
            .collect();
        ext_points
            .iter()
            .map(|&p| extrapolate_over_subspace(base, &bits, p))
            .collect()
    }

    /// The core correctness gate: embed(Gf8 NTT) == GF128 NTT over embed(H).
    fn check_extend_embeds<const WB: usize>() {
        let ntt = Gf8Ntt::<WB>::new();
        let (base, _full) = embed_subspaces::<WB>();
        let ext_points: Vec<F128> = (0..WB).map(|j| embed(Gf8((WB + j) as u8))).collect();

        let samples: Vec<u64> = [0u64, 1, 7, 123, 9999, 0x8000_0001, 0xFFFF_FFFF]
            .iter()
            .map(|&s| sample_word(s, WB))
            .chain([0u64, u64::from(u32::MAX), sample_word(424242, WB)])
            .collect();
        for &w in &samples {
            let g = ntt.extend_word(w);
            let reference = gf128_extend_over_embed::<WB>(w, &base, &ext_points);
            for j in 0..WB {
                assert_eq!(embed(g[j]), reference[j], "WB={WB} word={w:#x} ext {j}");
            }
        }
    }

    #[test]
    fn gf8_extend_embeds_to_gf128_extend_d32() {
        check_extend_embeds::<32>();
    }
    #[test]
    fn gf8_extend_embeds_to_gf128_extend_d64() {
        check_extend_embeds::<64>();
    }

    /// The full round message via GF(2^8) must equal the GF(2^128) computation
    /// over the same embed(H) subspace.
    fn check_round_message_matches<const WB: usize>() {
        let (base, _full) = embed_subspaces::<WB>();
        let ext_points: Vec<F128> = (0..WB).map(|j| embed(Gf8((WB + j) as u8))).collect();
        let ntt = Gf8Ntt::<WB>::new();

        for n in [4usize, 6] {
            let num = 1usize << n;
            let a: Vec<u64> = (0..num).map(|i| sample_word(i as u64 * 3 + 1, WB)).collect();
            let b: Vec<u64> = (0..num).map(|i| sample_word(i as u64 * 5 + 2, WB)).collect();
            let c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let eq: Vec<F128> = (0..num).map(|i| sample128(i as u64 + 70)).collect();

            let got = gf8_round_message::<WB>(&a, &b, &c, &eq, &ntt);

            // Reference in GF(2^128) over embed(H).
            let mut want = [F128::zero(); WB];
            for x in 0..num {
                let ae = gf128_extend_over_embed::<WB>(a[x], &base, &ext_points);
                let be = gf128_extend_over_embed::<WB>(b[x], &base, &ext_points);
                let ce = gf128_extend_over_embed::<WB>(c[x], &base, &ext_points);
                for j in 0..WB {
                    want[j] += (ae[j] * be[j] - ce[j]) * eq[x];
                }
            }
            assert_eq!(got, want, "gf8 round message mismatch at WB={WB} n={n}");
        }
    }

    #[test]
    fn gf8_round_message_matches_gf128_over_embed_subspace_d32() {
        check_round_message_matches::<32>();
    }
    #[test]
    fn gf8_round_message_matches_gf128_over_embed_subspace_d64() {
        check_round_message_matches::<64>();
    }

    /// Prove with the GF(2^8) scheme + verify over embed(H): the full Phase-1+2
    /// protocol must round-trip (honest AND), validating that the GF(2^8) round
    /// message, the embed-base fold, and the embed-full R₀ reconstruction are
    /// mutually consistent — at the given lane width `WB`.
    fn check_scheme_round_trips<const WB: usize>() {
        use super::super::oblong_and::{
            ReplayChannel, prove_oblong_and_channel, verify_oblong_and_channel,
        };
        for n in [4usize, 6] {
            let num = 1usize << n;
            let a: Vec<u64> = (0..num).map(|i| sample_word(i as u64 * 3 + 1, WB)).collect();
            let b: Vec<u64> = (0..num).map(|i| sample_word(i as u64 * 5 + 2, WB)).collect();
            let c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
            let scheme = Gf8Scheme::<WB>::new();

            // Shared sampled-challenge sequence for prove + verify: only the BIG
            // challenges (n−3, the 3 small ones are deterministic from the
            // scheme), then z, then the n sumcheck γ.
            let mut seq: Vec<F128> = (0..n - 3).map(|i| sample128(i as u64 + 1)).collect();
            seq.push(sample128(0xBEEF));
            seq.extend((0..n).map(|k| sample128(0xCAFE + k as u64)));

            let mut pch = ReplayChannel::new(seq.clone());
            let (proof, prover_point) = prove_oblong_and_channel(&mut pch, &a, &b, &c, &scheme);

            let mut vch = ReplayChannel::new(seq);
            let out = verify_oblong_and_channel(
                &mut vch,
                &proof,
                n,
                scheme.full_subspace(),
                scheme.small_challenges(),
            )
            .expect("GF(2^8) scheme must verify honest AND");
            assert_eq!(out.eval_point.len(), n + 1);
            // The prover's exposed eval-point must equal the verifier's re-derived
            // one (same `[z, γ…]` drawn in the same order) — the invariant the
            // sound ψ_z binding relies on to recombine projected columns at γ.
            assert_eq!(prover_point, out.eval_point);
        }
    }

    #[test]
    fn gf8_scheme_full_protocol_round_trips_d32() {
        check_scheme_round_trips::<32>();
    }
    #[test]
    fn gf8_scheme_full_protocol_round_trips_d64() {
        check_scheme_round_trips::<64>();
    }

    /// extend_word = Σ L_b over set bits ⇒ linear: ext(w1^w2)=ext(w1)+ext(w2).
    fn check_extend_is_f2_linear<const WB: usize>() {
        let ntt = Gf8Ntt::<WB>::new();
        let w1 = sample_word(11, WB);
        let w2 = sample_word(22, WB);
        let e1 = ntt.extend_word(w1);
        let e2 = ntt.extend_word(w2);
        let e12 = ntt.extend_word(w1 ^ w2);
        for j in 0..WB {
            assert_eq!(e12[j], e1[j] + e2[j]);
        }
    }

    #[test]
    fn gf8_extend_is_f2_linear_d32() {
        check_extend_is_f2_linear::<32>();
    }
    #[test]
    fn gf8_extend_is_f2_linear_d64() {
        check_extend_is_f2_linear::<64>();
    }

    /// Release-mode timing: GF(2^8) byte-lookup round message vs the naive
    /// GF(2^128) one, at the nvars=16 discharge scale, for both `WB ∈ {32, 64}`.
    /// Run with:
    /// `cargo test -p zinc-poly --release --features simd \
    ///   oblong_and_gf8::tests::bench -- --ignored --nocapture`
    fn bench_round_message<const WB: usize>() {
        use super::super::oblong_and::{AdditiveNtt, eq_indicator, univariate_round_message};
        use std::time::Instant;

        let n = 16usize;
        let num = 1usize << n;
        let a: Vec<u64> = (0..num).map(|i| sample_word(i as u64 * 3 + 1, WB)).collect();
        let b: Vec<u64> = (0..num).map(|i| sample_word(i as u64 * 5 + 2, WB)).collect();
        let c: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        let r: Vec<F128> = (0..n).map(|i| sample128(i as u64 + 9)).collect();
        let eq = eq_indicator(&r);

        let gf128_ntt = AdditiveNtt::<WB>::new();
        let gf8_ntt = Gf8Ntt::<WB>::new();

        let reps = 5;
        let mut best_128 = f64::MAX;
        let mut best_8 = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let m1 = univariate_round_message(&a, &b, &c, &eq, &gf128_ntt);
            best_128 = best_128.min(t.elapsed().as_secs_f64());
            std::hint::black_box(m1);

            let t = Instant::now();
            let m2 = gf8_round_message::<WB>(&a, &b, &c, &eq, &gf8_ntt);
            best_8 = best_8.min(t.elapsed().as_secs_f64());
            std::hint::black_box(m2);
        }
        let per = |s: f64| s * 1e9 / num as f64;
        println!(
            "\n  round message, WB={WB} n={n} ({num} words), best of {reps}:\n    \
             GF(2^128) naive : {:8.2} ms  ({:6.2} ns/word)\n    \
             GF(2^8)  lookup : {:8.2} ms  ({:6.2} ns/word)\n    \
             speedup         : {:.2}x  (NTT+products in GF(2^8); eq-weighting still GF(2^128))\n",
            best_128 * 1e3,
            per(best_128),
            best_8 * 1e3,
            per(best_8),
            best_128 / best_8,
        );
    }

    #[test]
    #[ignore = "timing benchmark; run with --release --ignored --nocapture"]
    fn bench_gf8_vs_gf128_round_message() {
        bench_round_message::<32>();
        bench_round_message::<64>();
    }
}
