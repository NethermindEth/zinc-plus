use crypto_primitives::{BaseFieldConfig, ProjectElementWithConfig, boolean::Boolean};
use itertools::Itertools;
use num_traits::Zero;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, binary_ref::BinaryRefPoly, binary_u64::BinaryU64Poly},
};
use zinc_utils::{add, mul};

/// Fold a trace of one type into a trace of another (similar but smaller) type.
///
/// Note that folding will increase number of variables for MLEs by
/// `ilog2(Self::FOLDING_FACTOR)`.
pub trait FoldTrace<From, To> {
    /// Folding factor, a positive power of 2.
    const FOLDING_FACTOR: usize;

    fn fold_trace_mle(mle: &DenseMultilinearExtension<From>) -> DenseMultilinearExtension<To>;

    /// Verifier-side: compute one column's contribution to the folded PCS
    /// eval-claim at the extended evaluation point
    /// `r_0 || gamma_1 || ... || gamma_k`, where `k = log2(FOLDING_FACTOR)`.
    ///
    /// # Inputs:
    /// - `bar_u_coeffs`: the column's unfolded lifted-eval coefficients in
    ///   `F_q[X]`, of formal length `D`. May be shorter than `D` if trailing
    ///   zero coefficients were trimmed.
    /// - `alphas`: per-poly PCS alphas, of length `D / FOLDING_FACTOR`,
    ///   matching the folded entry size.
    /// - `folding_challenges`: `k` field-typed challenges sampled by the
    ///   verifier, in sampling order (`gamma_1` first, `gamma_k` last).
    ///
    /// Returns the value `<alphas, bar_u_folded>` where `bar_u_folded` is the
    /// column's lifted-eval polynomial after applying `k` chained 2x splits
    /// and pinning the appended boolean variables to `(gamma_1, ..., gamma_k)`.
    fn fold_eval_claim<C, A>(
        bar_u_coeffs: &[C::Element],
        alphas: &[A],
        folding_challenges: &[C::Element],
        field_cfg: &C,
    ) -> C::Element
    where
        C: BaseFieldConfig + ProjectElementWithConfig<A>,
    {
        // MSB-first contiguous chained-2x-split chunking.
        //
        // `bar_u_coeffs` is partitioned into `FOLDING_FACTOR` contiguous chunks of
        // length `D / FOLDING_FACTOR`.
        //
        // Their inner products with `alphas` form a `k`-variable multilinear polynomial
        // which is then evaluated at `(gamma_1, ..., gamma_k)` with `gamma_1` paired
        // with the high bit of the chunk index.
        //
        // This covers `NoopFoldTrace` (k = 0, degenerating to a single inner product)
        // as well as chained 2x folds.

        debug_assert_eq!(
            1usize << folding_challenges.len(),
            Self::FOLDING_FACTOR,
            "fold_eval_claim: 1 << folding_challenges.len() must equal FOLDING_FACTOR",
        );
        debug_assert!(
            bar_u_coeffs.len() <= mul!(alphas.len(), Self::FOLDING_FACTOR),
            "fold_eval_claim: bar_u_coeffs.len() must not exceed alphas.len() * FOLDING_FACTOR",
        );

        let chunk_size = alphas.len();
        let alphas = alphas.iter().map(|a| field_cfg.project(a)).collect_vec();

        // Step 1: Per-chunk inner products
        //   P_i = sum_{j < chunk_size} alphas[j] * bar_u_coeffs[i*chunk_size + j],
        // Trimmed (missing-trailing-zero) coefficients are treated as zero.
        let chunk_evals: Vec<C::Element> = (0..Self::FOLDING_FACTOR)
            .map(|i| {
                let start = mul!(i, chunk_size);
                let mut acc = field_cfg.zero();
                for (j, alpha) in alphas.iter().enumerate() {
                    if let Some(coeff) = bar_u_coeffs.get(add!(start, j)) {
                        field_cfg.add_assign(&mut acc, &field_cfg.mul(alpha, coeff));
                    }
                }
                acc
            })
            .collect();

        // Step 2: MLE-evaluate the per-chunk inner products at folding_challenges,
        // MSB-first (gamma_1 = high bit, peeled first).
        mle_eval_msb_first(field_cfg, chunk_evals, folding_challenges)
    }
}

//
// NOOP fold
//

pub struct NoopFoldTrace;

impl<T: Clone> FoldTrace<T, T> for NoopFoldTrace {
    const FOLDING_FACTOR: usize = 1;

    fn fold_trace_mle(mle: &DenseMultilinearExtension<T>) -> DenseMultilinearExtension<T> {
        mle.clone()
    }
}

//
// Binary folds
//

pub struct FoldBinaryTrace2x<const D: usize, const HALF_D: usize>;

impl<const D: usize, const HALF_D: usize> FoldTrace<BinaryPoly<D>, BinaryPoly<HALF_D>>
    for FoldBinaryTrace2x<D, HALF_D>
{
    const FOLDING_FACTOR: usize = 2;

    fn fold_trace_mle(
        mle: &DenseMultilinearExtension<BinaryPoly<D>>,
    ) -> DenseMultilinearExtension<BinaryPoly<HALF_D>> {
        split_binary_poly_mle(mle)
    }
}

pub struct FoldBinaryTrace4x<const D: usize, const HALF_D: usize, const QUARTER_D: usize>;

impl<const D: usize, const HALF_D: usize, const QUARTER_D: usize>
    FoldTrace<BinaryPoly<D>, BinaryPoly<QUARTER_D>> for FoldBinaryTrace4x<D, HALF_D, QUARTER_D>
{
    const FOLDING_FACTOR: usize = 4;

    fn fold_trace_mle(
        mle: &DenseMultilinearExtension<BinaryPoly<D>>,
    ) -> DenseMultilinearExtension<BinaryPoly<QUARTER_D>> {
        let mle = split_binary_poly_mle::<D, HALF_D>(mle);
        split_binary_poly_mle::<HALF_D, QUARTER_D>(&mle)
    }
}

//
// Helper functions
//

/// Split a column of `BinaryPoly<D>` entries into a concatenated column
/// of `BinaryPoly<HALF_D>` entries.
///
/// Each entry `v[i]` with `D` binary coefficients is split into:
/// - `u[i]` = low `HALF_D` coefficients (indices `0..HALF_D`)
/// - `w[i]` = high `HALF_D` coefficients (indices `HALF_D..D`)
///
/// so that `v[i] = u[i] + X^HALF_D · w[i]`.
///
/// Returns a column of length `2n` where:
/// - `v'[0..n]   = u[0..n]`  (low halves)
/// - `v'[n..2n]  = w[0..n]`  (high halves)
///
/// The returned MLE has `num_vars + 1` variables, with the last variable
/// selecting between the low half (0) and high half (1).
///
/// Panics at compile-time if `D != 2 * HALF_D`.
fn split_binary_poly_mle<const D: usize, const HALF_D: usize>(
    mle: &DenseMultilinearExtension<BinaryPoly<D>>,
) -> DenseMultilinearExtension<BinaryPoly<HALF_D>> {
    const {
        assert!(D == 2 * HALF_D, "split_column: D must equal 2 * HALF_D");
    }

    #[cfg(not(feature = "simd"))]
    let res = split_binary_poly_mle_ref(mle);

    #[cfg(feature = "simd")]
    let res = split_binary_poly_mle_u64(mle);

    res
}

#[allow(dead_code)]
fn split_binary_poly_mle_ref<const D: usize, const HALF_D: usize>(
    mle: &DenseMultilinearExtension<BinaryRefPoly<D>>,
) -> DenseMultilinearExtension<BinaryRefPoly<HALF_D>> {
    let n = mle.evaluations.len();
    let mut lo_evals = Vec::with_capacity(n);
    let mut hi_evals = Vec::with_capacity(n);

    for entry in &mle.evaluations {
        let lo_arr: [Boolean; HALF_D] = std::array::from_fn(|i| entry[i]);
        let hi_arr: [Boolean; HALF_D] = std::array::from_fn(|i| entry[add!(HALF_D, i)]);
        lo_evals.push(BinaryRefPoly::<HALF_D>::new(lo_arr));
        hi_evals.push(BinaryRefPoly::<HALF_D>::new(hi_arr));
    }

    // Concatenate: v' = u || w (low halves first, high halves second).
    lo_evals.extend(hi_evals);

    DenseMultilinearExtension::from_evaluations_vec(add!(mle.num_vars, 1), lo_evals, Zero::zero())
}

#[allow(dead_code)]
fn split_binary_poly_mle_u64<const D: usize, const HALF_D: usize>(
    mle: &DenseMultilinearExtension<BinaryU64Poly<D>>,
) -> DenseMultilinearExtension<BinaryU64Poly<HALF_D>> {
    let n = mle.evaluations.len();
    let mut lo_evals: Vec<BinaryU64Poly<HALF_D>> = Vec::with_capacity(n);
    let mut hi_evals: Vec<BinaryU64Poly<HALF_D>> = Vec::with_capacity(n);

    for entry in &mle.evaluations {
        let bits: u64 = *entry.inner();
        // `From<u64>` masks off bits at positions `>= HALF_D` so each half
        // upholds the `BinaryU64Poly<HALF_D>` invariant.
        lo_evals.push(BinaryU64Poly::<HALF_D>::from(bits));
        hi_evals.push(BinaryU64Poly::<HALF_D>::from(bits >> HALF_D));
    }

    // Concatenate: v' = u || w (low halves first, high halves second), matching
    // the layout produced by `split_binary_poly_mle_ref`.
    lo_evals.extend(hi_evals);

    DenseMultilinearExtension::from_evaluations_vec(add!(mle.num_vars, 1), lo_evals, Zero::zero())
}

/// Multilinear evaluation of `values` (length `2^gammas.len()`, MSB-first
/// indexed) at point `gammas`. Peels `gammas[0]` (the high bit, equivalently
/// the first sampled challenge) at each recursive step, splitting `values`
/// into a lower half (high bit = 0) and an upper half (high bit = 1).
fn mle_eval_msb_first<C: BaseFieldConfig>(
    cfg: &C,
    values: Vec<C::Element>,
    gammas: &[C::Element],
) -> C::Element {
    if gammas.is_empty() {
        debug_assert_eq!(values.len(), 1);
        return values.into_iter().next().expect("non-empty values");
    }
    debug_assert_eq!(values.len(), 1usize << gammas.len());

    let half = values.len() >> 1;
    let g = &gammas[0];
    let one_minus_g = cfg.sub(&cfg.one(), g);

    let mut next: Vec<C::Element> = Vec::with_capacity(half);
    for i in 0..half {
        let mut lo = cfg.mul(&one_minus_g, &values[i]);
        cfg.add_assign(&mut lo, &cfg.mul(g, &values[add!(i, half)]));
        next.push(lo);
    }
    mle_eval_msb_first(cfg, next, &gammas[1..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::Wrapper;
    use rand::prelude::*;
    use zinc_transcript::traits::GenTranscribable;

    /// Build two MLEs (`BinaryRefPoly<D>` and `BinaryU64Poly<D>`) carrying the
    /// same coefficient pattern from a list of bit-packed `u64` entries.
    fn build_matched_mles<const D: usize>(
        bits_list: &[u64],
    ) -> (
        DenseMultilinearExtension<BinaryRefPoly<D>>,
        DenseMultilinearExtension<BinaryU64Poly<D>>,
    ) {
        let n = bits_list.len();
        assert!(n.is_power_of_two(), "n must be a power of two");
        let num_vars = n.trailing_zeros() as usize;

        let ref_entries: Vec<BinaryRefPoly<D>> = bits_list
            .iter()
            .map(|&bits| BinaryRefPoly::read_transcription_bytes_exact(&bits.to_le_bytes()))
            .collect();
        let u64_entries: Vec<BinaryU64Poly<D>> = bits_list
            .iter()
            .map(|&bits| BinaryU64Poly::from(bits))
            .collect();

        let ref_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, ref_entries, Zero::zero());
        let u64_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, u64_entries, Zero::zero());

        (ref_mle, u64_mle)
    }

    /// Run both splitters on matched inputs and assert that every output
    /// coefficient agrees bit-for-bit.
    fn assert_split_matches<const D: usize, const HALF_D: usize>(bits_list: Vec<u64>) {
        let (ref_mle, u64_mle) = build_matched_mles::<D>(&bits_list);

        let split_ref = split_binary_poly_mle_ref::<D, HALF_D>(&ref_mle);
        let split_u64 = split_binary_poly_mle_u64::<D, HALF_D>(&u64_mle);

        assert_eq!(split_ref.num_vars, split_u64.num_vars);
        assert_eq!(split_ref.evaluations.len(), split_u64.evaluations.len());
        for (idx, (r, u)) in split_ref
            .evaluations
            .iter()
            .zip(split_u64.evaluations.iter())
            .enumerate()
        {
            // Compare bits in two different ways - directly, as via iterator

            for i in 0..HALF_D {
                let r_bit = *r[i].inner();
                let u_bit = ((*u.inner()) >> i) & 1 != 0;
                assert_eq!(
                    r_bit, u_bit,
                    "mismatch at output entry {idx}, coefficient bit {i}",
                );
            }

            for (i, pair) in r.iter().zip_longest(u.iter()).enumerate() {
                match pair {
                    itertools::EitherOrBoth::Both(r_bit, u_bit) => {
                        assert_eq!(
                            *r_bit.inner(),
                            *u_bit,
                            "mismatch at output entry {idx}, coefficient bit {i}",
                        );
                    }
                    itertools::EitherOrBoth::Left(_) | itertools::EitherOrBoth::Right(_) => {
                        panic!("mismatch in number of coefficients at output entry {idx}");
                    }
                }
            }
        }
    }

    #[test]
    fn split_ref_and_u64_match_d4_exhaustive() {
        // Single-entry input: enumerate all 16 bit patterns.
        for bits in 0u64..16 {
            assert_split_matches::<4, 2>(vec![bits]);
        }

        // 4-entry input: enumerate all 16^4 = 65536 patterns is too much; sample.
        let mut rng = rand::rng();
        for _ in 0..32 {
            let bits_list: Vec<u64> = (0..4).map(|_| rng.random::<u64>() & 0xF).collect();
            assert_split_matches::<4, 2>(bits_list);
        }
    }

    #[test]
    fn split_ref_and_u64_match_random() {
        let mut rng = rand::rng();

        for n_log in 0..=3 {
            let n = 1usize << n_log;
            let bits_list: Vec<u64> = (0..n).map(|_| rng.random::<u64>() & 0xF).collect();
            assert_split_matches::<4, 2>(bits_list);
        }

        for n_log in 0..=4 {
            let n = 1usize << n_log;
            let bits_list: Vec<u64> = (0..n).map(|_| rng.random::<u64>() & 0xFF).collect();
            assert_split_matches::<8, 4>(bits_list);
        }

        for n_log in 0..=5 {
            let n = 1usize << n_log;
            let bits_list: Vec<u64> = (0..n).map(|_| rng.random::<u64>() & 0xFFFF_FFFF).collect();
            assert_split_matches::<32, 16>(bits_list);
        }

        for n_log in 0..=6 {
            let n = 1usize << n_log;
            let bits_list: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
            assert_split_matches::<64, 32>(bits_list);
        }
    }

    #[test]
    fn split_u64_pins_all_zero_entries() {
        // Zero input should round-trip to all-zero output regardless of D.
        let bits_list = vec![0u64; 8];
        assert_split_matches::<8, 4>(bits_list.clone());
        assert_split_matches::<32, 16>(bits_list.clone());
        assert_split_matches::<64, 32>(bits_list);
    }

    #[test]
    fn split_u64_handles_all_ones_d64() {
        // All-ones (every bit set) is the high-edge case for D=64 since the
        // mask `(1 << 64) - 1` is not directly representable. Expect lo = hi =
        // 2^32 - 1 for D=64, HALF_D=32.
        let bits_list = vec![u64::MAX; 4];
        assert_split_matches::<64, 32>(bits_list);
    }
}
