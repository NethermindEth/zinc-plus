//! Arity-8 Merkle-compiled limbwise-folding IOPP over the radix-8 chain.
//!
//! One fold round binds *three* multilinear variables: each limb splits into
//! its eight mod-8 coefficient classes, all `8 k` pieces are folded with
//! fresh challenges, and the folded vector is re-decomposed into balanced
//! base-`2^p` limbs (`p >= lambda + 4` keeps the limb count at two). Claims
//! are tracked linearly as in the arity-2 protocol, with the split check
//! using the three-variable Lagrange weights
//! `w_i = prod_l ((1 - z_l) or z_l)` over the bits of the class index `i`.
//!
//! Leaves of the round-`r` tree bundle the full 8-coset
//! `(u_t[j + c * H])_{c in [8]}` of every limb (`H = n_r / 8`), so each
//! query opens one leaf per round; the cross-round consistency check at the
//! chained position `j = s mod n_r` derives the sub-code symbols through the
//! lifted 8x8 DFT block by solving the transposed system `M_j^T z = alpha`
//! fraction-free ([`Radix8Chain::solve_scaled`]) and verifying
//! `d * sum_j' 2^{p(j'-1)} child_{j'}[j] == sum_t z_t^T c_t` over the
//! integers --- no rationals, no precomputed inverses.

use super::{
    BasefoldError,
    chain::ChainConfig,
    chain8::{ARITY, LOG_ARITY, Radix8Chain},
    iopp::{
        absorb_field, absorb_ints, balanced_digit_bounds, eval_projected_mle, mul_u32,
        squeeze_challenge, two_pow,
    },
    limbs::{decompose_balanced, next_limb_count_arity},
};
use crate::merkle::{MerkleProof, MerkleTree, MtHash};
use crypto_primitives::{FromWithConfig, PrimeField, crypto_bigint_int::Int};
use num_traits::ConstZero;
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_utils::{add, inner_transparent_field::InnerTransparentField, mul, mul_by_scalar::MulByScalar, sub};

/// Challenge bitlength implied by the challenge width parameter.
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)] // NCU is a small const
const fn lambda_bits(ncu: usize) -> u32 {
    64 * (ncu as u32)
}

/// Parameters of the arity-8 limbwise-folding IOPP.
#[derive(Clone, Debug)]
pub struct Arity8Params<C: ChainConfig> {
    pub chain: Radix8Chain<C>,
    /// Balanced digit width; must satisfy `p >= lambda + 4`.
    pub p: u32,
    /// Number of folding rounds `R <= chain.num_levels()`.
    pub num_rounds: usize,
    /// `log2` of the witness length (`= 3 * num_levels + log2(tail)`).
    pub num_vars: usize,
    /// Number of spot-check queries.
    pub num_queries: usize,
}

impl<C: ChainConfig> Arity8Params<C> {
    pub fn new(
        chain: Radix8Chain<C>,
        p: u32,
        num_rounds: usize,
        num_queries: usize,
    ) -> Result<Self, BasefoldError> {
        if num_queries == 0 {
            return Err(BasefoldError::InvalidParams(
                "num_queries must be positive".to_owned(),
            ));
        }
        if num_rounds == 0 || num_rounds > chain.num_levels() {
            return Err(BasefoldError::InvalidParams(format!(
                "num_rounds ({num_rounds}) must be in 1..={}",
                chain.num_levels()
            )));
        }
        let num_vars = chain.msg_len_at(0).trailing_zeros() as usize;
        Ok(Self {
            chain,
            p,
            num_rounds,
            num_vars,
            num_queries,
        })
    }
}

/// Commitment / hint / proof mirror the arity-2 compiled protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arity8Commitment {
    pub root: MtHash,
}

#[derive(Clone, Debug)]
pub struct Arity8Hint<const NK: usize> {
    codewords: Vec<Vec<Int<NK>>>,
    tree: MerkleTree,
}

#[derive(Clone, Debug)]
pub struct Leaf8Opening<const NK: usize> {
    /// Row-ordered values: for each limb `t`, the coset
    /// `(u_t[j + c * H])_{c in [8]}`.
    pub values: Vec<Int<NK>>,
    pub proof: MerkleProof,
}

#[derive(Clone, Debug)]
pub struct Arity8Proof<F: PrimeField, const ND: usize, const NK: usize> {
    pub round_roots: Vec<MtHash>,
    pub tail: Vec<Vec<Int<ND>>>,
    /// Per round, per limb: the eight class claims.
    pub class_claims: Vec<Vec<[F; ARITY]>>,
    /// Per round `1..num_rounds`: new-limb claims.
    pub limb_claims: Vec<Vec<F>>,
    pub queries: Vec<Vec<Leaf8Opening<NK>>>,
}

impl<F: PrimeField, const ND: usize, const NK: usize> Arity8Proof<F, ND, NK> {
    #[allow(clippy::arithmetic_side_effects)] // counting
    pub fn size_bytes(&self, field_elem_bytes: usize) -> usize {
        let hash = 32usize;
        let nd = <Int<ND> as ConstTranscribable>::NUM_BYTES;
        let nk = <Int<NK> as ConstTranscribable>::NUM_BYTES;
        let roots = self.round_roots.len() * hash;
        let tail: usize = self.tail.iter().map(|l| l.len() * nd).sum();
        let claims = (ARITY * self.class_claims.iter().map(Vec::len).sum::<usize>()
            + self.limb_claims.iter().map(Vec::len).sum::<usize>())
            * field_elem_bytes;
        let queries: usize = self
            .queries
            .iter()
            .flat_map(|q| q.iter())
            .map(|o| o.values.len() * nk + o.proof.siblings.len() * hash + 16)
            .sum();
        roots + tail + claims + queries
    }
}

/// Rows of the round tree: limb-major, the eight `H`-length segments of each
/// limb codeword, so column `j` is the 8-coset of every limb.
#[allow(clippy::arithmetic_side_effects)] // segment arithmetic over powers of two
fn build_round_tree<const NK: usize>(codewords: &[Vec<Int<NK>>]) -> MerkleTree {
    let h = codewords.first().map(|cw| cw.len() / ARITY).unwrap_or_default();
    let rows: Vec<&[Int<NK>]> = codewords
        .iter()
        .flat_map(|cw| (0..ARITY).map(move |c| &cw[c * h..(c + 1) * h]))
        .collect();
    MerkleTree::new(&rows)
}

fn check_digit_range<const ND: usize>(
    values: &[Int<ND>],
    p: u32,
    what: &str,
) -> Result<(), BasefoldError> {
    let (neg_half, half) = balanced_digit_bounds::<ND>(p);
    if values.iter().any(|d| *d >= half || *d < neg_half) {
        return Err(BasefoldError::InvalidParams(format!(
            "{what} entries must lie in the balanced digit range"
        )));
    }
    Ok(())
}

pub fn commit<C, const ND: usize, const NK: usize, const CHECK: bool>(
    params: &Arity8Params<C>,
    witness: &[Int<ND>],
) -> Result<(Arity8Commitment, Arity8Hint<NK>), BasefoldError>
where
    C: ChainConfig,
{
    if witness.len() != params.chain.msg_len_at(0) {
        return Err(BasefoldError::InvalidParams(
            "witness length mismatch".to_owned(),
        ));
    }
    check_digit_range(witness, params.p, "witness")?;
    let codewords = vec![
        params
            .chain
            .encode_at_level::<Int<ND>, Int<NK>, CHECK>(0, witness),
    ];
    let tree = build_round_tree(&codewords);
    Ok((
        Arity8Commitment { root: tree.root() },
        Arity8Hint { codewords, tree },
    ))
}

/// The three-variable Lagrange weights of the class index bits:
/// `w_i = prod_{l < 3} ((1 - z_l) if bit l of i is 0 else z_l)`.
fn class_weights<F>(z: &[F], cfg: &F::Config) -> [F; ARITY]
where
    F: PrimeField + Clone,
{
    let one = F::one_with_cfg(cfg);
    std::array::from_fn(|i| {
        let mut w = F::one_with_cfg(cfg);
        for (l, z_l) in z.iter().enumerate().take(LOG_ARITY as usize) {
            let factor = if (i >> l) & 1 == 0 {
                one.clone() - z_l.clone()
            } else {
                z_l.clone()
            };
            w *= factor;
        }
        w
    })
}

fn split_classes<const ND: usize>(limb: &[Int<ND>]) -> Vec<Vec<Int<ND>>> {
    (0..ARITY)
        .map(|i| limb.iter().skip(i).step_by(ARITY).copied().collect())
        .collect()
}

/// `sum_t sum_i alpha_{t,i} * class_{t,i}`, accumulated wide.
#[allow(clippy::arithmetic_side_effects)] // index arithmetic over verified lengths
fn fold_classes<const ND: usize, const NW: usize, const CHECK: bool>(
    limbs: &[Vec<Int<ND>>],
    challenges: &[[Int<NW>; ARITY]],
) -> Vec<Int<NW>> {
    debug_assert_eq!(limbs.len(), challenges.len());
    let out_len = limbs.first().map_or(0, |l| l.len() / ARITY);
    (0..out_len)
        .map(|idx| {
            let mut acc = Int::<NW>::ZERO;
            for (limb, alphas) in limbs.iter().zip(challenges) {
                for (i, alpha) in alphas.iter().enumerate() {
                    let term = limb[ARITY * idx + i]
                        .resize::<NW>()
                        .mul_by_scalar::<CHECK>(alpha)
                        .expect("fold term overflow");
                    acc = add!(acc, term);
                }
            }
            acc
        })
        .collect()
}

fn squeeze_class_challenges<const NW: usize, const NCU: usize>(
    transcript: &mut impl Transcript,
    k: usize,
) -> Vec<[Int<NW>; ARITY]> {
    (0..k)
        .map(|_| std::array::from_fn(|_| squeeze_challenge::<NW, NCU>(transcript)))
        .collect()
}

fn validate_widths8<const NW: usize, const NCU: usize>(
    params: &Arity8Params<impl ChainConfig>,
) -> Result<(), BasefoldError> {
    let lambda = lambda_bits(NCU);
    if params.p < add!(lambda, add!(LOG_ARITY, 1u32)) {
        return Err(BasefoldError::InvalidParams(format!(
            "digit width p={} must be at least lambda + {} = {}",
            params.p,
            add!(LOG_ARITY, 1u32),
            add!(lambda, add!(LOG_ARITY, 1u32))
        )));
    }
    if u32::try_from(NW)
        .ok()
        .and_then(|nw| nw.checked_mul(64))
        .is_none_or(|bits| bits < add!(params.p, add!(lambda, 16u32)))
    {
        return Err(BasefoldError::InvalidParams(
            "wide accumulator width NW too small".to_owned(),
        ));
    }
    Ok(())
}

/// Prover. Caller has absorbed `commitment.root` and sampled `cfg`, `point`.
#[allow(clippy::too_many_lines)]
pub fn prove<
    C,
    F,
    const ND: usize,
    const NK: usize,
    const NW: usize,
    const NCU: usize,
    const CHECK: bool,
>(
    params: &Arity8Params<C>,
    witness: &[Int<ND>],
    hint: &Arity8Hint<NK>,
    point: &[F],
    cfg: &F::Config,
    transcript: &mut impl Transcript,
) -> Result<(Arity8Proof<F, ND, NK>, F), BasefoldError>
where
    C: ChainConfig,
    F: InnerTransparentField
        + for<'a> FromWithConfig<&'a Int<ND>>
        + for<'a> FromWithConfig<&'a Int<NW>>
        + Clone,
    F::Inner: Transcribable + ConstTranscribable,
    F::Modulus: Transcribable,
{
    validate_widths8::<NW, NCU>(params)?;
    check_digit_range(witness, params.p, "witness")?;
    if point.len() != params.num_vars
        || params.num_vars < mul!(3usize, params.num_rounds)
    {
        return Err(BasefoldError::InvalidParams(
            "point length / round count mismatch".to_owned(),
        ));
    }

    let r_max = params.num_rounds;
    let eval = eval_projected_mle::<F, ND>(witness, point, cfg)?;

    let mut cur_limbs: Vec<Vec<Int<ND>>> = vec![witness.to_vec()];
    let mut class_claims_all: Vec<Vec<[F; ARITY]>> = Vec::with_capacity(r_max);
    let mut limb_claims_all: Vec<Vec<F>> = Vec::with_capacity(r_max.saturating_sub(1));
    let mut tail: Vec<Vec<Int<ND>>> = Vec::new();
    let mut round_roots: Vec<MtHash> = Vec::with_capacity(r_max.saturating_sub(1));
    let mut inter_codewords: Vec<Vec<Vec<Int<NK>>>> = Vec::with_capacity(r_max.saturating_sub(1));
    let mut inter_trees: Vec<MerkleTree> = Vec::with_capacity(r_max.saturating_sub(1));

    for r in 1..=r_max {
        let suffix = &point[mul!(3usize, r)..];

        // 1. Class claims for every current limb.
        let mut round_class_claims: Vec<[F; ARITY]> = Vec::with_capacity(cur_limbs.len());
        for limb in &cur_limbs {
            let classes = split_classes(limb);
            let claims: [F; ARITY] = std::array::from_fn(|i| {
                eval_projected_mle::<F, ND>(&classes[i], suffix, cfg)
                    .expect("class MLE evaluation")
            });
            for e in &claims {
                absorb_field(transcript, e);
            }
            round_class_claims.push(claims);
        }

        // 2. Challenges (eight per limb).
        let challenges = squeeze_class_challenges::<NW, NCU>(transcript, cur_limbs.len());

        // 3. Fold all classes, re-decompose, commit or send tail.
        let folded = fold_classes::<ND, NW, CHECK>(&cur_limbs, &challenges);
        let k_next =
            next_limb_count_arity(lambda_bits(NCU), params.p, cur_limbs.len(), LOG_ARITY);
        let new_limbs = decompose_balanced::<NW, ND>(&folded, params.p, k_next)?;

        if r < r_max {
            let codewords: Vec<Vec<Int<NK>>> = new_limbs
                .iter()
                .map(|limb| {
                    params
                        .chain
                        .encode_at_level::<Int<ND>, Int<NK>, CHECK>(r, limb)
                })
                .collect();
            let tree = build_round_tree(&codewords);
            let root = tree.root();
            transcript.absorb_slice(&root.0);
            round_roots.push(root);
            inter_codewords.push(codewords);
            inter_trees.push(tree);

            let mut claims_r: Vec<F> = Vec::with_capacity(k_next);
            for limb in &new_limbs {
                let e = eval_projected_mle::<F, ND>(limb, suffix, cfg)?;
                absorb_field(transcript, &e);
                claims_r.push(e);
            }
            limb_claims_all.push(claims_r);
        } else {
            for limb in &new_limbs {
                absorb_ints(transcript, limb);
            }
            tail = new_limbs.clone();
        }

        class_claims_all.push(round_class_claims);
        cur_limbs = new_limbs;
    }

    // 4. Query phase.
    let mut queries: Vec<Vec<Leaf8Opening<NK>>> = Vec::with_capacity(params.num_queries);
    for _ in 0..params.num_queries {
        let s = squeeze_query_index(transcript, params.chain.codeword_len_at(1));
        let mut openings: Vec<Leaf8Opening<NK>> = Vec::with_capacity(r_max);
        for r in 0..r_max {
            let (codewords, tree) = if r == 0 {
                (&hint.codewords, &hint.tree)
            } else {
                (
                    &inter_codewords[sub!(r, 1usize)],
                    &inter_trees[sub!(r, 1usize)],
                )
            };
            let h = params.chain.codeword_len_at(add!(r, 1usize));
            let leaf_index = s & sub!(h, 1usize);
            let mut values = Vec::with_capacity(mul!(codewords.len(), ARITY));
            for cw in codewords {
                for c in 0..ARITY {
                    values.push(cw[add!(leaf_index, mul!(c, h))]);
                }
            }
            let proof = tree.prove(leaf_index).map_err(|e| {
                BasefoldError::InvalidParams(format!("merkle proving failed: {e}"))
            })?;
            openings.push(Leaf8Opening { values, proof });
        }
        queries.push(openings);
    }

    Ok((
        Arity8Proof {
            round_roots,
            tail,
            class_claims: class_claims_all,
            limb_claims: limb_claims_all,
            queries,
        },
        eval,
    ))
}

/// Verifier. Caller has absorbed `commitment.root` and sampled `cfg`, `point`.
#[allow(clippy::too_many_lines)]
pub fn verify<
    C,
    F,
    const ND: usize,
    const NK: usize,
    const NW: usize,
    const NCU: usize,
    const CHECK: bool,
>(
    params: &Arity8Params<C>,
    commitment: &Arity8Commitment,
    proof: &Arity8Proof<F, ND, NK>,
    point: &[F],
    claimed_eval: &F,
    cfg: &F::Config,
    transcript: &mut impl Transcript,
) -> Result<(), BasefoldError>
where
    C: ChainConfig,
    F: InnerTransparentField
        + for<'a> FromWithConfig<&'a Int<ND>>
        + for<'a> FromWithConfig<&'a Int<NW>>
        + Clone
        + PartialEq,
    F::Inner: Transcribable + ConstTranscribable,
    F::Modulus: Transcribable,
{
    validate_widths8::<NW, NCU>(params)?;
    let r_max = params.num_rounds;
    if proof.class_claims.len() != r_max
        || proof.limb_claims.len() != sub!(r_max, 1usize)
        || proof.round_roots.len() != sub!(r_max, 1usize)
        || proof.queries.len() != params.num_queries
        || proof.queries.iter().any(|q| q.len() != r_max)
        || point.len() != params.num_vars
    {
        return Err(BasefoldError::InvalidParams(
            "proof shape does not match parameters".to_owned(),
        ));
    }

    let mut cur_claims: Vec<F> = vec![claimed_eval.clone()];
    let mut all_challenges: Vec<Vec<[Int<NW>; ARITY]>> = Vec::with_capacity(r_max);
    let mut limb_counts: Vec<usize> = vec![1];
    let mut tail_codewords: Vec<Vec<Int<NK>>> = Vec::new();

    for r in 1..=r_max {
        let z_round = &point[mul!(3usize, sub!(r, 1usize))..mul!(3usize, r)];
        let weights = class_weights::<F>(z_round, cfg);
        let k_prev = cur_claims.len();

        // 1. Class-claim split identities.
        let round_class_claims = &proof.class_claims[sub!(r, 1usize)];
        if round_class_claims.len() != k_prev {
            return Err(BasefoldError::ClaimCheckFailed { round: r });
        }
        for (t, claims) in round_class_claims.iter().enumerate() {
            let mut recombined = F::zero_with_cfg(cfg);
            for (w, e) in weights.iter().zip(claims.iter()) {
                recombined += w.clone() * e.clone();
            }
            if recombined != cur_claims[t] {
                return Err(BasefoldError::ClaimCheckFailed { round: r });
            }
            for e in claims {
                absorb_field(transcript, e);
            }
        }

        // 2. Challenges.
        let challenges = squeeze_class_challenges::<NW, NCU>(transcript, k_prev);

        let mut folded_claim = F::zero_with_cfg(cfg);
        for (alphas, claims) in challenges.iter().zip(round_class_claims) {
            for (alpha, e) in alphas.iter().zip(claims.iter()) {
                folded_claim += F::from_with_cfg(alpha, cfg) * e.clone();
            }
        }

        // 3. Root / tail absorption and the recombination check.
        let k_next =
            next_limb_count_arity(lambda_bits(NCU), params.p, k_prev, LOG_ARITY);
        let claims_r: Vec<F> = if r < r_max {
            transcript.absorb_slice(&proof.round_roots[sub!(r, 1usize)].0);
            let claims = proof.limb_claims[sub!(r, 1usize)].clone();
            if claims.len() != k_next {
                return Err(BasefoldError::ClaimCheckFailed { round: r });
            }
            for e in &claims {
                absorb_field(transcript, e);
            }
            claims
        } else {
            if proof.tail.len() != k_next {
                return Err(BasefoldError::TailCheckFailed(format!(
                    "expected {k_next} tail limbs, got {}",
                    proof.tail.len()
                )));
            }
            let mut claims = Vec::with_capacity(proof.tail.len());
            for limb in &proof.tail {
                if limb.len() != params.chain.msg_len_at(r_max) {
                    return Err(BasefoldError::TailCheckFailed(
                        "tail limb length mismatch".to_owned(),
                    ));
                }
                check_digit_range(limb, params.p, "tail")
                    .map_err(|_| BasefoldError::TailCheckFailed("digit range".to_owned()))?;
                absorb_ints(transcript, limb);
                claims.push(eval_projected_mle::<F, ND>(
                    limb,
                    &point[mul!(3usize, r)..],
                    cfg,
                )?);
                tail_codewords.push(
                    params
                        .chain
                        .encode_at_level::<Int<ND>, Int<NK>, CHECK>(r_max, limb),
                );
            }
            claims
        };

        let mut recombined = F::zero_with_cfg(cfg);
        for (j, e) in claims_r.iter().enumerate() {
            let weight = two_pow::<F, NW>(mul_u32(params.p, j), cfg);
            recombined += weight * e.clone();
        }
        if recombined != folded_claim {
            return Err(BasefoldError::ClaimCheckFailed { round: r });
        }

        all_challenges.push(challenges);
        limb_counts.push(k_next);
        cur_claims = claims_r;
    }

    // 4. Query phase.
    for (q_idx, openings) in proof.queries.iter().enumerate() {
        let s = squeeze_query_index(transcript, params.chain.codeword_len_at(1));

        // Merkle verification of every opened leaf.
        for (r, opening) in openings.iter().enumerate() {
            let h = params.chain.codeword_len_at(add!(r, 1usize));
            let leaf_index = s & sub!(h, 1usize);
            if opening.values.len() != mul!(limb_counts[r], ARITY) {
                return Err(BasefoldError::InvalidParams(format!(
                    "query {q_idx}: round-{r} leaf has wrong arity"
                )));
            }
            let root = if r == 0 {
                &commitment.root
            } else {
                &proof.round_roots[sub!(r, 1usize)]
            };
            opening
                .proof
                .verify(root, &opening.values, leaf_index)
                .map_err(|_| BasefoldError::MerkleCheckFailed {
                    round: r,
                    query: q_idx,
                })?;
        }

        // Cross-round consistency through the lifted DFT blocks.
        for r in 1..=r_max {
            let n_r = params.chain.codeword_len_at(r);
            let j = s & sub!(n_r, 1usize);
            let challenges = &all_challenges[sub!(r, 1usize)];
            let (d, zs) = params
                .chain
                .solve_scaled::<NW>(sub!(r, 1usize), j, challenges)?;

            // RHS: sum over limbs of z_t^T c_t (coset values, c-major order).
            let parent = &openings[sub!(r, 1usize)];
            let mut rhs = Int::<NW>::ZERO;
            for (t, z) in zs.iter().enumerate() {
                for (c, z_c) in z.iter().enumerate() {
                    let val = parent.values[add!(mul!(t, ARITY), c)].resize::<NW>();
                    let term = val
                        .mul_by_scalar::<CHECK>(z_c)
                        .ok_or(BasefoldError::ConsistencyCheckFailed {
                            round: r,
                            position: j,
                        })?;
                    rhs = add!(rhs, term);
                }
            }

            // LHS: d * sum_j' 2^{p(j'-1)} child_{j'}[j].
            let child_at_j: Vec<Int<NK>> = if r < r_max {
                let h = params.chain.codeword_len_at(add!(r, 1usize));
                // h is a power of two (codeword lengths).
                let slot = j >> h.trailing_zeros();
                let leaf = &openings[r];
                (0..limb_counts[r])
                    .map(|t| leaf.values[add!(mul!(t, ARITY), slot)])
                    .collect()
            } else {
                tail_codewords.iter().map(|cw| cw[j]).collect()
            };
            let mut w = Int::<NW>::ZERO;
            #[allow(clippy::arithmetic_side_effects)] // shl overflow-checked
            for v in child_at_j.iter().rev() {
                w = add!(w << params.p, v.resize::<NW>());
            }
            let lhs = w
                .mul_by_scalar::<CHECK>(&d)
                .ok_or(BasefoldError::ConsistencyCheckFailed {
                    round: r,
                    position: j,
                })?;

            if lhs != rhs {
                return Err(BasefoldError::ConsistencyCheckFailed {
                    round: r,
                    position: j,
                });
            }
        }
    }

    Ok(())
}

fn squeeze_query_index(transcript: &mut impl Transcript, cap: usize) -> usize {
    debug_assert!(cap.is_power_of_two());
    let x: u32 = transcript.get_challenge();
    (x as usize) & sub!(cap, 1usize)
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::basefold::chain::ChainConfigF167772161;
    use crypto_primitives::crypto_bigint_monty::MontyField;
    use num_traits::ConstOne;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use zinc_primality::MillerRabin;
    use zinc_transcript::Blake3Transcript;

    type F = MontyField<4>;
    const ND: usize = 3;
    const NK: usize = 5; // 320-bit leaves (radix-8 norm growth is tight)
    const NW: usize = 12;
    const NCU: usize = 2; // lambda = 128
    const P: u32 = 132; // lambda + 4
    const Q: usize = 30;

    fn random_witness(rng: &mut StdRng, len: usize) -> Vec<Int<ND>> {
        (0..len)
            .map(|_| {
                let mut words = [0u64; ND];
                words[0] = rng.random();
                words[1] = rng.random();
                words[2] = rng.random::<u64>() & 7; // 131-bit magnitude
                let v = Int::<ND>::new(crypto_bigint::Int::from_words(words));
                if rng.random::<bool>() { -v } else { v }
            })
            .collect()
    }

    struct Setup {
        params: Arity8Params<ChainConfigF167772161>,
        witness: Vec<Int<ND>>,
        commitment: Arity8Commitment,
        hint: Arity8Hint<NK>,
    }

    fn setup(num_vars: usize, num_rounds: usize) -> Setup {
        let msg_len = 1usize << num_vars;
        let chain =
            Radix8Chain::<ChainConfigF167772161>::new(msg_len, 4, num_rounds).unwrap();
        let params = Arity8Params::new(chain, P, num_rounds, Q).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let witness = random_witness(&mut rng, msg_len);
        let (commitment, hint) = commit::<_, ND, NK, true>(&params, &witness).unwrap();
        Setup {
            params,
            witness,
            commitment,
            hint,
        }
    }

    fn open_transcript(
        commitment: &Arity8Commitment,
        num_vars: usize,
    ) -> (Blake3Transcript, <F as PrimeField>::Config, Vec<F>) {
        let mut t = Blake3Transcript::new();
        t.absorb_slice(&commitment.root.0);
        let cfg = t.get_random_field_cfg::<F, crypto_primitives::crypto_bigint_uint::Uint<4>, MillerRabin>();
        let point: Vec<F> = t.get_field_challenges(num_vars, &cfg);
        (t, cfg, point)
    }

    #[test]
    fn complete_end_to_end_arity8() {
        let s = setup(9, 3);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 9);
        let (proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 9);
        verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params,
            &s.commitment,
            &proof,
            &point_v,
            &eval,
            &cfg_v,
            &mut tv,
        )
        .unwrap();

        eprintln!(
            "arity-8 basefold proof (m=2^9, R=3, Q={Q}): ~{} bytes",
            proof.size_bytes(32)
        );
    }

    #[test]
    fn rejects_wrong_eval_arity8() {
        let s = setup(9, 3);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 9);
        let (proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        let bad = eval + F::one_with_cfg(&cfg);
        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 9);
        let res = verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params,
            &s.commitment,
            &proof,
            &point_v,
            &bad,
            &cfg_v,
            &mut tv,
        );
        assert!(matches!(res, Err(BasefoldError::ClaimCheckFailed { round: 1 })));
    }

    #[test]
    fn rejects_tampered_leaf_arity8() {
        let s = setup(9, 3);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 9);
        let (mut proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        proof.queries[0][1].values[2] = add!(proof.queries[0][1].values[2], Int::<NK>::ONE);
        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 9);
        let res = verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params,
            &s.commitment,
            &proof,
            &point_v,
            &eval,
            &cfg_v,
            &mut tv,
        );
        assert!(matches!(
            res,
            Err(BasefoldError::MerkleCheckFailed { round: 1, query: 0 })
        ));
    }

    #[test]
    fn rejects_tampered_tail_arity8() {
        let s = setup(9, 3);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 9);
        let (mut proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        proof.tail[0][0] = add!(proof.tail[0][0], Int::<ND>::ONE);
        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 9);
        let res = verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params,
            &s.commitment,
            &proof,
            &point_v,
            &eval,
            &cfg_v,
            &mut tv,
        );
        assert!(res.is_err());
    }

    /// Arity-8 depth dial at m = 2^12 with production-flavored parameters;
    /// compare against the radix-2 numbers in
    /// `compiled::tests::depth_dial_report`.
    #[test]
    #[ignore = "experiment; run with --ignored --nocapture"]
    fn depth_dial_report_arity8() {
        use std::time::Instant;
        const QQ: usize = 100;
        let num_vars = 12usize;
        let msg_len = 1usize << num_vars;
        let mut rng = StdRng::seed_from_u64(99);
        let witness = random_witness(&mut rng, msg_len);

        eprintln!(
            "arity-8 depth dial: m = 2^{num_vars}, rep = 4, q = 5*2^25+1, \
             p = {P}, lambda = 128, Q = {QQ}, leaf ints {} B",
            <Int<NK> as ConstTranscribable>::NUM_BYTES
        );
        for num_rounds in [2usize, 3, 4] {
            let chain = Radix8Chain::<ChainConfigF167772161>::new(
                msg_len, 4, num_rounds,
            )
            .unwrap();
            let params = Arity8Params::new(chain, P, num_rounds, QQ).unwrap();

            let t0 = Instant::now();
            let (commitment, hint) = commit::<_, ND, NK, true>(&params, &witness).unwrap();
            let t_commit = t0.elapsed();

            let (mut tp, cfg, point) = open_transcript(&commitment, num_vars);
            let t0 = Instant::now();
            let (proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
                &params, &witness, &hint, &point, &cfg, &mut tp,
            )
            .unwrap();
            let t_prove = t0.elapsed();

            let (mut tv, cfg_v, point_v) = open_transcript(&commitment, num_vars);
            let t0 = Instant::now();
            verify::<_, F, ND, NK, NW, NCU, true>(
                &params,
                &commitment,
                &proof,
                &point_v,
                &eval,
                &cfg_v,
                &mut tv,
            )
            .unwrap();
            let t_verify = t0.elapsed();

            let mut max_leaf_bits = 0u32;
            for o in proof.queries.iter().flat_map(|q| q.iter()) {
                for v in &o.values {
                    max_leaf_bits = max_leaf_bits.max(v.inner().abs().bits());
                }
            }
            let tail_exp = num_vars - 3 * num_rounds;
            eprintln!(
                "R = {num_rounds}: tail 2^{tail_exp} | {:7} B total | max leaf bits \
                 {max_leaf_bits:3} | commit {t_commit:?} prove {t_prove:?} verify {t_verify:?}",
                proof.size_bytes(32)
            );
        }
    }
}
