//! Merkle-compiled limbwise-folding IOPP with a spot-check query phase.
//!
//! Same round logic and Fiat--Shamir schedule as [`super::iopp`], with two
//! changes of transport:
//! - each round's limb codewords are committed under one Merkle tree with
//!   *pair-bundled leaves*: leaf `j in [n_{r+1}]` of the round-`r` tree
//!   hashes, limb-major, the values `(u_{r,t}[j], u_{r,t}[j + n_{r+1}])` ---
//!   exactly the opening set both the round-`r` and round-`(r+1)`
//!   consistency checks need at a chained query position;
//! - instead of checking every position, the verifier draws `num_queries`
//!   positions `s` from the transcript after the last round and checks, per
//!   query, the cross-round consistency at the chained positions
//!   `j_{r-1} = s mod n_r`, against Merkle-verified leaf openings. One leaf
//!   opening per round per query.
//!
//! The transcript ritual for callers is as in [`super::iopp`], with the
//! commitment now being the round-0 Merkle root: absorb `commitment.root`,
//! sample the claim field and the evaluation point, then call
//! `prove` / `verify` on the same transcript.

use super::{
    BasefoldError,
    chain::ChainConfig,
    iopp::{
        BasefoldParams, absorb_field, absorb_ints, balanced_digit_bounds,
        check_consistency_values, check_witness_digits, eval_projected_mle, fold_limbs,
        lambda_bits, mul_u32, split_even_odd, squeeze_round_challenges, two_pow,
        validate_widths,
    },
    limbs::{decompose_balanced, next_limb_count},
};
use crate::merkle::{MerkleProof, MerkleTree, MtHash};
use crypto_primitives::{FromWithConfig, PrimeField, crypto_bigint_int::Int};
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_utils::{add, inner_transparent_field::InnerTransparentField, sub};

/// The commitment of the compiled IOPP: the round-0 Merkle root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasefoldCommitment {
    pub root: MtHash,
}

/// Prover-side opening hint: the round-0 limb codewords and their tree.
#[derive(Clone, Debug)]
pub struct BasefoldHint<const NK: usize> {
    codewords: Vec<Vec<Int<NK>>>,
    tree: MerkleTree,
}

/// One Merkle-verified leaf opening: the pair-bundled column values in row
/// order (`u_t[j], u_t[j + half]` for each limb `t`) plus the path.
#[derive(Clone, Debug)]
pub struct LeafOpening<const NK: usize> {
    pub values: Vec<Int<NK>>,
    pub proof: MerkleProof,
}

/// The compiled proof.
#[derive(Clone, Debug)]
pub struct CompiledProof<F: PrimeField, const ND: usize, const NK: usize> {
    /// Roots of the intermediate-round trees (rounds `1..num_rounds`,
    /// exclusive of the tail round).
    pub round_roots: Vec<MtHash>,
    /// Cleartext tail limbs (level `num_rounds` digits).
    pub tail: Vec<Vec<Int<ND>>>,
    /// Per round, per limb: the half-claims.
    pub half_claims: Vec<Vec<(F, F)>>,
    /// Per round `1..num_rounds` (exclusive of tail): new-limb claims.
    pub limb_claims: Vec<Vec<F>>,
    /// Per query, per round `0..num_rounds`: one leaf opening.
    pub queries: Vec<Vec<LeafOpening<NK>>>,
}

impl<F: PrimeField, const ND: usize, const NK: usize> CompiledProof<F, ND, NK> {
    /// Serialized-size estimate in bytes (hashes are 32 bytes; integers use
    /// their fixed transcription width; field elements `field_elem_bytes`).
    #[allow(clippy::arithmetic_side_effects)] // counting
    pub fn size_bytes(&self, field_elem_bytes: usize) -> usize {
        let hash = 32usize;
        let nd = <Int<ND> as ConstTranscribable>::NUM_BYTES;
        let nk = <Int<NK> as ConstTranscribable>::NUM_BYTES;
        let roots = self.round_roots.len() * hash;
        let tail: usize = self.tail.iter().map(|l| l.len() * nd).sum();
        let claims = (2 * self
            .half_claims
            .iter()
            .map(Vec::len)
            .sum::<usize>()
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

/// Build the round tree with pair-bundled leaves: rows are, limb-major, the
/// low and high halves of each limb codeword; leaf `j` then hashes
/// `(u_t[j], u_t[j + half])` for all `t`.
#[allow(clippy::arithmetic_side_effects)] // codeword lengths are even powers of two
fn build_round_tree<const NK: usize>(codewords: &[Vec<Int<NK>>]) -> MerkleTree {
    let half = codewords
        .first()
        .map(|cw| cw.len() / 2)
        .unwrap_or_default();
    let rows: Vec<&[Int<NK>]> = codewords
        .iter()
        .flat_map(|cw| [&cw[..half], &cw[half..]])
        .collect();
    MerkleTree::new(&rows)
}

/// Commit to the witness: encode its limbs at chain level 0 and build the
/// pair-bundled Merkle tree.
pub fn commit<C, const ND: usize, const NK: usize, const CHECK: bool>(
    params: &BasefoldParams<C>,
    witness: &[Int<ND>],
) -> Result<(BasefoldCommitment, BasefoldHint<NK>), BasefoldError>
where
    C: ChainConfig,
{
    check_witness_digits(params, witness)?;
    let codewords = vec![
        params
            .chain
            .encode_at_level::<Int<ND>, Int<NK>, CHECK>(0, witness),
    ];
    let tree = build_round_tree(&codewords);
    Ok((
        BasefoldCommitment { root: tree.root() },
        BasefoldHint { codewords, tree },
    ))
}

/// Prover. The caller must already have absorbed `commitment.root` into
/// `transcript` and sampled `cfg` and `point` from it.
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
    params: &BasefoldParams<C>,
    witness: &[Int<ND>],
    hint: &BasefoldHint<NK>,
    point: &[F],
    cfg: &F::Config,
    transcript: &mut impl Transcript,
) -> Result<(CompiledProof<F, ND, NK>, F), BasefoldError>
where
    C: ChainConfig,
    F: InnerTransparentField
        + for<'a> FromWithConfig<&'a Int<ND>>
        + for<'a> FromWithConfig<&'a Int<NW>>
        + Clone,
    F::Inner: Transcribable + ConstTranscribable,
    F::Modulus: Transcribable,
{
    validate_widths::<ND, NK, NW, NCU>(params)?;
    check_witness_digits(params, witness)?;
    if point.len() != params.num_vars {
        return Err(BasefoldError::InvalidParams(format!(
            "point has {} coordinates, expected {}",
            point.len(),
            params.num_vars
        )));
    }

    let r_max = params.num_rounds;
    let eval = eval_projected_mle::<F, ND>(witness, point, cfg)?;

    let mut cur_limbs: Vec<Vec<Int<ND>>> = vec![witness.to_vec()];
    let mut half_claims_all: Vec<Vec<(F, F)>> = Vec::with_capacity(r_max);
    let mut limb_claims_all: Vec<Vec<F>> = Vec::with_capacity(r_max.saturating_sub(1));
    let mut tail: Vec<Vec<Int<ND>>> = Vec::new();
    let mut round_roots: Vec<MtHash> = Vec::with_capacity(r_max.saturating_sub(1));
    // Intermediate-round transport state for the query phase.
    let mut inter_codewords: Vec<Vec<Vec<Int<NK>>>> = Vec::with_capacity(r_max.saturating_sub(1));
    let mut inter_trees: Vec<MerkleTree> = Vec::with_capacity(r_max.saturating_sub(1));

    for r in 1..=r_max {
        let suffix = &point[r..];

        // 1. Half-claims.
        let mut round_half_claims: Vec<(F, F)> = Vec::with_capacity(cur_limbs.len());
        for limb in &cur_limbs {
            let (even, odd) = split_even_odd(limb);
            let e_e = eval_projected_mle::<F, ND>(&even, suffix, cfg)?;
            let e_o = eval_projected_mle::<F, ND>(&odd, suffix, cfg)?;
            absorb_field(transcript, &e_e);
            absorb_field(transcript, &e_o);
            round_half_claims.push((e_e, e_o));
        }

        // 2. Challenges.
        let challenges = squeeze_round_challenges::<NW, NCU>(transcript, cur_limbs.len());

        // 3. Fold, decompose, commit (root) or send tail.
        let folded = fold_limbs::<ND, NW, CHECK>(&cur_limbs, &challenges);
        let k_next = next_limb_count(lambda_bits(NCU), params.p, cur_limbs.len());
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

        half_claims_all.push(round_half_claims);
        cur_limbs = new_limbs;
    }

    // 4. Query phase: open one pair-bundled leaf per round per query.
    let mut queries: Vec<Vec<LeafOpening<NK>>> = Vec::with_capacity(params.num_queries);
    for _ in 0..params.num_queries {
        let s = squeeze_query_index(transcript, params.chain.codeword_len_at(1));
        let mut openings: Vec<LeafOpening<NK>> = Vec::with_capacity(r_max);
        for r in 0..r_max {
            let (codewords, tree) = if r == 0 {
                (&hint.codewords, &hint.tree)
            } else {
                (&inter_codewords[sub!(r, 1usize)], &inter_trees[sub!(r, 1usize)])
            };
            let half = params.chain.codeword_len_at(add!(r, 1usize));
            let leaf_index = leaf_index_for(s, half);
            let values = leaf_values(codewords, leaf_index, half);
            let proof = tree.prove(leaf_index).map_err(|e| {
                BasefoldError::InvalidParams(format!("merkle proving failed: {e}"))
            })?;
            openings.push(LeafOpening { values, proof });
        }
        queries.push(openings);
    }

    Ok((
        CompiledProof {
            round_roots,
            tail,
            half_claims: half_claims_all,
            limb_claims: limb_claims_all,
            queries,
        },
        eval,
    ))
}

/// Verifier. The caller must already have absorbed `commitment.root` into
/// `transcript` and sampled `cfg` and `point` from it.
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
    params: &BasefoldParams<C>,
    commitment: &BasefoldCommitment,
    proof: &CompiledProof<F, ND, NK>,
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
    validate_widths::<ND, NK, NW, NCU>(params)?;
    let r_max = params.num_rounds;
    if proof.half_claims.len() != r_max
        || proof.limb_claims.len() != sub!(r_max, 1usize)
        || proof.round_roots.len() != sub!(r_max, 1usize)
        || proof.queries.len() != params.num_queries
        || proof.queries.iter().any(|q| q.len() != r_max)
    {
        return Err(BasefoldError::InvalidParams(
            "proof shape does not match parameters".to_owned(),
        ));
    }
    if point.len() != params.num_vars {
        return Err(BasefoldError::InvalidParams(format!(
            "point has {} coordinates, expected {}",
            point.len(),
            params.num_vars
        )));
    }

    let mut cur_claims: Vec<F> = vec![claimed_eval.clone()];
    let mut all_challenges: Vec<Vec<(Int<NW>, Int<NW>)>> = Vec::with_capacity(r_max);
    let mut limb_counts: Vec<usize> = vec![1]; // k_r per round, r = 0..r_max
    let mut tail_codewords: Vec<Vec<Int<NK>>> = Vec::new();

    for r in 1..=r_max {
        let z_r = &point[sub!(r, 1usize)];
        let one = F::one_with_cfg(cfg);
        let k_prev = cur_claims.len();

        // 1. Half-claim split identities.
        let round_half_claims = &proof.half_claims[sub!(r, 1usize)];
        if round_half_claims.len() != k_prev {
            return Err(BasefoldError::ClaimCheckFailed { round: r });
        }
        for (t, (e_e, e_o)) in round_half_claims.iter().enumerate() {
            let recombined = (one.clone() - z_r.clone()) * e_e.clone() + z_r.clone() * e_o.clone();
            if recombined != cur_claims[t] {
                return Err(BasefoldError::ClaimCheckFailed { round: r });
            }
            absorb_field(transcript, e_e);
            absorb_field(transcript, e_o);
        }

        // 2. Challenges.
        let challenges = squeeze_round_challenges::<NW, NCU>(transcript, k_prev);

        let mut folded_claim = F::zero_with_cfg(cfg);
        for ((alpha_e, alpha_o), (e_e, e_o)) in challenges.iter().zip(round_half_claims) {
            let fe = F::from_with_cfg(alpha_e, cfg);
            let fo = F::from_with_cfg(alpha_o, cfg);
            folded_claim = folded_claim + fe * e_e.clone() + fo * e_o.clone();
        }

        // 3. Root / tail absorption and new-limb claims.
        let k_next = next_limb_count(lambda_bits(NCU), params.p, k_prev);
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
            let (neg_half, half) = balanced_digit_bounds::<ND>(params.p);
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
                if limb.iter().any(|d| *d >= half || *d < neg_half) {
                    return Err(BasefoldError::TailCheckFailed(
                        "tail digit out of balanced range".to_owned(),
                    ));
                }
                absorb_ints(transcript, limb);
                claims.push(eval_projected_mle::<F, ND>(limb, &point[r..], cfg)?);
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

        // Merkle-verify every opened leaf against its absorbed root.
        for (r, opening) in openings.iter().enumerate() {
            let half = params.chain.codeword_len_at(add!(r, 1usize));
            let leaf_index = leaf_index_for(s, half);
            let expected_values = mul_usize(2, limb_counts[r]);
            if opening.values.len() != expected_values {
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

        // Cross-round consistency at the chained positions.
        for r in 1..=r_max {
            let n_r = params.chain.codeword_len_at(r);
            let j_prev = leaf_index_for(s, n_r);
            let tau = params.chain.twiddle(sub!(r, 1usize), j_prev);
            let parent_pairs: Vec<(Int<NK>, Int<NK>)> = openings[sub!(r, 1usize)]
                .values
                .chunks_exact(2)
                .map(|c| (c[0], c[1]))
                .collect();
            let child_at_j: Vec<Int<NK>> = if r < r_max {
                let half = params.chain.codeword_len_at(add!(r, 1usize));
                let slot = usize::from(j_prev >= half);
                openings[r]
                    .values
                    .chunks_exact(2)
                    .map(|c| c[slot])
                    .collect()
            } else {
                tail_codewords.iter().map(|cw| cw[j_prev]).collect()
            };
            check_consistency_values::<NK, NW, CHECK>(
                params.p,
                tau,
                &parent_pairs,
                &child_at_j,
                &all_challenges[sub!(r, 1usize)],
                r,
                j_prev,
            )?;
        }
    }

    Ok(())
}

/// Draw a query position in `[0, n_leaves_round0)` (a power of two).
fn squeeze_query_index(transcript: &mut impl Transcript, cap: usize) -> usize {
    debug_assert!(cap.is_power_of_two());
    let x: u32 = transcript.get_challenge();
    (x as usize) & sub!(cap, 1usize)
}

/// The pair-bundled leaf index of query position `s` in a tree with `half`
/// leaves (a power of two).
fn leaf_index_for(s: usize, half: usize) -> usize {
    debug_assert!(half.is_power_of_two());
    s & sub!(half, 1usize)
}

/// Column values of the pair-bundled leaf, in tree row order.
fn leaf_values<const NK: usize>(
    codewords: &[Vec<Int<NK>>],
    leaf_index: usize,
    half: usize,
) -> Vec<Int<NK>> {
    let mut values = Vec::with_capacity(codewords.len().saturating_mul(2));
    for cw in codewords {
        values.push(cw[leaf_index]);
        values.push(cw[add!(leaf_index, half)]);
    }
    values
}

#[allow(clippy::arithmetic_side_effects)]
fn mul_usize(a: usize, b: usize) -> usize {
    a.checked_mul(b).expect("index overflow")
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
    use crate::basefold::chain::{ChainConfigF65537, FoldableIprsChain};
    use crypto_primitives::crypto_bigint_monty::MontyField;
    use num_traits::ConstOne;
    use zinc_primality::MillerRabin;
    use zinc_transcript::Blake3Transcript;

    type F = MontyField<4>;
    const ND: usize = 3;
    const NK: usize = 8;
    const NW: usize = 8;
    const NCU: usize = 2; // lambda = 128
    const P: u32 = 130;
    const Q: usize = 30;

    struct Setup {
        params: BasefoldParams<ChainConfigF65537>,
        witness: Vec<Int<ND>>,
        commitment: BasefoldCommitment,
        hint: BasefoldHint<NK>,
    }

    fn setup(num_vars: usize, num_rounds: usize) -> Setup {
        let msg_len = 1usize << num_vars;
        let chain =
            FoldableIprsChain::<ChainConfigF65537>::new(msg_len, 4, num_rounds).unwrap();
        let params = BasefoldParams::new(chain, P, num_rounds, Q).unwrap();
        let witness: Vec<Int<ND>> = (0..msg_len as i32)
            .map(|i| Int::from(31 * i - 1000))
            .collect();
        let (commitment, hint) = commit::<_, ND, NK, true>(&params, &witness).unwrap();
        Setup {
            params,
            witness,
            commitment,
            hint,
        }
    }

    fn open_transcript(
        commitment: &BasefoldCommitment,
        num_vars: usize,
    ) -> (Blake3Transcript, <F as PrimeField>::Config, Vec<F>) {
        let mut t = Blake3Transcript::new();
        t.absorb_slice(&commitment.root.0);
        let cfg = t.get_random_field_cfg::<F, crypto_primitives::crypto_bigint_uint::Uint<4>, MillerRabin>();
        let point: Vec<F> = t.get_field_challenges(num_vars, &cfg);
        (t, cfg, point)
    }

    #[test]
    fn complete_end_to_end_compiled() {
        let s = setup(8, 6);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 8);
        let (proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 8);
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
            "compiled basefold proof (m=2^8, R=6, Q={Q}): ~{} bytes",
            proof.size_bytes(32)
        );
    }

    #[test]
    fn rejects_wrong_eval_compiled() {
        let s = setup(6, 4);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 6);
        let (proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        let bad_eval = eval + F::one_with_cfg(&cfg);
        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 6);
        let res = verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params,
            &s.commitment,
            &proof,
            &point_v,
            &bad_eval,
            &cfg_v,
            &mut tv,
        );
        assert!(matches!(res, Err(BasefoldError::ClaimCheckFailed { round: 1 })));
    }

    #[test]
    fn rejects_tampered_leaf_opening() {
        let s = setup(6, 4);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 6);
        let (mut proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        proof.queries[0][1].values[0] = add!(proof.queries[0][1].values[0], Int::<NK>::ONE);
        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 6);
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
    fn rejects_forged_round_root() {
        let s = setup(6, 4);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 6);
        let (mut proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        let mut bytes = proof.round_roots[0].0;
        bytes[0] ^= 1;
        proof.round_roots[0] = MtHash(bytes);
        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 6);
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

    #[test]
    fn rejects_tampered_tail_compiled() {
        let s = setup(6, 4);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 6);
        let (mut proof, eval) = prove::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &s.witness, &s.hint, &point, &cfg, &mut tp,
        )
        .unwrap();

        proof.tail[0][1] = add!(proof.tail[0][1], Int::<ND>::ONE);
        let (mut tv, cfg_v, point_v) = open_transcript(&s.commitment, 6);
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
}
