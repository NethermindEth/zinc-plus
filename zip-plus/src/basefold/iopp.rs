//! Interactive limbwise-folding IOPP: prover and verifier.
//!
//! Milestone-1 form of the Zip++ protocol: oracles are transmitted in full
//! (no Merkle compression yet) and the verifier evaluates the cross-round
//! consistency checks at *every* position; the Fiat--Shamir schedule is
//! already final, so the Merkle/query layer can replace the oracle transport
//! without touching the round logic.
//!
//! Protocol shape (one fold round, arity 2, limbs `v_{r-1,1..k}`):
//! 1. Prover sends per-limb half-claims `(e^e_t, e^o_t)`; verifier checks
//!    `(1 - z_r) e^e_t + z_r e^o_t = e_{r-1,t}` in `F` (the multilinear
//!    split identity; one coordinate of the evaluation point is consumed
//!    per round).
//! 2. Verifier sends fresh `lambda`-bit challenges `alpha^e_t, alpha^o_t`.
//! 3. Prover folds `w_r = sum_t alpha^e_t v^e_{r-1,t} + alpha^o_t
//!    v^o_{r-1,t}`, re-decomposes `w_r` into balanced base-`2^p` limbs,
//!    commits their level-`r` chain encodings (or, in the last round, sends
//!    the digits in the clear), and sends the new limb claims `e_{r,j}`.
//!    Verifier checks `sum_j 2^{p(j-1)} e_{r,j} = sum_t alpha^e_t e^e_t +
//!    alpha^o_t e^o_t` in `F`.
//! 4. (Query phase; here: all positions) For every position `j` of round
//!    `r`, with `tau = lift(omega_{r-1}^j)` and parent openings
//!    `c1_t = u_{r-1,t}[j]`, `c2_t = u_{r-1,t}[j + n_r]`, check over `Z`
//!    (denominators cleared):
//!    `2 tau * sum_j' 2^{p(j'-1)} u_{r,j'}[j]
//!       = sum_t alpha^e_t tau (c1_t + c2_t) + alpha^o_t (c1_t - c2_t)`.
//!
//! Fiat--Shamir schedule (identical in `prove` and `verify`): the caller
//! absorbs the round-0 oracle (the commitment) and samples the claim field
//! `F` and the evaluation point *before* calling; each round then absorbs
//! the half-claims, squeezes the challenges, and absorbs the new oracle and
//! limb claims (or the cleartext tail).

use super::{
    BasefoldError,
    chain::{ChainConfig, FoldableIprsChain},
    limbs::{decompose_balanced, next_limb_count},
};
use crypto_primitives::{FromWithConfig, PrimeField, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use num_traits::{ConstOne, ConstZero};
use std::fmt::Debug;
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_utils::{add, inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar, sub};

/// Parameters of the limbwise-folding IOPP.
#[derive(Clone, Debug)]
pub struct BasefoldParams<C: ChainConfig> {
    pub chain: FoldableIprsChain<C>,
    /// Balanced digit width; must satisfy `p >= lambda + 2` for the limb
    /// count to stabilize at two.
    pub p: u32,
    /// Number of folding rounds `R <= chain.num_levels()`.
    pub num_rounds: usize,
    /// `log2` of the witness length.
    pub num_vars: usize,
}

impl<C: ChainConfig> BasefoldParams<C> {
    pub fn new(
        chain: FoldableIprsChain<C>,
        p: u32,
        num_rounds: usize,
    ) -> Result<Self, BasefoldError> {
        if num_rounds == 0 || num_rounds > chain.num_levels() {
            return Err(BasefoldError::InvalidParams(format!(
                "num_rounds ({num_rounds}) must be in 1..={}",
                chain.num_levels()
            )));
        }
        let msg_len = chain.msg_len_at(0);
        let num_vars = msg_len.trailing_zeros() as usize;
        Ok(Self {
            chain,
            p,
            num_rounds,
            num_vars,
        })
    }
}

/// The (interactive-milestone) proof: full oracles plus tracked claims.
#[derive(Clone, Debug)]
pub struct BasefoldProof<F: PrimeField, const ND: usize, const NK: usize> {
    /// Limb codeword oracles for rounds `0..num_rounds` (round 0 is the
    /// witness commitment at chain level 0; round `r` lives at level `r`).
    pub round_oracles: Vec<Vec<Vec<Int<NK>>>>,
    /// Cleartext tail: the level-`num_rounds` limb digits.
    pub tail: Vec<Vec<Int<ND>>>,
    /// Per round `r = 1..=R`, per limb `t`: the half-claims `(e^e, e^o)`.
    pub half_claims: Vec<Vec<(F, F)>>,
    /// Per round `r = 1..R` (exclusive: the tail round sends digits
    /// instead): the new-limb claims `e_{r,j}`.
    pub limb_claims: Vec<Vec<F>>,
}

/// Challenge bitlength implied by the challenge width parameter.
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_truncation)] // NCU is a small const
const fn lambda_bits(ncu: usize) -> u32 {
    64 * (ncu as u32)
}

/// Encode the witness limbs at chain level 0 (the commitment message of the
/// IOPP). With the milestone restriction `k_0 = 1` this is one codeword.
pub fn commit<C, const ND: usize, const NK: usize, const CHECK: bool>(
    params: &BasefoldParams<C>,
    witness: &[Int<ND>],
) -> Result<Vec<Vec<Int<NK>>>, BasefoldError>
where
    C: ChainConfig,
{
    check_witness_digits(params, witness)?;
    Ok(vec![
        params
            .chain
            .encode_at_level::<Int<ND>, Int<NK>, CHECK>(0, witness),
    ])
}

fn check_witness_digits<C: ChainConfig, const ND: usize>(
    params: &BasefoldParams<C>,
    witness: &[Int<ND>],
) -> Result<(), BasefoldError> {
    if witness.len() != params.chain.msg_len_at(0) {
        return Err(BasefoldError::InvalidParams(format!(
            "witness length {} does not match chain message length {}",
            witness.len(),
            params.chain.msg_len_at(0)
        )));
    }
    let (neg_half, half) = balanced_digit_bounds::<ND>(params.p);
    if witness.iter().any(|d| *d >= half || *d < neg_half) {
        return Err(BasefoldError::InvalidParams(
            "witness entries must lie in the balanced digit range".to_owned(),
        ));
    }
    Ok(())
}

/// The balanced digit range `[-2^{p-1}, 2^{p-1})` as `(lower, upper)`.
#[allow(clippy::arithmetic_side_effects)] // p is validated against the digit width
fn balanced_digit_bounds<const ND: usize>(p: u32) -> (Int<ND>, Int<ND>) {
    let half = Int::<ND>::ONE << sub!(p, 1u32);
    (-half, half)
}

/// Prover. The caller must already have absorbed the `commit` output into
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
    point: &[F],
    cfg: &F::Config,
    transcript: &mut impl Transcript,
) -> Result<(BasefoldProof<F, ND, NK>, F), BasefoldError>
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
    // round_oracles[0] = the witness-limb codewords (k_0 = 1).
    let mut round_oracles: Vec<Vec<Vec<Int<NK>>>> =
        vec![commit::<C, ND, NK, CHECK>(params, witness)?];

    let eval = eval_projected_mle::<F, ND>(witness, point, cfg)?;

    let mut cur_limbs: Vec<Vec<Int<ND>>> = vec![witness.to_vec()];
    let mut half_claims_all: Vec<Vec<(F, F)>> = Vec::with_capacity(r_max);
    let mut limb_claims_all: Vec<Vec<F>> = Vec::with_capacity(r_max.saturating_sub(1));
    let mut tail: Vec<Vec<Int<ND>>> = Vec::new();

    for r in 1..=r_max {
        let suffix = &point[r..];

        // 1. Half-claims for every current limb.
        let mut round_half_claims: Vec<(F, F)> = Vec::with_capacity(cur_limbs.len());
        for limb in &cur_limbs {
            let (even, odd) = split_even_odd(limb);
            let e_e = eval_projected_mle::<F, ND>(&even, suffix, cfg)?;
            let e_o = eval_projected_mle::<F, ND>(&odd, suffix, cfg)?;
            absorb_field(transcript, &e_e);
            absorb_field(transcript, &e_o);
            round_half_claims.push((e_e, e_o));
        }

        // 2. Challenges (after all half-claims of the round).
        let challenges = squeeze_round_challenges::<NW, NCU>(transcript, cur_limbs.len());

        // 3. Fold and re-decompose.
        let folded = fold_limbs::<ND, NW, CHECK>(&cur_limbs, &challenges);
        let k_next = next_limb_count(lambda_bits(NCU), params.p, cur_limbs.len());
        let new_limbs = decompose_balanced::<NW, ND>(&folded, params.p, k_next)?;

        if r < r_max {
            let mut oracles_r: Vec<Vec<Int<NK>>> = Vec::with_capacity(k_next);
            let mut claims_r: Vec<F> = Vec::with_capacity(k_next);
            for limb in &new_limbs {
                let cw = params
                    .chain
                    .encode_at_level::<Int<ND>, Int<NK>, CHECK>(r, limb);
                absorb_ints(transcript, &cw);
                oracles_r.push(cw);
            }
            for limb in &new_limbs {
                let e = eval_projected_mle::<F, ND>(limb, suffix, cfg)?;
                absorb_field(transcript, &e);
                claims_r.push(e);
            }
            round_oracles.push(oracles_r);
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

    Ok((
        BasefoldProof {
            round_oracles,
            tail,
            half_claims: half_claims_all,
            limb_claims: limb_claims_all,
        },
        eval,
    ))
}

/// Verifier. The caller must already have absorbed `proof.round_oracles[0]`
/// into `transcript` and sampled `cfg` and `point` from it.
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
    proof: &BasefoldProof<F, ND, NK>,
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
    validate_proof_shape::<C, F, ND, NK>(params, proof)?;
    if point.len() != params.num_vars {
        return Err(BasefoldError::InvalidParams(format!(
            "point has {} coordinates, expected {}",
            point.len(),
            params.num_vars
        )));
    }

    let mut cur_claims: Vec<F> = vec![claimed_eval.clone()];
    let mut all_challenges: Vec<Vec<(Int<NW>, Int<NW>)>> = Vec::with_capacity(r_max);
    // Tail oracles get computed by the verifier itself from the cleartext.
    let mut tail_oracles: Vec<Vec<Int<NK>>> = Vec::new();

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

        // Folded claim: e' = sum_t alpha^e_t e^e_t + alpha^o_t e^o_t.
        let mut folded_claim = F::zero_with_cfg(cfg);
        for ((alpha_e, alpha_o), (e_e, e_o)) in challenges.iter().zip(round_half_claims) {
            let fe = F::from_with_cfg(alpha_e, cfg);
            let fo = F::from_with_cfg(alpha_o, cfg);
            folded_claim = folded_claim + fe * e_e.clone() + fo * e_o.clone();
        }

        // 3. New-limb claims and their recombination check.
        let k_next = next_limb_count(lambda_bits(NCU), params.p, k_prev);
        let claims_r: Vec<F> = if r < r_max {
            for cw in &proof.round_oracles[r] {
                absorb_ints(transcript, cw);
            }
            let claims = proof.limb_claims[sub!(r, 1usize)].clone();
            if claims.len() != k_next {
                return Err(BasefoldError::ClaimCheckFailed { round: r });
            }
            for e in &claims {
                absorb_field(transcript, e);
            }
            claims
        } else {
            // Cleartext tail: digit-range checks, then verifier-computed
            // claims and encodings.
            let (neg_half, half) = balanced_digit_bounds::<ND>(params.p);
            let mut claims = Vec::with_capacity(proof.tail.len());
            if proof.tail.len() != k_next {
                return Err(BasefoldError::TailCheckFailed(format!(
                    "expected {k_next} tail limbs, got {}",
                    proof.tail.len()
                )));
            }
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
                tail_oracles.push(
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
        cur_claims = claims_r;
    }

    // 4. Cross-round consistency at every position (interactive milestone).
    for r in 1..=r_max {
        let parent = &proof.round_oracles[sub!(r, 1usize)];
        let child: &Vec<Vec<Int<NK>>> = if r < r_max {
            &proof.round_oracles[r]
        } else {
            &tail_oracles
        };
        let n_r = params.chain.codeword_len_at(r);
        let challenges = &all_challenges[sub!(r, 1usize)];
        for j in 0..n_r {
            check_consistency_at::<C, NK, NW, CHECK>(params, r, j, parent, child, challenges)?;
        }
    }

    Ok(())
}

/// The cleared-denominator consistency check at one position.
///
/// `2 tau * sum_j' 2^{p(j'-1)} child[j'][j] ==
///   sum_t alpha^e_t * tau * (c1_t + c2_t) + alpha^o_t * (c1_t - c2_t)`.
#[allow(clippy::arithmetic_side_effects)] // widths budgeted; checked ops via macros
fn check_consistency_at<C, const NK: usize, const NW: usize, const CHECK: bool>(
    params: &BasefoldParams<C>,
    round: usize,
    j: usize,
    parent: &[Vec<Int<NK>>],
    child: &[Vec<Int<NK>>],
    challenges: &[(Int<NW>, Int<NW>)],
) -> Result<(), BasefoldError>
where
    C: ChainConfig,
{
    let n_r = params.chain.codeword_len_at(round);
    let tau = params.chain.twiddle(sub!(round, 1usize), j);
    let two_tau = mul_i64(2, tau);

    // LHS: recombine the child limbs at position j, times 2*tau.
    let mut lhs = Int::<NW>::ZERO;
    for limb_cw in child.iter().rev() {
        lhs = add!(lhs << params.p, limb_cw[j].resize::<NW>());
    }
    let lhs = lhs
        .mul_by_scalar::<CHECK>(&two_tau)
        .ok_or_else(overflow_err(round, j))?;

    // RHS: sum over parent limbs of alpha^e * tau * (c1 + c2) + alpha^o * (c1 - c2).
    let mut rhs = Int::<NW>::ZERO;
    for (limb_cw, (alpha_e, alpha_o)) in parent.iter().zip(challenges) {
        let c1 = limb_cw[j].resize::<NW>();
        let c2 = limb_cw[add!(j, n_r)].resize::<NW>();
        let even_part = add!(c1, c2)
            .mul_by_scalar::<CHECK>(&tau)
            .ok_or_else(overflow_err(round, j))?
            .mul_by_scalar::<CHECK>(alpha_e)
            .ok_or_else(overflow_err(round, j))?;
        let odd_part = sub!(c1, c2)
            .mul_by_scalar::<CHECK>(alpha_o)
            .ok_or_else(overflow_err(round, j))?;
        rhs = add!(rhs, add!(even_part, odd_part));
    }

    if lhs == rhs {
        Ok(())
    } else {
        Err(BasefoldError::ConsistencyCheckFailed {
            round,
            position: j,
        })
    }
}

fn overflow_err(round: usize, position: usize) -> impl Fn() -> BasefoldError {
    move || BasefoldError::ConsistencyCheckFailed { round, position }
}

// ---------- helpers ----------

fn validate_widths<const ND: usize, const NK: usize, const NW: usize, const NCU: usize>(
    params: &BasefoldParams<impl ChainConfig>,
) -> Result<(), BasefoldError> {
    let lambda = lambda_bits(NCU);
    if params.p < add!(lambda, 2u32) {
        return Err(BasefoldError::InvalidParams(format!(
            "digit width p={} must be at least lambda + 2 = {}",
            params.p,
            add!(lambda, 2u32)
        )));
    }
    if u32::try_from(NW)
        .ok()
        .and_then(|nw| nw.checked_mul(64))
        .is_none_or(|bits| bits < add!(params.p, add!(lambda, 8u32)))
    {
        return Err(BasefoldError::InvalidParams(
            "wide accumulator width NW too small for p + lambda".to_owned(),
        ));
    }
    let _ = (ND, NK);
    Ok(())
}

fn validate_proof_shape<C, F, const ND: usize, const NK: usize>(
    params: &BasefoldParams<C>,
    proof: &BasefoldProof<F, ND, NK>,
) -> Result<(), BasefoldError>
where
    C: ChainConfig,
    F: PrimeField,
{
    let r_max = params.num_rounds;
    if proof.round_oracles.len() != r_max
        || proof.half_claims.len() != r_max
        || proof.limb_claims.len() != sub!(r_max, 1usize)
    {
        return Err(BasefoldError::InvalidParams(
            "proof shape does not match parameters".to_owned(),
        ));
    }
    for (r, oracles_r) in proof.round_oracles.iter().enumerate() {
        let expected_len = params.chain.codeword_len_at(r);
        if oracles_r.is_empty() || oracles_r.iter().any(|cw| cw.len() != expected_len) {
            return Err(BasefoldError::InvalidParams(format!(
                "round-{r} oracle has wrong length"
            )));
        }
    }
    Ok(())
}

fn split_even_odd<const ND: usize>(limb: &[Int<ND>]) -> (Vec<Int<ND>>, Vec<Int<ND>>) {
    let even = limb.iter().step_by(2).copied().collect();
    let odd = limb.iter().skip(1).step_by(2).copied().collect();
    (even, odd)
}

/// `sum_t alpha^e_t * even_t + alpha^o_t * odd_t`, accumulated wide.
#[allow(clippy::arithmetic_side_effects)] // index arithmetic over verified lengths
fn fold_limbs<const ND: usize, const NW: usize, const CHECK: bool>(
    limbs: &[Vec<Int<ND>>],
    challenges: &[(Int<NW>, Int<NW>)],
) -> Vec<Int<NW>> {
    debug_assert_eq!(limbs.len(), challenges.len());
    let half_len = limbs.first().map_or(0, |l| l.len() / 2);
    (0..half_len)
        .map(|i| {
            let mut acc = Int::<NW>::ZERO;
            for (limb, (alpha_e, alpha_o)) in limbs.iter().zip(challenges) {
                let even = limb[mul_usize(2, i)].resize::<NW>();
                let odd = limb[add!(mul_usize(2, i), 1usize)].resize::<NW>();
                let te = even
                    .mul_by_scalar::<CHECK>(alpha_e)
                    .expect("fold even-term overflow");
                let to = odd
                    .mul_by_scalar::<CHECK>(alpha_o)
                    .expect("fold odd-term overflow");
                acc = add!(acc, add!(te, to));
            }
            acc
        })
        .collect()
}

/// Project the digits mod `q0` and evaluate the resulting MLE at `point`.
fn eval_projected_mle<F, const ND: usize>(
    digits: &[Int<ND>],
    point: &[F],
    cfg: &F::Config,
) -> Result<F, BasefoldError>
where
    F: InnerTransparentField + for<'a> FromWithConfig<&'a Int<ND>> + Clone,
{
    let evaluations: Vec<F::Inner> = digits
        .iter()
        .map(|d| F::from_with_cfg(d, cfg).into_inner())
        .collect();
    let num_vars = evaluations.len().trailing_zeros() as usize;
    let mle = DenseMultilinearExtension {
        num_vars,
        evaluations,
    };
    mle.evaluate_with_config(point, cfg)
        .map_err(|e| BasefoldError::InvalidParams(format!("MLE evaluation failed: {e:?}")))
}

/// Squeeze one non-negative `64 * NCU`-bit challenge, zero-extended wide.
fn squeeze_challenge<const NW: usize, const NCU: usize>(
    transcript: &mut impl Transcript,
) -> Int<NW> {
    let u: Uint<NCU> = transcript.get_challenge();
    let wide: crypto_bigint::Uint<NW> = u.inner().resize();
    Int::<NW>::new(crypto_bigint::Int::from_words(wide.to_words()))
}

fn squeeze_round_challenges<const NW: usize, const NCU: usize>(
    transcript: &mut impl Transcript,
    k: usize,
) -> Vec<(Int<NW>, Int<NW>)> {
    (0..k)
        .map(|_| {
            let alpha_e = squeeze_challenge::<NW, NCU>(transcript);
            let alpha_o = squeeze_challenge::<NW, NCU>(transcript);
            (alpha_e, alpha_o)
        })
        .collect()
}

fn absorb_ints<T: ConstTranscribable>(transcript: &mut impl Transcript, vals: &[T]) {
    let mut buf = vec![0u8; T::NUM_BYTES];
    for v in vals {
        v.write_transcription_bytes_exact(&mut buf);
        transcript.absorb_slice(&buf);
    }
}

fn absorb_field<F>(transcript: &mut impl Transcript, v: &F)
where
    F: PrimeField,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    let mut buf = vec![0u8; v.inner().get_num_bytes()];
    transcript.absorb_random_field(v, &mut buf);
}

/// `2^bits` as a field element.
#[allow(clippy::arithmetic_side_effects)] // shl is overflow-checked by crypto-bigint
fn two_pow<F, const NW: usize>(bits: u32, cfg: &F::Config) -> F
where
    F: PrimeField + for<'a> FromWithConfig<&'a Int<NW>>,
{
    let v = Int::<NW>::ONE << bits;
    F::from_with_cfg(&v, cfg)
}

#[allow(clippy::arithmetic_side_effects)]
fn mul_u32(a: u32, b: usize) -> u32 {
    a.checked_mul(u32::try_from(b).expect("limb index fits u32"))
        .expect("power-of-two exponent overflow")
}

#[allow(clippy::arithmetic_side_effects)]
fn mul_usize(a: usize, b: usize) -> usize {
    a.checked_mul(b).expect("index overflow")
}

#[allow(clippy::arithmetic_side_effects)]
fn mul_i64(a: i64, b: i64) -> i64 {
    a.checked_mul(b).expect("twiddle doubling overflow")
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
    use crate::basefold::chain::ChainConfigF65537;
    use crypto_primitives::crypto_bigint_monty::MontyField;
    use zinc_primality::MillerRabin;
    use zinc_transcript::Blake3Transcript;

    type F = MontyField<4>;
    const ND: usize = 3;
    const NK: usize = 8;
    const NW: usize = 8;
    const NCU: usize = 2; // lambda = 128
    const P: u32 = 130;

    struct Setup {
        params: BasefoldParams<ChainConfigF65537>,
        witness: Vec<Int<ND>>,
        commitment: Vec<Vec<Int<NK>>>,
    }

    fn setup(num_vars: usize, num_rounds: usize) -> Setup {
        let msg_len = 1usize << num_vars;
        let chain =
            FoldableIprsChain::<ChainConfigF65537>::new(msg_len, 4, num_rounds).unwrap();
        let params = BasefoldParams::new(chain, P, num_rounds).unwrap();
        // Signed witness entries, small but representative.
        let witness: Vec<Int<ND>> = (0..msg_len as i32)
            .map(|i| Int::from(31 * i - 1000))
            .collect();
        let commitment = commit::<_, ND, NK, true>(&params, &witness).unwrap();
        Setup {
            params,
            witness,
            commitment,
        }
    }

    /// The caller-side transcript ritual: absorb the commitment, sample the
    /// claim field and the evaluation point.
    fn open_transcript(
        commitment: &[Vec<Int<NK>>],
        num_vars: usize,
    ) -> (Blake3Transcript, <F as PrimeField>::Config, Vec<F>) {
        let mut t = Blake3Transcript::new();
        for cw in commitment {
            absorb_ints(&mut t, cw);
        }
        let cfg = t.get_random_field_cfg::<F, Uint<4>, MillerRabin>();
        let point: Vec<F> = t.get_field_challenges(num_vars, &cfg);
        (t, cfg, point)
    }

    #[test]
    fn complete_end_to_end() {
        let s = setup(8, 6);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 8);
        let (proof, eval) =
            prove::<_, F, ND, NK, NW, NCU, true>(&s.params, &s.witness, &point, &cfg, &mut tp)
                .unwrap();

        let (mut tv, cfg_v, point_v) = open_transcript(&proof.round_oracles[0], 8);
        verify::<_, F, ND, NK, NW, NCU, true>(&s.params, &proof, &point_v, &eval, &cfg_v, &mut tv)
            .unwrap();
    }

    #[test]
    fn rejects_wrong_eval() {
        let s = setup(6, 4);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 6);
        let (proof, eval) =
            prove::<_, F, ND, NK, NW, NCU, true>(&s.params, &s.witness, &point, &cfg, &mut tp)
                .unwrap();

        let bad_eval = eval + F::one_with_cfg(&cfg);
        let (mut tv, cfg_v, point_v) = open_transcript(&proof.round_oracles[0], 6);
        let res = verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &proof, &point_v, &bad_eval, &cfg_v, &mut tv,
        );
        assert!(matches!(res, Err(BasefoldError::ClaimCheckFailed { round: 1 })));
    }

    #[test]
    fn rejects_tampered_half_claim() {
        let s = setup(6, 4);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 6);
        let (mut proof, eval) =
            prove::<_, F, ND, NK, NW, NCU, true>(&s.params, &s.witness, &point, &cfg, &mut tp)
                .unwrap();

        let (e_e, e_o) = proof.half_claims[1][0].clone();
        proof.half_claims[1][0] = (e_e + F::one_with_cfg(&cfg), e_o);
        let (mut tv, cfg_v, point_v) = open_transcript(&proof.round_oracles[0], 6);
        let res = verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &proof, &point_v, &eval, &cfg_v, &mut tv,
        );
        assert!(res.is_err());
    }

    #[test]
    fn rejects_tampered_tail_digit() {
        let s = setup(6, 4);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 6);
        let (mut proof, eval) =
            prove::<_, F, ND, NK, NW, NCU, true>(&s.params, &s.witness, &point, &cfg, &mut tp)
                .unwrap();

        proof.tail[0][1] = add!(proof.tail[0][1], Int::<ND>::ONE);
        let (mut tv, cfg_v, point_v) = open_transcript(&proof.round_oracles[0], 6);
        let res = verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &proof, &point_v, &eval, &cfg_v, &mut tv,
        );
        assert!(res.is_err());
    }

    #[test]
    fn rejects_tampered_intermediate_oracle() {
        let s = setup(6, 4);
        let (mut tp, cfg, point) = open_transcript(&s.commitment, 6);
        let (mut proof, eval) =
            prove::<_, F, ND, NK, NW, NCU, true>(&s.params, &s.witness, &point, &cfg, &mut tp)
                .unwrap();

        // Round-1 oracle, limb 0, one position: breaks either the transcript
        // binding (challenges diverge) or the consistency check.
        proof.round_oracles[1][0][3] = add!(proof.round_oracles[1][0][3], Int::<NK>::ONE);
        let (mut tv, cfg_v, point_v) = open_transcript(&proof.round_oracles[0], 6);
        let res = verify::<_, F, ND, NK, NW, NCU, true>(
            &s.params, &proof, &point_v, &eval, &cfg_v, &mut tv,
        );
        assert!(res.is_err());
    }
}
