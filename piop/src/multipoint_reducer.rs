//! Multi-point MLE claim reducer.
//!
//! Given a set of trace MLEs `v_0, …, v_{C-1}` over `n` variables and a
//! set of claims `{v_j(r^(l)) = a_{l,j}}` at `L` unrelated points
//! `r^(1), …, r^(L)`, reduces them to a single claim
//! `{v_j(r_final) = c_j}` at a fresh point `r_final` via one
//! sumcheck of degree 2.
//!
//! # Protocol
//!
//! 1. Verifier samples per-point column batching coefficients
//!    `γ_{l,j}` (one per (point, column) pair).
//! 2. Define the combined claim
//!
//!    ```text
//!    S = Σ_{l,j} γ_{l,j} · a_{l,j}
//!    ```
//!
//!    which, by the MLE identity, equals
//!
//!    ```text
//!    S = Σ_b [ Σ_l eq(b, r^(l)) · Σ_j γ_{l,j} · v_j(b) ]
//!    ```
//!
//!    Each inner `Σ_j γ_{l,j} · v_j(b)` is precomputed as a single
//!    combined MLE `P_l`, giving the sumcheck summand
//!    `Σ_l eq(b, r^(l)) · P_l(b)` of degree 2.
//!
//! 3. Prover runs the sumcheck; at the end, both sides know a fresh
//!    point `r_final` and the sumcheck's expected evaluation.
//! 4. Prover sends `v_j(r_final)` for each column; the verifier checks
//!    that the reconstructed sum
//!    `Σ_l eq(r_final, r^(l)) · Σ_j γ_{l,j} · v_j(r_final)` matches
//!    the sumcheck's expected evaluation.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    utils::{ArithErrors, build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

use crate::sumcheck::{MLSumcheck, SumCheckError, SumcheckProof};

/// A single claim group: one point and one value per column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiClaim<F> {
    pub point: Vec<F>,
    /// Claimed MLE evaluations `v_j(point)` at this point, indexed by
    /// the column order the caller uses consistently across claims.
    pub evals: Vec<F>,
}

/// Proof of the multi-point reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiPointReduceProof<F> {
    pub sumcheck_proof: SumcheckProof<F>,
    /// `v_j(r_final)` for each column j. Carried in the proof since
    /// the verifier needs them to finalize the reduction check.
    pub tail_evals: Vec<F>,
}

/// Subclaim returned to the caller after a successful verification.
#[derive(Clone, Debug)]
pub struct MultiPointReduceSubclaim<F> {
    /// Fresh reduction point.
    pub r_final: Vec<F>,
    /// `v_j(r_final)` — the reduced claim set.
    pub evals: Vec<F>,
}

#[derive(Debug, Error)]
pub enum MultiPointReduceError<F> {
    #[error("arithmetic error: {0}")]
    Arith(#[from] ArithErrors),
    #[error("sumcheck failed: {0}")]
    Sumcheck(#[from] SumCheckError<F>),
    #[error("expected sum mismatch: got {got:?}, expected {expected:?}")]
    ExpectedSumMismatch { got: Box<F>, expected: Box<F> },
    #[error("final reconstruction check failed")]
    FinalEvalMismatch,
    #[error("invalid shape: {0}")]
    InvalidShape(&'static str),
}

pub struct MultiPointReducer<F>(PhantomData<F>);

impl<F> MultiPointReducer<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + PrimeField + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    /// Reduce claims at `L` unrelated points to a single claim at a
    /// fresh point `r_final`.
    ///
    /// All `claims` must agree on `evals.len() == trace_mles.len()`
    /// and all `point.len() == num_vars`. `trace_mles` must all share
    /// `num_vars`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove(
        transcript: &mut impl Transcript,
        trace_mles: &[DenseMultilinearExtension<F::Inner>],
        claims: &[MultiClaim<F>],
        cfg: &F::Config,
    ) -> Result<(MultiPointReduceProof<F>, MultiPointReduceSubclaim<F>), MultiPointReduceError<F>>
    {
        assert!(!trace_mles.is_empty(), "need at least one trace MLE");
        assert!(!claims.is_empty(), "need at least one claim group");

        let num_cols = trace_mles.len();
        let num_vars = trace_mles[0].num_vars;
        for mle in trace_mles {
            assert_eq!(mle.num_vars, num_vars, "trace MLEs must share num_vars");
        }
        for claim in claims {
            assert_eq!(claim.point.len(), num_vars, "claim point dim mismatch");
            assert_eq!(claim.evals.len(), num_cols, "claim evals length mismatch");
        }

        // Sample γ_{l,j}: flattened (L * C) challenges, row-major by
        // point then column so that `gammas[l * C + j] = γ_{l,j}`.
        let total_coeffs = claims.len() * num_cols;
        let gammas: Vec<F> = transcript.get_field_challenges(total_coeffs, cfg);

        // Build per-point eq MLEs and per-point precombined MLEs.
        let zero = F::zero_with_cfg(cfg);
        let zero_inner = zero.inner();

        let mut eq_mles: Vec<DenseMultilinearExtension<F::Inner>> = Vec::with_capacity(claims.len());
        let mut precombined_mles: Vec<DenseMultilinearExtension<F::Inner>> =
            Vec::with_capacity(claims.len());

        for (l, claim) in claims.iter().enumerate() {
            let eq_l = build_eq_x_r_inner::<F>(&claim.point, cfg)?;
            eq_mles.push(eq_l);

            // precombined_l[b] = Σ_j γ_{l,j} * v_j(b).
            let base = l * num_cols;
            let evaluations: Vec<F::Inner> = cfg_into_iter!(0..1usize << num_vars)
                .map(|b| {
                    let mut acc = zero.clone();
                    for j in 0..num_cols {
                        let v_eval = F::new_unchecked_with_cfg(
                            trace_mles[j].evaluations[b].clone(),
                            cfg,
                        );
                        acc = acc + gammas[base + j].clone() * &v_eval;
                    }
                    acc.into_inner()
                })
                .collect();
            precombined_mles.push(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evaluations,
                zero_inner.clone(),
            ));
        }

        // Pack MLEs interleaved: [eq_0, P_0, eq_1, P_1, ...].
        let mut mles = Vec::with_capacity(2 * claims.len());
        for (eq, pc) in eq_mles.into_iter().zip(precombined_mles.into_iter()) {
            mles.push(eq);
            mles.push(pc);
        }

        let num_groups = claims.len();

        // Combiner: Σ_l v[2l] * v[2l+1]. Starts from the first product
        // so we never need to synthesize a zero F from scratch inside
        // the closure (which would require capturing cfg and bumps
        // against the `'static` bound required by the sumcheck).
        let comb_fn = move |vals: &[F]| -> F {
            let mut acc = vals[0].clone() * &vals[1];
            for l in 1..num_groups {
                acc = acc + vals[2 * l].clone() * &vals[2 * l + 1];
            }
            acc
        };

        let (sumcheck_proof, sumcheck_state) =
            MLSumcheck::prove_as_subprotocol(transcript, mles, num_vars, 2, comb_fn, cfg);

        let r_final: Vec<F> = sumcheck_state.randomness.clone();

        // Prover sends v_j(r_final) for each column; clone-evaluate.
        let tail_evals: Vec<F> = trace_mles
            .iter()
            .map(|mle| {
                mle.clone()
                    .evaluate_with_config(&r_final, cfg)
                    .expect("dimension checked above")
            })
            .collect();

        // Absorb tail_evals so later challenges in the outer protocol
        // bind to them.
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        for v in &tail_evals {
            transcript.absorb_random_field(v, &mut buf);
        }

        Ok((
            MultiPointReduceProof {
                sumcheck_proof,
                tail_evals: tail_evals.clone(),
            },
            MultiPointReduceSubclaim {
                r_final,
                evals: tail_evals,
            },
        ))
    }

    /// Verify a multi-point reduction.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify(
        transcript: &mut impl Transcript,
        proof: &MultiPointReduceProof<F>,
        claims: &[MultiClaim<F>],
        num_cols: usize,
        num_vars: usize,
        cfg: &F::Config,
    ) -> Result<MultiPointReduceSubclaim<F>, MultiPointReduceError<F>> {
        if claims.is_empty() {
            return Err(MultiPointReduceError::InvalidShape("need at least one claim"));
        }
        if proof.tail_evals.len() != num_cols {
            return Err(MultiPointReduceError::InvalidShape("tail_evals length mismatch"));
        }
        for claim in claims {
            if claim.point.len() != num_vars || claim.evals.len() != num_cols {
                return Err(MultiPointReduceError::InvalidShape("claim shape mismatch"));
            }
        }

        // Match prover's challenge sampling exactly.
        let total_coeffs = claims.len() * num_cols;
        let gammas: Vec<F> = transcript.get_field_challenges(total_coeffs, cfg);

        // Expected sum: Σ_{l,j} γ_{l,j} * a_{l,j}.
        let zero = F::zero_with_cfg(cfg);
        let mut expected_sum = zero.clone();
        for (l, claim) in claims.iter().enumerate() {
            let base = l * num_cols;
            for (j, a) in claim.evals.iter().enumerate() {
                expected_sum = expected_sum + gammas[base + j].clone() * a;
            }
        }

        if proof.sumcheck_proof.claimed_sum != expected_sum {
            return Err(MultiPointReduceError::ExpectedSumMismatch {
                got: Box::new(proof.sumcheck_proof.claimed_sum.clone()),
                expected: Box::new(expected_sum),
            });
        }

        let sumcheck_sub = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            2,
            &proof.sumcheck_proof,
            cfg,
        )?;

        let r_final = &sumcheck_sub.point;

        // Reconstruct the sumcheck's expected evaluation from tail_evals.
        //
        //  Σ_l eq(r_final, r^(l)) · Σ_j γ_{l,j} · v_j(r_final)
        let one = F::from_with_cfg(1u64, cfg);
        let mut reconstructed = zero.clone();
        for (l, claim) in claims.iter().enumerate() {
            let eq_l = eq_eval(r_final, &claim.point, one.clone())?;
            let base = l * num_cols;
            let mut inner = zero.clone();
            for j in 0..num_cols {
                inner = inner + gammas[base + j].clone() * &proof.tail_evals[j];
            }
            reconstructed = reconstructed + eq_l * &inner;
        }

        if reconstructed != sumcheck_sub.expected_evaluation {
            return Err(MultiPointReduceError::FinalEvalMismatch);
        }

        // Absorb tail_evals (matching prover order).
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        for v in &proof.tail_evals {
            transcript.absorb_random_field(v, &mut buf);
        }

        Ok(MultiPointReduceSubclaim {
            r_final: r_final.clone(),
            evals: proof.tail_evals.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::{Field, crypto_bigint_const_monty::ConstMontyField};
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;

    fn random_mle(num_vars: usize, rng: &mut impl RngCore) -> DenseMultilinearExtension<<F as Field>::Inner> {
        let size = 1usize << num_vars;
        let zero_inner = F::zero_with_cfg(&()).into_inner();
        let evals: Vec<_> = (0..size).map(|_| F::from(rng.next_u64()).into_inner()).collect();
        DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
    }

    fn lift(mle: &DenseMultilinearExtension<<F as Field>::Inner>) -> DenseMultilinearExtension<F> {
        let cfg = ();
        let evals: Vec<F> = mle
            .evaluations
            .iter()
            .map(|x| F::new_unchecked_with_cfg(x.clone(), &cfg))
            .collect();
        DenseMultilinearExtension::from_evaluations_vec(mle.num_vars, evals, F::zero_with_cfg(&cfg))
    }

    fn random_point(num_vars: usize, rng: &mut impl RngCore) -> Vec<F> {
        (0..num_vars).map(|_| F::from(rng.next_u64())).collect()
    }

    /// Build claims for `trace_mles` at `points` by direct evaluation.
    fn make_claims(
        trace_mles: &[DenseMultilinearExtension<<F as Field>::Inner>],
        points: &[Vec<F>],
    ) -> Vec<MultiClaim<F>> {
        let cfg = ();
        points
            .iter()
            .map(|p| {
                let evals: Vec<F> = trace_mles
                    .iter()
                    .map(|m| lift(m).evaluate(p, F::zero_with_cfg(&cfg)).unwrap())
                    .collect();
                MultiClaim {
                    point: p.clone(),
                    evals,
                }
            })
            .collect()
    }

    #[test]
    fn reduce_two_points_roundtrip() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(0xABCD);
        let num_vars = 3;
        let num_cols = 4;

        let trace_mles: Vec<_> = (0..num_cols).map(|_| random_mle(num_vars, &mut rng)).collect();
        let points = vec![
            random_point(num_vars, &mut rng),
            random_point(num_vars, &mut rng),
        ];
        let claims = make_claims(&trace_mles, &points);

        let mut p_ts = Blake3Transcript::new();
        let (proof, _prover_sub) =
            MultiPointReducer::<F>::prove(&mut p_ts, &trace_mles, &claims, &cfg).unwrap();

        let mut v_ts = Blake3Transcript::new();
        let sub = MultiPointReducer::<F>::verify(
            &mut v_ts, &proof, &claims, num_cols, num_vars, &cfg,
        )
        .expect("verify must succeed");

        // Tail evals must equal the true evaluations at r_final.
        for (j, claim_eval) in sub.evals.iter().enumerate() {
            let direct = lift(&trace_mles[j])
                .evaluate(&sub.r_final, F::zero_with_cfg(&cfg))
                .unwrap();
            assert_eq!(*claim_eval, direct, "column {j} tail_eval mismatch");
        }
    }

    #[test]
    fn reduce_single_point_roundtrip() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(7);
        let num_vars = 4;
        let num_cols = 2;

        let trace_mles: Vec<_> = (0..num_cols).map(|_| random_mle(num_vars, &mut rng)).collect();
        let points = vec![random_point(num_vars, &mut rng)];
        let claims = make_claims(&trace_mles, &points);

        let mut p_ts = Blake3Transcript::new();
        let (proof, _) =
            MultiPointReducer::<F>::prove(&mut p_ts, &trace_mles, &claims, &cfg).unwrap();

        let mut v_ts = Blake3Transcript::new();
        let res = MultiPointReducer::<F>::verify(
            &mut v_ts, &proof, &claims, num_cols, num_vars, &cfg,
        );
        assert!(res.is_ok(), "single-point reduction must succeed");
    }

    #[test]
    fn reduce_three_points_roundtrip() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(42);
        let num_vars = 3;
        let num_cols = 3;

        let trace_mles: Vec<_> = (0..num_cols).map(|_| random_mle(num_vars, &mut rng)).collect();
        let points = vec![
            random_point(num_vars, &mut rng),
            random_point(num_vars, &mut rng),
            random_point(num_vars, &mut rng),
        ];
        let claims = make_claims(&trace_mles, &points);

        let mut p_ts = Blake3Transcript::new();
        let (proof, _) =
            MultiPointReducer::<F>::prove(&mut p_ts, &trace_mles, &claims, &cfg).unwrap();

        let mut v_ts = Blake3Transcript::new();
        MultiPointReducer::<F>::verify(&mut v_ts, &proof, &claims, num_cols, num_vars, &cfg)
            .expect("3-point reduction must succeed");
    }

    #[test]
    fn tampered_claim_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(99);
        let num_vars = 3;
        let num_cols = 2;

        let trace_mles: Vec<_> = (0..num_cols).map(|_| random_mle(num_vars, &mut rng)).collect();
        let points = vec![random_point(num_vars, &mut rng), random_point(num_vars, &mut rng)];
        let mut claims = make_claims(&trace_mles, &points);

        let mut p_ts = Blake3Transcript::new();
        let (proof, _) =
            MultiPointReducer::<F>::prove(&mut p_ts, &trace_mles, &claims, &cfg).unwrap();

        // Tamper a claimed eval AFTER proving — verifier must reject.
        claims[0].evals[0] = claims[0].evals[0].clone() + F::from(1u64);

        let mut v_ts = Blake3Transcript::new();
        let res =
            MultiPointReducer::<F>::verify(&mut v_ts, &proof, &claims, num_cols, num_vars, &cfg);
        match res {
            Err(MultiPointReduceError::ExpectedSumMismatch { .. }) => {}
            other => panic!("expected ExpectedSumMismatch, got {other:?}"),
        }
    }

    #[test]
    fn tampered_tail_eval_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(1234);
        let num_vars = 3;
        let num_cols = 2;

        let trace_mles: Vec<_> = (0..num_cols).map(|_| random_mle(num_vars, &mut rng)).collect();
        let points = vec![random_point(num_vars, &mut rng), random_point(num_vars, &mut rng)];
        let claims = make_claims(&trace_mles, &points);

        let mut p_ts = Blake3Transcript::new();
        let (mut proof, _) =
            MultiPointReducer::<F>::prove(&mut p_ts, &trace_mles, &claims, &cfg).unwrap();

        // Tamper a tail eval — reconstruction check must fail.
        proof.tail_evals[0] = proof.tail_evals[0].clone() + F::from(1u64);

        let mut v_ts = Blake3Transcript::new();
        let res =
            MultiPointReducer::<F>::verify(&mut v_ts, &proof, &claims, num_cols, num_vars, &cfg);
        match res {
            Err(MultiPointReduceError::FinalEvalMismatch) => {}
            other => panic!("expected FinalEvalMismatch, got {other:?}"),
        }
    }
}
