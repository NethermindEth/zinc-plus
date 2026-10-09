//! Multi-point reducer for integer columns.
//!
//! The [`BinMultipointReducer`](crate::bin_multipoint_reducer) for the
//! integer batch. Same reduction, one coefficient per column instead of
//! `D`: an integer cell is a number, so its lifted evaluation is a
//! degree-zero polynomial and the γ-mixing is one scalar per column
//! rather than one per bit.
//!
//! # Setting
//!
//! Let `(col_0, …, col_{n-1})` be the ψ_α-projected witness integer MLEs
//! over `n_vars` variables — the batch committed via Zip+. Inputs are `T`
//! claims: for each `t`, a point `r^(t)` and per-column lifted
//! evaluations `lift_t[j]`, claimed to equal `MLE[col_j](r^(t))`.
//!
//! # Reducer
//!
//! 1. Sample `gammas[j]` (n scalars) and `betas[t]` (T scalars).
//! 2. Run a degree-2 sumcheck on `Y = Σ_x P(x) · M(x)` where
//!    `P(x) = Σ_j gammas[j] · col_j(x)` and
//!    `M(x) = Σ_t betas[t] · eq(x, r^(t))`, against the prover-supplied
//!    target `Y = Σ_t betas[t] · Σ_j gammas[j] · lift_t[j]`.
//! 3. The output `(r*, v*)` reduces all `T` claims to `P(r*) = v* / M(r*)`
//!    at the single point `r*`, which the caller binds to the Zip+
//!    commitment with one opening.
//!
//! Soundness is the binary reducer's, coefficient count aside: β folds
//! the `T` claims, the sumcheck reduces the identity to a point, γ
//! batches the per-column claims, and the single Zip+ open binds `P(r*)`
//! to the committed columns and so every `lift_t[j]` through it.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use std::marker::PhantomData;
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    utils::build_eq_x_r_inner,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField};

use crate::{
    bin_multipoint_reducer::{BinClaim, Proof, Reduced, ReducerError},
    sumcheck::MLSumcheck,
};

pub struct IntMultipointReducer<F>(PhantomData<F>);

impl<F> IntMultipointReducer<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    /// Run the reducer prover.
    ///
    /// `claims[t].lifts[j]` MUST equal `MLE[int_cols[j]](claims[t].point)`
    /// for an honest prover, carried as a one-coefficient polynomial.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove(
        transcript: &mut impl Transcript,
        int_cols: &[DenseMultilinearExtension<F::Inner>],
        claims: &[BinClaim<F>],
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, Reduced<F>), ReducerError<F>> {
        assert!(!int_cols.is_empty(), "reducer needs at least one int col");
        assert!(!claims.is_empty(), "reducer needs at least one claim");
        let n_cols = int_cols.len();
        let zero = F::zero_with_cfg(field_cfg);
        let zero_inner = zero.inner().clone();

        let gammas_flat: Vec<F> = transcript.get_field_challenges(n_cols, field_cfg);
        let betas: Vec<F> = transcript.get_field_challenges(claims.len(), field_cfg);

        // P(x) = Σ_j γ_j · col_j(x) at the hypercube.
        let n_hyper = 1usize << num_vars;
        let p_evals: Vec<F::Inner> = cfg_into_iter!(0..n_hyper)
            .map(|x_idx| {
                let mut s = zero.clone();
                for (j, col) in int_cols.iter().enumerate() {
                    let v = F::new_unchecked_with_cfg(col.evaluations[x_idx].clone(), field_cfg);
                    s = s + &(gammas_flat[j].clone() * &v);
                }
                s.into_inner()
            })
            .collect();
        let p_mle =
            DenseMultilinearExtension::from_evaluations_vec(num_vars, p_evals, zero_inner.clone());

        // M(x) = Σ_t β_t · eq(x, r^(t)) at the hypercube.
        let eq_tables: Vec<DenseMultilinearExtension<F::Inner>> = cfg_iter!(claims)
            .map(|c| build_eq_x_r_inner::<F>(&c.point, field_cfg).expect("eq build"))
            .collect();
        let m_evals: Vec<F::Inner> = cfg_into_iter!(0..n_hyper)
            .map(|x_idx| {
                let mut s = zero.clone();
                for (t, eq_t) in eq_tables.iter().enumerate() {
                    let e_f =
                        F::new_unchecked_with_cfg(eq_t.evaluations[x_idx].clone(), field_cfg);
                    s = s + &(betas[t].clone() * &e_f);
                }
                s.into_inner()
            })
            .collect();
        let m_mle = DenseMultilinearExtension::from_evaluations_vec(num_vars, m_evals, zero_inner);

        let (sumcheck_proof, sumcheck_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            vec![p_mle.clone(), m_mle],
            num_vars,
            2,
            |v: &[F]| v[0].clone() * &v[1],
            field_cfg,
        );

        let r_star = sumcheck_state.randomness.clone();
        let p_at_r_star = p_mle
            .evaluate_with_config(&r_star, field_cfg)
            .expect("p_mle eval at r*");

        Ok((
            Proof { sumcheck_proof },
            Reduced {
                point: r_star,
                gammas_flat,
                p_eval: p_at_r_star,
            },
        ))
    }

    /// Run the reducer verifier.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify(
        transcript: &mut impl Transcript,
        proof: &Proof<F>,
        claims: &[BinClaim<F>],
        n_cols: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Reduced<F>, ReducerError<F>> {
        assert!(!claims.is_empty(), "reducer needs at least one claim");
        let gammas_flat: Vec<F> = transcript.get_field_challenges(n_cols, field_cfg);
        let betas: Vec<F> = transcript.get_field_challenges(claims.len(), field_cfg);

        let zero = F::zero_with_cfg(field_cfg);
        let mut total = zero.clone();
        for (index, (claim, beta)) in claims.iter().zip(betas.iter()).enumerate() {
            if claim.lifts.len() != n_cols {
                return Err(ReducerError::MalformedClaim { index });
            }
            let mut y = zero.clone();
            for (gamma, lift) in gammas_flat.iter().zip(claim.lifts.iter()) {
                // An integer column's lifted evaluation is one coefficient;
                // `new_trimmed` drops it entirely when it is zero.
                match lift.coeffs.as_slice() {
                    [] => {}
                    [c] => y = y + &(gamma.clone() * c),
                    _ => return Err(ReducerError::MalformedClaim { index }),
                }
            }
            total = total + &(beta.clone() * &y);
        }

        if proof.sumcheck_proof.claimed_sum != total {
            return Err(ReducerError::ClaimedSumMismatch {
                got: proof.sumcheck_proof.claimed_sum.clone(),
                expected: total,
            });
        }

        let sub = MLSumcheck::verify_as_subprotocol(
            transcript,
            num_vars,
            2,
            &proof.sumcheck_proof,
            field_cfg,
        )?;

        let r_star = sub.point.clone();
        let one = F::one_with_cfg(field_cfg);
        let mut m_at_r_star = zero.clone();
        for (claim, beta) in claims.iter().zip(betas.iter()) {
            let eq = zinc_poly::utils::eq_eval(&r_star, &claim.point, one.clone())?;
            m_at_r_star = m_at_r_star + &(beta.clone() * &eq);
        }
        if m_at_r_star == zero {
            return Err(ReducerError::ZeroMSelector);
        }
        let p_at_r_star = sub.expected_evaluation.clone() * &(one / &m_at_r_star);

        Ok(Reduced {
            point: r_star,
            gammas_flat,
            p_eval: p_at_r_star,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;

    fn rand_int_col(
        n_vars: usize,
        rng: &mut impl RngCore,
    ) -> DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner> {
        let evals: Vec<_> = (0..1usize << n_vars)
            .map(|_| F::from(rng.next_u64()).into_inner())
            .collect();
        DenseMultilinearExtension::from_evaluations_vec(
            n_vars,
            evals,
            F::zero_with_cfg(&()).into_inner(),
        )
    }

    fn col_lift_at(
        col: &DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>,
        point: &[F],
        cfg: &<F as PrimeField>::Config,
    ) -> DynamicPolynomialF<F> {
        DynamicPolynomialF::new_trimmed(vec![
            col.clone().evaluate_with_config(point, cfg).unwrap(),
        ])
    }

    /// Three claims at three points over four columns reduce to the one
    /// point and the one value both sides agree on.
    #[test]
    fn round_trip_t3_n4() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(7);
        let n_vars = 6;
        let cols: Vec<_> = (0..4).map(|_| rand_int_col(n_vars, &mut rng)).collect();

        let claims: Vec<BinClaim<F>> = (0..3)
            .map(|_| {
                let r: Vec<F> = (0..n_vars).map(|_| F::from(rng.next_u64())).collect();
                BinClaim {
                    lifts: cols.iter().map(|c| col_lift_at(c, &r, &cfg)).collect(),
                    point: r,
                }
            })
            .collect();

        let mut p_ts = Blake3Transcript::new();
        let (proof, p_red) =
            IntMultipointReducer::<F>::prove(&mut p_ts, &cols, &claims, n_vars, &cfg)
                .expect("prove");

        let mut v_ts = Blake3Transcript::new();
        let v_red = IntMultipointReducer::<F>::verify(
            &mut v_ts,
            &proof,
            &claims,
            cols.len(),
            n_vars,
            &cfg,
        )
        .expect("verify");

        assert_eq!(p_red.point, v_red.point);
        assert_eq!(p_red.gammas_flat, v_red.gammas_flat);
        assert_eq!(p_red.p_eval, v_red.p_eval);
    }

    /// A lift the prover did not prove disagrees with the sumcheck's own
    /// claimed sum, so the verifier catches it before the sumcheck runs.
    #[test]
    fn tampered_lift_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(11);
        let n_vars = 5;
        let cols: Vec<_> = (0..3).map(|_| rand_int_col(n_vars, &mut rng)).collect();
        let r: Vec<F> = (0..n_vars).map(|_| F::from(rng.next_u64())).collect();
        let honest = BinClaim {
            lifts: cols.iter().map(|c| col_lift_at(c, &r, &cfg)).collect(),
            point: r,
        };

        let mut p_ts = Blake3Transcript::new();
        let (proof, _) = IntMultipointReducer::<F>::prove(
            &mut p_ts,
            &cols,
            std::slice::from_ref(&honest),
            n_vars,
            &cfg,
        )
        .expect("prove");

        let mut tampered = honest;
        let c = &mut tampered.lifts[1].coeffs[0];
        *c = c.clone() + c.clone();

        let mut v_ts = Blake3Transcript::new();
        let res = IntMultipointReducer::<F>::verify(
            &mut v_ts,
            &proof,
            &[tampered],
            cols.len(),
            n_vars,
            &cfg,
        );
        assert!(matches!(res, Err(ReducerError::ClaimedSumMismatch { .. })));
    }
}
