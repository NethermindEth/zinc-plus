//! Multi-point reducer for binary_poly columns.
//!
//! Reduces multiple polynomial-valued MLE evaluation claims at distinct
//! points on a shared batch of `binary_poly<D>` columns into a single
//! Zip+ opening at one reduced point, saving `T-1` Zip+ opens (one per
//! claim beyond the first).
//!
//! # Setting
//!
//! Let `(col_0, …, col_{n-1})` be a batch of `binary_poly<D>` MLEs over
//! `n_vars` variables — typically the witness binary_poly batch
//! committed via Zip+. Each col has degree-< D bit-poly entries; we view
//! the l-th bit of `col_j` as a 0/1-valued MLE `B_{j,l}(x)`.
//!
//! Inputs (T claims):
//! - For each `t ∈ 0..T`: a point `r^(t) ∈ F_q^{n_vars}` and a vector of
//!   polynomial-valued evals `lift_t[j] ∈ F_q[X]_{<D}`, claimed to equal
//!   `MLE[col_j](r^(t))` (i.e. `lift_t[j].coeff[l] = B_{j,l}(r^(t))`).
//!
//! # Reducer
//!
//! 1. Sample `gammas[j][l]` (n × D scalars) and `betas[t]` (T scalars)
//!    from the Fiat-Shamir transcript. (`gammas` doubles as the Zip+
//!    alpha vector for the final open.)
//! 2. Compute per-claim batched evals
//!    `y_t = Σ_{j,l} gammas[j][l] · lift_t[j].coeff[l]`.
//! 3. Compute the total `Y = Σ_t betas[t] · y_t` (the prover-supplied
//!    target). The verifier compares this against the sumcheck's own
//!    `claimed_sum` to catch tampering of any `lift_t[j]`.
//! 4. Run a degree-2 sumcheck on the identity
//!    `Y = Σ_x P(x) · M(x)` where
//!    - `P(x) = Σ_{j,l} gammas[j][l] · B_{j,l}(x)` (precombined bits-MLE)
//!    - `M(x) = Σ_t betas[t] · eq(x, r^(t))` (multi-point selector).
//! 5. The sumcheck output `(r*, v* = P(r*) · M(r*))` reduces all T
//!    claims to a single claim `P(r*) = v* / M(r*)` at point `r*`. The
//!    caller invokes `ZipPlus::verify_with_alphas` once at `r*` with
//!    `gammas` (reshaped to `[n_cols][D]`) and expected eval `P(r*)` to
//!    bind to the Zip+ commitment.
//!
//! # Soundness
//!
//! - The β-mixing folds T claims into one with negligible loss
//!   (Schwartz-Zippel over the sumcheck domain).
//! - The sumcheck reduces a multilinear identity to a single point.
//! - The γ-mixing batches `n × D` per-bit claims into one scalar so the
//!   sumcheck is on a single MLE.
//! - The final Zip+ open binds `P(r*)` to the actual committed cols,
//!   transitively binding every `lift_t[j]` (any tampered `lift_t[j]`
//!   would force `Σ γ · (lift - true)` to vanish for random γ).

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use std::marker::PhantomData;
use thiserror::Error;
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF},
    utils::{ArithErrors, build_eq_x_r_inner},
};
use zinc_transcript::{
    delegate_transcribable,
    traits::{ConstTranscribable, Transcript},
};
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

use crate::sumcheck::{MLSumcheck, SumCheckError, SumcheckProof};

/// One claim handed to the reducer: a point `r^(t)` and the prover-
/// supplied polynomial-valued evals at that point (one per col).
#[derive(Clone, Debug)]
pub struct BinClaim<F: PrimeField> {
    pub point: Vec<F>,
    /// `lifts[j]` is the polynomial-valued MLE evaluation of `col_j` at
    /// `point`, i.e. `MLE[col_j](point) ∈ F_q[X]_{<D}`.
    pub lifts: Vec<DynamicPolynomialF<F>>,
}

/// Proof emitted by `prove`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    pub sumcheck_proof: SumcheckProof<F>,
}

delegate_transcribable!(Proof<F> { sumcheck_proof: SumcheckProof<F> }
    where F: PrimeField, F::Inner: ConstTranscribable, F::Modulus: ConstTranscribable);

/// Output of `prove` / `verify`: the reduced point + per-bit-MLE alpha
/// vector + claimed `P(r*)`. The caller uses these to drive a single
/// `ZipPlus::verify_with_alphas` at `r*`.
#[derive(Clone, Debug)]
pub struct Reduced<F: PrimeField> {
    /// Reduced evaluation point.
    pub point: Vec<F>,
    /// Per-(col, coeff) batching scalar. Reshape to `[n_cols][D]` to use
    /// as `per_poly_alphas` in `ZipPlus::verify_with_alphas`.
    pub gammas_flat: Vec<F>,
    /// Claimed value `P(r*)` — the alpha-projected MLE eval at the
    /// reduced point. Used as `expected_eval` for Zip+.
    pub p_eval: F,
}

#[derive(Debug, Error)]
pub enum ReducerError<F: PrimeField> {
    #[error("sumcheck error: {0}")]
    Sumcheck(SumCheckError<F>),
    #[error("eq build error: {0}")]
    Eq(ArithErrors),
    #[error("reducer claimed_sum mismatch: got {got}, expected {expected}")]
    ClaimedSumMismatch { got: F, expected: F },
    #[error("M(r*) is zero — degenerate reducer instance")]
    ZeroMSelector,
}

impl<F: PrimeField> From<SumCheckError<F>> for ReducerError<F> {
    fn from(e: SumCheckError<F>) -> Self {
        Self::Sumcheck(e)
    }
}

impl<F: PrimeField> From<ArithErrors> for ReducerError<F> {
    fn from(e: ArithErrors) -> Self {
        Self::Eq(e)
    }
}

pub struct BinMultipointReducer<F, const D: usize>(PhantomData<F>);

impl<F, const D: usize> BinMultipointReducer<F, D>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    /// Run the reducer prover.
    ///
    /// `claims[t].lifts[j]` MUST equal `MLE[bin_cols[j]](claims[t].point)`
    /// in `F_q[X]_{<D}` for an honest prover.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn prove(
        transcript: &mut impl Transcript,
        bin_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
        claims: &[BinClaim<F>],
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, Reduced<F>), ReducerError<F>> {
        assert!(!bin_cols.is_empty(), "reducer needs at least one bin col");
        assert!(!claims.is_empty(), "reducer needs at least one claim");
        let n_cols = bin_cols.len();
        let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();

        let n_gamma = n_cols * D;
        let gammas_flat: Vec<F> = transcript.get_field_challenges(n_gamma, field_cfg);
        let betas: Vec<F> = transcript.get_field_challenges(claims.len(), field_cfg);

        let zero = F::zero_with_cfg(field_cfg);

        // Build P(x) = Σ_{j,l} γ_{j,l} · B_{j,l}(x) at hypercube.
        let p_evals: Vec<F::Inner> = cfg_into_iter!(0..1usize << num_vars)
            .map(|x_idx| {
                let mut s = zero.clone();
                for j in 0..n_cols {
                    let bp = &bin_cols[j].evaluations[x_idx];
                    let coeffs = bp.inner().coeffs;
                    for l in 0..D {
                        if coeffs[l].into_inner() {
                            s = s + &gammas_flat[j * D + l];
                        }
                    }
                }
                s.into_inner()
            })
            .collect();
        let p_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            p_evals,
            zero_inner.clone(),
        );

        // Build M(x) = Σ_t β_t · eq(x, r^(t)) at hypercube.
        let mut m_evals_f: Vec<F> = vec![zero.clone(); 1usize << num_vars];
        for (t, claim) in claims.iter().enumerate() {
            let eq_t = build_eq_x_r_inner::<F>(&claim.point, field_cfg)?;
            for (acc, e) in m_evals_f.iter_mut().zip(eq_t.evaluations.iter()) {
                let e_f = F::new_unchecked_with_cfg(e.clone(), field_cfg);
                *acc = acc.clone() + &(betas[t].clone() * &e_f);
            }
        }
        let m_evals: Vec<F::Inner> = m_evals_f.into_iter().map(F::into_inner).collect();
        let m_mle = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            m_evals,
            zero_inner,
        );

        // Sumcheck on P · M (degree 2).
        let mles = vec![p_mle.clone(), m_mle];
        let (sumcheck_proof, sumcheck_state) = MLSumcheck::prove_as_subprotocol(
            transcript,
            mles,
            num_vars,
            2,
            |v: &[F]| v[0].clone() * &v[1],
            field_cfg,
        );

        let r_star = sumcheck_state.randomness.clone();

        // P(r*) honest evaluation: evaluate p_mle at r*.
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
    ///
    /// `claims[t].lifts[j]` are the prover-supplied polynomial-valued
    /// evals — the verifier batches them into per-claim scalar evals
    /// using the same γ as the prover, and folds them via β.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    pub fn verify(
        transcript: &mut impl Transcript,
        proof: &Proof<F>,
        claims: &[BinClaim<F>],
        n_cols: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<Reduced<F>, ReducerError<F>> {
        assert!(!claims.is_empty(), "reducer needs at least one claim");
        let n_gamma = n_cols * D;
        let gammas_flat: Vec<F> = transcript.get_field_challenges(n_gamma, field_cfg);
        let betas: Vec<F> = transcript.get_field_challenges(claims.len(), field_cfg);

        let zero = F::zero_with_cfg(field_cfg);
        let y_t: Vec<F> = claims
            .iter()
            .map(|c| {
                assert_eq!(c.lifts.len(), n_cols);
                let mut s = zero.clone();
                for (j, lift) in c.lifts.iter().enumerate() {
                    for (l, coeff) in lift.coeffs.iter().enumerate() {
                        s = s + &(gammas_flat[j * D + l].clone() * coeff);
                    }
                }
                s
            })
            .collect();
        let total: F = betas
            .iter()
            .zip(y_t.iter())
            .fold(zero.clone(), |acc, (b, y)| acc + &(b.clone() * y));

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
        let m_at_r_star = m_evaluation_at_point(claims, &betas, &r_star, field_cfg)?;
        if m_at_r_star == zero {
            return Err(ReducerError::ZeroMSelector);
        }
        // P(r*) = sub.expected_evaluation / M(r*).
        let one = F::one_with_cfg(field_cfg);
        let m_inv = one / &m_at_r_star;
        let p_at_r_star = sub.expected_evaluation.clone() * &m_inv;

        Ok(Reduced {
            point: r_star,
            gammas_flat,
            p_eval: p_at_r_star,
        })
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn m_evaluation_at_point<F: PrimeField>(
    claims: &[BinClaim<F>],
    betas: &[F],
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<F, ArithErrors> {
    let one = F::one_with_cfg(field_cfg);
    let zero = F::zero_with_cfg(field_cfg);
    let mut s = zero;
    for (claim, beta) in claims.iter().zip(betas.iter()) {
        let eq = zinc_poly::utils::eq_eval(r_star, &claim.point, one.clone())?;
        s = s + &(beta.clone() * &eq);
    }
    Ok(s)
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
    use zinc_transcript::Blake3Transcript;

    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, { U128::LIMBS }>;
    const D: usize = 32;

    fn rand_bin_col(
        n_vars: usize,
        rng: &mut impl RngCore,
    ) -> DenseMultilinearExtension<BinaryPoly<D>> {
        let len = 1usize << n_vars;
        let evals: Vec<BinaryPoly<D>> =
            (0..len).map(|_| BinaryPoly::<D>::from(rng.next_u32())).collect();
        DenseMultilinearExtension::from_evaluations_vec(n_vars, evals, BinaryPoly::<D>::zero())
    }

    fn col_lift_at(
        col: &DenseMultilinearExtension<BinaryPoly<D>>,
        point: &[F],
        cfg: &<F as PrimeField>::Config,
    ) -> DynamicPolynomialF<F> {
        let zero = F::zero_with_cfg(cfg);
        let eq = zinc_poly::utils::build_eq_x_r_vec(point, cfg).unwrap();
        let mut coeffs = vec![zero; D];
        for (i, entry) in col.iter().enumerate() {
            for (l, c) in entry.inner().coeffs.iter().enumerate() {
                if c.into_inner() {
                    coeffs[l] = coeffs[l].clone() + &eq[i];
                }
            }
        }
        DynamicPolynomialF::new_trimmed(coeffs)
    }

    #[test]
    fn round_trip_t2_n3() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(31);
        let n_vars = 6;
        let cols: Vec<_> = (0..3).map(|_| rand_bin_col(n_vars, &mut rng)).collect();

        let r1: Vec<F> = (0..n_vars).map(|_| F::from(rng.next_u64())).collect();
        let r2: Vec<F> = (0..n_vars).map(|_| F::from(rng.next_u64())).collect();
        let claim1 = BinClaim {
            point: r1.clone(),
            lifts: cols.iter().map(|c| col_lift_at(c, &r1, &cfg)).collect(),
        };
        let claim2 = BinClaim {
            point: r2.clone(),
            lifts: cols.iter().map(|c| col_lift_at(c, &r2, &cfg)).collect(),
        };
        let claims = vec![claim1, claim2];

        let mut p_ts = Blake3Transcript::new();
        let (proof, p_red) =
            BinMultipointReducer::<F, D>::prove(&mut p_ts, &cols, &claims, n_vars, &cfg)
                .expect("prove");

        let mut v_ts = Blake3Transcript::new();
        let v_red = BinMultipointReducer::<F, D>::verify(
            &mut v_ts, &proof, &claims, cols.len(), n_vars, &cfg,
        )
        .expect("verify");

        assert_eq!(p_red.point, v_red.point);
        assert_eq!(p_red.gammas_flat, v_red.gammas_flat);
        assert_eq!(p_red.p_eval, v_red.p_eval);
    }

    /// Tampering one of the prover-supplied lifts must cause the
    /// reducer's `claimed_sum` (over true cols) to disagree with the
    /// verifier-recomputed `total` (over tampered lifts).
    #[test]
    fn tampered_lift_rejected() {
        let cfg = ();
        let mut rng = StdRng::seed_from_u64(17);
        let n_vars = 5;
        let cols: Vec<_> = (0..2).map(|_| rand_bin_col(n_vars, &mut rng)).collect();
        let r1: Vec<F> = (0..n_vars).map(|_| F::from(rng.next_u64())).collect();
        let mut claim = BinClaim {
            point: r1.clone(),
            lifts: cols.iter().map(|c| col_lift_at(c, &r1, &cfg)).collect(),
        };

        // Tamper one coefficient.
        claim.lifts[0].coeffs.swap(0, 1);
        let claims = vec![claim];

        let mut p_ts = Blake3Transcript::new();
        let (proof, _) =
            BinMultipointReducer::<F, D>::prove(&mut p_ts, &cols, &claims, n_vars, &cfg)
                .expect("prove");

        let mut v_ts = Blake3Transcript::new();
        let res = BinMultipointReducer::<F, D>::verify(
            &mut v_ts, &proof, &claims, cols.len(), n_vars, &cfg,
        );
        assert!(matches!(
            res,
            Err(ReducerError::ClaimedSumMismatch { .. })
        ));
    }
}
