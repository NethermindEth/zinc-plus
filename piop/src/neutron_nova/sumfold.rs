use crypto_primitives::PrimeField;
use thiserror::Error;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{ArithErrors, build_eq_x_r_inner, build_eq_x_r_vec, eq_eval},
};
use zinc_utils::{
    delayed_reduction::DelayedFieldProductSum, inner_transparent_field::InnerTransparentField,
};

use crate::sumcheck::{
    multi_degree::{MultiDegreeSumcheckGroup, PrefixFastPath, PrefixRoundOutput},
    prover::ProverState as SumcheckProverState,
};

/// Errors produced by linear SumFold prefix-table helpers.
#[derive(Clone, Debug, Error)]
pub enum SumFoldError {
    #[error("linear instance claims cannot be empty")]
    EmptyClaims,
    #[error("instance count must be a power of two, got {len}")]
    InstanceCountNotPowerOfTwo { len: usize },
    #[error("instance count mismatch for ell={ell}: got {got}, expected {expected}")]
    InstanceCountMismatch {
        ell: usize,
        got: usize,
        expected: usize,
    },
    #[error("domain size is too large for ell={ell}")]
    DomainTooLarge { ell: usize },
    #[error("ell0={ell0} must be at most ell={ell}")]
    Ell0TooLarge { ell0: usize, ell: usize },
    #[error("beta length mismatch: got {got}, expected {expected}")]
    BetaLengthMismatch { got: usize, expected: usize },
    #[error("hybrid sumfold requires ell0 < ell, got ell0={ell0}, ell={ell}")]
    HybridPrefixNeedsTail { ell0: usize, ell: usize },
    #[error("equality table construction failed: {0}")]
    EqTable(#[from] ArithErrors),
}

/// Dense per-instance linear claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearInstanceClaims<F: PrimeField> {
    claims: Vec<F>,
    ell: usize,
}

impl<F: PrimeField> LinearInstanceClaims<F> {
    pub fn new(claims: Vec<F>) -> Result<Self, SumFoldError> {
        if claims.is_empty() {
            return Err(SumFoldError::EmptyClaims);
        }
        if !claims.len().is_power_of_two() {
            return Err(SumFoldError::InstanceCountNotPowerOfTwo { len: claims.len() });
        }

        let ell =
            usize::try_from(claims.len().trailing_zeros()).expect("trailing_zeros fits usize");
        Ok(Self { claims, ell })
    }

    pub fn from_claims_for_ell(claims: Vec<F>, ell: usize) -> Result<Self, SumFoldError> {
        let expected = checked_domain_size(ell)?;
        if claims.len() != expected {
            return Err(SumFoldError::InstanceCountMismatch {
                ell,
                got: claims.len(),
                expected,
            });
        }
        Self::new(claims)
    }

    pub fn claims(&self) -> &[F] {
        &self.claims
    }

    pub fn ell(&self) -> usize {
        self.ell
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }
}

/// Prefix table over the first `ell0` instance variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearPrefixTable<F: PrimeField> {
    values: Vec<F>,
    ell: usize,
    ell0: usize,
}

impl<F: PrimeField> LinearPrefixTable<F> {
    pub(crate) fn from_values_for_prefix_vars(
        values: Vec<F>,
        ell: usize,
        prefix_vars: usize,
    ) -> Result<Self, SumFoldError> {
        if prefix_vars > ell {
            return Err(SumFoldError::Ell0TooLarge {
                ell0: prefix_vars,
                ell,
            });
        }
        let expected = checked_domain_size(prefix_vars)?;
        if values.len() != expected {
            return Err(SumFoldError::InstanceCountMismatch {
                ell: prefix_vars,
                got: values.len(),
                expected,
            });
        }
        Ok(Self {
            values,
            ell,
            ell0: prefix_vars,
        })
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn build(
        instance_claims: &LinearInstanceClaims<F>,
        beta: &[F],
        ell0: usize,
        field_cfg: &F::Config,
    ) -> Result<Self, SumFoldError> {
        let ell = instance_claims.ell();
        if ell0 > ell {
            return Err(SumFoldError::Ell0TooLarge { ell0, ell });
        }
        if beta.len() != ell {
            return Err(SumFoldError::BetaLengthMismatch {
                got: beta.len(),
                expected: ell,
            });
        }

        let prefix_len = checked_domain_size(ell0)?;
        let tail_vars = ell - ell0;
        let tail_len = checked_domain_size(tail_vars)?;
        let tail_weights = if tail_vars == 0 {
            vec![F::one_with_cfg(field_cfg)]
        } else {
            build_eq_x_r_vec(&beta[ell0..], field_cfg)?
        };

        debug_assert_eq!(tail_weights.len(), tail_len);
        let mut values = vec![F::zero_with_cfg(field_cfg); prefix_len];
        for (tail_weight, claims_chunk) in tail_weights
            .iter()
            .zip(instance_claims.claims().chunks_exact(prefix_len))
        {
            for (value, claim) in values.iter_mut().zip(claims_chunk) {
                *value += tail_weight.clone() * claim;
            }
        }

        Ok(Self { values, ell, ell0 })
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn ell(&self) -> usize {
        self.ell
    }

    pub fn ell0(&self) -> usize {
        self.ell0
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn to_mle(&self, field_cfg: &F::Config) -> DenseMultilinearExtension<F::Inner> {
        let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
        DenseMultilinearExtension::from_evaluations_vec(
            self.ell0,
            self.values
                .iter()
                .map(|value| value.inner().clone())
                .collect(),
            zero_inner,
        )
    }
}

struct LinearSumFoldPrefixFastPath<F: PrimeField> {
    instance_claims: LinearInstanceClaims<F>,
    beta: Vec<F>,
    ell0: usize,
    prefix_state: SumcheckProverState<F>,
}

impl<F> LinearSumFoldPrefixFastPath<F>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
{
    fn new(
        instance_claims: LinearInstanceClaims<F>,
        beta: Vec<F>,
        ell0: usize,
        field_cfg: &F::Config,
    ) -> Result<Self, SumFoldError> {
        let table = LinearPrefixTable::build(&instance_claims, &beta, ell0, field_cfg)?;
        let eq_prefix = build_eq_x_r_inner(&beta[..ell0], field_cfg)?;
        let table_mle = table.to_mle(field_cfg);
        let prefix_state = SumcheckProverState::new(vec![eq_prefix, table_mle], ell0, 2);

        Ok(Self {
            instance_claims,
            beta,
            ell0,
            prefix_state,
        })
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn finish_tail_mles(
        self,
        prefix_challenges: &[F],
        field_cfg: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        debug_assert_eq!(prefix_challenges.len(), self.ell0);
        let ell = self.instance_claims.ell();
        let tail_vars = ell - self.ell0;
        let prefix_len = checked_domain_size(self.ell0).expect("validated ell0 fits usize");
        let tail_len = checked_domain_size(tail_vars).expect("validated tail vars fit usize");
        let prefix_weights = build_eq_x_r_vec(prefix_challenges, field_cfg)
            .expect("prefix challenge equality table should build");
        let beta_tail_weights = build_eq_x_r_vec(&self.beta[self.ell0..], field_cfg)
            .expect("tail beta equality table should build");
        let eq_prefix_at_r = eq_eval(
            prefix_challenges,
            &self.beta[..self.ell0],
            F::one_with_cfg(field_cfg),
        )
        .expect("prefix challenge and beta prefix lengths match");

        let mut bound_claims = vec![F::zero_with_cfg(field_cfg); tail_len];
        for (tail, value) in bound_claims.iter_mut().enumerate() {
            for (prefix, weight) in prefix_weights.iter().enumerate().take(prefix_len) {
                let idx = prefix + (tail << self.ell0);
                *value += weight.clone() * &self.instance_claims.claims()[idx];
            }
        }

        let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
        let scaled_eq_tail = DenseMultilinearExtension::from_evaluations_vec(
            tail_vars,
            beta_tail_weights
                .iter()
                .map(|weight| (eq_prefix_at_r.clone() * weight).inner().clone())
                .collect(),
            zero_inner.clone(),
        );
        let bound_claims = DenseMultilinearExtension::from_evaluations_vec(
            tail_vars,
            bound_claims
                .iter()
                .map(|value| value.inner().clone())
                .collect(),
            zero_inner,
        );

        vec![scaled_eq_tail, bound_claims]
    }
}

impl<F> PrefixFastPath<F> for LinearSumFoldPrefixFastPath<F>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
{
    fn prefix_len(&self) -> usize {
        self.ell0
    }

    fn prove_prefix_round(
        &mut self,
        verifier_msg: &Option<F>,
        config: &F::Config,
    ) -> PrefixRoundOutput<F> {
        let msg = self.prefix_state.prove_round(
            verifier_msg,
            |values: &[F]| values[0].clone() * &values[1],
            config,
        );
        let asserted_sum = if self.prefix_state.round == 1 {
            self.prefix_state.asserted_sum.clone()
        } else {
            None
        };

        PrefixRoundOutput {
            asserted_sum,
            tail_evaluations: msg.0.tail_evaluations,
        }
    }

    fn finish_prefix(
        self: Box<Self>,
        prefix_challenges: &[F],
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        self.finish_tail_mles(prefix_challenges, config)
    }
}

impl<F> LinearInstanceClaims<F>
where
    F: InnerTransparentField + DelayedFieldProductSum + Send + Sync + 'static,
{
    pub fn build_full_sumcheck_group(
        &self,
        beta: &[F],
        field_cfg: &F::Config,
    ) -> Result<MultiDegreeSumcheckGroup<F>, SumFoldError> {
        if beta.len() != self.ell {
            return Err(SumFoldError::BetaLengthMismatch {
                got: beta.len(),
                expected: self.ell,
            });
        }

        let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
        let eq_beta = build_eq_x_r_inner(beta, field_cfg)?;
        let claims = DenseMultilinearExtension::from_evaluations_vec(
            self.ell,
            self.claims
                .iter()
                .map(|value| value.inner().clone())
                .collect(),
            zero_inner,
        );

        Ok(MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_beta, claims],
            Box::new(|values: &[F]| values[0].clone() * &values[1]),
        ))
    }

    pub fn build_hybrid_sumcheck_group(
        &self,
        beta: &[F],
        ell0: usize,
        field_cfg: &F::Config,
    ) -> Result<MultiDegreeSumcheckGroup<F>, SumFoldError> {
        if beta.len() != self.ell {
            return Err(SumFoldError::BetaLengthMismatch {
                got: beta.len(),
                expected: self.ell,
            });
        }
        if ell0 == 0 {
            return self.build_full_sumcheck_group(beta, field_cfg);
        }
        if ell0 >= self.ell {
            return Err(SumFoldError::HybridPrefixNeedsTail {
                ell0,
                ell: self.ell,
            });
        }

        let fast_path =
            LinearSumFoldPrefixFastPath::new(self.clone(), beta.to_vec(), ell0, field_cfg)?;

        Ok(MultiDegreeSumcheckGroup::with_prefix_fast(
            2,
            Vec::new(),
            Box::new(|values: &[F]| values[0].clone() * &values[1]),
            Box::new(fast_path),
        ))
    }
}

pub(crate) fn checked_domain_size(ell: usize) -> Result<usize, SumFoldError> {
    let shift = u32::try_from(ell).map_err(|_| SumFoldError::DomainTooLarge { ell })?;
    1usize
        .checked_shl(shift)
        .ok_or(SumFoldError::DomainTooLarge { ell })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof},
        test_utils::test_config,
    };
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};
    use zinc_poly::{
        mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
        utils::{build_eq_x_r_vec, eq_eval},
    };
    use zinc_transcript::Blake3Transcript;

    type F = MontyField<4>;

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn claims_for_ell(ell: usize) -> LinearInstanceClaims<F> {
        let claims = (0..(1usize << ell))
            .map(|idx| {
                let idx = u64::try_from(idx).expect("test index fits u64");
                f(idx + 2)
            })
            .collect();
        LinearInstanceClaims::from_claims_for_ell(claims, ell).unwrap()
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn direct_prefix_value(
        claims: &LinearInstanceClaims<F>,
        beta: &[F],
        ell0: usize,
        prefix: usize,
    ) -> F {
        let cfg = test_config();
        let ell = claims.ell();
        let tail_vars = ell - ell0;
        let tail_weights = if tail_vars == 0 {
            vec![F::one_with_cfg(&cfg)]
        } else {
            build_eq_x_r_vec(&beta[ell0..], &cfg).unwrap()
        };

        let mut acc = F::zero_with_cfg(&cfg);
        for (tail, weight) in tail_weights.iter().enumerate() {
            let idx = prefix + (tail << ell0);
            acc += weight.clone() * &claims.claims()[idx];
        }
        acc
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn direct_full_beta_sum(claims: &LinearInstanceClaims<F>, beta: &[F]) -> F {
        let cfg = test_config();
        let weights = build_eq_x_r_vec(beta, &cfg).unwrap();
        let mut acc = F::zero_with_cfg(&cfg);
        for (weight, claim) in weights.iter().zip(claims.claims()) {
            acc += weight.clone() * claim;
        }
        acc
    }

    fn claims_mle(
        claims: &LinearInstanceClaims<F>,
    ) -> DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner> {
        let cfg = test_config();
        let zero_inner = F::zero_with_cfg(&cfg).inner().clone();
        DenseMultilinearExtension::from_evaluations_vec(
            claims.ell(),
            claims
                .claims()
                .iter()
                .map(|claim| claim.inner().clone())
                .collect(),
            zero_inner,
        )
    }

    fn prove_and_verify(
        group: MultiDegreeSumcheckGroup<F>,
        ell: usize,
    ) -> (MultiDegreeSumcheckProof<F>, Vec<F>, Vec<F>) {
        let cfg = test_config();
        let mut prover_transcript = Blake3Transcript::new();
        let (proof, _states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![group],
            ell,
            &cfg,
        );

        let mut verifier_transcript = Blake3Transcript::new();
        let subclaims =
            MultiDegreeSumcheck::verify_as_subprotocol(&mut verifier_transcript, ell, &proof, &cfg)
                .expect("sumcheck proof should verify");

        (
            proof,
            subclaims.point().to_vec(),
            subclaims.expected_evaluations().to_vec(),
        )
    }

    fn proof_satisfies_dense_claim(
        proof: &MultiDegreeSumcheckProof<F>,
        ell: usize,
        beta: &[F],
        claims: &LinearInstanceClaims<F>,
    ) -> bool {
        let cfg = test_config();
        let mut verifier_transcript = Blake3Transcript::new();
        let Ok(subclaims) =
            MultiDegreeSumcheck::verify_as_subprotocol(&mut verifier_transcript, ell, proof, &cfg)
        else {
            return false;
        };

        let eq_at_point =
            eq_eval(subclaims.point(), beta, F::one_with_cfg(&cfg)).expect("same length");
        let claim_at_point = claims_mle(claims)
            .evaluate_with_config(subclaims.point(), &cfg)
            .unwrap();
        subclaims.expected_evaluations()[0] == eq_at_point * claim_at_point
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn beta_for_ell(ell: usize) -> Vec<F> {
        (0..ell)
            .map(|idx| f(3 + 2 * u64::try_from(idx).expect("test index fits u64")))
            .collect()
    }

    fn assert_hybrid_matches_full_sumcheck(ell: usize, ell0: usize) {
        let cfg = test_config();
        let claims = claims_for_ell(ell);
        let beta = beta_for_ell(ell);

        let full_group = claims.build_full_sumcheck_group(&beta, &cfg).unwrap();
        let optimized_group = claims
            .build_hybrid_sumcheck_group(&beta, ell0, &cfg)
            .unwrap();

        let (full_proof, full_point, full_expected) = prove_and_verify(full_group, ell);
        let (optimized_proof, optimized_point, optimized_expected) =
            prove_and_verify(optimized_group, ell);

        assert_eq!(optimized_proof, full_proof);
        assert_eq!(optimized_point, full_point);
        assert_eq!(optimized_expected, full_expected);
        assert_eq!(
            full_proof.claimed_sums()[0],
            direct_full_beta_sum(&claims, &beta)
        );

        let eq_at_point = eq_eval(&full_point, &beta, F::one_with_cfg(&cfg)).expect("same length");
        let claim_at_point = claims_mle(&claims)
            .evaluate_with_config(&full_point, &cfg)
            .unwrap();
        assert_eq!(full_expected[0], eq_at_point * claim_at_point);
    }

    #[test]
    fn prefix_table_matches_direct_tail_fold_for_all_ell0_cases() {
        let cfg = test_config();
        let claims = claims_for_ell(3);
        let beta = vec![f(3), f(5), f(7)];

        for ell0 in 0..=3 {
            let table = LinearPrefixTable::build(&claims, &beta, ell0, &cfg).unwrap();
            assert_eq!(table.ell(), 3);
            assert_eq!(table.ell0(), ell0);
            assert_eq!(table.len(), 1usize << ell0);

            for prefix in 0..table.len() {
                assert_eq!(
                    table.values()[prefix],
                    direct_prefix_value(&claims, &beta, ell0, prefix)
                );
            }
        }
    }

    #[test]
    fn hybrid_sumfold_proof_matches_full_ordinary_sumcheck() {
        for (ell, ell0) in [(3, 1), (4, 2), (5, 3), (4, 0), (5, 4)] {
            assert_hybrid_matches_full_sumcheck(ell, ell0);
        }
    }

    #[test]
    fn hybrid_sumfold_rejects_tampered_prefix_and_tail_messages() {
        let cfg = test_config();
        let ell = 4;
        let ell0 = 2;
        let claims = claims_for_ell(ell);
        let beta = beta_for_ell(ell);

        let group = claims
            .build_hybrid_sumcheck_group(&beta, ell0, &cfg)
            .unwrap();
        let (proof, _point, _expected) = prove_and_verify(group, ell);

        let mut prefix_tampered = proof.clone();
        prefix_tampered.group_messages_mut_for_testing()[0][0]
            .0
            .tail_evaluations[0] += f(1);
        assert!(!proof_satisfies_dense_claim(
            &prefix_tampered,
            ell,
            &beta,
            &claims
        ));

        let mut tail_tampered = proof;
        tail_tampered.group_messages_mut_for_testing()[0][ell0]
            .0
            .tail_evaluations[0] += f(1);
        assert!(!proof_satisfies_dense_claim(
            &tail_tampered,
            ell,
            &beta,
            &claims
        ));
    }

    #[test]
    fn linear_sumfold_validation_errors_are_reported() {
        let cfg = test_config();

        assert!(matches!(
            LinearInstanceClaims::<F>::new(Vec::new()),
            Err(SumFoldError::EmptyClaims)
        ));
        assert!(matches!(
            LinearInstanceClaims::new(vec![f(1), f(2), f(3)]),
            Err(SumFoldError::InstanceCountNotPowerOfTwo { len: 3 })
        ));
        assert!(matches!(
            LinearInstanceClaims::from_claims_for_ell(vec![f(1), f(2), f(3)], 2),
            Err(SumFoldError::InstanceCountMismatch {
                ell: 2,
                got: 3,
                expected: 4
            })
        ));

        let claims = claims_for_ell(2);
        assert!(matches!(
            LinearPrefixTable::build(&claims, &[f(3), f(5)], 3, &cfg),
            Err(SumFoldError::Ell0TooLarge { ell0: 3, ell: 2 })
        ));
        assert!(matches!(
            LinearPrefixTable::build(&claims, &[f(3)], 1, &cfg),
            Err(SumFoldError::BetaLengthMismatch {
                got: 1,
                expected: 2
            })
        ));

        assert!(matches!(
            claims
                .build_hybrid_sumcheck_group(&[f(3), f(5)], 2, &cfg)
                .err(),
            Some(SumFoldError::HybridPrefixNeedsTail { ell0: 2, ell: 2 })
        ));
    }
}
