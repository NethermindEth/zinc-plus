use crypto_primitives::PrimeField;
use thiserror::Error;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    utils::{ArithErrors, build_eq_x_r_inner, build_eq_x_r_vec},
};
use zinc_utils::{
    UNCHECKED,
    delayed_reduction::DelayedFieldProductSum,
    inner_product::{FieldFieldInnerProduct, InnerProduct},
};

use crate::sumcheck::multi_degree::MultiDegreeSumcheckGroup;

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
    #[error("{label} length mismatch: got {got}, expected {expected}")]
    WeightLengthMismatch {
        label: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("sumcheck group construction requires ell0 > 0")]
    SumcheckNeedsNonzeroEll0,
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

impl<F> LinearPrefixTable<F>
where
    F: PrimeField + DelayedFieldProductSum + 'static,
    F::Inner: num_traits::Zero,
{
    pub fn build_sumcheck_claim(
        &self,
        prefix_eq_weights: &[F],
        field_cfg: &F::Config,
    ) -> Result<F, SumFoldError> {
        if prefix_eq_weights.len() != self.values.len() {
            return Err(SumFoldError::WeightLengthMismatch {
                label: "prefix_eq_weights",
                got: prefix_eq_weights.len(),
                expected: self.values.len(),
            });
        }

        Ok(FieldFieldInnerProduct::inner_product::<UNCHECKED>(
            prefix_eq_weights,
            &self.values,
            F::zero_with_cfg(field_cfg),
        )
        .expect("prefix equality weights and table values have matching lengths"))
    }

    pub fn build_sumcheck_group_from_prefix_weights(
        &self,
        prefix_eq_weights: &[F],
        field_cfg: &F::Config,
    ) -> Result<MultiDegreeSumcheckGroup<F>, SumFoldError> {
        if self.ell0 == 0 {
            return Err(SumFoldError::SumcheckNeedsNonzeroEll0);
        }
        if prefix_eq_weights.len() != self.values.len() {
            return Err(SumFoldError::WeightLengthMismatch {
                label: "prefix_eq_weights",
                got: prefix_eq_weights.len(),
                expected: self.values.len(),
            });
        }

        let zero_inner = F::zero_with_cfg(field_cfg).inner().clone();
        let eq_prefix = DenseMultilinearExtension::from_evaluations_vec(
            self.ell0,
            prefix_eq_weights
                .iter()
                .map(|value| value.inner().clone())
                .collect(),
            zero_inner,
        );
        let table = self.to_mle(field_cfg);

        Ok(MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_prefix, table],
            Box::new(|values: &[F]| values[0].clone() * &values[1]),
        ))
    }

    pub fn build_sumcheck_group(
        &self,
        beta_prefix: &[F],
        field_cfg: &F::Config,
    ) -> Result<MultiDegreeSumcheckGroup<F>, SumFoldError> {
        if self.ell0 == 0 {
            return Err(SumFoldError::SumcheckNeedsNonzeroEll0);
        }
        if beta_prefix.len() != self.ell0 {
            return Err(SumFoldError::BetaLengthMismatch {
                got: beta_prefix.len(),
                expected: self.ell0,
            });
        }

        let eq_beta_prefix = build_eq_x_r_inner(beta_prefix, field_cfg)?;
        let table = self.to_mle(field_cfg);

        Ok(MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_beta_prefix, table],
            Box::new(|values: &[F]| values[0].clone() * &values[1]),
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
    use crate::{sumcheck::multi_degree::MultiDegreeSumcheck, test_utils::test_config};
    use crypto_primitives::{FromWithConfig, crypto_bigint_monty::MontyField};
    use zinc_poly::{
        mle::MultilinearExtensionWithConfig,
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
    fn linear_sumfold_group_proves_weighted_instance_sum() {
        let cfg = test_config();
        let claims = claims_for_ell(3);
        let beta = vec![f(3), f(5), f(7)];
        let ell0 = 2;
        let table = LinearPrefixTable::build(&claims, &beta, ell0, &cfg).unwrap();

        let group = table
            .build_sumcheck_group(&beta[..ell0], &cfg)
            .expect("ell0 > 0 should build a group");
        let mut prover_transcript = Blake3Transcript::new();
        let (proof, _states) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![group],
            ell0,
            &cfg,
        );

        assert_eq!(
            proof.claimed_sums()[0],
            direct_full_beta_sum(&claims, &beta)
        );

        let mut verifier_transcript = Blake3Transcript::new();
        let subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
            &mut verifier_transcript,
            ell0,
            &proof,
            &cfg,
        )
        .expect("linear sumfold group should verify");

        let point = subclaims.point();
        let eq_at_point =
            eq_eval(point, &beta[..ell0], F::one_with_cfg(&cfg)).expect("same length");
        let table_eval = table
            .to_mle(&cfg)
            .evaluate_with_config(point, &cfg)
            .unwrap();
        assert_eq!(
            subclaims.expected_evaluations()[0],
            eq_at_point * table_eval
        );
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

        let table = LinearPrefixTable::build(&claims, &[f(3), f(5)], 0, &cfg).unwrap();
        assert!(matches!(
            table.build_sumcheck_group(&[], &cfg),
            Err(SumFoldError::SumcheckNeedsNonzeroEll0)
        ));
    }
}
