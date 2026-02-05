use crypto_primitives::{FixedSemiring, FromWithConfig, PrimeField, Semiring};
use num_traits::Zero;
use zinc_uair::{
    ideal::{Ideal, IdealCheck},
    ideal_collector::CollectedIdeal,
};
use zinc_utils::from_ref::FromRef;

use crate::{EvaluatablePolynomial, univariate::dynamic::over_field::DynamicPolynomialF};

#[derive(Clone, Copy, Debug)]
pub struct DegreeOneIdeal<R: Semiring> {
    generating_root: R,
}

impl<R: FixedSemiring> DegreeOneIdeal<R> {
    pub fn new(generating_root: R) -> Self {
        Self { generating_root }
    }
}

impl<R: Semiring> FromRef<DegreeOneIdeal<R>> for DegreeOneIdeal<R> {
    #[inline(always)]
    fn from_ref(ideal: &DegreeOneIdeal<R>) -> Self {
        ideal.clone()
    }
}

impl<R: Semiring> Ideal for DegreeOneIdeal<R> {}

impl<F: PrimeField> DegreeOneIdeal<F> {
    pub fn from_with_cfg<R>(ideal_over_ring: &DegreeOneIdeal<R>, field_cfg: &F::Config) -> Self
    where
        R: Semiring,
        F: FromWithConfig<R>,
    {
        Self {
            generating_root: F::from_with_cfg(ideal_over_ring.generating_root.clone(), field_cfg),
        }
    }
}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for CollectedIdeal<DegreeOneIdeal<F>> {
    fn contains(&self, value: &DynamicPolynomialF<F>) -> bool {
        if value.is_zero() {
            return true;
        }

        match &self.ideal_or_zero {
            Some(DegreeOneIdeal { generating_root }) => {
                let field_cfg = value.coeffs[0].cfg();
                let root_in_field = F::from_with_cfg(generating_root.clone(), field_cfg);
                F::is_zero(
                    &value
                        .evaluate_at_point(&root_in_field)
                        .expect("arithmetic overflow"),
                )
            }
            None => value.is_zero(),
        }
    }
}
