use crypto_primitives::{FixedSemiring, FromWithConfig, PrimeField, Semiring};
use num_traits::Zero;
use zinc_uair::ideal::{Ideal, IdealCheck};
use zinc_utils::from_ref::FromRef;

use crate::{EvaluatablePolynomial, univariate::dynamic::DynamicPolynomial};


#[derive(Clone, Copy, Debug)]
pub enum DegreeOneIdeal<R: Semiring> {
    DegreeOneIdeal { generating_root: R },
    ZeroIdeal,
}

impl<R: FixedSemiring> DegreeOneIdeal<R> {
    pub fn new(generating_root: R) -> Self {
        Self::DegreeOneIdeal { generating_root }
    }
}

impl<R: Semiring> FromRef<DegreeOneIdeal<R>> for DegreeOneIdeal<R> {
    #[inline(always)]
    fn from_ref(ideal: &DegreeOneIdeal<R>) -> Self {
        ideal.clone()
    }
}

impl<R: Semiring> Ideal for DegreeOneIdeal<R> {
    #[inline(always)]
    fn zero_ideal() -> Self {
        Self::ZeroIdeal
    }
}

impl<F: PrimeField> DegreeOneIdeal<F> {
    pub fn from_with_cfg<R>(ideal_over_ring: &DegreeOneIdeal<R>, field_cfg: &F::Config) -> Self
    where
        R: Semiring,
        F: FromWithConfig<R>,
    {
        match ideal_over_ring {
            DegreeOneIdeal::DegreeOneIdeal { generating_root } => Self::DegreeOneIdeal {
                generating_root: F::from_with_cfg(generating_root.clone(), field_cfg),
            },

            DegreeOneIdeal::ZeroIdeal => Self::ZeroIdeal,
        }
    }
}

impl<F: PrimeField> IdealCheck<DegreeOneIdeal<F>> for DynamicPolynomial<F> {
    fn is_contained_in(&self, ideal: &DegreeOneIdeal<F>) -> bool {
        if self.is_zero() {
            return true;
        }

        match ideal {
            DegreeOneIdeal::DegreeOneIdeal { generating_root } => {
                let field_cfg = self.coeffs[0].cfg();
                let root_in_field = F::from_with_cfg(generating_root.clone(), field_cfg);
                F::is_zero(
                    &self
                        .evaluate_at_point(&root_in_field)
                        .expect("arithmetic overflow"),
                )
            }
            DegreeOneIdeal::ZeroIdeal => self.is_zero(),
        }
    }
}
