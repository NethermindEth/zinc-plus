use crypto_primitives::{FixedSemiring, FromWithConfig, Semiring};
use zinc_uair::ideal::{Ideal, IdealCheck};
use zinc_utils::from_ref::FromRef;

use crate::{EvaluatablePolynomial, univariate::dynamic::DynamicPolynomial};

use super::dense::DensePolynomial;

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

impl<R, F> IdealCheck<DegreeOneIdeal<R>> for DynamicPolynomial<F>
where
    R: Semiring,
    F: FromWithConfig<R>,
{
    fn is_contained_in(&self, ideal: &DegreeOneIdeal<R>, zero: &Self) -> bool {
        match ideal {
            DegreeOneIdeal::DegreeOneIdeal { generating_root } => {
                let field_cfg = zero.coeffs[0].cfg();
                let root_in_field = F::from_with_cfg(generating_root.clone(), field_cfg);
                self.evaluate_at_point(&root_in_field)
                    .expect("arithmetic overflow")
                    == zero
                        .evaluate_at_point(&root_in_field)
                        .expect("should be fine")
            }
            DegreeOneIdeal::ZeroIdeal => self == zero,
        }
    }
}
