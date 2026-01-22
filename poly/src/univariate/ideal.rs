use crypto_primitives::{FixedSemiring, FromWithConfig, Semiring};
use zinc_uair::ideal::{Ideal, IdealCheck};
use zinc_utils::from_ref::FromRef;

use crate::EvaluatablePolynomial;

use super::dense::DensePolynomial;

#[derive(Clone, Copy, Debug)]
pub enum DegreeOneIdeal<R: Semiring, Inner: Ideal, const DEGREE_PLUS_ONE: usize> {
    DegreeOneIdeal { generating_root: R },
    Inner(Inner),
}

impl<R: FixedSemiring, Inner: Ideal, const DEGREE_PLUS_ONE: usize>
    DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>
{
    pub fn new(generating_root: R) -> Self {
        Self::DegreeOneIdeal { generating_root }
    }
}

impl<R: Semiring, Inner: Ideal, const DEGREE_PLUS_ONE: usize>
    FromRef<DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>>
    for DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from_ref(ideal: &DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>) -> Self {
        ideal.clone()
    }
}

impl<R: Semiring, Inner: Ideal, const DEGREE_PLUS_ONE: usize> Ideal
    for DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn zero_ideal() -> Self {
        Self::Inner(Inner::zero_ideal())
    }
}

impl<R, F, Inner, const DEGREE_PLUS_ONE: usize>
    IdealCheck<DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>> for DensePolynomial<F, DEGREE_PLUS_ONE>
where
    R: Semiring,
    F: FromWithConfig<R>,
    Inner: Ideal,
    DensePolynomial<F, DEGREE_PLUS_ONE>: IdealCheck<Inner>,
{
    fn is_contained_in_with_zero(
        &self,
        ideal: &DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>,
        zero: &Self,
    ) -> bool {
        match ideal {
            DegreeOneIdeal::DegreeOneIdeal { generating_root } => {
                let field_cfg = zero.coeffs[0].cfg();
                self.evaluate_at_point(&F::from_with_cfg(generating_root.clone(), field_cfg))
                    .expect("arithmetic overflow")
                    == zero.coeffs[0]
            }
            DegreeOneIdeal::Inner(inner) => self.is_contained_in_with_zero(inner, zero),
        }
    }
}
