use crypto_primitives::{FixedSemiring, Semiring};
use num_traits::Zero;
use zinc_uair::ideal::{Ideal, IdealCheck};
use zinc_utils::from_ref::FromRef;

use crate::EvaluatablePolynomial;

use super::dense::DensePolynomial;

#[derive(Clone, Copy, Debug)]
pub enum DegreeOneIdeal<R: FixedSemiring, Inner: Ideal, const DEGREE_PLUS_ONE: usize> {
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

impl<R: FixedSemiring, Inner: Ideal, const DEGREE_PLUS_ONE: usize>
    FromRef<DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>>
    for DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn from_ref(ideal: &DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>) -> Self {
        ideal.clone()
    }
}

impl<R: FixedSemiring, Inner: Ideal, const DEGREE_PLUS_ONE: usize> Ideal
    for DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>
{
    #[inline(always)]
    fn zero_ideal() -> Self {
        Self::Inner(Inner::zero_ideal())
    }
}

impl<R, Inner, const DEGREE_PLUS_ONE: usize> IdealCheck<DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>>
    for DensePolynomial<R, DEGREE_PLUS_ONE>
where
    R: FixedSemiring,
    Inner: Ideal,
    DensePolynomial<R, DEGREE_PLUS_ONE>: IdealCheck<Inner>,
{
    fn is_contained_in(&self, ideal: &DegreeOneIdeal<R, Inner, DEGREE_PLUS_ONE>) -> bool {
        match ideal {
            DegreeOneIdeal::DegreeOneIdeal { generating_root } => self
                .evaluate_at_point(generating_root)
                .expect("arithmetic overflow")
                .is_zero(),
            DegreeOneIdeal::Inner(inner) => self.is_contained_in(inner),
        }
    }
}
