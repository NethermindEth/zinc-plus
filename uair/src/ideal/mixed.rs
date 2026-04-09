use crate::{
    ideal::{Ideal, IdealCheck, degree_one::DegreeOneIdeal, xn_minus_one::XnMinusOneIdeal},
    ideal_collector::IdealOrZero,
};
use crypto_primitives::{FromWithConfig, PrimeField, Semiring};
use num_traits::Zero;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_utils::from_ref::FromRef;

/// An ideal that is either a degree-one ideal `(X - a)` or an
/// `X^N - 1` ideal. Combined with [`IdealOrZero`], this lets a
/// UAIR mix the zero ideal, `(X - a)` ideals, and the `(X^N - 1)`
/// ideal within a single constraint system.
#[derive(Clone, Copy, Debug)]
pub enum MixedDegreeOneOrXnMinusOne<R: Semiring, const N: usize> {
    DegreeOne(DegreeOneIdeal<R>),
    XnMinusOne(XnMinusOneIdeal<N>),
}

impl<R: Semiring, const N: usize> FromRef<MixedDegreeOneOrXnMinusOne<R, N>>
    for MixedDegreeOneOrXnMinusOne<R, N>
{
    #[inline(always)]
    fn from_ref(value: &MixedDegreeOneOrXnMinusOne<R, N>) -> Self {
        value.clone()
    }
}

impl<R: Semiring, const N: usize> Ideal for MixedDegreeOneOrXnMinusOne<R, N> {}

impl<F: PrimeField, const N: usize> MixedDegreeOneOrXnMinusOne<F, N> {
    pub fn from_with_cfg<R>(
        ring_ideal: &MixedDegreeOneOrXnMinusOne<R, N>,
        field_cfg: &F::Config,
    ) -> Self
    where
        R: Semiring,
        F: for<'a> FromWithConfig<&'a R>,
    {
        match ring_ideal {
            MixedDegreeOneOrXnMinusOne::DegreeOne(i) => {
                Self::DegreeOne(DegreeOneIdeal::from_with_cfg(i, field_cfg))
            }
            MixedDegreeOneOrXnMinusOne::XnMinusOne(_) => Self::XnMinusOne(XnMinusOneIdeal::<N>),
        }
    }
}

impl<F: PrimeField, const N: usize> IdealCheck<DynamicPolynomialF<F>>
    for IdealOrZero<MixedDegreeOneOrXnMinusOne<F, N>>
{
    fn contains(&self, value: &DynamicPolynomialF<F>) -> bool {
        if value.is_zero() {
            return true;
        }

        match &self.ideal_or_zero {
            None => value.is_zero(),
            Some(MixedDegreeOneOrXnMinusOne::DegreeOne(i)) => {
                // Delegate to the DegreeOneIdeal check impl so the
                // arithmetic lives in a single place.
                let wrapped: IdealOrZero<DegreeOneIdeal<F>> = IdealOrZero {
                    ideal_or_zero: Some(i.clone()),
                };
                wrapped.contains(value)
            }
            Some(MixedDegreeOneOrXnMinusOne::XnMinusOne(_)) => {
                let wrapped: IdealOrZero<XnMinusOneIdeal<N>> = IdealOrZero {
                    ideal_or_zero: Some(XnMinusOneIdeal::<N>),
                };
                wrapped.contains(value)
            }
        }
    }
}
