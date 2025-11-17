use crypto_primitives::{PrimeField, crypto_bigint_int::Int};
use num_traits::{CheckedAdd, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{add, mul_by_scalar::MulByScalar};

/// A trait for types that can be dot-multiplied.
pub trait InnerProduct<Rhs> {
    /// The resulting type we get after the inner product.
    type Output;

    /// The main entry point for the inner product.
    fn inner_product(&self, rhs: &[Rhs]) -> Self::Output;
}

impl<Lhs, Rhs> InnerProduct<Rhs> for &[Lhs]
where
    Lhs: Clone + Zero + CheckedAdd + for<'z> MulByScalar<&'z Rhs> + Send + Sync,
    Rhs: Send + Sync,
{
    type Output = Lhs;

    fn inner_product(&self, rhs: &[Rhs]) -> Self::Output {
        inner_product_aux(self, rhs, Lhs::zero)
    }
}

impl<Lhs, Rhs> InnerProduct<Rhs> for Vec<Lhs>
where
    for<'a> &'a [Lhs]: InnerProduct<Rhs, Output = Lhs>,
{
    type Output = Lhs;

    fn inner_product(&self, rhs: &[Rhs]) -> Self::Output {
        self.as_slice().inner_product(rhs)
    }
}

impl<Lhs, Rhs, const N: usize> InnerProduct<Rhs> for [Lhs; N]
where
    for<'a> &'a [Lhs]: InnerProduct<Rhs, Output = Lhs>,
{
    type Output = Lhs;

    fn inner_product(&self, rhs: &[Rhs]) -> Self::Output {
        self.as_slice().inner_product(rhs)
    }
}

impl<T, const LIMBS: usize> InnerProduct<T> for Int<LIMBS> {
    type Output = Self;

    fn inner_product(&self, point: &[T]) -> Self::Output {
        if !point.is_empty() {
            // TODO: Do we really want the RHS be empty?
            //       Or do we want it to be one point?
            panic!("The RHS should be empty");
        }
        *self
    }
}

/// A trait for types can be dot-multiplied
/// but with a catch: they need a config for
/// getting the zero (e.g. `PrimeField`s).
pub trait InnerProductWithConfig<Rhs> {
    type Output;
    type Config;

    fn inner_product(&self, rhs: &[Rhs], config: &Self::Config) -> Self::Output;
}

impl<Lhs, Rhs> InnerProductWithConfig<Rhs> for &[Lhs]
where
    Lhs: PrimeField + CheckedAdd + for<'z> MulByScalar<&'z Rhs>,
    Rhs: Send + Sync,
{
    type Output = Lhs;
    type Config = Lhs::Config;

    fn inner_product(&self, rhs: &[Rhs], config: &Lhs::Config) -> Self::Output {
        inner_product_aux(self, rhs, || Lhs::zero_with_cfg(config))
    }
}

impl<Lhs: PrimeField, Rhs> InnerProductWithConfig<Rhs> for Vec<Lhs>
where
    for<'a> &'a [Lhs]: InnerProductWithConfig<Rhs, Output = Lhs, Config = Lhs::Config>,
{
    type Output = Lhs;
    type Config = Lhs::Config;

    fn inner_product(&self, rhs: &[Rhs], config: &Lhs::Config) -> Self::Output {
        self.as_slice().inner_product(rhs, config)
    }
}

// TODO: can these two merged into one?

#[cfg(not(feature = "parallel"))]
fn inner_product_aux<Lhs, Rhs, Z>(lhs: &[Lhs], rhs: &[Rhs], zero: Z) -> Lhs
where
    Lhs: Clone + CheckedAdd + for<'z> MulByScalar<&'z Rhs>,
    Z: Fn() -> Lhs,
{
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| {
            lhs.mul_by_scalar(rhs)
                .expect("Cannot multiply an element by a coefficient")
        })
        .reduce(|acc, product| add!(acc, &product))
        .unwrap_or(zero())
}

#[cfg(feature = "parallel")]
fn inner_product_aux<Lhs, Rhs, Z>(lhs: &[Lhs], rhs: &[Rhs], zero: Z) -> Lhs
where
    Lhs: Clone + Sync + Send + CheckedAdd + for<'z> MulByScalar<&'z Rhs>,
    Rhs: Send + Sync,
    Z: Fn() -> Lhs + Send + Sync,
{
    lhs.par_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| {
            lhs.mul_by_scalar(rhs)
                .expect("Cannot multiply an element by a coefficient")
        })
        .reduce(zero, |acc, product| add!(acc, &product))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_inner_product_basic() {
        let lhs = [1, 2, 3];
        let rhs = [4, 5, 6];
        assert_eq!(lhs.inner_product(&rhs), 4 + 2 * 5 + 3 * 6);
    }
}
