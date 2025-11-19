use crypto_primitives::crypto_bigint_int::Int;
use num_traits::CheckedAdd;
use thiserror::Error;

use crate::mul_by_scalar::MulByScalar;

/// A trait for types that can be dot-multiplied.
pub trait InnerProduct<Rhs> {
    /// The resulting type we get after the inner product.
    type Output;

    /// The main entry point for the inner product.
    fn inner_product(
        &self,
        rhs: &[Rhs],
        zero: Self::Output,
    ) -> Result<Self::Output, InnerProductError>;
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum InnerProductError {
    #[error("The length of LHS and RHS does not match: LHS={lhs}, RHS={rhs}")]
    LengthMismatch { lhs: usize, rhs: usize },
    #[error("Arithmetic overflow")]
    Overflow,
}

impl<Lhs, Rhs> InnerProduct<Rhs> for &[Lhs]
where
    Lhs: Clone + CheckedAdd + for<'z> MulByScalar<&'z Rhs>,
{
    type Output = Lhs;

    fn inner_product(&self, rhs: &[Rhs], zero: Lhs) -> Result<Self::Output, InnerProductError> {
        self.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs.mul_by_scalar(rhs).ok_or(InnerProductError::Overflow))
            .reduce(|acc, product| {
                acc?.checked_add(&product?)
                    .ok_or(InnerProductError::Overflow)
            })
            .unwrap_or(Ok(zero))
    }
}

impl<Lhs, Rhs> InnerProduct<Rhs> for Vec<Lhs>
where
    for<'a> &'a [Lhs]: InnerProduct<Rhs, Output = Lhs>,
{
    type Output = Lhs;

    #[inline]
    fn inner_product(&self, rhs: &[Rhs], zero: Lhs) -> Result<Self::Output, InnerProductError> {
        self.as_slice().inner_product(rhs, zero)
    }
}

impl<Lhs, Rhs, const N: usize> InnerProduct<Rhs> for [Lhs; N]
where
    for<'a> &'a [Lhs]: InnerProduct<Rhs, Output = Lhs>,
{
    type Output = Lhs;

    #[inline]
    fn inner_product(&self, rhs: &[Rhs], zero: Lhs) -> Result<Self::Output, InnerProductError> {
        self.as_slice().inner_product(rhs, zero)
    }
}

impl<T, const LIMBS: usize> InnerProduct<T> for Int<LIMBS> {
    type Output = Self;

    fn inner_product(&self, point: &[T], _zero: Self) -> Result<Self::Output, InnerProductError> {
        if !point.is_empty() {
            // TODO: Do we really want the RHS be empty?
            //       Or do we want it to be one point?
            Err(InnerProductError::LengthMismatch {
                lhs: 0,
                rhs: point.len(),
            })
        } else {
            Ok(*self)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_inner_product_basic() {
        let lhs = [1, 2, 3];
        let rhs = [4, 5, 6];
        assert_eq!(lhs.inner_product(&rhs, 0), Ok(4 + 2 * 5 + 3 * 6));
    }
}
