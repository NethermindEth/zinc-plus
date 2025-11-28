use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int};
use num_traits::CheckedAdd;
use thiserror::Error;

use crate::{from_ref::FromRef, mul_by_scalar::MulByScalar};

/// A trait for types that can be dot-multiplied.
pub trait InnerProduct<Rhs, Output> {
    /// The main entry point for the inner product.
    fn inner_product(&self, rhs: &[Rhs], zero: Output) -> Result<Output, InnerProductError>;
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum InnerProductError {
    #[error("The length of LHS and RHS does not match: LHS={lhs}, RHS={rhs}")]
    LengthMismatch { lhs: usize, rhs: usize },
    #[error("Arithmetic overflow")]
    Overflow,
}

impl<Lhs, Rhs, Out> InnerProduct<Rhs, Out> for &[Lhs]
where
    Lhs: CheckedAdd + for<'a> MulByScalar<&'a Rhs>,
    Out: From<Lhs> + CheckedAdd,
{
    fn inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs.mul_by_scalar(rhs).ok_or(InnerProductError::Overflow))
            .try_fold(zero, |acc, product| {
                acc.checked_add(&product?.into())
                    .ok_or(InnerProductError::Overflow)
            })
    }
}

impl<Lhs, Rhs, Out> InnerProduct<Rhs, Out> for Vec<Lhs>
where
    for<'a> &'a [Lhs]: InnerProduct<Rhs, Out>,
{
    #[inline]
    fn inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.as_slice().inner_product(rhs, zero)
    }
}

impl<Lhs, Rhs, Out, const N: usize> InnerProduct<Rhs, Out> for [Lhs; N]
where
    for<'a> &'a [Lhs]: InnerProduct<Rhs, Out>,
{
    #[inline]
    fn inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.as_slice().inner_product(rhs, zero)
    }
}

impl<T, Out, const LIMBS: usize> InnerProduct<T, Out> for Int<LIMBS>
where
    Int<LIMBS>: for<'a> MulByScalar<&'a T>,
    Out: FromRef<Self>,
{
    fn inner_product(&self, point: &[T], _zero: Out) -> Result<Out, InnerProductError> {
        if point.len() != 1 {
            Err(InnerProductError::LengthMismatch {
                lhs: 1,
                rhs: point.len(),
            })
        } else {
            Ok(Out::from_ref(
                &self
                    .mul_by_scalar(&point[0])
                    .ok_or(InnerProductError::Overflow)?,
            ))
        }
    }
}

impl<const M: usize> InnerProduct<i128, Int<M>> for &[Boolean] {
    /// If we have a slice of booleans we do not need to multiply at all.
    /// The inner product is just a sum!
    fn inner_product(&self, rhs: &[i128], zero: Int<M>) -> Result<Int<M>, InnerProductError> {
        if self.len() != rhs.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: self.len(),
                rhs: rhs.len(),
            });
        }

        (0..self.len())
            .filter(|&i| self[i].into_inner())
            .try_fold(zero, |acc, i| {
                acc.checked_add(&Int::from(rhs[i]))
                    .ok_or(InnerProductError::Overflow)
            })
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
