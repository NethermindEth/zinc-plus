use std::ops::{Add, Mul};

use crypto_primitives::crypto_bigint_int::Int;
use num_traits::{CheckedAdd, CheckedMul, One, Zero};
use thiserror::Error;

use crate::{from_ref::FromRef, mul_by_scalar::MulByScalar};

/// A trait for types that can be dot-multiplied.
pub trait InnerProduct<Rhs, Output> {
    /// The main entry point for the inner product.
    fn inner_product(&self, rhs: &[Rhs], zero: Output) -> Result<Output, InnerProductError>;
}

/// A trait for types that can be dot-multiplied ignoring integer overflows.
pub trait InnerProductUnchecked<Rhs, Output> {
    /// The main entry point for the inner product.
    fn inner_product_unchecked(
        &self,
        rhs: &[Rhs],
        zero: Output,
    ) -> Result<Output, InnerProductError>;
}

/// A trait for types that can be dot-multiplied ignoring integer overflows
/// but with widened lhs and rhs values.
pub trait WideningInnerProduct<Rhs, Output> {
    /// The main entry point for the inner product.
    fn widening_inner_product(
        &self,
        rhs: &[Rhs],
        zero: Output,
    ) -> Result<Output, InnerProductError>;
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
        if self.len() != rhs.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: self.len(),
                rhs: rhs.len(),
            });
        }

        self.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs.mul_by_scalar(rhs).ok_or(InnerProductError::Overflow))
            .try_fold(zero, |acc, product| {
                acc.checked_add(&product?.into())
                    .ok_or(InnerProductError::Overflow)
            })
    }
}

impl<Lhs, Rhs, Out> InnerProductUnchecked<Rhs, Out> for &[Lhs]
where
    Lhs: Clone + for<'a> Mul<&'a Rhs, Output = Lhs>,
    Out: for<'a> Add<&'a Lhs, Output = Out>,
{
    #[allow(clippy::arithmetic_side_effects)] // by design
    fn inner_product_unchecked(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        if self.len() != rhs.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: self.len(),
                rhs: rhs.len(),
            });
        }

        Ok(self
            .iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs.clone() * rhs)
            .fold(zero, |acc, product| acc + &product))
    }
}

impl<Lhs, Rhs, Out> WideningInnerProduct<Rhs, Out> for &[Lhs]
where
    Out: FromRef<Lhs>
        + FromRef<Rhs>
        + for<'a> Add<&'a Out, Output = Out>
        + for<'a> Mul<&'a Out, Output = Out>,
{
    #[allow(clippy::arithmetic_side_effects)] // by design
    fn widening_inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        if self.len() != rhs.len() {
            return Err(InnerProductError::LengthMismatch {
                lhs: self.len(),
                rhs: rhs.len(),
            });
        }

        Ok(self
            .iter()
            .zip(rhs)
            .map(|(lhs, rhs)| Out::from_ref(lhs) * &(Out::from_ref(rhs)))
            .fold(zero, |acc, product| acc + &product))
    }
}

impl<Lhs, Rhs, Out> InnerProduct<Rhs, Out> for Vec<Lhs>
where
    for<'a> &'a [Lhs]: InnerProduct<Rhs, Out>,
{
    #[inline(always)]
    fn inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.as_slice().inner_product(rhs, zero)
    }
}

impl<Lhs, Rhs, Out> InnerProductUnchecked<Rhs, Out> for Vec<Lhs>
where
    for<'a> &'a [Lhs]: InnerProductUnchecked<Rhs, Out>,
{
    #[inline(always)]
    fn inner_product_unchecked(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.as_slice().inner_product_unchecked(rhs, zero)
    }
}

impl<Lhs, Rhs, Out> WideningInnerProduct<Rhs, Out> for Vec<Lhs>
where
    for<'a> &'a [Lhs]: WideningInnerProduct<Rhs, Out>,
{
    #[inline(always)]
    fn widening_inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.as_slice().widening_inner_product(rhs, zero)
    }
}

impl<Lhs, Rhs, Out, const N: usize> InnerProduct<Rhs, Out> for [Lhs; N]
where
    for<'a> &'a [Lhs]: InnerProduct<Rhs, Out>,
{
    #[inline(always)]
    fn inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.as_slice().inner_product(rhs, zero)
    }
}

impl<Lhs, Rhs, Out, const N: usize> InnerProductUnchecked<Rhs, Out> for [Lhs; N]
where
    for<'a> &'a [Lhs]: InnerProductUnchecked<Rhs, Out>,
{
    #[inline(always)]
    fn inner_product_unchecked(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.as_slice().inner_product_unchecked(rhs, zero)
    }
}

impl<Lhs, Rhs, Out, const N: usize> WideningInnerProduct<Rhs, Out> for [Lhs; N]
where
    for<'a> &'a [Lhs]: WideningInnerProduct<Rhs, Out>,
{
    #[inline(always)]
    fn widening_inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.as_slice().widening_inner_product(rhs, zero)
    }
}

macro_rules! impl_inner_product_for_primitives {
    ($($t:ty),*) => {
       $(
           impl<T, Out> InnerProduct<T, Out> for $t
           where
               T: Zero + One + PartialEq,
               Out: Zero + One + From<$t> + for<'a> From<&'a T> + CheckedMul,
           {
                fn inner_product(&self, point: &[T], _zero: Out) -> Result<Out, InnerProductError> {
                        if point.len() != 1 {
                            return Err(InnerProductError::LengthMismatch {
                                lhs: 1,
                                rhs: point.len(),
                            });
                        }

                        if point[0].is_one() {
                            return Ok(Out::from(*self))
                        }


                        if point[0].is_zero() {
                            return Ok(Out::zero());
                        }


                        Out::from(*self).checked_mul(&Out::from(&point[0])).ok_or(InnerProductError::Overflow)

                    }
           }

           impl<T, Out> InnerProductUnchecked<T, Out> for $t
           where
               T: Zero + One + PartialEq,
               Out: Zero + One + From<$t> + for<'a> From<&'a T> + for<'a> Mul<&'a Out, Output = Out>,
           {
                #[allow(clippy::arithmetic_side_effects)]
                fn inner_product_unchecked(&self, point: &[T], _zero: Out) -> Result<Out, InnerProductError> {
                        if point.len() != 1 {
                            return Err(InnerProductError::LengthMismatch {
                                lhs: 1,
                                rhs: point.len(),
                            });
                        }

                        if point[0].is_one() {
                            return Ok(Out::from(*self))
                        }


                        if point[0].is_zero() {
                            return Ok(Out::zero());
                        }


                        Ok(Out::from(*self) * &(Out::from(&point[0])))
                    }
           }
       )*
    };
}

impl_inner_product_for_primitives!(i8, i16, i32, i64, i128);

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

pub trait InnerProductWrapper<T> {
    fn new_ref(value: &T) -> &Self;
}

impl<T> InnerProductWrapper<T> for T {
    fn new_ref(value: &T) -> &Self {
        value
    }
}

/// A zero cost newtype wrapper for forcing
/// unchecked inner product implementation at type level.
///
/// If a type `T` implements `InnerProductUnchecked` just implement
/// `Borrow<ForceUncheckedInnerProduct<T>>` for it and you can
/// use `ForceUncheckedInnerProduct<T>` where `zip-plus` expects
/// something inner-productable.
#[repr(transparent)]
pub struct ForceUncheckedInnerProduct<T>(T);

impl<T> InnerProductWrapper<T> for ForceUncheckedInnerProduct<T> {
    #[inline(always)]
    fn new_ref(value: &T) -> &Self {
        // Safety: ForceUncheckedInnerProduct<T> is #[repr(transparent)] and is
        // guaranteed to have the same memory layout as T
        unsafe { &*(value as *const T as *const Self) }
    }
}

impl<T, Rhs, Out> InnerProduct<Rhs, Out> for ForceUncheckedInnerProduct<T>
where
    T: InnerProductUnchecked<Rhs, Out>,
{
    #[inline(always)]
    fn inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.0.inner_product_unchecked(rhs, zero)
    }
}

/// A zero cost newtype wrapper for forcing
/// unchecked widening inner product implementation at type level.
///
/// If a type `T` implements `WideningInnerProduct` just implement
/// `Borrow<ForceWideningInnerProduct<T>>` for it and you can
/// use `ForceWideningInnerProduct<T>` where `zip-plus` expects
/// something inner-productable.
#[repr(transparent)]
pub struct ForceWideningInnerProduct<T>(T);

impl<T> InnerProductWrapper<T> for ForceWideningInnerProduct<T> {
    #[inline(always)]
    fn new_ref(value: &T) -> &Self {
        // Safety: ForceWideningInnerProduct is #[repr(transparent)] and is
        // guaranteed to have the same memory layout as T
        unsafe { &*(value as *const T as *const Self) }
    }
}

impl<T, Rhs, Out> InnerProduct<Rhs, Out> for ForceWideningInnerProduct<T>
where
    T: WideningInnerProduct<Rhs, Out>,
{
    #[inline(always)]
    fn inner_product(&self, rhs: &[Rhs], zero: Out) -> Result<Out, InnerProductError> {
        self.0.widening_inner_product(rhs, zero)
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
