use std::{marker::PhantomData, ops::Mul};

use crypto_primitives::{FromWithConfig, PrimeField};
use zinc_utils::mul_by_scalar::{MulByScalar, WideningMulByScalar};

use crate::code::iprs::PnttInt;

/// A trait for various wrappers used
/// to pick the right multiplication
/// by twiddles.
pub trait MulByTwiddle<T> {
    fn new_ref(value: &T) -> &Self;
}

impl<T> MulByTwiddle<T> for T {
    fn new_ref(value: &T) -> &Self {
        value
    }
}

#[repr(transparent)]
pub struct ForceWideningMulByTwiddle<T, WideningMBS>(T, PhantomData<WideningMBS>);

impl<T, WideningMBS> MulByTwiddle<T> for ForceWideningMulByTwiddle<T, WideningMBS> {
    #[inline(always)]
    fn new_ref(value: &T) -> &Self {
        // Safety: ForceWideningMulByTwiddle is #[repr(transparent)] and is
        // guaranteed to have the same memory layout as T
        unsafe { &*(value as *const T as *const Self) }
    }
}

impl<T, WideningMBS> Mul<&PnttInt> for &ForceWideningMulByTwiddle<T, WideningMBS>
where
    WideningMBS: WideningMulByScalar<T, PnttInt>,
{
    type Output = WideningMBS::Output;

    fn mul(self, rhs: &PnttInt) -> Self::Output {
        WideningMBS::mul_by_scalar_widen(&self.0, rhs)
    }
}

/// The twiddle multiplication that
/// uses the `MulByScalar` implementation.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub(crate) struct ForceMulByScalar<T>(T);

impl<T> MulByTwiddle<T> for ForceMulByScalar<T> {
    #[inline(always)]
    fn new_ref(value: &T) -> &Self {
        // Safety: ForceMulByScalar is #[repr(transparent)] and is
        // guaranteed to have the same memory layout as T
        unsafe { &*(value as *const T as *const Self) }
    }
}

impl<T> Mul<&PnttInt> for &ForceMulByScalar<T>
where
    T: for<'a> MulByScalar<&'a PnttInt>,
{
    type Output = T;

    fn mul(self, rhs: &PnttInt) -> Self::Output {
        self.0
            .mul_by_scalar(rhs)
            .expect("Twiddle multiplication overflow")
    }
}

/// If we deal with fields we do not have to
/// worry about overflows. Multiplication by twiddle
/// is done by conversions and field operations.
#[derive(Debug, Clone)]
#[repr(transparent)]
pub(crate) struct FieldMulByTwiddle<F: PrimeField>(F);

impl<T: PrimeField> MulByTwiddle<T> for FieldMulByTwiddle<T> {
    #[inline(always)]
    fn new_ref(value: &T) -> &Self {
        // Safety: ForceMulByScalar is #[repr(transparent)] and is
        // guaranteed to have the same memory layout as T
        unsafe { &*(value as *const T as *const Self) }
    }
}

impl<F> Mul<&PnttInt> for &FieldMulByTwiddle<F>
where
    F: PrimeField + FromWithConfig<PnttInt>,
{
    type Output = F;

    fn mul(self, rhs: &PnttInt) -> Self::Output {
        F::from_with_cfg(*rhs, self.0.cfg()) * &self.0
    }
}
