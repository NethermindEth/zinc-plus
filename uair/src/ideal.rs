use std::fmt::Debug;
use std::marker::PhantomData;

use crypto_primitives::{FixedSemiring, Semiring};
use zinc_utils::from_ref::FromRef;

pub trait Ideal: FromRef<Self> + Clone + Debug + Send + Sync {
    fn zero_ideal() -> Self;
}

pub trait IdealCheck<I: Ideal> {
    fn is_contained_in_with_zero(&self, ideal: &I, zero: &Self) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub struct ZeroIdeal;

impl Ideal for ZeroIdeal {
    #[inline(always)]
    fn zero_ideal() -> Self {
        ZeroIdeal
    }
}

impl<R: Semiring> IdealCheck<ZeroIdeal> for R {
    #[inline(always)]
    fn is_contained_in_with_zero(&self, _ideal: &ZeroIdeal, zero: &R) -> bool {
        self == zero
    }
}

impl FromRef<ZeroIdeal> for ZeroIdeal {
    #[inline(always)]
    fn from_ref(_ideal: &ZeroIdeal) -> Self {
        ZeroIdeal
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DummyIdeal;

impl Ideal for DummyIdeal {
    #[inline(always)]
    fn zero_ideal() -> Self {
        DummyIdeal
    }
}

impl<R: Semiring> IdealCheck<DummyIdeal> for R {
    #[inline(always)]
    fn is_contained_in_with_zero(&self, _ideal: &DummyIdeal, _zero: &R) -> bool {
        false
    }
}

impl<I: Ideal> FromRef<I> for DummyIdeal {
    #[inline(always)]
    fn from_ref(_ideal: &I) -> Self {
        DummyIdeal
    }
}
