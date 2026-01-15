use std::fmt::Debug;
use std::marker::PhantomData;

use crypto_primitives::{FixedSemiring, Semiring};

pub trait Ideal<R: Semiring>: Clone + Debug + Send + Sync {
    fn zero_ideal() -> Self;

    fn contains(&self, elem: R) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub struct ZeroIdeal<R: FixedSemiring>(PhantomData<R>);

impl<R: FixedSemiring> Ideal<R> for ZeroIdeal<R> {
    #[inline(always)]
    fn zero_ideal() -> Self {
        ZeroIdeal(Default::default())
    }

    #[inline(always)]
    fn contains(&self, elem: R) -> bool {
        elem.is_zero()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DummyIdeal<R: Semiring, I: Ideal<R>>(PhantomData<(R, I)>);

impl<R: Semiring, I: Ideal<R>> Ideal<R> for DummyIdeal<R, I> {
    #[inline(always)]
    fn zero_ideal() -> Self {
        DummyIdeal(Default::default())
    }

    #[inline(always)]
    fn contains(&self, _elem: R) -> bool {
        false
    }
}

impl<R: Semiring, I: Ideal<R>> From<I> for DummyIdeal<R, I> {
    #[inline(always)]
    fn from(_ideal: I) -> Self {
        DummyIdeal(Default::default())
    }
}
