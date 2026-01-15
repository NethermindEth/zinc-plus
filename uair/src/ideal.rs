use std::fmt::Debug;
use std::marker::PhantomData;

use crypto_primitives::{FixedSemiring, Semiring};
use zinc_utils::from_ref::FromRef;

pub trait Ideal<R: Semiring>: FromRef<Self> + Clone + Debug + Send + Sync {
    fn zero_ideal() -> Self;

    fn contains(&self, elem: &R) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub struct ZeroIdeal<R: FixedSemiring>(PhantomData<R>);

impl<R: FixedSemiring> Ideal<R> for ZeroIdeal<R> {
    #[inline(always)]
    fn zero_ideal() -> Self {
        ZeroIdeal(Default::default())
    }

    #[inline(always)]
    fn contains(&self, elem: &R) -> bool {
        elem.is_zero()
    }
}

impl<R: FixedSemiring> FromRef<ZeroIdeal<R>> for ZeroIdeal<R> {
    #[inline(always)]
    fn from_ref(ideal: &ZeroIdeal<R>) -> Self {
        ideal.clone()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DummyIdeal<R: Semiring>(PhantomData<R>);

impl<R: Semiring> Ideal<R> for DummyIdeal<R> {
    #[inline(always)]
    fn zero_ideal() -> Self {
        DummyIdeal(Default::default())
    }

    #[inline(always)]
    fn contains(&self, _elem: &R) -> bool {
        false
    }
}

impl<R: Semiring, I: Ideal<R>> FromRef<I> for DummyIdeal<R> {
    #[inline(always)]
    fn from_ref(_ideal: &I) -> Self {
        DummyIdeal(Default::default())
    }
}
