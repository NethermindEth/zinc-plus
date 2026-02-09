use std::marker::PhantomData;

use crypto_primitives::Semiring;
use zinc_uair::Uair;
use zinc_utils::inner_transparent_field::InnerTransparentField;

pub struct CombinedPolyResolver<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField> CombinedPolyResolver<F> {
    pub fn prove_as_subprotocol<R: Semiring + 'static, U: Uair<R>>() {}
}
