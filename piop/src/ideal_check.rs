use std::{marker::PhantomData, sync::Arc};

use crypto_primitives::{FixedSemiring, Semiring};
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_transcript::traits::Transcript;
use zinc_uair::{ConstraintBuilder, Uair};

pub struct IdealCheckProtocol<R: Semiring>(PhantomData<R>);

impl<R: Semiring> IdealCheckProtocol<R> {
    pub fn prove_as_subprotocol<U: Uair<R>>(
        transcript: &mut impl Transcript,
        cs_up: &[DenseMultilinearExtension<R>],
        cs_down: &[DenseMultilinearExtension<R>],
        num_constraints: usize,
    ) {
    }
}

pub(crate) struct IdealCheckConstraintBuilder<R: Semiring> {
    pub uair_poly_mles: Vec<DenseMultilinearExtension<R>>,
    num_constraints: usize,
    curr_mle: usize,
    idx: usize,
}

impl<R: Semiring> IdealCheckConstraintBuilder<R> {
    pub fn new(idx: usize, num_constraints: usize) -> Self {
        Self {
            uair_poly_mles: Vec::with_capacity(num_constraints),
            curr_mle: 0,
            idx,
            num_constraints,
        }
    }
}

impl<R: FixedSemiring> ConstraintBuilder for IdealCheckConstraintBuilder<R> {
    type Expr = R;

    #[allow(clippy::arithmetic_side_effects)]
    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal_generator: &Self::Expr) {
        self.uair_poly_mles[self.curr_mle % self.num_constraints].evaluations[self.idx] = expr;

        self.curr_mle += 1;
    }
}

pub(crate) struct IdealCollector<R: Semiring, B: ConstraintBuilder> {
    pub ideals: Vec<R>,
    pub inner: B,
}

impl<R: Semiring, B: ConstraintBuilder> IdealCollector<R, B> {
    pub fn new(num_constraints: usize, inner: B) -> Self {
        Self {
            ideals: Vec::with_capacity(num_constraints),
            inner,
        }
    }
}

impl<R, B> ConstraintBuilder for IdealCollector<R, B>
where
    R: FixedSemiring,
    B: ConstraintBuilder<Expr = R>,
{
    type Expr = R;

    fn assert_in_ideal(&mut self, expr: Self::Expr, ideal_generator: &Self::Expr) {
        self.ideals.push(ideal_generator.clone());

        self.inner.assert_in_ideal(expr, ideal_generator);
    }
}
