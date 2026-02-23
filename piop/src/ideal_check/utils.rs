use crate::ideal_check::IdealCheckField;
use crypto_primitives::{FromWithConfig, Semiring, crypto_bigint_int::Int};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use zinc_poly::{
    CoefficientProjectable,
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF},
};
use zinc_uair::{Uair, collect_scalars::collect_scalars};
use zinc_utils::cfg_iter;

/// A ring type whose elements can be projected to dynamic polynomials over a
/// prime field `F`. This abstracts over the specific polynomial representation
/// (e.g. `BinaryPoly<D>` or `DensePolynomial<i64, D>`) and allows the
/// `IdealCheckProtocol` to operate generically on any projectable ring.
pub trait ProjectToField<F: IdealCheckField>: Semiring + 'static {
    fn project_to_field(&self, projecting_element: &F) -> DynamicPolynomialF<F>;
}

/// BinaryPoly<D> can be projected to any IdealCheckField since IdealCheckField
/// requires `FromWithConfig<Boolean>`.
impl<F: IdealCheckField, const D: usize> ProjectToField<F> for BinaryPoly<D> {
    fn project_to_field(&self, projecting_element: &F) -> DynamicPolynomialF<F> {
        self.project_coefficients(projecting_element).into()
    }
}

/// DensePolynomial<i64, D> can be projected to any IdealCheckField that
/// implements `FromWithConfig<i64>`.
impl<F: IdealCheckField + FromWithConfig<i64>, const D: usize> ProjectToField<F>
    for DensePolynomial<i64, D>
{
    fn project_to_field(&self, projecting_element: &F) -> DynamicPolynomialF<F> {
        self.project_coefficients(projecting_element).into()
    }
}

/// Int<LIMBS> (256-bit integer, etc.) can be projected to any IdealCheckField
/// that implements `FromWithConfig<Int<LIMBS>>`. Since Int<N> is a degree-0
/// polynomial (a scalar), projection produces a 1-coefficient DynamicPolynomialF.
impl<F: IdealCheckField + FromWithConfig<Int<LIMBS>>, const LIMBS: usize> ProjectToField<F>
    for Int<LIMBS>
{
    fn project_to_field(&self, projecting_element: &F) -> DynamicPolynomialF<F> {
        self.project_coefficients(projecting_element).into()
    }
}

/// Apply projection to coefficients of coefficients of the input trace.
pub(crate) fn project_trace_matrix<F: IdealCheckField, R: ProjectToField<F>>(
    trace: &[DenseMultilinearExtension<R>],
    projecting_element: &F,
) -> Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>> {
    cfg_iter!(trace)
        .map(|trace_col| {
            cfg_iter!(trace_col)
                .map(|x| x.project_to_field(projecting_element))
                .collect()
        })
        .collect()
}

pub(crate) fn project_scalars<F: IdealCheckField, U: Uair<R>, R: ProjectToField<F>>(
    projecting_element: &F,
) -> HashMap<R, DynamicPolynomialF<F>> {
    let uair_scalars = collect_scalars::<R, U>();

    // TODO(Ilia): if there's a lot of scalars
    //             we should do this in parallel probably.
    uair_scalars
        .into_iter()
        .map(|scalar| {
            let mut dynamic_poly = scalar.project_to_field(projecting_element);
            dynamic_poly.trim();
            (scalar, dynamic_poly)
        })
        .collect()
}
