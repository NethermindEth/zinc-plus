use crypto_primitives::{FromWithConfig, PrimeField, Semiring};
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;
use zinc_poly::{
    EvaluatablePolynomial, EvaluationError,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_uair::{Uair, collect_scalars::collect_scalars};
use zinc_utils::{
    cfg_extend, cfg_iter, from_ref::FromRef, projectable_to_field::ProjectableToField,
};

/// Project a multi-typed trace onto `F[X]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn project_trace_coeffs<F, PolyCoeff, Int, const DEGREE_PLUS_ONE: usize>(
    binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    arbitrary_poly_trace: &[DenseMultilinearExtension<
        DensePolynomial<PolyCoeff, DEGREE_PLUS_ONE>,
    >],
    int_trace: &[DenseMultilinearExtension<Int>],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>>
where
    F: FromWithConfig<PolyCoeff> + FromWithConfig<Int> + Send + Sync,
    PolyCoeff: Clone + Send + Sync,
    Int: Clone + Send + Sync,
{
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let mut result =
        Vec::with_capacity(binary_poly_trace.len() + arbitrary_poly_trace.len() + int_trace.len());

    cfg_extend!(
        result,
        cfg_iter!(binary_poly_trace).map(|column| {
            cfg_iter!(column)
                .map(|binary_poly| {
                    binary_poly
                        .iter()
                        .map(|coeff| {
                            if coeff.into_inner() {
                                one.clone()
                            } else {
                                zero.clone()
                            }
                        })
                        .collect()
                })
                .collect()
        })
    );

    cfg_extend!(
        result,
        cfg_iter!(arbitrary_poly_trace).map(|column| {
            cfg_iter!(column)
                .map(|arbitrary_poly| {
                    arbitrary_poly
                        .iter()
                        .map(|coeff| F::from_with_cfg(coeff.clone(), field_cfg))
                        .collect()
                })
                .collect()
        })
    );

    cfg_extend!(
        result,
        cfg_iter!(int_trace).map(|column| {
            cfg_iter!(column)
                .map(|int| DynamicPolynomialF {
                    coeffs: vec![F::from_with_cfg(int.clone(), field_cfg)],
                })
                .collect()
        })
    );

    result
}

/// Project a multi-typed traces along F[X]->F.
/// Note, that we do not need Montgomery forms for the
/// binary polynomials as they can be projected just
/// using a field config. Projecting int trace is just
/// turning constant `DynamicPolynomialF`s into `F`s.
/// The real work may happen only for the arbitrary polynomials
/// which we have no better solution than just evaluate them on
/// the projecting element.
#[allow(clippy::arithmetic_side_effects)]
pub fn project_trace_to_field<F: PrimeField + FromRef<F> + 'static, const DEGREE_PLUS_ONE: usize>(
    binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    arbitrary_poly_trace: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
    int_trace: &[DenseMultilinearExtension<DynamicPolynomialF<F>>],
    projecting_element: &F,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F::Inner: Default,
{
    let binary_poly_projection =
        BinaryPoly::<DEGREE_PLUS_ONE>::prepare_projection(projecting_element);

    let mut result =
        Vec::with_capacity(binary_poly_trace.len() + arbitrary_poly_trace.len() + int_trace.len());

    cfg_extend!(
        result,
        cfg_iter!(binary_poly_trace).map(|column| {
            cfg_iter!(column)
                .map(|poly| binary_poly_projection(poly).inner().clone())
                .collect()
        })
    );

    cfg_extend!(
        result,
        cfg_iter!(arbitrary_poly_trace).map(|column| {
            cfg_iter!(column)
                .map(|poly| {
                    poly.evaluate_at_point(projecting_element)
                        .expect("dynamic poly evaluation does not fail")
                        .inner()
                        .clone()
                })
                .collect()
        })
    );

    result.extend(int_trace.iter().map(|column| {
        column
            .iter()
            .map(|i| {
                if i.coeffs.is_empty() {
                    F::Inner::default()
                } else {
                    i.coeffs[0].inner().clone()
                }
            })
            .collect()
    }));

    result
}

/// Project scalars of a UAIR onto F[X].
pub fn project_scalars<F: PrimeField, U: Uair>(
    project: impl Fn(&U::Scalar) -> DynamicPolynomialF<F>,
) -> HashMap<U::Scalar, DynamicPolynomialF<F>> {
    let uair_scalars = collect_scalars::<U>();

    // TODO(Ilia): if there's a lot of scalars
    //             we should do this in parallel probably.
    uair_scalars
        .into_iter()
        .map(|scalar| {
            (scalar.clone(), {
                let mut dynamic_poly = project(&scalar);

                dynamic_poly.trim();

                dynamic_poly
            })
        })
        .collect()
}

/// Project scalars of a UAIR along F[X] -> F.
pub fn project_scalars_to_field<R: Semiring + 'static, F: PrimeField>(
    scalars: HashMap<R, DynamicPolynomialF<F>>,
    projecting_element: &F,
) -> Result<HashMap<R, F>, (R, F, EvaluationError)> {
    // TODO(Ilia): Parallelising this might be good for big UAIRs.
    //             We'd conditionally route between sequential and parallel
    //             projection depending on how many scalars the UAIR has.
    scalars
        .into_iter()
        .map(
            |(scalar, value)| -> Result<(R, F), (R, F, EvaluationError)> {
                Ok((
                    scalar.clone(),
                    value
                        .evaluate_at_point(projecting_element)
                        .map_err(|err| (scalar.clone(), projecting_element.clone(), err))?,
                ))
            },
        )
        .try_collect()
}
