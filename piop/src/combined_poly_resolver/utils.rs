use itertools::Itertools;
use std::collections::HashMap;

use crypto_primitives::{PrimeField, Semiring};
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;

use crate::combined_poly_resolver::CombinedPolyResolverError;

pub(crate) fn project_scalars_to_field<R: Semiring + 'static, F: PrimeField>(
    scalars: HashMap<R, DynamicPolynomialF<F>>,
    projection_powers: &[F],
    projecting_element: &F,
) -> Result<HashMap<R, F>, CombinedPolyResolverError<F>> {
    scalars
        .into_iter()
        .map(
            |(scalar, value)| -> Result<(R, F), CombinedPolyResolverError<F>> {
                Ok((
                    scalar,
                    value
                        .evaluate_with_powers(projection_powers)
                        .map_err(|err| {
                            CombinedPolyResolverError::ProjectionError(
                                value.clone(),
                                projecting_element.clone(),
                                err,
                            )
                        })?,
                ))
            },
        )
        .try_collect()
}
