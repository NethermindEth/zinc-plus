#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{FromWithConfig, PrimeField};
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_utils::{cfg_extend, cfg_iter};

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
    F: FromWithConfig<PolyCoeff> + FromWithConfig<Int>,
    PolyCoeff: Clone,
    Int: Clone,
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
                    binary_poly.map_coeffs(|coeff| if coeff { one.clone() } else { zero.clone() })
                })
                .collect()
        })
    );

    cfg_extend!(
        result,
        cfg_iter!(arbitrary_poly_trace).map(|column| {
            cfg_iter!(column)
                .map(|arbitrary_poly| {
                    arbitrary_poly.map_coeffs(|coeff| F::from_with_cfg(coeff.clone(), field_cfg))
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
