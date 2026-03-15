use std::collections::HashMap;

use crate::{
    ideal_check::{IdealCheckProtocol, Proof, ProverState},
    projections::{
        ColumnMajorTrace, RowMajorTrace, project_scalars, project_trace_coeffs_column_major,
        project_trace_coeffs_row_major,
    },
};
use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::{FromWithConfig, crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
use num_traits::Zero;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF},
};
use zinc_test_uair::GenerateSingleTypeWitness;
use zinc_transcript::traits::Transcript;
use zinc_uair::{Uair, constraint_counter::count_constraints};

pub const LIMBS: usize = 4;

pub fn test_config() -> MontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

type F = MontyField<4>;

/// Run ideal check prover using MLE-first approach (for linear constraints).
/// Uses column-indexed trace.
#[allow(clippy::type_complexity)]
pub fn run_ideal_check_prover_linear<U, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
    trace: &[DenseMultilinearExtension<DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>],
    transcript: &mut impl Transcript,
) -> (
    Proof<F>,
    ProverState<F>,
    HashMap<U::Scalar, DynamicPolynomialF<F>>,
    ColumnMajorTrace<F>,
)
where
    U: GenerateSingleTypeWitness<Witness = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + IdealCheckProtocol,
    F: FromWithConfig<Int<5>>,
{
    assert!(
        U::signature().binary_poly_cols.is_zero() && U::signature().int_cols.is_zero(),
        "the signature should be single typed"
    );

    let field_cfg = test_config();

    let num_constraints = count_constraints::<U>();

    let scalars = project_scalars::<F, U>(|scalar| {
        scalar
            .iter()
            .map(|coeff| F::from_with_cfg(coeff, &field_cfg))
            .collect()
    });

    let trace = project_trace_coeffs_column_major::<F, Int<5>, Int<5>, DEGREE_PLUS_ONE>(
        &[],
        trace,
        &[],
        &field_cfg,
    );

    let (proof, state) = U::prove_linear(
        transcript,
        &trace,
        &scalars,
        num_constraints,
        num_vars,
        &field_cfg,
    )
    .unwrap();

    (proof, state, scalars, trace)
}

/// Run ideal check prover using combined polynomial approach (for any
/// constraints). Uses row-indexed (transposed) trace.
#[allow(clippy::type_complexity)]
pub fn run_ideal_check_prover_combined<U, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
    trace: &[DenseMultilinearExtension<DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>],
    transcript: &mut impl Transcript,
) -> (
    Proof<F>,
    ProverState<F>,
    HashMap<U::Scalar, DynamicPolynomialF<F>>,
    RowMajorTrace<F>,
)
where
    U: GenerateSingleTypeWitness<Witness = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + IdealCheckProtocol,
    F: FromWithConfig<Int<5>>,
{
    assert!(
        U::signature().binary_poly_cols.is_zero() && U::signature().int_cols.is_zero(),
        "the signature should be single typed"
    );

    let field_cfg = test_config();

    let num_constraints = count_constraints::<U>();

    let scalars = project_scalars::<F, U>(|scalar| {
        scalar
            .iter()
            .map(|coeff| F::from_with_cfg(coeff, &field_cfg))
            .collect()
    });

    let trace = project_trace_coeffs_row_major::<F, Int<5>, Int<5>, DEGREE_PLUS_ONE>(
        &[],
        trace,
        &[],
        &field_cfg,
    );

    let (proof, state) = U::prove_combined(
        transcript,
        &trace,
        &scalars,
        num_constraints,
        num_vars,
        &field_cfg,
    )
    .unwrap();

    (proof, state, scalars, trace)
}
