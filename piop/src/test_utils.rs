#![allow(clippy::type_complexity, clippy::arithmetic_side_effects)]

use crate::{
    ideal_check::{IdealCheckProtocol, Proof as IdealCheckProof},
    projections::{
        ColumnMajorTrace, ProjectedScalars, RowMajorTrace, project_scalars,
        project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
};
use crypto_bigint::{Odd, modular::FixedMontyParams};
use crypto_primitives::{FromWithConfig, crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
use zinc_poly::univariate::{dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF};
use zinc_test_uair::GenerateRandomTrace;
use zinc_transcript::traits::Transcript;
use zinc_uair::{Uair, UairTrace, constraint_counter::count_constraints};

pub const LIMBS: usize = 4;

pub fn test_config() -> FixedMontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    FixedMontyParams::new(modulus)
}

type F = MontyField<4>;

/// Run ideal check prover using MLE-first approach (for linear constraints).
/// Uses column-indexed trace.
pub fn run_ideal_check_prover_linear<U, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
    trace: &UairTrace<Int<5>, Int<5>, DEGREE_PLUS_ONE, DEGREE_PLUS_ONE>,
    prime_idx: Option<usize>,
    transcript: &mut impl Transcript,
) -> (
    IdealCheckProof<F>,
    Vec<F>,
    ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
    ColumnMajorTrace<F>,
)
where
    U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>,
    F: FromWithConfig<Int<5>>,
{
    // These helpers intentionally accept mixed-type signatures. The projection
    // layer handles binary_poly, arbitrary_poly, and int columns uniformly, and
    // mixed fixtures are needed to regression-test down-row splicing.
    let field_cfg = test_config();

    let num_constraints = count_constraints::<U>();
    let num_constraints = prime_idx
        .map(|i| num_constraints.for_prime(i))
        .unwrap_or(num_constraints.q);
    let family_idx = prime_idx.map_or(0, |i| i + 1);

    let scalars = project_scalars::<F, U>(|scalar| {
        scalar
            .iter()
            .map(|coeff| F::from_with_cfg(coeff, &field_cfg))
            .collect()
    });

    let trace: ColumnMajorTrace<F> = project_trace_coeffs_column_major(trace, &field_cfg);

    let evaluation_point: Vec<F> = transcript.get_field_challenges(num_vars, &field_cfg);

    let proof = IdealCheckProtocol::<U>::prove_mle_first::<_, DEGREE_PLUS_ONE>(
        transcript,
        &trace,
        &scalars,
        family_idx,
        num_constraints,
        &evaluation_point,
        &field_cfg,
    )
    .unwrap();

    (proof, evaluation_point, scalars, trace)
}

/// Run ideal check prover using combined polynomial approach (for any
/// constraints). Uses row-indexed (transposed) trace.
pub fn run_ideal_check_prover_combined<U, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
    trace: &UairTrace<Int<5>, Int<5>, DEGREE_PLUS_ONE, DEGREE_PLUS_ONE>,
    prime_idx: Option<usize>,
    transcript: &mut impl Transcript,
) -> (
    IdealCheckProof<F>,
    Vec<F>,
    ProjectedScalars<U::Scalar, DynamicPolynomialF<F>>,
    RowMajorTrace<F>,
)
where
    U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>,
    F: FromWithConfig<Int<5>>,
{
    // These helpers intentionally accept mixed-type signatures. The projection
    // layer handles binary_poly, arbitrary_poly, and int columns uniformly, and
    // mixed fixtures are needed to regression-test down-row splicing.
    let field_cfg = test_config();

    let num_constraints = count_constraints::<U>();
    let num_constraints = prime_idx
        .map(|i| num_constraints.for_prime(i))
        .unwrap_or(num_constraints.q);
    let family_idx = prime_idx.map_or(0, |i| i + 1);

    let scalars = project_scalars::<F, U>(|scalar| {
        scalar
            .iter()
            .map(|coeff| F::from_with_cfg(coeff, &field_cfg))
            .collect()
    });

    let trace: RowMajorTrace<F> = project_trace_coeffs_row_major(trace, &field_cfg);

    let evaluation_point: Vec<F> = transcript.get_field_challenges(num_vars, &field_cfg);

    let proof = IdealCheckProtocol::<U>::prove_combined::<_, DEGREE_PLUS_ONE>(
        transcript,
        &trace,
        &scalars,
        family_idx,
        num_constraints,
        &evaluation_point,
        &field_cfg,
    )
    .unwrap();

    (proof, evaluation_point, scalars, trace)
}
