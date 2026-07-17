#![allow(clippy::type_complexity, clippy::arithmetic_side_effects)]

use crate::{
    ideal_check::{IdealCheckProtocol, Proof as IdealCheckProof},
    projections::{
        ColumnMajorTrace, ProjectedScalars, RowMajorTrace, project_scalars,
        project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
};
use crypto_primitives::{
    BaseFieldConfig, ProjectElementWithConfig,
    crypto_bigint_int::Int,
    crypto_bigint_monty::{MontyField, MontyFieldElement},
    crypto_bigint_uint::Uint,
};
use zinc_poly::univariate::{dense::DensePolynomial, dynamic::DynamicPolynomial};
use zinc_test_uair::GenerateRandomTrace;
use zinc_transcript::traits::Transcript;
use zinc_uair::{Uair, UairTrace, constraint_counter::count_constraints};

pub const LIMBS: usize = 4;

pub fn test_config() -> MontyField<LIMBS> {
    let modulus =
        Uint::from_be_hex("0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33");
    MontyField::new(&modulus).expect("modulus should be a valid odd prime")
}

type F = MontyField<4>;
type E = MontyFieldElement<4>;

/// Run ideal check prover using MLE-first approach (for linear constraints).
/// Uses column-indexed trace.
pub fn run_ideal_check_prover_linear<U, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
    trace: &UairTrace<Int<5>, Int<5>, DEGREE_PLUS_ONE, DEGREE_PLUS_ONE>,
    prime_idx: Option<usize>,
    transcript: &mut impl Transcript,
) -> (
    IdealCheckProof<E>,
    Vec<E>,
    ProjectedScalars<U::Scalar, DynamicPolynomial<E>>,
    ColumnMajorTrace<E>,
)
where
    U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>,
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

    let scalars = project_scalars::<F, U>(&field_cfg, |scalar| {
        scalar
            .iter()
            .map(|coeff| field_cfg.project(coeff))
            .collect()
    });

    let trace: ColumnMajorTrace<E> = project_trace_coeffs_column_major(trace, &field_cfg);

    let evaluation_point: Vec<E> = transcript.get_field_challenges(num_vars, &field_cfg);

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
    IdealCheckProof<E>,
    Vec<E>,
    ProjectedScalars<U::Scalar, DynamicPolynomial<E>>,
    RowMajorTrace<E>,
)
where
    U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>,
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

    let scalars = project_scalars::<F, U>(&field_cfg, |scalar| {
        scalar
            .iter()
            .map(|coeff| field_cfg.project(coeff))
            .collect()
    });

    let trace: RowMajorTrace<E> = project_trace_coeffs_row_major(trace, &field_cfg);

    let evaluation_point: Vec<E> = transcript.get_field_challenges(num_vars, &field_cfg);

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
