use std::collections::HashMap;

use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::{FromWithConfig, crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
use num_traits::Zero;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF},
};
use zinc_test_uair::GenerateSingleTypeWitness;
use zinc_transcript::traits::Transcript;
use zinc_uair::{Uair, collect_scalars::collect_scalars, constraint_counter::count_constraints};

use crate::{
    ideal_check::{self, IdealCheckProtocol},
    projections::{project_scalars, project_trace_coeffs},
};

pub const LIMBS: usize = 4;

pub fn test_config() -> MontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

type WitnessCoeff = Int<5>;
type Witness = DensePolynomial<Int<5>, 32>;

type F = MontyField<4>;

pub fn run_ideal_check_prover_single_type<U, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
    trace: &[DenseMultilinearExtension<DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>],
    transcript: &mut impl Transcript,
) -> (
    ideal_check::Proof<F>,
    ideal_check::ProverState<F>,
    HashMap<U::Scalar, DynamicPolynomialF<F>>,
    Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>>,
)
where
    U: GenerateSingleTypeWitness<Witness = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
        + Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>,
    F: FromWithConfig<Int<5>>,
{
    assert!(
        U::signature().binary_poly_cols.is_zero() && U::signature().int_cols.is_zero(),
        "the signature should be single ttyped"
    );

    let field_cfg = test_config();

    let num_constraints = count_constraints::<U>();

    let scalars = project_scalars::<F, U>(|scalar| {
        scalar.map_coeffs(|coeff| F::from_with_cfg(coeff, &field_cfg))
    });

    let trace =
        project_trace_coeffs::<F, Int<5>, Int<5>, DEGREE_PLUS_ONE>(&[], trace, &[], &field_cfg);

    let (proof, state) = IdealCheckProtocol::prove_as_subprotocol::<U>(
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
