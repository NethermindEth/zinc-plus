use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::crypto_bigint_monty::MontyField;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::binary::BinaryPoly};
use zinc_test_uair::GenerateWitness;
use zinc_transcript::traits::Transcript;
use zinc_uair::constraint_counter::count_constraints;

use crate::ideal_check::{self, IdealCheckProtocol, ProjectToField};

pub const LIMBS: usize = 4;

pub fn test_config() -> MontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

pub type TestIcField = MontyField<4>;

pub fn run_ideal_check_prover<U, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
    trace: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    transcript: &mut impl Transcript,
) -> (
    ideal_check::Proof<TestIcField>,
    ideal_check::ProverState<TestIcField, BinaryPoly<DEGREE_PLUS_ONE>>,
)
where
    U: GenerateWitness<BinaryPoly<DEGREE_PLUS_ONE>>,
    BinaryPoly<DEGREE_PLUS_ONE>: ProjectToField<TestIcField>,
{
    let field_cfg = test_config();

    let num_constraints = count_constraints::<BinaryPoly<DEGREE_PLUS_ONE>, U>();

    IdealCheckProtocol::<TestIcField>::prove_as_subprotocol::<BinaryPoly<DEGREE_PLUS_ONE>, U>(
        transcript,
        trace,
        num_constraints,
        num_vars,
        &field_cfg,
    )
    .unwrap()
}
