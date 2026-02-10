use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_test_uair::GenerateWitness;
use zinc_transcript::traits::Transcript;
use zinc_uair::constraint_counter::count_constraints;

use crate::ideal_check::{self, IdealCheckProtocol, IdealCheckTypes};

pub const LIMBS: usize = 4;

pub fn test_config() -> MontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

pub struct TestIcTypes;

impl<const DEGREE_PLUS_ONE: usize> IdealCheckTypes<DEGREE_PLUS_ONE> for TestIcTypes {
    type WitnessCoeff = Int<5>;
    type Witness = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>;

    type F = MontyField<4>;
}

pub fn run_ideal_check_prover<U, const DEGREE_PLUS_ONE: usize>(
    num_vars: usize,
    trace: &[DenseMultilinearExtension<DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>],
    transcript: &mut impl Transcript,
) -> (
    ideal_check::Proof<TestIcTypes, DEGREE_PLUS_ONE>,
    ideal_check::ProverState<TestIcTypes, DEGREE_PLUS_ONE>,
)
where
    U: GenerateWitness<DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>,
{
    let field_cfg = test_config();

    let num_constraints = count_constraints::<<TestIcTypes as IdealCheckTypes<_>>::Witness, U>();

    IdealCheckProtocol::<TestIcTypes, _>::prove_as_subprotocol::<U>(
        transcript,
        trace,
        num_constraints,
        num_vars,
        &field_cfg,
    )
    .unwrap()
}
