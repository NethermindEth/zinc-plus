use rand::RngCore;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_uair::Uair;

/// A trait for UAIRs for generating single-typed witness,
/// e.g. consisting of only arbitrary polynomials.
pub trait GenerateSingleTypeWitness: Uair {
    type Witness;

    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<Self::Witness>>;
}
