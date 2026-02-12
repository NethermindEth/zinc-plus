use crypto_primitives::Semiring;
use rand::RngCore;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_uair::Uair;

pub trait GenerateSingleTypeWitness: Uair {
    type Witness;

    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<Self::Witness>>;
}
