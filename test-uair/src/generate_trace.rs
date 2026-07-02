use rand::Rng;
use zinc_uair::{Uair, UairTrace};

/// A trait for UAIRs for generating random trace (for both public and witness
/// columns).
pub trait GenerateRandomTrace<const DEGREE_PLUS_ONE: usize>: Uair {
    type PolyCoeff: Clone;
    type Int: Clone;

    #[allow(clippy::type_complexity)]
    fn generate_random_trace<G: Rng + ?Sized>(
        num_vars: usize,
        rng: &mut G,
    ) -> UairTrace<'static, Self::PolyCoeff, Self::Int, DEGREE_PLUS_ONE, DEGREE_PLUS_ONE>;
}
