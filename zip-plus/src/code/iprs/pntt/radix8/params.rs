use ark_ff::{FftField, Field, FpConfig, MontBackend};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use itertools::Itertools;

/// Configuration of radix-8 pseudo NTT.
pub trait Config: Copy + Send + Sync {
    /// The integer types of twiddles.
    type Int: Clone + std::fmt::Debug + Eq + Send + Sync;
    /// The field used to generate the twiddle factors
    /// and the base matrix for this pseudo NTT.
    type Field: FftField;

    /// The number of steps where NTT is performed
    /// recursivily.
    const DEPTH: usize;
    /// The number of columns of the base matrix.
    const BASE_LEN: usize;
    /// The number of rows of the base matrix.
    const BASE_DIM: usize;
    /// The coefficients used to combine subresults.
    /// They are the 8-th roots of unity from the field `Self::Field`
    /// lifted to `Self::Int`.
    const TWIDDLES: [Self::Int; 8];

    /// The length of the pseudo NTT's input.
    const INPUT_LEN: usize = Self::BASE_LEN * (1 << (3 * Self::DEPTH));
    /// The length of the pseudo NTT's output.
    const OUTPUT_LEN: usize = Self::BASE_DIM * (1 << (3 * Self::DEPTH));

    /// A helper to get an integer representation that
    /// lies in the range `[-(p - 1)/2, (p - 1)/2]` from a field element.
    fn field_to_int_normalized(x: Self::Field) -> Self::Int;
}

// Precomputed parameters needed for
// pseudo NTT algorithm.
#[derive(Clone, Debug)]
pub struct Radix8PNTTParams<C: Config> {
    /// The base matrix of the pseudo NTT.
    pub base_matrix: Vec<Vec<C::Int>>,
    /// The roots of unity of degree `C::M` lifted to integers.
    pub roots_of_unity: Vec<C::Int>,
}

impl<C: Config> Radix8PNTTParams<C> {
    /// Precompute pseudo NTT parameters.
    pub fn new() -> Self {
        Self {
            base_matrix: Self::precompute_base_matrix(),
            // TODO(Ilia): There's no reason to not use the roots of unity
            //             in precomputation of `base_matrix`.
            roots_of_unity: Self::precompute_roots_of_unity(),
        }
    }

    fn precompute_base_matrix() -> Vec<Vec<C::Int>> {
        let domain = Radix2EvaluationDomain::<C::Field>::new(C::BASE_DIM)
            .expect("Failed to create NTT domain");

        let mut matrix = Vec::with_capacity(C::BASE_DIM);

        for root in domain.elements() {
            matrix.push(
                (0..C::BASE_LEN)
                    .map(move |i| C::field_to_int_normalized(root.pow([i as u64])))
                    .collect_vec(),
            )
        }

        matrix
    }

    fn precompute_roots_of_unity() -> Vec<C::Int> {
        let domain = Radix2EvaluationDomain::<C::Field>::new(C::OUTPUT_LEN)
            .expect("Failed to create NTT domain");

        domain
            .elements()
            .map(C::field_to_int_normalized)
            .collect_vec()
    }
}

/// Pseudo NTT configuration derived from
/// the field `Fp` for `p = 2^16 + 1`.
///
/// Supports `DEPTH` up to `3`.
#[derive(Clone, Copy)]
pub struct PNTTConfigF2_16_1<const DEPTH: usize>;

pub(crate) mod fq {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "65537"]
    #[generator = "3"]
    pub struct FqConfig;
    pub type Fq = Fp64<MontBackend<FqConfig, 1>>;
}

use fq::*;

impl<const DEPTH: usize> Config for PNTTConfigF2_16_1<DEPTH> {
    type Int = i64;
    type Field = Fq;
    const BASE_LEN: usize = 32;
    const BASE_DIM: usize = 64;
    const DEPTH: usize = DEPTH;
    const TWIDDLES: [Self::Int; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> Self::Int {
        let big_int = MontBackend::<FqConfig, 1>::into_bigint(x);

        normalize_field_element(big_int.0[0], (1 << 16) + 1)
    }
}

/// Field normalization for at most 32-bit fields.
/// Might have unpleasant overflows if used for bigger fields.
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::cast_possible_wrap)]
fn normalize_field_element(x: u64, p: u64) -> i64 {
    if x >= (p - 1) / 2 {
        x as i64 - p as i64
    } else {
        x as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Twiddles are indeed the 8th roots of unity.
    fn check_twiddles_generic<C: Config>() {
        let domain = Radix2EvaluationDomain::<C::Field>::new(8).unwrap();

        let expected = domain
            .elements()
            .map(C::field_to_int_normalized)
            .collect_vec();

        let our = C::TWIDDLES.to_vec();

        assert_eq!(expected, our);
    }

    #[test]
    fn check_twiddles() {
        check_twiddles_generic::<PNTTConfigF2_16_1<1>>();
    }
}
