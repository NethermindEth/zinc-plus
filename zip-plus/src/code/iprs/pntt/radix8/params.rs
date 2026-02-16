use ark_ff::{FftField, FpConfig};
use std::marker::PhantomData;

/// The integer types of twiddles.
pub type PnttInt = i64;

/// Configuration of radix-8 pseudo NTT.
pub trait Config: Copy + Send + Sync {
    /// The field used to generate the twiddle factors
    /// and the base matrix for this pseudo NTT.
    type Field: FftField;
    const FIELD_MODULUS: u32;

    /// The number of steps where NTT is performed
    /// recursivily.
    const DEPTH: usize;
    /// The number of columns of the base matrix.
    const BASE_LEN: usize;
    /// The number of rows of the base matrix, always a power of 2.
    const BASE_DIM: usize;
    /// log2 of the number of rows of the base matrix.
    const BASE_DIM_LOG2: u32 = Self::BASE_DIM.trailing_zeros();
    /// The mask to compute `i % Self::BASE_DIM`.
    const BASE_DIM_MASK: usize = Self::BASE_DIM - 1;
    /// The coefficients used to combine subresults.
    /// They are the 8-th roots of unity from the field `Self::Field`
    /// lifted to `Self::Int`.
    const BASE_TWIDDLES: [PnttInt; 8];

    /// The length of the pseudo NTT's input.
    const INPUT_LEN: usize = Self::BASE_LEN * (1 << (3 * Self::DEPTH));
    /// The length of the pseudo NTT's output.
    const OUTPUT_LEN: usize = Self::BASE_DIM * (1 << (3 * Self::DEPTH));

    /// A helper to get an integer representation that
    /// lies in the range `[-(p - 1)/2, (p - 1)/2]` from a field element.
    // TODO(alex): If we need it somewhere else, it would make sense to make this a
    // property of the field.
    fn field_to_int_normalized(x: Self::Field) -> PnttInt;

    /// Validate that the depth fits the NTT domain for this configuration.
    fn assert_depth_valid() {}
}

mod precompute {
    use super::{Config, PnttInt};
    use ark_ff::Field;
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use itertools::Itertools;
    use std::array;
    use zinc_utils::mul;

    #[allow(clippy::arithmetic_side_effects)]
    pub(super) fn precompute_butterfly_twiddles<C: Config>() -> Vec<Vec<[[PnttInt; 8]; 7]>> {
        let roots_of_unity = precompute_roots_of_unity::<C>(C::OUTPUT_LEN);

        (0..C::DEPTH)
            .map(|k| {
                let sub_chunk_length = C::BASE_DIM * (1 << (3 * k));
                let curr_prim_root_power = 1 << (3 * (C::DEPTH - 1 - k));

                (0..sub_chunk_length)
                    .map(|i| {
                        array::from_fn(|j_minus_1| {
                            let root = roots_of_unity[curr_prim_root_power * i * (j_minus_1 + 1)];

                            array::from_fn(|twiddle_idx| {
                                mul_and_normalize_twiddle(
                                    C::BASE_TWIDDLES[twiddle_idx],
                                    root,
                                    C::FIELD_MODULUS,
                                )
                            })
                        })
                    })
                    .collect()
            })
            .collect()
    }

    pub(super) fn precompute_base_matrix<C: Config>() -> Vec<Vec<PnttInt>> {
        let domain = Radix2EvaluationDomain::<C::Field>::new(C::BASE_DIM)
            .expect("Failed to create NTT domain");

        domain
            .elements()
            .map(|root| {
                (0..C::BASE_LEN)
                    .map(move |i| C::field_to_int_normalized(root.pow([i as u64])))
                    .collect_vec()
            })
            .collect()
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub(super) fn precompute_roots_of_unity<C: Config>(n: usize) -> Vec<PnttInt> {
        let domain =
            Radix2EvaluationDomain::<C::Field>::new(n).expect("Failed to create NTT domain");

        domain
            .elements()
            .map(C::field_to_int_normalized)
            .collect_vec()
    }

    /// Field normalization for at most 32-bit fields.
    /// Might have unpleasant overflows if used for bigger fields.
    #[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
    pub(super) fn normalize_field_element(x: u64, p: u32) -> i64 {
        debug_assert!(x <= i64::MAX as u64);
        let x = x as i64;
        let p = i64::from(p);
        if x >= (p - 1) / 2 { x - p } else { x }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn mul_and_normalize_twiddle(twiddle: PnttInt, root: PnttInt, modulus: u32) -> PnttInt {
        let twiddle_mod = to_positive_mod_repr(twiddle, modulus);
        let root_mod = to_positive_mod_repr(root, modulus);
        let product = mul!(twiddle_mod, root_mod) % u64::from(modulus);

        normalize_field_element(product, modulus)
    }

    #[allow(clippy::cast_sign_loss)]
    fn to_positive_mod_repr(value: PnttInt, modulus: u32) -> u64 {
        value.rem_euclid(i64::from(modulus)) as u64
    }
}

// Precomputed parameters needed for
// pseudo NTT algorithm.
#[derive(Clone, Debug)]
pub struct Radix8PnttParams<C: Config> {
    /// The base matrix of the pseudo NTT.
    pub base_matrix: Vec<Vec<PnttInt>>, // TODO(Alex): Maybe use DenseRowMatrix for this?
    /// Precomputed twiddles for every stage that already contain the relevant
    /// root-of-unity factor. This lets the butterfly apply a single
    /// multiplication per term instead of two.
    pub butterfly_twiddles: Vec<Vec<[[PnttInt; 8]; 7]>>,
    _phantom: PhantomData<C>,
}

impl<C: Config> Default for Radix8PnttParams<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Config> Radix8PnttParams<C> {
    /// Precompute pseudo NTT parameters.
    pub fn new() -> Self {
        C::assert_depth_valid();
        Self {
            base_matrix: precompute::precompute_base_matrix::<C>(),
            butterfly_twiddles: precompute::precompute_butterfly_twiddles::<C>(),
            _phantom: PhantomData,
        }
    }
}

/// Pseudo NTT configuration derived from
/// the field `Fp` for `p = 2^16 + 1`.
///
/// Supports `DEPTH` up to `3`.
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16_1<const DEPTH: usize>;

mod fq {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "65537"]
    #[generator = "3"]
    pub struct FqConfig;

    pub type FqBackend = MontBackend<FqConfig, 1>;
    pub type Fq = Fp64<FqBackend>;

    #[allow(clippy::cast_possible_truncation)] // We know modulus is small enough.
    pub const MODULUS: u32 = FqConfig::MODULUS.0[0] as u32;
}

impl<const DEPTH: usize> Config for PnttConfigF2_16_1<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 32;
    const BASE_DIM: usize = 64;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }

    fn assert_depth_valid() {
        assert!(
            DEPTH <= 3,
            "DEPTH {DEPTH} exceeds max 3 for F65537 PNTT (BASE_DIM=64)"
        );
    }
}

// ==================== F65537 rate 1/4 configurations ====================

/// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=1, BASE_DIM=4 (rate 1/4).
/// NTT domain up to 2^16. Row lengths: 8 (D=1), 64 (D=2), 512 (D=3), 4096 (D=4).
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16R4B1<const DEPTH: usize>;

/// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=2, BASE_DIM=8 (rate 1/4).
/// NTT domain up to 2^16. Row lengths: 16 (D=1), 128 (D=2), 1024 (D=3), 8192 (D=4).
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16R4B2<const DEPTH: usize>;

/// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=4, BASE_DIM=16 (rate 1/4).
/// NTT domain up to 2^16. Row lengths: 32 (D=1), 256 (D=2), 2048 (D=3), 16384 (D=4).
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16R4B4<const DEPTH: usize>;

/// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=16, BASE_DIM=64 (rate 1/4).
/// NTT domain up to 2^16, enabling row lengths up to 2^13.
/// Row lengths: 128 (D=1), 1024 (D=2), 8192 (D=3).
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16R4B16<const DEPTH: usize>;

/// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=32, BASE_DIM=128 (rate 1/4).
/// NTT domain up to 2^16, enabling row lengths up to 2^14.
/// Row lengths: 256 (D=1), 2048 (D=2), 16384 (D=3).
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16R4B32<const DEPTH: usize>;

/// Pseudo NTT configuration for F65537 (2^16 + 1) with BASE_LEN=64, BASE_DIM=256 (rate 1/4).
/// NTT domain up to 2^16, enabling row lengths up to 2^12.
/// Row lengths: 512 (D=1), 4096 (D=2).
#[derive(Clone, Copy)]
pub struct PnttConfigF2_16R4B64<const DEPTH: usize>;

impl<const DEPTH: usize> Config for PnttConfigF2_16R4B1<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 1;
    const BASE_DIM: usize = 4;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }

    fn assert_depth_valid() {
        assert!(
            DEPTH <= 4,
            "DEPTH {DEPTH} exceeds max 4 for F65537 PNTT (BASE_DIM=4)"
        );
    }
}

impl<const DEPTH: usize> Config for PnttConfigF2_16R4B2<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 2;
    const BASE_DIM: usize = 8;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }

    fn assert_depth_valid() {
        assert!(
            DEPTH <= 4,
            "DEPTH {DEPTH} exceeds max 4 for F65537 PNTT (BASE_DIM=8)"
        );
    }
}

impl<const DEPTH: usize> Config for PnttConfigF2_16R4B4<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 4;
    const BASE_DIM: usize = 16;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }

    fn assert_depth_valid() {
        assert!(
            DEPTH <= 4,
            "DEPTH {DEPTH} exceeds max 4 for F65537 PNTT (BASE_DIM=16)"
        );
    }
}

impl<const DEPTH: usize> Config for PnttConfigF2_16R4B16<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 16;
    const BASE_DIM: usize = 64;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }

    fn assert_depth_valid() {
        assert!(
            DEPTH <= 3,
            "DEPTH {DEPTH} exceeds max 3 for F65537 PNTT (BASE_DIM=64)"
        );
    }
}

impl<const DEPTH: usize> Config for PnttConfigF2_16R4B32<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 32;
    const BASE_DIM: usize = 128;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }

    fn assert_depth_valid() {
        assert!(
            DEPTH <= 3,
            "DEPTH {DEPTH} exceeds max 3 for F65537 PNTT (BASE_DIM=128)"
        );
    }
}

impl<const DEPTH: usize> Config for PnttConfigF2_16R4B64<DEPTH> {
    type Field = fq::Fq;
    const FIELD_MODULUS: u32 = fq::MODULUS;
    const BASE_LEN: usize = 64;
    const BASE_DIM: usize = 256;
    const DEPTH: usize = DEPTH;
    const BASE_TWIDDLES: [PnttInt; 8] = [1, 4096, -256, 16, -1, -4096, 256, -16];

    fn field_to_int_normalized(x: Self::Field) -> PnttInt {
        let big_int = fq::FqBackend::into_bigint(x);

        precompute::normalize_field_element(big_int.0[0], Self::FIELD_MODULUS)
    }

    fn assert_depth_valid() {
        assert!(
            DEPTH <= 2,
            "DEPTH {DEPTH} exceeds max 2 for F65537 PNTT (BASE_DIM=256)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Twiddles are indeed the 8th roots of unity.
    fn check_twiddles_generic<C: Config>() {
        let expected = precompute::precompute_roots_of_unity::<C>(8);

        let our = C::BASE_TWIDDLES.to_vec();

        assert_eq!(expected, our);
    }

    #[test]
    fn check_twiddles_f65537() {
        check_twiddles_generic::<PnttConfigF2_16_1<1>>();
        check_twiddles_generic::<PnttConfigF2_16R4B1<1>>();
        check_twiddles_generic::<PnttConfigF2_16R4B2<1>>();
        check_twiddles_generic::<PnttConfigF2_16R4B4<1>>();
        check_twiddles_generic::<PnttConfigF2_16R4B16<1>>();
        check_twiddles_generic::<PnttConfigF2_16R4B32<1>>();
        check_twiddles_generic::<PnttConfigF2_16R4B64<1>>();
    }
}
