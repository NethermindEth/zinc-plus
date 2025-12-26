use super::params::{Config, PnttInt};
use ark_ff::{Field, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use itertools::Itertools;
use std::array;

#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
pub(crate) fn precompute_butterfly_twiddles<C: Config>() -> Vec<Vec<[[PnttInt; 8]; 7]>> {
    let roots_of_unity = precompute_roots_of_unity::<C>(C::OUTPUT_LEN);

    let modulus = <C::Field as Field>::BasePrimeField::MODULUS.as_ref()[0];
    let modulus_i64 = modulus as i64;

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
                                modulus_i64,
                                modulus,
                            )
                        })
                    })
                })
                .collect()
        })
        .collect()
}

pub(crate) fn precompute_base_matrix<C: Config>() -> Vec<Vec<PnttInt>> {
    let domain =
        Radix2EvaluationDomain::<C::Field>::new(C::BASE_DIM).expect("Failed to create NTT domain");

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
pub(crate) fn precompute_roots_of_unity<C: Config>(n: usize) -> Vec<PnttInt> {
    let domain = Radix2EvaluationDomain::<C::Field>::new(n).expect("Failed to create NTT domain");

    domain
        .elements()
        .map(C::field_to_int_normalized)
        .collect_vec()
}

/// Field normalization for at most 32-bit fields.
/// Might have unpleasant overflows if used for bigger fields.
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
pub(crate) fn normalize_field_element(x: u64, p: u64) -> i64 {
    if x >= (p - 1) / 2 {
        x as i64 - p as i64
    } else {
        x as i64
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn mul_and_normalize_twiddle(
    twiddle: PnttInt,
    root: PnttInt,
    modulus_i64: i64,
    modulus_u64: u64,
) -> PnttInt {
    let twiddle_mod = to_positive_mod_repr(twiddle, modulus_i64);
    let root_mod = to_positive_mod_repr(root, modulus_i64);
    let product = (twiddle_mod * root_mod) % modulus_u64;

    normalize_field_element(product, modulus_u64)
}

#[allow(clippy::arithmetic_side_effects, clippy::cast_sign_loss)]
fn to_positive_mod_repr(value: PnttInt, modulus: i64) -> u64 {
    let mut repr = value % modulus;
    if repr < 0 {
        repr += modulus;
    }
    repr as u64
}
