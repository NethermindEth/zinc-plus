//! Pseudo number theoretic transform of radix 8.

#[macro_use]
mod butterfly;
mod mul_by_twiddle;
mod octet_reversal;

pub mod params;

use ark_std::{cfg_chunks_mut, cfg_into_iter};
use itertools::Itertools;
use num_traits::{CheckedAdd, CheckedMul};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{array, fmt::Debug, iter::Sum};
use zinc_utils::add;

use butterfly::*;
use octet_reversal::*;
use params::*;

pub(crate) use mul_by_twiddle::*;

/// The main entrypoint of the radix-8 pseudo NTT algorithm.
pub(crate) fn pntt<R, C, M>(input: &[R], params: &Radix8PnttParams<C>, mul_by_twiddle: &M) -> Vec<R>
where
    C: Config,
    R: CheckedAdd + CheckedMul + Sum + Clone + Send + Sync + Debug,
    M: MulByTwiddle<R, PnttInt>,
{
    assert_eq!(
        C::INPUT_LEN,
        input.len(),
        "PNTT expects length = {}, got {}",
        C::INPUT_LEN,
        input.len()
    );

    let mut output = base_multiply_into_output(input, params, mul_by_twiddle);

    combine_stages(&mut output, params, mul_by_twiddle);

    output
}

/// Performs the butterfly steps of the radix-8 pseudo NTT algorithm.
/// Assumes `out` contains the result of multiplications of the base chunks
/// with the `base_matrix`.
#[allow(clippy::arithmetic_side_effects)]
fn combine_stages<R, C, M>(out: &mut [R], params: &Radix8PnttParams<C>, mul_by_twiddle: &M)
where
    C: Config,
    R: CheckedAdd + CheckedMul + Clone + Send + Sync + Debug,
    M: MulByTwiddle<R, PnttInt>,
{
    for k in 0..C::DEPTH {
        // The length of chunks in the current layer.
        let sub_chunk_length = C::BASE_DIM * (1 << (3 * k));

        // On each step of recursive radix-8 NTT
        // we divide the length of the evaluation domain by 8.
        // This is done via raising the current primitive root \omega
        // to 8. Hence on the recursive step k the current primitive
        // root of unity can be found as the original \omega
        // raised to 8^k.
        //
        // Since we are going from bottom up we are successively
        // taking
        //      \omega^(8 ^ (C::DEPTH - 1))
        //      \omega^(8 ^ (C::DEPTH - 2))
        //      ...
        //      \omega^(8 ^ 0) = \omega
        // These factors are already absorbed into `layer_twiddles`.
        let layer_twiddles = &params.butterfly_twiddles[k];
        debug_assert_eq!(layer_twiddles.len(), sub_chunk_length);

        // Work separately on combining each chunk of the next layer.
        cfg_chunks_mut!(out, 8 * sub_chunk_length).for_each(|chunk: &mut [R]| {
            for i in 0..sub_chunk_length {
                // Prepare subresults without applying roots of unity; the
                // per-layer twiddles already include those factors.
                let subresults: [R; 8] =
                    array::from_fn(|j| chunk[j * sub_chunk_length + i].clone());

                #[allow(unused_mut)] // false alarm
                let ys: [&mut R; 8] = chunk
                    .chunks_mut(sub_chunk_length)
                    .map(|mut subchunk| &mut subchunk[i])
                    .collect_vec()
                    .try_into()
                    .expect("We are guaranteed to have the right length here");

                // Perform butterflies.
                apply_radix_8_butterflies(ys, &subresults, &layer_twiddles[i], mul_by_twiddle);
            }
        });
    }
}

/// Allocates the output vector and performs base layer multiplications.
#[allow(clippy::arithmetic_side_effects)]
fn base_multiply_into_output<R, C, M>(
    input: &[R],
    params: &Radix8PnttParams<C>,
    mul_by_twiddle: &M,
) -> Vec<R>
where
    C: Config,
    R: Clone + CheckedAdd + CheckedMul + Sum + Send + Sync,
    M: MulByTwiddle<R, PnttInt>,
{
    cfg_into_iter!(0..C::OUTPUT_LEN)
        .map(|i| {
            let chunk = i >> C::BASE_DIM_LOG2; // i / C::BASE_DIM
            let row = i & C::BASE_DIM_MASK; // i % C::BASE_DIM

            // If we'd done all the divide steps of the NTT recursively
            // we'd end up with chunks of original indices
            // combined together according to their `3 * C::DEPTH`
            // least significant bits. Moreover, the value of these
            // least significant bits correspond to the number of the chunk
            // in octet-reverse order.
            let oct_rev_chunk = octet_reversal(chunk, C::DEPTH);

            // We always know that the first column of the Vandermonde matrix
            // consists of 1's.
            params.base_matrix[row][1..].iter().enumerate().fold(
                input[oct_rev_chunk].clone(),
                |acc, (col, bm_row_col)| {
                    let term = mul_by_twiddle.mul_by_twiddle(
                        &input[oct_rev_chunk | ((col + 1) << (3 * C::DEPTH))],
                        bm_row_col,
                    );

                    add!(acc, &term)
                },
            )
        })
        .collect()
}

// TODO: make unchecked versions of the above.

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use ark_ff::{Field, PrimeField, Zero};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use crypto_primitives::crypto_bigint_int::Int;
    use itertools::Itertools;
    use octet_reversal::octet_reversal;
    use zinc_utils::mul_by_scalar::MulByScalar;

    use super::*;

    fn compare_to_arkworks_ntt_base_layer_generic<C: Config>()
    where
        C::Field: From<PnttInt>,
    {
        let input = (0i64..(32i64 * PnttInt::from(1 << (3 * C::DEPTH)))).collect_vec();

        let arkworks_res = {
            let mut result = Vec::with_capacity(64 * (1 << (3 * C::DEPTH)));
            let domain = Radix2EvaluationDomain::<C::Field>::new(64).unwrap();

            for chunk in 0..(1 << (3 * C::DEPTH)) {
                let mut input = input
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        i & ((1 << (3 * C::DEPTH)) - 1) == octet_reversal(chunk, C::DEPTH)
                    })
                    .map(|(_, x)| C::Field::from(*x))
                    .collect_vec();

                input.resize(64, C::Field::zero());

                result.extend(domain.fft(&input));
            }

            result
        };

        let our_res = {
            let params = Radix8PnttParams::<C>::new();

            let output = base_multiply_into_output(&input, &params, &MBSMulByTwiddle);

            output.into_iter().map(C::Field::from).collect_vec()
        };

        assert_eq!(arkworks_res, our_res);
    }

    #[test]
    fn compare_to_arkworks_ntt_base_layer_multiply() {
        compare_to_arkworks_ntt_base_layer_generic::<PnttConfigF2_16_1<1>>();
        compare_to_arkworks_ntt_base_layer_generic::<PnttConfigF2_16_1<2>>();
        compare_to_arkworks_ntt_base_layer_generic::<PnttConfigF2_16_1<3>>();
    }

    fn pntt_against_arkworks_generic<C: Config>()
    where
        C::Field: From<PnttInt>,
        Int<4>: From<PnttInt> + for<'a> MulByScalar<&'a PnttInt>,
    {
        let input: Vec<PnttInt> = (0..C::INPUT_LEN)
            .map(|x| PnttInt::try_from(x).unwrap())
            .collect_vec();

        let arkworks_res = {
            let domain = Radix2EvaluationDomain::<C::Field>::new(C::OUTPUT_LEN).unwrap();

            let mut input = input.iter().map(|i| C::Field::from(*i)).collect_vec();

            input.resize(C::OUTPUT_LEN, C::Field::zero());

            domain.fft_in_place(&mut input);

            input
        };

        let our_res = {
            let input: Vec<Int<4>> = input.into_iter().map(|x| x.into()).collect_vec();

            let params = Radix8PnttParams::<C>::new();

            let res: Vec<Int<4>> = pntt(&input, &params, &MBSMulByTwiddle);

            res.into_iter()
                .map(|x| {
                    let x_reduced = {
                        let modulus = <C::Field as Field>::BasePrimeField::MODULUS.as_ref()[0];
                        let x_reduced = x % Int::from_i64(modulus.try_into().unwrap());

                        let x_reduced: Int<1> = x_reduced.checked_resize().unwrap();

                        i64::from(x_reduced.into_inner())
                    };

                    C::Field::from(x_reduced)
                })
                .collect_vec()
        };

        assert_eq!(arkworks_res, our_res);
    }

    #[test]
    fn pntt_against_arkworks() {
        pntt_against_arkworks_generic::<PnttConfigF2_16_1<1>>();
        pntt_against_arkworks_generic::<PnttConfigF2_16_1<2>>();
        pntt_against_arkworks_generic::<PnttConfigF2_16_1<3>>();
    }
}
