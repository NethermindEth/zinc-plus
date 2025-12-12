//! Pseudo number theoretic transform of radix 8.

#[macro_use]
mod butterfly;
mod mul_by_twiddle;
mod octet_reversal;
mod params;

use ark_std::{cfg_chunks_mut, cfg_iter_mut};
use num_traits::{CheckedAdd, CheckedMul};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{array, iter::Sum};
use zinc_utils::add;

use butterfly::*;
use octet_reversal::*;

pub(crate) use mul_by_twiddle::*;

pub use params::*;

/// The main entrypoint of the radix-8 pseudo NTT algorithm.
pub(crate) fn pntt<In, Out, C, M>(
    input: &[In],
    zero: Out,
    params: &Radix8PNTTParams<C>,
    mul_by_twiddle: M,
) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: Clone + From<In> + CheckedAdd + CheckedMul + Send + Sync + Sum,
    for<'a> &'a Out: From<&'a In>,
    M: MulByTwiddle<Out, C::Int>,
{
    assert_eq!(
        C::N,
        input.len(),
        "PNTT expects length = {}, got {}",
        C::N,
        input.len()
    );

    let mut output = base_multiply_into_output(input, params, zero, mul_by_twiddle.clone());

    combine_stages(&mut output, params, mul_by_twiddle);

    output
}

#[allow(clippy::arithmetic_side_effects)]
/// Performs the butterfly steps of the radix-8 pseudo NTT algorithm.
/// Assumes `out` contains the result of multiplications of the base chunks
/// with the `base_matrix`.
fn combine_stages<Out, C, M>(out: &mut [Out], params: &Radix8PNTTParams<C>, mul_by_twiddle: M)
where
    C: Config,
    Out: Clone + CheckedAdd + CheckedMul + Send + Sync,
    M: MulByTwiddle<Out, C::Int>,
{
    for k in 0..C::DEPTH {
        // The length of chunks in the current layer.
        let sub_chunk_length = C::BASE_DIM * (1 << (3 * k));

        // Work separately on combining each chunk of the next layer.
        cfg_chunks_mut!(out, 8 * sub_chunk_length).for_each(|chunk: &mut [Out]| {
            for i in 0..sub_chunk_length {
                // Prepare subresults. Multiply them by the right roots of unity.
                let subresults: [Out; 8] = array::from_fn(|j| {
                    mul_by_twiddle.mul_by_twiddle(
                        &chunk[j * sub_chunk_length + i],
                        &params.roots_of_unity[j * i * (1 << (3 * (C::DEPTH - 1 - k)))],
                    )
                });

                // Perform butterflies.
                (
                    chunk[i],
                    chunk[sub_chunk_length + i],
                    chunk[2 * sub_chunk_length + i],
                    chunk[3 * sub_chunk_length + i],
                    chunk[4 * sub_chunk_length + i],
                    chunk[5 * sub_chunk_length + i],
                    chunk[6 * sub_chunk_length + i],
                    chunk[7 * sub_chunk_length + i],
                ) = do_all_butterflies!(&subresults, &(C::TWIDDLES), mul_by_twiddle.clone());
            }
        });
    }
}

#[allow(clippy::arithmetic_side_effects)]
/// Allocates the output vector and performs base layer multiplications.
fn base_multiply_into_output<In, Out, C, M>(
    input: &[In],
    params: &Radix8PNTTParams<C>,
    zero: Out,
    mul_by_twiddle: M,
) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: Clone + From<In> + CheckedAdd + CheckedMul + Sum + Send + Sync + CheckedMul + Sum + Send,
    for<'a> &'a Out: From<&'a In>,
    M: MulByTwiddle<Out, C::Int>,
{
    let mut output = vec![zero; C::M];

    cfg_iter_mut!(output).enumerate().for_each(|(i, out)| {
        let chunk = i / C::BASE_DIM;
        let row = i % C::BASE_DIM;

        // If we'd done all the divide steps of the NTT recursively
        // we'd end up with chunks of original indices
        // combined together according to their `3 * C::DEPTH`
        // least significant bits. Moreover, the value of these
        // least significant bits correspond to the number of the chunk
        // in octet-reverse order.
        let oct_rev_chunk = octet_reversal(chunk, C::DEPTH);

        // We always know that the first column of the Vandermonde matrix
        // consists of 1's.
        let result = params.base_matrix[row][1..].iter().enumerate().fold(
            Out::from(input[oct_rev_chunk].clone()),
            |acc, (col, bm_row_col)| {
                let term = mul_by_twiddle.mul_by_twiddle(
                    (&input[oct_rev_chunk | ((col + 1) << (3 * C::DEPTH))]).into(),
                    bm_row_col,
                );

                add!(acc, &term)
            },
        );

        *out = result;
    });

    output
}

// TODO: make unchecked versions of the above.

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use ark_ff::{Field, PrimeField, Zero};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use crypto_primitives::crypto_bigint_int::Int;
    use itertools::Itertools;
    use num_traits::ConstZero;
    use octet_reversal::octet_reversal;
    use zinc_utils::mul_by_scalar::MulByScalar;

    use super::*;

    fn compare_to_arkworks_ntt_base_layer_generic<C: Config>()
    where
        C::Field: From<i64>,
        for<'a> i64: MulByScalar<&'a C::Int>,
    {
        let input = (0i64..(32i64 * Into::<i64>::into(1 << (3 * C::DEPTH)))).collect_vec();

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
            let params = Radix8PNTTParams::<C>::new();

            let output = base_multiply_into_output(&input, &params, 0, MBSMulByTwiddle);

            output.into_iter().map(C::Field::from).collect_vec()
        };

        assert_eq!(arkworks_res, our_res);
    }

    #[test]
    fn compare_to_arkworks_ntt_base_layer_multiply() {
        compare_to_arkworks_ntt_base_layer_generic::<PNTTConfigF2_16_1<1>>();
        compare_to_arkworks_ntt_base_layer_generic::<PNTTConfigF2_16_1<2>>();
        compare_to_arkworks_ntt_base_layer_generic::<PNTTConfigF2_16_1<3>>();
    }

    fn pntt_against_arkworks_generic<C: Config>()
    where
        C::Field: From<C::Int>,
        C::Int: From<i64> + Into<i64>,
        Int<4>: From<C::Int> + for<'a> MulByScalar<&'a C::Int>,
    {
        let input: Vec<C::Int> = (0..C::N)
            .map(|x| i64::try_from(x).unwrap().into())
            .collect_vec();

        let arkworks_res = {
            let domain = Radix2EvaluationDomain::<C::Field>::new(C::M).unwrap();

            let mut input = input
                .iter()
                .map(|i| C::Field::from(i.clone()))
                .collect_vec();

            input.resize(C::M, C::Field::zero());

            domain.fft_in_place(&mut input);

            input
        };

        let our_res = {
            let input: Vec<Int<4>> = input.into_iter().map(|x| x.into().into()).collect_vec();

            let params = Radix8PNTTParams::<C>::new();

            let res: Vec<Int<4>> = pntt(&input, Int::<4>::ZERO, &params, MBSMulByTwiddle);

            res.into_iter()
                .map(|x| {
                    let x_reduced = {
                        let modulus = <C::Field as Field>::BasePrimeField::MODULUS.as_ref()[0];
                        let x_reduced = x % Int::from_i64(modulus.try_into().unwrap());

                        let x_reduced: Int<1> = x_reduced.checked_resize().unwrap();

                        i64::from(x_reduced.into_inner())
                    };

                    C::Field::from(x_reduced.into())
                })
                .collect_vec()
        };

        assert_eq!(arkworks_res, our_res);
    }

    #[test]
    fn pntt_against_arkworks() {
        pntt_against_arkworks_generic::<PNTTConfigF2_16_1<1>>();
        pntt_against_arkworks_generic::<PNTTConfigF2_16_1<2>>();
        pntt_against_arkworks_generic::<PNTTConfigF2_16_1<3>>();
    }
}
