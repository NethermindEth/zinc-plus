//! Pseudo number theoretic transform of radix 8.

#[macro_use]
mod butterfly;
mod octet_reversal;

pub mod params;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{array, fmt::Debug};
use zinc_utils::{cfg_chunks_mut, cfg_into_iter};

use butterfly::*;
use octet_reversal::*;
use params::*;

/// The main entrypoint of the radix-8 pseudo NTT algorithm.
///
/// All arithmetic on the output type is provided by the caller.
pub(crate) fn pntt<In, Out, C>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    map_to_out: impl Fn(&In) -> Out + Copy + Sync,
    mul_in_by_twiddle: impl Fn(&In, &PnttInt) -> Out + Copy + Sync,
    mul_out_by_twiddle: impl Fn(&Out, &PnttInt) -> Out + Copy + Sync,
    add_out: impl Fn(Out, &Out) -> Out + Copy + Sync,
) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: Clone + Send + Sync + Debug,
{
    assert_eq!(
        params.row_len,
        input.len(),
        "PNTT expects length = {}, got {}",
        params.row_len,
        input.len()
    );

    let mut output =
        base_multiply_into_output(input, params, map_to_out, mul_in_by_twiddle, add_out);

    combine_stages(&mut output, params, mul_out_by_twiddle, add_out);

    output
}

/// Performs the butterfly steps of the radix-8 pseudo NTT algorithm.
/// Assumes `out` contains the result of multiplications of the base chunks
/// with the `base_matrix`.
#[allow(clippy::arithmetic_side_effects)]
fn combine_stages<R, C>(
    out: &mut [R],
    params: &Radix8PnttParams<C>,
    mul_by_twiddle: impl Fn(&R, &PnttInt) -> R + Copy + Sync,
    add_out: impl Fn(R, &R) -> R + Copy + Sync,
) where
    C: Config,
    R: Clone + Send + Sync + Debug,
{
    for k in 0..params.depth {
        // The length of chunks in the current layer.
        let sub_chunk_length = params.base_dim * (1 << (3 * k));

        // On each step of recursive radix-8 NTT
        // we divide the length of the evaluation domain by 8.
        // This is done via raising the current primitive root \omega
        // to 8. Hence on the recursive step k the current primitive
        // root of unity can be found as the original \omega
        // raised to 8^k.
        //
        // Since we are going from bottom up we are successively
        // taking
        //      \omega^(8 ^ (params.depth - 1))
        //      \omega^(8 ^ (params.depth - 2))
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
                apply_radix_8_butterflies(
                    ys,
                    &subresults,
                    &layer_twiddles[i],
                    mul_by_twiddle,
                    add_out,
                );
            }
        });
    }
}

/// Allocates the output vector and performs base layer multiplications.
#[allow(clippy::arithmetic_side_effects)]
fn base_multiply_into_output<In, Out, C>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    map_to_out: impl Fn(&In) -> Out + Copy + Sync,
    mul_by_twiddle: impl Fn(&In, &PnttInt) -> Out + Copy + Sync,
    add_out: impl Fn(Out, &Out) -> Out + Copy + Sync,
) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: Clone + Send + Sync,
{
    cfg_into_iter!(0..params.codeword_len)
        .map(|i| {
            let chunk = i >> params.base_dim_log2; // i / BASE_DIM
            let row = i & params.base_dim_mask; // i % BASE_DIM

            // If we'd done all the divide steps of the NTT recursively
            // we'd end up with chunks of original indices
            // combined together according to their `3 * params.depth`
            // least significant bits. Moreover, the value of these
            // least significant bits correspond to the number of the chunk
            // in octet-reverse order.
            let oct_rev_chunk = octet_reversal(chunk, params.depth);

            // We always know that the first column of the Vandermonde matrix
            // consists of 1's.
            params.base_matrix[row][1..].iter().enumerate().fold(
                map_to_out(&input[oct_rev_chunk]),
                |acc, (col, bm_row_col)| {
                    let term = mul_by_twiddle(
                        &input[oct_rev_chunk | ((col + 1) << (3 * params.depth))],
                        bm_row_col,
                    );

                    add_out(acc, &term)
                },
            )
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, clippy::clone_on_copy)]
mod tests {
    use ark_ff::{Field, PrimeField, Zero};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use crypto_primitives::{Wrapper, crypto_bigint_int::Int};
    use itertools::Itertools;
    use num_traits::CheckedAdd;
    use octet_reversal::octet_reversal;
    use zinc_utils::{CHECKED, mul_by_scalar::MulByScalar};

    use super::*;

    fn compare_to_arkworks_ntt_base_layer_generic<C: Config>(params: &Radix8PnttParams<C>)
    where
        C::Field: From<PnttInt>,
    {
        let input = (0i64..(32i64 * PnttInt::from(1 << (3 * params.depth)))).collect_vec();

        let arkworks_res = {
            let mut result = Vec::with_capacity(64 * (1 << (3 * params.depth)));
            let domain = Radix2EvaluationDomain::<C::Field>::new(64).unwrap();

            for chunk in 0..(1 << (3 * params.depth)) {
                let mut input = input
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        i & ((1 << (3 * params.depth)) - 1) == octet_reversal(chunk, params.depth)
                    })
                    .map(|(_, x)| C::Field::from(*x))
                    .collect_vec();

                input.resize(64, C::Field::zero());

                result.extend(domain.fft(&input));
            }

            result
        };

        let our_res = {
            let output = base_multiply_into_output(
                &input,
                params,
                |v| *v,
                |a, b| (*a).mul_by_scalar::<CHECKED>(b).unwrap(),
                |a: PnttInt, b: &PnttInt| a.checked_add(*b).unwrap(),
            );

            output.into_iter().map(C::Field::from).collect_vec()
        };

        assert_eq!(arkworks_res, our_res);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // long running
    fn compare_to_arkworks_ntt_base_layer_multiply() {
        for depth in 1..=3 {
            let row_len = 32 * (1 << (3 * depth));
            let params = Radix8PnttParams::new(row_len, depth, 2).unwrap();
            compare_to_arkworks_ntt_base_layer_generic::<PnttConfigF65537>(&params);
        }
    }

    fn pntt_against_arkworks_generic<C: Config>(params: &Radix8PnttParams<C>)
    where
        C::Field: From<PnttInt>,
        Int<4>: From<PnttInt>,
    {
        let input: Vec<PnttInt> = (0..params.row_len)
            .map(|x| PnttInt::try_from(x).unwrap())
            .collect_vec();

        let arkworks_res = {
            let domain = Radix2EvaluationDomain::<C::Field>::new(params.codeword_len).unwrap();

            let mut input = input.iter().map(|i| C::Field::from(*i)).collect_vec();

            input.resize(params.codeword_len, C::Field::zero());

            domain.fft_in_place(&mut input);

            input
        };

        let our_res = {
            let input: Vec<Int<4>> = input.into_iter().map(|x| x.into()).collect_vec();

            let res: Vec<Int<4>> = pntt(
                &input,
                params,
                |v| v.clone(),
                |a: &Int<4>, b| (*a).mul_by_scalar::<CHECKED>(b).unwrap(),
                |a: &Int<4>, b| (*a).mul_by_scalar::<CHECKED>(b).unwrap(),
                |a: Int<4>, b: &Int<4>| a.checked_add(b).unwrap(),
            );

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
    #[cfg_attr(miri, ignore)] // long running
    fn pntt_against_arkworks() {
        let base_len = 32;
        for depth in 1..=3 {
            let row_len = base_len * (1 << (3 * depth));
            pntt_against_arkworks_generic::<PnttConfigF65537>(
                &Radix8PnttParams::new(row_len, depth, 2).unwrap(),
            );
        }
    }
}
