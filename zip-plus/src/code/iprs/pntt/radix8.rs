//! Pseudo number theoretic transform of radix 8.

mod butterfly;
mod mul_by_twiddle;
mod octet_reversal;

pub mod params;

#[cfg(not(feature = "unchecked-butterfly"))]
use butterfly::apply_radix_8_butterflies;
#[cfg(feature = "unchecked-butterfly")]
use butterfly::apply_radix_8_butterflies_unchecked;
use num_traits::{CheckedAdd, CheckedMul};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{array, fmt::Debug, mem::MaybeUninit};
#[cfg(feature = "unchecked-butterfly")]
use std::ops::AddAssign;
#[cfg(not(feature = "unchecked-butterfly"))]
use zinc_utils::add;
use zinc_utils::{cfg_chunks_mut, cfg_into_iter, from_ref::FromRef};

#[cfg(feature = "pntt-timing")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "pntt-timing")]
static BASE_LAYER_NANOS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "pntt-timing")]
static BUTTERFLY_NANOS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "pntt-timing")]
static PNTT_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
/// Reset timing counters. Call before starting a benchmark.
#[cfg(feature = "pntt-timing")]
pub fn reset_timing() {
    BASE_LAYER_NANOS.store(0, Ordering::Relaxed);
    BUTTERFLY_NANOS.store(0, Ordering::Relaxed);
    PNTT_CALL_COUNT.store(0, Ordering::Relaxed);
}

/// Print timing statistics.
#[cfg(feature = "pntt-timing")]
pub fn print_timing() {
    let base_ns = BASE_LAYER_NANOS.load(Ordering::Relaxed);
    let butterfly_ns = BUTTERFLY_NANOS.load(Ordering::Relaxed);
    let calls = PNTT_CALL_COUNT.load(Ordering::Relaxed);
    let total_ns = base_ns + butterfly_ns;
    
    println!("\n=== PNTT Timing Statistics ===");
    println!("Total PNTT calls: {}", calls);
    println!("Base layer:   {:>10.3} ms ({:>5.1}%)", base_ns as f64 / 1_000_000.0, 100.0 * base_ns as f64 / total_ns as f64);
    println!("Butterfly:    {:>10.3} ms ({:>5.1}%)", butterfly_ns as f64 / 1_000_000.0, 100.0 * butterfly_ns as f64 / total_ns as f64);
    println!("Total:        {:>10.3} ms", total_ns as f64 / 1_000_000.0);
    if calls > 0 {
        println!("Avg per call: {:>10.3} ms", total_ns as f64 / 1_000_000.0 / calls as f64);
        println!("  Base layer: {:>10.3} µs", base_ns as f64 / 1_000.0 / calls as f64);
        println!("  Butterfly:  {:>10.3} µs", butterfly_ns as f64 / 1_000.0 / calls as f64);
    }
    println!("==============================\n");
}

use octet_reversal::*;
use params::*;

pub(crate) use mul_by_twiddle::*;

/// The main entrypoint of the radix-8 pseudo NTT algorithm.
#[cfg(not(feature = "unchecked-butterfly"))]
pub(crate) fn pntt<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd + CheckedMul + FromRef<In> + Clone + Send + Sync + Debug,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    pntt_impl::<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(input, params)
}

/// In-place PNTT that writes into an existing output buffer.
#[cfg(not(feature = "unchecked-butterfly"))]
pub(crate) fn pntt_into<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    out: &mut [MaybeUninit<Out>],
)
where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd + CheckedMul + FromRef<In> + Clone + Send + Sync + Debug,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    pntt_into_impl::<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(input, params, out)
}

/// In-place PNTT that writes into an existing output buffer.
#[cfg(feature = "unchecked-butterfly")]
pub(crate) fn pntt_into<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    out: &mut [MaybeUninit<Out>],
)
where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd
        + CheckedMul
        + FromRef<In>
        + Clone
        + Send
        + Sync
        + Debug
        + for<'a> AddAssign<&'a Out>,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    pntt_into_impl::<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(input, params, out)
}

/// The main entrypoint of the radix-8 pseudo NTT algorithm.
/// This version uses unchecked addition in butterflies for better performance.
#[cfg(feature = "unchecked-butterfly")]
pub(crate) fn pntt<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd + CheckedMul + FromRef<In> + Clone + Send + Sync + Debug + for<'a> AddAssign<&'a Out>,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    pntt_impl::<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(input, params)
}

/// Implementation of the radix-8 pseudo NTT algorithm.
#[cfg(not(feature = "unchecked-butterfly"))]
fn pntt_impl<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd + CheckedMul + FromRef<In> + Clone + Send + Sync + Debug,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    assert_eq!(
        C::INPUT_LEN,
        input.len(),
        "PNTT expects length = {}, got {}",
        C::INPUT_LEN,
        input.len()
    );

    #[cfg(feature = "pntt-timing")]
    PNTT_CALL_COUNT.fetch_add(1, Ordering::Relaxed);

    #[cfg(feature = "pntt-timing")]
    let base_start = std::time::Instant::now();
    
    let mut output = base_multiply_into_output::<_, _, _, MulInByTwiddle>(input, params);

    #[cfg(feature = "pntt-timing")]
    {
        let base_elapsed = base_start.elapsed().as_nanos() as u64;
        BASE_LAYER_NANOS.fetch_add(base_elapsed, Ordering::Relaxed);
    }

    #[cfg(feature = "pntt-timing")]
    let butterfly_start = std::time::Instant::now();
    
    combine_stages::<_, _, MulOutByTwiddle>(&mut output, params);

    #[cfg(feature = "pntt-timing")]
    {
        let butterfly_elapsed = butterfly_start.elapsed().as_nanos() as u64;
        BUTTERFLY_NANOS.fetch_add(butterfly_elapsed, Ordering::Relaxed);
    }

    output
}

#[cfg(not(feature = "unchecked-butterfly"))]
fn pntt_into_impl<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    out: &mut [MaybeUninit<Out>],
) where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd + CheckedMul + FromRef<In> + Clone + Send + Sync + Debug,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    assert_eq!(
        C::INPUT_LEN,
        input.len(),
        "PNTT expects length = {}, got {}",
        C::INPUT_LEN,
        input.len()
    );

    assert_eq!(
        C::OUTPUT_LEN,
        out.len(),
        "PNTT output expects length = {}, got {}",
        C::OUTPUT_LEN,
        out.len()
    );

    #[cfg(feature = "pntt-timing")]
    PNTT_CALL_COUNT.fetch_add(1, Ordering::Relaxed);

    #[cfg(feature = "pntt-timing")]
    let base_start = std::time::Instant::now();

    base_multiply_into_output_in_place::<_, _, _, MulInByTwiddle>(input, params, out);

    #[cfg(feature = "pntt-timing")]
    {
        let base_elapsed = base_start.elapsed().as_nanos() as u64;
        BASE_LAYER_NANOS.fetch_add(base_elapsed, Ordering::Relaxed);
    }

    // Safe because we have just initialized all output elements.
    let out_init = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut Out, out.len())
    };

    #[cfg(feature = "pntt-timing")]
    let butterfly_start = std::time::Instant::now();

    combine_stages::<_, _, MulOutByTwiddle>(out_init, params);

    #[cfg(feature = "pntt-timing")]
    {
        let butterfly_elapsed = butterfly_start.elapsed().as_nanos() as u64;
        BUTTERFLY_NANOS.fetch_add(butterfly_elapsed, Ordering::Relaxed);
    }
}

#[cfg(feature = "unchecked-butterfly")]
fn pntt_into_impl<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    out: &mut [MaybeUninit<Out>],
) where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd
        + CheckedMul
        + FromRef<In>
        + Clone
        + Send
        + Sync
        + Debug
        + for<'a> AddAssign<&'a Out>,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    assert_eq!(
        C::INPUT_LEN,
        input.len(),
        "PNTT expects length = {}, got {}",
        C::INPUT_LEN,
        input.len()
    );

    assert_eq!(
        C::OUTPUT_LEN,
        out.len(),
        "PNTT output expects length = {}, got {}",
        C::OUTPUT_LEN,
        out.len()
    );

    #[cfg(feature = "pntt-timing")]
    PNTT_CALL_COUNT.fetch_add(1, Ordering::Relaxed);

    #[cfg(feature = "pntt-timing")]
    let base_start = std::time::Instant::now();

    base_multiply_into_output_in_place_unchecked::<_, _, _, MulInByTwiddle>(input, params, out);

    #[cfg(feature = "pntt-timing")]
    {
        let base_elapsed = base_start.elapsed().as_nanos() as u64;
        BASE_LAYER_NANOS.fetch_add(base_elapsed, Ordering::Relaxed);
    }

    let out_init = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut Out, out.len())
    };

    #[cfg(feature = "pntt-timing")]
    let butterfly_start = std::time::Instant::now();

    combine_stages::<_, _, MulOutByTwiddle>(out_init, params);

    #[cfg(feature = "pntt-timing")]
    {
        let butterfly_elapsed = butterfly_start.elapsed().as_nanos() as u64;
        BUTTERFLY_NANOS.fetch_add(butterfly_elapsed, Ordering::Relaxed);
    }
}

/// Implementation of the radix-8 pseudo NTT algorithm with unchecked butterflies.
#[cfg(feature = "unchecked-butterfly")]
fn pntt_impl<In, Out, C, MulInByTwiddle, MulOutByTwiddle>(
    input: &[In],
    params: &Radix8PnttParams<C>,
) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: CheckedAdd + CheckedMul + FromRef<In> + Clone + Send + Sync + Debug + for<'a> AddAssign<&'a Out>,
    MulInByTwiddle: MulByTwiddle<In, PnttInt, Output = Out>,
    MulOutByTwiddle: MulByTwiddle<Out, PnttInt, Output = Out>,
{
    assert_eq!(
        C::INPUT_LEN,
        input.len(),
        "PNTT expects length = {}, got {}",
        C::INPUT_LEN,
        input.len()
    );

    #[cfg(feature = "pntt-timing")]
    PNTT_CALL_COUNT.fetch_add(1, Ordering::Relaxed);

    #[cfg(feature = "pntt-timing")]
    let base_start = std::time::Instant::now();
    
    let mut output = base_multiply_into_output_unchecked::<_, _, _, MulInByTwiddle>(input, params);

    #[cfg(feature = "pntt-timing")]
    {
        let base_elapsed = base_start.elapsed().as_nanos() as u64;
        BASE_LAYER_NANOS.fetch_add(base_elapsed, Ordering::Relaxed);
    }

    #[cfg(feature = "pntt-timing")]
    let butterfly_start = std::time::Instant::now();
    
    combine_stages::<_, _, MulOutByTwiddle>(&mut output, params);

    #[cfg(feature = "pntt-timing")]
    {
        let butterfly_elapsed = butterfly_start.elapsed().as_nanos() as u64;
        BUTTERFLY_NANOS.fetch_add(butterfly_elapsed, Ordering::Relaxed);
    }

    output
}

/// Performs the butterfly steps of the radix-8 pseudo NTT algorithm.
/// Assumes `out` contains the result of multiplications of the base chunks
/// with the `base_matrix`.
#[cfg(not(feature = "unchecked-butterfly"))]
#[allow(clippy::arithmetic_side_effects)]
fn combine_stages<R, C, M>(out: &mut [R], params: &Radix8PnttParams<C>)
where
    C: Config,
    R: CheckedAdd + CheckedMul + Clone + Send + Sync + Debug,
    M: MulByTwiddle<R, PnttInt, Output = R>,
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

                // Build mutable references to the 8 output positions directly
                // via pointer arithmetic, avoiding a Vec heap allocation.
                let chunk_ptr = chunk.as_mut_ptr();
                let ys: [&mut R; 8] = array::from_fn(|j| unsafe {
                    &mut *chunk_ptr.add(j * sub_chunk_length + i)
                });

                // Perform butterflies.
                apply_radix_8_butterflies::<_, _, M>(ys, &subresults, &layer_twiddles[i]);
            }
        });
    }
}

/// Performs the butterfly steps of the radix-8 pseudo NTT algorithm.
/// This version uses unchecked addition (no overflow checks) for better performance.
/// Use only when overflow is guaranteed not to occur.
#[cfg(feature = "unchecked-butterfly")]
#[allow(clippy::arithmetic_side_effects)]
fn combine_stages<R, C, M>(out: &mut [R], params: &Radix8PnttParams<C>)
where
    C: Config,
    R: Clone + Send + Sync + Debug + for<'a> AddAssign<&'a R>,
    M: MulByTwiddle<R, PnttInt, Output = R>,
{
    for k in 0..C::DEPTH {
        let sub_chunk_length = C::BASE_DIM * (1 << (3 * k));
        let layer_twiddles = &params.butterfly_twiddles[k];
        debug_assert_eq!(layer_twiddles.len(), sub_chunk_length);

        cfg_chunks_mut!(out, 8 * sub_chunk_length).for_each(|chunk: &mut [R]| {
            for i in 0..sub_chunk_length {
                let subresults: [R; 8] =
                    array::from_fn(|j| chunk[j * sub_chunk_length + i].clone());

                // Build mutable references to the 8 output positions directly
                // via pointer arithmetic, avoiding a Vec heap allocation.
                let chunk_ptr = chunk.as_mut_ptr();
                let ys: [&mut R; 8] = array::from_fn(|j| unsafe {
                    &mut *chunk_ptr.add(j * sub_chunk_length + i)
                });

                // Perform unchecked butterflies (no overflow checking).
                apply_radix_8_butterflies_unchecked::<_, _, M>(ys, &subresults, &layer_twiddles[i]);
            }
        });
    }
}

/// Allocates the output vector and performs base layer multiplications.
#[cfg(not(feature = "unchecked-butterfly"))]
#[allow(clippy::arithmetic_side_effects)]
fn base_multiply_into_output<In, Out, C, M>(input: &[In], params: &Radix8PnttParams<C>) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: Clone + CheckedAdd + CheckedMul + FromRef<In> + Send + Sync,
    M: MulByTwiddle<In, PnttInt, Output = Out>,
{
    cfg_into_iter!(0..C::OUTPUT_LEN)
        .map(|i| {
            let chunk = i >> C::BASE_DIM_LOG2; // i / C::BASE_DIM
            let row = i & C::BASE_DIM_MASK; // i % C::BASE_DIM
            let oct_rev_chunk = params.oct_rev_table[chunk];

            // We always know that the first column of the Vandermonde matrix
            // consists of 1's.
            params.base_matrix[row][1..].iter().enumerate().fold(
                Out::from_ref(&input[oct_rev_chunk]),
                |acc, (col, bm_row_col)| {
                    let term = M::mul_by_twiddle(
                        &input[oct_rev_chunk | ((col + 1) << (3 * C::DEPTH))],
                        bm_row_col,
                    );

                    add!(acc, &term)
                },
            )
        })
        .collect()
}

/// Writes base layer multiplication output directly into an uninitialized buffer.
#[cfg(not(feature = "unchecked-butterfly"))]
#[allow(clippy::arithmetic_side_effects)]
fn base_multiply_into_output_in_place<In, Out, C, M>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    out: &mut [MaybeUninit<Out>],
) where
    C: Config,
    In: Clone + Send + Sync,
    Out: Clone + CheckedAdd + CheckedMul + FromRef<In> + Send + Sync,
    M: MulByTwiddle<In, PnttInt, Output = Out>,
{
    cfg_chunks_mut!(out, C::BASE_DIM)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            // All elements within a chunk share the same oct_rev_chunk.
            let oct_rev_chunk = params.oct_rev_table[chunk_idx];
            for (row, out_cell) in chunk.iter_mut().enumerate() {
                let value = params.base_matrix[row][1..].iter().enumerate().fold(
                    Out::from_ref(&input[oct_rev_chunk]),
                    |acc, (col, bm_row_col)| {
                        let term = M::mul_by_twiddle(
                            &input[oct_rev_chunk | ((col + 1) << (3 * C::DEPTH))],
                            bm_row_col,
                        );

                        add!(acc, &term)
                    },
                );

                out_cell.write(value);
            }
        });
}

/// Allocates the output vector and performs base layer multiplications.
/// This version uses unchecked addition for better performance.
#[cfg(feature = "unchecked-butterfly")]
#[allow(clippy::arithmetic_side_effects)]
fn base_multiply_into_output_unchecked<In, Out, C, M>(input: &[In], params: &Radix8PnttParams<C>) -> Vec<Out>
where
    C: Config,
    In: Clone + Send + Sync,
    Out: Clone + FromRef<In> + Send + Sync + for<'a> AddAssign<&'a Out>,
    M: MulByTwiddle<In, PnttInt, Output = Out>,
{
    cfg_into_iter!(0..C::OUTPUT_LEN)
        .map(|i| {
            let chunk = i >> C::BASE_DIM_LOG2; // i / C::BASE_DIM
            let row = i & C::BASE_DIM_MASK; // i % C::BASE_DIM
            let oct_rev_chunk = params.oct_rev_table[chunk];

            // We always know that the first column of the Vandermonde matrix
            // consists of 1's. Use fused mul_by_twiddle_and_add to avoid temporaries.
            let mut acc = Out::from_ref(&input[oct_rev_chunk]);
            for (col, bm_row_col) in params.base_matrix[row][1..].iter().enumerate() {
                M::mul_by_twiddle_and_add(
                    &mut acc,
                    &input[oct_rev_chunk | ((col + 1) << (3 * C::DEPTH))],
                    bm_row_col,
                );
            }
            acc
        })
        .collect()
}

/// Writes base layer multiplication output directly into an uninitialized buffer.
/// This version uses unchecked addition for better performance.
#[cfg(feature = "unchecked-butterfly")]
#[allow(clippy::arithmetic_side_effects)]
fn base_multiply_into_output_in_place_unchecked<In, Out, C, M>(
    input: &[In],
    params: &Radix8PnttParams<C>,
    out: &mut [MaybeUninit<Out>],
) where
    C: Config,
    In: Clone + Send + Sync,
    Out: Clone + FromRef<In> + Send + Sync + for<'a> AddAssign<&'a Out>,
    M: MulByTwiddle<In, PnttInt, Output = Out>,
{
    cfg_chunks_mut!(out, C::BASE_DIM)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            // All elements within a chunk share the same oct_rev_chunk.
            let oct_rev_chunk = params.oct_rev_table[chunk_idx];
            for (row, out_cell) in chunk.iter_mut().enumerate() {
                // Use fused mul_by_twiddle_and_add to avoid temporaries.
                let mut acc = Out::from_ref(&input[oct_rev_chunk]);
                for (col, bm_row_col) in params.base_matrix[row][1..].iter().enumerate() {
                    M::mul_by_twiddle_and_add(
                        &mut acc,
                        &input[oct_rev_chunk | ((col + 1) << (3 * C::DEPTH))],
                        bm_row_col,
                    );
                }
                out_cell.write(acc);
            }
        });
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use ark_ff::{Field, PrimeField, Zero};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use crypto_primitives::crypto_bigint_int::Int;
    use itertools::Itertools;
    use octet_reversal::octet_reversal;
    use zinc_utils::{CHECKED, mul_by_scalar::MulByScalar};

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

            #[cfg(not(feature = "unchecked-butterfly"))]
            let output =
                base_multiply_into_output::<_, _, _, MBSMulByTwiddle<CHECKED>>(&input, &params);
            #[cfg(feature = "unchecked-butterfly")]
            let output =
                base_multiply_into_output_unchecked::<_, _, _, MBSMulByTwiddle<CHECKED>>(&input, &params);

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

            let res: Vec<Int<4>> =
                pntt::<_, _, _, MBSMulByTwiddle<CHECKED>, MBSMulByTwiddle<CHECKED>>(
                    &input, &params,
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
    fn pntt_against_arkworks() {
        pntt_against_arkworks_generic::<PnttConfigF2_16_1<1>>();
        pntt_against_arkworks_generic::<PnttConfigF2_16_1<2>>();
        pntt_against_arkworks_generic::<PnttConfigF2_16_1<3>>();
    }
}
