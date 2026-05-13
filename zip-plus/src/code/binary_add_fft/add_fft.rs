//! Iterative radix-2 additive Reed-Solomon FFT over `Z[X] / f̃`.
//!
//! Mirrors the LCH (Lin-Chung-Han) novel-polynomial-basis FFT
//! structurally, using the simplification `s_i(β_i) = 1` granted by
//! the Cantor basis. Inputs are interpreted as novel-basis
//! coefficients; outputs are evaluations at the Cantor-basis-enumerated
//! points of `V_d` where `d = log2(codeword_len)`. The algorithm goes
//! natural-order to natural-order — no bit-reversal step.
//!
//! ## Algorithm sketch
//!
//! For each stage `i = m-1, m-2, …, 0`, partition `data[0..n]` into
//! blocks of size `2 · 2^i`. Within each block, pair entry `k` with
//! entry `k + 2^i`. For each pair `(a, b)`, apply the butterfly
//!
//! ```text
//!     a' = a + t · b
//!     b' = a' + b   (= a + (t + 1) · b)
//! ```
//!
//! where the twiddle `t = s_i(α_block)` is the subspace polynomial `s_i`
//! evaluated at the "block offset" — the `GF(2^32)` element formed by
//! summing the Cantor basis vectors `β_l` for each set bit `l ≥ i+1`
//! of `block_start`. The butterfly is `F_2`-linear in `(a, b)` over
//! `GF(2^32)`; structurally lifting the same equations to `Z[X] / f̃`
//! gives a code whose mod-2 reduction is the original additive RS code.
//!
//! ## Correctness witnesses (tests in this module)
//!
//! 1. `gf32_fft_matches_naive_eval` — the pure-`GF(2^32)` version of
//!    this algorithm agrees with the naive
//!    `P(α) = ∑ c_i · X_i(α)` evaluator. (Validates the FFT structure.)
//!
//! 2. `lifted_fft_mod_2_matches_gf32_fft` — running the *lifted*
//!    `Z[X] / f̃` FFT on lifted inputs and reducing the outputs mod 2
//!    yields the `GF(2^32)` FFT output. (The **core soundness witness**
//!    — the lift commutes with the encoding.)
//!
//! 3. `lifted_fft_is_linear` — `FFT(a + b) = FFT(a) + FFT(b)` in
//!    `Z[X] / f̃`. (The `LinearCode` invariant.)

use super::basis::Gf2_32;
use super::params::{Config, Radix2AddFftParams, evaluate_subspace_poly_at_gf32};
use super::ring_ops::FftRingElement;
use zinc_utils::cfg_chunks_mut;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Apply the radix-2 additive FFT in place.
///
/// `data` must have length `params.codeword_len`. On entry it holds
/// novel-basis coefficients (with zero padding beyond `params.row_len`
/// when called from the encoder). On exit, `data[k]` holds the
/// evaluation of the polynomial at `params.eval_points[k]`, in the
/// `T`-flavoured ring.
///
/// The kernel is polymorphic over the ring element type `T` via
/// [`FftRingElement`]. Typically `T = DensePolynomial<Int<N>, 32>`
/// for the lifted-`Z[X]/f̃` encoder, or `T = Gf2_32` for the reference
/// implementation used in tests.
///
/// **Coefficient growth.** Each butterfly maps `(a, b) -> (a + t·b, a + (t+1)·b)`
/// where `t` is a lifted `Gf2_32`. The multiplication `t·b` can blow
/// each `Int<N>` coefficient up by up to `D = 32` (the number of
/// summands inside the lifted-twiddle multiplication), plus the final
/// `+ b` adds one more `b`'s worth, giving a per-stage growth factor
/// of at most `D + 1 = 33`. After `m` stages the worst case
/// coefficient magnitude is bounded by `33^m · max(|input|)`. Callers
/// must size `N` accordingly; the kernel itself performs no overflow
/// checking.
pub fn additive_fft<C: Config, T: FftRingElement>(
    data: &mut [T],
    params: &Radix2AddFftParams<C>,
) {
    assert_eq!(
        data.len(),
        params.codeword_len,
        "additive_fft: data length {} does not match codeword_len {}",
        data.len(),
        params.codeword_len
    );

    let m = params.log2_codeword_len;
    for i in (0..m).rev() {
        let h = 1usize << i;
        let block_size = h << 1;
        let s_i = &params.subspace_polys[i];
        let cantor_basis = &params.cantor_basis;

        // Stages are strictly sequential (each butterfly reads results
        // of the previous stage), but the blocks WITHIN a stage are
        // independent — `chunks_exact_mut(block_size)` gives each task
        // a disjoint sub-slice. Under `parallel`, rayon parallelizes
        // across blocks; without the feature, this is a plain
        // sequential `chunks_mut`. For late stages with only one or
        // two blocks, rayon's adaptive split avoids spawning extra
        // tasks, so the parallel branch isn't a regression.
        cfg_chunks_mut!(data, block_size)
            .enumerate()
            .for_each(|(block_idx, block_data)| {
                let block_start = block_idx * block_size;
                let block_offset = block_offset_for_stage(block_start, i, m, cantor_basis);
                let twiddle = evaluate_subspace_poly_at_gf32(s_i, block_offset);

                for k in 0..h {
                    butterfly(block_data, k, k + h, twiddle);
                }
            });
    }
}

/// The single butterfly operation, in place.
///
/// `(data[idx_a], data[idx_b]) := (a + t·b, a + (t+1)·b)`, where
/// `a = data[idx_a]`, `b = data[idx_b]`, and `t` is the lifted-`Gf2_32`
/// twiddle.
///
/// Fast path: when `twiddle == 0` (which happens for the first block
/// of every stage, since `block_offset = 0 ∈ V_i` and `s_i` vanishes
/// on `V_i`), the butterfly degenerates to `(a, b) → (a, a + b)` and
/// the `mul_by_lifted_gf32` call (~512 `Int<N>` additions) is skipped
/// entirely.
#[inline]
fn butterfly<T: FftRingElement>(
    data: &mut [T],
    idx_a: usize,
    idx_b: usize,
    twiddle: Gf2_32,
) {
    if twiddle == Gf2_32::ZERO {
        // `new_a = a`, `new_b = a + b`. Avoid cloning `b` since we
        // never multiply it.
        let new_b = data[idx_a].fft_add(&data[idx_b]);
        data[idx_b] = new_b;
        return;
    }
    let b = data[idx_b].clone();
    let t_b = b.fft_mul_lifted_gf32(twiddle);
    let new_a = data[idx_a].fft_add(&t_b);
    let new_b = new_a.fft_add(&b);
    data[idx_a] = new_a;
    data[idx_b] = new_b;
}

/// Compute the `GF(2^32)` "block offset" at FFT stage `i` for block
/// starting at index `block_start`. This is the sum of Cantor basis
/// vectors `β_l` for each bit `l` in the range `(i, m)` that is set
/// in `block_start`.
#[inline]
fn block_offset_for_stage(
    block_start: usize,
    i: usize,
    m: usize,
    cantor_basis: &[Gf2_32],
) -> Gf2_32 {
    let mut offset = Gf2_32::ZERO;
    for l in (i + 1)..m {
        if (block_start >> l) & 1 != 0 {
            offset = offset.add(cantor_basis[l]);
        }
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::params::AddFftConfigGF2_32;
    use super::super::ring_ops::{REDUCED_LEN, lift_gf32, reduce_to_gf32};
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_poly::univariate::dense::DensePolynomial;

    type P = Radix2AddFftParams<AddFftConfigGF2_32>;

    /// Naive `P(α) = ∑ c_i · X_i(α)` evaluator over `GF(2^32)`.
    fn naive_eval_at_gf32<C: Config>(
        coeffs: &[Gf2_32],
        params: &Radix2AddFftParams<C>,
        alpha: Gf2_32,
    ) -> Gf2_32 {
        let m = params.log2_codeword_len;
        let s_at_alpha: Vec<Gf2_32> = (0..m)
            .map(|j| evaluate_subspace_poly_at_gf32(&params.subspace_polys[j], alpha))
            .collect();
        let mut acc = Gf2_32::ZERO;
        for (i, &c_i) in coeffs.iter().enumerate() {
            let mut x_i = Gf2_32::ONE;
            for j in 0..m {
                if (i >> j) & 1 != 0 {
                    x_i = x_i.mul(s_at_alpha[j]);
                }
            }
            acc = acc.add(c_i.mul(x_i));
        }
        acc
    }

    /// Lift a `Gf2_32` to `DensePolynomial<Int<N>, 32>` (the lifted
    /// `Z[X]/f̃` representation as a `DensePolynomial`).
    fn lift_to_poly<const N: usize>(g: Gf2_32) -> DensePolynomial<Int<N>, REDUCED_LEN> {
        DensePolynomial { coeffs: lift_gf32(g) }
    }

    #[test]
    fn gf32_fft_matches_naive_eval() {
        for log_n in 1..=5 {
            let n = 1usize << log_n;
            let p = P::new(n, n).expect("valid params");
            let coeffs: Vec<Gf2_32> = (0..n)
                .map(|i| {
                    Gf2_32(
                        (i as u32)
                            .wrapping_mul(0xdead_beef)
                            .wrapping_add(0xcafe_babe),
                    )
                })
                .collect();
            let expected: Vec<Gf2_32> = p
                .eval_points
                .iter()
                .map(|&alpha| naive_eval_at_gf32(&coeffs, &p, alpha))
                .collect();
            let mut got = coeffs.clone();
            additive_fft(&mut got, &p);
            assert_eq!(got, expected, "FFT disagrees with naive at log_n={log_n}");
        }
    }

    #[test]
    fn lifted_fft_mod_2_matches_gf32_fft() {
        for log_n in 1..=5 {
            let n = 1usize << log_n;
            let p = P::new(n, n).expect("valid params");
            let coeffs: Vec<Gf2_32> = (0..n)
                .map(|i| {
                    Gf2_32(
                        (i as u32)
                            .wrapping_mul(0xdead_beef)
                            .wrapping_add(0xcafe_babe),
                    )
                })
                .collect();
            let mut gf_result = coeffs.clone();
            additive_fft(&mut gf_result, &p);
            let mut lifted: Vec<DensePolynomial<Int<4>, REDUCED_LEN>> =
                coeffs.iter().map(|&c| lift_to_poly::<4>(c)).collect();
            additive_fft(&mut lifted, &p);
            let lifted_mod_2: Vec<Gf2_32> =
                lifted.iter().map(|d| reduce_to_gf32(&d.coeffs)).collect();
            assert_eq!(
                lifted_mod_2, gf_result,
                "lift does not commute with FFT at log_n={log_n}"
            );
        }
    }

    /// Same as `lifted_fft_mod_2_matches_gf32_fft` but storing the
    /// lifted coefficients in native `i128`s instead of `Int<N>`s.
    /// Exercises the `FftRingElement` impl for `DensePolynomial<i128, 32>`.
    #[test]
    fn lifted_fft_mod_2_matches_gf32_fft_i128() {
        for log_n in 1..=5 {
            let n = 1usize << log_n;
            let p = P::new(n, n).expect("valid params");
            let coeffs: Vec<Gf2_32> = (0..n)
                .map(|i| {
                    Gf2_32(
                        (i as u32)
                            .wrapping_mul(0xdead_beef)
                            .wrapping_add(0xcafe_babe),
                    )
                })
                .collect();
            let mut gf_result = coeffs.clone();
            additive_fft(&mut gf_result, &p);
            // Lift to i128 coefficients (each Gf2_32 bit -> 0i128 or 1i128).
            let mut lifted: Vec<DensePolynomial<i128, REDUCED_LEN>> = coeffs
                .iter()
                .map(|&c| {
                    let mut arr = [0i128; REDUCED_LEN];
                    for (i, slot) in arr.iter_mut().enumerate() {
                        if c.coeff(i) {
                            *slot = 1;
                        }
                    }
                    DensePolynomial { coeffs: arr }
                })
                .collect();
            additive_fft(&mut lifted, &p);
            // Reduce mod 2 by inspecting the parity of each i128 coefficient.
            let lifted_mod_2: Vec<Gf2_32> = lifted
                .iter()
                .map(|d| {
                    let mut bits = 0u32;
                    for (i, c) in d.coeffs.iter().enumerate() {
                        if c.rem_euclid(2) == 1 {
                            bits |= 1 << i;
                        }
                    }
                    Gf2_32(bits)
                })
                .collect();
            assert_eq!(
                lifted_mod_2, gf_result,
                "i128-lifted FFT does not commute with FFT at log_n={log_n}"
            );
        }
    }

    #[test]
    fn lifted_fft_is_linear() {
        let log_n = 5;
        let n = 1usize << log_n;
        let p = P::new(n, n).expect("valid params");

        let a_coeffs: Vec<Gf2_32> = (0..n)
            .map(|i| Gf2_32((i as u32).wrapping_mul(0x0123_4567)))
            .collect();
        let b_coeffs: Vec<Gf2_32> = (0..n)
            .map(|i| Gf2_32((i as u32).wrapping_mul(0x89ab_cdef)))
            .collect();

        let mut a_lift: Vec<DensePolynomial<Int<4>, REDUCED_LEN>> =
            a_coeffs.iter().map(|&c| lift_to_poly::<4>(c)).collect();
        let mut b_lift: Vec<DensePolynomial<Int<4>, REDUCED_LEN>> =
            b_coeffs.iter().map(|&c| lift_to_poly::<4>(c)).collect();
        let mut sum_lift: Vec<DensePolynomial<Int<4>, REDUCED_LEN>> = (0..n)
            .map(|i| a_lift[i].fft_add(&b_lift[i]))
            .collect();

        additive_fft(&mut a_lift, &p);
        additive_fft(&mut b_lift, &p);
        additive_fft(&mut sum_lift, &p);

        for i in 0..n {
            let expected = a_lift[i].fft_add(&b_lift[i]);
            assert_eq!(sum_lift[i], expected, "linearity broken at i={i}");
        }
    }

    /// Encoding setting: padded input. `row_len < codeword_len` with
    /// the high entries zero — verifies the FFT handles zero-padded
    /// inputs the same as the explicit-novel-basis input would.
    #[test]
    fn lifted_fft_handles_zero_padding() {
        let row_len = 4;
        let codeword_len = 16;
        let p = P::new(row_len, codeword_len).expect("valid params");

        let row: Vec<Gf2_32> = vec![Gf2_32(0x1234), Gf2_32(0x5678), Gf2_32(0x9abc), Gf2_32(0xdef0)];
        let mut padded: Vec<Gf2_32> = row.clone();
        padded.resize(codeword_len, Gf2_32::ZERO);

        let mut gf_result = padded.clone();
        additive_fft(&mut gf_result, &p);

        let mut lifted: Vec<DensePolynomial<Int<3>, REDUCED_LEN>> =
            padded.iter().map(|&c| lift_to_poly::<3>(c)).collect();
        additive_fft(&mut lifted, &p);
        let lifted_mod_2: Vec<Gf2_32> =
            lifted.iter().map(|d| reduce_to_gf32(&d.coeffs)).collect();

        assert_eq!(lifted_mod_2, gf_result);
    }
}
