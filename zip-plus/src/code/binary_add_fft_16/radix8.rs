//! Radix-8 additive Reed-Solomon FFT over `Z[X] / f̃` (D = 16 version).
//!
//! Each meta-stage covers three consecutive subspace polynomials
//! `s_{3K}, s_{3K+1}, s_{3K+2}` and applies, per block of 8 sub-blocks,
//! an 8×8 Vandermonde matrix `V_{K,B}` whose entry `(j, i)` is the
//! `i`-th novel-basis polynomial for the 3-dimensional sub-recursion
//! evaluated at the `j`-th sub-block point:
//!
//! ```text
//! V_{K,B}[j][i] = ∏_{a ∈ bits(i)} s_{3K+a}(v_{K,B,j}),
//! v_{K,B,j}     = outer_block_offset(B)
//!               + j_0·β_{3K} + j_1·β_{3K+1} + j_2·β_{3K+2}.
//! ```
//!
//! Entries live in `GF(2^16)`; at FFT time each entry is lifted as a
//! `{0,1}`-coefficient polynomial in `Z[X]/f̃` and the matvec is carried
//! out in `Z[X]/f̃` via `T::fft_acc_mul_lifted_gf16` plus
//! `T::fft_finalize_acc`. Z-carries propagate across meta-stages — the
//! resulting linear code is not the same as the radix-2 lifted FFT's
//! image (the two share the same mod-2 reduction but differ on Z).
//!
//! The kernel requires `log2_codeword_len % 3 == 0`.

use super::basis::Gf2_16;
use super::params::{Config16, Radix2AddFftParams16, evaluate_subspace_poly_at_gf16};
use super::ring_ops::FftRingElement16;
use crate::ZipError;
use std::{fmt::Debug, marker::PhantomData};
use zinc_utils::cfg_chunks_mut;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Precomputed parameters for the radix-8 additive FFT.
///
/// We require `codeword_len.is_power_of_two()`,
/// `row_len.is_power_of_two()`, `row_len ≤ codeword_len ≤ 2^16`, and
/// crucially `log2(codeword_len) % 3 == 0` so the kernel is a clean
/// sequence of `m / 3` radix-8 meta-stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Radix8FftParams16<C: Config16> {
    /// Underlying radix-2 parameters (Cantor basis, subspace polys,
    /// eval points, sizes).
    pub base: Radix2AddFftParams16<C>,

    /// Number of radix-8 meta-stages: `log2(codeword_len) / 3`.
    pub num_meta_stages: usize,

    /// Precomputed 8×8 Vandermonde matrices, indexed
    /// `[meta_stage_K][outer_block_idx]`. Each matrix entry is a
    /// `Gf2_16` to be lifted at FFT time.
    pub vandermondes: Vec<Vec<[[Gf2_16; 8]; 8]>>,

    _phantom: PhantomData<C>,
}

impl<C: Config16> Radix8FftParams16<C> {
    /// Construct precomputed parameters. Validates that
    /// `log2(codeword_len) % 3 == 0` in addition to the radix-2
    /// validation.
    pub fn new(row_len: usize, codeword_len: usize) -> Result<Self, ZipError> {
        let base = Radix2AddFftParams16::<C>::new(row_len, codeword_len)?;
        let m = base.log2_codeword_len;
        if !m.is_multiple_of(3) {
            return Err(ZipError::InvalidPcsParam(format!(
                "Radix8FftParams16: log2(codeword_len) ({m}) must be divisible by 3"
            )));
        }
        let num_meta_stages = m / 3;
        let vandermondes = precompute_vandermondes(&base, num_meta_stages);
        Ok(Self {
            base,
            num_meta_stages,
            vandermondes,
            _phantom: PhantomData,
        })
    }

    #[inline]
    pub fn row_len(&self) -> usize {
        self.base.row_len
    }

    #[inline]
    pub fn codeword_len(&self) -> usize {
        self.base.codeword_len
    }

    #[inline]
    pub fn log2_codeword_len(&self) -> usize {
        self.base.log2_codeword_len
    }
}

/// Compute the `GF(2^16)` "outer block offset" at meta-stage `K`. For
/// outer block starting at codeword index `outer_start` (which is a
/// multiple of `2^(3(K+1))`), this is
/// `∑_{l ≥ 3(K+1)} (outer_start_bit_l) · β_l`.
#[inline]
fn outer_block_offset_for_meta_stage(
    outer_start: usize,
    meta_stage_k: usize,
    m: usize,
    cantor_basis: &[Gf2_16],
) -> Gf2_16 {
    let mut offset = Gf2_16::ZERO;
    let low = 3 * (meta_stage_k + 1);
    for l in low..m {
        if (outer_start >> l) & 1 != 0 {
            offset = offset.add(cantor_basis[l]);
        }
    }
    offset
}

/// Precompute the 8×8 Vandermonde matrices for every meta-stage K and
/// every outer block in that meta-stage.
fn precompute_vandermondes<C: Config16>(
    base: &Radix2AddFftParams16<C>,
    num_meta_stages: usize,
) -> Vec<Vec<[[Gf2_16; 8]; 8]>> {
    let m = base.log2_codeword_len;
    let cantor = &base.cantor_basis;
    let codeword_len = base.codeword_len;

    let mut out: Vec<Vec<[[Gf2_16; 8]; 8]>> = Vec::with_capacity(num_meta_stages);

    for k in 0..num_meta_stages {
        let outer_block_size = 1usize << (3 * (k + 1));
        let num_blocks = codeword_len / outer_block_size;
        let mut per_meta: Vec<[[Gf2_16; 8]; 8]> = Vec::with_capacity(num_blocks);

        let s0 = &base.subspace_polys[3 * k];
        let s1 = &base.subspace_polys[3 * k + 1];
        let s2 = &base.subspace_polys[3 * k + 2];

        let beta0 = cantor[3 * k];
        let beta1 = cantor[3 * k + 1];
        let beta2 = cantor[3 * k + 2];

        for b in 0..num_blocks {
            let outer_start = b * outer_block_size;
            let outer_off = outer_block_offset_for_meta_stage(outer_start, k, m, cantor);

            // Build the 8 sub-block evaluation points
            // v_j = outer_off + j_0·β_{3K} + j_1·β_{3K+1} + j_2·β_{3K+2}.
            let mut v_points = [Gf2_16::ZERO; 8];
            for j in 0..8 {
                let mut v = outer_off;
                if (j >> 0) & 1 != 0 {
                    v = v.add(beta0);
                }
                if (j >> 1) & 1 != 0 {
                    v = v.add(beta1);
                }
                if (j >> 2) & 1 != 0 {
                    v = v.add(beta2);
                }
                v_points[j] = v;
            }

            // Evaluate the three subspace polynomials at each v_j.
            let mut s_evals: [[Gf2_16; 8]; 3] =
                [[Gf2_16::ZERO; 8], [Gf2_16::ZERO; 8], [Gf2_16::ZERO; 8]];
            for j in 0..8 {
                s_evals[0][j] = evaluate_subspace_poly_at_gf16(s0, v_points[j]);
                s_evals[1][j] = evaluate_subspace_poly_at_gf16(s1, v_points[j]);
                s_evals[2][j] = evaluate_subspace_poly_at_gf16(s2, v_points[j]);
            }

            // Build the 8×8 matrix: M[j][i] = ∏_{a ∈ bits(i)} s_{3K+a}(v_j).
            let mut matrix = [[Gf2_16::ZERO; 8]; 8];
            for j in 0..8 {
                for i in 0..8 {
                    let mut x_i = Gf2_16::ONE;
                    if (i >> 0) & 1 != 0 {
                        x_i = x_i.mul(s_evals[0][j]);
                    }
                    if (i >> 1) & 1 != 0 {
                        x_i = x_i.mul(s_evals[1][j]);
                    }
                    if (i >> 2) & 1 != 0 {
                        x_i = x_i.mul(s_evals[2][j]);
                    }
                    matrix[j][i] = x_i;
                }
            }

            per_meta.push(matrix);
        }

        out.push(per_meta);
    }

    out
}

/// Apply the radix-8 additive FFT in place.
///
/// Iterates meta-stages from `num_meta_stages - 1` down to `0`. At each
/// meta-stage K:
/// - Outer block size = `2^(3(K+1))` codeword positions.
/// - Sub-block stride = `2^(3K)` codeword positions.
/// - For each outer block B (looked up in the precomputed matrices),
///   for each offset `t ∈ 0..stride`, perform the 8×8 matvec
///   `output[j] = Σ_i V_{K,B}[j][i] · input[i]` on the 8-tuple of
///   sub-block elements at positions `outer_start + i·stride + t`,
///   carried out in `Z[X]/f̃` via the `FftRingElement16` trait.
pub fn additive_fft_radix8_16<C: Config16, T: FftRingElement16>(
    data: &mut [T],
    params: &Radix8FftParams16<C>,
) {
    assert_eq!(
        data.len(),
        params.base.codeword_len,
        "additive_fft_radix8_16: data length {} does not match codeword_len {}",
        data.len(),
        params.base.codeword_len
    );

    // Meta-stages run from high to low, matching the radix-2 ordering.
    for k in (0..params.num_meta_stages).rev() {
        let outer_block_size = 1usize << (3 * (k + 1));
        let stride = 1usize << (3 * k);
        let matrices = &params.vandermondes[k];

        cfg_chunks_mut!(data, outer_block_size)
            .enumerate()
            .for_each(|(block_idx, block_data)| {
                let matrix = &matrices[block_idx];
                radix8_meta_stage_block::<T>(block_data, stride, matrix);
            });
    }
}

/// Apply one meta-stage's 8×8 matvec to a single outer block.
#[inline]
fn radix8_meta_stage_block<T: FftRingElement16>(
    block_data: &mut [T],
    stride: usize,
    matrix: &[[Gf2_16; 8]; 8],
) {
    // For each offset t within the sub-block, gather the 8 inputs
    // from sub-blocks i = 0..8 at offset t and apply the matvec.
    //
    // We snapshot the relevant 8 inputs into a small stack array per t.
    for t in 0..stride {
        // Gather inputs.
        let inputs: [T; 8] = [
            block_data[0 * stride + t].clone(),
            block_data[1 * stride + t].clone(),
            block_data[2 * stride + t].clone(),
            block_data[3 * stride + t].clone(),
            block_data[4 * stride + t].clone(),
            block_data[5 * stride + t].clone(),
            block_data[6 * stride + t].clone(),
            block_data[7 * stride + t].clone(),
        ];

        // Compute outputs[j] = Σ_i matrix[j][i] · inputs[i] in Z[X]/f̃.
        for j in 0..8 {
            let mut acc = T::fft_unreduced_zero();
            for i in 0..8 {
                let tw = matrix[j][i];
                if tw == Gf2_16::ZERO {
                    continue;
                }
                T::fft_acc_mul_lifted_gf16(&inputs[i], tw, &mut acc);
            }
            block_data[j * stride + t] = T::fft_finalize_acc(acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::params::AddFftConfigGF2_16;

    type P8 = Radix8FftParams16<AddFftConfigGF2_16>;

    #[test]
    fn rejects_log_codeword_not_divisible_by_3() {
        // log_n = 4 is not a multiple of 3.
        assert!(P8::new(16, 16).is_err());
        assert!(P8::new(2, 32).is_err());
    }

    #[test]
    fn accepts_log_codeword_divisible_by_3() {
        assert!(P8::new(8, 8).is_ok());
        assert!(P8::new(8, 64).is_ok());
        assert!(P8::new(64, 4096).is_ok());
    }
}
