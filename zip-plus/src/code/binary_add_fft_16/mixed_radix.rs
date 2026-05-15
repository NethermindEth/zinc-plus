//! Mixed-radix additive Reed-Solomon FFT over `Z[X] / f̃` (D = 16).
//!
//! Generalises the radix-8 evaluator: the `m`-bit FFT
//! (`m = log2(codeword_len)`) is decomposed into a sequence of stages,
//! each consuming either three bits (a radix-8, 8×8 Vandermonde matvec)
//! or four bits (a radix-16, 16×16 matvec). Radix-16 stages are placed
//! at the base — the lowest bit positions, smallest stride.
//!
//! A stage covering subspace polynomials `s_{off}, …, s_{off+r-1}`
//! applies, per outer block, the matrix
//!
//! ```text
//! V[j][i] = ∏_{a ∈ bits(i)} s_{off+a}(v_j),
//! v_j     = outer_block_offset(B) + ∑_a j_a · β_{off+a},
//! ```
//!
//! with `i, j ∈ [0, 2^r)`. Entries live in `GF(2^16)`; at FFT time each
//! entry is lifted as a `{0,1}`-coefficient polynomial in `Z[X]/f̃` and
//! the matvec is carried out in `Z[X]/f̃` via the [`FftRingElement16`]
//! trait. Z-carries propagate across stages.
//!
//! Pure radix-8 (`m % 3 == 0`) is the special case with no radix-16
//! stages, and is bit-for-bit identical to the historic radix-8 kernel.
//! Allowing one or two radix-16 base stages lifts the `m % 3 == 0`
//! constraint: an `m`-bit FFT is feasible whenever `m = 3a + 4b` for
//! non-negative `a, b`, i.e. for every `m` except `{1, 2, 5}`.

use super::basis::Gf2_16;
use super::params::{Config16, Radix2AddFftParams16, evaluate_subspace_poly_at_gf16};
use super::ring_ops::FftRingElement16;
use crate::ZipError;
use std::{fmt::Debug, marker::PhantomData};
use zinc_utils::cfg_chunks_mut;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// One stage of the mixed-radix evaluator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadixStage {
    /// `log2` of this stage's radix: `3` (radix-8) or `4` (radix-16).
    pub radix_log: usize,

    /// Bit offset of the lowest subspace polynomial this stage covers;
    /// the stage spans `s_{bit_offset}, …, s_{bit_offset+radix_log-1}`.
    pub bit_offset: usize,

    /// One flat `2^radix_log × 2^radix_log` Vandermonde matrix per
    /// outer block, row-major (`matrix[j * radix + i]`). Entries are
    /// `Gf2_16` to be lifted at FFT time.
    pub vandermondes: Vec<Vec<Gf2_16>>,
}

/// Precomputed parameters for the mixed-radix additive FFT.
///
/// We require `codeword_len.is_power_of_two()`,
/// `row_len.is_power_of_two()`, `row_len ≤ codeword_len ≤ 2^16`, and
/// that `m = log2(codeword_len)` is expressible as `3a + 4b` with
/// non-negative `a, b` (every `m` except `{1, 2, 5}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedRadixFftParams16<C: Config16> {
    /// Underlying radix-2 parameters (Cantor basis, subspace polys,
    /// eval points, sizes).
    pub base: Radix2AddFftParams16<C>,

    /// Stages, ordered base → top (ascending `bit_offset`).
    pub stages: Vec<RadixStage>,

    _phantom: PhantomData<C>,
}

/// Decompose an `m`-bit FFT into stage radices, base → top.
///
/// Returns the `log2`-radices (`3` or `4`) of each stage. Radix-16
/// stages (`4`) come first (at the base). Errors if `m` cannot be
/// written as `3a + 4b`, i.e. `m ∈ {1, 2, 5}`.
pub fn decompose_radices(m: usize) -> Result<Vec<usize>, ZipError> {
    let (num16, num8) = match m % 3 {
        0 => (0usize, m / 3),
        1 if m >= 4 => (1usize, (m - 4) / 3),
        2 if m >= 8 => (2usize, (m - 8) / 3),
        _ => {
            return Err(ZipError::InvalidPcsParam(format!(
                "MixedRadixFftParams16: log2(codeword_len) ({m}) is not expressible \
                 as 3a+4b (only m ∈ {{1, 2, 5}} are excluded)"
            )));
        }
    };
    let mut radices = vec![4usize; num16];
    radices.extend(std::iter::repeat(3usize).take(num8));
    Ok(radices)
}

impl<C: Config16> MixedRadixFftParams16<C> {
    /// Construct precomputed parameters. Validates the radix-2
    /// preconditions plus the `3a + 4b` representability of
    /// `log2(codeword_len)`.
    pub fn new(row_len: usize, codeword_len: usize) -> Result<Self, ZipError> {
        let base = Radix2AddFftParams16::<C>::new(row_len, codeword_len)?;
        let m = base.log2_codeword_len;
        let radix_logs = decompose_radices(m)?;

        let mut stages = Vec::with_capacity(radix_logs.len());
        let mut bit_offset = 0usize;
        for &radix_log in &radix_logs {
            let vandermondes = precompute_stage_vandermondes(&base, bit_offset, radix_log);
            stages.push(RadixStage {
                radix_log,
                bit_offset,
                vandermondes,
            });
            bit_offset += radix_log;
        }
        debug_assert_eq!(bit_offset, m);

        Ok(Self {
            base,
            stages,
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

    /// Number of radix-16 stages (`0`, `1`, or `2`).
    #[inline]
    pub fn num_radix16_stages(&self) -> usize {
        self.stages.iter().filter(|s| s.radix_log == 4).count()
    }
}

/// Precompute the per-block Vandermonde matrices for one stage.
///
/// The stage covers subspace polynomials `s_{off}, …, s_{off+r-1}`,
/// where `off = bit_offset` and `r = radix_log`. Each outer block has
/// size `2^(off+r)`; there are `codeword_len / 2^(off+r)` of them.
fn precompute_stage_vandermondes<C: Config16>(
    base: &Radix2AddFftParams16<C>,
    bit_offset: usize,
    radix_log: usize,
) -> Vec<Vec<Gf2_16>> {
    let m = base.log2_codeword_len;
    let cantor = &base.cantor_basis;
    let codeword_len = base.codeword_len;

    let radix = 1usize << radix_log;
    let outer_block_size = 1usize << (bit_offset + radix_log);
    let num_blocks = codeword_len / outer_block_size;

    let betas: Vec<Gf2_16> = (0..radix_log).map(|a| cantor[bit_offset + a]).collect();

    let mut out: Vec<Vec<Gf2_16>> = Vec::with_capacity(num_blocks);

    for b in 0..num_blocks {
        let outer_start = b * outer_block_size;

        // Outer block offset: ∑_{l ≥ bit_offset+radix_log} bit_l(outer_start) · β_l.
        let mut outer_off = Gf2_16::ZERO;
        for l in (bit_offset + radix_log)..m {
            if (outer_start >> l) & 1 != 0 {
                outer_off = outer_off.add(cantor[l]);
            }
        }

        // Sub-block evaluation points v_j = outer_off + ∑_a j_a · β_{off+a}.
        let mut v_points = vec![Gf2_16::ZERO; radix];
        for j in 0..radix {
            let mut v = outer_off;
            for a in 0..radix_log {
                if (j >> a) & 1 != 0 {
                    v = v.add(betas[a]);
                }
            }
            v_points[j] = v;
        }

        // s_evals[a][j] = s_{off+a}(v_j).
        let mut s_evals = vec![vec![Gf2_16::ZERO; radix]; radix_log];
        for a in 0..radix_log {
            let s_a = &base.subspace_polys[bit_offset + a];
            for j in 0..radix {
                s_evals[a][j] = evaluate_subspace_poly_at_gf16(s_a, v_points[j]);
            }
        }

        // matrix[j * radix + i] = ∏_{a ∈ bits(i)} s_{off+a}(v_j).
        let mut matrix = vec![Gf2_16::ZERO; radix * radix];
        for j in 0..radix {
            for i in 0..radix {
                let mut x_i = Gf2_16::ONE;
                for a in 0..radix_log {
                    if (i >> a) & 1 != 0 {
                        x_i = x_i.mul(s_evals[a][j]);
                    }
                }
                matrix[j * radix + i] = x_i;
            }
        }

        out.push(matrix);
    }

    out
}

/// Apply the mixed-radix additive FFT in place.
///
/// Stages run from top (highest `bit_offset`) down to the base, matching
/// the radix-2 stage ordering. For each stage, every outer block is an
/// independent `2^radix_log × 2^radix_log` matvec carried out in
/// `Z[X]/f̃`.
pub fn additive_fft_mixed_radix_16<C: Config16, T: FftRingElement16>(
    data: &mut [T],
    params: &MixedRadixFftParams16<C>,
) {
    assert_eq!(
        data.len(),
        params.base.codeword_len,
        "additive_fft_mixed_radix_16: data length {} does not match codeword_len {}",
        data.len(),
        params.base.codeword_len
    );

    for stage in params.stages.iter().rev() {
        let radix = 1usize << stage.radix_log;
        let outer_block_size = 1usize << (stage.bit_offset + stage.radix_log);
        let stride = 1usize << stage.bit_offset;
        let matrices = &stage.vandermondes;

        cfg_chunks_mut!(data, outer_block_size)
            .enumerate()
            .for_each(|(block_idx, block_data)| {
                mixed_radix_stage_block::<T>(block_data, stride, radix, &matrices[block_idx]);
            });
    }
}

/// Apply one stage's `radix × radix` matvec to a single outer block.
#[inline]
fn mixed_radix_stage_block<T: FftRingElement16>(
    block_data: &mut [T],
    stride: usize,
    radix: usize,
    matrix: &[Gf2_16],
) {
    // Snapshot buffer for the `radix` inputs at a given offset `t`.
    let mut inputs: Vec<T> = (0..radix).map(|_| T::fft_zero()).collect();

    for t in 0..stride {
        for i in 0..radix {
            inputs[i] = block_data[i * stride + t].clone();
        }

        // outputs[j] = Σ_i matrix[j][i] · inputs[i] in Z[X]/f̃.
        for j in 0..radix {
            let mut acc = T::fft_unreduced_zero();
            for i in 0..radix {
                let tw = matrix[j * radix + i];
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
    use super::super::params::{AddFftConfigGF2_16, compute_vandermonde_matrix_gf16};
    use super::*;

    type P = MixedRadixFftParams16<AddFftConfigGF2_16>;

    #[test]
    fn decompose_pure_radix8() {
        assert_eq!(decompose_radices(0).unwrap(), Vec::<usize>::new());
        assert_eq!(decompose_radices(3).unwrap(), vec![3]);
        assert_eq!(decompose_radices(6).unwrap(), vec![3, 3]);
        assert_eq!(decompose_radices(12).unwrap(), vec![3, 3, 3, 3]);
    }

    #[test]
    fn decompose_one_radix16_base() {
        // m % 3 == 1: one radix-16 stage at the base.
        assert_eq!(decompose_radices(4).unwrap(), vec![4]);
        assert_eq!(decompose_radices(7).unwrap(), vec![4, 3]);
        assert_eq!(decompose_radices(13).unwrap(), vec![4, 3, 3, 3]);
        assert_eq!(decompose_radices(16).unwrap(), vec![4, 3, 3, 3, 3]);
    }

    #[test]
    fn decompose_two_radix16_base() {
        // m % 3 == 2: two radix-16 stages at the base.
        assert_eq!(decompose_radices(8).unwrap(), vec![4, 4]);
        assert_eq!(decompose_radices(11).unwrap(), vec![4, 4, 3]);
        assert_eq!(decompose_radices(14).unwrap(), vec![4, 4, 3, 3]);
    }

    #[test]
    fn decompose_rejects_unrepresentable() {
        for m in [1usize, 2, 5] {
            assert!(decompose_radices(m).is_err(), "m={m} should be rejected");
        }
    }

    #[test]
    fn params_reject_unrepresentable_codeword_len() {
        // m = 1, 2, 5 → codeword_len 2, 4, 32.
        assert!(P::new(2, 2).is_err());
        assert!(P::new(2, 4).is_err());
        assert!(P::new(2, 32).is_err());
    }

    #[test]
    fn params_accept_representable_codeword_len() {
        assert!(P::new(8, 8).is_ok()); // m = 3
        assert!(P::new(16, 16).is_ok()); // m = 4 (one radix-16)
        assert!(P::new(8, 256).is_ok()); // m = 8 (two radix-16)
        assert!(P::new(8, 4096).is_ok()); // m = 12 (pure radix-8)
        assert!(P::new(8, 8192).is_ok()); // m = 13 (radix-16 + radix-8)
    }

    /// The mixed-radix FFT computes the multipoint evaluation of the
    /// novel-basis polynomial: `data[k] = Σ_i input[i] · X_i(v_k)`. We
    /// check this against the explicit Vandermonde, for sizes that
    /// exercise pure radix-8, one radix-16, and two radix-16 stages.
    #[test]
    fn fft_matches_direct_evaluation() {
        for &m in &[3usize, 4, 6, 7, 8] {
            let codeword_len = 1usize << m;
            let params = P::new(codeword_len, codeword_len).expect("valid params");

            // Deterministic pseudo-random GF(2^16) input.
            let input: Vec<Gf2_16> = (0..codeword_len)
                .map(|i| Gf2_16(((i as u64).wrapping_mul(0x9E37_79B9) ^ 0xABCD) as u16))
                .collect();

            // Reference: cw_ref[k] = Σ_i input[i] · V[k][i].
            let vmat = compute_vandermonde_matrix_gf16(
                &params.base.eval_points,
                &params.base.subspace_polys,
                m,
            );
            let mut cw_ref = vec![Gf2_16::ZERO; codeword_len];
            for k in 0..codeword_len {
                let mut acc = Gf2_16::ZERO;
                for i in 0..codeword_len {
                    acc = acc.add(input[i].mul(vmat[k][i]));
                }
                cw_ref[k] = acc;
            }

            // Mixed-radix FFT.
            let mut data = input.clone();
            additive_fft_mixed_radix_16(&mut data, &params);

            assert_eq!(data, cw_ref, "mixed-radix FFT mismatch at m={m}");
        }
    }

    /// The lifted (`i64`) FFT reduced mod 2 must equal the `Gf2_16`
    /// FFT — the structural-lift invariant — across radix-16 stages.
    #[test]
    fn lifted_fft_reduces_to_gf16_fft() {
        use super::super::ring_ops::REDUCED_LEN_16;
        use zinc_poly::univariate::dense::DensePolynomial;

        for &m in &[4usize, 7, 8] {
            let codeword_len = 1usize << m;
            let params = P::new(codeword_len, codeword_len).expect("valid params");

            let input: Vec<Gf2_16> = (0..codeword_len)
                .map(|i| Gf2_16(((i as u64).wrapping_mul(0xC2B2_AE35) ^ 0x1357) as u16))
                .collect();

            // GF(2^16) FFT.
            let mut gf_data = input.clone();
            additive_fft_mixed_radix_16(&mut gf_data, &params);

            // Lifted i64 FFT, starting from the {0,1}-lift of the input.
            let mut lifted: Vec<DensePolynomial<i64, REDUCED_LEN_16>> = input
                .iter()
                .map(|&g| {
                    let mut coeffs = [0i64; REDUCED_LEN_16];
                    for j in 0..REDUCED_LEN_16 {
                        coeffs[j] = i64::from(g.coeff(j));
                    }
                    DensePolynomial { coeffs }
                })
                .collect();
            additive_fft_mixed_radix_16(&mut lifted, &params);

            for k in 0..codeword_len {
                let mut bits = 0u16;
                for j in 0..REDUCED_LEN_16 {
                    if lifted[k].coeffs[j] & 1 != 0 {
                        bits |= 1u16 << j;
                    }
                }
                assert_eq!(
                    Gf2_16(bits),
                    gf_data[k],
                    "lifted FFT mod 2 mismatch at m={m}, k={k}"
                );
            }
        }
    }
}
