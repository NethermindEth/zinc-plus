//! Mixed-radix additive Reed-Solomon FFT over `Z[X] / f̃` (D = 16).
//!
//! Generalises the radix-8 evaluator: the `m`-bit FFT
//! (`m = log2(codeword_len)`) is decomposed into a sequence of stages,
//! each consuming either three bits (a radix-8, 8×8 Vandermonde matvec)
//! or two bits (a radix-4, 4×4 matvec). Radix-4 stages are placed
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
//! Pure radix-8 (`m % 3 == 0`) is the special case with no radix-4
//! stages, and is bit-for-bit identical to the historic radix-8 kernel.
//! Allowing one or two radix-4 base stages lifts the `m % 3 == 0`
//! constraint: an `m`-bit FFT is feasible whenever `m = 3a + 2b` for
//! non-negative `a, b`, i.e. for every `m` except `m = 1`.
//!
//! Radix-4 (not radix-16) fills the gap because it is the cheapest
//! radix per bit: a radix-`r` matvec costs `r / log2(r)` twiddle
//! multiplications per output per bit — `2` for radix-4, `2.67` for
//! radix-8, `4` for radix-16. Two radix-4 stages cover the same four
//! bits as one radix-16 stage at half the multiplications, with
//! essentially the same lifted-coefficient growth.

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
    /// `log2` of this stage's radix: `2` (radix-4) or `3` (radix-8);
    /// `1` (radix-2) is also supported via [`MixedRadixFftParams16::new_with_radix_logs`].
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
/// that `m = log2(codeword_len)` is expressible as `3a + 2b` with
/// non-negative `a, b` (every `m` except `m = 1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedRadixFftParams16<C: Config16> {
    /// Underlying radix-2 parameters (Cantor basis, subspace polys,
    /// eval points, sizes).
    pub base: Radix2AddFftParams16<C>,

    /// Stages, ordered base → top (ascending `bit_offset`).
    pub stages: Vec<RadixStage>,

    _phantom: PhantomData<C>,
}

/// Decompose an `m`-bit FFT into stage radix-logs, base → top.
///
/// Radix-8 (`log` = `3`) is the workhorse — fewest stages, so the
/// smallest lifted-coefficient growth. Radix-4 (`log` = `2`) fills the
/// gap when `m` is not a multiple of `3`, and the radix-4 stages sit at
/// the base. `m = 2·(num4) + 3·(num8)`; only `m = 1` is infeasible.
pub fn decompose_radices(m: usize) -> Result<Vec<usize>, ZipError> {
    let (num4, num8) = match m % 3 {
        0 => (0usize, m / 3),
        1 if m >= 4 => (2usize, (m - 4) / 3),
        2 => (1usize, (m - 2) / 3),
        _ => {
            return Err(ZipError::InvalidPcsParam(format!(
                "MixedRadixFftParams16: log2(codeword_len) ({m}) must be ≥ 2 \
                 (only m = 1 is unsupported)"
            )));
        }
    };
    let mut radices = vec![2usize; num4];
    radices.extend(std::iter::repeat(3usize).take(num8));
    Ok(radices)
}

impl<C: Config16> MixedRadixFftParams16<C> {
    /// Construct precomputed parameters with the shipped radix
    /// decomposition ([`decompose_radices`]: radix-8 workhorse, radix-4
    /// filler). Validates the radix-2 preconditions plus
    /// `m = log2(codeword_len) ≥ 2`.
    pub fn new(row_len: usize, codeword_len: usize) -> Result<Self, ZipError> {
        let base = Radix2AddFftParams16::<C>::new(row_len, codeword_len)?;
        let radix_logs = decompose_radices(base.log2_codeword_len)?;
        Ok(Self::from_base_and_radices(base, radix_logs))
    }

    /// Construct with an explicit per-stage radix decomposition:
    /// `radix_logs[k]` is `log2` of stage `k`'s radix, ordered base →
    /// top, and the entries must sum to `log2(codeword_len)`. Intended
    /// for experiments and coefficient-size measurement; [`Self::new`]
    /// picks the shipped decomposition.
    pub fn new_with_radix_logs(
        row_len: usize,
        codeword_len: usize,
        radix_logs: Vec<usize>,
    ) -> Result<Self, ZipError> {
        let base = Radix2AddFftParams16::<C>::new(row_len, codeword_len)?;
        let sum: usize = radix_logs.iter().sum();
        if sum != base.log2_codeword_len {
            return Err(ZipError::InvalidPcsParam(format!(
                "MixedRadixFftParams16: radix_logs sum to {sum}, expected \
                 log2(codeword_len) = {}",
                base.log2_codeword_len
            )));
        }
        Ok(Self::from_base_and_radices(base, radix_logs))
    }

    /// Build the precomputed stages from a base and a base→top list of
    /// radix-logs (assumed to sum to `base.log2_codeword_len`).
    fn from_base_and_radices(base: Radix2AddFftParams16<C>, radix_logs: Vec<usize>) -> Self {
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
        debug_assert_eq!(bit_offset, base.log2_codeword_len);

        Self {
            base,
            stages,
            _phantom: PhantomData,
        }
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

    /// Count of stages by radix: `(radix-4 count, radix-8 count)`.
    #[inline]
    pub fn radix_stage_counts(&self) -> (usize, usize) {
        let r4 = self.stages.iter().filter(|s| s.radix_log == 2).count();
        let r8 = self.stages.iter().filter(|s| s.radix_log == 3).count();
        (r4, r8)
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

/// Apply the mixed-radix additive FFT in place over a fully-dense
/// input. Thin wrapper over [`additive_fft_mixed_radix_16_padded`].
pub fn additive_fft_mixed_radix_16<C: Config16, T: FftRingElement16>(
    data: &mut [T],
    params: &MixedRadixFftParams16<C>,
) {
    let len = data.len();
    additive_fft_mixed_radix_16_padded(data, params, len);
}

/// Apply the mixed-radix additive FFT in place, exploiting zero
/// padding. `num_nonzero` is the count of nonzero leading cells of
/// `data`; the rest are zero — the Reed-Solomon rate-`1/REP` padding
/// added by `encode` / `encode_wide`.
///
/// Stages run from top (highest `bit_offset`) down to the base. The
/// top stage runs first and is the only one that sees the zero
/// padding: it spans the whole array as a single outer block, and its
/// later sub-blocks lie entirely in `[num_nonzero, len)`. So its
/// `radix × radix` matvec sums only the `⌈num_nonzero / stride⌉`
/// non-zero input sub-blocks instead of all `radix` — a `1 − 1/REP`
/// saving on that stage. Every later stage operates on already-dense
/// data and processes all `radix` inputs.
pub fn additive_fft_mixed_radix_16_padded<C: Config16, T: FftRingElement16>(
    data: &mut [T],
    params: &MixedRadixFftParams16<C>,
    num_nonzero: usize,
) {
    assert_eq!(
        data.len(),
        params.base.codeword_len,
        "additive_fft_mixed_radix_16: data length {} does not match codeword_len {}",
        data.len(),
        params.base.codeword_len
    );

    let mut is_first_stage = true;
    for stage in params.stages.iter().rev() {
        let radix = 1usize << stage.radix_log;
        let outer_block_size = 1usize << (stage.bit_offset + stage.radix_log);
        let stride = 1usize << stage.bit_offset;
        let matrices = &stage.vandermondes;

        // The first stage processed is the top stage — a single outer
        // block spanning all of `data`. Sub-blocks lying entirely in
        // the zero-padding region `[num_nonzero, len)` contribute
        // nothing to the matvec and are skipped. Skipping a zero input
        // is exact: `twiddle · 0 = 0`. Later stages see dense data.
        let nonzero_inputs = if is_first_stage {
            debug_assert_eq!(outer_block_size, data.len());
            num_nonzero.div_ceil(stride).min(radix)
        } else {
            radix
        };
        is_first_stage = false;

        cfg_chunks_mut!(data, outer_block_size)
            .enumerate()
            .for_each(|(block_idx, block_data)| {
                mixed_radix_stage_block::<T>(
                    block_data,
                    stride,
                    radix,
                    &matrices[block_idx],
                    nonzero_inputs,
                );
            });
    }
}

/// Apply one stage's `radix × radix` matvec to a single outer block.
///
/// Only the first `nonzero_inputs` (`≤ radix`) input sub-blocks are
/// read and fed into the matvec; the rest are known-zero (the FFT's
/// zero-padding, on the top stage) and contribute nothing. All `radix`
/// outputs are still produced. For a dense stage `nonzero_inputs ==
/// radix`.
#[inline]
fn mixed_radix_stage_block<T: FftRingElement16>(
    block_data: &mut [T],
    stride: usize,
    radix: usize,
    matrix: &[Gf2_16],
    nonzero_inputs: usize,
) {
    debug_assert!(radix <= 8, "mixed-radix kernel supports radix ≤ 8");
    debug_assert!(nonzero_inputs <= radix);
    // Stack-allocated snapshot of the input sub-blocks at offset `t` —
    // radix is at most 8 (radix-8 is the largest shipped stage), so
    // this avoids a per-block heap allocation.
    let mut inputs: [T; 8] = std::array::from_fn(|_| T::fft_zero());

    for t in 0..stride {
        for i in 0..nonzero_inputs {
            inputs[i] = block_data[i * stride + t].clone();
        }

        // outputs[j] = Σ_i matrix[j][i] · inputs[i] in Z[X]/f̃.
        //
        // Iterate `i` outermost: each input is touched once per `j`, so
        // keeping it in the outer loop lets its 16 coefficients stay
        // resident while it is scattered into all `radix` accumulators.
        let mut accs: [T::UnreducedAcc; 8] =
            std::array::from_fn(|_| T::fft_unreduced_zero());
        for i in 0..nonzero_inputs {
            let input_i = &inputs[i];
            for j in 0..radix {
                let tw = matrix[j * radix + i];
                if tw == Gf2_16::ZERO {
                    continue;
                }
                T::fft_acc_mul_lifted_gf16(input_i, tw, &mut accs[j]);
            }
        }
        for j in 0..radix {
            block_data[j * stride + t] = T::fft_finalize_acc(accs[j].clone());
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
        // m % 3 == 0: no radix-4 stages.
        assert_eq!(decompose_radices(0).unwrap(), Vec::<usize>::new());
        assert_eq!(decompose_radices(3).unwrap(), vec![3]);
        assert_eq!(decompose_radices(6).unwrap(), vec![3, 3]);
        assert_eq!(decompose_radices(12).unwrap(), vec![3, 3, 3, 3]);
    }

    #[test]
    fn decompose_two_radix4_base() {
        // m % 3 == 1: two radix-4 stages at the base.
        assert_eq!(decompose_radices(4).unwrap(), vec![2, 2]);
        assert_eq!(decompose_radices(7).unwrap(), vec![2, 2, 3]);
        assert_eq!(decompose_radices(13).unwrap(), vec![2, 2, 3, 3, 3]);
        assert_eq!(decompose_radices(16).unwrap(), vec![2, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn decompose_one_radix4_base() {
        // m % 3 == 2: one radix-4 stage at the base.
        assert_eq!(decompose_radices(2).unwrap(), vec![2]);
        assert_eq!(decompose_radices(5).unwrap(), vec![2, 3]);
        assert_eq!(decompose_radices(8).unwrap(), vec![2, 3, 3]);
        assert_eq!(decompose_radices(11).unwrap(), vec![2, 3, 3, 3]);
        assert_eq!(decompose_radices(14).unwrap(), vec![2, 3, 3, 3, 3]);
    }

    #[test]
    fn decompose_rejects_only_m_equals_one() {
        assert!(decompose_radices(1).is_err(), "m=1 should be rejected");
        for m in [0usize, 2, 3, 4, 5, 6, 7, 8] {
            assert!(decompose_radices(m).is_ok(), "m={m} should be accepted");
        }
    }

    #[test]
    fn params_reject_unrepresentable_codeword_len() {
        // Only m = 1 (codeword_len 2) is unsupported.
        assert!(P::new(2, 2).is_err());
    }

    #[test]
    fn params_accept_representable_codeword_len() {
        assert!(P::new(2, 4).is_ok()); // m = 2 (one radix-4)
        assert!(P::new(8, 8).is_ok()); // m = 3 (pure radix-8)
        assert!(P::new(16, 16).is_ok()); // m = 4 (two radix-4)
        assert!(P::new(8, 32).is_ok()); // m = 5 (radix-4 + radix-8)
        assert!(P::new(8, 4096).is_ok()); // m = 12 (pure radix-8)
        assert!(P::new(8, 8192).is_ok()); // m = 13 (two radix-4 + radix-8)
    }

    #[test]
    fn new_with_radix_logs_rejects_wrong_sum() {
        // m = 12, but radix_logs sum to 11.
        assert!(P::new_with_radix_logs(8, 4096, vec![3, 3, 3, 2]).is_err());
        // Correct sum is accepted.
        assert!(P::new_with_radix_logs(8, 4096, vec![2, 2, 2, 2, 2, 2]).is_ok());
    }

    /// The mixed-radix FFT computes the multipoint evaluation of the
    /// novel-basis polynomial: `data[k] = Σ_i input[i] · X_i(v_k)`. We
    /// check this against the explicit Vandermonde, for sizes that
    /// exercise pure radix-8, one radix-4, and two radix-4 stages.
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
    /// FFT — the structural-lift invariant — across radix-4 stages.
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

    /// Measure the empirical max |codeword coefficient| of the lifted
    /// (`i64`) FFT for different radix decompositions, on a real encode
    /// (`row_len` random binary cells, zero-padded to `codeword_len`).
    /// Lower-radix decompositions use more, shallower stages — fewer
    /// multiplications, but more carry-accumulating depth, so larger
    /// coefficients. This pins down whether an all-radix-4 encoder
    /// still fits `BITS_CW = 32`.
    #[test]
    fn codeword_coef_bits_by_radix() {
        use super::super::ring_ops::REDUCED_LEN_16;
        use zinc_poly::univariate::dense::DensePolynomial;

        // (label, row_len, codeword_len, radix_logs).
        let configs: [(&str, usize, usize, Vec<usize>); 5] = [
            ("m=12 radix-8 (shipped 1/4)", 1024, 4096, vec![3, 3, 3, 3]),
            ("m=12 radix-4 only", 1024, 4096, vec![2, 2, 2, 2, 2, 2]),
            ("m=13 radix-4+radix-8 (shipped 1/8)", 1024, 8192, vec![2, 2, 3, 3, 3]),
            ("m=13 radix-2+radix-4", 1024, 8192, vec![1, 2, 2, 2, 2, 2, 2]),
            ("m=16 radix-4+radix-8 (nvars=13 rate 1/4)", 16384, 65536, vec![2, 2, 3, 3, 3, 3]),
        ];

        for (label, row_len, codeword_len, radix_logs) in configs {
            let params = MixedRadixFftParams16::<AddFftConfigGF2_16>::new_with_radix_logs(
                row_len,
                codeword_len,
                radix_logs,
            )
            .expect("valid radix decomposition");

            // row_len random binary cells (bit j of a pseudo-random
            // u16 → coefficient j), zero-padded to codeword_len.
            let mut data: Vec<DensePolynomial<i64, REDUCED_LEN_16>> = (0..codeword_len)
                .map(|i| {
                    let mut coeffs = [0i64; REDUCED_LEN_16];
                    if i < row_len {
                        let g = (i as u64)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add(0xABCD) as u16;
                        for (j, c) in coeffs.iter_mut().enumerate() {
                            *c = ((g >> j) & 1) as i64;
                        }
                    }
                    DensePolynomial { coeffs }
                })
                .collect();

            additive_fft_mixed_radix_16(&mut data, &params);

            let mut max_abs: i64 = 0;
            for cell in &data {
                for &c in &cell.coeffs {
                    max_abs = max_abs.max(c.unsigned_abs() as i64);
                }
            }
            let bits = if max_abs == 0 {
                0
            } else {
                65 - (max_abs as u64).leading_zeros() as u64
            };
            eprintln!(
                "  {label}: max |coef| = {max_abs} (~2^{:.2}), {bits} bits incl. sign{}",
                if max_abs == 0 { 0.0 } else { (max_abs as f64).log2() },
                if bits <= 32 { "  [fits BITS_CW=32]" } else { "  [EXCEEDS 32]" },
            );
        }
    }
}
