pub mod f2_lin_comb;
pub mod iprs;
pub mod raa;
pub mod raa_f2;
pub mod raa_sign_flip;

use crate::pcs::structs::ZipTypes;
use crypto_primitives::{DenseRowMatrix, FromPrimitiveWithConfig};
use num_traits::Zero;
use std::fmt::Debug;
use std::mem::MaybeUninit;
use zinc_poly::univariate::binary_f2_wide::BinaryF2Poly;
use zinc_poly::univariate::binary_gf128::GF128Poly;
use zinc_utils::{cfg_chunks, cfg_chunks_mut, from_ref::FromRef};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Extension trait providing an `F_2[X]`-linear encoder over arbitrary
/// `BinaryF2Poly<W>` widths. Used by the F_2[X] MLE-opening protocol's
/// proximity check (see `protocol/src/f2_open_plan.md`), which encodes
/// a width-4 combined row and checks consistency against committed
/// codeword cells at sampled columns.
///
/// The encoder must agree with `LinearCode::encode` on
/// `Cw = BinaryPoly<D>` rows when those rows are lifted to
/// `BinaryF2Poly<W>` — i.e., it implements the *same* linear map,
/// just over the wider F_2[X] cell type. The natural fit is
/// `RaaF2Code`, whose Repeat / Permute / XOR-Accumulate kernel is
/// uniformly F_2[X]-linear.
pub trait F2LinearOpener {
    fn encode_f2_lin_open<const W: usize>(
        &self,
        row: &[BinaryF2Poly<W>],
    ) -> Vec<BinaryF2Poly<W>>;

    /// `GF(2^128)`-coefficient variant for the **un-lifted** open: the same
    /// `F_2`-linear Repeat/Permute/XOR-Accumulate map, applied per coefficient
    /// of a `GF128Poly<D>` row (reusing the generic `encode_f2_lin` kernel — the
    /// encoder is `F_2`-linear, hence `GF(2^128)`-linear per coefficient). Used
    /// by the un-lifted open's combined-row proximity, where the eq-tensor stays
    /// in `GF(2^128)` so the combined row carries `D` `GF(2^128)` coefficients
    /// (the bit-slice evals) instead of a lifted wide `F_2[X]` poly.
    fn encode_gf128_lin_open<const D: usize>(
        &self,
        row: &[GF128Poly<D>],
    ) -> Vec<GF128Poly<D>>;
}

pub trait LinearCode<Zt: ZipTypes>: Debug + Clone + Eq + Sync + Send {
    /// Repetition factor, a.k.a. inverse rate, the ratio of codeword length to
    /// input row length. Has to be at a power of 2.
    ///
    /// Note: Ideally, this should be a generic constant, but due to the fact
    /// that generic parameters may not be used in const operations, this
    /// makes using it too much of a hassle.
    const REPETITION_FACTOR: usize;

    /// Length of each input row before encoding
    fn row_len(&self) -> usize;

    /// Length of each encoded codeword (output length after encoding)
    fn codeword_len(&self) -> usize;

    /// String representation of the parameters of this linear code, used for
    /// benchmarks. Should start with "row_len=X".
    fn params_string(&self) -> String;

    /// Encodes a row of cryptographic integers using this linear encoding
    /// scheme.
    ///
    /// This function is optimized for the prover's context where we work with
    /// cryptographic integers. It's more efficient than `encode_f` as it
    /// avoids field conversions.
    ///
    /// # Parameters
    /// - `row`: Slice of cryptographic integers to encode
    ///
    /// # Returns
    /// A vector of cryptographic integers representing the encoded row
    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw>;

    /// Encodes a row of cryptographic integers using this linear encoding
    /// scheme.
    ///
    /// This function is optimized for the prover's context where we work with
    /// cryptographic integers. It's more efficient than `encode_f` as it
    /// avoids field conversions.
    ///
    /// # Parameters
    /// - `row`: Slice of cryptographic integers to encode
    ///
    /// # Returns
    /// A vector of cryptographic integers representing the encoded row
    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR>;

    /// Encodes a row of field elements using this linear encoding scheme.
    ///
    /// This function is used when working with field elements directly and
    /// performs the encoding by first converting the sparse matrices to
    /// field elements.
    ///
    /// # Parameters
    /// - `row`: Slice of field elements to encode
    /// - `field`: Field configuration for the conversion
    ///
    /// # Returns
    /// A vector of field elements representing the encoded row
    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F>;

    /// Encode `row` directly into the (uninitialised) destination slot
    /// `dst`, optionally reusing `scratch` as intermediate storage. Lets
    /// the caller pre-allocate the destination row of the codeword matrix
    /// and a thread-local scratch buffer once per task block, avoiding
    /// the per-row `Vec` allocation pattern of [`Self::encode`].
    ///
    /// `dst.len()` must equal `self.codeword_len()`. `scratch.len()` must
    /// be at least `self.codeword_len()`; its initial contents are not
    /// observed. On return every slot of `dst` is initialised.
    ///
    /// The default implementation calls [`Self::encode`] and writes the
    /// returned `Vec` slot-by-slot. Codes with an efficient in-place kernel
    /// (e.g. [`super::raa_f2::RaaF2Code`]) should override to drop the
    /// intermediate `Vec`.
    fn encode_into_with_scratch(
        &self,
        row: &[Zt::Eval],
        _scratch: &mut [Zt::Cw],
        dst: &mut [MaybeUninit<Zt::Cw>],
    ) {
        let encoded = self.encode(row);
        debug_assert_eq!(dst.len(), encoded.len());
        for (slot, val) in dst.iter_mut().zip(encoded.into_iter()) {
            slot.write(val);
        }
    }

    /// Batched encoder: encode `num_rows` consecutive rows from `evals`
    /// into the (uninitialised) destination matrix `out`. Lets codes
    /// with a cross-row-vectorisable kernel (e.g.
    /// [`super::raa_f2::RaaF2Code`]) override with a packed
    /// implementation; the default loops over rows with per-task
    /// scratch reuse.
    ///
    /// `evals.len()` must equal `num_rows * self.row_len()`. `out` must
    /// be sized `num_rows × self.codeword_len()`. On return every slot
    /// of `out.data` is initialised.
    fn encode_rows_batched(
        &self,
        evals: &[Zt::Eval],
        num_rows: usize,
        out: &mut DenseRowMatrix<MaybeUninit<Zt::Cw>>,
    ) {
        let row_len = self.row_len();
        let codeword_len = self.codeword_len();
        debug_assert_eq!(evals.len(), num_rows * row_len);
        debug_assert_eq!(out.data.len(), num_rows * codeword_len);

        // Chunk row-encodes per task so each rayon task amortises (a) its
        // scratch buffer over many encodes and (b) the per-task scheduling
        // overhead.
        const ROWS_PER_TASK: usize = 64;
        let task_rows = ROWS_PER_TASK.min(num_rows.max(1));
        let dst_block_stride = codeword_len * task_rows;
        let src_block_stride = row_len * task_rows;

        cfg_chunks_mut!(out.data, dst_block_stride)
            .zip(cfg_chunks!(evals, src_block_stride))
            .for_each(|(dst_block, src_block)| {
                let mut scratch: Vec<Zt::Cw> = vec![Zt::Cw::zero(); codeword_len];
                for (dst_row, src_row) in dst_block
                    .chunks_mut(codeword_len)
                    .zip(src_block.chunks(row_len))
                {
                    self.encode_into_with_scratch(src_row, &mut scratch, dst_row);
                }
            });
    }
}
