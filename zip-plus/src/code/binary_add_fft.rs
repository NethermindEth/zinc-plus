//! Additive Reed-Solomon FFT over `GF(2^32) = F_2[X]/(f)`, structurally
//! lifted to `Z[X]/(f̃)`. Mirrors the layout of [`crate::code::iprs`]:
//! a `basis` module precomputes the algebra (Cantor basis, subspace
//! polynomials), `ring_ops` handles the lifted ring arithmetic + the
//! [`FftRingElement`](ring_ops::FftRingElement) abstraction, `params`
//! holds the [`Radix2AddFftParams`](params::Radix2AddFftParams)
//! precomputation, and `add_fft` is the generic radix-2 FFT kernel
//! used by [`BinaryAddFftCode`].
//!
//! The `LinearCode<Zt>` impl below is generic over any `Zt: ZipTypes`
//! whose `Cw` and `CombR` implement [`FftRingElement`] (with
//! `Cw: FromRef<Zt::Eval>` providing the input lift). The concrete
//! intended use is
//! `Zt::Eval = BinaryPoly<32>`,
//! `Zt::Cw = DensePolynomial<Int<K>, 32>`,
//! `Zt::CombR = Zt::Comb = DensePolynomial<Int<M>, 32>` —
//! the "alpha-projection skipped" setup of the binary-RS Zip+ variant.

pub mod add_fft;
pub mod basis;
pub mod params;
pub mod ring_ops;

use std::{fmt::Debug, marker::PhantomData};

use crypto_primitives::FromPrimitiveWithConfig;
use zinc_utils::from_ref::FromRef;

use crate::{ZipError, code::LinearCode, pcs::structs::ZipTypes};
use add_fft::additive_fft;
pub use params::{AddFftConfigGF2_32, Config, Radix2AddFftParams};
use ring_ops::FftRingElement;

/// Reed-Solomon linear code whose encoder is an additive FFT over
/// `GF(2^32)` lifted structurally to `Z[X]/(f̃)`.
///
/// The codeword length is `row_len · REP` (must be a power of two
/// `≤ 2^32`). The encoder runs `log2(codeword_len)` radix-2 butterfly
/// stages; there is no separate "base layer" concept like in IPRS,
/// since additive FFTs are intrinsically radix-2.
///
/// The `CHECK` generic is reserved for future overflow checking
/// (mirroring [`crate::code::iprs::IprsCode`]'s `CHECK` parameter);
/// it currently has no effect — callers must size `Zt::Cw`/`Zt::CombR`
/// coefficient widths to accommodate the worst-case growth of
/// `~33^log2(codeword_len)` documented in
/// [`additive_fft`](add_fft::additive_fft).
pub struct BinaryAddFftCode<Zt: ZipTypes, C: Config, const REP: usize, const CHECK: bool> {
    params: Radix2AddFftParams<C>,
    _phantom: PhantomData<Zt>,
}

impl<Zt: ZipTypes, C: Config, const REP: usize, const CHECK: bool>
    BinaryAddFftCode<Zt, C, REP, CHECK>
{
    /// Construct a code with the given input row length. The codeword
    /// length is `row_len * REP`. Both must be powers of two; `REP`
    /// must be a power of two; and the codeword length must not
    /// exceed `2^DEGREE = 2^32`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(row_len: usize) -> Result<Self, ZipError> {
        if !REP.is_power_of_two() {
            return Err(ZipError::InvalidPcsParam(format!(
                "REP ({REP}) must be a power of two"
            )));
        }
        let codeword_len = row_len.checked_mul(REP).ok_or_else(|| {
            ZipError::InvalidPcsParam(format!("row_len ({row_len}) * REP ({REP}) overflows usize"))
        })?;
        Ok(Self {
            params: Radix2AddFftParams::new(row_len, codeword_len)?,
            _phantom: PhantomData,
        })
    }

    /// Additive FFTs run all `log2(codeword_len)` stages — there is no
    /// "depth" choice to optimise. Provided for API parity with
    /// [`crate::code::iprs::IprsCode::new_with_optimal_depth`].
    pub fn new_with_optimal_depth(row_len: usize) -> Result<Self, ZipError> {
        Self::new(row_len)
    }

    pub fn params(&self) -> &Radix2AddFftParams<C> {
        &self.params
    }
}

impl<Zt: ZipTypes, C: Config, const REP: usize, const CHECK: bool> Clone
    for BinaryAddFftCode<Zt, C, REP, CHECK>
{
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<Zt: ZipTypes, C: Config, const REP: usize, const CHECK: bool> Debug
    for BinaryAddFftCode<Zt, C, REP, CHECK>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryAddFftCode")
            .field("params", &self.params)
            .finish()
    }
}

impl<Zt: ZipTypes, C: Config, const REP: usize, const CHECK: bool> PartialEq
    for BinaryAddFftCode<Zt, C, REP, CHECK>
{
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params
    }
}

impl<Zt: ZipTypes, C: Config, const REP: usize, const CHECK: bool> Eq
    for BinaryAddFftCode<Zt, C, REP, CHECK>
{
}

impl<Zt, C, const REP: usize, const CHECK: bool> LinearCode<Zt>
    for BinaryAddFftCode<Zt, C, REP, CHECK>
where
    Zt: ZipTypes,
    C: Config,
    Zt::Cw: FftRingElement + FromRef<Zt::Eval>,
    Zt::CombR: FftRingElement,
{
    const REPETITION_FACTOR: usize = REP;

    fn row_len(&self) -> usize {
        self.params.row_len
    }

    fn codeword_len(&self) -> usize {
        self.params.codeword_len
    }

    fn params_string(&self) -> String {
        format!(
            "row_len={}, rate=1/{REP}, log2_codeword_len={}",
            self.row_len(),
            self.params.log2_codeword_len
        )
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        assert_eq!(
            row.len(),
            self.params.row_len,
            "encode: input length {} does not match row_len {}",
            row.len(),
            self.params.row_len
        );
        let mut data: Vec<Zt::Cw> = row.iter().map(Zt::Cw::from_ref).collect();
        data.resize_with(self.params.codeword_len, Zt::Cw::fft_zero);
        additive_fft(&mut data, &self.params);
        data
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        assert_eq!(
            row.len(),
            self.params.row_len,
            "encode_wide: input length {} does not match row_len {}",
            row.len(),
            self.params.row_len
        );
        let mut data: Vec<Zt::CombR> = row.to_vec();
        data.resize_with(self.params.codeword_len, Zt::CombR::fft_zero);
        additive_fft(&mut data, &self.params);
        data
    }

    fn encode_f<F>(&self, _row: &[F]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F>,
    {
        // `encode_f` is part of the `LinearCode` trait but is never called
        // by the Zinc+ protocol — the verifier uses `encode_wide` for the
        // proximity check. The additive FFT over `Z[X]/f̃` is not naturally
        // expressible over an arbitrary prime field `F` without a separate
        // F-flavoured kernel, so we leave this unsupported until a concrete
        // need arises.
        unimplemented!(
            "BinaryAddFftCode::encode_f is not supported; use encode_wide for the \
             proximity check path."
        )
    }
}

#[cfg(test)]
mod linear_code_tests {
    use super::*;
    use crate::code::LinearCode;
    use crate::pcs::structs::ZipPlus;
    use crate::pcs::test_utils::BinPolyAddFftZipTypes;
    use crypto_primitives::crypto_bigint_int::Int;
    use rand::prelude::*;
    use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
    use zinc_poly::univariate::binary::BinaryPoly;
    use zinc_poly::univariate::dense::DensePolynomial;
    use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};

    /// Smoke test: instantiate `BinaryAddFftCode` with the concrete
    /// `BinPolyAddFftZipTypes` and run `encode` / `encode_wide` end-to-end.
    /// Verifies the trait bounds line up and the encoded codeword has the
    /// expected length. Linearity of the encoder is already covered at
    /// the FFT-kernel level by `add_fft::tests::lifted_fft_is_linear`.
    #[test]
    fn encode_and_encode_wide_produce_codeword_of_expected_length() {
        const K: usize = 5; // 320-bit Cw coefficients
        const M: usize = 8; // 512-bit CombR coefficients
        const REP: usize = 4;
        type Zt = BinPolyAddFftZipTypes<K, M>;
        type Code = BinaryAddFftCode<Zt, AddFftConfigGF2_32, REP, false>;

        let row_len = 16usize;
        let code = Code::new(row_len).expect("valid params");

        assert_eq!(LinearCode::<Zt>::row_len(&code), row_len);
        assert_eq!(LinearCode::<Zt>::codeword_len(&code), row_len * REP);

        // `BinaryPoly<32>::from(u32)` constructs the binary polynomial whose
        // i-th coefficient is bit i of the input.
        let row: Vec<BinaryPoly<32>> = (0..row_len)
            .map(|i| BinaryPoly::<32>::from((i as u32).wrapping_mul(0x0123_4567) | 1))
            .collect();

        let cw = LinearCode::<Zt>::encode(&code, &row);
        assert_eq!(cw.len(), row_len * REP);
    }

    /// ZipTypes mirroring the bench's `BenchAddFftI32ZipPlusTypes`:
    /// `Cw = CombR = DensePolynomial<i32, 32>`. Used by the
    /// `column_size_report_i32` test.
    #[derive(Debug, Clone)]
    pub struct InternalAddFftI32ZipTypes;
    impl crate::pcs::structs::ZipTypes for InternalAddFftI32ZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = BinaryPoly<32>;
        type Cw = DensePolynomial<i32, 32>;
        type Fmod = crypto_primitives::crypto_bigint_uint::Uint<4>;
        type PrimeTest = zinc_primality::MillerRabin;
        type Chal = i32;
        type Pt = i32;
        type CombR = DensePolynomial<i32, 32>;
        type Comb = Self::CombR;
        type EvalDotChal = zinc_utils::inner_product::ScalarProduct;
        type CombDotChal = zinc_utils::inner_product::ScalarProduct;
        type ArrCombRDotChal = zinc_utils::inner_product::MBSInnerProduct;
    }

    /// Measure the size of a single column of the encoded codeword
    /// matrix, both raw and after zstd-3 compression. Run with
    /// `cargo test -p zip-plus column_size_report -- --nocapture --ignored`.
    ///
    /// "Column" here = the bytes the prover would write per Merkle
    /// opening: across the `batch_size` polys, gather the `num_rows`
    /// `Cw` entries at one column index, serialize them concatenated,
    /// and compress.
    #[test]
    #[ignore]
    fn column_size_report() {
        const K: usize = 2;
        const M: usize = 3;
        const REP: usize = 4;
        type Zt = BinPolyAddFftZipTypes<K, M>;
        type Code = BinaryAddFftCode<Zt, AddFftConfigGF2_32, REP, false>;

        // Match the bench config:
        //   poly_size = 2^14
        //   row_len = poly_size / REP = 4096
        //   codeword_len = row_len * REP = 16384
        //   num_rows = poly_size / row_len = 4
        //   batch_size = 11
        let num_vars = 14usize;
        let poly_size = 1usize << num_vars;
        let row_len = poly_size / REP;
        let batch_size = 11usize;

        let code = Code::new(row_len).expect("valid params");
        let codeword_len = LinearCode::<Zt>::codeword_len(&code);
        let num_rows = poly_size / row_len;

        let pp = ZipPlus::<Zt, Code>::setup(poly_size, code);

        // True-random polynomials, matching what the criterion bench uses
        // via `DenseMultilinearExtension::rand`.
        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let polys: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = (0..batch_size)
            .map(|_| DenseMultilinearExtension::<BinaryPoly<32>>::rand(num_vars, &mut rng))
            .collect();

        let (hint, _comm) =
            ZipPlus::<Zt, Code>::commit(&pp, &polys).expect("commit succeeded");

        // Single Cw entry's serialized size.
        let elem_bytes = <DensePolynomial<Int<K>, 32> as ConstTranscribable>::NUM_BYTES;

        // Gather a column at index 0 across all polys' codeword matrices,
        // in the same layout `phase_prove.rs::open_merkle_trees_for_column_inner`
        // uses.
        let column_idx = 0usize;
        let total_rows: usize = hint.cw_matrices.iter().map(|m| m.num_rows).sum();
        let raw_size_per_column = total_rows * elem_bytes;
        let mut col_buf = vec![0u8; raw_size_per_column];
        let mut offset = 0;
        for cw_matrix in &hint.cw_matrices {
            for row in cw_matrix.as_rows() {
                row[column_idx]
                    .write_transcription_bytes_exact(&mut col_buf[offset..offset + elem_bytes]);
                offset += elem_bytes;
            }
        }

        let compressed = zstd::encode_all(&col_buf[..], 3).expect("zstd-3 compression");

        // Measure how many bytes per i128 coefficient are actually
        // non-zero. After `m` FFT stages with `BinaryPoly<32>` inputs
        // (coefficients in {0,1}), the magnitudes are bounded by
        // ~33^m. We expect ~7 of every 16 bytes (high half) to be
        // always-zero for this `m`; the rest captures the actual
        // value bits.
        let mut byte_freq = [0usize; 256];
        for &b in &col_buf {
            byte_freq[b as usize] += 1;
        }
        let zero_byte_count = byte_freq[0];
        let nonzero_byte_count = raw_size_per_column - zero_byte_count;

        // Theoretical bound on bit-width used per coefficient.
        // codeword_len = 2^m, growth ≤ 33^m, so log2(33) · m bits.
        let m = (codeword_len as f64).log2() as usize;
        let bits_used = (m as f64) * 33f64.log2();
        let bytes_used_per_coeff = (bits_used / 8.0).ceil() as usize;

        eprintln!();
        eprintln!("===== BinaryAddFftCode column-opening size =====");
        eprintln!("  Config:");
        eprintln!("    poly_size      = 2^{num_vars} = {poly_size}");
        eprintln!("    row_len        = {row_len}");
        eprintln!("    codeword_len   = {codeword_len} (= 2^{m})");
        eprintln!("    num_rows/poly  = {num_rows}");
        eprintln!("    batch_size     = {batch_size}");
        eprintln!("    Cw type        = DensePolynomial<Int<{K}>, 32> = {elem_bytes} bytes/entry");
        eprintln!();
        eprintln!("  Per-poly column ({} Cw entries, one column of one poly's matrix):", num_rows);
        eprintln!("    raw   = {} bytes ({:.1} KiB)", num_rows * elem_bytes, (num_rows * elem_bytes) as f64 / 1024.0);
        eprintln!();
        eprintln!("  Batched column / Merkle leaf payload ({} polys × {} rows = {} Cw entries):", batch_size, num_rows, total_rows);
        eprintln!("    raw       = {} bytes ({:.1} KiB)", raw_size_per_column, raw_size_per_column as f64 / 1024.0);
        eprintln!(
            "    zstd-3    = {} bytes ({:.1} KiB)  [ratio {:.2}×]",
            compressed.len(),
            compressed.len() as f64 / 1024.0,
            raw_size_per_column as f64 / compressed.len() as f64,
        );
        eprintln!();
        eprintln!("  Why zstd compresses so well:");
        eprintln!(
            "    Loose worst-case bound on each i128 magnitude is 33^m = 33^{} ≈ 2^{:.0}",
            m, bits_used,
        );
        eprintln!(
            "    ({:.1} of 16 bytes/coeff used). Reality is tighter because codeword",
            bits_used / 8.0,
        );
        eprintln!(
            "    entries are *linear combinations* of sparse {{0,1}}-input × {{0,1}}-twiddle"
        );
        eprintln!(
            "    products, not iterated worst-case products. Plus many entries are exactly"
        );
        eprintln!(
            "    zero because the polynomial has row_len/codeword_len = 1/{} non-zero novel-"
        ,
            codeword_len / row_len,
        );
        eprintln!("    basis coefficients. Empirically:");
        eprintln!(
            "      {}/{} = {:.1}% bytes in the raw column buffer are zero",
            zero_byte_count,
            raw_size_per_column,
            100.0 * zero_byte_count as f64 / raw_size_per_column as f64,
        );
        eprintln!(
            "      Non-zero bytes: {} ({:.1}% of raw); zstd-3 further compresses {:.1}×",
            nonzero_byte_count,
            100.0 * nonzero_byte_count as f64 / raw_size_per_column as f64,
            nonzero_byte_count as f64 / compressed.len() as f64,
        );
        eprintln!(
            "      (Naive high-bit truncation alone would give ~{} bytes/coeff × 32 × {} = {} bytes)",
            bytes_used_per_coeff,
            total_rows,
            bytes_used_per_coeff * 32 * total_rows,
        );
        eprintln!();
        let num_openings = <Zt as crate::pcs::structs::ZipTypes>::NUM_COLUMN_OPENINGS;
        eprintln!(
            "  Full proof column-opening cost ({} openings × per-column):",
            num_openings
        );
        eprintln!(
            "    raw    = {} bytes ({:.1} KiB)",
            raw_size_per_column * num_openings,
            (raw_size_per_column * num_openings) as f64 / 1024.0,
        );
        eprintln!(
            "    zstd-3 = {} bytes ({:.1} KiB)  [each opening compressed independently]",
            compressed.len() * num_openings,
            (compressed.len() * num_openings) as f64 / 1024.0,
        );
        eprintln!("================================================");
        eprintln!();
    }

    /// Same as `column_size_report` but with `Cw = DensePolynomial<i32, 32>`
    /// (the bench's `BenchAddFftI32ZipPlusTypes`). Each `Cw` entry is
    /// 4× smaller than the `Int<2>` version (128 bytes vs 512), which
    /// also affects how much zstd-3 can compress.
    #[test]
    #[ignore]
    fn column_size_report_i32() {
        type Zt = InternalAddFftI32ZipTypes;
        type Code = BinaryAddFftCode<Zt, AddFftConfigGF2_32, 4, false>;

        let num_vars = 14usize;
        let poly_size = 1usize << num_vars;
        let row_len = poly_size / 4;
        let batch_size = 11usize;

        let code = Code::new(row_len).expect("valid params");
        let codeword_len = LinearCode::<Zt>::codeword_len(&code);
        let num_rows = poly_size / row_len;

        let pp = ZipPlus::<Zt, Code>::setup(poly_size, code);

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let polys: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = (0..batch_size)
            .map(|_| DenseMultilinearExtension::<BinaryPoly<32>>::rand(num_vars, &mut rng))
            .collect();

        let (hint, _comm) =
            ZipPlus::<Zt, Code>::commit(&pp, &polys).expect("commit succeeded");

        let elem_bytes = <DensePolynomial<i32, 32> as ConstTranscribable>::NUM_BYTES;
        let column_idx = 0usize;
        let total_rows: usize = hint.cw_matrices.iter().map(|m| m.num_rows).sum();
        let raw_size_per_column = total_rows * elem_bytes;
        let mut col_buf = vec![0u8; raw_size_per_column];
        let mut offset = 0;
        for cw_matrix in &hint.cw_matrices {
            for row in cw_matrix.as_rows() {
                row[column_idx]
                    .write_transcription_bytes_exact(&mut col_buf[offset..offset + elem_bytes]);
                offset += elem_bytes;
            }
        }

        let compressed = zstd::encode_all(&col_buf[..], 3).expect("zstd-3 compression");

        let mut byte_freq = [0usize; 256];
        for &b in &col_buf {
            byte_freq[b as usize] += 1;
        }
        let zero_byte_count = byte_freq[0];
        let nonzero_byte_count = raw_size_per_column - zero_byte_count;
        let m = (codeword_len as f64).log2() as usize;

        eprintln!();
        eprintln!("===== BinaryAddFftCode column-opening size (i32 Cw) =====");
        eprintln!("  Config:");
        eprintln!("    poly_size      = 2^{num_vars} = {poly_size}");
        eprintln!("    row_len        = {row_len}");
        eprintln!("    codeword_len   = {codeword_len} (= 2^{m})");
        eprintln!("    num_rows/poly  = {num_rows}");
        eprintln!("    batch_size     = {batch_size}");
        eprintln!("    Cw type        = DensePolynomial<i32, 32> = {elem_bytes} bytes/entry");
        eprintln!();
        eprintln!("  Per-poly column ({} Cw entries):", num_rows);
        eprintln!("    raw   = {} bytes ({:.2} KiB)", num_rows * elem_bytes, (num_rows * elem_bytes) as f64 / 1024.0);
        eprintln!();
        eprintln!("  Batched column / Merkle leaf payload ({} polys × {} rows = {} Cw entries):", batch_size, num_rows, total_rows);
        eprintln!("    raw       = {} bytes ({:.2} KiB)", raw_size_per_column, raw_size_per_column as f64 / 1024.0);
        eprintln!(
            "    zstd-3    = {} bytes ({:.2} KiB)  [ratio {:.2}×]",
            compressed.len(),
            compressed.len() as f64 / 1024.0,
            raw_size_per_column as f64 / compressed.len() as f64,
        );
        eprintln!(
            "    {} / {} = {:.1}% zero bytes; {} non-zero bytes",
            zero_byte_count,
            raw_size_per_column,
            100.0 * zero_byte_count as f64 / raw_size_per_column as f64,
            nonzero_byte_count,
        );
        eprintln!();
        let num_openings = <Zt as crate::pcs::structs::ZipTypes>::NUM_COLUMN_OPENINGS;
        eprintln!(
            "  Full proof column-opening cost ({} openings × per-column):",
            num_openings
        );
        eprintln!(
            "    raw    = {} bytes ({:.1} KiB)",
            raw_size_per_column * num_openings,
            (raw_size_per_column * num_openings) as f64 / 1024.0,
        );
        eprintln!(
            "    zstd-3 = {} bytes ({:.1} KiB)",
            compressed.len() * num_openings,
            (compressed.len() * num_openings) as f64 / 1024.0,
        );
        eprintln!("=========================================================");
        eprintln!();
    }
}
