//! Binary additive RS-FFT code over `GF(2^16) = F_2[X]/(X^16 + X^5 + X^3
//! + X^2 + 1)` lifted structurally to `Z[X]/f̃` with `f̃ = X^16 + X^5 +
//! X^3 + X^2 + 1`, using a **mixed-radix** evaluator: each stage applies
//! a radix-8 (8×8) or radix-4 (4×4) `GF(2^16)` Vandermonde matvec
//! whose entries are precomputed in GF and lifted to 0/1-coef
//! polynomials in `Z[X]/f̃`. Radix-4 stages sit at the base.
//!
//! Intended use: the "1× folded" Zip+ variant for the binary lane that
//! commits `BinaryPoly<16>` rows, as the protocol's `FoldedZincTypes`
//! path expects (each `BinaryPoly<32>` witness column is split into 2
//! `BinaryPoly<16>` halves).

pub mod basis;
pub mod ext_field;
pub mod mixed_radix;
pub mod packed_cw;
pub mod params;
pub mod poly_ext;
pub mod ring_ops;

use std::{fmt::Debug, marker::PhantomData};

use crypto_primitives::FromPrimitiveWithConfig;
use zinc_poly::{ConstCoeffBitWidth, univariate::dense::DensePolynomial};
use zinc_utils::from_ref::FromRef;

use crate::{ZipError, code::LinearCode, pcs::structs::ZipTypes};
pub use mixed_radix::MixedRadixFftParams16;
pub use params::{AddFftConfigGF2_16, Config16};
use ring_ops::{FftRingElement16, I32FftConvert};

/// Fixed seed for the signed `{-1, 0, 1}` twiddle lift used by
/// [`BinaryAddFft16Code::new_signed`]. The signs are public,
/// deterministic FFT parameters; this value is chosen so the certified
/// worst-case codeword bound for the shipped folded-lane sizes (m = 12,
/// m = 13) stays below `2^23`, i.e. fits a signed 24-bit pack — see the
/// `certified_worst_case_bound` test in `mixed_radix`.
pub(crate) const SIGN_SEED: u64 = 3;

/// Codeword lengths for which the signed-lift encode is certified to
/// stay within the `i32` fast path — *every* intermediate FFT stage,
/// not only the final codeword (the `i32` path's element array holds
/// the intermediates). Verified by the `certified_worst_case_bound`
/// test in `mixed_radix`; other lengths fall back to the `i64` path.
fn signed_lift_i32_safe(codeword_len: usize) -> bool {
    matches!(codeword_len, 4096 | 8192)
}


/// Reed-Solomon linear code whose encoder is a **mixed-radix** additive
/// FFT over `GF(2^16)` lifted structurally to `Z[X]/(f̃)`. Each stage
/// covers three (radix-8) or two (radix-4) subspace polynomials and
/// is applied as an 8×8 or 4×4 `GF(2^16)` Vandermonde matvec carried
/// out in `Z[X]/f̃` (per-entry twiddles are 0/1-coef GF-lifted).
///
/// Requires `m = log2(row_len * REP)` to be expressible as `3a + 2b`
/// with non-negative `a, b` — every `m` except `m = 1`. Pure
/// radix-8 (`m % 3 == 0`) uses no radix-4 stages.
pub struct BinaryAddFft16Code<
    Zt: ZipTypes,
    C: Config16,
    const REP: usize,
    const CHECK: bool,
> {
    params: MixedRadixFftParams16<C>,
    /// Whether the narrow-`i32` encode fast path is provably safe for
    /// these parameters — decided once in [`Self::new`] by a sound
    /// worst-case probe (see [`Self::probe_i32_safe`]).
    i32_safe: bool,
    _phantom: PhantomData<Zt>,
}

impl<Zt: ZipTypes, C: Config16, const REP: usize, const CHECK: bool>
    BinaryAddFft16Code<Zt, C, REP, CHECK>
{
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
        let params = MixedRadixFftParams16::new(row_len, codeword_len)?;
        let i32_safe = Self::probe_i32_safe(&params);
        Ok(Self {
            params,
            i32_safe,
            _phantom: PhantomData,
        })
    }

    pub fn new_with_optimal_depth(row_len: usize) -> Result<Self, ZipError> {
        Self::new(row_len)
    }

    /// Like [`Self::new`], but lifts the Vandermonde twiddles with a
    /// randomized `{-1, 0, 1}` signed lift
    /// ([`MixedRadixFftParams16::randomize_signs`], seed [`SIGN_SEED`]).
    ///
    /// The signs introduce cancellation that cuts the lifted integer
    /// coefficient growth by ~6 bits (certified worst case) — enough to
    /// commit the codeword in a 24-bit pack instead of 32. Any sign
    /// choice is a valid lift (`±1 ≡ 1 mod 2`), so the code reduced mod
    /// 2 is unchanged.
    ///
    /// The `i32` encode fast path is used when the codeword length is
    /// one for which the signed lift is *certified* to keep every FFT
    /// stage within `i32` ([`signed_lift_i32_safe`]) — the cheap
    /// uniform-input probe `new` runs is not sound for signed twiddles
    /// (it sees the cancelled, not the worst-case, sum). Other lengths
    /// fall back to the `i64` path.
    pub fn new_signed(row_len: usize) -> Result<Self, ZipError> {
        let mut code = Self::new(row_len)?;
        code.params.randomize_signs(SIGN_SEED);
        code.i32_safe = signed_lift_i32_safe(code.params.codeword_len());
        Ok(code)
    }

    pub fn params(&self) -> &MixedRadixFftParams16<C> {
        &self.params
    }

    /// Whether `encode` runs on the narrow-`i32` fast path for these
    /// parameters.
    pub fn uses_i32_fast_path(&self) -> bool {
        self.i32_safe
    }

    /// Sound worst-case check of whether the whole transform stays
    /// within `i32` for these parameters.
    ///
    /// The lifted additive FFT is a *non-negative* integer-linear map:
    /// every codeword coefficient is a sum (twiddles are 0/1-coef, the
    /// mod-`f̃` reduction only adds) of input coefficients. So running
    /// the real FFT with every input coefficient pinned to a sound
    /// upper bound `b_in` yields, per output position, exactly the
    /// largest magnitude any codeword coefficient can reach. `i128`
    /// arithmetic carries the probe without itself overflowing
    /// (`codeword_len ≤ 2^16`, so the bound stays well under `2^64`).
    ///
    /// `b_in = 2^(Eval::COEFF_BIT_WIDTH)` is a sound upper bound on a
    /// lifted input coefficient (the lift never widens a coefficient
    /// beyond the evaluation ring's magnitude range).
    fn probe_i32_safe(params: &MixedRadixFftParams16<C>) -> bool {
        let b_in: i128 = 1i128 << Zt::Eval::COEFF_BIT_WIDTH;
        let input = DensePolynomial::<i128, 16> { coeffs: [b_in; 16] };
        let zero = <DensePolynomial<i128, 16> as FftRingElement16>::fft_zero();

        let mut probe = vec![input; params.row_len()];
        probe.resize(params.codeword_len(), zero);
        mixed_radix::additive_fft_mixed_radix_16_padded(&mut probe, params, params.row_len());

        probe
            .iter()
            .flat_map(|p| p.coeffs.iter())
            .all(|&c| c <= i128::from(i32::MAX) && c >= i128::from(i32::MIN))
    }
}

impl<Zt: ZipTypes, C: Config16, const REP: usize, const CHECK: bool> Clone
    for BinaryAddFft16Code<Zt, C, REP, CHECK>
{
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            i32_safe: self.i32_safe,
            _phantom: PhantomData,
        }
    }
}

impl<Zt: ZipTypes, C: Config16, const REP: usize, const CHECK: bool> Debug
    for BinaryAddFft16Code<Zt, C, REP, CHECK>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryAddFft16Code")
            .field("params", &self.params)
            .finish()
    }
}

impl<Zt: ZipTypes, C: Config16, const REP: usize, const CHECK: bool> PartialEq
    for BinaryAddFft16Code<Zt, C, REP, CHECK>
{
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params
    }
}

impl<Zt: ZipTypes, C: Config16, const REP: usize, const CHECK: bool> Eq
    for BinaryAddFft16Code<Zt, C, REP, CHECK>
{
}

impl<Zt, C, const REP: usize, const CHECK: bool> LinearCode<Zt>
    for BinaryAddFft16Code<Zt, C, REP, CHECK>
where
    Zt: ZipTypes,
    C: Config16,
    Zt::Cw: FftRingElement16 + FromRef<Zt::Eval> + I32FftConvert,
    Zt::CombR: FftRingElement16,
{
    const REPETITION_FACTOR: usize = REP;

    fn row_len(&self) -> usize {
        self.params.row_len()
    }

    fn codeword_len(&self) -> usize {
        self.params.codeword_len()
    }

    fn params_string(&self) -> String {
        format!(
            "row_len={}, rate=1/{REP}, log2_codeword_len={} (mixed-radix: {} radix-8 + {} radix-4)",
            self.row_len(),
            self.params.log2_codeword_len(),
            self.params.radix_stage_counts().1,
            self.params.radix_stage_counts().0,
        )
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        assert_eq!(
            row.len(),
            self.params.row_len(),
            "encode: input length {} does not match row_len {}",
            row.len(),
            self.params.row_len()
        );
        if self.i32_safe {
            // Fast path: run the FFT on 64-byte `i32` elements instead
            // of 128-byte `i64` ones. `probe_i32_safe` proved the whole
            // transform stays within `i32`, so the narrowed buffer
            // yields a codeword bit-identical to the `i64` path.
            let mut data: Vec<DensePolynomial<i32, 16>> = row
                .iter()
                .map(|e| Zt::Cw::from_ref(e).narrow_to_i32_fft())
                .collect();
            data.resize_with(
                self.params.codeword_len(),
                <DensePolynomial<i32, 16> as FftRingElement16>::fft_zero,
            );
            mixed_radix::additive_fft_mixed_radix_16_padded(&mut data, &self.params, row.len());
            return data.iter().map(Zt::Cw::widen_from_i32_fft).collect();
        }

        let mut data: Vec<Zt::Cw> = row.iter().map(Zt::Cw::from_ref).collect();
        data.resize_with(self.params.codeword_len(), Zt::Cw::fft_zero);
        mixed_radix::additive_fft_mixed_radix_16_padded(&mut data, &self.params, row.len());
        data
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        assert_eq!(
            row.len(),
            self.params.row_len(),
            "encode_wide: input length {} does not match row_len {}",
            row.len(),
            self.params.row_len()
        );
        let mut data: Vec<Zt::CombR> = row.to_vec();
        data.resize_with(self.params.codeword_len(), Zt::CombR::fft_zero);
        mixed_radix::additive_fft_mixed_radix_16_padded(&mut data, &self.params, row.len());
        data
    }

    fn encode_f<F>(&self, _row: &[F]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F>,
    {
        unimplemented!(
            "BinaryAddFft16Code::encode_f is not supported; use encode_wide for the \
             proximity check path."
        )
    }
}

#[cfg(test)]
mod linear_code_tests {
    use super::*;
    use crate::code::LinearCode;
    use crate::code::binary_add_fft_16::ext_field::{Bit4Poly16, Gf2_16Ext, P_16_DEFAULT};
    use crate::code::binary_add_fft_16::packed_cw::PackedI64Poly16;
    use crate::pcs::structs::{ZipPlus, ZipTypes};
    use crate::pcs_transcript::PcsProverTranscript;
    use crypto_primitives::crypto_bigint_uint::Uint;
    use rand::prelude::*;
    use zinc_poly::mle::DenseMultilinearExtension;
    use zinc_poly::univariate::binary::BinaryPoly;
    use zinc_poly::univariate::dense::DensePolynomial;
    use zinc_transcript::traits::Transcript;
    use zinc_utils::CHECKED;

    /// ZipTypes for the D=16 polynomial-valued variant with direct i64
    /// Cw / CombR backing (no Int wrappers). i64 is needed because
    /// codeword_len at the bench dims (1024 * REP = 4096) amplifies the
    /// coefficient magnitudes enough to overflow i32.
    ///
    /// - `Eval = BinaryPoly<16>` — natural shape of a 1×-folded binary
    ///   witness (16 bits per entry instead of 32).
    /// - `Cw = CombR = DensePolynomial<i64, 16>`.
    /// - `Chal = Pt = Bit4Poly16` — 16-nibble polynomial challenge.
    #[derive(Debug, Clone)]
    struct PolyChalZt16I64;
    impl ZipTypes for PolyChalZt16I64 {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = BinaryPoly<16>;
        type Cw = DensePolynomial<i64, 16>;
        type Fmod = Uint<16>; // matches Gf2_16Ext::Modulus
        type PrimeTest = zinc_primality::MillerRabin;
        type Chal = Bit4Poly16;
        type Pt = Bit4Poly16;
        type CombR = DensePolynomial<i64, 16>;
        type Comb = Self::CombR;
        type EvalDotChal = zinc_utils::inner_product::ScalarProduct;
        type CombDotChal = zinc_utils::inner_product::ScalarProduct;
        type ArrCombRDotChal = zinc_utils::inner_product::MBSInnerProduct;
    }

    /// Polychal `ZipTypes` with the production *packed* codeword:
    /// `Cw = PackedI64Poly16<24>` (the 24-bit pack enabled by the
    /// signed lift). `CombR` keeps an 8-bit pack (combined-row coefs
    /// are tiny at `num_rows = 1`).
    #[derive(Debug, Clone)]
    struct PolyChalZt16Packed24;
    impl ZipTypes for PolyChalZt16Packed24 {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = BinaryPoly<16>;
        type Cw = PackedI64Poly16<24>;
        type Fmod = Uint<16>;
        type PrimeTest = zinc_primality::MillerRabin;
        type Chal = Bit4Poly16;
        type Pt = Bit4Poly16;
        type CombR = PackedI64Poly16<8>;
        type Comb = Self::CombR;
        type EvalDotChal = zinc_utils::inner_product::ScalarProduct;
        type CombDotChal = zinc_utils::inner_product::ScalarProduct;
        type ArrCombRDotChal = zinc_utils::inner_product::MBSInnerProduct;
    }

    /// End-to-end check of the *shipped* polychal configuration: signed
    /// `{-1,0,1}` lift (`new_signed`) **and** the 24-bit codeword pack.
    /// If any codeword coefficient exceeded the signed 24-bit range the
    /// `write_packed` debug-assert would fire (or the proximity check
    /// would fail) — so a passing roundtrip certifies the pack is
    /// lossless for these dimensions.
    #[test]
    fn commit_prove_verify_polychal_signed_packed24() {
        const REP: usize = 4;
        type Zt = PolyChalZt16Packed24;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;
        type F = Gf2_16Ext<P_16_DEFAULT>;

        let num_vars = 10usize;
        let row_len = 1024usize;
        let batch = 11usize;
        let code = Code::new_signed(row_len).expect("valid radix-8 params");
        let poly_size = 1usize << num_vars;
        let pp = ZipPlus::<Zt, Code>::setup(poly_size, code);

        let mut rng = StdRng::seed_from_u64(0x7a4b_24cd);
        let polys: Vec<DenseMultilinearExtension<BinaryPoly<16>>> = (0..batch)
            .map(|_| {
                let evaluations: Vec<BinaryPoly<16>> = (0..poly_size)
                    .map(|_| BinaryPoly::<16>::from(rng.random::<u16>() as u64))
                    .collect();
                DenseMultilinearExtension::<BinaryPoly<16>> { num_vars, evaluations }
            })
            .collect();

        let (hint, comm) = ZipPlus::<Zt, Code>::commit(&pp, &polys).unwrap();
        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = ();
        let point: Vec<Bit4Poly16> = (0..num_vars).map(|i| Bit4Poly16(2 + i as i64)).collect();

        let eval_f = ZipPlus::<Zt, Code>::prove::<F, CHECKED>(
            &mut transcript, &pp, &polys, &point, &hint, &field_cfg,
        )
        .expect("prove succeeds");

        let point_f: Vec<F> = point
            .iter()
            .map(|v| <F as crypto_primitives::FromWithConfig<&Bit4Poly16>>::from_with_cfg(v, &field_cfg))
            .collect();
        let mut transcript = transcript.into_verification_transcript();
        transcript.fs_transcript.absorb_slice(&comm.root.0);

        let result = ZipPlus::<Zt, Code>::verify::<F, CHECKED>(
            &mut transcript, &pp, &comm, &field_cfg, &point_f, &eval_f,
        );
        assert!(result.is_ok(), "signed+packed24 polychal verify failed: {result:?}");
    }

    /// Encode-only smoke test for the radix-8 variant at the bench dims:
    /// `row_len = 1024`, REP = 4 ⇒ codeword_len = 4096, log2 = 12 (✓ %3 = 0).
    #[test]
    fn encode_smoke_16() {
        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;

        let row_len = 1024usize;
        let code = Code::new(row_len).expect("valid radix-8 params");

        assert_eq!(LinearCode::<Zt>::row_len(&code), row_len);
        assert_eq!(LinearCode::<Zt>::codeword_len(&code), row_len * REP);

        let row: Vec<BinaryPoly<16>> = (0..row_len)
            .map(|i| BinaryPoly::<16>::from((i as u64).wrapping_mul(0x0123_4567) | 1))
            .collect();

        let cw = LinearCode::<Zt>::encode(&code, &row);
        assert_eq!(cw.len(), row_len * REP);
    }

    /// Bench-dim commit→prove→verify roundtrip for the radix-8 variant:
    /// `num_vars=10, row_len=1024, batch=11`.
    #[test]
    fn commit_prove_verify_polychal_smoke_16() {
        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;
        type F = Gf2_16Ext<P_16_DEFAULT>;

        let num_vars = 10usize;
        let row_len = 1024usize;
        let batch = 11usize;
        let code = Code::new(row_len).expect("valid radix-8 params");
        let poly_size = 1usize << num_vars;
        let pp = ZipPlus::<Zt, Code>::setup(poly_size, code);

        let mut rng = StdRng::seed_from_u64(0xdead_b00f);
        let polys: Vec<DenseMultilinearExtension<BinaryPoly<16>>> = (0..batch)
            .map(|_| {
                let evaluations: Vec<BinaryPoly<16>> = (0..poly_size)
                    .map(|_| BinaryPoly::<16>::from(rng.random::<u16>() as u64))
                    .collect();
                DenseMultilinearExtension::<BinaryPoly<16>> { num_vars, evaluations }
            })
            .collect();

        let (hint, comm) = ZipPlus::<Zt, Code>::commit(&pp, &polys).unwrap();
        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = ();
        let point: Vec<Bit4Poly16> = (0..num_vars).map(|i| Bit4Poly16(2 + i as i64)).collect();

        let eval_f = ZipPlus::<Zt, Code>::prove::<F, CHECKED>(
            &mut transcript, &pp, &polys, &point, &hint, &field_cfg,
        )
        .expect("prove succeeds");

        let point_f: Vec<F> = point
            .iter()
            .map(|v| <F as crypto_primitives::FromWithConfig<&Bit4Poly16>>::from_with_cfg(v, &field_cfg))
            .collect();
        let mut transcript = transcript.into_verification_transcript();
        transcript.fs_transcript.absorb_slice(&comm.root.0);

        let result = ZipPlus::<Zt, Code>::verify::<F, CHECKED>(
            &mut transcript, &pp, &comm, &field_cfg, &point_f, &eval_f,
        );
        assert!(result.is_ok(), "radix-8 polychal batch verify failed: {result:?}");
    }

    /// Commit→prove→verify roundtrip with the **signed** `{-1,0,1}`
    /// twiddle lift (`new_signed`): exercises the signed FFT end to end
    /// through the PCS. The signed lift changes the integer codeword
    /// but prover and verifier share the same signed parameters, so the
    /// opening still verifies.
    #[test]
    fn commit_prove_verify_polychal_signed_16() {
        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;
        type F = Gf2_16Ext<P_16_DEFAULT>;

        let num_vars = 10usize;
        let row_len = 1024usize;
        let batch = 11usize;
        let code = Code::new_signed(row_len).expect("valid radix-8 params");
        assert!(
            code.uses_i32_fast_path(),
            "signed m=12 is certified i32-safe, so it must use the i32 encode path"
        );
        let poly_size = 1usize << num_vars;
        let pp = ZipPlus::<Zt, Code>::setup(poly_size, code);

        let mut rng = StdRng::seed_from_u64(0x5169_64ed);
        let polys: Vec<DenseMultilinearExtension<BinaryPoly<16>>> = (0..batch)
            .map(|_| {
                let evaluations: Vec<BinaryPoly<16>> = (0..poly_size)
                    .map(|_| BinaryPoly::<16>::from(rng.random::<u16>() as u64))
                    .collect();
                DenseMultilinearExtension::<BinaryPoly<16>> { num_vars, evaluations }
            })
            .collect();

        let (hint, comm) = ZipPlus::<Zt, Code>::commit(&pp, &polys).unwrap();
        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = ();
        let point: Vec<Bit4Poly16> = (0..num_vars).map(|i| Bit4Poly16(2 + i as i64)).collect();

        let eval_f = ZipPlus::<Zt, Code>::prove::<F, CHECKED>(
            &mut transcript, &pp, &polys, &point, &hint, &field_cfg,
        )
        .expect("prove succeeds");

        let point_f: Vec<F> = point
            .iter()
            .map(|v| <F as crypto_primitives::FromWithConfig<&Bit4Poly16>>::from_with_cfg(v, &field_cfg))
            .collect();
        let mut transcript = transcript.into_verification_transcript();
        transcript.fs_transcript.absorb_slice(&comm.root.0);

        let result = ZipPlus::<Zt, Code>::verify::<F, CHECKED>(
            &mut transcript, &pp, &comm, &field_cfg, &point_f, &eval_f,
        );
        assert!(result.is_ok(), "signed polychal batch verify failed: {result:?}");
    }

    /// The narrow-`i32` encode fast path must produce a codeword
    /// bit-identical to the reference `i64` FFT, and the soundness
    /// probe must only accept it when every coefficient fits `i32`.
    #[test]
    fn i32_fast_path_matches_i64() {
        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;

        let row_len = 1024usize; // m = 12
        let code = Code::new(row_len).expect("valid radix-8 params");
        assert!(
            code.uses_i32_fast_path(),
            "m=12 (peak ~2^26) should select the i32 fast path"
        );

        let mut rng = StdRng::seed_from_u64(0x1234_5678);
        let row: Vec<BinaryPoly<16>> = (0..row_len)
            .map(|_| BinaryPoly::<16>::from(rng.random::<u16>() as u64))
            .collect();

        // Fast path, via the public encoder.
        let cw_fast = LinearCode::<Zt>::encode(&code, &row);

        // Reference: the i64 FFT run directly.
        let mut reference: Vec<DensePolynomial<i64, 16>> =
            row.iter().map(<DensePolynomial<i64, 16>>::from_ref).collect();
        reference.resize_with(
            row_len * REP,
            <DensePolynomial<i64, 16> as FftRingElement16>::fft_zero,
        );
        mixed_radix::additive_fft_mixed_radix_16(&mut reference, code.params());

        assert_eq!(
            cw_fast, reference,
            "i32 fast-path codeword differs from the i64 reference"
        );
        for cell in &cw_fast {
            for &c in &cell.coeffs {
                assert!(
                    i32::try_from(c).is_ok(),
                    "coefficient {c} exceeds i32 but the probe accepted the fast path"
                );
            }
        }
    }

    /// The soundness probe must reject the `i32` path when the codeword
    /// is large enough to overflow it (m=16 peaks at ~2^39).
    #[test]
    fn probe_discriminates_by_codeword_size() {
        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;

        // m = 12, codeword 4096: peak ~2^26 → i32-safe.
        assert!(Code::new(1024).unwrap().uses_i32_fast_path());
        // m = 16, codeword 65536: peak ~2^39 → must fall back to i64.
        assert!(!Code::new(16384).unwrap().uses_i32_fast_path());
    }

    /// Controlled A/B of the encode FFT: `i32` vs `i64` working buffer,
    /// interleaved in one process so machine noise cancels. Ignored by
    /// default; run with `cargo test -p zip-plus --release --features
    /// parallel encode_i32_vs_i64_ab -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing measurement"]
    fn encode_i32_vs_i64_ab() {
        use std::time::{Duration, Instant};

        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;

        for &log_row in &[10usize, 11] {
            let row_len = 1usize << log_row;
            let code = Code::new(row_len).expect("valid radix-8 params");
            let code_signed = Code::new_signed(row_len).expect("valid radix-8 params");
            let cw_len = row_len * REP;
            let mut rng = StdRng::seed_from_u64(0xa11ce);
            let row: Vec<BinaryPoly<16>> = (0..row_len)
                .map(|_| BinaryPoly::<16>::from(rng.random::<u16>() as u64))
                .collect();

            let build_i64 = || {
                let mut d: Vec<DensePolynomial<i64, 16>> =
                    row.iter().map(<DensePolynomial<i64, 16>>::from_ref).collect();
                d.resize_with(cw_len, <DensePolynomial<i64, 16> as FftRingElement16>::fft_zero);
                d
            };
            let build_i32 = || {
                let mut d: Vec<DensePolynomial<i32, 16>> = row
                    .iter()
                    .map(|e| <DensePolynomial<i64, 16>>::from_ref(e).narrow_to_i32_fft())
                    .collect();
                d.resize_with(cw_len, <DensePolynomial<i32, 16> as FftRingElement16>::fft_zero);
                d
            };

            let iters = 400;
            let (mut t32, mut t64) = (Duration::MAX, Duration::MAX);
            let (mut t32s, mut t64s) = (Duration::MAX, Duration::MAX);
            for _ in 0..iters {
                // unsigned i32 fast path (the commit-2 baseline)
                let mut d32 = build_i32();
                let s = Instant::now();
                mixed_radix::additive_fft_mixed_radix_16(&mut d32, code.params());
                t32 = t32.min(s.elapsed());
                std::hint::black_box(&d32);

                // unsigned i64
                let mut d64 = build_i64();
                let s = Instant::now();
                mixed_radix::additive_fft_mixed_radix_16(&mut d64, code.params());
                t64 = t64.min(s.elapsed());
                std::hint::black_box(&d64);

                // signed i32 (the shipped polychal encode path for m=12/13)
                let mut d32s = build_i32();
                let s = Instant::now();
                mixed_radix::additive_fft_mixed_radix_16(&mut d32s, code_signed.params());
                t32s = t32s.min(s.elapsed());
                std::hint::black_box(&d32s);

                // signed i64
                let mut d64s = build_i64();
                let s = Instant::now();
                mixed_radix::additive_fft_mixed_radix_16(&mut d64s, code_signed.params());
                t64s = t64s.min(s.elapsed());
                std::hint::black_box(&d64s);
            }
            println!(
                "single-row FFT m={} (codeword {cw_len}) : \
                 i32={t32:?}  i64={t64:?}  signed-i32={t32s:?}  signed-i64={t64s:?}  \
                 (signed-i32-vs-unsigned-i32 {:.2}x)",
                log_row + 2,
                t32s.as_secs_f64() / t32.as_secs_f64(),
            );
        }
    }

    /// Commit-time breakdown: how much of `commit` is RS-FFT encoding
    /// vs Merkle-tree construction. Ignored by default; run with
    /// `cargo test -p zip-plus --release --features parallel \
    ///   commit_time_breakdown -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing measurement"]
    fn commit_time_breakdown() {
        use crate::merkle::MerkleTree;
        use std::time::{Duration, Instant};
        use zinc_utils::cfg_iter;
        #[cfg(feature = "parallel")]
        use rayon::prelude::*;

        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;

        // (num_vars, batch); num_rows = 1 so row_len = poly_size.
        for &(num_vars, batch) in &[(10usize, 11usize), (13usize, 11usize)] {
            let row_len = 1usize << num_vars;
            let poly_size = row_len;
            let code = Code::new(row_len).expect("valid radix-8 params");
            let pp = ZipPlus::<Zt, Code>::setup(poly_size, code);

            let mut rng = StdRng::seed_from_u64(0xdead_b00f);
            let polys: Vec<DenseMultilinearExtension<BinaryPoly<16>>> = (0..batch)
                .map(|_| {
                    let evaluations: Vec<BinaryPoly<16>> = (0..poly_size)
                        .map(|_| BinaryPoly::<16>::from(rng.random::<u16>() as u64))
                        .collect();
                    DenseMultilinearExtension::<BinaryPoly<16>> { num_vars, evaluations }
                })
                .collect();

            let iters = 5;
            let (mut enc_min, mut mt_min) = (Duration::MAX, Duration::MAX);
            for _ in 0..iters {
                let t0 = Instant::now();
                let cw_matrices: Vec<_> = cfg_iter!(polys)
                    .map(|p| ZipPlus::<Zt, Code>::encode_rows(&pp, p))
                    .collect();
                let enc = t0.elapsed();

                let t1 = Instant::now();
                let all_rows: Vec<&[<Zt as ZipTypes>::Cw]> =
                    cw_matrices.iter().flat_map(|m| m.as_rows()).collect();
                let mt = MerkleTree::new(&all_rows);
                let mt_elapsed = t1.elapsed();
                std::hint::black_box(mt.root());

                enc_min = enc_min.min(enc);
                mt_min = mt_min.min(mt_elapsed);
            }
            let total = enc_min + mt_min;
            let pct = |d: Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64();
            println!(
                "num_vars={num_vars} batch={batch} codeword_len={} : \
                 encode={enc_min:?} ({:.1}%)  merkle={mt_min:?} ({:.1}%)  total={total:?}",
                poly_size * REP,
                pct(enc_min),
                pct(mt_min),
            );
        }
    }

    /// Measure the empirical max |coefficient| of the radix-8 codeword
    /// at the bench dims, with random binary input. Compares directly
    /// against the radix-2 measurement (~2^47, 49 bits incl. sign).
    #[test]
    fn codeword_coef_bits() {
        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;

        for (label, row_len) in [
            ("radix-8 REP=4, row_len=1024 (=1<<10)", 1024usize),
        ] {
            if row_len * REP > (1usize << 16) {
                eprintln!("  skip {label}: codeword_len > 2^16");
                continue;
            }
            let code = Code::new(row_len).expect("valid radix-8 params");
            let mut rng = StdRng::seed_from_u64(0xdead_b00f);
            let row: Vec<BinaryPoly<16>> = (0..row_len)
                .map(|_| BinaryPoly::<16>::from(rng.random::<u64>()))
                .collect();
            let cw = LinearCode::<Zt>::encode(&code, &row);
            let mut max_abs: i64 = 0;
            for cell in &cw {
                for &c in &cell.coeffs {
                    let a = c.unsigned_abs() as i64;
                    if a > max_abs {
                        max_abs = a;
                    }
                }
            }
            let bits = if max_abs == 0 {
                0
            } else {
                65 - (max_abs as u64).leading_zeros() as u64
            };
            eprintln!(
                "  {label}: codeword_len={}, max |coef| = {max_abs} (~ 2^{:.2}), bits-incl-sign = {}",
                row_len * REP,
                if max_abs == 0 { 0.0 } else { (max_abs as f64).log2() },
                bits,
            );
        }
    }

    /// Measure the empirical max |coefficient| of the codeword at the
    /// rate-1/8 SHA dims (`row_len = 1024`, `REP = 8`, `codeword_len =
    /// 8192`, `m = 13`). This `m` is not divisible by 3, so the encoder
    /// runs two radix-4 base stages plus three radix-8 stages. The
    /// result must stay under `2^32` for `BITS_CW = 32` to be lossless.
    #[test]
    fn codeword_coef_bits_rep8_radix4() {
        const REP: usize = 8;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;

        let row_len = 1024usize;
        let code = Code::new(row_len).expect("valid mixed-radix params");
        // m = 13 → two radix-4 + three radix-8 stages.
        assert_eq!(code.params().radix_stage_counts(), (2, 3));

        let mut rng = StdRng::seed_from_u64(0xdead_b00f);
        let row: Vec<BinaryPoly<16>> = (0..row_len)
            .map(|_| BinaryPoly::<16>::from(rng.random::<u64>()))
            .collect();
        let cw = LinearCode::<Zt>::encode(&code, &row);
        let mut max_abs: i64 = 0;
        for cell in &cw {
            for &c in &cell.coeffs {
                let a = c.unsigned_abs() as i64;
                if a > max_abs {
                    max_abs = a;
                }
            }
        }
        let bits = if max_abs == 0 {
            0
        } else {
            65 - (max_abs as u64).leading_zeros() as u64
        };
        eprintln!(
            "  mixed-radix REP=8, row_len=1024: codeword_len={}, m=13, \
             max |coef| = {max_abs} (~ 2^{:.2}), bits-incl-sign = {}",
            row_len * REP,
            if max_abs == 0 { 0.0 } else { (max_abs as f64).log2() },
            bits,
        );
        assert!(
            bits <= 32,
            "rate-1/8 codeword coefficient needs {bits} bits, exceeds BITS_CW=32"
        );
    }

    /// Tamper test for the radix-8 variant.
    #[test]
    fn polychal_tamper_detection_16() {
        const REP: usize = 4;
        type Zt = PolyChalZt16I64;
        type Code = BinaryAddFft16Code<Zt, AddFftConfigGF2_16, REP, false>;
        type F = Gf2_16Ext<P_16_DEFAULT>;

        // num_vars=9, row_len=512 (codeword_len=2048, m=11) — NOT divisible by 3.
        // Use row_len=128, REP=4 → codeword_len=512, m=9 (divisible by 3 ✓).
        let num_vars = 9usize;
        let row_len = 128usize;
        let code = Code::new(row_len).expect("valid radix-8 params");
        let poly_size = 1usize << num_vars;
        let pp = ZipPlus::<Zt, Code>::setup(poly_size, code);

        let evaluations: Vec<BinaryPoly<16>> = (0..poly_size)
            .map(|j| BinaryPoly::<16>::from((j as u64).wrapping_mul(0xA5A5_3CB7_C001_C0DE) | 1))
            .collect();
        let poly = DenseMultilinearExtension::<BinaryPoly<16>> { num_vars, evaluations };

        let (hint, comm) = ZipPlus::<Zt, Code>::commit_single(&pp, &poly).unwrap();
        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = ();

        let point: Vec<Bit4Poly16> = (0..num_vars).map(|i| Bit4Poly16(2 + i as i64)).collect();

        let eval_f = ZipPlus::<Zt, Code>::prove_single::<F, CHECKED>(
            &mut transcript, &pp, &poly, &point, &hint, &field_cfg,
        )
        .expect("prove succeeds");

        let point_f: Vec<F> = point
            .iter()
            .map(|v| <F as crypto_primitives::FromWithConfig<&Bit4Poly16>>::from_with_cfg(v, &field_cfg))
            .collect();

        let tampered = eval_f + F::ONE_VAL;

        let mut transcript = transcript.into_verification_transcript();
        transcript.fs_transcript.absorb_slice(&comm.root.0);

        let result = ZipPlus::<Zt, Code>::verify::<F, CHECKED>(
            &mut transcript, &pp, &comm, &field_cfg, &point_f, &tampered,
        );
        assert!(result.is_err(), "radix-8 tampered eval should fail");
    }
}
