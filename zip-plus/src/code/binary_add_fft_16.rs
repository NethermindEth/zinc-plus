//! Binary additive RS-FFT code over `GF(2^16) = F_2[X]/(X^16 + X^5 + X^3
//! + X^2 + 1)` lifted structurally to `Z[X]/f̃` with `f̃ = X^16 + X^5 +
//! X^3 + X^2 + 1`, using a **radix-8** evaluator: each meta-stage applies
//! an 8×8 `GF(2^16)` Vandermonde matvec whose entries are precomputed in
//! GF and lifted to 0/1-coef polynomials in `Z[X]/f̃`.
//!
//! Intended use: the "1× folded" Zip+ variant for the binary lane that
//! commits `BinaryPoly<16>` rows, as the protocol's `FoldedZincTypes`
//! path expects (each `BinaryPoly<32>` witness column is split into 2
//! `BinaryPoly<16>` halves).

pub mod basis;
pub mod ext_field;
pub mod packed_cw;
pub mod params;
pub mod radix8;
pub mod ring_ops;

use std::{fmt::Debug, marker::PhantomData};

use crypto_primitives::FromPrimitiveWithConfig;
use zinc_utils::from_ref::FromRef;

use crate::{ZipError, code::LinearCode, pcs::structs::ZipTypes};
pub use params::{AddFftConfigGF2_16, Config16};
pub use radix8::Radix8FftParams16;
use ring_ops::FftRingElement16;


/// Reed-Solomon linear code whose encoder is a **radix-8** additive
/// FFT over `GF(2^16)` lifted structurally to `Z[X]/(f̃)`. Each radix-8
/// meta-stage covers three subspace polynomials and is applied as an
/// 8×8 `GF(2^16)` Vandermonde matvec carried out in `Z[X]/f̃` (per-entry
/// twiddles are 0/1-coef GF-lifted).
///
/// Requires `log2(row_len * REP) % 3 == 0` (e.g. `row_len=1024, REP=4`
/// gives `m=12`, `m/3 = 4` meta-stages).
pub struct BinaryAddFft16Code<
    Zt: ZipTypes,
    C: Config16,
    const REP: usize,
    const CHECK: bool,
> {
    params: Radix8FftParams16<C>,
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
        let params = Radix8FftParams16::new(row_len, codeword_len)?;
        Ok(Self {
            params,
            _phantom: PhantomData,
        })
    }

    pub fn new_with_optimal_depth(row_len: usize) -> Result<Self, ZipError> {
        Self::new(row_len)
    }

    pub fn params(&self) -> &Radix8FftParams16<C> {
        &self.params
    }
}

impl<Zt: ZipTypes, C: Config16, const REP: usize, const CHECK: bool> Clone
    for BinaryAddFft16Code<Zt, C, REP, CHECK>
{
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
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
    Zt::Cw: FftRingElement16 + FromRef<Zt::Eval>,
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
            "row_len={}, rate=1/{REP}, log2_codeword_len={} (radix-8)",
            self.row_len(),
            self.params.log2_codeword_len()
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
        let mut data: Vec<Zt::Cw> = row.iter().map(Zt::Cw::from_ref).collect();
        data.resize_with(self.params.codeword_len(), Zt::Cw::fft_zero);
        radix8::additive_fft_radix8_16(&mut data, &self.params);
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
        radix8::additive_fft_radix8_16(&mut data, &self.params);
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
