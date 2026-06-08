#![allow(clippy::arithmetic_side_effects)]

use std::{
    fmt::Debug,
    io::{Cursor, Read, Write},
    marker::PhantomData,
};

use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInteger, PrimeField as ArkPrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use crypto_bigint::{BoxedUint, modular::BoxedMontyForm};
use crypto_primitives::{
    FromWithConfig, IntRing, PrimeField, crypto_bigint_boxed_monty::BoxedMontyField,
    crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use num_integer::Integer;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};

use crate::{
    ZipError,
    pcs::{
        generic::{FoldablePCS, PCS},
        msm_commitment::{
            BoolSubsetMsm, MsmCommitmentEngine, MsmCommitmentKey, MsmError, RowMsmStrategy,
            ScalarPippengerMsm,
        },
    },
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};

#[derive(Clone, Debug)]
pub struct HyraxPCS<C: AffineRepr, Lanes>(PhantomData<(C, Lanes)>);

impl<C, Lanes> HyraxPCS<C, Lanes>
where
    C: AffineRepr,
{
    /// Open a folded Hyrax commitment whose lane values are already scalar
    /// field elements.
    ///
    /// This is needed for instance-axis folds of binary commitments: after
    /// folding by transcript-derived weights, each bit lane is a scalar field
    /// linear combination of bits, not a `bool`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_open_scalar_lanes<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        ck: &HyraxCommitmentKey<C>,
        scalar_lanes: &[Vec<Vec<C::ScalarField>>],
        point: &[F],
        prover_data: &HyraxProverData<C>,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: HyraxFieldBridge<C>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
        Lanes: Clone + Debug + Send + Sync,
    {
        let _ = CHECK_FOR_OVERFLOW;
        if scalar_lanes.is_empty() {
            return Ok(());
        }
        validate_scalar_lanes::<C>(ck, scalar_lanes, point.len(), prover_data)?;

        let n = scalar_lanes[0][0].len();
        let point_scalar = point.iter().map(F::field_to_scalar).collect::<Vec<_>>();
        let row_vars = prover_data.num_rows.ilog2() as usize;
        let q0_f = eq_tensor_f::<F>(&point[..row_vars], field_cfg);
        let q1_scalar = eq_tensor_scalar::<C>(&point_scalar[row_vars..]);
        let alphas = sample_scalars::<C>(
            &mut transcript.fs_transcript,
            scalar_lanes.len() * prover_data.num_lanes,
        );

        let mut b_scalar = vec![C::ScalarField::zero(); prover_data.num_rows];
        for (poly_idx, lanes) in scalar_lanes.iter().enumerate() {
            for (lane, values) in lanes.iter().enumerate() {
                let alpha = alphas[alpha_index_dynamic(prover_data.num_lanes, poly_idx, lane)];
                for (row_idx, row) in values.chunks(ck.num_cols).enumerate() {
                    let mut row_eval = C::ScalarField::zero();
                    for (col_idx, value) in row.iter().enumerate() {
                        if let Some(weight) = q1_scalar.get(col_idx) {
                            row_eval += *value * weight;
                        }
                    }
                    b_scalar[row_idx] += alpha * row_eval;
                }
            }
        }

        let b_f = b_scalar
            .iter()
            .map(|value| F::scalar_to_field(value, field_cfg))
            .collect::<Vec<_>>();
        transcript.write_field_elements(&b_f)?;

        let row_coeffs = if prover_data.num_rows == 1 {
            vec![C::ScalarField::from(1u64)]
        } else {
            sample_scalars::<C>(&mut transcript.fs_transcript, prover_data.num_rows)
        };

        let mut combined_row = vec![C::ScalarField::zero(); ck.num_cols];
        let mut rho_star = C::ScalarField::zero();
        for (poly_idx, lanes) in scalar_lanes.iter().enumerate() {
            for (lane, values) in lanes.iter().enumerate() {
                let alpha = alphas[alpha_index_dynamic(prover_data.num_lanes, poly_idx, lane)];
                for (row_idx, row) in values.chunks(ck.num_cols).enumerate() {
                    let coeff = alpha * row_coeffs[row_idx];
                    if ck.blinding_mode.is_blinded() {
                        let blind_idx = commitment_index_dynamic(
                            prover_data.num_lanes,
                            poly_idx,
                            lane,
                            row_idx,
                            prover_data.num_rows,
                        );
                        rho_star += coeff * prover_data.blinds[blind_idx];
                    }
                    for (col_idx, value) in row.iter().enumerate() {
                        combined_row[col_idx] += coeff * value;
                    }
                }
            }
        }

        write_scalars::<C>(transcript, &combined_row)?;
        if ck.blinding_mode.is_blinded() {
            write_scalar::<C>(transcript, &rho_star)?;
        }

        if q0_f.len() != b_f.len() || n != (1usize << point.len()) {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax folded scalar-lane opening shape mismatch".to_string(),
            ));
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyraxBlindingMode {
    Blinded,
    Unblinded,
}

impl HyraxBlindingMode {
    fn as_u8(self) -> u8 {
        match self {
            Self::Blinded => 1,
            Self::Unblinded => 0,
        }
    }

    fn is_blinded(self) -> bool {
        matches!(self, Self::Blinded)
    }
}

#[derive(Clone, Debug)]
pub struct HyraxCommitmentKey<C: AffineRepr> {
    pub(crate) num_cols: usize,
    pub(crate) blinding_mode: HyraxBlindingMode,
    pub(crate) msm_ck: MsmCommitmentKey<C>,
}

#[derive(Clone, Debug)]
pub struct HyraxVerifierKey<C: AffineRepr> {
    pub(crate) num_cols: usize,
    pub(crate) bases: Vec<C>,
    pub(crate) h: C::Group,
    pub(crate) blinding_mode: HyraxBlindingMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyraxCommitment<C: AffineRepr> {
    pub(crate) batch_size: usize,
    pub(crate) num_lanes: usize,
    pub(crate) num_rows: usize,
    pub(crate) blinding_mode: HyraxBlindingMode,
    pub(crate) comm: Vec<C::Group>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyraxProverData<C: AffineRepr> {
    pub(crate) batch_size: usize,
    pub(crate) num_lanes: usize,
    pub(crate) num_rows: usize,
    pub(crate) blinding_mode: HyraxBlindingMode,
    pub(crate) blinds: Vec<C::ScalarField>,
}

pub trait HyraxFieldBridge<C: AffineRepr>: PrimeField {
    fn field_to_scalar(value: &Self) -> C::ScalarField;
    fn scalar_to_field(value: &C::ScalarField, cfg: &Self::Config) -> Self;
}

impl<C, const LIMBS: usize> HyraxFieldBridge<C> for MontyField<LIMBS>
where
    C: AffineRepr,
{
    fn field_to_scalar(value: &Self) -> C::ScalarField {
        assert_curve_scalar_modulus::<C, LIMBS>(&value.modulus());

        let canonical = value.retrieve();
        let mut bytes = vec![0u8; <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES];
        canonical.write_transcription_bytes_exact(&mut bytes);
        C::ScalarField::from_le_bytes_mod_order(&bytes)
    }

    fn scalar_to_field(value: &C::ScalarField, cfg: &Self::Config) -> Self {
        let actual_modulus = Uint::<LIMBS>::new(cfg.modulus().get());
        assert_curve_scalar_modulus::<C, LIMBS>(&actual_modulus);

        let scalar_bigint: <C::ScalarField as ArkPrimeField>::BigInt = value.clone().into();
        let scalar_uint = uint_from_le_bytes::<LIMBS>(&scalar_bigint.to_bytes_le());
        MontyField::<LIMBS>::from_with_cfg(&scalar_uint, cfg)
    }
}

impl<C> HyraxFieldBridge<C> for BoxedMontyField
where
    C: AffineRepr,
{
    fn field_to_scalar(value: &Self) -> C::ScalarField {
        assert_curve_scalar_modulus_boxed::<C>(&value.modulus());

        let canonical = BoxedMontyForm::from(value.clone()).retrieve();
        C::ScalarField::from_le_bytes_mod_order(&canonical.to_le_bytes())
    }

    fn scalar_to_field(value: &C::ScalarField, cfg: &Self::Config) -> Self {
        let actual_modulus = cfg.modulus().clone().get();
        assert_curve_scalar_modulus_boxed::<C>(&actual_modulus);

        let scalar_bigint: <C::ScalarField as ArkPrimeField>::BigInt = value.clone().into();
        let scalar_uint = BoxedUint::from_le_slice(
            &scalar_bigint.to_bytes_le(),
            actual_modulus.bits_precision(),
        )
        .expect("curve scalar must fit protocol field precision");
        BoxedMontyField::from_with_cfg(&scalar_uint, cfg)
    }
}

fn validate_fold_inputs<T>(values: &[T], theta_len: usize, label: &str) -> Result<(), ZipError> {
    if values.is_empty() {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax cannot fold empty {label}"
        )));
    }
    if values.len() != theta_len {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax fold {label} count mismatch: got {}, expected {theta_len}",
            values.len()
        )));
    }
    Ok(())
}

pub trait HyraxLanes<C, Eval, const D: usize>: Clone + Debug + Send + Sync
where
    C: AffineRepr,
    Eval: Clone + Debug + Send + Sync,
{
    type LaneValue: Copy + Send + Sync;
    type Strategy: RowMsmStrategy<C, Self::LaneValue>;

    const NUM_LANES: usize;

    fn lane_value(eval: &Eval, lane: usize) -> Result<Self::LaneValue, ZipError>;

    fn lane_to_scalar(value: Self::LaneValue) -> C::ScalarField;

    fn lifted_eval<F>(
        lifted_eval: &DynamicPolynomialF<F>,
        lane: usize,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField;
}

#[derive(Clone, Debug)]
pub struct BinaryLanes;

#[derive(Clone, Debug)]
pub struct IntScalarLane;

#[derive(Clone, Debug)]
pub struct DensePolyScalarLanes;

impl<C: AffineRepr, const D: usize> HyraxLanes<C, BinaryPoly<D>, D> for BinaryLanes {
    type LaneValue = bool;
    type Strategy = BoolSubsetMsm<6>;

    const NUM_LANES: usize = D;

    fn lane_value(eval: &BinaryPoly<D>, lane: usize) -> Result<Self::LaneValue, ZipError> {
        eval.iter()
            .nth(lane)
            .map(|bit| bit.inner())
            .ok_or_else(|| ZipError::InvalidPcsParam(format!("binary lane {lane} out of range")))
    }

    fn lane_to_scalar(value: Self::LaneValue) -> C::ScalarField {
        if value {
            C::ScalarField::from(1u64)
        } else {
            C::ScalarField::zero()
        }
    }

    fn lifted_eval<F>(
        lifted_eval: &DynamicPolynomialF<F>,
        lane: usize,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField,
    {
        Ok(lifted_eval
            .coeffs
            .get(lane)
            .cloned()
            .unwrap_or_else(|| F::zero_with_cfg(field_cfg)))
    }
}

impl<C: AffineRepr, const LIMBS: usize, const D: usize> HyraxLanes<C, Int<LIMBS>, D>
    for IntScalarLane
{
    type LaneValue = C::ScalarField;
    type Strategy = ScalarPippengerMsm;

    const NUM_LANES: usize = 1;

    fn lane_value(eval: &Int<LIMBS>, lane: usize) -> Result<Self::LaneValue, ZipError> {
        if lane != 0 {
            return Err(ZipError::InvalidPcsParam(format!(
                "int lane {lane} out of range"
            )));
        }
        int_to_scalar::<C, LIMBS>(eval)
    }

    fn lane_to_scalar(value: Self::LaneValue) -> C::ScalarField {
        value
    }

    fn lifted_eval<F>(
        lifted_eval: &DynamicPolynomialF<F>,
        lane: usize,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField,
    {
        if lane != 0 {
            return Err(ZipError::InvalidPcsParam(format!(
                "lifted int lane {lane} out of range"
            )));
        }
        Ok(lifted_eval
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(|| F::zero_with_cfg(field_cfg)))
    }
}

impl<C: AffineRepr, const LIMBS: usize, const D: usize>
    HyraxLanes<C, DensePolynomial<Int<LIMBS>, D>, D> for DensePolyScalarLanes
{
    type LaneValue = C::ScalarField;
    type Strategy = ScalarPippengerMsm;

    const NUM_LANES: usize = D;

    fn lane_value(
        eval: &DensePolynomial<Int<LIMBS>, D>,
        lane: usize,
    ) -> Result<Self::LaneValue, ZipError> {
        eval.coeffs
            .get(lane)
            .ok_or_else(|| ZipError::InvalidPcsParam(format!("dense lane {lane} out of range")))
            .and_then(int_to_scalar::<C, LIMBS>)
    }

    fn lane_to_scalar(value: Self::LaneValue) -> C::ScalarField {
        value
    }

    fn lifted_eval<F>(
        lifted_eval: &DynamicPolynomialF<F>,
        lane: usize,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField,
    {
        Ok(lifted_eval
            .coeffs
            .get(lane)
            .cloned()
            .unwrap_or_else(|| F::zero_with_cfg(field_cfg)))
    }
}

impl<C: AffineRepr, Lanes> HyraxPCS<C, Lanes> {
    pub fn setup_from_bases(
        width: usize,
        bases: Vec<C>,
        h: C::Group,
    ) -> Result<(HyraxCommitmentKey<C>, HyraxVerifierKey<C>), ZipError> {
        Self::setup_from_bases_with_blinding(width, bases, h, HyraxBlindingMode::Blinded)
    }

    pub fn setup_from_bases_with_blinding(
        width: usize,
        bases: Vec<C>,
        h: C::Group,
        blinding_mode: HyraxBlindingMode,
    ) -> Result<(HyraxCommitmentKey<C>, HyraxVerifierKey<C>), ZipError> {
        if !width.is_power_of_two() {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax row width must be a power of two, got {width}"
            )));
        }
        let msm_ck = msm_key(width, bases.clone(), h)?;
        Ok((
            HyraxCommitmentKey {
                num_cols: width,
                blinding_mode,
                msm_ck,
            },
            HyraxVerifierKey {
                num_cols: width,
                bases,
                h,
                blinding_mode,
            },
        ))
    }
}

impl<F, C, Lanes, Eval, const D: usize> PCS<F, Eval, D> for HyraxPCS<C, Lanes>
where
    F: HyraxFieldBridge<C>,
    C: AffineRepr,
    Eval: Clone + Debug + Send + Sync,
    Lanes: HyraxLanes<C, Eval, D>,
{
    type CommitmentKey = HyraxCommitmentKey<C>;
    type VerifierKey = HyraxVerifierKey<C>;
    type Commitment = HyraxCommitment<C>;
    type ProverData = HyraxProverData<C>;
    type OpeningProof = Vec<u8>;

    fn precompute_ck(ck: &Self::CommitmentKey) {
        Lanes::Strategy::precompute_ck(&ck.msm_ck);
    }

    fn commit(
        ck: &Self::CommitmentKey,
        polys: &[DenseMultilinearExtension<Eval>],
    ) -> Result<(Self::ProverData, Self::Commitment), ZipError> {
        if polys.is_empty() {
            return Ok((
                HyraxProverData {
                    batch_size: 0,
                    num_lanes: Lanes::NUM_LANES,
                    num_rows: 0,
                    blinding_mode: ck.blinding_mode,
                    blinds: Vec::new(),
                },
                HyraxCommitment {
                    batch_size: 0,
                    num_lanes: Lanes::NUM_LANES,
                    num_rows: 0,
                    blinding_mode: ck.blinding_mode,
                    comm: Vec::new(),
                },
            ));
        }

        validate_polys(polys)?;
        let n = polys[0].evaluations.len();
        let num_rows = num_rows(n, ck.num_cols)?;
        let mut all_comm = Vec::with_capacity(polys.len() * Lanes::NUM_LANES * num_rows);
        let mut all_blinds = if ck.blinding_mode.is_blinded() {
            Vec::with_capacity(polys.len() * Lanes::NUM_LANES * num_rows)
        } else {
            Vec::new()
        };

        for poly in polys {
            for lane in 0..Lanes::NUM_LANES {
                let values = lane_values::<C, Lanes, Eval, D>(poly, lane)?;
                let commitment = if ck.blinding_mode.is_blinded() {
                    let blind = MsmCommitmentEngine::<C>::blind(&ck.msm_ck, values.len());
                    let commitment = MsmCommitmentEngine::<C>::commit_with::<_, Lanes::Strategy>(
                        &ck.msm_ck, &values, &blind,
                    )
                    .map_err(msm_err)?;
                    all_blinds.extend(blind.blind);
                    commitment
                } else {
                    MsmCommitmentEngine::<C>::commit_unblinded_with::<_, Lanes::Strategy>(
                        &ck.msm_ck, &values,
                    )
                    .map_err(msm_err)?
                };
                all_comm.extend(commitment.comm);
            }
        }

        Ok((
            HyraxProverData {
                batch_size: polys.len(),
                num_lanes: Lanes::NUM_LANES,
                num_rows,
                blinding_mode: ck.blinding_mode,
                blinds: all_blinds,
            },
            HyraxCommitment {
                batch_size: polys.len(),
                num_lanes: Lanes::NUM_LANES,
                num_rows,
                blinding_mode: ck.blinding_mode,
                comm: all_comm,
            },
        ))
    }

    fn absorb_commitment<T: Transcript>(transcript: &mut T, commitment: &Self::Commitment) {
        transcript.absorb_slice(b"hyrax_commitment_begin");
        transcript.absorb_slice(&(commitment.batch_size as u64).to_le_bytes());
        transcript.absorb_slice(&(commitment.num_lanes as u64).to_le_bytes());
        transcript.absorb_slice(&(commitment.num_rows as u64).to_le_bytes());
        transcript.absorb_slice(&[commitment.blinding_mode.as_u8()]);
        for comm in &commitment.comm {
            let bytes = group_bytes::<C>(comm).unwrap_or_default();
            transcript.absorb_slice(&bytes);
        }
        transcript.absorb_slice(b"hyrax_commitment_end");
    }

    fn commitment_num_bytes(commitment: &Self::Commitment) -> usize {
        let group_size = C::zero().serialized_size(Compress::Yes);
        3 * core::mem::size_of::<u64>() + 1 + commitment.comm.len() * group_size
    }

    fn write_commitment_bytes(commitment: &Self::Commitment, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(commitment.batch_size as u64).to_le_bytes());
        buf.extend_from_slice(&(commitment.num_lanes as u64).to_le_bytes());
        buf.extend_from_slice(&(commitment.num_rows as u64).to_le_bytes());
        buf.push(commitment.blinding_mode.as_u8());
        for comm in &commitment.comm {
            let bytes = group_bytes::<C>(comm).expect("Hyrax commitment must serialize");
            buf.extend_from_slice(&bytes);
        }
    }

    fn batch_size(commitment: &Self::Commitment) -> usize {
        commitment.batch_size
    }

    fn prove_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        ck: &Self::CommitmentKey,
        polys: &[DenseMultilinearExtension<Eval>],
        point: &[F],
        prover_data: &Self::ProverData,
        field_cfg: &F::Config,
    ) -> Result<Self::OpeningProof, ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let _ = CHECK_FOR_OVERFLOW;
        let start = transcript.stream.position() as usize;
        if polys.is_empty() {
            return Ok(Vec::new());
        }
        validate_polys(polys)?;
        validate_hyrax_shape::<C, Lanes, Eval, D>(
            ck.num_cols,
            ck.blinding_mode,
            polys,
            prover_data,
        )?;

        let n = polys[0].evaluations.len();
        if n != (1usize << point.len()) {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax open expected point for {n} evals, got {} variables",
                point.len()
            )));
        }

        let point_scalar = point.iter().map(F::field_to_scalar).collect::<Vec<_>>();
        let row_vars = prover_data.num_rows.ilog2() as usize;
        let q0_f = eq_tensor_f::<F>(&point[..row_vars], field_cfg);
        let q1_scalar = eq_tensor_scalar::<C>(&point_scalar[row_vars..]);
        let alphas = sample_scalars::<C>(
            &mut transcript.fs_transcript,
            polys.len() * Lanes::NUM_LANES,
        );

        let mut b_scalar = vec![C::ScalarField::zero(); prover_data.num_rows];
        for (poly_idx, poly) in polys.iter().enumerate() {
            for lane in 0..Lanes::NUM_LANES {
                let alpha = alphas[alpha_index_dynamic(Lanes::NUM_LANES, poly_idx, lane)];
                for (row_idx, row) in poly.evaluations.chunks(ck.num_cols).enumerate() {
                    let mut row_eval = C::ScalarField::zero();
                    for (col_idx, eval) in row.iter().enumerate() {
                        let value = Lanes::lane_to_scalar(Lanes::lane_value(eval, lane)?);
                        if let Some(weight) = q1_scalar.get(col_idx) {
                            row_eval += value * weight;
                        }
                    }
                    b_scalar[row_idx] += alpha * row_eval;
                }
            }
        }

        let b_f = b_scalar
            .iter()
            .map(|value| F::scalar_to_field(value, field_cfg))
            .collect::<Vec<_>>();
        transcript.write_field_elements(&b_f)?;

        let row_coeffs = if prover_data.num_rows == 1 {
            vec![C::ScalarField::from(1u64)]
        } else {
            sample_scalars::<C>(&mut transcript.fs_transcript, prover_data.num_rows)
        };

        let mut combined_row = vec![C::ScalarField::zero(); ck.num_cols];
        let mut rho_star = C::ScalarField::zero();
        for (poly_idx, poly) in polys.iter().enumerate() {
            for lane in 0..Lanes::NUM_LANES {
                let alpha = alphas[alpha_index_dynamic(Lanes::NUM_LANES, poly_idx, lane)];
                for (row_idx, row) in poly.evaluations.chunks(ck.num_cols).enumerate() {
                    let coeff = alpha * row_coeffs[row_idx];
                    if ck.blinding_mode.is_blinded() {
                        let blind_idx = commitment_index_dynamic(
                            Lanes::NUM_LANES,
                            poly_idx,
                            lane,
                            row_idx,
                            prover_data.num_rows,
                        );
                        rho_star += coeff * prover_data.blinds[blind_idx];
                    }
                    for (col_idx, eval) in row.iter().enumerate() {
                        let value = Lanes::lane_to_scalar(Lanes::lane_value(eval, lane)?);
                        combined_row[col_idx] += coeff * value;
                    }
                }
            }
        }

        write_scalars::<C>(transcript, &combined_row)?;
        if ck.blinding_mode.is_blinded() {
            write_scalar::<C>(transcript, &rho_star)?;
        }

        if q0_f.len() != b_f.len() {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax b vector shape mismatch".to_string(),
            ));
        }

        let end = transcript.stream.position() as usize;
        Ok(transcript.stream.get_ref()[start..end].to_vec())
    }

    fn verify_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        lifted_evals: &[DynamicPolynomialF<F>],
        opening_proof: &Self::OpeningProof,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let _ = CHECK_FOR_OVERFLOW;
        if !opening_proof.is_empty() {
            let original_stream =
                std::mem::replace(&mut transcript.stream, Cursor::new(opening_proof.clone()));
            let result = <Self as PCS<F, Eval, D>>::verify_open::<CHECK_FOR_OVERFLOW>(
                transcript,
                vk,
                commitment,
                point,
                lifted_evals,
                &Vec::new(),
                field_cfg,
            );
            let consumed = transcript.stream.position() == opening_proof.len() as u64;
            transcript.stream = original_stream;
            result?;
            if !consumed {
                return Err(ZipError::InvalidPcsOpen(
                    "PCS opening proof has trailing bytes".to_string(),
                ));
            }
            return Ok(());
        }

        if commitment.batch_size == 0 {
            return Ok(());
        }
        if commitment.blinding_mode != vk.blinding_mode {
            return Err(ZipError::InvalidPcsParam(
                "Hyrax commitment blinding mode mismatch".to_string(),
            ));
        }
        validate_commitment_shape::<C, Lanes, Eval, D>(commitment)?;
        if lifted_evals.len() != commitment.batch_size {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax verifier expected {} lifted evals, got {}",
                commitment.batch_size,
                lifted_evals.len()
            )));
        }

        let n = 1usize << point.len();
        let expected_rows = num_rows(n, vk.num_cols)?;
        if expected_rows != commitment.num_rows {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax verifier expected {expected_rows} rows from point, commitment has {}",
                commitment.num_rows
            )));
        }

        let row_vars = commitment.num_rows.ilog2() as usize;
        let q0_f = eq_tensor_f::<F>(&point[..row_vars], field_cfg);
        let point_scalar = point.iter().map(F::field_to_scalar).collect::<Vec<_>>();
        let q1_scalar = eq_tensor_scalar::<C>(&point_scalar[row_vars..]);
        let alphas = sample_scalars::<C>(
            &mut transcript.fs_transcript,
            commitment.batch_size * commitment.num_lanes,
        );

        let b_f = transcript.read_field_elements::<F>(commitment.num_rows)?;
        if b_f.len() != q0_f.len() {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax b vector shape mismatch".to_string(),
            ));
        }

        let mut expected_eval = F::zero_with_cfg(field_cfg);
        for (poly_idx, lifted_eval) in lifted_evals.iter().enumerate() {
            for lane in 0..commitment.num_lanes {
                let alpha = F::scalar_to_field(
                    &alphas[alpha_index_dynamic(commitment.num_lanes, poly_idx, lane)],
                    field_cfg,
                );
                let mut term = Lanes::lifted_eval::<F>(lifted_eval, lane, field_cfg)?;
                term *= &alpha;
                expected_eval += &term;
            }
        }

        let mut b_eval = F::zero_with_cfg(field_cfg);
        for (weight, b) in q0_f.iter().zip(b_f.iter()) {
            let mut term = weight.clone();
            term *= b;
            b_eval += &term;
        }
        if b_eval != expected_eval {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax evaluation consistency failure".to_string(),
            ));
        }

        let b_scalar = b_f.iter().map(F::field_to_scalar).collect::<Vec<_>>();
        let row_coeffs = if commitment.num_rows == 1 {
            vec![C::ScalarField::from(1u64)]
        } else {
            sample_scalars::<C>(&mut transcript.fs_transcript, commitment.num_rows)
        };

        let combined_row = read_scalars::<C>(transcript, vk.num_cols)?;
        let rho_star = if vk.blinding_mode.is_blinded() {
            Some(read_scalar::<C>(transcript)?)
        } else {
            None
        };

        let mut lhs = C::ScalarField::zero();
        for (value, weight) in combined_row.iter().zip(q1_scalar.iter()) {
            lhs += *value * weight;
        }
        let mut rhs = C::ScalarField::zero();
        for (coeff, b) in row_coeffs.iter().zip(b_scalar.iter()) {
            rhs += *coeff * b;
        }
        if lhs != rhs {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax row coherence failure".to_string(),
            ));
        }

        let mut comm_lc = C::Group::zero();
        for poly_idx in 0..commitment.batch_size {
            for lane in 0..commitment.num_lanes {
                let alpha = alphas[alpha_index_dynamic(commitment.num_lanes, poly_idx, lane)];
                for (row_idx, row_coeff) in row_coeffs.iter().enumerate() {
                    let idx = commitment_index_dynamic(
                        commitment.num_lanes,
                        poly_idx,
                        lane,
                        row_idx,
                        commitment.num_rows,
                    );
                    comm_lc += commitment.comm[idx] * (alpha * row_coeff);
                }
            }
        }

        let msm_ck = msm_key(vk.num_cols, vk.bases.clone(), vk.h)?;
        let mut expected = <ScalarPippengerMsm as RowMsmStrategy<C, C::ScalarField>>::msm_row(
            &msm_ck,
            &combined_row,
        )
        .map_err(msm_err)?;
        if let Some(rho_star) = rho_star {
            expected += vk.h * rho_star;
        }

        if comm_lc != expected {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax commitment opening failure".to_string(),
            ));
        }

        Ok(())
    }
}

impl<F, C, Lanes, Eval, const D: usize> FoldablePCS<F, Eval, D> for HyraxPCS<C, Lanes>
where
    F: HyraxFieldBridge<C>,
    C: AffineRepr,
    Eval: Clone + Debug + Send + Sync,
    Lanes: HyraxLanes<C, Eval, D>,
{
    fn fold_commitments(
        commitments: &[Self::Commitment],
        theta: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self::Commitment, ZipError> {
        let _ = field_cfg;
        validate_fold_inputs(commitments, theta.len(), "commitments")?;
        let first = &commitments[0];
        validate_commitment_shape::<C, Lanes, Eval, D>(first)?;

        let mut folded = vec![C::Group::zero(); first.comm.len()];
        for (commitment, weight) in commitments.iter().zip(theta) {
            validate_commitment_shape::<C, Lanes, Eval, D>(commitment)?;
            if !same_commitment_shape(first, commitment) {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax commitment fold shape mismatch".to_string(),
                ));
            }
            let scalar = F::field_to_scalar(weight);
            for (out, value) in folded.iter_mut().zip(commitment.comm.iter()) {
                *out += value.clone() * scalar;
            }
        }

        Ok(HyraxCommitment {
            batch_size: first.batch_size,
            num_lanes: first.num_lanes,
            num_rows: first.num_rows,
            blinding_mode: first.blinding_mode,
            comm: folded,
        })
    }

    fn fold_prover_data(
        prover_data: &[Self::ProverData],
        theta: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self::ProverData, ZipError> {
        let _ = field_cfg;
        validate_fold_inputs(prover_data, theta.len(), "prover data")?;
        let first = &prover_data[0];

        let mut folded_blinds = vec![C::ScalarField::zero(); first.blinds.len()];
        for (data, weight) in prover_data.iter().zip(theta) {
            if !same_prover_data_shape(first, data) {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax prover-data fold shape mismatch".to_string(),
                ));
            }
            let scalar = F::field_to_scalar(weight);
            for (out, blind) in folded_blinds.iter_mut().zip(data.blinds.iter()) {
                *out += *blind * scalar;
            }
        }

        Ok(HyraxProverData {
            batch_size: first.batch_size,
            num_lanes: first.num_lanes,
            num_rows: first.num_rows,
            blinding_mode: first.blinding_mode,
            blinds: folded_blinds,
        })
    }
}

fn validate_polys<Eval: Clone>(polys: &[DenseMultilinearExtension<Eval>]) -> Result<(), ZipError> {
    if let Some(first) = polys.first() {
        for poly in polys {
            if poly.num_vars != first.num_vars || poly.evaluations.len() != first.evaluations.len()
            {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax batch polynomial shape mismatch".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_scalar_lanes<C>(
    ck: &HyraxCommitmentKey<C>,
    scalar_lanes: &[Vec<Vec<C::ScalarField>>],
    point_len: usize,
    prover_data: &HyraxProverData<C>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
{
    let expected_n = 1usize
        .checked_shl(u32::try_from(point_len).map_err(|_| {
            ZipError::InvalidPcsParam(format!("Hyrax point length {point_len} is too large"))
        })?)
        .ok_or_else(|| {
            ZipError::InvalidPcsParam(format!("Hyrax point length {point_len} is too large"))
        })?;
    let expected_rows = num_rows(expected_n, ck.num_cols)?;
    if prover_data.batch_size != scalar_lanes.len()
        || prover_data.num_rows != expected_rows
        || prover_data.blinding_mode != ck.blinding_mode
    {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax scalar-lane prover data shape mismatch".to_string(),
        ));
    }
    let expected_blinds = if ck.blinding_mode.is_blinded() {
        prover_data.batch_size * prover_data.num_lanes * prover_data.num_rows
    } else {
        0
    };
    if prover_data.blinds.len() != expected_blinds {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax scalar-lane blind count mismatch".to_string(),
        ));
    }
    for lanes in scalar_lanes {
        if lanes.len() != prover_data.num_lanes {
            return Err(ZipError::InvalidPcsParam(
                "Hyrax scalar-lane count mismatch".to_string(),
            ));
        }
        for values in lanes {
            if values.len() != expected_n {
                return Err(ZipError::InvalidPcsParam(format!(
                    "Hyrax scalar-lane length mismatch: got {}, expected {expected_n}",
                    values.len()
                )));
            }
        }
    }
    Ok(())
}

fn same_commitment_shape<C: AffineRepr>(
    lhs: &HyraxCommitment<C>,
    rhs: &HyraxCommitment<C>,
) -> bool {
    lhs.batch_size == rhs.batch_size
        && lhs.num_lanes == rhs.num_lanes
        && lhs.num_rows == rhs.num_rows
        && lhs.blinding_mode == rhs.blinding_mode
        && lhs.comm.len() == rhs.comm.len()
}

fn same_prover_data_shape<C: AffineRepr>(
    lhs: &HyraxProverData<C>,
    rhs: &HyraxProverData<C>,
) -> bool {
    lhs.batch_size == rhs.batch_size
        && lhs.num_lanes == rhs.num_lanes
        && lhs.num_rows == rhs.num_rows
        && lhs.blinding_mode == rhs.blinding_mode
        && lhs.blinds.len() == rhs.blinds.len()
}

fn validate_hyrax_shape<C, Lanes, Eval, const D: usize>(
    width: usize,
    blinding_mode: HyraxBlindingMode,
    polys: &[DenseMultilinearExtension<Eval>],
    prover_data: &HyraxProverData<C>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
    Lanes: HyraxLanes<C, Eval, D>,
    Eval: Clone + Debug + Send + Sync,
{
    let n = polys[0].evaluations.len();
    let num_rows = num_rows(n, width)?;
    let expected_blinds = if blinding_mode.is_blinded() {
        polys.len() * Lanes::NUM_LANES * num_rows
    } else {
        0
    };
    if prover_data.batch_size != polys.len()
        || prover_data.num_lanes != Lanes::NUM_LANES
        || prover_data.num_rows != num_rows
        || prover_data.blinding_mode != blinding_mode
        || prover_data.blinds.len() != expected_blinds
    {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax prover data shape mismatch".to_string(),
        ));
    }
    Ok(())
}

fn validate_commitment_shape<C, Lanes, Eval, const D: usize>(
    commitment: &HyraxCommitment<C>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
    Lanes: HyraxLanes<C, Eval, D>,
    Eval: Clone + Debug + Send + Sync,
{
    if commitment.num_lanes != Lanes::NUM_LANES {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax commitment lane mismatch: expected {}, got {}",
            Lanes::NUM_LANES,
            commitment.num_lanes
        )));
    }
    let expected = commitment.batch_size * commitment.num_lanes * commitment.num_rows;
    if commitment.comm.len() != expected {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax commitment expected {expected} row commitments, got {}",
            commitment.comm.len()
        )));
    }
    Ok(())
}

fn lane_values<C, Lanes, Eval, const D: usize>(
    poly: &DenseMultilinearExtension<Eval>,
    lane: usize,
) -> Result<Vec<Lanes::LaneValue>, ZipError>
where
    C: AffineRepr,
    Lanes: HyraxLanes<C, Eval, D>,
    Eval: Clone + Debug + Send + Sync,
{
    poly.evaluations
        .iter()
        .map(|eval| Lanes::lane_value(eval, lane))
        .collect()
}

fn int_to_scalar<C: AffineRepr, const LIMBS: usize>(
    value: &Int<LIMBS>,
) -> Result<C::ScalarField, ZipError> {
    let (abs, is_negative) = if value.is_negative() {
        (
            value.checked_abs().ok_or_else(|| {
                ZipError::InvalidPcsParam("cannot convert minimum Int to scalar".to_string())
            })?,
            true,
        )
    } else {
        (*value, false)
    };
    let mut scalar = unsigned_int_to_scalar::<C, LIMBS>(&abs);
    if is_negative && !scalar.is_zero() {
        scalar = -scalar;
    }
    Ok(scalar)
}

fn unsigned_int_to_scalar<C: AffineRepr, const LIMBS: usize>(value: &Int<LIMBS>) -> C::ScalarField {
    let mut bytes = Vec::with_capacity(LIMBS * core::mem::size_of::<crypto_bigint::Word>());
    for word in value.as_uint().as_words() {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    C::ScalarField::from_le_bytes_mod_order(&bytes)
}

fn assert_curve_scalar_modulus<C, const LIMBS: usize>(actual: &Uint<LIMBS>)
where
    C: AffineRepr,
{
    let expected =
        uint_from_le_bytes::<LIMBS>(&<C::ScalarField as ArkPrimeField>::MODULUS.to_bytes_le());
    assert_eq!(
        actual, &expected,
        "Hyrax field mismatch: protocol field modulus must equal curve scalar modulus",
    );
}

fn assert_curve_scalar_modulus_boxed<C>(actual: &BoxedUint)
where
    C: AffineRepr,
{
    let expected = BoxedUint::from_le_slice(
        &<C::ScalarField as ArkPrimeField>::MODULUS.to_bytes_le(),
        actual.bits_precision(),
    )
    .expect("curve scalar modulus must fit protocol field precision");
    assert_eq!(
        actual, &expected,
        "Hyrax field mismatch: protocol field modulus must equal curve scalar modulus",
    );
}

fn uint_from_le_bytes<const LIMBS: usize>(bytes: &[u8]) -> Uint<LIMBS> {
    let num_bytes = <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES;
    assert!(
        bytes.len() <= num_bytes,
        "integer encoding does not fit in target Uint",
    );
    let mut padded = vec![0u8; num_bytes];
    padded[..bytes.len()].copy_from_slice(bytes);
    Uint::<LIMBS>::read_transcription_bytes_exact(&padded)
}

fn msm_key<C: AffineRepr>(
    width: usize,
    bases: Vec<C>,
    h: C::Group,
) -> Result<MsmCommitmentKey<C>, ZipError> {
    MsmCommitmentEngine::<C>::setup_from_bases(width, bases, h)
        .map(|(ck, _)| ck)
        .map_err(msm_err)
}

fn num_rows(n: usize, width: usize) -> Result<usize, ZipError> {
    if width == 0 {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax row width must be non-zero".to_string(),
        ));
    }
    Ok(<usize as Integer>::div_ceil(&n, &width))
}

fn alpha_index_dynamic(num_lanes: usize, poly_idx: usize, lane: usize) -> usize {
    poly_idx * num_lanes + lane
}

fn commitment_index_dynamic(
    num_lanes: usize,
    poly_idx: usize,
    lane: usize,
    row_idx: usize,
    num_rows: usize,
) -> usize {
    ((poly_idx * num_lanes + lane) * num_rows) + row_idx
}

fn eq_tensor_f<F: PrimeField>(point: &[F], cfg: &F::Config) -> Vec<F> {
    let mut tensor = vec![F::one_with_cfg(cfg)];
    for r in point {
        let one_minus = {
            let mut value = F::one_with_cfg(cfg);
            value -= r;
            value
        };
        let current = tensor.clone();
        tensor.clear();
        for value in &current {
            let mut lo = value.clone();
            lo *= &one_minus;
            tensor.push(lo);
        }
        for value in current {
            let mut hi = value;
            hi *= r;
            tensor.push(hi);
        }
    }
    tensor
}

fn eq_tensor_scalar<C: AffineRepr>(point: &[C::ScalarField]) -> Vec<C::ScalarField> {
    let mut tensor = vec![C::ScalarField::from(1u64)];
    for r in point {
        let one_minus = C::ScalarField::from(1u64) - r;
        let current = tensor.clone();
        tensor.clear();
        for value in &current {
            tensor.push(*value * one_minus);
        }
        for value in current {
            tensor.push(value * r);
        }
    }
    tensor
}

fn sample_scalars<C: AffineRepr>(
    transcript: &mut impl Transcript,
    n: usize,
) -> Vec<C::ScalarField> {
    (0..n)
        .map(|_| {
            let mut bytes = Vec::with_capacity(64);
            for _ in 0..8 {
                let word = transcript.get_challenge::<u64>();
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            C::ScalarField::from_le_bytes_mod_order(&bytes)
        })
        .collect()
}

fn write_scalars<C: AffineRepr>(
    transcript: &mut PcsProverTranscript,
    scalars: &[C::ScalarField],
) -> Result<(), ZipError> {
    for scalar in scalars {
        write_scalar::<C>(transcript, scalar)?;
    }
    Ok(())
}

fn write_scalar<C: AffineRepr>(
    transcript: &mut PcsProverTranscript,
    scalar: &C::ScalarField,
) -> Result<(), ZipError> {
    let bytes = scalar_bytes::<C>(scalar)?;
    transcript.fs_transcript.absorb_slice(&bytes);
    transcript.stream.write_all(&bytes)?;
    Ok(())
}

fn read_scalars<C: AffineRepr>(
    transcript: &mut PcsVerifierTranscript,
    n: usize,
) -> Result<Vec<C::ScalarField>, ZipError> {
    (0..n).map(|_| read_scalar::<C>(transcript)).collect()
}

fn read_scalar<C: AffineRepr>(
    transcript: &mut PcsVerifierTranscript,
) -> Result<C::ScalarField, ZipError> {
    let size = C::ScalarField::zero().serialized_size(Compress::Yes);
    let mut bytes = vec![0u8; size];
    transcript.stream.read_exact(&mut bytes)?;
    transcript.fs_transcript.absorb_slice(&bytes);
    C::ScalarField::deserialize_compressed(bytes.as_slice()).map_err(ark_err)
}

fn scalar_bytes<C: AffineRepr>(scalar: &C::ScalarField) -> Result<Vec<u8>, ZipError> {
    let mut bytes = Vec::with_capacity(scalar.serialized_size(Compress::Yes));
    scalar.serialize_compressed(&mut bytes).map_err(ark_err)?;
    Ok(bytes)
}

fn group_bytes<C: AffineRepr>(group: &C::Group) -> Result<Vec<u8>, ZipError> {
    let affine = group.into_affine();
    let mut bytes = Vec::with_capacity(affine.serialized_size(Compress::Yes));
    affine.serialize_compressed(&mut bytes).map_err(ark_err)?;
    Ok(bytes)
}

fn msm_err(err: MsmError) -> ZipError {
    ZipError::InvalidPcsParam(err.to_string())
}

fn ark_err(err: ark_serialize::SerializationError) -> ZipError {
    ZipError::Serialization(format!("ark serialization error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_ec::PrimeGroup;
    use ark_ff::Field as ArkField;
    use crypto_primitives::FromWithConfig;

    fn cfg_from_curve<C: AffineRepr>() -> <MontyField<4> as PrimeField>::Config {
        let modulus =
            uint_from_le_bytes::<4>(&<C::ScalarField as ArkPrimeField>::MODULUS.to_bytes_le());
        <MontyField<4> as PrimeField>::make_cfg(&modulus)
            .expect("curve scalar modulus must be prime")
    }

    fn assert_bridge_round_trip<C: AffineRepr>() {
        let cfg = cfg_from_curve::<C>();
        for value in [0u64, 1, 2, 17, 123, 1 << 20] {
            let field = MontyField::<4>::from_with_cfg(value, &cfg);
            let scalar = <MontyField<4> as HyraxFieldBridge<C>>::field_to_scalar(&field);
            assert_eq!(scalar, C::ScalarField::from(value));

            let field_again =
                <MontyField<4> as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg);
            assert_eq!(field_again, field);
        }

        let large_values = [
            C::ScalarField::from(2u64).inverse().unwrap(),
            -C::ScalarField::from(1u64),
            C::ScalarField::from_le_bytes_mod_order(&[0xA5; 64]),
        ];
        for scalar in large_values {
            let field = <MontyField<4> as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg);
            let scalar_again = <MontyField<4> as HyraxFieldBridge<C>>::field_to_scalar(&field);
            assert_eq!(scalar_again, scalar);
        }
    }

    #[test]
    fn bridge_round_trips_bn254_scalar_field() {
        assert_bridge_round_trip::<ark_bn254::G1Affine>();
    }

    #[test]
    fn bridge_round_trips_secp256k1_scalar_field() {
        assert_bridge_round_trip::<ark_secp256k1::Affine>();
    }

    #[test]
    #[should_panic(expected = "Hyrax field mismatch")]
    fn bridge_rejects_mismatched_field_config() {
        let bn_cfg = cfg_from_curve::<ark_bn254::G1Affine>();
        let bn_field = MontyField::<4>::from_with_cfg(1u64, &bn_cfg);
        let _ =
            <MontyField<4> as HyraxFieldBridge<ark_secp256k1::Affine>>::field_to_scalar(&bn_field);
    }

    fn binary_hyrax_open_verify_round_trip_with_modes(
        commit_mode: HyraxBlindingMode,
        verify_mode: HyraxBlindingMode,
    ) -> Result<(), ZipError> {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        fn bp(bits: u32) -> BinaryPoly<D> {
            BinaryPoly::<D>::from(bits)
        }

        let cfg = cfg_from_curve::<C>();
        let width = 512;
        let generator = <C as AffineRepr>::Group::generator();
        let bases = (1..=width)
            .map(|idx| (generator * <C as AffineRepr>::ScalarField::from(idx as u64)).into_affine())
            .collect::<Vec<_>>();
        let h = generator * <C as AffineRepr>::ScalarField::from((width + 1) as u64);
        let (ck, _) = HyraxPCS::<C, BinaryLanes>::setup_from_bases_with_blinding(
            width,
            bases.clone(),
            h,
            commit_mode,
        )?;
        let (_, vk) = HyraxPCS::<C, BinaryLanes>::setup_from_bases_with_blinding(
            width,
            bases,
            h,
            verify_mode,
        )?;

        let evals0 = (0..width)
            .map(|idx| bp((idx as u32).wrapping_mul(0x9E37_79B1)))
            .collect::<Vec<_>>();
        let evals1 = (0..width)
            .map(|idx| bp(!((idx as u32).wrapping_mul(0x85EB_CA6B))))
            .collect::<Vec<_>>();
        let polys = vec![
            DenseMultilinearExtension::from_evaluations_vec(9, evals0, bp(0)),
            DenseMultilinearExtension::from_evaluations_vec(9, evals1, bp(0)),
        ];
        let (prover_data, commitment) =
            <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, &polys)?;

        let point = [
            [0x11u8; 64],
            [0x22u8; 64],
            [0x33u8; 64],
            [0x44u8; 64],
            [0x55u8; 64],
            [0x66u8; 64],
            [0x77u8; 64],
            [0x88u8; 64],
            [0xA5u8; 64],
        ]
        .iter()
        .map(|bytes| {
            let scalar = <C as AffineRepr>::ScalarField::from_le_bytes_mod_order(bytes);
            <F as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg)
        })
        .collect::<Vec<_>>();
        let eq = eq_tensor_f::<F>(&point, &cfg);
        let lifted_evals = polys
            .iter()
            .map(|poly| {
                let mut coeffs = vec![F::zero_with_cfg(&cfg); D];
                for (weight, eval) in eq.iter().zip(poly.evaluations.iter()) {
                    for (lane, bit) in eval.iter().enumerate() {
                        if bit.inner() {
                            coeffs[lane] += weight;
                        }
                    }
                }
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect::<Vec<_>>();

        let mut prover_transcript = PcsProverTranscript {
            fs_transcript: Default::default(),
            stream: Default::default(),
        };
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut prover_transcript.fs_transcript,
            &commitment,
        );
        let mut transcription_buf = vec![0u8; <F as crypto_primitives::Field>::Inner::NUM_BYTES];
        for lifted_eval in &lifted_evals {
            prover_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted_eval.coeffs, &mut transcription_buf);
        }
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::prove_open::<true>(
            &mut prover_transcript,
            &ck,
            &polys,
            &point,
            &prover_data,
            &cfg,
        )?;

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut verifier_transcript.fs_transcript,
            &commitment,
        );
        let mut transcription_buf = vec![0u8; <F as crypto_primitives::Field>::Inner::NUM_BYTES];
        for lifted_eval in &lifted_evals {
            verifier_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted_eval.coeffs, &mut transcription_buf);
        }
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut verifier_transcript,
            &vk,
            &commitment,
            &point,
            &lifted_evals,
            &Vec::new(),
            &cfg,
        )
    }

    #[test]
    fn binary_hyrax_open_verify_round_trip() {
        binary_hyrax_open_verify_round_trip_with_modes(
            HyraxBlindingMode::Blinded,
            HyraxBlindingMode::Blinded,
        )
        .unwrap();
    }

    #[test]
    fn unblinded_binary_hyrax_open_verify_round_trip() {
        binary_hyrax_open_verify_round_trip_with_modes(
            HyraxBlindingMode::Unblinded,
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();
    }

    #[test]
    fn hyrax_rejects_blinding_mode_mismatch() {
        let result = binary_hyrax_open_verify_round_trip_with_modes(
            HyraxBlindingMode::Unblinded,
            HyraxBlindingMode::Blinded,
        );
        assert!(result.is_err());
    }

    #[test]
    fn folded_binary_hyrax_commitment_opens_from_scalar_lanes() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        fn bp(bits: u32) -> BinaryPoly<D> {
            BinaryPoly::<D>::from(bits)
        }

        let cfg = cfg_from_curve::<C>();
        let n = 8;
        let width = n;
        let generator = <C as AffineRepr>::Group::generator();
        let bases = (1..=width)
            .map(|idx| (generator * <C as AffineRepr>::ScalarField::from(idx as u64)).into_affine())
            .collect::<Vec<_>>();
        let h = generator * <C as AffineRepr>::ScalarField::from((width + 1) as u64);
        let (ck, vk) = HyraxPCS::<C, BinaryLanes>::setup_from_bases_with_blinding(
            width,
            bases,
            h,
            HyraxBlindingMode::Blinded,
        )
        .unwrap();

        let instance_polys = [
            vec![
                DenseMultilinearExtension::from_evaluations_vec(
                    3,
                    (0..n).map(|idx| bp((idx as u32) * 17 + 3)).collect(),
                    bp(0),
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    3,
                    (0..n).map(|idx| bp(!((idx as u32) * 11))).collect(),
                    bp(0),
                ),
            ],
            vec![
                DenseMultilinearExtension::from_evaluations_vec(
                    3,
                    (0..n).map(|idx| bp((idx as u32) * 23 + 9)).collect(),
                    bp(0),
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    3,
                    (0..n).map(|idx| bp(!((idx as u32) * 5 + 7))).collect(),
                    bp(0),
                ),
            ],
        ];

        let mut prover_data = Vec::new();
        let mut commitments = Vec::new();
        for polys in &instance_polys {
            let (data, commitment) =
                <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, polys).unwrap();
            prover_data.push(data);
            commitments.push(commitment);
        }

        let theta = [F::from_with_cfg(3u64, &cfg), F::from_with_cfg(5u64, &cfg)];
        let folded_commitment =
            <HyraxPCS<C, BinaryLanes> as FoldablePCS<F, BinaryPoly<D>, D>>::fold_commitments(
                &commitments,
                &theta,
                &cfg,
            )
            .unwrap();
        let folded_data =
            <HyraxPCS<C, BinaryLanes> as FoldablePCS<F, BinaryPoly<D>, D>>::fold_prover_data(
                &prover_data,
                &theta,
                &cfg,
            )
            .unwrap();

        let theta_scalar = theta
            .iter()
            .map(<F as HyraxFieldBridge<C>>::field_to_scalar)
            .collect::<Vec<_>>();
        let mut scalar_lanes =
            vec![vec![vec![<C as AffineRepr>::ScalarField::zero(); n]; D]; instance_polys[0].len()];
        for (instance_idx, polys) in instance_polys.iter().enumerate() {
            for (poly_idx, poly) in polys.iter().enumerate() {
                for (eval_idx, eval) in poly.evaluations.iter().enumerate() {
                    for (lane, bit) in eval.iter().enumerate() {
                        if bit.inner() {
                            scalar_lanes[poly_idx][lane][eval_idx] += theta_scalar[instance_idx];
                        }
                    }
                }
            }
        }

        let point = [[0x11u8; 64], [0x22u8; 64], [0x33u8; 64]]
            .iter()
            .map(|bytes| {
                let scalar = <C as AffineRepr>::ScalarField::from_le_bytes_mod_order(bytes);
                <F as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg)
            })
            .collect::<Vec<_>>();
        let eq = eq_tensor_f::<F>(&point, &cfg);
        let folded_lifted_evals = scalar_lanes
            .iter()
            .map(|lanes| {
                let coeffs = lanes
                    .iter()
                    .map(|values| {
                        values.iter().zip(eq.iter()).fold(
                            F::zero_with_cfg(&cfg),
                            |mut acc, (value, weight)| {
                                acc += <F as HyraxFieldBridge<C>>::scalar_to_field(value, &cfg)
                                    * weight;
                                acc
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect::<Vec<_>>();

        let mut prover_transcript = PcsProverTranscript {
            fs_transcript: Default::default(),
            stream: Default::default(),
        };
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut prover_transcript.fs_transcript,
            &folded_commitment,
        );
        let mut transcription_buf = vec![0u8; <F as crypto_primitives::Field>::Inner::NUM_BYTES];
        for lifted_eval in &folded_lifted_evals {
            prover_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted_eval.coeffs, &mut transcription_buf);
        }
        HyraxPCS::<C, BinaryLanes>::prove_open_scalar_lanes::<F, true>(
            &mut prover_transcript,
            &ck,
            &scalar_lanes,
            &point,
            &folded_data,
            &cfg,
        )
        .unwrap();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut verifier_transcript.fs_transcript,
            &folded_commitment,
        );
        let mut transcription_buf = vec![0u8; <F as crypto_primitives::Field>::Inner::NUM_BYTES];
        for lifted_eval in &folded_lifted_evals {
            verifier_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted_eval.coeffs, &mut transcription_buf);
        }
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut verifier_transcript,
            &vk,
            &folded_commitment,
            &point,
            &folded_lifted_evals,
            &Vec::new(),
            &cfg,
        )
        .unwrap();
    }

    #[test]
    fn hyrax_fold_rejects_commitment_shape_mismatch() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        let cfg = cfg_from_curve::<C>();
        let generator = <C as AffineRepr>::Group::generator();
        let bases = (1..=4)
            .map(|idx| (generator * <C as AffineRepr>::ScalarField::from(idx as u64)).into_affine())
            .collect::<Vec<_>>();
        let h = generator * <C as AffineRepr>::ScalarField::from(5u64);
        let (ck, _) = HyraxPCS::<C, BinaryLanes>::setup_from_bases_with_blinding(
            4,
            bases,
            h,
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();
        let polys_one = vec![DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![BinaryPoly::<D>::from(1u32); 4],
            BinaryPoly::<D>::from(0u32),
        )];
        let polys_two = vec![DenseMultilinearExtension::from_evaluations_vec(
            3,
            vec![BinaryPoly::<D>::from(2u32); 8],
            BinaryPoly::<D>::from(0u32),
        )];
        let (_, c0) =
            <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, &polys_one)
                .unwrap();
        let (_, c1) =
            <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, &polys_two)
                .unwrap();

        let theta = [F::from_with_cfg(1u64, &cfg), F::from_with_cfg(2u64, &cfg)];
        assert!(matches!(
            <HyraxPCS<C, BinaryLanes> as FoldablePCS<F, BinaryPoly<D>, D>>::fold_commitments(
                &[c0, c1],
                &theta,
                &cfg,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));
    }
}
