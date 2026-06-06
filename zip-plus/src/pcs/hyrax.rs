#![allow(clippy::arithmetic_side_effects)]

use std::{
    collections::HashSet,
    fmt::Debug,
    io::{Read, Write},
    marker::PhantomData,
};

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger, PrimeField as ArkPrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use crypto_primitives::{
    FromWithConfig, IntRing, PrimeField, crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
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
        generic::PCS,
        msm_commitment::{
            BoolSubsetMsm, MsmCommitmentEngine, MsmCommitmentKey, MsmError, RowMsmStrategy,
            ScalarPippengerMsm,
        },
    },
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};

#[derive(Clone, Debug)]
pub struct HyraxPCS<C: AffineRepr, Lanes>(PhantomData<(C, Lanes)>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyraxBlindingMode {
    Blinded,
    Unblinded,
}

impl Default for HyraxBlindingMode {
    fn default() -> Self {
        Self::Unblinded
    }
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
    fn field_to_scalar(value: &Self) -> Result<C::ScalarField, ZipError>;
    fn scalar_to_field(value: &C::ScalarField, cfg: &Self::Config) -> Result<Self, ZipError>;
}

impl<C, const LIMBS: usize> HyraxFieldBridge<C> for MontyField<LIMBS>
where
    C: AffineRepr,
{
    fn field_to_scalar(value: &Self) -> Result<C::ScalarField, ZipError> {
        validate_curve_scalar_modulus::<C, LIMBS>(&value.modulus())?;

        let canonical = value.retrieve();
        let mut bytes = vec![0u8; <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES];
        canonical.write_transcription_bytes_exact(&mut bytes);
        Ok(C::ScalarField::from_le_bytes_mod_order(&bytes))
    }

    fn scalar_to_field(value: &C::ScalarField, cfg: &Self::Config) -> Result<Self, ZipError> {
        let actual_modulus = Uint::<LIMBS>::new(cfg.modulus().get());
        validate_curve_scalar_modulus::<C, LIMBS>(&actual_modulus)?;

        let scalar_bigint: <C::ScalarField as ArkPrimeField>::BigInt = value.clone().into();
        let scalar_uint = uint_from_le_bytes::<LIMBS>(&scalar_bigint.to_bytes_le());
        Ok(MontyField::<LIMBS>::from_with_cfg(&scalar_uint, cfg))
    }
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

    fn commit_poly(
        _ck: &HyraxCommitmentKey<C>,
        _poly: &DenseMultilinearExtension<Eval>,
        _num_rows: usize,
    ) -> Option<Result<(Vec<C::Group>, Vec<C::ScalarField>), ZipError>> {
        None
    }

    fn accumulate_b(
        row: &[Eval],
        lane: usize,
        q1_scalar: &[C::ScalarField],
    ) -> Result<C::ScalarField, ZipError> {
        let mut row_eval = C::ScalarField::zero();
        for (col_idx, eval) in row.iter().enumerate() {
            let value = Self::lane_to_scalar(Self::lane_value(eval, lane)?);
            if let Some(weight) = q1_scalar.get(col_idx) {
                row_eval += value * weight;
            }
        }
        Ok(row_eval)
    }

    fn accumulate_combined_row(
        row: &[Eval],
        lane: usize,
        coeff: C::ScalarField,
        combined_row: &mut [C::ScalarField],
    ) -> Result<(), ZipError> {
        for (col_idx, eval) in row.iter().enumerate() {
            let value = Self::lane_to_scalar(Self::lane_value(eval, lane)?);
            combined_row[col_idx] += coeff * value;
        }
        Ok(())
    }

    fn accumulate_single_row_opening(
        row: &[Eval],
        lane: usize,
        alpha: C::ScalarField,
        q1_scalar: &[C::ScalarField],
        b_scalar: &mut C::ScalarField,
        combined_row: &mut [C::ScalarField],
    ) -> Result<(), ZipError> {
        for (col_idx, eval) in row.iter().enumerate() {
            let value = Self::lane_to_scalar(Self::lane_value(eval, lane)?);
            let scaled = alpha * value;
            if let Some(weight) = q1_scalar.get(col_idx) {
                *b_scalar += scaled * weight;
            }
            combined_row[col_idx] += scaled;
        }
        Ok(())
    }

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

    fn commit_poly(
        ck: &HyraxCommitmentKey<C>,
        poly: &DenseMultilinearExtension<BinaryPoly<D>>,
        num_rows: usize,
    ) -> Option<Result<(Vec<C::Group>, Vec<C::ScalarField>), ZipError>> {
        let expected_comm = <Self as HyraxLanes<C, BinaryPoly<D>, D>>::NUM_LANES * num_rows;
        let mut comm = Vec::with_capacity(expected_comm);
        let mut blinds = if ck.blinding_mode.is_blinded() {
            Vec::with_capacity(expected_comm)
        } else {
            Vec::new()
        };

        Some((|| {
            for lane in 0..<Self as HyraxLanes<C, BinaryPoly<D>, D>>::NUM_LANES {
                let lane_blinds = if ck.blinding_mode.is_blinded() {
                    Some(MsmCommitmentEngine::<C>::blind(
                        &ck.msm_ck,
                        poly.evaluations.len(),
                    ))
                } else {
                    None
                };

                for (row_idx, row) in poly.evaluations.chunks(ck.num_cols).enumerate() {
                    let values = row
                        .iter()
                        .map(|eval| {
                            <Self as HyraxLanes<C, BinaryPoly<D>, D>>::lane_value(eval, lane)
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let mut row_comm = if values.iter().copied().any(|bit| bit) {
                        <Self as HyraxLanes<C, BinaryPoly<D>, D>>::Strategy::msm_row(
                            &ck.msm_ck, &values,
                        )
                        .map_err(msm_err)?
                    } else {
                        C::Group::zero()
                    };

                    if let Some(lane_blinds) = lane_blinds.as_ref() {
                        row_comm += ck.msm_ck.h * lane_blinds.blind[row_idx];
                    }
                    comm.push(row_comm);
                }

                if let Some(lane_blinds) = lane_blinds {
                    blinds.extend(lane_blinds.blind);
                }
            }
            Ok((comm, blinds))
        })())
    }

    fn accumulate_b(
        row: &[BinaryPoly<D>],
        lane: usize,
        q1_scalar: &[C::ScalarField],
    ) -> Result<C::ScalarField, ZipError> {
        let mut row_eval = C::ScalarField::zero();
        for (col_idx, eval) in row.iter().enumerate() {
            if <Self as HyraxLanes<C, BinaryPoly<D>, D>>::lane_value(eval, lane)? {
                if let Some(weight) = q1_scalar.get(col_idx) {
                    row_eval += weight;
                }
            }
        }
        Ok(row_eval)
    }

    fn accumulate_combined_row(
        row: &[BinaryPoly<D>],
        lane: usize,
        coeff: C::ScalarField,
        combined_row: &mut [C::ScalarField],
    ) -> Result<(), ZipError> {
        for (col_idx, eval) in row.iter().enumerate() {
            if <Self as HyraxLanes<C, BinaryPoly<D>, D>>::lane_value(eval, lane)? {
                combined_row[col_idx] += coeff;
            }
        }
        Ok(())
    }

    fn accumulate_single_row_opening(
        row: &[BinaryPoly<D>],
        lane: usize,
        alpha: C::ScalarField,
        q1_scalar: &[C::ScalarField],
        b_scalar: &mut C::ScalarField,
        combined_row: &mut [C::ScalarField],
    ) -> Result<(), ZipError> {
        let mut row_eval = C::ScalarField::zero();
        for (col_idx, eval) in row.iter().enumerate() {
            if <Self as HyraxLanes<C, BinaryPoly<D>, D>>::lane_value(eval, lane)? {
                if let Some(weight) = q1_scalar.get(col_idx) {
                    row_eval += weight;
                }
                combined_row[col_idx] += alpha;
            }
        }
        *b_scalar += alpha * row_eval;
        Ok(())
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
    pub fn setup(
        width: usize,
        domain: impl AsRef<[u8]>,
        blinding_mode: HyraxBlindingMode,
    ) -> Result<(HyraxCommitmentKey<C>, HyraxVerifierKey<C>), ZipError> {
        let domain = domain.as_ref();
        let bases = (0..width)
            .map(|idx| hash_to_curve::<C>(domain, b"basis", idx))
            .collect::<Result<Vec<_>, _>>()?;
        let h = hash_to_curve::<C>(domain, b"blinding", 0)?.into_group();
        Self::setup_from_trusted_bases(width, bases, h, blinding_mode)
    }

    pub fn setup_from_trusted_bases(
        width: usize,
        bases: Vec<C>,
        h: C::Group,
        blinding_mode: HyraxBlindingMode,
    ) -> Result<(HyraxCommitmentKey<C>, HyraxVerifierKey<C>), ZipError> {
        validate_trusted_bases(width, &bases, &h)?;
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
            if let Some(result) = Lanes::commit_poly(ck, poly, num_rows) {
                let (comm, blinds) = result?;
                all_comm.extend(comm);
                all_blinds.extend(blinds);
                continue;
            }
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
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let _ = CHECK_FOR_OVERFLOW;
        if polys.is_empty() {
            if prover_data.batch_size != 0
                || prover_data.num_lanes != Lanes::NUM_LANES
                || prover_data.num_rows != 0
                || prover_data.blinding_mode != ck.blinding_mode
                || !prover_data.blinds.is_empty()
            {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax prover data must be canonical for an empty batch".to_string(),
                ));
            }
            return Ok(());
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

        let point_scalar = point
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        let row_vars = prover_data.num_rows.ilog2() as usize;
        let q0_f = eq_tensor_f::<F>(&point[..row_vars], field_cfg);
        let q1_scalar = eq_tensor_scalar::<C>(&point_scalar[row_vars..]);
        let alphas = sample_scalars::<C>(
            &mut transcript.fs_transcript,
            polys.len() * Lanes::NUM_LANES,
        );

        let mut combined_row = vec![C::ScalarField::zero(); ck.num_cols];
        let mut rho_star = C::ScalarField::zero();

        let mut b_scalar = vec![C::ScalarField::zero(); prover_data.num_rows];
        if prover_data.num_rows == 1 {
            for (poly_idx, poly) in polys.iter().enumerate() {
                for lane in 0..Lanes::NUM_LANES {
                    let alpha = alphas[alpha_index_dynamic(Lanes::NUM_LANES, poly_idx, lane)];
                    if ck.blinding_mode.is_blinded() {
                        let blind_idx = commitment_index_dynamic(
                            Lanes::NUM_LANES,
                            poly_idx,
                            lane,
                            0,
                            prover_data.num_rows,
                        );
                        rho_star += alpha * prover_data.blinds[blind_idx];
                    }
                    Lanes::accumulate_single_row_opening(
                        &poly.evaluations,
                        lane,
                        alpha,
                        &q1_scalar,
                        &mut b_scalar[0],
                        &mut combined_row,
                    )?;
                }
            }
        } else {
            for (poly_idx, poly) in polys.iter().enumerate() {
                for lane in 0..Lanes::NUM_LANES {
                    let alpha = alphas[alpha_index_dynamic(Lanes::NUM_LANES, poly_idx, lane)];
                    for (row_idx, row) in poly.evaluations.chunks(ck.num_cols).enumerate() {
                        let row_eval = Lanes::accumulate_b(row, lane, &q1_scalar)?;
                        b_scalar[row_idx] += alpha * row_eval;
                    }
                }
            }

            let b_f = b_scalar
                .iter()
                .map(|value| F::scalar_to_field(value, field_cfg))
                .collect::<Result<Vec<_>, _>>()?;
            transcript.write_field_elements(&b_f)?;

            let row_coeffs =
                sample_scalars::<C>(&mut transcript.fs_transcript, prover_data.num_rows);

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
                        Lanes::accumulate_combined_row(row, lane, coeff, &mut combined_row)?;
                    }
                }
            }
        }

        if prover_data.num_rows == 1 {
            let b_f = b_scalar
                .iter()
                .map(|value| F::scalar_to_field(value, field_cfg))
                .collect::<Result<Vec<_>, _>>()?;
            transcript.write_field_elements(&b_f)?;
        }

        write_scalars::<C>(transcript, &combined_row)?;
        if ck.blinding_mode.is_blinded() {
            write_scalar::<C>(transcript, &rho_star)?;
        }

        if q0_f.len() != b_scalar.len() {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax b vector shape mismatch".to_string(),
            ));
        }

        Ok(())
    }

    fn verify_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        lifted_evals: &[DynamicPolynomialF<F>],
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let _ = CHECK_FOR_OVERFLOW;
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
        if commitment.batch_size == 0 {
            if commitment.num_rows != 0 {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax empty batch must use the canonical empty commitment".to_string(),
                ));
            }
            return Ok(());
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
        let point_scalar = point
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
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
                )?;
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

        let b_scalar = b_f
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
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

        let mut comm_lc_scalars = Vec::with_capacity(commitment.comm.len());
        for poly_idx in 0..commitment.batch_size {
            for lane in 0..commitment.num_lanes {
                let alpha = alphas[alpha_index_dynamic(commitment.num_lanes, poly_idx, lane)];
                comm_lc_scalars.extend(row_coeffs.iter().map(|row_coeff| alpha * row_coeff));
            }
        }

        let comm_bases = C::Group::normalize_batch(&commitment.comm);
        let comm_lc = msm_unchecked::<C>(&comm_bases, &comm_lc_scalars)?;

        let mut expected = msm_unchecked::<C>(&vk.bases[..combined_row.len()], &combined_row)?;
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

fn validate_trusted_bases<C: AffineRepr>(
    width: usize,
    bases: &[C],
    h: &C::Group,
) -> Result<(), ZipError> {
    if !width.is_power_of_two() {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax row width must be a power of two, got {width}"
        )));
    }
    if bases.len() != width {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax expected {width} bases, got {}",
            bases.len()
        )));
    }

    let mut seen = HashSet::with_capacity(bases.len());
    for (idx, base) in bases.iter().copied().enumerate() {
        if base.is_zero() {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax base {idx} is the identity"
            )));
        }
        if !seen.insert(base) {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax base {idx} duplicates an earlier base"
            )));
        }
    }

    let h_affine = h.clone().into_affine();
    if h_affine.is_zero() {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax blinding base is the identity".to_string(),
        ));
    }
    if seen.contains(&h_affine) {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax blinding base duplicates a witness base".to_string(),
        ));
    }

    Ok(())
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

fn hash_to_curve<C: AffineRepr>(domain: &[u8], label: &[u8], index: usize) -> Result<C, ZipError> {
    let point_bytes = C::zero().serialized_size(Compress::Yes);
    let mut counter = 0u64;
    loop {
        let mut hasher = blake3::Hasher::new();
        absorb_hash_part(&mut hasher, b"zinc-plus-hyrax-setup-v1")?;
        absorb_hash_part(&mut hasher, domain)?;
        absorb_hash_part(&mut hasher, label)?;
        hasher.update(
            &u64::try_from(index)
                .map_err(|_| {
                    ZipError::InvalidPcsParam("Hyrax setup index does not fit u64".to_string())
                })?
                .to_le_bytes(),
        );
        hasher.update(&counter.to_le_bytes());

        let mut bytes = vec![0u8; point_bytes];
        hasher.finalize_xof().fill(&mut bytes);
        if let Some(point) = C::from_random_bytes(&bytes).map(|point| point.clear_cofactor()) {
            if !point.is_zero() {
                return Ok(point);
            }
        }

        counter = counter.checked_add(1).ok_or_else(|| {
            ZipError::InvalidPcsParam("Hyrax hash-to-curve setup exhausted counters".to_string())
        })?;
    }
}

fn absorb_hash_part(hasher: &mut blake3::Hasher, part: &[u8]) -> Result<(), ZipError> {
    hasher.update(
        &u64::try_from(part.len())
            .map_err(|_| {
                ZipError::InvalidPcsParam(
                    "Hyrax setup domain component length does not fit u64".to_string(),
                )
            })?
            .to_le_bytes(),
    );
    hasher.update(part);
    Ok(())
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

fn validate_curve_scalar_modulus<C, const LIMBS: usize>(
    actual: &Uint<LIMBS>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
{
    let expected =
        uint_from_le_bytes::<LIMBS>(&<C::ScalarField as ArkPrimeField>::MODULUS.to_bytes_le());
    if actual != &expected {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax field mismatch: protocol field modulus must equal curve scalar modulus"
                .to_string(),
        ));
    }
    Ok(())
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

fn msm_unchecked<C: AffineRepr>(
    bases: &[C],
    scalars: &[C::ScalarField],
) -> Result<C::Group, ZipError> {
    if bases.len() != scalars.len() {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax MSM expected {} bases, got {}",
            scalars.len(),
            bases.len()
        )));
    }
    Ok(<C::Group as VariableBaseMSM>::msm_unchecked(bases, scalars))
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

    fn assert_bridge_round_trip<C: AffineRepr>() -> Result<(), ZipError> {
        let cfg = cfg_from_curve::<C>();
        for value in [0u64, 1, 2, 17, 123, 1 << 20] {
            let field = MontyField::<4>::from_with_cfg(value, &cfg);
            let scalar = <MontyField<4> as HyraxFieldBridge<C>>::field_to_scalar(&field)?;
            assert_eq!(scalar, C::ScalarField::from(value));

            let field_again =
                <MontyField<4> as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg)?;
            assert_eq!(field_again, field);
        }

        let large_values = [
            C::ScalarField::from(2u64).inverse().unwrap(),
            -C::ScalarField::from(1u64),
            C::ScalarField::from_le_bytes_mod_order(&[0xA5; 64]),
        ];
        for scalar in large_values {
            let field = <MontyField<4> as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg)?;
            let scalar_again = <MontyField<4> as HyraxFieldBridge<C>>::field_to_scalar(&field)?;
            assert_eq!(scalar_again, scalar);
        }
        Ok(())
    }

    #[test]
    fn bridge_round_trips_bn254_scalar_field() {
        assert_bridge_round_trip::<ark_bn254::G1Affine>().unwrap();
    }

    #[test]
    fn bridge_round_trips_secp256k1_scalar_field() {
        assert_bridge_round_trip::<ark_secp256k1::Affine>().unwrap();
    }

    #[test]
    fn bridge_rejects_mismatched_field_config() {
        let bn_cfg = cfg_from_curve::<ark_bn254::G1Affine>();
        let bn_field = MontyField::<4>::from_with_cfg(1u64, &bn_cfg);
        let result =
            <MontyField<4> as HyraxFieldBridge<ark_secp256k1::Affine>>::field_to_scalar(&bn_field);
        assert!(matches!(result, Err(ZipError::InvalidPcsParam(_))));
    }

    #[test]
    fn setup_derives_distinct_deterministic_bases() {
        type C = ark_bn254::G1Affine;
        let width = 32;
        let (ck_0, vk_0) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-setup-test",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();
        let (ck_1, vk_1) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-setup-test",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();

        assert_eq!(ck_0.msm_ck.bases, ck_1.msm_ck.bases);
        assert_eq!(vk_0.bases, vk_1.bases);
        assert_eq!(ck_0.msm_ck.h, ck_1.msm_ck.h);
        assert_eq!(vk_0.h, vk_1.h);
        assert_eq!(ck_0.blinding_mode, HyraxBlindingMode::Unblinded);
        assert_eq!(vk_0.blinding_mode, HyraxBlindingMode::Unblinded);
        assert!(ck_0.msm_ck.bases.iter().all(|base| !base.is_zero()));
        assert!(!ck_0.msm_ck.h.is_zero());

        let seen = ck_0.msm_ck.bases.iter().copied().collect::<HashSet<_>>();
        assert_eq!(seen.len(), width);
        assert!(!seen.contains(&ck_0.msm_ck.h.into_affine()));
    }

    #[test]
    fn trusted_setup_rejects_bad_bases() {
        type C = ark_bn254::G1Affine;
        let width = 8;
        let generator = <C as AffineRepr>::Group::generator();
        let bases = (1..=width)
            .map(|idx| (generator * <C as AffineRepr>::ScalarField::from(idx as u64)).into_affine())
            .collect::<Vec<_>>();
        let h = generator * <C as AffineRepr>::ScalarField::from((width + 1) as u64);

        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                0,
                Vec::new(),
                <C as AffineRepr>::Group::zero(),
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                bases[..width - 1].to_vec(),
                h,
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        let mut identity_bases = bases.clone();
        identity_bases[0] = C::zero();
        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                identity_bases,
                h,
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        let mut duplicate_bases = bases.clone();
        duplicate_bases[1] = duplicate_bases[0];
        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                duplicate_bases,
                h,
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                bases.clone(),
                <C as AffineRepr>::Group::zero(),
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                bases.clone(),
                bases[0].into_group(),
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));
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
        let (ck, _) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-round-trip-test",
            commit_mode,
        )?;
        let (_, vk) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-round-trip-test",
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
        .collect::<Result<Vec<_>, _>>()?;
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
    fn hyrax_rejects_empty_commitment_with_nonempty_lifted_evals() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        let width = 8;
        let (_, vk) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-empty-reject-test",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();

        let commitment = HyraxCommitment::<C> {
            batch_size: 0,
            num_lanes: D,
            num_rows: 0,
            blinding_mode: HyraxBlindingMode::Unblinded,
            comm: Vec::new(),
        };
        let cfg = cfg_from_curve::<C>();
        let lifted_evals = vec![DynamicPolynomialF::new_trimmed(vec![F::zero_with_cfg(
            &cfg,
        )])];
        let mut verifier_transcript = PcsVerifierTranscript {
            fs_transcript: Default::default(),
            stream: Default::default(),
        };

        let result = <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut verifier_transcript,
            &vk,
            &commitment,
            &[],
            &lifted_evals,
            &cfg,
        );
        assert!(matches!(result, Err(ZipError::InvalidPcsParam(_))));
    }

    #[test]
    fn hyrax_rejects_noncanonical_empty_commitment() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        let width = 8;
        let (_, vk) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-empty-reject-test-2",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();

        let commitment = HyraxCommitment::<C> {
            batch_size: 0,
            num_lanes: D,
            num_rows: 1,
            blinding_mode: HyraxBlindingMode::Unblinded,
            comm: Vec::new(),
        };
        let cfg = cfg_from_curve::<C>();
        let mut verifier_transcript = PcsVerifierTranscript {
            fs_transcript: Default::default(),
            stream: Default::default(),
        };

        let result = <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut verifier_transcript,
            &vk,
            &commitment,
            &[],
            &[],
            &cfg,
        );
        assert!(matches!(result, Err(ZipError::InvalidPcsParam(_))));
    }
}
