//! Zinc+ PIOP for UCS - end-to-end protocol.
//!
//! Implements the Zinc+ compiler pipeline (cf. paper, Section "Zinc+
//! Compiler"):
//!
//! ```text
//! Z[X]  --\phi_q-->  F_q[X]  --MLE eval-->  F_q[X]  --\psi_a-->  F_q
//!         Step 1               Step 2                  Step 3
//! ```
//!
//! After the three compiler steps, the protocol continues with:
//!
//! - Step 4: combined CPR + Lookup multi-degree sumcheck (CPR group at degree
//!   `max_deg+2`, one lookup group per table type; shared eval point `r*`)
//! - Step 5: multi-point evaluation sumcheck (combines up/down evals at r* into
//!   a single evaluation point r_0)
//! - Step 6: lift-and-project (unprojected MLE evaluations at r_0)
//! - Step 7: Zip+ PCS open/verify at r_0

pub mod prover;
pub mod verifier;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{ConstIntRing, ConstIntSemiring, FromWithConfig, PrimeField, Semiring};
use std::marker::PhantomData;
use thiserror::Error;
use zinc_piop::{
    combined_poly_resolver::{CombinedPolyResolverError, Proof as CombinedPolyResolverProof},
    ideal_check::{IdealCheckError, Proof as IdealCheckProof},
    lookup::LookupError,
    multipoint_eval::{MultipointEvalError, Proof as MultipointEvalProof},
    projections::ProjectedTrace,
    sumcheck::multi_degree::MultiDegreeSumcheckProof,
};
use zinc_poly::{
    ConstCoeffBitWidth, EvaluationError as PolyEvaluationError,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly,
        dense::DensePolynomial,
        dynamic::over_field::{DynamicPolyVecF, DynamicPolynomialF},
    },
};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_uair::{Uair, ideal::Ideal};
use zinc_utils::{cfg_extend, cfg_into_iter, cfg_iter, named::Named};
use zip_plus::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlusCommitment, ZipTypes},
};

//
// Data structures
//

/// Lookup-specific proof data (commitments, evaluations, lifted evals).
///
/// Present only when the UAIR has lookup specs (`Proof::lookup` is `Some`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupProofData<F: PrimeField> {
    /// Zip+ commitments for lookup auxiliary columns.
    /// One pair `(comm_m, comm_u)` per lookup group.
    pub commitments: Vec<(ZipPlusCommitment, ZipPlusCommitment)>,
    /// Lookup auxiliary column evaluations at the shared sumcheck point r*.
    /// Order: per group `[m_0..m_{L-1}, u_0..u_{L-1}]`.
    pub aux_evals: Vec<F>,
    /// Lookup auxiliary column lifted MLE evaluations at r_0.
    /// Each is a `DynamicPolynomialF` (single field element).
    /// Same ordering as `aux_evals`.
    pub lifted_evals: Vec<DynamicPolynomialF<F>>,
    /// Zip+ commitments for decomposed chunk columns.
    /// One commitment per decomposed lookup group (batching L·K chunk MLEs).
    /// Empty when no lookup group uses decomposition.
    pub chunk_commitments: Vec<ZipPlusCommitment>,
    /// Chunk column evaluations at the shared sumcheck point r*.
    /// Order: per decomposed group, flattened `[chunk_{0,0}, ...,
    /// chunk_{L-1,K-1}]`.
    pub chunk_evals: Vec<F>,
    /// Chunk column lifted MLE evaluations at r_0 for PCS opening.
    /// Same ordering as `chunk_evals`.
    pub chunk_lifted_evals: Vec<DynamicPolynomialF<F>>,
}

/// Full proof produced by the Zinc+ PIOP for UCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    /// Zip+ commitments to the witness columns.
    pub commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip: Vec<u8>,
    /// Randomized ideal check proof.
    pub ideal_check: IdealCheckProof<F>,
    /// Combined polynomial resolver proof (up_evals + down_evals).
    pub resolver: CombinedPolyResolverProof<F>,
    /// Multi-degree sumcheck proof (CPR group + future lookup groups).
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Multi-point evaluation sumcheck proof (combines up_evals and
    /// down_evals at r' into a single evaluation point r_0).
    pub multipoint_eval: MultipointEvalProof<F>,
    /// Witness-only polynomial MLE evaluations at r_0 in F_q[X]
    /// (after \phi_q, before \psi_a), ordered as
    /// `[wit_bin..., wit_arb..., wit_int...]`.
    /// The verifier recomputes public lifted_evals from public data,
    /// interleaves them with these, and derives scalar open_evals via
    /// \psi_a for the sumcheck consistency check and Zip+ PCS verify.
    pub witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    /// Lookup argument proof data. `None` when the UAIR has no lookup specs.
    pub lookup: Option<LookupProofData<F>>,
}

impl<F> GenTranscribable for Proof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (commit0, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);
        let (commit1, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);
        let (commit2, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);

        let (zip_len, bytes) = u32::read_transcription_bytes_subset(bytes);
        let zip_len = usize::try_from(zip_len).expect("zip length must fit into usize");
        let (zip_bytes, bytes) = bytes.split_at(zip_len);
        let zip = zip_bytes.to_vec();

        let (ideal_check, bytes) = IdealCheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (resolver, bytes) =
            CombinedPolyResolverProof::<F>::read_transcription_bytes_subset(bytes);
        let (combined_sumcheck, bytes) =
            MultiDegreeSumcheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (multipoint_eval, bytes) =
            MultipointEvalProof::<F>::read_transcription_bytes_subset(bytes);

        let (witness_vec, bytes) = DynamicPolyVecF::<F>::read_transcription_bytes_subset(bytes);
        let witness_lifted_evals = witness_vec.0;

        // Lookup data: u8 flag (0 = None, 1 = Some) then fields
        let (has_lookup, mut bytes) = u8::read_transcription_bytes_subset(bytes);
        let lookup = if has_lookup == 0 {
            None
        } else {
            let (aux_evals, rest) = Vec::<F>::read_transcription_bytes_subset(bytes);
            let (lifted_vec, rest) = DynamicPolyVecF::<F>::read_transcription_bytes_subset(rest);

            let (num_comms, rest) = u32::read_transcription_bytes_subset(rest);
            let num_comms =
                usize::try_from(num_comms).expect("lookup comm count must fit into usize");
            let mut commitments = Vec::with_capacity(num_comms);
            let mut rest = rest;
            for _ in 0..num_comms {
                let (c1, r) = ZipPlusCommitment::read_transcription_bytes_subset(rest);
                let (c2, r) = ZipPlusCommitment::read_transcription_bytes_subset(r);
                commitments.push((c1, c2));
                rest = r;
            }

            let (num_chunk_comms, rest) = u32::read_transcription_bytes_subset(rest);
            let num_chunk_comms =
                usize::try_from(num_chunk_comms).expect("chunk comm count must fit into usize");
            let mut chunk_commitments = Vec::with_capacity(num_chunk_comms);
            let mut rest = rest;
            for _ in 0..num_chunk_comms {
                let (c, r) = ZipPlusCommitment::read_transcription_bytes_subset(rest);
                chunk_commitments.push(c);
                rest = r;
            }

            let (chunk_evals, rest) = Vec::<F>::read_transcription_bytes_subset(rest);
            let (chunk_lifted_vec, rest) =
                DynamicPolyVecF::<F>::read_transcription_bytes_subset(rest);
            bytes = rest;

            Some(LookupProofData {
                commitments,
                aux_evals,
                lifted_evals: lifted_vec.0,
                chunk_commitments,
                chunk_evals,
                chunk_lifted_evals: chunk_lifted_vec.0,
            })
        };

        assert!(bytes.is_empty(), "All bytes should be consumed");

        Self {
            commitments: (commit0, commit1, commit2),
            zip,
            ideal_check,
            resolver,
            combined_sumcheck,
            multipoint_eval,
            witness_lifted_evals,
            lookup,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        // 3 commitments (ConstTranscribable - no length prefix)
        buf = self.commitments.0.write_transcription_bytes_subset(buf);
        buf = self.commitments.1.write_transcription_bytes_subset(buf);
        buf = self.commitments.2.write_transcription_bytes_subset(buf);

        // zip: u32 length + raw bytes
        let zip_len = u32::try_from(self.zip.len()).expect("zip length must fit into u32");
        zip_len.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        buf = &mut buf[u32::NUM_BYTES..];
        buf[..self.zip.len()].copy_from_slice(&self.zip);
        buf = &mut buf[self.zip.len()..];

        // ideal_check: u32 length prefix + data
        buf = self.ideal_check.write_transcription_bytes_subset(buf);

        // resolver: u32 length prefix + data
        buf = self.resolver.write_transcription_bytes_subset(buf);

        // combined_sumcheck: u32 length prefix + data
        buf = self.combined_sumcheck.write_transcription_bytes_subset(buf);

        // multipoint_eval: u32 length prefix + data
        buf = self.multipoint_eval.write_transcription_bytes_subset(buf);

        // witness_lifted_evals: u32 length prefix + DynamicPolyVecF encoding
        buf = DynamicPolyVecF::reinterpret(&self.witness_lifted_evals)
            .write_transcription_bytes_subset(buf);

        // Lookup data: u8 flag + fields
        let has_lookup: u8 = u8::from(self.lookup.is_some());
        has_lookup.write_transcription_bytes_exact(&mut buf[..u8::NUM_BYTES]);
        buf = &mut buf[u8::NUM_BYTES..];

        if let Some(ref lk) = self.lookup {
            buf = lk.aux_evals.write_transcription_bytes_subset(buf);
            buf = DynamicPolyVecF::reinterpret(&lk.lifted_evals)
                .write_transcription_bytes_subset(buf);

            let num_comms = u32::try_from(lk.commitments.len()).expect("comms count fits u32");
            num_comms.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
            buf = &mut buf[u32::NUM_BYTES..];
            for (c1, c2) in &lk.commitments {
                buf = c1.write_transcription_bytes_subset(buf);
                buf = c2.write_transcription_bytes_subset(buf);
            }

            let num_chunk_comms =
                u32::try_from(lk.chunk_commitments.len()).expect("chunk comms count fits u32");
            num_chunk_comms.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
            buf = &mut buf[u32::NUM_BYTES..];
            for c in &lk.chunk_commitments {
                buf = c.write_transcription_bytes_subset(buf);
            }

            buf = lk.chunk_evals.write_transcription_bytes_subset(buf);
            buf = DynamicPolyVecF::reinterpret(&lk.chunk_lifted_evals)
                .write_transcription_bytes_subset(buf);
        }

        let _ = buf;
    }
}

impl<F> Transcribable for Proof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let witness_vec = DynamicPolyVecF::reinterpret(&self.witness_lifted_evals);
        3 * ZipPlusCommitment::NUM_BYTES
            + u32::NUM_BYTES
            + self.zip.len()
            + IdealCheckProof::<F>::LENGTH_NUM_BYTES
            + self.ideal_check.get_num_bytes()
            + CombinedPolyResolverProof::<F>::LENGTH_NUM_BYTES
            + self.resolver.get_num_bytes()
            + MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES
            + self.combined_sumcheck.get_num_bytes()
            + MultipointEvalProof::<F>::LENGTH_NUM_BYTES
            + self.multipoint_eval.get_num_bytes()
            + DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
            + witness_vec.get_num_bytes()
            + u8::NUM_BYTES
            + self.lookup.as_ref().map_or(0, |lk| {
                Vec::<F>::LENGTH_NUM_BYTES
                    + lk.aux_evals.get_num_bytes()
                    + DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
                    + DynamicPolyVecF::reinterpret(&lk.lifted_evals).get_num_bytes()
                    + u32::NUM_BYTES
                    + lk.commitments.len() * 2 * ZipPlusCommitment::NUM_BYTES
                    + u32::NUM_BYTES
                    + lk.chunk_commitments.len() * ZipPlusCommitment::NUM_BYTES
                    + Vec::<F>::LENGTH_NUM_BYTES
                    + lk.chunk_evals.get_num_bytes()
                    + DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
                    + DynamicPolyVecF::reinterpret(&lk.chunk_lifted_evals).get_num_bytes()
            })
    }
}

/// Trait bundling the various type parameters for the public inputs (NYI),
/// witness and Zinc+ PIOP.
pub trait ZincTypes<const DEGREE_PLUS_ONE: usize> {
    /// Main integer type for the protocol, used as a coefficient type for the
    /// arbitrary polynomial trace columns and for the integer trace columns.
    type Int: Semiring
        + ConstTranscribable
        + ConstCoeffBitWidth
        + Named
        + Default
        + Clone
        + Send
        + Sync
        + 'static;

    /// Projecting element to project Zip+ evaluations and UAIR scalars to the
    /// field.
    type Chal: ConstIntRing + ConstTranscribable + Named;

    /// Evaluation point type, used for all column types in Zip+ to evaluate
    /// multilinear polynomials.
    type Pt: ConstIntRing;

    type CombR;

    /// Randomly sampled field modulus type, used throughout the protocol for
    /// finite field operations.
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;

    /// Primality test for the field modulus.
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// Zip+ types for the binary polynomial trace columns.
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns.
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the integer trace columns.
    type IntZt: ZipTypes<
            Eval = Self::Int,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Linear code used in Zip+ for the binary polynomial trace columns.
    type BinaryLc: LinearCode<Self::BinaryZt>;

    /// Linear code used in Zip+ for the arbitrary polynomial trace columns.
    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;

    /// Linear code used in Zip+ for the integer trace columns.
    type IntLc: LinearCode<Self::IntZt>;

    /// Zip+ types for lookup auxiliary columns (m, u, v) — field-element-sized
    /// integers.
    type LookupZt: ZipTypes<
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Linear code for lookup auxiliary columns.
    type LookupLc: LinearCode<Self::LookupZt>;
}

/// Main struct for the Zinc+ PIOP. The protocol is implemented as associated
/// functions on it.
///
/// (Note that type parameters are further constrained in the impl blocks for
/// the prover and verifier)
#[derive(Copy, Clone, Default, Debug)]
pub struct ZincPlusPiop<Zt, U, F, const DEGREE_PLUS_ONE: usize>(PhantomData<(Zt, U, F)>)
where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    U: Uair,
    F: PrimeField;

/// Error type for error happening during the protocol execution (prover and
/// verifier).
#[derive(Debug, Error)]
pub enum ProtocolError<F: PrimeField, I: Ideal> {
    #[error("ideal check failed: {0}")]
    IdealCheck(#[from] IdealCheckError<F, I>),
    #[error("combined poly resolver failed: {0}")]
    Resolver(#[from] CombinedPolyResolverError<F>),
    #[error("scalar projection failed: {0}")]
    ScalarProjection(PolyEvaluationError),
    #[error("multi-point evaluation failed: {0}")]
    MultipointEval(#[from] MultipointEvalError<F>),
    #[error("lifted eval psi_a projection failed: {0}")]
    LiftedEvalProjection(PolyEvaluationError),
    #[error("lookup argument failed: {0}")]
    Lookup(#[from] LookupError),
    #[error("PCS error: {0}")]
    Pcs(#[from] ZipError),
    #[error("PCS verification failed at column {0}: {1}")]
    PcsVerification(usize, ZipError),
}

//
// Helper functions
//

/// Absorb public column entries into the Fiat-Shamir transcript.
///
/// Each entry is serialized via `ConstTranscribable::write_transcription_bytes`
/// and absorbed. This must be called in the same order by both prover and
/// verifier, after commitments and before the random prime draw.
fn absorb_public_columns<T: ConstTranscribable>(
    transcript: &mut impl Transcript,
    cols: &[DenseMultilinearExtension<T>],
) {
    let mut buf = vec![0u8; T::NUM_BYTES];
    for col in cols {
        for entry in col.iter() {
            entry.write_transcription_bytes_exact(&mut buf);
            transcript.absorb_slice(&buf);
        }
    }
}

/// Compute per-column lifted MLE evaluations at `point`.
///
/// For each column j, returns `\sum_b eq(b, point) * v_j(b)` as a polynomial
/// in `F_q[X]` (coefficient-wise MLE evaluation). Dispatches on the trace
/// layout internally.
///
/// Binary columns exploit the 0/1 structure for conditional additions only.
/// The `eq(point, *)` table is built once and reused across all columns.
#[allow(clippy::arithmetic_side_effects)]
fn compute_lifted_evals<F: PrimeField, const D: usize>(
    point: &[F],
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_trace: &ProjectedTrace<F>,
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>> {
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_lifted_evals: eq table build failed");

    let n_bin = trace_bin_poly.len();
    let zero = F::zero_with_cfg(field_cfg);

    // Binary columns: exploit 0/1 structure for conditional additions.
    let mut result: Vec<DynamicPolynomialF<F>> = cfg_iter!(trace_bin_poly)
        .map(|col| {
            let mut coeffs = vec![zero.clone(); D];
            for (b, entry) in col.iter().enumerate() {
                for (l, coeff) in entry.iter().enumerate() {
                    if coeff.into_inner() {
                        coeffs[l] += &eq_table[b];
                    }
                }
            }
            DynamicPolynomialF::new_trimmed(coeffs)
        })
        .collect();

    // Non-binary columns: coefficient-wise eq-weighted sum.
    fn weighted_eq_sum<'a, F2: PrimeField + 'a>(
        col: impl Iterator<Item = &'a DynamicPolynomialF<F2>> + Clone,
        eq_table: &[F2],
        zero: &F2,
    ) -> DynamicPolynomialF<F2> {
        let num_coeffs = col.clone().map(|e| e.coeffs.len()).max().unwrap_or(0);
        let mut coeffs = vec![zero.clone(); num_coeffs];
        for (b, entry) in col.enumerate() {
            for (l, coeff) in entry.coeffs.iter().enumerate() {
                let mut term = eq_table[b].clone();
                term *= coeff;
                coeffs[l] += &term;
            }
        }
        DynamicPolynomialF::new_trimmed(coeffs)
    }

    match projected_trace {
        ProjectedTrace::RowMajor(t) => {
            let num_cols = t.first().map(|r| r.len()).unwrap_or(0);
            cfg_extend!(
                result,
                cfg_into_iter!(n_bin..num_cols).map(|col_idx| weighted_eq_sum(
                    t.iter().map(|row| &row[col_idx]),
                    &eq_table,
                    &zero,
                ))
            );
        }
        ProjectedTrace::ColumnMajor(t) => {
            cfg_extend!(
                result,
                cfg_iter!(t[n_bin..]).map(|col_mle| weighted_eq_sum(
                    col_mle.iter(),
                    &eq_table,
                    &zero,
                ))
            );
        }
    }

    result
}

/// Project a DensePolynomial scalar to DynamicPolynomialF by projecting each
/// coefficient via \phi_q.
pub fn project_scalar_fn<R, F, const D: usize>(
    scalar: &DensePolynomial<R, D>,
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: PrimeField + for<'a> FromWithConfig<&'a R>,
{
    scalar
        .iter()
        .map(|coeff| F::from_with_cfg(coeff, field_cfg))
        .collect()
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::U64;
    use crypto_primitives::{
        Field, crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use rand::rng;
    use zinc_piop::{
        combined_poly_resolver::CombinedPolyResolverError, multipoint_eval::MultipointEvalError,
    };
    use zinc_poly::univariate::{binary::BinaryPolyInnerProduct, dense::DensePolyInnerProduct};
    use zinc_primality::MillerRabin;
    use zinc_test_uair::{
        BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, BitPolyLookupUair,
        DecomposedBitPolyUair, DecomposedWordUair, GenerateRandomTrace, MultiColLookupUair,
        MultiGroupLookupUair, RangeCheck8Uair, SimpleLookupUair, TestAirNoMultiplication,
        TestUairMixedShifts, TestUairSimpleMultiplication, Word8LookupUair,
    };
    use zinc_uair::{
        degree_counter::count_max_degree, ideal::degree_one::DegreeOneIdeal,
        ideal_collector::IdealOrZero,
    };
    use zinc_utils::{
        CHECKED,
        from_ref::FromRef,
        inner_product::{MBSInnerProduct, ScalarProduct},
        projectable_to_field::ProjectableToField,
    };
    use zip_plus::{
        code::{
            iprs::{IprsCode, PnttConfigF65537},
            raa::{RaaCode, RaaConfig},
        },
        pcs::structs::{ZipPlus, ZipPlusParams},
        pcs_transcript::PcsProverTranscript,
    };

    const INT_LIMBS: usize = U64::LIMBS;
    const FIELD_LIMBS: usize = U64::LIMBS * 3;
    const DEGREE_PLUS_ONE: usize = 32;

    // Zip+ type parameters.

    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    const REP: usize = 4;

    type F = MontyField<FIELD_LIMBS>;

    pub struct BinPolyZipTypes {}
    impl ZipTypes for BinPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    pub struct ArbitraryPolyZipTypesIprs {}
    impl ZipTypes for ArbitraryPolyZipTypesIprs {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal =
            DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Arbitrary poly ZipTypes with wider codewords for RAA encoding.
    /// RAA accumulation grows the bit-width, so Cw needs more bits than Eval.
    pub struct ArbitraryPolyZipTypesRaa {}
    impl ZipTypes for ArbitraryPolyZipTypesRaa {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<K>, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal =
            DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    type ZtInt = i64;

    pub struct IntZipTypes {}
    impl ZipTypes for IntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = ZtInt;
        type Cw = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Zip+ types for lookup auxiliary columns (m, u, v).
    /// Eval = Int<LOOKUP_EVAL_LIMBS> holds any element of F_q.
    /// FIELD_LIMBS + 1 because Int is signed: Int<3> maxes at 2^191-1,
    /// but field elements can be up to q-1 ≈ 2^192-1.
    const LOOKUP_EVAL_LIMBS: usize = FIELD_LIMBS + 1;

    pub struct LookupZipTypes {}
    impl ZipTypes for LookupZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = Int<LOOKUP_EVAL_LIMBS>;
        type Cw = Int<M>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    struct TestZincTypesIprs;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesIprs {
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;
        type LookupZt = LookupZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
        type LookupLc = IprsCode<Self::LookupZt, PnttConfigF65537, REP, CHECKED>;
    }

    #[derive(Copy, Clone)]
    struct TestRaaConfig;
    impl RaaConfig for TestRaaConfig {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = true;
    }

    struct TestZincTypesRaa;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesRaa {
        type Int = i64;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesRaa;
        type IntZt = IntZipTypes;
        type LookupZt = LookupZipTypes;

        type BinaryLc = RaaCode<Self::BinaryZt, TestRaaConfig, REP>;
        type ArbitraryLc = RaaCode<Self::ArbitraryZt, TestRaaConfig, REP>;
        type IntLc = RaaCode<Self::IntZt, TestRaaConfig, REP>;
        type LookupLc = RaaCode<Self::LookupZt, TestRaaConfig, REP>;
    }

    /// Use row size equal to poly size, resulting in flat single-row matrices
    fn make_iprs<Zt: ZipTypes>(num_vars: usize) -> IprsCode<Zt, PnttConfigF65537, REP, CHECKED> {
        let poly_size = 1 << num_vars;
        IprsCode::new_with_optimal_depth(poly_size).unwrap()
    }

    /// Set up Zip+ PCS parameters for a given number of MLE variables.
    #[allow(clippy::type_complexity)]
    fn setup_pp<Zt>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc, Zt::LookupLc),
    ) -> (
        ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
        ZipPlusParams<Zt::IntZt, Zt::IntLc>,
        ZipPlusParams<Zt::LookupZt, Zt::LookupLc>,
    )
    where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
    {
        let poly_size = 1 << num_vars;
        (
            ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::setup(poly_size, linear_codes.0),
            ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::setup(poly_size, linear_codes.1),
            ZipPlus::<Zt::IntZt, Zt::IntLc>::setup(poly_size, linear_codes.2),
            ZipPlus::<Zt::LookupZt, Zt::LookupLc>::setup(poly_size, linear_codes.3),
        )
    }

    macro_rules! default_project_ideal {
        () => {
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
        };
    }

    #[allow(clippy::result_large_err)]
    fn do_test<Zt, U>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc, Zt::LookupLc),
        project_ideal: impl Fn(
            &IdealOrZero<U::Ideal>,
            &<F as PrimeField>::Config,
        ) -> IdealOrZero<DegreeOneIdeal<F>>
        + Copy,
        tamper: impl Fn(&mut Proof<F>),
        check_verification: impl Fn(Result<(), ProtocolError<F, IdealOrZero<DegreeOneIdeal<F>>>>),
    ) where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
        <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
        <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
        U: Uair<Scalar = DensePolynomial<Zt::Int, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Zt::Int, Int = Zt::Int>
            + 'static,
        F: for<'a> FromWithConfig<&'a Zt::Int>
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>,
        <F as Field>::Inner: FromRef<Zt::Fmod>,
        <F as Field>::Modulus: FromRef<Zt::Fmod>,
        <Zt::LookupZt as ZipTypes>::Eval: FromRef<<F as Field>::Inner>,
    {
        let mut rng = rng();
        let pp = setup_pp::<Zt>(num_vars, linear_codes);

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = U::signature();
        let public_trace = trace.public(&sig);

        macro_rules! run_protocol {
            ($mle_first:ident) => {
                let mut proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<
                    { $mle_first },
                    CHECKED,
                >(&pp, &trace, num_vars, project_scalar_fn)
                .expect("Prover failed");

                // Checking that the proof can be properly serialized and deserialized
                let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
                transcript.write(&proof).expect("Failed to serialize proof");
                let mut transcript = transcript.into_verification_transcript();
                let proof_2 = transcript
                    .read()
                    .expect("Failed to deserialize proof after serialization");
                assert_eq!(proof, proof_2);

                tamper(&mut proof);

                let verification_result =
                    ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                        &pp,
                        proof,
                        &public_trace,
                        num_vars,
                        project_scalar_fn,
                        project_ideal,
                    );
                check_verification(verification_result);
            };
        }

        run_protocol!(false);

        if count_max_degree::<U>() <= 1 {
            // For linear constraints, also test the MLE-first ideal check approach.
            run_protocol!(true);
        }
    }

    /// End-to-end test: TestAirNoMultiplication.
    ///
    /// UAIR constraint: a + b - c \in (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = <X - 2>).
    #[test]
    fn test_e2e_no_multiplication() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestAirNoMultiplication<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: TestUairSimpleMultiplication.
    ///
    /// UAIR constraints (3 total, no ideals):
    ///   up[0] * up[1] = down[0]
    ///   up[1] * up[2] = down[1]
    ///   up[0] * up[2] = down[2]
    ///
    /// Uses RAA code with small num_vars (2) because chained polynomial
    /// multiplication causes exponential growth in both degree and coefficient
    /// magnitude. With num_vars=2 (4 rows), max degree=6 and max coefficient
    /// ~= 127^8 ~= 2^56, which fits in i64.
    #[test]
    fn test_e2e_simple_multiplication() {
        let num_vars = 2;
        do_test::<TestZincTypesRaa, TestUairSimpleMultiplication<ZtInt>>(
            num_vars,
            (
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: TestUairMixedShifts.
    ///
    /// Uses mixed shift amounts (col a: shift 1, col b: shift 2).
    /// Constraints: a[i+1] = a[i] + b[i], c[i] = b[i+2].
    #[test]
    fn test_e2e_mixed_shifts() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairMixedShifts<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: BinaryDecompositionUair.
    ///
    /// Uses binary_poly (1 col) and int (1 col) trace types.
    /// UAIR constraint: binary_poly[0] - int[0] \in <X - 2>
    #[test]
    fn test_e2e_binary_decomposition() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BinaryDecompositionUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: BigLinearUair.
    ///
    /// Uses 16 binary_poly cols and 1 int col.
    /// UAIR constraints:
    ///   sum(up.binary_poly[0..16]) - up.int[0] \in <X - 1>
    ///   down.binary_poly[0] - up.int[0] \in <X - 2>
    ///   up.binary_poly[i] - down.binary_poly[i] = 0, for i=1..15
    #[test]
    fn test_e2e_big_linear() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: BigLinearUairWithPublicInput.
    ///
    /// Same as [`BigLinearUair`], but with the first few binary_poly columns as
    /// public inputs.
    #[test]
    fn test_e2e_big_linear_with_public_input() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: SimpleLookupUair.
    ///
    /// 2 int cols, constraint: a - b = 0, column 0 looked up against Word(4).
    #[test]
    fn test_e2e_simple_lookup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, SimpleLookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: Word8LookupUair (Word(8) lookup, no decomp).
    #[test]
    fn test_e2e_word8_lookup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, Word8LookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: RangeCheck8Uair (8-bit binary decomp, no lookup).
    #[test]
    fn test_e2e_range_check_8() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, RangeCheck8Uair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    //
    // Negative tests for SimpleLookupUair
    //

    #[test]
    fn test_lookup_tamper_aux_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, SimpleLookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |proof| {
                let lk = proof.lookup.as_mut().unwrap();
                let cfg = *lk.aux_evals[0].cfg();
                lk.aux_evals[0] += &F::from_with_cfg(1u64, &cfg);
            },
            |res| {
                assert!(res.is_err(), "tampered lookup_aux_evals should be rejected");
            },
        );
    }

    #[test]
    fn test_lookup_tamper_lifted_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, SimpleLookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |proof| {
                let lk = proof.lookup.as_mut().unwrap();
                if let Some(c) = lk.lifted_evals[0].coeffs.first_mut() {
                    let one = F::from_with_cfg(1u64, &c.cfg().clone());
                    *c += &one;
                }
            },
            |res| {
                assert!(
                    res.is_err(),
                    "tampered lookup_lifted_evals should be rejected"
                );
            },
        );
    }

    #[test]
    fn test_lookup_tamper_commitment() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, SimpleLookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |proof| {
                if let Some(ref mut lk) = proof.lookup {
                    lk.commitments[0].0.root = Default::default();
                }
            },
            |res| {
                assert!(
                    res.is_err(),
                    "tampered lookup commitment should be rejected"
                );
            },
        );
    }

    /// End-to-end test: MultiColLookupUair (L=2, same Word(4) table).
    #[test]
    fn test_e2e_multi_col_lookup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, MultiColLookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: MultiGroupLookupUair (Word(4) + Word(8), two groups).
    #[test]
    fn test_e2e_multi_group_lookup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, MultiGroupLookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: BitPolyLookupUair (BitPoly(8) table type).
    #[test]
    fn test_e2e_bitpoly_lookup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitPolyLookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    //
    // Decomposed lookup tests
    //

    /// E2E: DecomposedWordUair — Word(8) with chunk_width=4, K=2.
    #[test]
    fn test_e2e_decomposed_word_lookup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, DecomposedWordUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// E2E: DecomposedBitPolyUair — BitPoly(8) with chunk_width=4, K=2.
    #[test]
    fn test_e2e_decomposed_bitpoly_lookup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, DecomposedBitPolyUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    #[test]
    fn test_decomposed_word_tamper_chunk_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, DecomposedWordUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |proof| {
                let lk = proof.lookup.as_mut().unwrap();
                let cfg = *lk.chunk_evals[0].cfg();
                lk.chunk_evals[0] += &F::from_with_cfg(1u64, &cfg);
            },
            |res| {
                assert!(res.is_err(), "tampered chunk_evals should be rejected");
            },
        );
    }

    #[test]
    fn test_decomposed_word_tamper_chunk_commitment() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, DecomposedWordUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |proof| {
                if let Some(ref mut lk) = proof.lookup {
                    lk.chunk_commitments[0].root = Default::default();
                }
            },
            |res| {
                assert!(res.is_err(), "tampered chunk commitment should be rejected");
            },
        );
    }

    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn test_decomposed_bitpoly_tamper_chunk_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, DecomposedBitPolyUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| {
                let lk = proof.lookup.as_mut().unwrap();
                let cfg = *lk.chunk_evals[0].cfg();
                lk.chunk_evals[0] += &F::from_with_cfg(1u64, &cfg);
            },
            |res| {
                assert!(res.is_err(), "tampered chunk_evals should be rejected");
            },
        );
    }

    #[test]
    fn test_decomposed_bitpoly_tamper_chunk_commitment() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, DecomposedBitPolyUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| {
                if let Some(ref mut lk) = proof.lookup {
                    lk.chunk_commitments[0].root = Default::default();
                }
            },
            |res| {
                assert!(res.is_err(), "tampered chunk commitment should be rejected");
            },
        );
    }

    //
    // Negative tests for BigLinearUairWithPublicInput: verify that proof
    // tampering is detected.
    //

    #[test]
    fn test_big_linear_tamper_lifted_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.witness_lifted_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                ));
            },
        );
    }

    #[test]
    fn test_big_linear_tamper_up_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.resolver.up_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(
                        CombinedPolyResolverError::ClaimValueDoesNotMatch { .. }
                    )
                ));
            },
        );
    }

    #[test]
    fn test_big_linear_tamper_down_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.resolver.down_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(
                        CombinedPolyResolverError::ClaimValueDoesNotMatch { .. }
                    )
                ));
            },
        );
    }

    // Tampering the commitment root causes the verifier to sample different
    // challenges. The ideal check fails first because the prover's
    // combined_mle_values were computed under the original transcript.
    #[test]
    fn test_big_linear_tamper_commitment() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.commitments.0.root = Default::default(),
            |res| {
                assert!(matches!(res.unwrap_err(), ProtocolError::IdealCheck(..)));
            },
        );
    }

    #[test]
    fn test_big_linear_tamper_ideal_check() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.ideal_check.combined_mle_values.swap(0, 1),
            |res| {
                assert!(matches!(res.unwrap_err(), ProtocolError::IdealCheck(..)));
            },
        );
    }
}
