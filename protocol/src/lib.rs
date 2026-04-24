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
use std::{fmt::Debug, marker::PhantomData};
use thiserror::Error;
use zinc_piop::{
    combined_poly_resolver::{CombinedPolyResolverError, Proof as CombinedPolyResolverProof},
    ideal_check::{IdealCheckError, Proof as IdealCheckProof},
    lookup::{LookupError, logup_gkr::LookupArgumentProof},
    multipoint_eval::{MultipointEvalError, Proof as MultipointEvalProof},
    multipoint_reducer::MultiPointReduceProof,
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
    /// Lookup argument proof. `None` when the UAIR has no lookup specs.
    pub lookup_proof: Vec<LookupArgumentProof<F>>,
    /// Phase 2f: reducer that folds (r_0, witness_evals) + each
    /// (ρ_row_g, witness_evals) into r_final. `None` when there are
    /// no lookups (opening stays at r_0).
    pub lookup_reducer: Option<LookupReducerProof<F>>,
}

/// Auxiliary data for the lookup reducer step.
///
/// The reducer binds prover-claimed scalar evaluations at r_0 and at
/// each lookup group's ρ_row to the true column MLEs, producing a
/// fresh r_final at which the PCS is opened.
///
/// Currently supports only UAIRs with zero public columns (asserted at
/// prove/verify entry points).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupReducerProof<F> {
    /// Witness column evals at r_0 (one per witness column). Used by
    /// the verifier as inputs to both the mp_subclaim check and the
    /// reducer's claimed-sum check.
    pub witness_evals_at_r_0: Vec<F>,
    /// Per lookup group: witness column evals at ρ_row_g (one vec per
    /// group, one F per witness column).
    pub witness_evals_at_rho_row: Vec<Vec<F>>,
    /// MultiPointReducer proof.
    pub reducer_proof: MultiPointReduceProof<F>,
}

impl<F: PrimeField> zinc_transcript::traits::GenTranscribable for LookupReducerProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (witness_evals_at_r_0, bytes) =
            Vec::<F>::read_transcription_bytes_subset(bytes);
        let (n_groups, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_groups = usize::try_from(n_groups).expect("group count fits usize");
        let mut witness_evals_at_rho_row = Vec::with_capacity(n_groups);
        for _ in 0..n_groups {
            let (v, rest) = Vec::<F>::read_transcription_bytes_subset(bytes);
            witness_evals_at_rho_row.push(v);
            bytes = rest;
        }
        let (reducer_proof, bytes) =
            MultiPointReduceProof::<F>::read_transcription_bytes_subset(bytes);
        assert!(bytes.is_empty(), "trailing bytes");
        Self {
            witness_evals_at_r_0,
            witness_evals_at_rho_row,
            reducer_proof,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let buf = self.witness_evals_at_r_0.write_transcription_bytes_subset(buf);
        let n_groups = u32::try_from(self.witness_evals_at_rho_row.len())
            .expect("group count fits u32");
        n_groups.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        let mut buf = &mut buf[u32::NUM_BYTES..];
        for v in &self.witness_evals_at_rho_row {
            buf = v.write_transcription_bytes_subset(buf);
        }
        let buf = self.reducer_proof.write_transcription_bytes_subset(buf);
        assert!(buf.is_empty(), "buffer size mismatch");
    }
}

impl<F: PrimeField> zinc_transcript::traits::Transcribable for LookupReducerProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let per_group_bytes: usize = self
            .witness_evals_at_rho_row
            .iter()
            .map(|v| Vec::<F>::LENGTH_NUM_BYTES + v.get_num_bytes())
            .sum();
        Vec::<F>::LENGTH_NUM_BYTES
            + self.witness_evals_at_r_0.get_num_bytes()
            + u32::NUM_BYTES
            + per_group_bytes
            + MultiPointReduceProof::<F>::LENGTH_NUM_BYTES
            + self.reducer_proof.get_num_bytes()
    }
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

        // lookup_proof: u32 count + each LookupArgumentProof length-prefixed.
        let (n_lookup, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_lookup = usize::try_from(n_lookup).expect("lookup count fits usize");
        let mut lookup_proof = Vec::with_capacity(n_lookup);
        for _ in 0..n_lookup {
            let (p, rest) = LookupArgumentProof::<F>::read_transcription_bytes_subset(bytes);
            lookup_proof.push(p);
            bytes = rest;
        }

        // lookup_reducer: u8 tag (1=Some, 0=None) + optional payload.
        let has_reducer = bytes[0];
        let bytes = &bytes[1..];
        let (lookup_reducer, bytes) = if has_reducer == 0 {
            (None, bytes)
        } else {
            let (r, rest) = LookupReducerProof::<F>::read_transcription_bytes_subset(bytes);
            (Some(r), rest)
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
            lookup_proof,
            lookup_reducer,
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

        // lookup_proof: u32 count + each LookupArgumentProof length-prefixed.
        let n_lookup = u32::try_from(self.lookup_proof.len()).expect("lookup count fits u32");
        n_lookup.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        buf = &mut buf[u32::NUM_BYTES..];
        for p in &self.lookup_proof {
            buf = p.write_transcription_bytes_subset(buf);
        }

        // lookup_reducer: u8 tag + optional payload.
        buf[0] = u8::from(self.lookup_reducer.is_some());
        buf = &mut buf[1..];
        if let Some(r) = &self.lookup_reducer {
            buf = r.write_transcription_bytes_subset(buf);
        }

        assert!(buf.is_empty(), "Entire buffer should be used");
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
        let lookup_bytes: usize = self
            .lookup_proof
            .iter()
            .map(|p| LookupArgumentProof::<F>::LENGTH_NUM_BYTES + p.get_num_bytes())
            .sum();
        let reducer_bytes = match &self.lookup_reducer {
            None => 0,
            Some(r) => LookupReducerProof::<F>::LENGTH_NUM_BYTES + r.get_num_bytes(),
        };
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
            + u32::NUM_BYTES // lookup count
            + lookup_bytes
            + 1 // reducer tag
            + reducer_bytes
    }
}

/// Trait bundling the various type parameters for the public inputs (NYI),
/// witness and Zinc+ PIOP.
pub trait ZincTypes<const DEGREE_PLUS_ONE: usize>: Clone + Debug {
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
        BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, GenerateRandomTrace,
        INT_LOOKUP_TABLE_WIDTH, IntLookupMultiUair, IntLookupUair, Sha256CompressionSliceUair,
        Sha256Ideal, TestUairMixedShifts, TestUairNoMultiplication,
        TestUairSimpleMultiplication,
    };
    use zinc_uair::{
        degree_counter::count_max_degree,
        ideal::{DegreeOneIdeal, rotation::RotationIdeal},
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

    #[derive(Debug, Clone)]
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

    #[derive(Debug, Clone)]
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
    #[derive(Debug, Clone)]
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

    #[derive(Debug, Clone)]
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

    #[derive(Clone, Debug)]
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

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    #[derive(Copy, Clone)]
    struct TestRaaConfig;
    impl RaaConfig for TestRaaConfig {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = true;
    }

    #[derive(Clone, Debug)]
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

        type BinaryLc = RaaCode<Self::BinaryZt, TestRaaConfig, REP>;
        type ArbitraryLc = RaaCode<Self::ArbitraryZt, TestRaaConfig, REP>;
        type IntLc = RaaCode<Self::IntZt, TestRaaConfig, REP>;
    }

    // -----------------------------------------------------------------
    // Wider parameterization for UAIRs that need 256-bit trace integers
    // (e.g. ECDSA over secp256k1). Uses `Int<5>` for the trace int type and
    // a 768-bit random prime field for the ideal check.
    // -----------------------------------------------------------------

    const ECDSA_INT_LIMBS: usize = zinc_test_uair::ECDSA_INT_LIMBS;
    const ECDSA_K: usize = ECDSA_INT_LIMBS * 2; // codeword widening
    const ECDSA_M: usize = ECDSA_INT_LIMBS * 4; // CombR widening
    const ECDSA_FIELD_LIMBS: usize = 12; // 768-bit random prime field

    type EcdsaInt = Int<ECDSA_INT_LIMBS>;
    type EcdsaF = MontyField<ECDSA_FIELD_LIMBS>;

    #[derive(Debug, Clone)]
    pub struct EcdsaBinPolyZipTypes {}
    impl ZipTypes for EcdsaBinPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        // Placeholder only: this UAIR has zero binary_poly cols, so Cw is
        // type-level dead code. Must be i64 to satisfy MulByScalar bounds.
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<ECDSA_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<ECDSA_M>;
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

    #[derive(Debug, Clone)]
    pub struct EcdsaArbPolyZipTypes {}
    impl ZipTypes for EcdsaArbPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = DensePolynomial<EcdsaInt, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<ECDSA_K>, DEGREE_PLUS_ONE>;
        type Fmod = Uint<ECDSA_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<ECDSA_M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = DensePolyInnerProduct<
            EcdsaInt,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    pub struct EcdsaIntZipTypes {}
    impl ZipTypes for EcdsaIntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = EcdsaInt;
        type Cw = Int<ECDSA_K>;
        type Fmod = Uint<ECDSA_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<ECDSA_M>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct EcdsaZincTypes;

    impl ZincTypes<DEGREE_PLUS_ONE> for EcdsaZincTypes {
        type Int = EcdsaInt;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<ECDSA_M>;
        type Fmod = Uint<ECDSA_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = EcdsaBinPolyZipTypes;
        type ArbitraryZt = EcdsaArbPolyZipTypes;
        type IntZt = EcdsaIntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
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
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
    ) -> (
        ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
        ZipPlusParams<Zt::IntZt, Zt::IntLc>,
    )
    where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
    {
        let poly_size = 1 << num_vars;
        (
            ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::setup(poly_size, linear_codes.0),
            ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::setup(poly_size, linear_codes.1),
            ZipPlus::<Zt::IntZt, Zt::IntLc>::setup(poly_size, linear_codes.2),
        )
    }

    macro_rules! default_project_ideal {
        () => {
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
        };
    }

    /// F-generic harness. Threads an explicit `FF` field through the prove /
    /// verify loop. Callers pick FF to match their UAIR's value-range needs.
    /// The existing [`do_test`] is a thin wrapper that hard-codes
    /// `FF = MontyField<FIELD_LIMBS>` (the 192-bit default).
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    fn do_test_f<FF, Zt, U, IdealOverF>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<FF as PrimeField>::Config) -> IdealOverF
        + Copy,
        tamper: impl Fn(&mut Proof<FF>),
        check_verification: impl Fn(Result<(), ProtocolError<FF, IdealOverF>>),
    ) where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
        Zt::Int: ProjectableToField<FF>,
        <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<FF>,
        <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<FF>,
        <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<FF>,
        <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<FF>,
        U: Uair<Scalar = DensePolynomial<Zt::Int, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Zt::Int, Int = Zt::Int>
            + 'static,
        FF: zinc_utils::inner_transparent_field::InnerTransparentField
            + crypto_primitives::FromPrimitiveWithConfig
            + FromRef<FF>
            + Send
            + Sync
            + 'static
            + for<'a> FromWithConfig<&'a Zt::Int>
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a FF>,
        <FF as Field>::Inner: FromRef<Zt::Fmod>
            + ConstIntSemiring
            + zinc_transcript::traits::ConstTranscribable
            + Send
            + Sync
            + num_traits::Zero
            + Default,
        <FF as Field>::Modulus: FromRef<Zt::Fmod>
            + zinc_transcript::traits::ConstTranscribable,
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<FF>>,
    {
        let mut rng = rng();
        let pp = setup_pp::<Zt>(num_vars, linear_codes);

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = U::signature();
        let public_trace = trace.public(&sig);

        macro_rules! run_protocol {
            ($mle_first:ident) => {
                let mut proof = ZincPlusPiop::<Zt, U, FF, DEGREE_PLUS_ONE>::prove::<
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
                    ZincPlusPiop::<Zt, U, FF, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
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

    /// Default F-parameterized harness used by all existing tests (F is the
    /// module-level `MontyField<FIELD_LIMBS>` = 192-bit). For tests that need
    /// a wider F (e.g. ECDSA over secp256k1), call [`do_test_f`] directly.
    #[allow(clippy::result_large_err)]
    fn do_test<Zt, U, IdealOverF>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &<F as PrimeField>::Config) -> IdealOverF
        + Copy,
        tamper: impl Fn(&mut Proof<F>),
        check_verification: impl Fn(Result<(), ProtocolError<F, IdealOverF>>),
    ) where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
        Zt::Int: ProjectableToField<F>,
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
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<F>>,
    {
        do_test_f::<F, Zt, U, IdealOverF>(
            num_vars,
            linear_codes,
            project_ideal,
            tamper,
            check_verification,
        );
    }

    /// End-to-end test: TestUairNoMultiplication.
    ///
    /// UAIR constraint: a + b - c \in (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = <X - 2>).
    #[test]
    fn test_e2e_no_multiplication() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairNoMultiplication<ZtInt>, _>(
            num_vars,
            (
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
        do_test::<TestZincTypesRaa, TestUairSimpleMultiplication<ZtInt>, _>(
            num_vars,
            (
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
        do_test::<TestZincTypesIprs, TestUairMixedShifts<ZtInt>, _>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: SHA-256 compression slice.
    ///
    /// UAIR constraints (see `test-uair/src/sha256.rs`):
    ///   1. s_sigma · (a_hat · rho_0(X) - sigma_0_hat) ∈ (X^32 - 1) mod 2
    ///   2. s_init  · (a_hat - y_a_public) == 0
    ///
    /// Exercises the new mod-2 (X^W - 1) ideal check at the verifier via
    /// `Sha256Ideal::RotXw1Mod2`, whose `contains` reduces remainder
    /// coefficients mod 2 (canonical parity) before accepting. See the
    /// module doc in `test-uair/src/sha256.rs` for the soundness caveat.
    #[test]
    fn test_e2e_sha256_slice() {
        let num_vars = 7; // 128 rows
        do_test::<TestZincTypesIprs, Sha256CompressionSliceUair<ZtInt>, Sha256Ideal<F>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |ideal_or_zero, field_cfg| {
                // Zero ideals are filtered out upstream of this closure (see
                // `piop/src/ideal_check.rs`), so we only receive NonZero.
                match ideal_or_zero {
                    IdealOrZero::NonZero(Sha256Ideal::RotX2(r)) => {
                        Sha256Ideal::RotX2(RotationIdeal::from_with_cfg(r, field_cfg))
                    }
                    IdealOrZero::NonZero(Sha256Ideal::RotXw1) => Sha256Ideal::RotXw1,
                    IdealOrZero::Zero => {
                        unreachable!("zero ideals are filtered before this closure runs")
                    }
                }
            },
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
        do_test::<TestZincTypesIprs, BinaryDecompositionUair<ZtInt>, _>(
            num_vars,
            (
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
        do_test::<TestZincTypesIprs, BigLinearUair<ZtInt>, _>(
            num_vars,
            (
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>, _>(
            num_vars,
            (
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
    // Negative tests for BigLinearUairWithPublicInput: verify that proof
    // tampering is detected.
    //

    #[test]
    fn test_big_linear_tamper_lifted_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>, _>(
            num_vars,
            (
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>, _>(
            num_vars,
            (
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>, _>(
            num_vars,
            (
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>, _>(
            num_vars,
            (
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>, _>(
            num_vars,
            (
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

    // -----------------------------------------------------------------------
    // IntLookupUair: exercises the protocol alongside a standalone lookup
    // argument on the same committed trace.
    //
    // The existing protocol pipeline currently *ignores* lookup_specs in
    // `UairSignature`, so Step 1–7 run as if no lookup were present. The
    // lookup is validated separately by LookupArgument::prove/verify on
    // the same witness/multiplicity int columns.
    //
    // This does not yet bind the LookupArgument's component evals to the
    // committed trace via the PCS opening — that binding is the remaining
    // protocol-wiring work (Phase 2d). What this test DOES validate:
    //   * The test UAIR (with a lookup spec declared) runs cleanly through
    //     the full protocol.
    //   * LookupArgument accepts an honest trace generated by the UAIR.
    //   * LookupArgument rejects when the multiplicity column is tampered.
    // -----------------------------------------------------------------------
    // Build a dynamic `F::Config` fresh from a blank transcript — any
    // valid cfg works for the standalone LookupArgument, since the
    // lookup argument uses a transcript disjoint from the protocol.
    fn sample_field_cfg() -> <F as crypto_primitives::PrimeField>::Config {
        use zinc_transcript::{Blake3Transcript, traits::Transcript};
        let mut ts = Blake3Transcript::new();
        ts.get_random_field_cfg::<F, crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS>, MillerRabin>()
    }

    fn lift_int_col(
        col: &zinc_poly::mle::DenseMultilinearExtension<ZtInt>,
        cfg: &<F as crypto_primitives::PrimeField>::Config,
    ) -> zinc_poly::mle::DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner> {
        use crypto_primitives::{FromWithConfig, PrimeField};
        let zero_inner = F::zero_with_cfg(cfg).into_inner();
        let evals: Vec<_> = col
            .evaluations
            .iter()
            .map(|v| {
                let signed: i64 = *v;
                assert!(signed >= 0, "witness/multiplicity must be non-negative");
                F::from_with_cfg(signed as u64, cfg).into_inner()
            })
            .collect();
        zinc_poly::mle::DenseMultilinearExtension::from_evaluations_vec(col.num_vars, evals, zero_inner)
    }

    fn build_identity_table(
        width: usize,
        cfg: &<F as crypto_primitives::PrimeField>::Config,
    ) -> zinc_poly::mle::DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner> {
        use crypto_primitives::{FromWithConfig, PrimeField};
        let size = 1usize << width;
        let zero_inner = F::zero_with_cfg(cfg).into_inner();
        let evals: Vec<_> = (0..size as u64)
            .map(|x| F::from_with_cfg(x, cfg).into_inner())
            .collect();
        zinc_poly::mle::DenseMultilinearExtension::from_evaluations_vec(width, evals, zero_inner)
    }

    // -----------------------------------------------------------------------
    // IntLookupUair: exercises the protocol alongside a standalone lookup
    // argument on the same committed trace.
    //
    // The existing protocol pipeline currently *ignores* lookup_specs in
    // `UairSignature`, so Step 1–7 run as if no lookup were present. The
    // lookup is validated separately by LookupArgument::prove/verify on
    // the same witness/multiplicity int columns.
    //
    // This does not yet bind the LookupArgument's component evals to the
    // committed trace via the PCS opening — that binding is the remaining
    // protocol-wiring work (Phase 2d).
    // -----------------------------------------------------------------------
    #[test]
    fn test_int_lookup_uair_protocol_runs() {
        let num_vars = INT_LOOKUP_TABLE_WIDTH;
        let mut rng = rng();

        let trace = IntLookupUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let sig = IntLookupUair::<ZtInt>::signature();
        let public_trace = trace.public(&sig);

        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (make_iprs(num_vars), make_iprs(num_vars), make_iprs(num_vars)),
        );
        let proof = ZincPlusPiop::<TestZincTypesIprs, IntLookupUair<ZtInt>, F, DEGREE_PLUS_ONE>::prove::<
            false, CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
            .expect("protocol prove must succeed on IntLookupUair");
        ZincPlusPiop::<TestZincTypesIprs, IntLookupUair<ZtInt>, F, DEGREE_PLUS_ONE>::verify::<
            _, CHECKED,
        >(&pp, proof, &public_trace, num_vars, project_scalar_fn, default_project_ideal!())
            .expect("protocol verify must succeed on IntLookupUair");
    }

    // Phase 2f soundness test: tampering the LookupArgument's component_evals
    // (post-prove) must be caught by the reducer's cross-check in step5
    // (witness_evals_at_rho_row vs component_evals).
    #[test]
    fn test_int_lookup_uair_tampered_component_evals_rejected() {
        let num_vars = INT_LOOKUP_TABLE_WIDTH;
        let mut rng = rng();

        let trace = IntLookupUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let sig = IntLookupUair::<ZtInt>::signature();
        let public_trace = trace.public(&sig);

        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (make_iprs(num_vars), make_iprs(num_vars), make_iprs(num_vars)),
        );
        let mut proof = ZincPlusPiop::<TestZincTypesIprs, IntLookupUair<ZtInt>, F, DEGREE_PLUS_ONE>::prove::<
            false, CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
            .expect("prove should succeed with honest trace");

        // Tamper: swap witness_evals[0] and multiplicity_eval in the
        // (single) lookup proof. With overwhelming probability the
        // two are distinct, so the reducer's cross-check catches the
        // mismatch.
        let w0 = proof.lookup_proof[0].component_evals.witness_evals[0].clone();
        let m = proof.lookup_proof[0].component_evals.multiplicity_eval.clone();
        proof.lookup_proof[0].component_evals.witness_evals[0] = m;
        proof.lookup_proof[0].component_evals.multiplicity_eval = w0;

        let result = ZincPlusPiop::<TestZincTypesIprs, IntLookupUair<ZtInt>, F, DEGREE_PLUS_ONE>::verify::<
            _, CHECKED,
        >(&pp, proof, &public_trace, num_vars, project_scalar_fn, default_project_ideal!());
        assert!(
            matches!(result.unwrap_err(), ProtocolError::Lookup(_)),
            "verifier must reject tampered component_evals via reducer cross-check"
        );
    }

    // Tampering the multiplicity column in the trace makes the lookup
    // identity fail. This exercises step4b_lookup inside the full
    // protocol pipeline.
    #[test]
    fn test_int_lookup_uair_tampered_trace_rejected_by_protocol() {
        let num_vars = INT_LOOKUP_TABLE_WIDTH;
        let mut rng = rng();

        let mut trace = IntLookupUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        // Tamper the multiplicity column (col index 1 of the int bucket).
        let int_cols = trace.int.to_mut();
        int_cols[1].evaluations[0] = int_cols[1].evaluations[0] + 1;

        let sig = IntLookupUair::<ZtInt>::signature();
        let public_trace = trace.public(&sig);

        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (make_iprs(num_vars), make_iprs(num_vars), make_iprs(num_vars)),
        );
        // The prover still builds a proof (it just proves the wrong thing).
        let proof = ZincPlusPiop::<TestZincTypesIprs, IntLookupUair<ZtInt>, F, DEGREE_PLUS_ONE>::prove::<
            false, CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
            .expect("tampered-trace prove should still produce a (soundness-invalid) proof");

        // Verifier must reject at step4b_lookup_verify.
        let result = ZincPlusPiop::<TestZincTypesIprs, IntLookupUair<ZtInt>, F, DEGREE_PLUS_ONE>::verify::<
            _, CHECKED,
        >(&pp, proof, &public_trace, num_vars, project_scalar_fn, default_project_ideal!());
        assert!(
            matches!(result.unwrap_err(), ProtocolError::Lookup(_)),
            "verifier must reject tampered multiplicities via step4b"
        );
    }

    #[test]
    fn test_int_lookup_uair_lookup_argument_accepts_honest_trace() {
        use zinc_piop::lookup::logup_gkr::LookupArgument;
        use zinc_transcript::Blake3Transcript;

        let num_vars = INT_LOOKUP_TABLE_WIDTH;
        let mut rng = rng();
        let trace = IntLookupUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);

        let cfg = sample_field_cfg();
        let wit_col = lift_int_col(&trace.int[0], &cfg);
        let mul_col = lift_int_col(&trace.int[1], &cfg);
        let table_mle = build_identity_table(INT_LOOKUP_TABLE_WIDTH, &cfg);

        let mut p_ts = Blake3Transcript::new();
        let (lookup_proof, _) =
            LookupArgument::<F>::prove(&mut p_ts, &[&wit_col], &table_mle, &mul_col, &cfg);

        let mut v_ts = Blake3Transcript::new();
        LookupArgument::<F>::verify(&mut v_ts, 1, INT_LOOKUP_TABLE_WIDTH, &lookup_proof, &cfg)
            .expect("honest lookup must verify");
    }

    #[test]
    fn test_int_lookup_uair_tampered_multiplicity_rejected() {
        use zinc_piop::lookup::logup_gkr::{LookupArgument, LookupArgumentError};
        use zinc_transcript::Blake3Transcript;

        let num_vars = INT_LOOKUP_TABLE_WIDTH;
        let mut rng = rng();
        let mut trace = IntLookupUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);

        // Tamper: increment the first multiplicity count.
        let tampered = trace.int.to_mut();
        tampered[1].evaluations[0] = tampered[1].evaluations[0] + 1;

        let cfg = sample_field_cfg();
        let wit_col = lift_int_col(&trace.int[0], &cfg);
        let mul_col = lift_int_col(&trace.int[1], &cfg);
        let table_mle = build_identity_table(INT_LOOKUP_TABLE_WIDTH, &cfg);

        let mut p_ts = Blake3Transcript::new();
        let (lookup_proof, _) =
            LookupArgument::<F>::prove(&mut p_ts, &[&wit_col], &table_mle, &mul_col, &cfg);

        let mut v_ts = Blake3Transcript::new();
        let result =
            LookupArgument::<F>::verify(&mut v_ts, 1, INT_LOOKUP_TABLE_WIDTH, &lookup_proof, &cfg);
        match result {
            Err(LookupArgumentError::NonzeroRootNumerator) => {}
            other => panic!("expected NonzeroRootNumerator, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Multi-column lookup: IntLookupMultiUair declares two int columns
    // (v_0, v_1) against the same Word{width:8} table; `group_lookup_specs`
    // collapses them into ONE group with column_indices = [0, 1]. The
    // multiplicity column (v_2) is shared.
    //
    // These tests exercise the `L > 1` path in step4b on both sides,
    // plus the per-column cross-check in step5.
    // -----------------------------------------------------------------------
    #[test]
    fn test_int_lookup_multi_uair_protocol_runs() {
        let num_vars = INT_LOOKUP_TABLE_WIDTH + 1; // 512 rows, 2 witness cols
        let mut rng = rng();

        let trace = IntLookupMultiUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let sig = IntLookupMultiUair::<ZtInt>::signature();
        let public_trace = trace.public(&sig);

        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (make_iprs(num_vars), make_iprs(num_vars), make_iprs(num_vars)),
        );
        let proof = ZincPlusPiop::<TestZincTypesIprs, IntLookupMultiUair<ZtInt>, F, DEGREE_PLUS_ONE>::prove::<
            false, CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
            .expect("protocol prove must succeed on IntLookupMultiUair");
        ZincPlusPiop::<TestZincTypesIprs, IntLookupMultiUair<ZtInt>, F, DEGREE_PLUS_ONE>::verify::<
            _, CHECKED,
        >(&pp, proof, &public_trace, num_vars, project_scalar_fn, default_project_ideal!())
            .expect("protocol verify must succeed on IntLookupMultiUair");
    }

    // Tampering v_0 only (leave v_1 and m honest) breaks the shared-group
    // cumulative sum: v_0's fake entry isn't covered by m. The verifier
    // must reject at step4b.
    #[test]
    fn test_int_lookup_multi_uair_tampered_v0_rejected() {
        let num_vars = INT_LOOKUP_TABLE_WIDTH + 1;
        let mut rng = rng();

        let mut trace = IntLookupMultiUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        // Push v_0[0] to a value that makes the multiplicity wrong.
        let int_cols = trace.int.to_mut();
        int_cols[0].evaluations[0] = int_cols[0].evaluations[0] + 1;

        let sig = IntLookupMultiUair::<ZtInt>::signature();
        let public_trace = trace.public(&sig);

        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (make_iprs(num_vars), make_iprs(num_vars), make_iprs(num_vars)),
        );
        let proof = ZincPlusPiop::<TestZincTypesIprs, IntLookupMultiUair<ZtInt>, F, DEGREE_PLUS_ONE>::prove::<
            false, CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
            .expect("tampered prove should still produce a (soundness-invalid) proof");

        let result = ZincPlusPiop::<TestZincTypesIprs, IntLookupMultiUair<ZtInt>, F, DEGREE_PLUS_ONE>::verify::<
            _, CHECKED,
        >(&pp, proof, &public_trace, num_vars, project_scalar_fn, default_project_ideal!());
        assert!(
            matches!(result.unwrap_err(), ProtocolError::Lookup(_)),
            "verifier must reject tampered v_0 via step4b"
        );
    }

    // Tampering the second witness column's component_eval entry
    // post-prove must be caught by the reducer's per-column cross-check
    // (witness_evals[1] vs reducer.witness_evals_at_rho_row[g][1]).
    #[test]
    fn test_int_lookup_multi_uair_tampered_second_component_eval_rejected() {
        let num_vars = INT_LOOKUP_TABLE_WIDTH + 1;
        let mut rng = rng();

        let trace = IntLookupMultiUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let sig = IntLookupMultiUair::<ZtInt>::signature();
        let public_trace = trace.public(&sig);

        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (make_iprs(num_vars), make_iprs(num_vars), make_iprs(num_vars)),
        );
        let mut proof = ZincPlusPiop::<TestZincTypesIprs, IntLookupMultiUair<ZtInt>, F, DEGREE_PLUS_ONE>::prove::<
            false, CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
            .expect("honest prove should succeed");

        // Swap witness_evals[0] and witness_evals[1] in the (single)
        // lookup proof. With overwhelming probability they are distinct,
        // so the step5 per-column cross-check catches the mismatch on [1].
        let w0 = proof.lookup_proof[0].component_evals.witness_evals[0].clone();
        let w1 = proof.lookup_proof[0].component_evals.witness_evals[1].clone();
        proof.lookup_proof[0].component_evals.witness_evals[0] = w1;
        proof.lookup_proof[0].component_evals.witness_evals[1] = w0;

        let result = ZincPlusPiop::<TestZincTypesIprs, IntLookupMultiUair<ZtInt>, F, DEGREE_PLUS_ONE>::verify::<
            _, CHECKED,
        >(&pp, proof, &public_trace, num_vars, project_scalar_fn, default_project_ideal!());
        assert!(
            matches!(result.unwrap_err(), ProtocolError::Lookup(_)),
            "verifier must reject swapped witness_evals via per-column cross-check"
        );
    }
}
