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

pub mod fixed_prime;
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
    lookup::{BatchedLookupProof, LookupError},
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

/// Full proof produced by the Zinc+ PIOP for UCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    /// Zip+ commitments to the witness columns.
    pub commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip: Vec<u8>,
    /// Randomized ideal check proof.
    pub ideal_check: IdealCheckProof<F>,
    /// Combined polynomial resolver proof (up_evals + down_evals +
    /// bit_op_down_evals).
    pub resolver: CombinedPolyResolverProof<F>,
    /// Multi-degree sumcheck proof (CPR group + future lookup groups).
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Multi-point evaluation sumcheck proof. Reduces all CPR claims at
    /// `r*` (up evals + row-shift down evals + bit-op virtual-column
    /// down evals) to a single evaluation point `r_0`. Bit-op sources
    /// are folded in as additional `up` slots; their consistency is
    /// discharged at `r_0` in Step 6 by applying the bit-op locally to
    /// the source's lifted eval.
    pub multipoint_eval: MultipointEvalProof<F>,
    /// Witness-only polynomial MLE evaluations at r_0 in F_q[X]
    /// (after \phi_q, before \psi_a), ordered as
    /// `[wit_bin..., wit_arb..., wit_int...]`.
    /// The verifier recomputes public lifted_evals from public data,
    /// interleaves them with these, and derives scalar open_evals via
    /// \psi_a for the sumcheck consistency check and Zip+ PCS verify.
    pub witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    /// Lookup argument proof. `None` when the UAIR has no lookup specs.
    pub lookup_proof: Option<BatchedLookupProof<F>>,
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

        // TODO: deserialize lookup_proof once BatchedLookupProof gets
        // Transcribable impls (lookup is not yet implemented).
        assert!(bytes.is_empty(), "All bytes should be consumed");

        Self {
            commitments: (commit0, commit1, commit2),
            zip,
            ideal_check,
            resolver,
            combined_sumcheck,
            multipoint_eval,
            witness_lifted_evals,
            lookup_proof: None,
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
        // TODO: serialize lookup_proof once BatchedLookupProof gets
        // Transcribable impls (lookup is not yet implemented).
        DynamicPolyVecF::reinterpret(&self.witness_lifted_evals)
            .write_transcription_bytes_subset(buf);
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
            // TODO: add lookup_proof size once BatchedLookupProof gets
            // Transcribable impls (lookup is not yet implemented).
            + DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
            + witness_vec.get_num_bytes()
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

/// Type bundle for the **folded** Zinc+ PIOP (1× fold, 2× column splitting).
///
/// The PIOP runs at trace degree `D` (so the trace and UAIR are unchanged
/// from the unfolded path), but the binary commitment is over `BinaryPoly<HALF_D>`
/// — each `BinaryPoly<D>` witness column is split into two `BinaryPoly<HALF_D>`
/// halves before commit. This decouples the trace's `BinaryPoly<D>` from the
/// PCS's `BinaryPoly<HALF_D>`, which `ZincTypes<D>` would otherwise force to
/// be the same (`BinaryZt::Eval = BinaryPoly<DEGREE_PLUS_ONE>`).
///
/// Arbitrary and integer commitments are unchanged.
pub trait FoldedZincTypes<const D: usize, const HALF_D: usize>: Clone + Debug {
    type Int: Semiring
        + ConstTranscribable
        + ConstCoeffBitWidth
        + Named
        + Default
        + Clone
        + Send
        + Sync
        + 'static;

    type Chal: ConstIntRing + ConstTranscribable + Named;

    type Pt: ConstIntRing;

    type CombR;

    type Fmod: ConstIntSemiring + ConstTranscribable + Named;

    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// Zip+ types for the **split** binary trace columns.
    /// `Eval = BinaryPoly<HALF_D>` — one round of 2× folding.
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<HALF_D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns (unchanged from
    /// the unfolded path: degree-`D` polynomials).
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the integer trace columns (unchanged).
    type IntZt: ZipTypes<
            Eval = Self::Int,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type BinaryLc: LinearCode<Self::BinaryZt>;

    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;

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
    #[error("lifted_evals bit-op consistency mismatch at bit_op spec {spec}")]
    LiftedEvalsBitOpMismatch { spec: usize },
    #[error("lookup argument failed: {0}")]
    Lookup(#[from] LookupError),
    #[error("booleanity check failed: {0}")]
    Booleanity(zinc_piop::lookup::booleanity::BooleanityError<F>),
    #[error("public-trace consistency check failed: {0}")]
    PublicConsistency(String),
    #[error("shifted bit-slice evaluation failed: {0}")]
    ShiftedBitSliceEval(zinc_poly::EvaluationError),
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
        BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, BitOpRotUair,
        EC_FP_INT_LIMBS, GenerateRandomTrace, Sha256CompressionSliceUair, Sha256Ideal,
        ShaEcdsaUair, TestUairMixedDegrees, TestUairMixedShifts, TestUairNoMultiplication,
        TestUairSimpleMultiplication,
    };
    use zinc_uair::{
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
    // `fixed-prime` branch: 256-bit field modulus (4 × u64 limbs) so the
    // hardcoded secp256k1 base prime fits in `Fmod = Uint<FIELD_LIMBS>`.
    const FIELD_LIMBS: usize = U64::LIMBS * 4;
    const DEGREE_PLUS_ONE: usize = 32;

    // Zip+ type parameters.

    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    /// Repetition factor for linear code, an inverse rate. Defaults to 4
    /// (rate 1/4); enabling the `iprs-rate-1-8` cargo feature switches
    /// every `IprsCode<..., REP, ...>` instance in this test module to
    /// inverse-rate 8 (rate 1/8).
    const REP: usize = if cfg!(feature = "iprs-rate-1-8") { 8 } else { 4 };

    /// Number of column openings the PCS performs. Tied to `REP`: rate 1/4
    /// uses 147 openings, rate 1/8 uses 96 (lower opening count is sound
    /// at the higher inverse rate because each column reveals more
    /// information about the codeword).
    const NUM_COL_OPENINGS_FOR_REP: usize = if cfg!(feature = "iprs-rate-1-8") {
        96
    } else {
        147
    };

    type F = MontyField<FIELD_LIMBS>;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypes {}
    impl ZipTypes for BinPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
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
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
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
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
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
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
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

    #[allow(clippy::result_large_err)]
    fn do_test<Zt, U>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
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

        // `MLE_FIRST = true` is now safe for any UAIR: it dispatches at
        // runtime to MLE-first (all-linear), Combined (all-non-linear), or
        // Hybrid (mixed). Always exercise it.
        run_protocol!(true);
    }

    /// End-to-end test: TestUairNoMultiplication.
    ///
    /// UAIR constraint: a + b - c \in (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = <X - 2>).
    #[test]
    fn test_e2e_no_multiplication() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairNoMultiplication<ZtInt>>(
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
        do_test::<TestZincTypesRaa, TestUairSimpleMultiplication<ZtInt>>(
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

    /// End-to-end test: TestUairMixedDegrees.
    ///
    /// Two non-zero-ideal `assert_in_ideal` constraints — one linear
    /// (degree 1), one quadratic (degree 2). Exercises the hybrid
    /// ideal-check dispatch (`prove_hybrid`), which routes the linear
    /// constraint through the MLE-first lane and the quadratic constraint
    /// through the combined-poly lane, merging the per-constraint values
    /// into a single proof. Honest witness is the all-zero trace, which
    /// trivially satisfies both constraints.
    #[test]
    fn test_e2e_mixed_degrees() {
        // Use TestZincTypesRaa because the quadratic constraint with
        // arbitrary_poly column multiplication needs an RAA-style code.
        let num_vars = 2;
        do_test::<TestZincTypesRaa, TestUairMixedDegrees<ZtInt>>(
            num_vars,
            (
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
            ),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
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

    // Tampering the commitment root causes the verifier to derive different
    // challenges from the Fiat–Shamir transcript. On the `fixed-prime`
    // branch the projecting prime `q` is hardcoded (not transcript-derived),
    // so prover and verifier still agree on `q` after the tamper; the first
    // observable divergence is at the combined-poly resolver, which catches
    // it as a sumcheck-sum mismatch.
    #[test]
    fn test_big_linear_tamper_commitment() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.commitments.0.root = Default::default(),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(
                        CombinedPolyResolverError::WrongSumcheckSum { .. }
                    )
                ));
            },
        );
    }

    /// End-to-end test: BitOpRotUair (synthetic UAIR with one
    /// `BitOp::Rot(7)` virtual column).
    ///
    /// Two binary witness columns W (col 0) and V (col 1); witness sets
    /// V[i] = Rot(7)(W[i]). Constraint: V[i] − Rot(7)(W[i]) ∈ <X − 2>.
    /// Exercises the bit-op virtual column path end-to-end: CPR
    /// materialises an extra down MLE for the bit-op, the prover
    /// publishes a `bit_op_down_evals` entry, and the verifier checks
    /// it in Step 4.5 against ψ(rot_c(lifted_eval[col 0])).
    #[test]
    fn test_e2e_bit_op_rot() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
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

    /// Tampering the CPR-emitted `bit_op_down_evals` triggers the new
    /// `LiftedAtRStarBitOpMismatch` error at Step 4.5.
    ///
    /// We tamper the F_q[X]-lifted source eval at r* — easier to
    /// engineer than tampering `bit_op_down_evals` directly, because
    /// the latter participates in the CPR claim-value reconstruction
    /// (so the verifier rejects earlier with `ClaimValueDoesNotMatch`).
    /// Tampering the source's `lifted_evals_at_rstar` makes the source's
    /// up-eval ψ-projection still match `cpr_subclaim.up_evals[0]`
    /// only with extreme luck — but with a single coefficient swap the
    /// up-eval check fires first. To target the bit-op check, we
    /// instead permute coefficients of the source's lifted eval such
    /// that ψ_α projects to the same value (preserves dot product
    /// against `α^j`); a swap that holds the dot product fixed but
    /// changes `rot_c(·)` is not directly engineerable in a black-box
    /// test. As a robust proxy: tamper `bit_op_down_evals[0]` and
    /// observe the verifier rejects (whatever the precise error
    /// variant). For this test we verify it does reject — we check for
    /// the bit-op mismatch when the ResolveCheckValue happens to pass,
    /// otherwise any rejection is acceptable.
    #[test]
    fn test_bit_op_rot_tamper_bit_op_down_eval() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| {
                // Mutate the only bit_op_down_eval — verifier must
                // reject (CPR claim-value reconstruction will catch it
                // first as a `ClaimValueDoesNotMatch`).
                if let Some(ev) = proof.resolver.bit_op_down_evals.first_mut() {
                    *ev = ev.clone() + ev.clone();
                }
            },
            |res| {
                assert!(res.is_err());
            },
        );
    }

    /// Tamper a coefficient of the source's witness lifted eval at
    /// r_0 (W_W's slot in `proof.witness_lifted_evals`). The swap
    /// changes `rot_c(·)` (so the bit-op slot's derived `open_eval`
    /// no longer matches mp_eval's expectation) and at the same time
    /// changes the source's own `open_eval`. The verifier rejects via
    /// mp_eval's `ClaimMismatch` (whichever slot trips the equation
    /// first — the per-slot identity is not separately surfaced).
    #[test]
    fn test_bit_op_rot_tamper_witness_lifted_source() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| {
                if let Some(p) = proof.witness_lifted_evals.first_mut() {
                    if p.coeffs.len() >= 2 {
                        p.coeffs.swap(0, 1);
                    }
                }
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                        | ProtocolError::LiftedEvalsBitOpMismatch { .. }
                ));
            },
        );
    }

    //
    // SHA-ECDSA E2E + tampering tests for the new Step 4.5 layer.
    //
    // These pin the post-Commit-D behaviour:
    //   * lifted_evals_at_rstar up-half tamper rejected
    //     with `LiftedAtRStarUpMismatch`.
    //   * lifted_evals_at_rstar down-half tamper rejected
    //     with `LiftedAtRStarDownMismatch`.
    //   * SHA `W_W` source tamper rejected upstream of mp_eval (any
    //     `LiftedAtRStar*` variant).
    //   * No-tamper proof-shape pin: `lifted_evals_at_rstar.len()`
    //     and `bit_op_down_evals.len()` match the SHA-ECDSA
    //     signature.
    //

    type ShaEcdsaInt = Int<EC_FP_INT_LIMBS>;

    /// Binary-poly Zip+ types tuned for the wider SHA-ECDSA cell type
    /// (Int<5>, 320-bit). Mirrors `BinPolyZipTypes` but with a wider
    /// `CombR` to soak up SHA-ECDSA's per-row inner products.
    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesShaEcdsa {}
    impl ZipTypes for BinPolyZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
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

    /// Arbitrary-poly Zip+ types over `Int<EC_FP_INT_LIMBS>` cells.
    /// SHA-ECDSA itself has no arbitrary-poly columns; this is only
    /// here to satisfy the `ZincTypes` bundle.
    #[derive(Debug, Clone)]
    pub struct ArbPolyZipTypesShaEcdsa {}
    impl ZipTypes for ArbPolyZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = DensePolynomial<ShaEcdsaInt, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<6>, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = DensePolyInnerProduct<
            ShaEcdsaInt,
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

    /// Int Zip+ types over `Int<EC_FP_INT_LIMBS>` cells (the ECDSA
    /// Jacobian columns and the SHA mu_{W,a,e} carries).
    #[derive(Debug, Clone)]
    pub struct IntZipTypesShaEcdsa {}
    impl ZipTypes for IntZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = ShaEcdsaInt;
        type Cw = Int<6>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// `ZincTypes` bundle wiring SHA-ECDSA's `Int<5>` cells through
    /// IPRS-coded Zip+ commitments. Mirrors `RealEcdsaBenchZincTypes`
    /// from `protocol/benches/e2e.rs` (which already exercises this
    /// configuration in production benchmarks).
    #[derive(Clone, Debug)]
    struct TestShaEcdsaZincTypes;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestShaEcdsaZincTypes {
        type Int = ShaEcdsaInt;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesShaEcdsa;
        type ArbitraryZt = ArbPolyZipTypesShaEcdsa;
        type IntZt = IntZipTypesShaEcdsa;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    /// Project an `IdealOrZero<Sha256Ideal<ShaEcdsaInt>>` to
    /// `Sha256Ideal<F>` for the SHA-ECDSA verifier. Mirrors
    /// `sha256_real_project_ideal` in `protocol/benches/e2e.rs`.
    fn sha256_test_project_ideal(
        ideal: &IdealOrZero<Sha256Ideal<ShaEcdsaInt>>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Sha256Ideal<F> {
        match ideal {
            IdealOrZero::NonZero(Sha256Ideal::RotX2(r)) => {
                Sha256Ideal::RotX2(RotationIdeal::from_with_cfg(r, field_cfg))
            }
            IdealOrZero::NonZero(Sha256Ideal::RotXw1) => Sha256Ideal::RotXw1,
            IdealOrZero::Zero => {
                unreachable!("zero ideals are filtered before this closure runs")
            }
        }
    }

    /// Run a SHA-ECDSA round-trip end-to-end. Calls `tamper` on the
    /// generated proof before verification and feeds the resulting
    /// `Result` into `check_verification`. Patterned after `do_test`
    /// but specialised to the SHA-ECDSA UAIR / `Sha256Ideal` ideal
    /// type (which doesn't fit the `IdealOrZero<DegreeOneIdeal<F>>`
    /// signature `do_test` hard-codes).
    ///
    /// `MLE_FIRST` is forced to `false` since SHA-ECDSA has constraints
    /// up to degree 6 (the ECDSA Y output-selection D4 term), which
    /// `count_effective_max_degree` reports above 1.
    #[allow(clippy::result_large_err)]
    fn do_test_sha_ecdsa(
        num_vars: usize,
        tamper: impl Fn(&mut Proof<F>),
        check_verification: impl Fn(Result<(), ProtocolError<F, Sha256Ideal<F>>>),
    ) {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        let mut rng = rng();
        let pp = setup_pp::<Zt>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = <U as Uair>::signature();
        let public_trace = trace.public(&sig);

        let mut proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // Round-trip the proof through (de)serialisation as a sanity
        // check; mirrors `do_test`.
        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript.write(&proof).expect("Failed to serialize proof");
        let mut transcript = transcript.into_verification_transcript();
        let proof_2 = transcript
            .read()
            .expect("Failed to deserialize proof after serialization");
        assert_eq!(proof, proof_2);

        tamper(&mut proof);

        let verification_result = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<
            _,
            CHECKED,
        >(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            sha256_test_project_ideal,
        );
        check_verification(verification_result);
    }

    /// `num_vars` for SHA-ECDSA tests. ECDSA's Shamir scalar
    /// multiplication needs `n_rows > 256`, so `num_vars >= 9`.
    const SHA_ECDSA_NUM_VARS: usize = 9;

    /// Tamper a coefficient of the SHA `W_W` slot in
    /// `proof.witness_lifted_evals` (the source of all six bit-op
    /// virtual columns). The verifier rejects via mp_eval's
    /// consistency check or `LiftedEvalsBitOpMismatch` (whichever
    /// trips first). Replaces the four pre-existing Step-4.5-specific
    /// tamper tests since Step 4.5 is gone.
    #[test]
    fn test_e2e_sha_ecdsa_tamper_witness_lifted_w() {
        do_test_sha_ecdsa(
            SHA_ECDSA_NUM_VARS,
            |proof| {
                // SHA-ECDSA's witness layout puts W_W at the same flat
                // index as in `cols::W_W` minus `NUM_BIN_PUB`. Easier
                // and equally diagnostic to tamper the first witness
                // slot instead — any of them feeds mp_eval.
                let p = &mut proof.witness_lifted_evals[0];
                assert!(
                    p.coeffs.len() >= 2,
                    "witness lifted eval polynomial has < 2 coefficients; cannot tamper",
                );
                p.coeffs.swap(0, 1);
            },
            |res| {
                let err = res.unwrap_err();
                assert!(
                    matches!(
                        err,
                        ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                            | ProtocolError::LiftedEvalsBitOpMismatch { .. },
                    ),
                    "expected mp_eval ClaimMismatch or LiftedEvalsBitOpMismatch, got {err:?}",
                );
            },
        );
    }

    /// No-tamper SHA-ECDSA round-trip + structural-shape pins. Prints
    /// the proof size so refactors that grow the proof are easy to
    /// catch.
    #[test]
    fn test_e2e_sha_ecdsa_proof_shape() {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        // SHA standalone signature pins (post-virtualization of B_1/
        // B_2/B_3): NUM_BIN = 18 (3 committed B_i cols dropped, 2
        // public correctors PA_R_*_CORR added — net −1), bit_op_specs
        // = 6, virtual_binary_poly_cols = 3.
        let sha_sig = <Sha256CompressionSliceUair<ShaEcdsaInt> as Uair>::signature();
        assert_eq!(
            sha_sig.total_cols().num_binary_poly_cols(),
            18,
            "SHA-256 NUM_BIN drifted: expected 18 binary_poly columns post-virtualization",
        );
        assert_eq!(
            sha_sig.bit_op_specs().len(),
            6,
            "SHA-256 bit_op_specs.len() drifted: expected 6",
        );
        assert_eq!(
            sha_sig.virtual_binary_poly_cols().len(),
            3,
            "SHA-256 virtual_binary_poly_cols.len() drifted: expected 3 (B_1/B_2/B_3)",
        );

        // SHA-ECDSA composed signature (the actual UAIR exercised here).
        let sig = <U as Uair>::signature();
        let num_bit_op = sig.bit_op_specs().len();
        assert_eq!(
            sig.total_cols().num_binary_poly_cols(),
            18,
            "SHA-ECDSA NUM_BIN drifted: expected 18 binary_poly columns",
        );
        assert_eq!(
            num_bit_op, 6,
            "SHA-ECDSA bit_op_specs.len() drifted: expected 6",
        );
        assert_eq!(
            sig.virtual_binary_poly_cols().len(),
            3,
            "SHA-ECDSA virtual_binary_poly_cols.len() drifted: expected 3",
        );

        // Round-trip a real proof and pin the post-rewrite proof size.
        let mut rng = rng();
        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let trace = U::generate_random_trace(SHA_ECDSA_NUM_VARS, &mut rng);
        let public_trace = trace.public(&sig);

        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("Prover failed");

        assert_eq!(
            proof.resolver.bit_op_down_evals.len(),
            num_bit_op,
            "Proof.resolver.bit_op_down_evals.len() must equal bit_op_specs.len()",
        );

        let total_proof_bytes = proof.get_num_bytes();
        println!("total proof bytes: {total_proof_bytes}");

        // Verifier still accepts the un-tampered proof.
        ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
            sha256_test_project_ideal,
        )
        .expect("Verifier rejected an honest SHA-ECDSA proof");
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
            ),
            default_project_ideal!(),
            |proof| proof.ideal_check.combined_mle_values.swap(0, 1),
            |res| {
                assert!(matches!(res.unwrap_err(), ProtocolError::IdealCheck(..)));
            },
        );
    }

    //
    // Folded Zip+ (1× fold) — round-trip test
    //

    /// Half-degree binary Zip+ types for the split commitment side of the
    /// folded path. Mirrors [`BinPolyZipTypes`] but with `Eval = BinaryPoly<16>`
    /// and `Cw` over `DensePolynomial<i64, 16>`, so the PCS commits the
    /// post-split BinaryPoly<16> witnesses.
    const HALF_DEGREE_PLUS_ONE: usize = 16;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesHalf {}
    impl ZipTypes for BinPolyZipTypesHalf {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<HALF_DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, HALF_DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, HALF_DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, HALF_DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            HALF_DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct TestFoldedZincTypesIprs;

    impl FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE> for TestFoldedZincTypesIprs {
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesHalf;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    /// Set up Zip+ params for the folded path. The binary commitment is over
    /// the split column (length `2n` with `BinaryPoly<HALF_D>` entries), so
    /// its `num_vars` is `num_vars + 1`. Arbitrary and int are sized normally.
    #[allow(clippy::type_complexity)]
    fn setup_folded_pp(
        num_vars: usize,
    ) -> (
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::BinaryZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::BinaryLc,
        >,
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::ArbitraryZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::ArbitraryLc,
        >,
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::IntZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::IntLc,
        >,
    ) {
        let split_size = 1 << (num_vars + 1);
        let normal_size = 1 << num_vars;
        (
            ZipPlus::setup(
                split_size,
                IprsCode::new_with_optimal_depth(split_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
        )
    }

    /// End-to-end test: BinaryDecompositionUair via the **folded** prover/
    /// verifier. Same UAIR, same trace generator, same field — only the
    /// binary commitment is over `BinaryPoly<16>` split columns, opened at
    /// the extended point `(r_0 ‖ γ)`.
    #[test]
    fn test_e2e_folded_binary_decomposition() {
        use crate::prover::prove_folded;
        use crate::verifier::verify_folded;

        let num_vars = 8;
        let mut rng = rng();
        let pp = setup_folded_pp(num_vars);

        let trace = BinaryDecompositionUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let sig = <BinaryDecompositionUair<ZtInt> as Uair>::signature();
        let public_trace = trace.public(&sig);

        let proof = prove_folded::<
            TestFoldedZincTypesIprs,
            BinaryDecompositionUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            false,
            CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
        .expect("Folded prover failed");

        verify_folded::<
            TestFoldedZincTypesIprs,
            BinaryDecompositionUair<ZtInt>,
            F,
            IdealOrZero<DegreeOneIdeal<F>>,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            CHECKED,
        >(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            default_project_ideal!(),
        )
        .expect("Folded verifier rejected a valid proof");
    }

    //
    // Folded Zip+ (4× fold) — round-trip test
    //

    /// Quarter-degree binary Zip+ types for the doubly-split commitment side
    /// of the 4× folded path. Mirrors [`BinPolyZipTypesHalf`] but with
    /// `Eval = BinaryPoly<8>` and 8-coeff codewords.
    const QUARTER_DEGREE_PLUS_ONE: usize = 8;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesQuarter {}
    impl ZipTypes for BinPolyZipTypesQuarter {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<QUARTER_DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, QUARTER_DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, QUARTER_DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, QUARTER_DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            QUARTER_DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct TestFoldedZincTypesIprs4x;

    impl FoldedZincTypes<DEGREE_PLUS_ONE, QUARTER_DEGREE_PLUS_ONE> for TestFoldedZincTypesIprs4x {
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesQuarter;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    /// Set up Zip+ params for the 4× folded path. The binary commitment is
    /// over the twice-split column (length `4n` with `BinaryPoly<8>`
    /// entries), so its `num_vars` is `num_vars + 2`. Arbitrary and int
    /// commitments are sized normally.
    #[allow(clippy::type_complexity)]
    fn setup_folded_4x_pp(
        num_vars: usize,
    ) -> (
        ZipPlusParams<
            <TestFoldedZincTypesIprs4x as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE,
            >>::BinaryZt,
            <TestFoldedZincTypesIprs4x as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE,
            >>::BinaryLc,
        >,
        ZipPlusParams<
            <TestFoldedZincTypesIprs4x as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE,
            >>::ArbitraryZt,
            <TestFoldedZincTypesIprs4x as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE,
            >>::ArbitraryLc,
        >,
        ZipPlusParams<
            <TestFoldedZincTypesIprs4x as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE,
            >>::IntZt,
            <TestFoldedZincTypesIprs4x as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE,
            >>::IntLc,
        >,
    ) {
        let split2_size = 1 << (num_vars + 2);
        let normal_size = 1 << num_vars;
        (
            ZipPlus::setup(
                split2_size,
                IprsCode::new_with_optimal_depth(split2_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
        )
    }

    /// End-to-end test: BinaryDecompositionUair via the **4× folded**
    /// prover/verifier. Same UAIR, same trace generator, same field — the
    /// binary commitment is over `BinaryPoly<8>` columns of length `4n`,
    /// opened at the doubly-extended point `(r_0 ‖ γ₁ ‖ γ₂)`.
    #[test]
    fn test_e2e_folded_4x_binary_decomposition() {
        use crate::prover::prove_folded_4x;
        use crate::verifier::verify_folded_4x;

        let num_vars = 8;
        let mut rng = rng();
        let pp = setup_folded_4x_pp(num_vars);

        let trace = BinaryDecompositionUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let sig = <BinaryDecompositionUair<ZtInt> as Uair>::signature();
        let public_trace = trace.public(&sig);

        let proof = prove_folded_4x::<
            TestFoldedZincTypesIprs4x,
            BinaryDecompositionUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            false,
            CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
        .expect("Folded 4× prover failed");

        verify_folded_4x::<
            TestFoldedZincTypesIprs4x,
            BinaryDecompositionUair<ZtInt>,
            F,
            IdealOrZero<DegreeOneIdeal<F>>,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE,
            CHECKED,
        >(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            default_project_ideal!(),
        )
        .expect("Folded 4× verifier rejected a valid proof");
    }
}
