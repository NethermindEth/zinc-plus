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
//! - Combined CPR + Booleanity + Lookup multi-degree sumcheck (CPR group at
//!   degree `max_deg+2`; optional booleanity group at degree 3 when the UAIR
//!   has witness binary-poly columns; one lookup group per table type; shared
//!   eval point `r*`)
//! - $\alpha'$ bridge: squeeze a fresh challenge $\alpha'$ after the booleanity
//!   `bit_slice_evals` are absorbed, and append one extra $\alpha'$-projected
//!   MLE + up-eval to the multipoint-eval inputs per witness binary-poly column
//!   (see `BooleanityChecker`)
//! - Multi-point evaluation sumcheck (combines up/down evals at `r*` into a
//!   single evaluation point `r_0`)
//! - Lift-and-project (unprojected MLE evaluations at `r_0`)
//! - Zip+ PCS open/verify at `r_0`

pub mod constraint_system;
pub mod fold;
pub mod prover;
pub mod r1cs_frontend;
pub mod r1cs_sparse_matrix;
pub mod shared_challenge;
pub mod uair_frontend;
pub mod verifier;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{constraint_system::Layout, fold::FoldTrace};
use crypto_primitives::{ConstIntRing, ConstIntSemiring, FromWithConfig, PrimeField, Semiring};
use std::{fmt::Debug, iter, marker::PhantomData};
use thiserror::Error;
use zinc_piop::{
    combined_poly_resolver::{CombinedPolyResolverError, Proof as CombinedPolyResolverProof},
    ideal_check::{IdealCheckError, Proof as IdealCheckProof},
    lookup::{LookupError, booleanity::BooleanityError},
    multipoint_eval::MultipointEvalError,
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
use zinc_utils::{cfg_extend, cfg_into_iter, cfg_iter, from_ref::FromRef, named::Named, powers};
use zip_plus::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlusCommitment, ZipTypes},
};

//
// Data structures
//

/// Full proof produced by the Zinc+ PIOP for UCS.
///
/// # Lifted-eval families
///
/// Witness lifted evals are sent **per family**: for each of the $n + 2$
/// families (Q[X] / $q_0$, the declared $q_1, \dots, q_n$, and the
/// PCS-only $q''$), the prover sends a vector of `DynamicPolynomialF<F>`
/// carrying the per-family coefficient lift of each witness column. The
/// verifier reads each family's lifts under that family's field cfg, no
/// per-coefficient `from_with_cfg` projection is needed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F: PrimeField, CP = crate::uair_frontend::UairConstraintProof<F>> {
    /// Zip+ commitments to the witness columns.
    pub commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip: Vec<u8>,
    /// Witness-only polynomial MLE evaluations at $r_0$, **per constraint
    /// family**.
    ///
    /// Indexing follows the standard family convention used throughout
    /// the protocol:
    /// * `witness_lifted_evals[0]` — Q[X] family under $q_0$, $\bar
    ///   u_j^{(0)}(X) = \sum_b \mathrm{eq}(b, r_0^{(0)}) \cdot u_j(b) \in
    ///   F_{q_0}[X]$.
    /// * `witness_lifted_evals[i]` for $i \in 1..=n$ — the $i$-th declared
    ///   prime family from [`zinc_uair::UairSignature::primes`], lifted into
    ///   $F_{q_i}[X]$ at $r_0$ projected mod $q_i$.
    ///
    /// Length is `n + 1` where `n = primes().len()`. Each inner Vec
    /// orders columns as `[wit_bin..., wit_arb..., wit_int...]`.
    ///
    /// The verifier recomputes per-family public lifted-evals from public
    /// data, interleaves them with these, evaluates at
    /// `projecting_elements[family_idx]` for the per-family MP-eval
    /// consistency check.
    pub witness_lifted_evals: Vec<Vec<DynamicPolynomialF<F>>>,
    /// Witness-only lifted MLE evaluations under the **PCS-only prime
    /// $q''$**, sampled fresh at step 7 start. Length equals the number of
    /// witness columns. The verifier uses these directly for the PCS
    /// evaluation check at $r^\star = r_0 \bmod q''$ — no
    /// per-coefficient $\phi_{q''}$ projection needed.
    ///
    /// Kept separate from `witness_lifted_evals` because $q''$ plays a
    /// distinct role (PCS-only; no MP-eval / constraint check happens
    /// under $q''$).
    ///
    /// If no $F_q[X]$ constraints are present, this will be `None` to indicate
    /// $q'' := q_0$ and this is identical to `witness_lifted_evals`.
    pub witness_lifted_evals_pp: Option<Vec<DynamicPolynomialF<F>>>,
    /// The constraint-argument sub-proof produced by the
    /// [`ConstraintSystem`](crate::constraint_system::ConstraintSystem)
    /// frontend (for UAIR:
    /// [`UairConstraintProof`](crate::uair_frontend::UairConstraintProof) —
    /// the ideal-check / CPR / multi-degree-sumcheck / multipoint-eval bundle
    /// plus the optional booleanity / lookup arguments, for the $Q[X]$ family
    /// and each declared prime).
    pub constraint_proof: CP,
}

/// The UAIR specialization of [`Proof`], carrying a
/// [`UairConstraintProof`](crate::uair_frontend::UairConstraintProof) as its
/// constraint sub-proof. This is the default type parameter for [`Proof`].
pub type UairProof<F> = Proof<F, crate::uair_frontend::UairConstraintProof<F>>;

impl<F, CP> GenTranscribable for Proof<F, CP>
where
    F: PrimeField,
    F::Integer: ConstTranscribable,
    CP: Transcribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (commit0, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);
        let (commit1, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);
        let (commit2, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);

        let (zip_len, bytes) = u32::read_transcription_bytes_subset(bytes);
        let zip_len = usize::try_from(zip_len).expect("zip length must fit into usize");
        let (zip_bytes, bytes) = bytes.split_at(zip_len);
        let zip = zip_bytes.to_vec();

        // witness_lifted_evals: u32 count (= n + 1, one per constraint
        // family) + length-prefixed DynamicPolyVecF entries. Each entry
        // carries its own field-cfg header.
        let (n_wlf, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_wlf = usize::try_from(n_wlf).expect("n_wlf must fit into usize");
        let mut witness_lifted_evals: Vec<Vec<DynamicPolynomialF<F>>> = Vec::with_capacity(n_wlf);
        for _ in 0..n_wlf {
            let (wv, rest) = DynamicPolyVecF::<F>::read_transcription_bytes_subset(bytes);
            witness_lifted_evals.push(wv.0);
            bytes = rest;
        }

        // witness_lifted_evals_pp: u32 presence flag, then (optionally) single
        // length-prefixed DynamicPolyVecF (q'' family).
        let (presence, bytes) = u32::read_transcription_bytes_subset(bytes);
        let (witness_lifted_evals_pp, bytes) = if presence != 0 {
            let (p, rest) = DynamicPolyVecF::<F>::read_transcription_bytes_subset(bytes);
            (Some(p.0), rest)
        } else {
            (None, bytes)
        };

        // constraint_proof: length-prefixed sub-proof body.
        let (constraint_proof, bytes) = CP::read_transcription_bytes_subset(bytes);

        assert!(bytes.is_empty(), "All bytes should be consumed");

        Self {
            commitments: (commit0, commit1, commit2),
            zip,
            witness_lifted_evals,
            witness_lifted_evals_pp,
            constraint_proof,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        // 3 commitments (ConstTranscribable - no length prefix)
        buf = self.commitments.0.write_transcription_bytes_subset(buf);
        buf = self.commitments.1.write_transcription_bytes_subset(buf);
        buf = self.commitments.2.write_transcription_bytes_subset(buf);

        // zip: u32 length + raw bytes
        let zip_len = u32::try_from(self.zip.len()).expect("zip length must fit into u32");
        buf = zip_len.write_transcription_bytes_subset(buf);
        buf[..self.zip.len()].copy_from_slice(&self.zip);
        buf = &mut buf[self.zip.len()..];

        // witness_lifted_evals (per constraint family, n + 1 entries):
        // u32 count + per-family DynamicPolyVecF (each carries its own
        // field-cfg header). Index 0 is Q[X] / q_0, indices 1..=n are
        // declared primes.
        let n_wlf = u32::try_from(self.witness_lifted_evals.len())
            .expect("witness_lifted_evals length must fit into u32");
        buf = n_wlf.write_transcription_bytes_subset(buf);
        for wlf in &self.witness_lifted_evals {
            buf = DynamicPolyVecF::reinterpret(wlf).write_transcription_bytes_subset(buf);
        }

        // witness_lifted_evals_pp: u32 presence flag, then (optionally) single
        // length-prefixed DynamicPolyVecF (q'' family).
        let presence = u32::from(self.witness_lifted_evals_pp.is_some());
        buf = presence.write_transcription_bytes_subset(buf);
        if let Some(ref lifted_pp) = self.witness_lifted_evals_pp {
            buf = DynamicPolyVecF::reinterpret(lifted_pp).write_transcription_bytes_subset(buf);
        }

        // constraint_proof: length-prefixed sub-proof body.
        buf = self.constraint_proof.write_transcription_bytes_subset(buf);

        let _ = buf;
    }
}

impl<F, CP> Transcribable for Proof<F, CP>
where
    F: PrimeField,
    F::Integer: ConstTranscribable,
    CP: Transcribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let witness_lifted_evals_bytes: usize = self
            .witness_lifted_evals
            .iter()
            .map(|wlf| {
                DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
                    + DynamicPolyVecF::reinterpret(wlf).get_num_bytes()
            })
            .sum();
        let witness_lifted_evals_pp_bytes = match &self.witness_lifted_evals_pp {
            Some(wpp) => {
                DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
                    + DynamicPolyVecF::reinterpret(wpp).get_num_bytes()
            }
            None => 0,
        };
        3 * ZipPlusCommitment::NUM_BYTES
            + u32::NUM_BYTES
            + self.zip.len()
            // witness_lifted_evals: count + sum of (length-prefix + body) per family
            + u32::NUM_BYTES
            + witness_lifted_evals_bytes
            // witness_lifted_evals_pp: presence flag + optional length-prefixed body
            + u32::NUM_BYTES
            + witness_lifted_evals_pp_bytes
            // constraint_proof: length-prefix + body
            + CP::LENGTH_NUM_BYTES
            + self.constraint_proof.get_num_bytes()
    }
}

/// Trait bundling the various type parameters for the public inputs (NYI),
/// witness and Zinc+ PIOP.
pub trait ZincTypes<const DEGREE_PLUS_ONE: usize, const FOLDED_DEG_PLUS_ONE: usize>:
    Clone + Debug
{
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
    type Fmod: ConstIntSemiring + ConstTranscribable + FromRef<Self::Fmod> + Named + Send + Sync;

    /// Primality test for the field modulus.
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// Zip+ types for the binary polynomial trace columns.
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<FOLDED_DEG_PLUS_ONE>,
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

    type BinaryFold: FoldTrace<BinaryPoly<DEGREE_PLUS_ONE>, BinaryPoly<FOLDED_DEG_PLUS_ONE>>;

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
pub struct ZincPlusPiop<
    Zt,
    CS,
    F,
    const DEGREE_PLUS_ONE: usize,
    const FOLDED_DEGREE_PLUS_ONE: usize,
>(PhantomData<(Zt, CS, F)>)
where
    Zt: ZincTypes<DEGREE_PLUS_ONE, FOLDED_DEGREE_PLUS_ONE>,
    CS: crate::constraint_system::ConstraintSystem,
    F: PrimeField;

/// Error type for error happening during the protocol execution (prover and
/// verifier).
#[derive(Debug, Error)]
pub enum ProtocolError<F: PrimeField> {
    #[error("ideal check failed: {0}")]
    IdealCheck(#[from] IdealCheckError<F>),
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
    #[error("booleanity argument failed: {0}")]
    Booleanity(#[from] BooleanityError<F>),
    #[error("booleanity proof missing from proof object")]
    BooleanityProofMissing,
    #[error("PCS error: {0}")]
    Pcs(#[from] ZipError),
    #[error("PCS verification failed at column {0}: {1}")]
    PcsVerification(usize, ZipError),
    #[error("F_q[X] ideal check failed at prime_idx {prime_idx} (q = {q}): {source}")]
    FqIdealCheck {
        prime_idx: usize,
        q: String,
        source: IdealCheckError<F>,
    },
    #[error("q'' witness lifted-evals length mismatch: got {got}, expected {expected}")]
    WitnessLiftedEvalsPpLengthMismatch { got: usize, expected: usize },
    #[error(
        "witness lifted-evals length mismatch at family {family_idx}: got {got}, expected {expected}"
    )]
    WitnessLiftedEvalsLengthMismatch {
        family_idx: usize,
        got: usize,
        expected: usize,
    },
    /// The R1CS/Spartan constraint-argument frontend
    /// ([`R1csFrontend`](crate::r1cs_frontend::R1csFrontend)) failed. Carries a
    /// human-readable description of the failing check (sumcheck consistency,
    /// matrix-MLE mismatch, witness-eval reconciliation, ...).
    #[error("R1CS constraint argument failed: {0}")]
    R1cs(String),
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

/// Compute the $\alpha'$ Schwartz-Zippel bridge scalars for the witness
/// binary-poly columns:
///
/// $$
///   c_j \;=\; \sum_{i=0}^{D-1} (\alpha')^{i} \cdot
///     \text{bit\_slice\_evals}[j \cdot D + i].
/// $$
///
/// One $c_j$ is produced per witness binary-poly column (in column-major
/// order, matching `BooleanityProof::bit_slice_evals`). The result is
/// appended to `MultipointEval`'s `up_evals` and bound to the committed
/// witness column by the MP sumcheck + PCS chain at a fresh random
/// $\alpha'$, replacing the previous (underconstrained for $D > 1$)
/// $\psi_a$ linear pin-down.
#[allow(clippy::arithmetic_side_effects)]
fn alpha_prime_bridge_up_evals<F: PrimeField, const D: usize>(
    bit_slice_evals: &[F],
    num_wit_bin: usize,
    alpha_prime: &F,
    field_cfg: &F::Config,
) -> Vec<F> {
    debug_assert_eq!(bit_slice_evals.len(), num_wit_bin * D);
    let one = F::one_with_cfg(field_cfg);
    let alpha_powers: Vec<F> = powers(alpha_prime.clone(), one, D);
    bit_slice_evals
        .chunks_exact(D)
        .map(|slice| {
            slice
                .iter()
                .zip(&alpha_powers)
                .fold(F::zero_with_cfg(field_cfg), |acc, (b, alpha_pow)| {
                    acc + b.clone() * alpha_pow
                })
        })
        .collect()
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

/// Build the list of per-family [`F::Config`]'s in family order:
/// `prime_cfgs[0]` is the $Q[X]$ family's sampled prime $q_0$,
/// `prime_cfgs[1..=n]` are the declared $q_1, ..., q_n$ in
/// [`Layout::primes`](crate::constraint_system::Layout::primes) order.
///
/// The family indexing convention follows the paper's
/// `prot:zincplus-ucs-pior`: family 0 = $Q[X]$,
/// families $i \ge 1$ = $F_{q_i}[X]$.
///
/// Primality is the UAIR author's responsibility (the UAIR is part of the
/// pre-agreed relation index); no runtime check needed here.
fn build_all_cfgs<F>(layout: &Layout<F::Integer>, qx_cfg: F::Config) -> Vec<F::Config>
where
    F: PrimeField,
{
    iter::once(qx_cfg)
        .chain(
            layout
                .primes()
                .iter()
                .map(|q| F::make_cfg(q).expect("declared prime is assumed prime")),
        )
        .collect()
}

//
// Tests
//

#[cfg(test)]
#[cfg(not(miri))] // long running
#[allow(
    clippy::arithmetic_side_effects,
    clippy::result_large_err,
    clippy::type_complexity
)]
mod tests {
    use super::*;
    use crate::{fold::FoldBinaryTrace4x, uair_frontend::UairFrontend};
    use crypto_bigint::U64;
    use crypto_primitives::{
        Field, HasPrimeFieldConfig, crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
        crypto_bigint_uint::Uint,
    };
    use rand::rng;
    use zinc_piop::{
        combined_poly_resolver::CombinedPolyResolverError, multipoint_eval::MultipointEvalError,
    };
    use zinc_poly::univariate::{binary::BinaryPolyInnerProduct, dense::DensePolyInnerProduct};
    use zinc_primality::MillerRabin;
    use zinc_test_uair::{
        BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, GenerateRandomTrace,
        ShaProxy, TestUairBitOpsFqFamily, TestUairFqLargePrime, TestUairMixedShifts,
        TestUairNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_uair::{
        Uair, constraint_counter::count_constraints, ideal::DegreeOneIdeal,
        ideal_collector::IdealOrZero,
    };
    use zinc_utils::{
        CHECKED,
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

    const D: usize = 32;
    const HALF_D: usize = D / 2;
    const QUARTER_D: usize = D / 4;

    // Zip+ type parameters.

    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    const REP_FACTOR: usize = 8;

    type F = MontyField<FIELD_LIMBS>;
    type ZtFmod = Uint<FIELD_LIMBS>;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypes {}
    impl ZipTypes for BinPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 100;
        type Eval = BinaryPoly<QUARTER_D>;
        type Cw = DensePolynomial<i64, QUARTER_D>;
        type Fmod = ZtFmod;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, QUARTER_D>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, QUARTER_D>;
        type CombDotChal =
            DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, QUARTER_D>;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    pub struct ArbitraryPolyZipTypesIprs {}
    impl ZipTypes for ArbitraryPolyZipTypesIprs {
        const NUM_COLUMN_OPENINGS: usize = 100;
        type Eval = DensePolynomial<i64, D>;
        type Cw = DensePolynomial<i64, D>;
        type Fmod = ZtFmod;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, D>;
        type EvalDotChal = DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, D>;
        type CombDotChal =
            DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D>;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Arbitrary poly ZipTypes with wider codewords for RAA encoding.
    /// RAA accumulation grows the bit-width, so Cw needs more bits than Eval.
    #[derive(Debug, Clone)]
    pub struct ArbitraryPolyZipTypesRaa {}
    impl ZipTypes for ArbitraryPolyZipTypesRaa {
        const NUM_COLUMN_OPENINGS: usize = 100;
        type Eval = DensePolynomial<i64, D>;
        type Cw = DensePolynomial<Int<K>, D>;
        type Fmod = ZtFmod;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, D>;
        type EvalDotChal = DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, D>;
        type CombDotChal =
            DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D>;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    type ZtInt = i64;

    #[derive(Debug, Clone)]
    pub struct IntZipTypes {}
    impl ZipTypes for IntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 100;
        type Eval = ZtInt;
        type Cw = i128;
        type Fmod = ZtFmod;
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

    impl ZincTypes<D, QUARTER_D> for TestZincTypesIprs {
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Fmod = ZtFmod;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;

        type BinaryFold = FoldBinaryTrace4x<D, HALF_D, QUARTER_D>;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP_FACTOR, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP_FACTOR, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP_FACTOR, CHECKED>;
    }

    #[derive(Copy, Clone)]
    struct TestRaaConfig;
    impl RaaConfig for TestRaaConfig {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = true;
    }

    #[derive(Clone, Debug)]
    struct TestZincTypesRaa;

    impl ZincTypes<D, QUARTER_D> for TestZincTypesRaa {
        type Int = i64;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Fmod = ZtFmod;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesRaa;
        type IntZt = IntZipTypes;

        type BinaryFold = FoldBinaryTrace4x<D, HALF_D, QUARTER_D>;

        type BinaryLc = RaaCode<Self::BinaryZt, TestRaaConfig, REP_FACTOR>;
        type ArbitraryLc = RaaCode<Self::ArbitraryZt, TestRaaConfig, REP_FACTOR>;
        type IntLc = RaaCode<Self::IntZt, TestRaaConfig, REP_FACTOR>;
    }

    /// Use row size equal to poly size, resulting in flat single-row matrices
    fn make_iprs<Zt: ZipTypes>(
        num_vars: usize,
    ) -> IprsCode<Zt, PnttConfigF65537, REP_FACTOR, CHECKED> {
        let poly_size = 1 << num_vars;
        IprsCode::new_with_optimal_depth(poly_size).unwrap()
    }

    /// Set up Zip+ PCS parameters for a given number of MLE variables.
    fn setup_pp<Zt>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
    ) -> (
        ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
        ZipPlusParams<Zt::IntZt, Zt::IntLc>,
    )
    where
        Zt: ZincTypes<D, QUARTER_D>,
    {
        let folded_num_vars = num_vars + Zt::BinaryFold::FOLDING_FACTOR.ilog2() as usize;

        let poly_size = 1 << num_vars;
        let folded_poly_size = 1 << folded_num_vars;
        (
            ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::setup(folded_poly_size, linear_codes.0),
            ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::setup(poly_size, linear_codes.1),
            ZipPlus::<Zt::IntZt, Zt::IntLc>::setup(poly_size, linear_codes.2),
        )
    }

    macro_rules! default_project_ideal {
        () => {
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
        };
    }

    /// Older test UAIRs declare no primes, so the F_q[X] ideal projection
    /// is never invoked at runtime. UAIRs that exercise the F_q[X] family
    /// must pass a concrete projection closure.
    macro_rules! default_project_fq_ideal {
        () => {
            |_ideal, _cfg| -> IdealOrZero<DegreeOneIdeal<F>> {
                unreachable!("this UAIR has no F_q[X] constraints")
            }
        };
    }

    fn do_test<Zt, U>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
        project_ideal: impl Fn(
            &IdealOrZero<U::Ideal>,
            &<F as HasPrimeFieldConfig>::Config,
        ) -> IdealOrZero<DegreeOneIdeal<F>>
        + Copy,
        project_fq_ideal: impl Fn(
            &IdealOrZero<U::FqIdeal>,
            &<F as HasPrimeFieldConfig>::Config,
        ) -> IdealOrZero<DegreeOneIdeal<F>>
        + Copy,
        tamper: impl Fn(&mut Proof<F>),
        check_verification: impl Fn(Result<(), ProtocolError<F>>),
    ) where
        Zt: ZincTypes<D, QUARTER_D>,
        <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
        <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
        U: Uair<Scalar = DensePolynomial<Zt::Int, D>, Prime = Zt::Fmod>
            + GenerateRandomTrace<D, PolyCoeff = Zt::Int, Int = Zt::Int>
            + 'static,
        F: Field<Integer = Zt::Fmod>
            + for<'a> FromWithConfig<&'a Zt::Int>
            + for<'a> FromWithConfig<&'a Zt::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>,
    {
        let mut rng = rng();
        let pp = setup_pp::<Zt>(num_vars, linear_codes);

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = U::signature();
        let public_trace = trace.public(&sig);

        macro_rules! run_protocol {
            ($mle_first:ident) => {
                let cs = UairFrontend::<U, F, D, QUARTER_D>::from_trace(&trace, project_scalar_fn);
                let mut proof =
                    ZincPlusPiop::<Zt, UairFrontend<U, F, D, QUARTER_D>, F, D, QUARTER_D>::prove::<
                        { $mle_first },
                        CHECKED,
                    >(&pp, &trace, num_vars, &cs)
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

                let cs_v = UairFrontend::<U, F, D, QUARTER_D>::new_verifier();
                let verification_result = ZincPlusPiop::<
                    Zt,
                    UairFrontend<U, F, D, QUARTER_D>,
                    F,
                    D,
                    QUARTER_D,
                >::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    &public_trace,
                    num_vars,
                    &cs_v,
                    project_scalar_fn,
                    project_ideal,
                    project_fq_ideal,
                );
                check_verification(verification_result);
            };
        }

        run_protocol!(false);

        run_protocol!(true);
    }

    /// End-to-end test: [`TestUairNoMultiplication`].
    ///
    /// UAIR constraint: `a + b - c \in (X - 2)`
    /// (one constraint, no polynomial multiplication, ideal = `<X - 2>`).
    #[test]
    fn test_e2e_no_multiplication() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairNoMultiplication<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: [`TestUairSimpleMultiplication`].
    ///
    /// UAIR constraints (3 total, no ideals):
    /// ```
    ///   up[0] * up[1] = down[0]
    ///   up[1] * up[2] = down[1]
    ///   up[0] * up[2] = down[2]
    /// ```
    ///
    /// Uses RAA code with small num_vars (2) because chained polynomial
    /// multiplication causes exponential growth in both degree and coefficient
    /// magnitude. With num_vars=2 (4 rows), max degree=6 and max coefficient
    /// ~= 127^8 ~= 2^56, which fits in i64.
    #[test]
    fn test_e2e_simple_multiplication() {
        let num_vars = 2;
        do_test::<TestZincTypesRaa, TestUairSimpleMultiplication<ZtInt, ZtFmod>>(
            num_vars,
            (
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            default_project_fq_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: [`TestUairMixedShifts`].
    ///
    /// Uses mixed shift amounts (col a: shift 1, col b: shift 2).
    /// Constraints: `a[i+1] = a[i] + b[i], c[i] = b[i+2]`.
    #[test]
    fn test_e2e_mixed_shifts() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairMixedShifts<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            default_project_fq_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: [`BinaryDecompositionUair`].
    ///
    /// Uses binary_poly (1 col) and int (1 col) trace types.
    /// UAIR constraint: `binary_poly[0] - int[0] \in <X - 2>`
    #[test]
    fn test_e2e_binary_decomposition() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BinaryDecompositionUair<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: [`TestUairFqLargePrime`] -- exercises the per-prime
    /// $F_{q_i}[X]$ ideal-check family with **two** large primes
    /// (`TEST_UAIR_FQ_LARGE_PRIME_0`, `TEST_UAIR_FQ_LARGE_PRIME_1`).
    ///
    /// UAIR has zero $Q[X]$ constraints and one $F_{q_i}[X]$
    /// constraint per prime, both of the form $\phi_{q_i}(a) \in (X - 0)$.
    #[test]
    fn test_e2e_fq_large_prime() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairFqLargePrime<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            // No Q[X] constraints -> Q[X] ideal projection is never invoked.
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            // F_q[X] ideal projection: `DegreeOneIdeal<R>` -> `DegreeOneIdeal<F>`
            // by lifting the generating root through the per-prime field cfg.
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: [`TestUairBitOpsFqFamily`] -- exercises bit-op virtual
    /// columns through both the Q[X] and F_q[X] constraint families.
    #[test]
    fn test_e2e_bit_ops_with_fq_family() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairBitOpsFqFamily<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: [`BigLinearUair`].
    ///
    /// Uses 16 binary_poly cols and 1 int col.
    /// UAIR constraints:
    /// ```
    ///   sum(up.binary_poly[0..16]) - up.int[0] \in <X - 1>
    ///   down.binary_poly[0] - up.int[0] \in <X - 2>
    ///   up.binary_poly[i] - down.binary_poly[i] = 0, for i=1..15
    /// ```
    #[test]
    fn test_e2e_big_linear() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUair<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: [`BigLinearUairWithPublicInput`].
    ///
    /// Same as [`BigLinearUair`], but with the first few binary_poly columns as
    /// public inputs.
    #[test]
    fn test_e2e_big_linear_with_public_input() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: [`ShaProxy`].
    ///
    /// SHA-flavored benchmarking UAIR: 14 binary_poly cols, 4 int cols, with
    /// asymmetric shifts (`bp[0]` by 1, `bp[4]` by 4). UAIR constraints:
    /// ```
    ///   bp[0][t+1] - bp[1] - bp[2] - bp[3] - int[0] - int[1] - int[2] \in <X - 2>
    ///   bp[4][t+4] - bp[5] - bp[6] - bp[7] - int[1] - int[2] - int[3] \in <X - 2>
    ///   bp[8] - int[0] \in <X - 2>
    ///   bp[9] - int[1] \in <X - 2>
    ///   bp[10] - X * bp[11] \in <X - 1>
    ///   bp[12] - X * bp[13] \in <X - 1>
    /// ```
    #[test]
    fn test_e2e_sha_proxy() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, ShaProxy<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| proof.witness_lifted_evals[0].swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                ));
            },
        );
    }

    /// Adversarial regression for the per-declared-prime family of the
    /// lifted-evals consistency check in step 6. [`TestUairFqLargePrime`]
    /// declares two primes, so `witness_lifted_evals` has shape
    /// `[Q, q_1, q_2]` (length 3).
    /// We perturb family `[1]` (declared prime $q_1$) and check that the
    /// per-prime [`MultipointEval::verify_subclaim`] call inside
    /// `step6_lifted_evals` rejects with `ClaimMismatch`.
    ///
    /// This complements [`test_big_linear_tamper_lifted_evals`] (which
    /// tampers the Q-family lift at `[0]`) by exercising the symmetric
    /// per-prime family — i.e. that the verifier independently binds each
    /// $\bar u_j^{(i)}$ to the $q_i$-projected trace at $r_0$, not just the
    /// Q-family.
    #[test]
    fn test_fq_large_prime_tamper_lifted_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairFqLargePrime<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            // No Q[X] constraints (mirrors test_e2e_fq_large_prime).
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            |proof| {
                // Family 1 = declared prime q_1. The UAIR has a single
                // (arbitrary-poly) witness column, so the inner Vec has
                // length 1; tamper that one lifted polynomial by swapping
                // two of its coefficients.
                let lifted = &mut proof.witness_lifted_evals[1][0];
                assert!(
                    lifted.coeffs.len() >= 2,
                    "lifted polynomial should have at least 2 coefficients to swap"
                );
                lifted.coeffs.swap(0, 1);
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                ));
            },
        );
    }

    /// Regression for source-lift tampering on a bit-op UAIR in a
    /// declared-prime family.
    ///
    /// [`TestUairBitOpsFqFamily`] constrains `ShR(w, 3)` through both Q[X]
    /// and F_q[X]. Perturbing the F_q-family lifted eval of source column `w`
    /// changes the committed-source opening and the verifier-derived bit-op
    /// opening, so this is an end-to-end regression rather than an isolated
    /// bit-op binding test.
    #[test]
    fn test_bit_ops_fq_family_tamper_source_lifted_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairBitOpsFqFamily<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            |proof| {
                // Family 1 = declared prime q_1. Source column 0 is `w`, the
                // source of the UAIR's single bit-op virtual `ShR(w, 3)`.
                let cfg = *proof.constraint_proof.cpr_proofs_fq[0].up_evals[0].cfg();
                let one = F::one_with_cfg(&cfg);
                let lifted = &mut proof.witness_lifted_evals[1][0];
                if lifted.coeffs.is_empty() {
                    lifted.coeffs.push(one);
                } else {
                    lifted.coeffs[0] += one;
                }
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                ));
            },
        );
    }

    /// Regression: a tampered F_q-family bit-op claim is rejected end-to-end.
    ///
    /// This perturbs the prover's claimed `ShR(w, 3)(r*)` value in the CPR
    /// proof. That value is shared by CPR and multipoint-eval; in practice,
    /// this mutation is caught by CPR's constraint reconstruction before the
    /// final multipoint-eval check.
    #[test]
    fn test_bit_ops_fq_family_tamper_bit_op_eval() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairBitOpsFqFamily<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            |proof| {
                let bit_op_eval = &mut proof.constraint_proof.cpr_proofs_fq[0].bit_op_evals[0];
                let cfg = *bit_op_eval.cfg();
                *bit_op_eval += F::one_with_cfg(&cfg);
            },
            |res| {
                assert!(
                    res.is_err(),
                    "tampered F_q-family bit-op evaluation must be rejected"
                );
            },
        );
    }

    /// Regression: a too-short per-family inner lifted-evals vector must be
    /// rejected with `WitnessLiftedEvalsLengthMismatch`, not panic in the
    /// `assemble_all` slices of `step6_lifted_evals`.
    #[test]
    fn test_fq_large_prime_truncated_lifted_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairFqLargePrime<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            // Drop family 1's only witness column, making its inner vec shorter
            // than the witness-column count.
            |proof| proof.witness_lifted_evals[1].clear(),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::WitnessLiftedEvalsLengthMismatch { family_idx: 1, .. }
                ));
            },
        );
    }

    /// Adversarial regression for the q''-family lifted-evals length guard.
    /// The q'' vector (`witness_lifted_evals_pp`) is PCS-only:
    /// `step7_pcs_verify` consumes only the witness-column ranges — yet the
    /// whole vector is absorbed into the FS transcript first. Without the
    /// guard, a malicious prover could append arbitrary polynomials as free
    /// transcript entropy to grind the later PCS folding/alpha challenges
    /// without opening any extra column.
    #[test]
    fn test_tamper_witness_lifted_evals_pp_extra_tail() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| {
                // Append a surplus polynomial to the q'' lifted-evals vector,
                // inflating its length past the expected witness-column count.
                // This UAIR has no F_q[X] constraints, so q'' is aliased to
                // q_0 and the prover sends `None`;
                // The surplus entry — sourced from the Q-family lift — must still be rejected
                // by the length guard before it can be absorbed as free
                // transcript entropy.
                let extra = proof.witness_lifted_evals[0][0].clone();
                proof.witness_lifted_evals_pp = Some(vec![extra]);
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::WitnessLiftedEvalsPpLengthMismatch { .. }
                ));
            },
        );
    }

    #[test]
    fn test_big_linear_tamper_up_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| proof.constraint_proof.cpr_proof.up_evals.swap(0, 1),
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| proof.constraint_proof.cpr_proof.down_evals.swap(0, 1),
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
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| proof.commitments.0.root = Default::default(),
            |res| {
                assert!(matches!(res.unwrap_err(), ProtocolError::IdealCheck(..)));
            },
        );
    }

    //
    // Booleanity-specific end-to-end tests. `BigLinearUair` has 16 binary-poly
    // witness columns, so the booleanity argument is exercised by all of the
    // protocol tests above. The tests here verify that tampering with the
    // booleanity proof and with witness bit values produces well-typed
    // verifier errors.
    //

    /// Perturbing one entry of `booleanity_proof.bit_slice_evals` by a
    /// non-trivial additive constant breaks the recomputed booleanity
    /// residue at `r*`, which the verifier's `finalize_verifier` catches
    /// against the sumcheck's `expected_evaluation`.
    #[test]
    fn test_big_linear_tamper_booleanity_evals() {
        use zinc_piop::lookup::booleanity::BooleanityError;
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUair<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| {
                let bp = proof
                    .constraint_proof
                    .booleanity_proof
                    .as_mut()
                    .expect("BigLinearUair has binary-poly witnesses");
                let two = {
                    let cfg = bp.bit_slice_evals[0].cfg();
                    let one = F::one_with_cfg(cfg);
                    one.clone() + one
                };
                bp.bit_slice_evals[0] += two;
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Booleanity(BooleanityError::ClaimValueDoesNotMatch { .. })
                ));
            },
        );
    }

    /// Removing entries from `booleanity_proof.bit_slice_evals` breaks the
    /// length invariant; `finalize_verifier` detects this via
    /// `WrongBitSliceEvalsNumber` before the residue check.
    #[test]
    fn test_big_linear_tamper_booleanity_evals_length() {
        use zinc_piop::lookup::booleanity::BooleanityError;
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUair<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| {
                let bp = proof
                    .constraint_proof
                    .booleanity_proof
                    .as_mut()
                    .expect("BigLinearUair has binary-poly witnesses");
                bp.bit_slice_evals.pop();
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Booleanity(BooleanityError::WrongBitSliceEvalsNumber { .. })
                ));
            },
        );
    }

    /// Removing `booleanity_proof` entirely when the UAIR has bin-poly
    /// witnesses produces `BooleanityProofMissing`.
    #[test]
    fn test_big_linear_drop_booleanity_proof() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUair<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| {
                proof.constraint_proof.booleanity_proof = None;
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::BooleanityProofMissing
                ));
            },
        );
    }

    /// Soundness regression: tamper $(\delta_0, \delta_1)$ on flat
    /// positions $(0, 1)$ of `bit_slice_evals` (witness column 0) that
    /// preserves both the booleanity residue at $r^\star$ and the OLD
    /// $\psi_a$ linear pin-down $\delta_0 + a \delta_1 = 0$ at the
    /// $\psi_a$ projecting element $a$ — caught by the $\alpha'$ bridge
    /// via the MP + PCS chain.
    ///
    /// Closed form ($\alpha$ = booleanity batching challenge):
    /// $$
    ///   \delta_0 = -\frac{(2 b_0 - 1) - (\alpha / a)(2 b_1 - 1)}
    ///                    {1 + \alpha / a^2}, \quad
    ///   \delta_1 = -\delta_0 / a.
    /// $$
    /// $a$ and $\alpha$ are recovered by replaying steps 0..=3 and
    /// driving the transcript through CPR + booleanity
    /// `prepare_verifier`.
    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn test_big_linear_alpha_prime_bridge_catches_pin_down_preserving_tamper() {
        use num_traits::Inv;
        use zinc_piop::{
            combined_poly_resolver::CombinedPolyResolver, lookup::booleanity::BooleanityChecker,
        };

        type Uair = BigLinearUair<ZtInt, ZtFmod>;
        type Piop<'a> = ZincPlusPiop<
            TestZincTypesIprs,
            UairFrontend<'a, Uair, F, D, QUARTER_D>,
            F,
            D,
            QUARTER_D,
        >;
        type Ideal = IdealOrZero<DegreeOneIdeal<F>>;

        let num_constraints = count_constraints::<Uair>();

        let num_vars = 8;
        let iprs = (
            make_iprs(num_vars),
            make_iprs(num_vars),
            make_iprs(num_vars),
        );
        let pp = setup_pp::<TestZincTypesIprs>(num_vars, iprs);
        let trace = Uair::generate_random_trace(num_vars, &mut rng());
        let public_trace = trace.public(&Uair::signature());
        // Prove-side frontend borrows the trace (needed for booleanity + the
        // alpha' bridge); it is reused for step0 replay and verify.
        let cs = UairFrontend::<Uair, F, D, QUARTER_D>::from_trace(&trace, project_scalar_fn);
        let mut proof = Piop::prove::<false, CHECKED>(&pp, &trace, num_vars, &cs).expect("prove");

        // Recover `a` and `\alpha` by replaying steps 0..=3 on a proof
        // clone, then advancing the transcript through CPR + booleanity
        // `prepare_verifier`.
        let (cfg, a, alpha) = {
            let mut v3 = Piop::step0_reconstruct_transcript::<Ideal>(
                &pp,
                proof.clone(),
                &public_trace,
                num_vars,
                &cs,
            )
            .and_then(|s| s.step1_prime_projection())
            .and_then(|s| {
                s.step2_ideal_check(default_project_ideal!(), default_project_fq_ideal!())
            })
            .and_then(|s| s.step3_eval_projection(project_scalar_fn))
            .expect("steps 0..=3");

            let cfg = *v3.field_cfg();
            let a = v3.projecting_element_f().clone();
            let nv = v3.num_vars();
            let claimed_sums = v3.proof_combined_sumcheck().claimed_sums().to_vec();
            let proof_cpr = v3.proof_cpr().clone();
            let ic_subclaim = v3.ic_subclaim().clone();

            let layout = v3.layout().clone();
            let num_wit_bin = layout.total_cols().num_binary_poly_cols()
                - layout.public_cols().num_binary_poly_cols();
            let transcript = v3.fs_transcript_mut();

            let folding_challenge: F = transcript.get_field_challenge(&cfg);
            CombinedPolyResolver::<F>::prepare_verifier::<BigLinearUair<ZtInt, ZtFmod>>(
                &proof_cpr,
                claimed_sums[0].clone(),
                &ic_subclaim,
                num_constraints.q,
                nv,
                &a,
                &folding_challenge,
                &cfg,
            )
            .expect("CPR prepare_verifier");

            let bool_anc = BooleanityChecker::<F>::prepare_verifier(
                transcript,
                &claimed_sums[1],
                num_wit_bin,
                D,
                nv,
                &cfg,
            )
            .expect("booleanity prepare_verifier");

            (cfg, a, bool_anc.alpha_powers[1].clone())
        };

        // Build (\delta_0, \delta_1) from the closed form (see doc-comment).
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);
        let two = one.clone() + &one;

        let a_inv: F = Inv::inv(a.clone()).expect("a != 0");
        let alpha_over_a: F = alpha.clone() * &a_inv;
        let alpha_over_a_sq: F = alpha_over_a.clone() * &a_inv;

        let bp = proof
            .constraint_proof
            .booleanity_proof
            .as_mut()
            .expect("BigLinearUair has binary-poly witnesses");
        let s0: F = two.clone() * &bp.bit_slice_evals[0] - &one; // 2 b_0 - 1
        let s1: F = two * &bp.bit_slice_evals[1] - &one; // 2 b_1 - 1

        let denom_inv: F = Inv::inv(one + &alpha_over_a_sq).expect("1 + α/a² != 0");
        let delta_0: F = zero.clone() - (s0.clone() - alpha_over_a * &s1) * &denom_inv;
        let delta_1: F = zero - a_inv * &delta_0;

        // Sanity: tamper is non-trivial and preserves both OLD checks.
        assert!(!F::is_zero(&delta_0), "tamper must be non-zero");
        assert!(
            F::is_zero(&(delta_0.clone() + a * &delta_1)),
            "must preserve OLD ψ_a linear pin-down"
        );
        let residue = (delta_0.clone() * &s0 + delta_0.clone() * &delta_0)
            + alpha * &(delta_1.clone() * &s1 + delta_1.clone() * &delta_1);
        assert!(F::is_zero(&residue), "must preserve booleanity residue");

        bp.bit_slice_evals[0] += delta_0;
        bp.bit_slice_evals[1] += delta_1;

        let err = Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            num_vars,
            &cs,
            project_scalar_fn,
            default_project_ideal!(),
            default_project_fq_ideal!(),
        )
        .expect_err("verifier must reject alpha-prime-tampered proof");
        assert!(
            matches!(
                err,
                ProtocolError::MultipointEval(_) | ProtocolError::PcsVerification(..)
            ),
            "expected MultipointEval / PCS-chain error, got: {err:?}"
        );
    }

    #[test]
    fn test_big_linear_tamper_ideal_check() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt, ZtFmod>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            default_project_fq_ideal!(),
            |proof| {
                proof
                    .constraint_proof
                    .ideal_check
                    .combined_mle_values
                    .swap(0, 1)
            },
            |res| {
                assert!(matches!(res.unwrap_err(), ProtocolError::IdealCheck(..)));
            },
        );
    }

    //
    // R1CS frontend end-to-end tests (stage 3): drive the full substrate
    // (commit / phi_q-projection / lift-and-project / Zip+ PCS) with the
    // Spartan-style `R1csFrontend` instead of `UairFrontend`.
    //

    /// Build the tiny satisfiable R1CS `z = [1, x, w]`, `w = x * x` (with
    /// `x=3`, `w=9`), commit its witness column as a single unfolded `int`
    /// column, and run the full `ZincPlusPiop` prove/verify. `num_public`
    /// chooses how much of the public prefix `[1, x]` is treated as
    /// verifier-known `io` (the constant `1` at z[0] is always public).
    /// `tamper` mutates the proof before verification; `check` inspects the
    /// verifier result.
    fn do_r1cs_test(
        num_public: usize,
        tamper: impl Fn(&mut Proof<F, crate::r1cs_frontend::R1csConstraintProof<F>>),
        check: impl Fn(Result<(), ProtocolError<F>>),
    ) {
        use crate::{
            r1cs_frontend::{R1csFrontend, R1csInstance},
            r1cs_sparse_matrix::SparseMatrix,
        };
        use std::borrow::Cow;
        use zinc_transcript::Blake3Transcript;
        use zinc_uair::UairTrace;

        let num_vars = 8usize;
        let n = 1usize << num_vars;
        let (x, w) = (3i64, 9i64);

        // Fixed R1CS field: a deterministic large prime (the substrate uses it as
        // the working field via `working_field`, instead of sampling). Sampled
        // off a throwaway transcript so it is reproducible and Ω(λ)-sized.
        let field_cfg = {
            let mut t = Blake3Transcript::new();
            t.get_random_field_cfg::<F, ZtFmod, MillerRabin>()
        };
        let f = |v: i64| F::from_with_cfg(&v, &field_cfg);

        // A z = z[1], B z = z[1], C z = z[2]; row 1 is the trivial 0 = 0
        // constraint (so #constraints = 2, s_x = 1 >= 1). Entries are field
        // elements in our variable-density sparse matrix.
        let instance = R1csInstance {
            a: SparseMatrix::from_rows(n, vec![vec![(1usize, f(1))], Vec::new()]),
            b: SparseMatrix::from_rows(n, vec![vec![(1usize, f(1))], Vec::new()]),
            c: SparseMatrix::from_rows(n, vec![vec![(2usize, f(1))], Vec::new()]),
            num_public_inputs: num_public,
        };
        let public_values: Vec<F> = [1i64, x, w][1..=num_public].iter().map(|&v| f(v)).collect();
        let frontend = R1csFrontend::<F>::new(instance, public_values, field_cfg);

        // Committed witness int column: z with the public prefix [0..=num_public]
        // (constant + io) zeroed; the frontend re-adds it inside the argument.
        // The witness values are small, so the substrate's `int` group (Zt::Int =
        // i64) holds their canonical reps directly; the substrate projects them to
        // the fixed field via phi (the identity here since q0 = the field).
        let mut z_wit = vec![0i64; n];
        z_wit[1] = x;
        z_wit[2] = w;
        for slot in z_wit.iter_mut().take(num_public + 1) {
            *slot = 0;
        }
        let witness_col = DenseMultilinearExtension::from_evaluations_vec(num_vars, z_wit, 0i64);
        let trace: UairTrace<'static, ZtInt, ZtInt, D, D> = UairTrace {
            binary_poly: Cow::Owned(vec![]),
            arbitrary_poly: Cow::Owned(vec![]),
            int: Cow::Owned(vec![witness_col]),
        };
        // R1CS declares no substrate public columns, so the public trace is empty.
        let public_trace: UairTrace<'static, ZtInt, ZtInt, D, D> = UairTrace {
            binary_poly: Cow::Owned(vec![]),
            arbitrary_poly: Cow::Owned(vec![]),
            int: Cow::Owned(vec![]),
        };

        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );

        type Piop = ZincPlusPiop<TestZincTypesIprs, R1csFrontend<F>, F, D, QUARTER_D>;
        type Ideal = IdealOrZero<DegreeOneIdeal<F>>;

        let mut proof =
            Piop::prove::<false, CHECKED>(&pp, &trace, num_vars, &frontend).expect("R1CS prove");
        tamper(&mut proof);

        // The three projection closures are never invoked (zero ideal, no psi_a).
        let res = Piop::verify::<Ideal, CHECKED>(
            &pp,
            proof,
            &public_trace,
            num_vars,
            &frontend,
            |_, _| unreachable!("R1CS: no psi_a scalar projection"),
            |_, _| unreachable!("R1CS: no ideal projection"),
            |_, _| unreachable!("R1CS: no fq-ideal projection"),
        );
        check(res);
    }

    /// End-to-end over the full substrate: R1CS with only the constant public.
    #[test]
    fn test_e2e_r1cs() {
        do_r1cs_test(0, |_| {}, |res| res.expect("R1CS verify"));
    }

    /// End-to-end with one genuine public input (`x` at z[1]) — exercises the
    /// nonzero `z_pub(r_y)` reconciliation against the substrate witness lift.
    #[test]
    fn test_e2e_r1cs_with_public_input() {
        do_r1cs_test(1, |_| {}, |res| res.expect("R1CS verify (public input)"));
    }

    /// Tamper negative: perturbing the frontend's `z_ry` breaks the inner
    /// sumcheck's `M~(r_x, r_y) * z_ry` consistency check.
    #[test]
    fn test_e2e_r1cs_tamper_z_ry() {
        do_r1cs_test(
            0,
            |proof| {
                let cfg = *proof.constraint_proof.z_ry.cfg();
                proof.constraint_proof.z_ry += &F::one_with_cfg(&cfg);
            },
            |res| assert!(res.is_err(), "tampered z_ry must be rejected"),
        );
    }

    /// Tamper negative: perturbing the substrate-assembled witness lift breaks
    /// `verify_lifted_evals`' `z(r_y) == z_pub(r_y) + z_wit(r_y)`
    /// reconciliation (the resolved risk-(f) path).
    #[test]
    fn test_e2e_r1cs_tamper_lifted_eval() {
        do_r1cs_test(
            0,
            |proof| {
                let cfg = *proof.constraint_proof.z_ry.cfg();
                let lift = &mut proof.witness_lifted_evals[0][0];
                if lift.coeffs.is_empty() {
                    lift.coeffs.push(F::one_with_cfg(&cfg));
                } else {
                    lift.coeffs[0] += &F::one_with_cfg(&cfg);
                }
            },
            |res| assert!(res.is_err(), "tampered witness lift must be rejected"),
        );
    }
}
