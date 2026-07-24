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

pub mod fold;
pub mod prover;
pub mod shared_challenge;
pub mod verifier;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::fold::FoldTrace;
use crypto_primitives::{
    BaseFieldConfig, ConstIntRing, ConstIntSemiring, ProjectElementWithConfig,
    ProjectPrimitiveIntegersWithConfig, Semiring, SetElement, Wrapper,
};
use std::{
    fmt::{Debug, Display},
    iter,
    marker::PhantomData,
};
use thiserror::Error;
use zinc_piop::{
    combined_poly_resolver::{CombinedPolyResolverError, Proof as CombinedPolyResolverProof},
    ideal_check::{IdealCheckError, Proof as IdealCheckProof},
    lookup::{
        BatchedLookupProof, LookupError,
        booleanity::{BooleanityError, BooleanityProof},
    },
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
        dynamic::{DynamicPolyVec, DynamicPolynomial, HasDynamicPolynomialConfig},
    },
};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_uair::{Uair, UairSignature};
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
/// PCS-only $q''$), the prover sends a vector of `DynamicPolynomial<F>`
/// carrying the per-family coefficient lift of each witness column. The
/// verifier reads each family's lifts under that family's field cfg, no
/// per-coefficient `cfg.project` projection is needed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F> {
    /// Zip+ commitments to the witness columns.
    pub commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip: Vec<u8>,
    /// Randomized ideal check proof (Q[X] family).
    pub ideal_check: IdealCheckProof<F>,
    /// Combined polynomial resolver proof (up_evals + down_evals).
    pub cpr_proof: CombinedPolyResolverProof<F>,
    /// Multi-degree sumcheck proof (CPR group + lookup groups).
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Multi-point evaluation sumcheck proof (combines up_evals and
    /// down_evals at `r*` into a single evaluation point `r_0`).
    pub multipoint_eval: MultipointEvalProof<F>,
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
    pub witness_lifted_evals: Vec<Vec<DynamicPolynomial<F>>>,
    /// Lookup argument proof. `None` when the UAIR has no lookup specs.
    pub lookup_proof: Option<BatchedLookupProof<F>>,
    /// Binary-polynomial booleanity argument proof. `None` when the UAIR
    /// has no witness binary-poly columns (the argument is omitted from
    /// the multi-degree sumcheck in that case).
    pub booleanity_proof: Option<BooleanityProof<F>>,
    /// Per-prime $F_{q_i}[X]$ ideal-check proofs, one per declared
    /// prime in [`zinc_uair::UairSignature::primes`], in the same order.
    /// Empty for UAIRs with $Q[X]$-only constraints.
    pub ideal_checks_fq: Vec<IdealCheckProof<F>>,
    /// Per-prime CPR proofs, one per declared prime, produced by the
    /// lockstep sumcheck in step 5. Empty for UAIRs with $Q[X]$ only
    /// constraints.
    pub cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>>,
    /// Per-prime multi-degree sumcheck proofs, one per declared prime,
    /// produced by the lockstep sumcheck driver in step 5.
    /// Empty for UAIRs with $Q[X]$ only constraints.
    pub combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>>,
    /// Per-prime multipoint-eval proofs, one per declared prime, produced
    /// by the lockstep multipoint-eval in step 6.
    /// Empty for UAIRs with $Q[X]$ only constraints.
    pub multipoint_evals_fq: Vec<MultipointEvalProof<F>>,
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
    pub witness_lifted_evals_pp: Option<Vec<DynamicPolynomial<F>>>,
}

impl<F> GenTranscribable for Proof<F>
where
    F: ConstTranscribable,
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

        // witness_lifted_evals: u32 count (= n + 1, one per constraint
        // family) + length-prefixed DynamicPolyVec entries. Each entry
        // carries its own field-cfg header.
        let (n_wlf, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_wlf = usize::try_from(n_wlf).expect("n_wlf must fit into usize");
        let mut witness_lifted_evals: Vec<Vec<DynamicPolynomial<F>>> = Vec::with_capacity(n_wlf);
        for _ in 0..n_wlf {
            let (wv, rest) = DynamicPolyVec::<F>::read_transcription_bytes_subset(bytes);
            witness_lifted_evals.push(wv.0);
            bytes = rest;
        }

        // booleanity_proof: presence flag (u32: 0 = absent, 1 = present)
        // followed by the proof body (length-prefixed) when present.
        let (presence, bytes) = u32::read_transcription_bytes_subset(bytes);
        let (booleanity_proof, bytes) = if presence != 0 {
            let (p, rest) = BooleanityProof::<F>::read_transcription_bytes_subset(bytes);
            (Some(p), rest)
        } else {
            (None, bytes)
        };

        // ideal_checks_fq: u32 count, then that many length-prefixed
        // IdealCheckProof entries (one per declared prime).
        let (n_fq, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_fq = usize::try_from(n_fq).expect("n_fq must fit into usize");
        let mut ideal_checks_fq: Vec<IdealCheckProof<F>> = Vec::with_capacity(n_fq);
        for _ in 0..n_fq {
            let (ic, rest) = IdealCheckProof::<F>::read_transcription_bytes_subset(bytes);
            ideal_checks_fq.push(ic);
            bytes = rest;
        }

        // cpr_proofs_fq: u32 count + length-prefixed entries.
        let (n_cpr_fq, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_cpr_fq = usize::try_from(n_cpr_fq).expect("n_cpr_fq must fit into usize");
        let mut cpr_proofs_fq: Vec<CombinedPolyResolverProof<F>> = Vec::with_capacity(n_cpr_fq);
        for _ in 0..n_cpr_fq {
            let (cpr, rest) =
                CombinedPolyResolverProof::<F>::read_transcription_bytes_subset(bytes);
            cpr_proofs_fq.push(cpr);
            bytes = rest;
        }

        // combined_sumchecks_fq: u32 count + length-prefixed entries.
        let (n_sum_fq, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_sum_fq = usize::try_from(n_sum_fq).expect("n_sum_fq must fit into usize");
        let mut combined_sumchecks_fq: Vec<MultiDegreeSumcheckProof<F>> =
            Vec::with_capacity(n_sum_fq);
        for _ in 0..n_sum_fq {
            let (sumcheck, rest) =
                MultiDegreeSumcheckProof::<F>::read_transcription_bytes_subset(bytes);
            combined_sumchecks_fq.push(sumcheck);
            bytes = rest;
        }

        // multipoint_evals_fq: u32 count + length-prefixed entries.
        let (n_mp_fq, mut bytes) = u32::read_transcription_bytes_subset(bytes);
        let n_mp_fq = usize::try_from(n_mp_fq).expect("n_mp_fq must fit into usize");
        let mut multipoint_evals_fq: Vec<MultipointEvalProof<F>> = Vec::with_capacity(n_mp_fq);
        for _ in 0..n_mp_fq {
            let (mp, rest) = MultipointEvalProof::<F>::read_transcription_bytes_subset(bytes);
            multipoint_evals_fq.push(mp);
            bytes = rest;
        }

        // witness_lifted_evals_pp: u32 presence flag, then (optionally) single
        // length-prefixed DynamicPolyVec (q'' family).
        let (presence, bytes) = u32::read_transcription_bytes_subset(bytes);
        let (witness_lifted_evals_pp, bytes) = if presence != 0 {
            let (p, rest) = DynamicPolyVec::<F>::read_transcription_bytes_subset(bytes);
            (Some(p.0), rest)
        } else {
            (None, bytes)
        };

        // TODO: deserialize lookup_proof once BatchedLookupProof gets
        // Transcribable impls (lookup is not yet implemented).
        assert!(bytes.is_empty(), "All bytes should be consumed");

        Self {
            commitments: (commit0, commit1, commit2),
            zip,
            ideal_check,
            cpr_proof: resolver,
            combined_sumcheck,
            multipoint_eval,
            witness_lifted_evals,
            lookup_proof: None,
            booleanity_proof,
            ideal_checks_fq,
            cpr_proofs_fq,
            combined_sumchecks_fq,
            multipoint_evals_fq,
            witness_lifted_evals_pp,
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

        // ideal_check: u32 length prefix + data
        buf = self.ideal_check.write_transcription_bytes_subset(buf);

        // resolver: u32 length prefix + data
        buf = self.cpr_proof.write_transcription_bytes_subset(buf);

        // combined_sumcheck: u32 length prefix + data
        buf = self.combined_sumcheck.write_transcription_bytes_subset(buf);

        // multipoint_eval: u32 length prefix + data
        buf = self.multipoint_eval.write_transcription_bytes_subset(buf);

        // witness_lifted_evals (per constraint family, n + 1 entries):
        // u32 count + per-family DynamicPolyVec (each carries its own
        // field-cfg header). Index 0 is Q[X] / q_0, indices 1..=n are
        // declared primes.
        let n_wlf = u32::try_from(self.witness_lifted_evals.len())
            .expect("witness_lifted_evals length must fit into u32");
        buf = n_wlf.write_transcription_bytes_subset(buf);
        for wlf in &self.witness_lifted_evals {
            buf = DynamicPolyVec::reinterpret(wlf).write_transcription_bytes_subset(buf);
        }

        // booleanity_proof: u32 presence flag, then (optionally) the body
        // with its own length prefix.
        let presence = u32::from(self.booleanity_proof.is_some());
        buf = presence.write_transcription_bytes_subset(buf);
        if let Some(ref bp) = self.booleanity_proof {
            buf = bp.write_transcription_bytes_subset(buf);
        }

        // ideal_checks_fq: u32 count + that many length-prefixed entries.
        let n_fq = u32::try_from(self.ideal_checks_fq.len())
            .expect("ideal_checks_fq length must fit into u32");
        buf = n_fq.write_transcription_bytes_subset(buf);
        for ic in &self.ideal_checks_fq {
            buf = ic.write_transcription_bytes_subset(buf);
        }

        // cpr_proofs_fq: u32 count + length-prefixed entries.
        let n_cpr_fq = u32::try_from(self.cpr_proofs_fq.len())
            .expect("cpr_proofs_fq length must fit into u32");
        buf = n_cpr_fq.write_transcription_bytes_subset(buf);
        for cpr in &self.cpr_proofs_fq {
            buf = cpr.write_transcription_bytes_subset(buf);
        }

        // combined_sumchecks_fq: u32 count + length-prefixed entries.
        let n_sum_fq = u32::try_from(self.combined_sumchecks_fq.len())
            .expect("combined_sumchecks_fq length must fit into u32");
        buf = n_sum_fq.write_transcription_bytes_subset(buf);
        for sumcheck in &self.combined_sumchecks_fq {
            buf = sumcheck.write_transcription_bytes_subset(buf);
        }

        // multipoint_evals_fq: u32 count + length-prefixed entries.
        let n_mp_fq = u32::try_from(self.multipoint_evals_fq.len())
            .expect("multipoint_evals_fq length must fit into u32");
        buf = n_mp_fq.write_transcription_bytes_subset(buf);
        for mp in &self.multipoint_evals_fq {
            buf = mp.write_transcription_bytes_subset(buf);
        }

        // witness_lifted_evals_pp: u32 presence flag, then (optionally) single
        // length-prefixed DynamicPolyVec (q'' family).
        let presence = u32::from(self.witness_lifted_evals_pp.is_some());
        buf = presence.write_transcription_bytes_subset(buf);
        if let Some(ref lifted_pp) = self.witness_lifted_evals_pp {
            buf = DynamicPolyVec::reinterpret(lifted_pp).write_transcription_bytes_subset(buf);
        }

        // TODO: serialize lookup_proof once BatchedLookupProof gets
        // Transcribable impls (lookup is not yet implemented).
        let _ = buf;
    }
}

impl<F> Transcribable for Proof<F>
where
    F: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let booleanity_bytes = match &self.booleanity_proof {
            Some(bp) => BooleanityProof::<F>::LENGTH_NUM_BYTES + bp.get_num_bytes(),
            None => 0,
        };
        let ideal_checks_fq_bytes: usize = self
            .ideal_checks_fq
            .iter()
            .map(|ic| IdealCheckProof::<F>::LENGTH_NUM_BYTES + ic.get_num_bytes())
            .sum();
        let cpr_proofs_fq_bytes: usize = self
            .cpr_proofs_fq
            .iter()
            .map(|cpr| CombinedPolyResolverProof::<F>::LENGTH_NUM_BYTES + cpr.get_num_bytes())
            .sum();
        let combined_sumchecks_fq_bytes: usize = self
            .combined_sumchecks_fq
            .iter()
            .map(|sc| MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES + sc.get_num_bytes())
            .sum();
        let multipoint_evals_fq_bytes: usize = self
            .multipoint_evals_fq
            .iter()
            .map(|mp| MultipointEvalProof::<F>::LENGTH_NUM_BYTES + mp.get_num_bytes())
            .sum();
        let witness_lifted_evals_bytes: usize = self
            .witness_lifted_evals
            .iter()
            .map(|wlf| {
                DynamicPolyVec::<F>::LENGTH_NUM_BYTES
                    + DynamicPolyVec::reinterpret(wlf).get_num_bytes()
            })
            .sum();
        let witness_lifted_evals_pp_bytes = match &self.witness_lifted_evals_pp {
            Some(wpp) => {
                DynamicPolyVec::<F>::LENGTH_NUM_BYTES
                    + DynamicPolyVec::reinterpret(wpp).get_num_bytes()
            }
            None => 0,
        };
        3 * ZipPlusCommitment::NUM_BYTES
            + u32::NUM_BYTES
            + self.zip.len()
            + IdealCheckProof::<F>::LENGTH_NUM_BYTES
            + self.ideal_check.get_num_bytes()
            + CombinedPolyResolverProof::<F>::LENGTH_NUM_BYTES
            + self.cpr_proof.get_num_bytes()
            + MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES
            + self.combined_sumcheck.get_num_bytes()
            + MultipointEvalProof::<F>::LENGTH_NUM_BYTES
            + self.multipoint_eval.get_num_bytes()
            // TODO: add lookup_proof size once BatchedLookupProof gets
            // Transcribable impls (lookup is not yet implemented).
            //
            // witness_lifted_evals: count + sum of (length-prefix + body) per family
            + u32::NUM_BYTES
            + witness_lifted_evals_bytes
            // booleanity presence flag + optional payload
            + u32::NUM_BYTES
            + booleanity_bytes
            // ideal_checks_fq: count + sum of (length-prefix + body) per entry
            + u32::NUM_BYTES
            + ideal_checks_fq_bytes
            // cpr_proofs_fq: count + sum of (length-prefix + body) per entry
            + u32::NUM_BYTES
            + cpr_proofs_fq_bytes
            // combined_sumchecks_fq: count + sum of (length-prefix + body) per entry
            + u32::NUM_BYTES
            + combined_sumchecks_fq_bytes
            // multipoint_evals_fq: count + sum of (length-prefix + body) per entry
            + u32::NUM_BYTES
            + multipoint_evals_fq_bytes
            // witness_lifted_evals_pp: single length-prefixed body
            + u32::NUM_BYTES
            + witness_lifted_evals_pp_bytes
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
    type Fmod: ConstIntSemiring
        + ConstTranscribable
        + FromRef<Self::Fmod>
        + Display
        + Named
        + Send
        + Sync;

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
pub struct ZincPlusPiop<Zt, U, C, const DEGREE_PLUS_ONE: usize, const FOLDED_DEGREE_PLUS_ONE: usize>(
    PhantomData<(Zt, U, C)>,
)
where
    Zt: ZincTypes<DEGREE_PLUS_ONE, FOLDED_DEGREE_PLUS_ONE>,
    U: Uair,
    C: BaseFieldConfig;

/// Error type for error happening during the protocol execution (prover and
/// verifier).
#[derive(Debug, Error)]
pub enum ProtocolError<F: SetElement> {
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
    #[error("non-canonical proof element: lifted integer >= family modulus")]
    NonCanonicalElement,
    #[error("proof family count mismatch: got {got}, expected {expected}")]
    FamilyCountMismatch { got: usize, expected: usize },
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
fn compute_lifted_evals<C: BaseFieldConfig, const D: usize>(
    point: &[C::Element],
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_trace: &ProjectedTrace<C::Element>,
    field_cfg: &C,
) -> Vec<DynamicPolynomial<C::Element>> {
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(field_cfg, point)
        .expect("compute_lifted_evals: eq table build failed");

    let n_bin = trace_bin_poly.len();
    let zero = field_cfg.zero();
    let poly_cfg = field_cfg.dyn_poly_cfg();

    // Binary columns: exploit 0/1 structure for conditional additions.
    let mut result: Vec<DynamicPolynomial<C::Element>> = cfg_iter!(trace_bin_poly)
        .map(|col| {
            let mut coeffs = vec![zero.clone(); D];
            for (b, entry) in col.iter().enumerate() {
                for (l, coeff) in entry.iter().enumerate() {
                    if *coeff.inner() {
                        field_cfg.add_assign(&mut coeffs[l], &eq_table[b]);
                    }
                }
            }
            poly_cfg.new_trimmed(coeffs)
        })
        .collect();

    // Non-binary columns: coefficient-wise eq-weighted sum.
    fn weighted_eq_sum<'a, C2: BaseFieldConfig>(
        cfg: &C2,
        col: impl Iterator<Item = &'a DynamicPolynomial<C2::Element>> + Clone,
        eq_table: &[C2::Element],
        zero: &C2::Element,
    ) -> DynamicPolynomial<C2::Element>
    where
        C2::Element: 'a,
    {
        let num_coeffs = col.clone().map(|e| e.coeffs.len()).max().unwrap_or(0);
        let mut coeffs = vec![zero.clone(); num_coeffs];
        for (b, entry) in col.enumerate() {
            for (l, coeff) in entry.coeffs.iter().enumerate() {
                let term = cfg.mul(&eq_table[b], coeff);
                cfg.add_assign(&mut coeffs[l], &term);
            }
        }
        cfg.dyn_poly_cfg().new_trimmed(coeffs)
    }

    match projected_trace {
        ProjectedTrace::RowMajor(t) => {
            let num_cols = t.first().map(|r| r.len()).unwrap_or(0);
            cfg_extend!(
                result,
                cfg_into_iter!(n_bin..num_cols).map(|col_idx| weighted_eq_sum(
                    field_cfg,
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
                    field_cfg,
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
fn alpha_prime_bridge_up_evals<C: BaseFieldConfig, const D: usize>(
    bit_slice_evals: &[C::Element],
    num_wit_bin: usize,
    alpha_prime: &C::Element,
    field_cfg: &C,
) -> Vec<C::Element> {
    debug_assert_eq!(bit_slice_evals.len(), num_wit_bin * D);
    let alpha_powers: Vec<C::Element> = powers(field_cfg, alpha_prime, D);
    bit_slice_evals
        .chunks_exact(D)
        .map(|slice| {
            slice
                .iter()
                .zip(&alpha_powers)
                .fold(field_cfg.zero(), |mut acc, (b, alpha_pow)| {
                    field_cfg.add_assign(&mut acc, &field_cfg.mul(b, alpha_pow));
                    acc
                })
        })
        .collect()
}

/// Project a DensePolynomial scalar to DynamicPolynomial by projecting each
/// coefficient via \phi_q.
pub fn project_scalar_fn<R, C, const D: usize>(
    scalar: &DensePolynomial<R, D>,
    field_cfg: &C,
) -> DynamicPolynomial<C::Element>
where
    C: BaseFieldConfig + ProjectElementWithConfig<R>,
{
    scalar
        .iter()
        .map(|coeff| field_cfg.project(coeff))
        .collect()
}

/// Projects a canonical lifted integer from the wire into the field
/// configured by `cfg`, rejecting non-canonical encodings: every field
/// value has exactly one accepted wire representation (`0 <= int < q`).
fn project_canonical<C, F>(cfg: &C, int: &C::Integer) -> Result<C::Element, ProtocolError<F>>
where
    C: BaseFieldConfig,
    F: SetElement,
{
    if *int < cfg.modulus() {
        Ok(cfg.project(int))
    } else {
        Err(ProtocolError::NonCanonicalElement)
    }
}

/// Build the list of per-family field configs in family order:
/// `prime_cfgs[0]` is the $Q[X]$ family's sampled prime $q_0$,
/// `prime_cfgs[1..=n]` are the declared $q_1, ..., q_n$ in
/// [`zinc_uair::UairSignature::primes`] order.
///
/// The family indexing convention follows the paper's
/// `prot:zincplus-ucs-pior`: family 0 = $Q[X]$,
/// families $i \ge 1$ = $F_{q_i}[X]$.
///
/// Primality is the UAIR author's responsibility (the UAIR is part of the
/// pre-agreed relation index); no runtime check needed here.
fn build_all_cfgs<C>(sig: &UairSignature<C::Integer>, qx_cfg: C) -> Vec<C>
where
    C: BaseFieldConfig,
{
    iter::once(qx_cfg)
        .chain(
            sig.primes()
                .iter()
                .map(|q| C::new(q).expect("declared prime is assumed prime")),
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
    clippy::type_complexity,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::clone_on_copy,
    clippy::redundant_clone
)]
mod tests {
    use super::*;
    use crate::fold::FoldBinaryTrace4x;
    use crypto_primitives::{
        FieldConfig, LiftElementWithConfig, RingConfig, SemiringConfig,
        crypto_bigint_int::Int,
        crypto_bigint_monty::{MontyField, MontyFieldElement},
        crypto_bigint_uint::{U64, Uint},
    };
    use num_traits::{ConstOne, Zero};
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
        constraint_counter::count_constraints, ideal::DegreeOneIdeal, ideal_collector::IdealOrZero,
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
    type E = MontyFieldElement<FIELD_LIMBS>;
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
        type CombDotChal = DensePolyInnerProduct<
            (),
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            QUARTER_D,
        >;
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
        type EvalDotChal =
            DensePolyInnerProduct<(), i64, Self::Chal, Self::CombR, MBSInnerProduct, D>;
        type CombDotChal =
            DensePolyInnerProduct<(), Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D>;
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
        type EvalDotChal =
            DensePolyInnerProduct<(), i64, Self::Chal, Self::CombR, MBSInnerProduct, D>;
        type CombDotChal =
            DensePolyInnerProduct<(), Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D>;
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
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::project(field_cfg, i))
        };
    }

    /// Older test UAIRs declare no primes, so the F_q[X] ideal projection
    /// is never invoked at runtime. UAIRs that exercise the F_q[X] family
    /// must pass a concrete projection closure.
    macro_rules! default_project_fq_ideal {
        () => {
            |_ideal, _cfg| -> IdealOrZero<DegreeOneIdeal<E>> {
                unreachable!("this UAIR has no F_q[X] constraints")
            }
        };
    }

    fn do_test<Zt, U>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>, &F) -> IdealOrZero<DegreeOneIdeal<E>> + Copy,
        project_fq_ideal: impl Fn(&IdealOrZero<U::FqIdeal>, &F) -> IdealOrZero<DegreeOneIdeal<E>> + Copy,
        tamper: impl Fn(&mut Proof<ZtFmod>),
        check_verification: impl Fn(Result<(), ProtocolError<E>>),
    ) where
        Zt: ZincTypes<D, QUARTER_D, Fmod = ZtFmod, Int = ZtInt, Chal = i128, CombR = Int<M>>,
        Zt::Int: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
        U: Uair<Scalar = DensePolynomial<Zt::Int, D>, Prime = Zt::Fmod>
            + GenerateRandomTrace<D, PolyCoeff = Zt::Int, Int = Zt::Int>
            + 'static,
    {
        let mut rng = rng();
        let pp = setup_pp::<Zt>(num_vars, linear_codes);

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = U::signature();
        let public_trace = trace.public(&sig);

        macro_rules! run_protocol {
            ($mle_first:ident) => {
                let mut proof = ZincPlusPiop::<Zt, U, F, D, QUARTER_D>::prove::<
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
                    ZincPlusPiop::<Zt, U, F, D, QUARTER_D>::verify::<_, CHECKED>(
                        &pp,
                        proof,
                        &public_trace,
                        num_vars,
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
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<E>>::zero(),
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
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<E>>::zero(),
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
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<E>>::zero(),
            // F_q[X] ideal projection: `DegreeOneIdeal<R>` -> `DegreeOneIdeal<F>`
            // by lifting the generating root through the per-prime field cfg.
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::project(field_cfg, i)),
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
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::project(field_cfg, i)),
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
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<E>>::zero(),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::project(field_cfg, i)),
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
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::project(field_cfg, i)),
            |proof| {
                // Family 1 = declared prime q_1. Source column 0 is `w`, the
                // source of the UAIR's single bit-op virtual `ShR(w, 3)`.
                // The family's declared prime is statically known.
                let sig = TestUairBitOpsFqFamily::<ZtInt, ZtFmod>::signature();
                let cfg = F::new(&sig.primes()[0]).expect("declared prime");
                let one = cfg.one();
                let lifted = &mut proof.witness_lifted_evals[1][0];
                if lifted.coeffs.is_empty() {
                    lifted.coeffs.push(cfg.lift(&one));
                } else {
                    let v = cfg.project(&lifted.coeffs[0]);
                    lifted.coeffs[0] = cfg.lift(&cfg.add(&v, &one));
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
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::project(field_cfg, i)),
            |proof| {
                // The family's declared prime is statically known.
                let sig = TestUairBitOpsFqFamily::<ZtInt, ZtFmod>::signature();
                let cfg = F::new(&sig.primes()[0]).expect("declared prime");
                let bit_op_eval = &mut proof.cpr_proofs_fq[0].bit_op_evals[0];
                let v = cfg.project(&*bit_op_eval);
                *bit_op_eval = cfg.lift(&cfg.add(&v, &cfg.one()));
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
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<E>>::zero(),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::project(field_cfg, i)),
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
            |proof| proof.cpr_proof.up_evals.swap(0, 1),
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
            |proof| proof.cpr_proof.down_evals.swap(0, 1),
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

    // A wire integer >= the family modulus
    #[test]
    fn test_big_linear_tamper_non_canonical_wire_integer() {
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
            |proof| proof.cpr_proof.up_evals[0] = ZtFmod::MAX,
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::NonCanonicalElement
                ));
            },
        );
    }

    // A *canonical* perturbation of the ideal-check opening values passes
    // the wire projection
    #[test]
    fn test_big_linear_tamper_ideal_check_values() {
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
                let coeffs = &mut proof.ideal_check.combined_mle_values[0].coeffs;
                if coeffs.is_empty() {
                    coeffs.push(ZtFmod::from(1u64));
                } else {
                    if coeffs[0].is_zero() {
                        coeffs[0] += ZtFmod::ONE;
                    } else {
                        coeffs[0] -= ZtFmod::ONE;
                    }
                }
            },
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
                    .booleanity_proof
                    .as_mut()
                    .expect("BigLinearUair has binary-poly witnesses");
                // The Q-family cfg (random q_0) is not carried by raw
                // elements; swapping two (distinct w.o.p.) evals is an
                // equally non-trivial perturbation of the residue.
                bp.bit_slice_evals.swap(0, 1);
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
                proof.booleanity_proof = None;
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
        use zinc_piop::{
            combined_poly_resolver::CombinedPolyResolver, lookup::booleanity::BooleanityChecker,
        };

        type Piop = ZincPlusPiop<TestZincTypesIprs, BigLinearUair<ZtInt, ZtFmod>, F, D, QUARTER_D>;
        type Ideal = IdealOrZero<DegreeOneIdeal<E>>;

        let num_constraints = count_constraints::<BigLinearUair<ZtInt, ZtFmod>>();

        let num_vars = 8;
        let iprs = (
            make_iprs(num_vars),
            make_iprs(num_vars),
            make_iprs(num_vars),
        );
        let pp = setup_pp::<TestZincTypesIprs>(num_vars, iprs);
        let trace = BigLinearUair::<ZtInt, ZtFmod>::generate_random_trace(num_vars, &mut rng());
        let public_trace = trace.public(&BigLinearUair::<ZtInt, ZtFmod>::signature());
        let mut proof =
            Piop::prove::<false, CHECKED>(&pp, &trace, num_vars, project_scalar_fn).expect("prove");

        // Recover `a` and `\alpha` by replaying steps 0..=3 on a proof
        // clone, then advancing the transcript through CPR + booleanity
        // `prepare_verifier`.
        let (cfg, a, alpha) = {
            let mut v3 = Piop::step0_reconstruct_transcript::<Ideal>(
                &pp,
                proof.clone(),
                &public_trace,
                num_vars,
            )
            .and_then(|s| s.step1_prime_projection())
            .and_then(|s| {
                s.step2_ideal_check(default_project_ideal!(), default_project_fq_ideal!())
            })
            .and_then(|s| s.step3_eval_projection(project_scalar_fn))
            .expect("steps 0..=3");

            let cfg = v3.field_cfg().clone();
            let a = v3.projecting_element_f().clone();
            let nv = v3.num_vars();
            let claimed_sums = v3.proof_combined_sumcheck().claimed_sums().to_vec();
            let proof_cpr = v3.proof_cpr().clone();
            let ic_subclaim = v3.ic_subclaim().clone();

            let sig = v3.uair_signature().clone();
            let num_wit_bin =
                sig.total_cols().num_binary_poly_cols() - sig.public_cols().num_binary_poly_cols();
            let transcript = v3.fs_transcript_mut();

            let folding_challenge: E = transcript.get_field_challenge(&cfg);
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
        let one = cfg.one();
        let two = cfg.add(&one, &one);

        let a_inv: E = cfg.inv(&a).expect("a != 0");
        let alpha_over_a: E = cfg.mul(&alpha, &a_inv);
        let alpha_over_a_sq: E = cfg.mul(&alpha_over_a, &a_inv);

        let bp = proof
            .booleanity_proof
            .as_mut()
            .expect("BigLinearUair has binary-poly witnesses");
        let b0: E = cfg.project(&bp.bit_slice_evals[0]);
        let b1: E = cfg.project(&bp.bit_slice_evals[1]);
        let s0: E = cfg.sub(&cfg.mul(&two, &b0), &one); // 2 b_0 - 1
        let s1: E = cfg.sub(&cfg.mul(&two, &b1), &one); // 2 b_1 - 1

        let denom_inv: E = cfg
            .inv(&cfg.add(&one, &alpha_over_a_sq))
            .expect("1 + α/a² != 0");
        let delta_0: E = cfg.neg(&cfg.mul(&cfg.sub(&s0, &cfg.mul(&alpha_over_a, &s1)), &denom_inv));
        let delta_1: E = cfg.neg(&cfg.mul(&a_inv, &delta_0));

        // Sanity: tamper is non-trivial and preserves both OLD checks.
        assert!(!cfg.is_zero(&delta_0), "tamper must be non-zero");
        assert!(
            cfg.is_zero(&cfg.add(&delta_0, &cfg.mul(&a, &delta_1))),
            "must preserve OLD ψ_a linear pin-down"
        );
        let residue = cfg.add(
            &cfg.add(&cfg.mul(&delta_0, &s0), &cfg.mul(&delta_0, &delta_0)),
            &cfg.mul(
                &alpha,
                &cfg.add(&cfg.mul(&delta_1, &s1), &cfg.mul(&delta_1, &delta_1)),
            ),
        );
        assert!(cfg.is_zero(&residue), "must preserve booleanity residue");

        bp.bit_slice_evals[0] = cfg.lift(&cfg.add(&b0, &delta_0));
        bp.bit_slice_evals[1] = cfg.lift(&cfg.add(&b1, &delta_1));

        let err = Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            num_vars,
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
            |proof| proof.ideal_check.combined_mle_values.swap(0, 1),
            |res| {
                assert!(matches!(res.unwrap_err(), ProtocolError::IdealCheck(..)));
            },
        );
    }
}
