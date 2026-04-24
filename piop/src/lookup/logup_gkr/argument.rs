//! Single-call lookup-argument API.
//!
//! Bundles the Phase-1 logup-GKR subprotocol and the Phase-2a leaf
//! constructor into a single `prove` / `verify` entry point suitable
//! for direct protocol wiring.
//!
//! # Protocol
//!
//! 1. Sample `alpha` from the transcript (lookup challenge).
//! 2. Prover: build leaf `(N, D)` MLEs from the trace data, run
//!    logup-GKR, and send `(LogupGkrProof, component_evals)` where
//!    `component_evals` carries `c_l(rho_row)`, `T(rho_row)`, and
//!    `m(rho_row)` for the GKR trace-level point `rho_row`.
//! 3. Verifier:
//!    * Re-sample `alpha`.
//!    * Run `LogupGkrVerifier`; obtain a subclaim with
//!      `(point, numerator_eval, denominator_eval)`.
//!    * Check `root_numerator == 0` — the lookup identity.
//!    * Reconstruct the expected `(N(rho), D(rho))` from
//!      `component_evals` via `expected_leaf_evals`; check it matches
//!      the GKR subclaim.
//!    * Return `LookupArgumentSubclaim { rho_row, component_evals, alpha }`
//!      for the OUTER protocol to bind the component evals to the
//!      actual committed columns (typically via a multipoint-eval
//!      reduction in a later step).

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use std::marker::PhantomData;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_utils::{add, inner_transparent_field::InnerTransparentField};

use super::{
    circuit::GrandSumCircuit,
    error::LogupGkrError,
    leaves::{LeafComponentEvals, build_lookup_leaves, expected_leaf_evals},
    proof::LogupGkrProof,
    prover::LogupGkrProver,
    verifier::LogupGkrVerifier,
};

/// Lookup-argument proof: the GKR proof + the four tail component
/// evaluations at the trace-level point the GKR descent produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupArgumentProof<F> {
    pub gkr_proof: LogupGkrProof<F>,
    pub component_evals: LeafComponentEvals<F>,
}

// Wire format: LogupGkrProof (Transcribable) + LeafComponentEvals
// (Transcribable), each with their own length prefix.
impl<F: PrimeField> GenTranscribable for LookupArgumentProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (gkr_proof, bytes) = LogupGkrProof::<F>::read_transcription_bytes_subset(bytes);
        let (component_evals, bytes) =
            LeafComponentEvals::<F>::read_transcription_bytes_subset(bytes);
        assert!(bytes.is_empty(), "trailing bytes");
        Self {
            gkr_proof,
            component_evals,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let buf = self.gkr_proof.write_transcription_bytes_subset(buf);
        let buf = self.component_evals.write_transcription_bytes_subset(buf);
        assert!(buf.is_empty(), "buffer size mismatch");
    }
}

impl<F: PrimeField> Transcribable for LookupArgumentProof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        add!(
            LogupGkrProof::<F>::LENGTH_NUM_BYTES,
            add!(
                self.gkr_proof.get_num_bytes(),
                add!(
                    LeafComponentEvals::<F>::LENGTH_NUM_BYTES,
                    self.component_evals.get_num_bytes()
                )
            )
        )
    }
}

/// Subclaim the verifier returns; the caller must validate that the
/// `component_evals` are consistent with the actual committed trace
/// MLEs at `rho_row`.
#[derive(Clone, Debug)]
pub struct LookupArgumentSubclaim<F> {
    /// Trace-level point ρ_row (length = `row_vars`).
    pub rho_row: Vec<F>,
    /// Prover-claimed evals at ρ_row that the outer protocol must bind.
    pub component_evals: LeafComponentEvals<F>,
    /// Lookup challenge sampled at the start of this subprotocol.
    pub alpha: F,
}

/// Errors for the lookup argument.
#[derive(Debug, thiserror::Error)]
pub enum LookupArgumentError<F> {
    #[error("GKR subprotocol failed: {0}")]
    Gkr(#[from] LogupGkrError<F>),
    /// The cumulative sum (root_numerator) is nonzero — the logup
    /// identity is violated, meaning either the lookup is invalid or
    /// the prover's data does not match.
    #[error("cumulative sum is nonzero; lookup identity fails")]
    NonzeroRootNumerator,
    /// The GKR denominator at the root is zero (makes the rational
    /// ill-defined); either a soundness failure or the lookup
    /// challenge collided with a table entry.
    #[error("root denominator is zero")]
    ZeroRootDenominator,
    /// GKR's `(N(ρ), D(ρ))` does not match the verifier-side
    /// reconstruction from the claimed component evals.
    #[error("leaf reconstruction mismatch")]
    LeafReconstructionMismatch,
    /// The verifier's expected `num_witness_columns` does not match
    /// the length of the proof's claimed witness evals.
    #[error(
        "witness column count mismatch: expected {expected}, got {got}"
    )]
    WitnessCountMismatch { expected: usize, got: usize },
}

/// Lookup-argument prover/verifier entry point.
pub struct LookupArgument<F>(PhantomData<F>);

impl<F> LookupArgument<F>
where
    F: PrimeField + InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    /// Sample `alpha` and absorb it. Used identically by prover and
    /// verifier so transcripts stay in sync.
    fn sample_alpha(transcript: &mut impl Transcript, cfg: &F::Config) -> F {
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        let alpha: F = transcript.get_field_challenge(cfg);
        transcript.absorb_random_field(&alpha, &mut buf);
        alpha
    }

    /// Absorb the component evals into the transcript (prover AND
    /// verifier, same order, so later challenges bind to them).
    fn absorb_component_evals(
        transcript: &mut impl Transcript,
        evals: &LeafComponentEvals<F>,
    ) {
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        for v in &evals.witness_evals {
            transcript.absorb_random_field(v, &mut buf);
        }
        transcript.absorb_random_field(&evals.table_eval, &mut buf);
        transcript.absorb_random_field(&evals.multiplicity_eval, &mut buf);
    }

    /// Prove the lookup relation.
    ///
    /// Returns the proof plus a [`LookupArgumentSubclaim`] for the
    /// caller to thread into the outer multipoint-eval / PCS
    /// reconciliation.
    pub fn prove(
        transcript: &mut impl Transcript,
        witness_columns: &[&DenseMultilinearExtension<F::Inner>],
        table: &DenseMultilinearExtension<F::Inner>,
        multiplicities: &DenseMultilinearExtension<F::Inner>,
        cfg: &F::Config,
    ) -> (LookupArgumentProof<F>, LookupArgumentSubclaim<F>) {
        assert!(!witness_columns.is_empty(), "need at least one witness column");

        let alpha = Self::sample_alpha(transcript, cfg);

        // Build the leaf (N, D) MLEs from the trace data and run the GKR.
        let leaves = build_lookup_leaves::<F>(
            witness_columns,
            table,
            multiplicities,
            &alpha,
            cfg,
        );
        let row_vars = leaves.row_vars;

        let circuit = GrandSumCircuit::<F>::build(
            leaves.numerator.clone(),
            leaves.denominator.clone(),
            cfg,
        );

        let (gkr_proof, gkr_subclaim) =
            LogupGkrProver::<F>::prove(transcript, &circuit, cfg);

        // ρ_row is the LOW `row_vars` coords of the GKR point.
        let rho_row: Vec<F> = gkr_subclaim.point[..row_vars].to_vec();

        // Compute the component evaluations the prover needs to send.
        let component_evals = compute_component_evals::<F>(
            witness_columns,
            table,
            multiplicities,
            &rho_row,
            cfg,
        );

        Self::absorb_component_evals(transcript, &component_evals);

        (
            LookupArgumentProof {
                gkr_proof,
                component_evals: component_evals.clone(),
            },
            LookupArgumentSubclaim {
                rho_row,
                component_evals,
                alpha,
            },
        )
    }

    /// Verify a lookup argument.
    ///
    /// On success returns `LookupArgumentSubclaim` that the caller
    /// must subsequently bind to the actual committed trace MLEs.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn verify(
        transcript: &mut impl Transcript,
        num_witness_columns: usize,
        row_vars: usize,
        proof: &LookupArgumentProof<F>,
        cfg: &F::Config,
    ) -> Result<LookupArgumentSubclaim<F>, LookupArgumentError<F>> {
        if proof.component_evals.witness_evals.len() != num_witness_columns {
            return Err(LookupArgumentError::WitnessCountMismatch {
                expected: num_witness_columns,
                got: proof.component_evals.witness_evals.len(),
            });
        }

        let alpha = Self::sample_alpha(transcript, cfg);

        // Layout determined by caller-provided shape:
        //   num_slots   = num_witness_columns + 1
        //   slot_vars   = ceil(log2(num_slots))
        //   total_vars  = row_vars + slot_vars
        let num_slots = num_witness_columns + 1;
        let slot_vars = num_slots.next_power_of_two().trailing_zeros() as usize;
        let total_vars = row_vars + slot_vars;

        let (gkr_sub, root_n, root_d) = LogupGkrVerifier::<F>::verify(
            transcript,
            total_vars,
            &proof.gkr_proof,
            cfg,
        )?;

        // Lookup identity: root_numerator / root_denominator == 0 iff the
        // lookup is valid. Since root_d is nonzero, root_n == 0 is the check.
        if root_d == F::zero_with_cfg(cfg) {
            return Err(LookupArgumentError::ZeroRootDenominator);
        }
        if root_n != F::zero_with_cfg(cfg) {
            return Err(LookupArgumentError::NonzeroRootNumerator);
        }

        // Split the final point: low `row_vars` = ρ_row, high = ρ_slot.
        let rho_row: Vec<F> = gkr_sub.point[..row_vars].to_vec();
        let rho_slot: &[F] = &gkr_sub.point[row_vars..];

        // Reconstruct the expected (N(ρ), D(ρ)) from the claimed component evals.
        let (expected_n, expected_d) = expected_leaf_evals::<F>(
            rho_slot,
            &proof.component_evals,
            &alpha,
            cfg,
        );

        if expected_n != gkr_sub.numerator_eval || expected_d != gkr_sub.denominator_eval {
            return Err(LookupArgumentError::LeafReconstructionMismatch);
        }

        Self::absorb_component_evals(transcript, &proof.component_evals);

        Ok(LookupArgumentSubclaim {
            rho_row,
            component_evals: proof.component_evals.clone(),
            alpha,
        })
    }
}

/// Helper: evaluate each trace MLE component at ρ_row to produce the
/// claimed values the prover must send.
fn compute_component_evals<F>(
    witness_columns: &[&DenseMultilinearExtension<F::Inner>],
    table: &DenseMultilinearExtension<F::Inner>,
    multiplicities: &DenseMultilinearExtension<F::Inner>,
    rho_row: &[F],
    cfg: &F::Config,
) -> LeafComponentEvals<F>
where
    F: PrimeField + InnerTransparentField,
{
    use zinc_poly::mle::MultilinearExtensionWithConfig;

    let witness_evals: Vec<F> = witness_columns
        .iter()
        .map(|col| {
            (*col)
                .clone()
                .evaluate_with_config(rho_row, cfg)
                .expect("witness column num_vars must equal row_vars")
        })
        .collect();
    let table_eval = table
        .clone()
        .evaluate_with_config(rho_row, cfg)
        .expect("table num_vars must equal row_vars");
    let multiplicity_eval = multiplicities
        .clone()
        .evaluate_with_config(rho_row, cfg)
        .expect("multiplicity num_vars must equal row_vars");

    LeafComponentEvals {
        witness_evals,
        table_eval,
        multiplicity_eval,
    }
}
