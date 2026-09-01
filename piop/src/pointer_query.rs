//! Pointer-query (composed read) subprotocol.
//!
//! Discharges the composed-read obligation `R(x) = V(b_1(x)..b_mu(x))`
//! for every row `x` of the trace cube via two chained sumchecks, with
//! the R-side anchored at the step-4 evaluation point `r*` whose
//! per-column evaluations (`up_evals`) the CPR resolver already emits
//! and the downstream cascade already discharges:
//!
//! ```text
//! R~(r*)  =  sum_y u~(y) * V~(y)          (sumcheck A, degree 2)
//! u~(r_A) =  sum_x eq(r*,x) *
//!            prod_nu (b_nu(x)*(r_A)_nu + (1-b_nu(x))*(1-(r_A)_nu))
//!                                          (sumcheck B, degree mu+1)
//! ```
//!
//! where `u(y) = sum_x eq(r*,x) * eqt(b(x), y)` is the eq-mass pushed
//! through the pointer map — prover-built in `O(2^mu)`, never
//! committed. Multiple reads batch under one transcript challenge
//! `alpha`. The endpoint claims (`V~_j(r_A)`, `b~_(j,nu)(r_B)`) are
//! discharged by the protocol layer against the int-batch commitment.
//! Soundness preconditions (bit booleanity, padding read-consistency,
//! range) and the staging plan: `documentation/pointer-query-design.md`.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
use thiserror::Error;
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    utils::{build_eq_x_r_inner, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_uair::ComposedReadSpec;
use zinc_utils::{add, inner_transparent_field::InnerTransparentField, mul};

use crate::sumcheck::{MLSumcheck, SumCheckError, SumcheckProof};

//
// Data structures
//

/// Proof for the pointer-query subprotocol: the two chained sumchecks
/// and the bridge evaluations `u~_j(r_A)` between them. The endpoint
/// claims on committed columns travel outside this struct, in the
/// protocol-level lifted evaluations at `r_A` and `r_B`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointerQueryProof<F: PrimeField> {
    pub sumcheck_a: SumcheckProof<F>,
    pub sumcheck_b: SumcheckProof<F>,
    /// `u~_j(r_A)` per read, in spec order: sumcheck A's endpoint needs
    /// them, and sumcheck B's claimed sum must equal their batch.
    pub u_evals_at_r_a: Vec<F>,
}

/// The two protocol-derived points the endpoint claims live at.
#[derive(Clone, Debug)]
pub struct PointerQueryPoints<F: PrimeField> {
    pub r_a: Vec<F>,
    pub r_b: Vec<F>,
}

#[derive(Error, Debug)]
pub enum PointerQueryError<F: PrimeField> {
    #[error("sumcheck error in the pointer query: {0}")]
    Sumcheck(#[from] SumCheckError<F>),
    #[error("bit_cols length {got} does not match the trace num_vars {num_vars}")]
    BitColumnCount { got: usize, num_vars: usize },
    #[error("sumcheck {which} claimed sum does not match its anchor")]
    ClaimedSumMismatch { which: &'static str },
    #[error("sumcheck {which} endpoint does not match the claimed evaluations")]
    EndpointMismatch { which: &'static str },
    #[error("proof carries {got} u-evaluations for {expected} reads")]
    UEvalCount { got: usize, expected: usize },
    #[error("arithmetic error building the eq vector")]
    EqBuild,
}

//
// Prover
//

/// Prove all declared composed reads. `int_cols_f` are the projected
/// int-section column MLEs (int-section order), `addrs` the per-read,
/// per-row addresses read off the raw trace, `r_star` the step-4
/// evaluation point. Draws the batching challenge `alpha`, runs
/// sumcheck A then B, absorbing the bridge evaluations in between.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_pointer_queries<F>(
    transcript: &mut impl Transcript,
    specs: &[ComposedReadSpec],
    int_flat_offset: usize,
    int_cols_f: &[DenseMultilinearExtension<F::Inner>],
    addrs: &[Vec<usize>],
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<(PointerQueryProof<F>, PointerQueryPoints<F>), PointerQueryError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default + Clone,
    F::Modulus: ConstTranscribable,
{
    let num_vars = r_star.len();
    for spec in specs {
        if spec.bit_cols.len() != num_vars {
            return Err(PointerQueryError::BitColumnCount {
                got: spec.bit_cols.len(),
                num_vars,
            });
        }
    }

    let alpha: F = transcript.get_field_challenge(field_cfg);
    let mut buf = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&alpha, &mut buf);

    let eq_r_star =
        build_eq_x_r_inner::<F>(r_star, field_cfg).map_err(|_| PointerQueryError::EqBuild)?;
    let zero = F::zero_with_cfg(field_cfg);

    // The eq-mass pushed through each read's pointer map:
    // u_j[y] = sum over rows x with address y of eq(r*, x).
    // Accumulated in F, carried into the MLE as inner values.
    let rows = 1usize << num_vars;
    let mut a_mles: Vec<DenseMultilinearExtension<F::Inner>> = Vec::with_capacity(2 * specs.len());
    for (spec, read_addrs) in specs.iter().zip(addrs) {
        let mut u = vec![zero.clone(); rows];
        for (x, &addr) in read_addrs.iter().enumerate() {
            let eq_x = F::new_unchecked_with_cfg(eq_r_star.evaluations[x].clone(), field_cfg);
            u[addr] = u[addr].clone() + eq_x;
        }
        a_mles.push(DenseMultilinearExtension {
            num_vars,
            evaluations: u.into_iter().map(|v| v.into_inner()).collect(),
        });
        a_mles.push(int_cols_f[spec.value_col - int_flat_offset].clone());
    }

    let alphas = alpha_powers(alpha, specs.len(), field_cfg);
    let a_alphas = alphas.clone();
    let a_zero = zero.clone();
    let (sumcheck_a, state_a) = MLSumcheck::prove_as_subprotocol(
        transcript,
        a_mles,
        num_vars,
        2,
        move |evals: &[F]| {
            let mut acc = a_zero.clone();
            for (j, alpha_j) in a_alphas.iter().enumerate() {
                acc = acc + alpha_j.clone() * evals[2 * j].clone() * evals[2 * j + 1].clone();
            }
            acc
        },
        field_cfg,
    );
    let r_a = state_a.randomness.clone();

    // Bridge: u~_j(r_A), needed by the verifier for A's endpoint and
    // as B's claimed sum; absorbed before B draws any challenge.
    let last = r_a[num_vars - 1].clone();
    let u_evals_at_r_a: Vec<F> = state_a
        .mles
        .iter()
        .step_by(2)
        .map(|mle| mle.clone().evaluate_with_config(std::slice::from_ref(&last), field_cfg))
        .collect::<Result<_, _>>()
        .map_err(|_| PointerQueryError::EqBuild)?;
    transcript.absorb_random_field_slice(&u_evals_at_r_a, &mut buf);

    let mut b_mles: Vec<DenseMultilinearExtension<F::Inner>> =
        Vec::with_capacity(1 + num_vars * specs.len());
    b_mles.push(eq_r_star);
    for spec in specs {
        for &bit_col in &spec.bit_cols {
            b_mles.push(int_cols_f[bit_col - int_flat_offset].clone());
        }
    }

    let r_a_for_b = r_a.clone();
    let b_alphas = alphas;
    let one = F::one_with_cfg(field_cfg);
    let b_zero = zero;
    let (sumcheck_b, state_b) = MLSumcheck::prove_as_subprotocol(
        transcript,
        b_mles,
        num_vars,
        num_vars + 1,
        move |evals: &[F]| {
            // evals: [eq(r*,·), b_{1,1}..b_{1,mu}, b_{2,1}.., ...]
            let mut acc = b_zero.clone();
            for (j, alpha_j) in b_alphas.iter().enumerate() {
                let mut prod = evals[0].clone();
                for (nu, r_nu) in r_a_for_b.iter().enumerate() {
                    let b = evals[1 + j * r_a_for_b.len() + nu].clone();
                    // f = b*r + (1-b)*(1-r)
                    let f = b.clone() * r_nu.clone()
                        + (one.clone() - b) * (one.clone() - r_nu.clone());
                    prod = prod * f;
                }
                acc = acc + alpha_j.clone() * prod;
            }
            acc
        },
        field_cfg,
    );
    let r_b = state_b.randomness.clone();

    Ok((
        PointerQueryProof {
            sumcheck_a,
            sumcheck_b,
            u_evals_at_r_a,
        },
        PointerQueryPoints { r_a, r_b },
    ))
}

//
// Verifier
//

/// Verify the pointer-query sumchecks against their anchors.
/// `result_up_evals[j]` are the resolver's up-evaluations of the
/// result columns (already discharged at `r*` by the downstream
/// cascade); `v_evals_at_r_a[j]` and `b_evals_at_r_b[j][nu]` are the
/// endpoint claims the caller reads off the lifted evaluations it
/// discharges against the int commitment at the returned points.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_pointer_queries<F>(
    transcript: &mut impl Transcript,
    specs: &[ComposedReadSpec],
    result_up_evals: &[F],
    v_evals_at_r_a: &[F],
    b_evals_at_r_b: &[Vec<F>],
    proof: &PointerQueryProof<F>,
    r_star: &[F],
    field_cfg: &F::Config,
) -> Result<PointerQueryPoints<F>, PointerQueryError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default + Clone,
    F::Modulus: ConstTranscribable,
{
    let num_vars = r_star.len();
    for spec in specs {
        if spec.bit_cols.len() != num_vars {
            return Err(PointerQueryError::BitColumnCount {
                got: spec.bit_cols.len(),
                num_vars,
            });
        }
    }
    if proof.u_evals_at_r_a.len() != specs.len() {
        return Err(PointerQueryError::UEvalCount {
            got: proof.u_evals_at_r_a.len(),
            expected: specs.len(),
        });
    }

    let alpha: F = transcript.get_field_challenge(field_cfg);
    let mut buf = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field(&alpha, &mut buf);
    let alphas = alpha_powers(alpha, specs.len(), field_cfg);
    let zero = F::zero_with_cfg(field_cfg);

    // Sumcheck A: anchored to the already-discharged result evals.
    let anchor = batched(&alphas, result_up_evals, &zero);
    if proof.sumcheck_a.claimed_sum != anchor {
        return Err(PointerQueryError::ClaimedSumMismatch { which: "A" });
    }
    let subclaim_a =
        MLSumcheck::verify_as_subprotocol(transcript, num_vars, 2, &proof.sumcheck_a, field_cfg)?;
    let r_a = subclaim_a.point.clone();

    let expected_a = alphas
        .iter()
        .zip(proof.u_evals_at_r_a.iter().zip(v_evals_at_r_a))
        .fold(zero.clone(), |acc, (alpha_j, (u, v))| {
            acc + alpha_j.clone() * u.clone() * v.clone()
        });
    if expected_a != subclaim_a.expected_evaluation {
        return Err(PointerQueryError::EndpointMismatch { which: "A" });
    }
    transcript.absorb_random_field_slice(&proof.u_evals_at_r_a, &mut buf);

    // Sumcheck B: anchored to the bridge evaluations.
    let anchor_b = batched(&alphas, &proof.u_evals_at_r_a, &zero);
    if proof.sumcheck_b.claimed_sum != anchor_b {
        return Err(PointerQueryError::ClaimedSumMismatch { which: "B" });
    }
    let subclaim_b = MLSumcheck::verify_as_subprotocol(
        transcript,
        num_vars,
        num_vars + 1,
        &proof.sumcheck_b,
        field_cfg,
    )?;
    let r_b = subclaim_b.point.clone();

    let one = F::one_with_cfg(field_cfg);
    let eq_r_star_r_b =
        eq_eval(r_star, &r_b, one.clone()).map_err(|_| PointerQueryError::EqBuild)?;
    let expected_b = alphas
        .iter()
        .zip(b_evals_at_r_b)
        .fold(zero.clone(), |acc, (alpha_j, bits)| {
            let prod = r_a
                .iter()
                .zip(bits)
                .fold(eq_r_star_r_b.clone(), |prod, (r_nu, b)| {
                    prod * (b.clone() * r_nu.clone()
                        + (one.clone() - b.clone()) * (one.clone() - r_nu.clone()))
                });
            acc + alpha_j.clone() * prod
        });
    if expected_b != subclaim_b.expected_evaluation {
        return Err(PointerQueryError::EndpointMismatch { which: "B" });
    }

    Ok(PointerQueryPoints { r_a, r_b })
}

//
// Helpers
//

#[allow(clippy::arithmetic_side_effects)]
fn alpha_powers<F>(alpha: F, count: usize, field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField + FromPrimitiveWithConfig,
{
    let mut powers = Vec::with_capacity(count);
    let mut acc = F::one_with_cfg(field_cfg);
    for _ in 0..count {
        powers.push(acc.clone());
        acc = acc * alpha.clone();
    }
    powers
}

#[allow(clippy::arithmetic_side_effects)]
fn batched<F: PrimeField>(alphas: &[F], values: &[F], zero: &F) -> F {
    alphas
        .iter()
        .zip(values)
        .fold(zero.clone(), |acc, (alpha_j, v)| {
            acc + alpha_j.clone() * v.clone()
        })
}

//
// Transcription: [u32 n_reads] and, when n_reads > 0, [field cfg]
// [u_evals: n_reads inner elements] [u32 a_bytes][sumcheck_a]
// [u32 b_bytes][sumcheck_b].
//

impl<F> GenTranscribable for PointerQueryProof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (n, bytes) = read_u32_prefix(bytes);
        assert!(
            n > 0,
            "an absent pointer-query proof is encoded as absence at the protocol layer"
        );
        let mod_size = F::Modulus::NUM_BYTES;
        let cfg = zinc_transcript::read_field_cfg::<F>(&bytes[..mod_size]);
        let bytes = &bytes[mod_size..];
        let u_end = mul!(n, F::Inner::NUM_BYTES);
        let u_evals_at_r_a = zinc_transcript::read_field_vec_with_cfg(&bytes[..u_end], &cfg);
        let bytes = &bytes[u_end..];
        let (a_len, bytes) = read_u32_prefix(bytes);
        let sumcheck_a = SumcheckProof::read_transcription_bytes_exact(&bytes[..a_len]);
        let bytes = &bytes[a_len..];
        let (b_len, bytes) = read_u32_prefix(bytes);
        let sumcheck_b = SumcheckProof::read_transcription_bytes_exact(&bytes[..b_len]);
        assert!(
            bytes[b_len..].is_empty(),
            "trailing bytes after PointerQueryProof"
        );
        Self {
            sumcheck_a,
            sumcheck_b,
            u_evals_at_r_a,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        let n = self.u_evals_at_r_a.len();
        assert!(n > 0, "an absent pointer-query proof is written as absence");
        buf = write_u32_prefix(buf, n);
        let modulus = self.u_evals_at_r_a[0].modulus();
        buf = zinc_transcript::append_field_cfg::<F>(buf, &modulus);
        for u in &self.u_evals_at_r_a {
            let (head, rest) = buf.split_at_mut(F::Inner::NUM_BYTES);
            u.clone().into_inner().write_transcription_bytes_exact(head);
            buf = rest;
        }
        for proof in [&self.sumcheck_a, &self.sumcheck_b] {
            let len = proof.get_num_bytes();
            buf = write_u32_prefix(buf, len);
            let (head, rest) = buf.split_at_mut(len);
            proof.write_transcription_bytes_exact(head);
            buf = rest;
        }
        assert!(
            buf.is_empty(),
            "PointerQueryProof leftover buffer (encoding underflow)"
        );
    }
}

impl<F> Transcribable for PointerQueryProof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        let n = self.u_evals_at_r_a.len();
        let mut total = add!(u32::NUM_BYTES, F::Modulus::NUM_BYTES);
        total = add!(total, mul!(n, F::Inner::NUM_BYTES));
        total = add!(total, add!(u32::NUM_BYTES, self.sumcheck_a.get_num_bytes()));
        total = add!(total, add!(u32::NUM_BYTES, self.sumcheck_b.get_num_bytes()));
        total
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn write_u32_prefix<'a>(buf: &'a mut [u8], val: usize) -> &'a mut [u8] {
    let (head, rest) = buf.split_at_mut(u32::NUM_BYTES);
    u32::try_from(val)
        .expect("length must fit into u32")
        .write_transcription_bytes_exact(head);
    rest
}

fn read_u32_prefix(bytes: &[u8]) -> (usize, &[u8]) {
    let (val, rest) = u32::read_transcription_bytes_subset(bytes);
    (
        usize::try_from(val).expect("length must fit into usize"),
        rest,
    )
}
