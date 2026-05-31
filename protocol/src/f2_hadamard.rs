//! Wiring-R Hadamard-discharge phase for the F_2 SHA-256 prover.
//!
//! Runs the coefficient-wise Hadamard zerocheck
//! ([`zinc_piop::lookup::hadamard`]) over the committed binary_poly
//! columns *before* the ψ_α projection is sampled, so α can later double
//! as the recombination element that pins the per-slice evals to the
//! column openings (see `protocol/src/f2_hadamard_plan.md`, Wiring R).
//!
//! This module provides the prover/verifier *phase* helpers. The
//! bit-slice evals they expose are discharged by the caller via
//! [`zinc_piop::lookup::booleanity::verify_bit_decomposition_consistency`]
//! once α and the Hadamard columns' α-evals at the Hadamard sumcheck
//! point `r*_H` are available.
//!
//! Phase-A scope: relations over primary committed columns with no row-
//! or bit-shifts (Δ = 0). Shifts, virtual operands, and the `W_β` carry
//! column land in later phases (see the plan).

use crypto_primitives::Field;
use zinc_piop::lookup::booleanity::compute_bit_slices_flat;
use zinc_piop::lookup::hadamard::{
    HadamardError, HadamardTriple, finalize_hadamard_prover, finalize_hadamard_verifier,
    prepare_hadamard_group, prepare_hadamard_verifier,
};
use zinc_piop::sumcheck::SumCheckError;
use zinc_piop::sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::binary_gf128::{BinaryFieldGF128, project_column_with_powers};
use zinc_transcript::traits::Transcript;

type Gf = BinaryFieldGF128;

/// A coefficient-wise Hadamard relation `W = U ⊙ V` between committed
/// binary_poly columns, by absolute column index. (Phase A: primary
/// columns, no shifts.)
#[derive(Clone, Copy, Debug)]
pub struct F2HadamardSpec {
    pub u_col: usize,
    pub v_col: usize,
    pub w_col: usize,
}

/// Proof data produced by the Hadamard phase: the zerocheck proof plus
/// the per-slice evals at `r*_H` (column-major over the distinct columns
/// the relations reference).
#[derive(Clone, Debug)]
pub struct F2HadamardProof {
    pub sumcheck_proof: MultiDegreeSumcheckProof<Gf>,
    pub bit_slice_evals: Vec<Gf>,
}

/// Errors emitted by [`verify_f2_hadamard_phase`].
#[derive(Debug, thiserror::Error)]
pub enum F2HadamardVerifyError {
    #[error("hadamard zerocheck: {0}")]
    Hadamard(#[from] HadamardError<Gf>),
    #[error("multi-degree sumcheck verify: {0:?}")]
    Sumcheck(SumCheckError<Gf>),
}

/// The distinct committed columns the relations reference, sorted. Both
/// prover and verifier derive this identically from the specs, so it
/// need not travel in the proof.
fn distinct_columns(specs: &[F2HadamardSpec]) -> Vec<usize> {
    let mut cols: Vec<usize> = specs
        .iter()
        .flat_map(|s| [s.u_col, s.v_col, s.w_col])
        .collect();
    cols.sort_unstable();
    cols.dedup();
    cols
}

fn relations_into_triples(specs: &[F2HadamardSpec], distinct: &[usize]) -> Vec<HadamardTriple> {
    let pos = |c: usize| {
        distinct
            .binary_search(&c)
            .expect("column referenced by a Hadamard spec must be in the distinct set")
    };
    specs
        .iter()
        .map(|s| HadamardTriple {
            u_col: pos(s.u_col),
            v_col: pos(s.v_col),
            w_col: pos(s.w_col),
        })
        .collect()
}

/// Prover half of the Wiring-R Hadamard phase. Returns `None` when there
/// are no relations. On success returns the proof plus the distinct
/// column list and the Hadamard sumcheck point `r*_H` — the caller needs
/// both after α to compute the columns' α-evals at `r*_H` for the
/// recombination discharge.
pub fn prove_f2_hadamard_phase<const D: usize>(
    transcript: &mut impl Transcript,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    specs: &[F2HadamardSpec],
    ic_evaluation_point: &[Gf],
    num_vars: usize,
) -> Option<(F2HadamardProof, Vec<usize>, Vec<Gf>)> {
    if specs.is_empty() {
        return None;
    }
    let cfg = ();
    let distinct = distinct_columns(specs);
    // Δ = 0 bit-slices for the distinct referenced columns. (Phase A:
    // clone the subset; the shift-aware builder lands with Phase B's
    // shifted operands.)
    let sub: Vec<DenseMultilinearExtension<BinaryPoly<D>>> =
        distinct.iter().map(|&c| columns[c].clone()).collect();
    let bit_slice_mles = compute_bit_slices_flat::<Gf, D>(&sub, &cfg);
    let relations = relations_into_triples(specs, &distinct);

    let (group, ancillary) =
        prepare_hadamard_group::<Gf, D>(transcript, bit_slice_mles, &relations, ic_evaluation_point, &cfg)
            .expect("hadamard group preparation")
            .expect("non-empty relations yield a group");

    let (sumcheck_proof, mut states) =
        MultiDegreeSumcheck::<Gf>::prove_as_subprotocol(transcript, vec![group], num_vars, &cfg);

    let r_star_h = states[0].randomness.clone();
    let bit_slice_evals = finalize_hadamard_prover::<Gf>(transcript, states.remove(0), ancillary, &cfg)
        .expect("hadamard prover finalize");

    Some((
        F2HadamardProof {
            sumcheck_proof,
            bit_slice_evals,
        },
        distinct,
        r_star_h,
    ))
}

/// Verifier half: checks the zerocheck (claimed sum zero + round
/// consistency + closing eval) and returns `(distinct_cols, r*_H)`. The
/// caller then computes each distinct column's α-eval at `r*_H` and runs
/// [`zinc_piop::lookup::booleanity::verify_bit_decomposition_consistency`]
/// against `proof.bit_slice_evals` to pin the slices to the commitments.
pub fn verify_f2_hadamard_phase<const D: usize>(
    transcript: &mut impl Transcript,
    proof: &F2HadamardProof,
    specs: &[F2HadamardSpec],
    ic_evaluation_point: &[Gf],
    num_vars: usize,
) -> Result<(Vec<usize>, Vec<Gf>), F2HadamardVerifyError> {
    let cfg = ();
    let distinct = distinct_columns(specs);
    let relations = relations_into_triples(specs, &distinct);

    let claimed = proof.sumcheck_proof.claimed_sums()[0];
    let ancillary = prepare_hadamard_verifier::<Gf, D>(
        transcript,
        claimed,
        &relations,
        ic_evaluation_point,
        &cfg,
    )?;
    let subclaims =
        MultiDegreeSumcheck::<Gf>::verify_as_subprotocol(transcript, num_vars, &proof.sumcheck_proof, &cfg)
            .map_err(F2HadamardVerifyError::Sumcheck)?;

    let r_star_h = subclaims.point().to_vec();
    finalize_hadamard_verifier::<Gf, D>(
        transcript,
        &proof.bit_slice_evals,
        &r_star_h,
        subclaims.expected_evaluations()[0],
        ancillary,
        &cfg,
    )?;

    Ok((distinct, r_star_h))
}

/// Compute the α-projected MLE evaluation of each `distinct` column at
/// `r_star` — the `parent_eval` that the recombination check
/// ([`zinc_piop::lookup::booleanity::verify_bit_decomposition_consistency`])
/// ties the per-slice evals to. `alpha_pows` must be `[1, α, …, α^{D-1}]`.
///
/// In Wiring R the prover computes these *after* α (so they share α with
/// the main projection) and ships them in the proof; the verifier
/// recombines `Σ_b α^b·v_b(r*_H) == parent_eval`. NOTE: until the
/// discharge opens these at `r*_H` via the (Issue-1) two-point
/// multipoint-eval, the parent evals are prover-supplied/trusted — the
/// in-flow check is completeness/honest-prover only.
pub fn alpha_parent_evals<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    distinct: &[usize],
    alpha_pows: &[BinaryFieldGF128],
    r_star: &[BinaryFieldGF128],
) -> Vec<BinaryFieldGF128> {
    let zero_inner = *BinaryFieldGF128::zero().inner();
    distinct
        .iter()
        .map(|&c| {
            let proj = project_column_with_powers::<D>(&columns[c].evaluations, alpha_pows);
            let mle = DenseMultilinearExtension::from_evaluations_vec(
                columns[c].num_vars,
                proj.iter().map(|x| *x.inner()).collect(),
                zero_inner,
            );
            mle.evaluate_with_config(r_star, &())
                .expect("parent α-eval at r*_H should succeed")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::Field;
    use crypto_primitives::boolean::Boolean;
    use zinc_piop::lookup::booleanity::verify_bit_decomposition_consistency;
    use zinc_poly::mle::MultilinearExtensionWithConfig;
    use zinc_poly::univariate::binary_gf128::{alpha_powers, project_column_with_powers};
    use zinc_transcript::Blake3Transcript;

    fn col_from_u32s<const D: usize>(
        patterns: &[u32],
    ) -> DenseMultilinearExtension<BinaryPoly<D>> {
        use std::array;
        let evaluations: Vec<BinaryPoly<D>> = patterns
            .iter()
            .map(|&p| {
                let coeffs: [Boolean; D] = array::from_fn(|i| Boolean::new((p >> i) & 1 != 0));
                BinaryPoly::<D>::new(coeffs)
            })
            .collect();
        let num_vars = patterns.len().next_power_of_two().trailing_zeros() as usize;
        DenseMultilinearExtension {
            num_vars,
            evaluations,
        }
    }

    fn ic_point() -> Vec<Gf> {
        vec![
            Gf::from_words([3, 0]),
            Gf::from_words([5, 0]),
            Gf::from_words([7, 0]),
        ]
    }

    /// α-projected MLE eval of each distinct column at `r*_H` — the
    /// `parent_eval` the recombination check ties the slices to.
    fn parent_evals<const D: usize>(
        columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
        distinct: &[usize],
        alpha: &Gf,
        r_star: &[Gf],
        num_vars: usize,
    ) -> Vec<Gf> {
        let pows = alpha_powers(alpha, D);
        let zero_i = *Gf::zero().inner();
        distinct
            .iter()
            .map(|&c| {
                let proj = project_column_with_powers::<D>(&columns[c].evaluations, &pows);
                let mle = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    proj.iter().map(|x| *x.inner()).collect(),
                    zero_i,
                );
                mle.evaluate_with_config(r_star, &()).unwrap()
            })
            .collect()
    }

    #[test]
    fn hadamard_phase_round_trip_and_recombination() {
        const D: usize = 4;
        let num_vars = 3;
        let specs = [F2HadamardSpec {
            u_col: 0,
            v_col: 1,
            w_col: 2,
        }];
        let ic = ic_point();

        let u = [0b1011u32, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let v = [0b1101u32, 0b1010, 0b0111, 0b1001, 0b0110, 0b1111, 0b1000, 0b0101];
        let w: Vec<u32> = u.iter().zip(&v).map(|(a, b)| a & b).collect();
        let columns = [
            col_from_u32s::<D>(&u),
            col_from_u32s::<D>(&v),
            col_from_u32s::<D>(&w),
        ];

        // ---- honest prover ----
        let mut pt = Blake3Transcript::new();
        let (proof, distinct_p, rstar_p) =
            prove_f2_hadamard_phase::<D>(&mut pt, &columns, &specs, &ic, num_vars).unwrap();

        // ---- verifier: zerocheck ----
        let mut vt = Blake3Transcript::new();
        let (distinct_v, rstar_v) =
            verify_f2_hadamard_phase::<D>(&mut vt, &proof, &specs, &ic, num_vars).unwrap();
        assert_eq!(distinct_p, distinct_v);
        assert_eq!(rstar_p, rstar_v);

        // ---- recombination: pin the slices to the columns at a test α ----
        let alpha = Gf::from_words([0x9E37_79B9, 0x1234_5678]);
        let parents = parent_evals::<D>(&columns, &distinct_v, &alpha, &rstar_v, num_vars);
        verify_bit_decomposition_consistency(&parents, &proof.bit_slice_evals, &alpha, D)
            .expect("bit-decomposition recombination");

        // ---- corrupt W: verifier's claimed-sum gate rejects ----
        let mut w_bad = w.clone();
        w_bad[0] ^= 1;
        let columns_bad = [
            col_from_u32s::<D>(&u),
            col_from_u32s::<D>(&v),
            col_from_u32s::<D>(&w_bad),
        ];
        let mut pt2 = Blake3Transcript::new();
        let (proof2, _, _) =
            prove_f2_hadamard_phase::<D>(&mut pt2, &columns_bad, &specs, &ic, num_vars).unwrap();
        let mut vt2 = Blake3Transcript::new();
        let rejected = verify_f2_hadamard_phase::<D>(&mut vt2, &proof2, &specs, &ic, num_vars);
        assert!(matches!(
            rejected,
            Err(F2HadamardVerifyError::Hadamard(
                HadamardError::NonZeroClaimedSum { .. }
            ))
        ));
    }

    #[test]
    fn empty_specs_is_a_noop() {
        const D: usize = 4;
        let mut pt = Blake3Transcript::new();
        let cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>> = vec![];
        assert!(prove_f2_hadamard_phase::<D>(&mut pt, &cols, &[], &ic_point(), 3).is_none());
    }
}
