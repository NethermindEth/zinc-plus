//! End-to-end prover/verifier pipeline for Zinc+ SNARKs.
//!
//! This module provides [`prove`] and [`verify`] functions that compose:
//! - Batched Zip+ PCS (commit → test → evaluate → verify)
//! - Zinc+ PIOP (ideal check → combined poly resolver → sumcheck)
//!
//! The pipeline works over `BinaryPoly<DEGREE_PLUS_ONE>` traces.

use std::time::{Duration, Instant};

use crypto_bigint::modular::MontyParams;
use crypto_bigint::Odd;
use crypto_primitives::crypto_bigint_monty::MontyField;
use crypto_primitives::{Field, FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig, PrimeField};
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::mle::MultilinearExtensionWithConfig;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_transcript::KeccakTranscript;
use zinc_uair::constraint_counter::count_constraints;
use zinc_uair::degree_counter::count_max_degree;
use zinc_uair::Uair;
use zinc_utils::from_ref::FromRef;
use zinc_utils::projectable_to_field::ProjectableToField;

use zip_plus::code::LinearCode;
use zip_plus::pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes};
use zip_plus::pcs::ZipPlusProof;

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;
use zinc_piop::lookup::{
    GkrPipelineLookupProof, PipelineLookupProof, LookupColumnSpec,
    LookupSumcheckGroup,
    prove_batched_lookup_with_indices,
    verify_gkr_batched_lookup,
    verify_batched_lookup,
    BatchedDecompLogupProtocol,
    group_lookup_specs,
    AffineLookupSpec,
};
use zinc_piop::lookup::pipeline::{
    LookupGroupMeta,
    build_lookup_instance_from_indices_pub,
    generate_table_and_shifts,
};
use zinc_piop::lookup::LookupWitnessSource;
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
use zinc_piop::sumcheck::{MLSumcheck, SumcheckProof};
use zinc_piop::sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof};
use zinc_piop::shift_sumcheck::{
    ShiftClaim, ShiftSumcheckProof, ShiftRoundPoly,
    shift_sumcheck_prove, shift_sumcheck_verify,
    shift_sumcheck_verify_pre, shift_sumcheck_verify_finalize,
};
#[cfg(feature = "parallel")]
use rayon::join as rayon_join;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_uair::ideal::{Ideal, IdealCheck};
use zinc_uair::ideal_collector::IdealOrZero;

/// Trivial ideal that always passes membership checks.
///
/// The IC verifier requires checking whether the combined MLE values
/// belong to the UAIR ideals lifted to the PIOP field. However, for
/// UAIRs defined over F₂\[X\] (like SHA-256), F₂ arithmetic (XOR = add)
/// does **not** lift to F_p arithmetic (1 + 1 = 2 ≠ 0). The prover
/// evaluates constraints over the projected field and the combined
/// polynomial outputs are not in the ideal when viewed over F_p.
///
/// Soundness is preserved because:
/// 1. The sumcheck / CPR verifies that the prover's claimed sums are
///    consistent with the committed trace (algebraic check over F_p).
/// 2. The PCS verifies proximity of the committed polynomials.
///
/// This matches the existing piop test pattern where the ideal check
/// always uses `IdealOrZero::Zero` (which bypasses membership testing).
#[derive(Clone, Copy, Debug)]
pub struct TrivialIdeal;

impl Ideal for TrivialIdeal {}

impl FromRef<TrivialIdeal> for TrivialIdeal {
    #[inline(always)]
    fn from_ref(ideal: &TrivialIdeal) -> Self {
        *ideal
    }
}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for TrivialIdeal {
    #[inline]
    fn contains(&self, _value: &DynamicPolynomialF<F>) -> bool {
        true
    }
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// Timing breakdown for a single prove/verify invocation.
#[derive(Clone, Debug, Default)]
pub struct TimingBreakdown {
    pub pcs_commit: Duration,
    pub ideal_check: Duration,
    pub combined_poly_resolver: Duration,
    pub lookup: Duration,
    pub pcs_prove: Duration,
    pub serialize: Duration,
    pub total: Duration,
}

/// Timing breakdown for verification.
#[derive(Clone, Debug, Default)]
pub struct VerifyTimingBreakdown {
    pub ideal_check_verify: Duration,
    pub combined_poly_resolver_verify: Duration,
    pub lookup_verify: Duration,
    pub pcs_verify: Duration,
    pub total: Duration,
}

/// Lookup proof variant: either GKR-based or classic batched decomposition.
#[derive(Clone, Debug)]
pub enum LookupProofData {
    /// GKR fractional sumcheck based (no chunks/inverses sent).
    Gkr(GkrPipelineLookupProof<PiopField>),
    /// Classic batched decomposition (chunks + inverses + sumcheck).
    Classic(PipelineLookupProof<PiopField>),
    /// CPR + classic lookup sumchecks batched into a multi-degree sumcheck.
    ///
    /// Group 0 is the CPR, groups 1..N are the lookup table groups.
    /// All share the same verifier challenges and evaluation point.
    BatchedClassic(BatchedCprLookupProof),
}

/// Combined CPR + Classic lookup proof using multi-degree sumcheck.
///
/// The CPR sumcheck (degree `max_degree + 2`) and one or more lookup
/// sumchecks (degree 2) are run in lockstep with shared verifier
/// randomness, producing a single evaluation point.
#[derive(Clone, Debug)]
pub struct BatchedCprLookupProof {
    /// Multi-degree sumcheck proof (group 0 = CPR, groups 1.. = lookup).
    pub md_proof: MultiDegreeSumcheckProof<PiopField>,
    /// CPR up evaluations at the shared point.
    pub cpr_up_evals: Vec<PiopField>,
    /// CPR down evaluations at the shared point.
    pub cpr_down_evals: Vec<PiopField>,
    /// Per-group lookup metadata.
    pub lookup_group_meta: Vec<LookupGroupMeta>,
    /// Per-group lookup proof data (chunks, mults, inverses — no sumcheck).
    pub lookup_group_proofs: Vec<zinc_piop::lookup::BatchedDecompLogupProof<PiopField>>,
}

/// Full proof data produced by the prover.
#[derive(Clone, Debug)]
pub struct ZincProof {
    /// Serialized PCS proof bytes.
    pub pcs_proof_bytes: Vec<u8>,
    /// PCS commitment (Merkle root + batch size).
    pub commitment: ZipPlusCommitment,
    /// IdealCheck proof: evaluated combined polynomials at random point.
    pub ic_proof_values: Vec<Vec<u8>>,
    /// CPR proof: sumcheck proof + up/down evaluations.
    pub cpr_sumcheck_messages: Vec<Vec<u8>>,
    pub cpr_sumcheck_claimed_sum: Vec<u8>,
    pub cpr_up_evals: Vec<Vec<u8>>,
    pub cpr_down_evals: Vec<Vec<u8>>,
    /// Batched decomposed LogUp proof for column typing constraints.
    /// `None` if no lookup columns were specified.
    pub lookup_proof: Option<LookupProofData>,
    /// Shift sumcheck proof. `None` if the UAIR has no explicit shift specs.
    pub shift_sumcheck: Option<SerializedShiftSumcheckProof>,
    /// PCS evaluation claims from CPR (evaluation point in the field).
    pub evaluation_point_bytes: Vec<Vec<u8>>,
    /// PCS evaluation values (evaluations of committed polys at the point).
    pub pcs_evals_bytes: Vec<Vec<u8>>,
    /// Prover timing breakdown.
    pub timing: TimingBreakdown,
}

/// Serialized shift sumcheck proof data.
#[derive(Clone, Debug)]
pub struct SerializedShiftSumcheckProof {
    /// Round polynomials, serialized: each round has 3 field element evaluations.
    pub rounds: Vec<Vec<u8>>,
    /// Per-claim v_finals (source column evaluations at challenge point).
    pub v_finals: Vec<Vec<u8>>,
}

/// Result of the verification.
#[derive(Clone, Debug)]
pub struct VerifyResult {
    pub accepted: bool,
    pub timing: VerifyTimingBreakdown,
}

// ─── Field configuration ────────────────────────────────────────────────────

/// 192-bit Montgomery field (3 × 64-bit limbs) used for the PIOP.
pub const FIELD_LIMBS: usize = 3;
pub type PiopField = MontyField<FIELD_LIMBS>;
/// Unsigned integer type matching the PIOP field's internal representation.
type PiopUint = crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS>;

/// Returns a fixed configuration for the PIOP field.
///
/// Uses a known-good 128-bit prime: 0x860995AE68FC80E1B1BD1E39D54B33.
pub fn piop_field_config() -> MontyParams<FIELD_LIMBS> {
    let modulus = crypto_bigint::Uint::<FIELD_LIMBS>::from_be_hex(
        "000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

/// Serialize a PiopField element to bytes (using Montgomery representation
/// for Fiat-Shamir consistency).
fn field_to_bytes(f: &PiopField) -> Vec<u8> {
    let mut buf = vec![0u8; <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES];
    f.inner().write_transcription_bytes(&mut buf);
    buf
}

/// Deserialize a PiopField element from bytes (Montgomery representation).
pub fn field_from_bytes(bytes: &[u8], cfg: &<PiopField as PrimeField>::Config) -> PiopField {
    let inner = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::read_transcription_bytes(bytes);
    PiopField::from_montgomery(inner, cfg)
}

/// Filter a trace to keep only private (non-public) columns.
///
/// Public columns are known to the verifier and need not be committed via the
/// PCS. Dropping them before `ZipPlus::commit` / `ZipPlus::prove` reduces
/// the Merkle-tree size and encoding work.
fn private_trace<T: Clone>(
    trace: &[DenseMultilinearExtension<T>],
    public_columns: &[usize],
) -> Vec<DenseMultilinearExtension<T>> {
    if public_columns.is_empty() {
        return trace.to_vec();
    }
    trace
        .iter()
        .enumerate()
        .filter(|(i, _)| !public_columns.contains(i))
        .map(|(_, col)| col.clone())
        .collect()
}

/// Convert a `PiopField` evaluation point to an arbitrary PCS field `F` by
/// going through the canonical unsigned integer representation.
///
/// Uses `Uint<FIELD_LIMBS>` as the intermediate representation, which can
/// hold any value in the PIOP field regardless of the modulus size.
/// This is used only in generic pipeline functions where `PcsF` may differ
/// from `PiopField`.
pub fn piop_point_to_pcs_field<F>(
    piop_point: &[PiopField],
    pcs_cfg: &F::Config,
) -> Vec<F>
where
    F: PrimeField + FromWithConfig<PiopUint>,
{
    piop_point
        .iter()
        .map(|f| {
            let uint = f.retrieve(); // canonical Uint<FIELD_LIMBS>
            uint.into_with_cfg(pcs_cfg)
        })
        .collect()
}

/// Derive the PCS evaluation point from a CPR evaluation point.
///
/// Takes the first `num_vars` coordinates of the CPR evaluation point and
/// converts them to `i128` (the PCS field for standard 128-bit pipelines).
/// This is used by benchmarks that run individual pipeline steps.
pub fn derive_pcs_point(cpr_eval_point: &[PiopField], num_vars: usize) -> Vec<i128> {
    cpr_eval_point[..num_vars]
        .iter()
        .map(|f| {
            let uint = f.retrieve();
            let words = uint.as_words();
            debug_assert!(
                words[2] == 0,
                "PIOP field element too large for i128: {words:?}"
            );
            (words[0] as i128) | ((words[1] as i128) << 64)
        })
        .collect()
}

/// Reconstruct the PCS evaluation point from a [`ZincProof`]'s serialized
/// `evaluation_point_bytes` as `PiopField` elements.
///
/// This is a convenience wrapper so that benchmarks that hold a proof
/// object can recover the real CPR evaluation point (which is now used
/// directly as the PCS evaluation point) without re-running the PIOP.
pub fn pcs_point_from_proof(proof: &ZincProof) -> Vec<PiopField> {
    let cfg = piop_field_config();
    proof
        .evaluation_point_bytes
        .iter()
        .map(|b| field_from_bytes(b, &cfg))
        .collect()
}

/// Reconstruct the full `up_evals` vector from private (prover-supplied)
/// and public (verifier-computed) column evaluations.
///
/// `private_evals` contains evaluations for non-public columns in order.
/// `public_evals` contains evaluations for public columns, aligned with
/// `public_columns` indices.
/// The result is a vector of length `total_cols` with each evaluation
/// placed at its correct flattened column index.
#[allow(clippy::arithmetic_side_effects)]
pub fn reconstruct_up_evals(
    private_evals: &[PiopField],
    public_evals: &[PiopField],
    public_columns: &[usize],
    total_cols: usize,
) -> Vec<PiopField> {
    let mut full = Vec::with_capacity(total_cols);
    let public_set: std::collections::HashMap<usize, usize> = public_columns
        .iter()
        .enumerate()
        .map(|(i, &col)| (col, i))
        .collect();
    let mut private_idx = 0usize;
    for col in 0..total_cols {
        if let Some(&pub_idx) = public_set.get(&col) {
            full.push(public_evals[pub_idx].clone());
        } else {
            full.push(private_evals[private_idx].clone());
            private_idx += 1;
        }
    }
    debug_assert_eq!(private_idx, private_evals.len());
    full
}

/// Reconstruct the full `v_finals` array for a shift sumcheck from
/// private (prover-supplied) and public (verifier-computed) entries.
///
/// `is_public_shift` returns `true` for shift indices whose source column
/// is public.  Private entries are consumed in order from `private_v_finals`;
/// public entries are consumed in order from `public_v_finals`.
#[allow(clippy::arithmetic_side_effects)]
fn reconstruct_shift_v_finals(
    private_v_finals: &[PiopField],
    public_v_finals: &[PiopField],
    num_shifts: usize,
    is_public_shift: impl Fn(usize) -> bool,
) -> Vec<PiopField> {
    let mut full = Vec::with_capacity(num_shifts);
    let mut priv_idx = 0usize;
    let mut pub_idx = 0usize;
    for i in 0..num_shifts {
        if is_public_shift(i) {
            full.push(public_v_finals[pub_idx].clone());
            pub_idx += 1;
        } else {
            full.push(private_v_finals[priv_idx].clone());
            priv_idx += 1;
        }
    }
    debug_assert_eq!(priv_idx, private_v_finals.len());
    debug_assert_eq!(pub_idx, public_v_finals.len());
    full
}

/// Project only the trace columns referenced by `lookup_specs` to field
/// elements, extracting raw integer indices at the same time.
///
/// Returns `(columns, raw_indices, remapped_specs)` where `columns` and
/// `raw_indices` contain only the needed columns and `remapped_specs` has
/// column indices adjusted to the 0-based position in `columns`.
/// Extract lookup columns from an already-projected field trace and compute
/// raw integer indices from the original BinaryPoly trace.
///
/// This avoids redundant field-element projection when the CPR step has
/// already projected the full trace with the same `projecting_element`.
///
/// Returns `(columns, raw_indices, remapped_specs, reverse_index_map)` where
/// `reverse_index_map[remapped_idx]` is the original trace column index.
fn extract_lookup_columns_from_field_trace<const D: usize>(
    bp_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    field_trace: &[DenseMultilinearExtension<<PiopField as Field>::Inner>],
    lookup_specs: &[LookupColumnSpec],
    field_cfg: &MontyParams<FIELD_LIMBS>,
) -> (Vec<Vec<PiopField>>, Vec<Vec<usize>>, Vec<LookupColumnSpec>, Vec<usize>) {
    use std::collections::BTreeMap;

    // Collect unique column indices (sorted for determinism).
    let mut needed: BTreeMap<usize, usize> = BTreeMap::new();
    for spec in lookup_specs {
        let next_id = needed.len();
        needed.entry(spec.column_index).or_insert(next_id);
    }

    let mut columns: Vec<Vec<PiopField>> = Vec::with_capacity(needed.len());
    let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());

    // Build the reverse map: remapped_index → original trace index.
    let mut reverse_index_map: Vec<usize> = vec![0; needed.len()];

    for (&orig_idx, &remapped_idx) in &needed {
        // Wrap Inner values into PiopField.
        let col_f: Vec<PiopField> = field_trace[orig_idx]
            .iter()
            .map(|inner| PiopField::new_unchecked_with_cfg(inner.clone(), field_cfg))
            .collect();

        // Compute raw indices from BinaryPoly (pure bit manipulation, no field ops).
        let col_idx: Vec<usize> = bp_trace[orig_idx]
            .iter()
            .map(|bp| {
                let mut idx = 0usize;
                for (j, coeff) in bp.iter().enumerate() {
                    if coeff.into_inner() {
                        idx |= 1usize << j;
                    }
                }
                idx
            })
            .collect();

        // BTreeMap iterates in key order, so we must place at remapped_idx.
        if remapped_idx >= columns.len() {
            columns.resize_with(remapped_idx + 1, Vec::new);
            raw_indices.resize_with(remapped_idx + 1, Vec::new);
        }
        columns[remapped_idx] = col_f;
        raw_indices[remapped_idx] = col_idx;
        reverse_index_map[remapped_idx] = orig_idx;
    }

    // Remap lookup specs to 0-based indices into the projected columns.
    let index_map: BTreeMap<usize, usize> = needed
        .keys()
        .enumerate()
        .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
        .collect();

    let remapped_specs: Vec<LookupColumnSpec> = lookup_specs
        .iter()
        .map(|spec| LookupColumnSpec {
            column_index: index_map[&spec.column_index],
            table_type: spec.table_type.clone(),
        })
        .collect();

    (columns, raw_indices, remapped_specs, reverse_index_map)
}

/// Build projected field values and raw integer indices for affine virtual
/// columns, and append them to the existing lookup column arrays.
///
/// Returns a `witness_source_map` that maps each remapped column index to
/// the corresponding [`LookupWitnessSource`].
fn append_affine_virtual_columns<const D: usize>(
    bp_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    affine_specs: &[AffineLookupSpec],
    projecting_element: &PiopField,
    _field_cfg: &MontyParams<FIELD_LIMBS>,
    num_vars: usize,
    // in / out:
    columns: &mut Vec<Vec<PiopField>>,
    raw_indices: &mut Vec<Vec<usize>>,
    remapped_specs: &mut Vec<LookupColumnSpec>,
    reverse_index_map: &[usize],
) -> Vec<LookupWitnessSource>
where
    BinaryPoly<D>: ProjectableToField<PiopField> + From<u32>,
{
    let num_rows = 1usize << num_vars;
    let projection = BinaryPoly::<D>::prepare_projection(projecting_element);

    // Build the witness source map for all existing (column) entries.
    let mut ws_map: Vec<LookupWitnessSource> = reverse_index_map
        .iter()
        .map(|&orig| LookupWitnessSource::Column { column_index: orig })
        .collect();

    for spec in affine_specs {
        let mut col_f = Vec::with_capacity(num_rows);
        let mut col_idx = Vec::with_capacity(num_rows);

        for t in 0..num_rows {
            let mut val: i64 = spec.constant_offset_bits as i64;
            for &(col_index, coeff) in &spec.terms {
                let bp = &bp_trace[col_index].evaluations[t];
                let mut bp_val: i64 = 0;
                for (j, c) in bp.iter().enumerate() {
                    if c.into_inner() {
                        bp_val |= 1i64 << j;
                    }
                }
                val += coeff * bp_val;
            }
            let val_u32 = val as u32;
            col_f.push(projection(&BinaryPoly::<D>::from(val_u32)));
            col_idx.push(val_u32 as usize);
        }

        let remapped_idx = columns.len();
        columns.push(col_f);
        raw_indices.push(col_idx);

        remapped_specs.push(LookupColumnSpec {
            column_index: remapped_idx,
            table_type: spec.table_type.clone(),
        });

        ws_map.push(LookupWitnessSource::Affine {
            terms: spec.terms.clone(),
            constant_offset_bits: spec.constant_offset_bits,
        });
    }

    ws_map
}

/// Compute the parent evaluation for an affine witness source.
///
/// `parent_eval = eq_sum_w · π(constant_offset) + Σ coeff · up_evals[col_idx]`.
///
/// The carry-free identity `π(v[t]) = π(offset) + Σ coeff · π(col[t])` holds
/// at every witness row.  When the lookup protocol's `num_vars` exceeds the
/// witness's `num_vars` (because the subtable is larger), the MLE of the
/// constant offset function over the witness domain evaluates to
/// `π(offset) · eq_sum_w` rather than `π(offset)`, where `eq_sum_w` =
/// `Σ_{j < witness_len} eq(j, x*)`.  Pass `eq_sum_w` to account for this.
pub fn eval_affine_parent<const D: usize>(
    terms: &[(usize, i64)],
    constant_offset_bits: u32,
    up_evals: &[PiopField],
    projecting_element: &PiopField,
    eq_sum_w: &PiopField,
    field_cfg: &MontyParams<FIELD_LIMBS>,
) -> PiopField
where
    BinaryPoly<D>: ProjectableToField<PiopField> + From<u32>,
{
    let projection = BinaryPoly::<D>::prepare_projection(projecting_element);
    let mut result = eq_sum_w.clone() * projection(&BinaryPoly::<D>::from(constant_offset_bits));

    let one = PiopField::one_with_cfg(field_cfg);
    for &(col_idx, coeff) in terms {
        // Build the field-element coefficient from the i64 value.
        let abs = coeff.unsigned_abs();
        let mut c = PiopField::zero_with_cfg(field_cfg);
        for _ in 0..abs {
            c += &one;
        }
        if coeff < 0 {
            c = PiopField::zero_with_cfg(field_cfg) - c;
        }
        result += c * up_evals[col_idx].clone();
    }
    result
}

// ─── Prover ─────────────────────────────────────────────────────────────────

/// Run the full Zinc+ prover pipeline for a UAIR over `BinaryPoly<D>`.
///
/// Returns a proof and timing breakdown.
///
/// # Type parameters
/// - `U`: the UAIR describing the constraint system.
/// - `Zt`: Zip+ type configuration.
/// - `Lc`: linear code for the PCS.
/// - `D`: `DEGREE_PLUS_ONE` for BinaryPoly.
/// - `CHECK`: overflow checking flag for the PCS.
///
/// # Arguments
/// - `params`: PCS parameters (poly size, num rows, code).
/// - `trace`: the witness trace as MLEs.
/// - `num_vars`: log₂(trace length).
/// - `lookup_specs`: column lookup specifications for batched decomposed
///   LogUp. Pass `&[]` to skip the lookup step.
#[allow(clippy::type_complexity)]
pub fn prove<U, Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_vars: usize,
    lookup_specs: &[LookupColumnSpec],
) -> ZincProof
where
    U: Uair<Scalar = BinaryPoly<D>>,
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D>>,
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
{
    let total_start = Instant::now();
    let sig = U::signature();
    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    // Only commit columns that are neither public (known to verifier)
    // nor shift sources (whose claims are resolved by shift sumcheck).
    let t0 = Instant::now();
    let pcs_trace = private_trace(trace, &sig.pcs_excluded_columns());
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, &pcs_trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: PIOP — Ideal Check ──────────────────────────────────
    let t1 = Instant::now();
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    let projected_scalars = project_scalars::<PiopField, U>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });

    let (ic_proof, ic_state) = if max_degree == 1 {
        // MLE-first path: evaluate column MLEs at the random point
        // directly from native BinaryPoly trace, avoiding full
        // trace projection to DynamicPolynomialF.
        IdealCheckProtocol::<PiopField>::prove_mle_first::<U, D>(
            &mut transcript,
            trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Ideal check prover failed")
    } else {
        // Generic path for non-linear constraints.
        let projected_trace = project_trace_coeffs::<PiopField, i64, i64, D>(
            trace, &[], &[], &field_cfg,
        );
        IdealCheckProtocol::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Ideal check prover failed")
    };
    let ideal_check_time = t1.elapsed();

    // ── Step 3: PIOP — Combined Poly Resolver ───────────────────────
    let t2 = Instant::now();
    // Get projecting element for F[X]→F projection (shared by CPR and lookup).
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("scalar projection failed");
    let field_trace = project_trace_to_field::<PiopField, D>(
        trace, &[], &[], &projecting_element,
    );

    // Extract lookup columns from the projected field trace before CPR
    // consumes it. This reuses the same projecting_element, avoiding a
    // redundant ~2ms re-projection of the BinaryPoly columns.
    let lookup_precomputed = if !lookup_specs.is_empty() {
        let (columns, raw_indices, remapped_specs, reverse_index_map) =
            extract_lookup_columns_from_field_trace(trace, &field_trace, lookup_specs, &field_cfg);
        Some((columns, raw_indices, remapped_specs, reverse_index_map))
    } else {
        None
    };

    // Extract source columns for shift sumcheck before CPR consumes field_trace.
    let shift_trace_columns: Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>> =
        sig.shifts.iter().map(|spec| field_trace[spec.source_col].clone()).collect();

    let (cpr_proof, cpr_state) =
        CombinedPolyResolver::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            field_trace,
            &ic_state.evaluation_point,
            &field_projected_scalars,
            num_constraints,
            num_vars,
            max_degree,
            &field_cfg,
        )
        .expect("Combined poly resolver failed");
    let cpr_time = t2.elapsed();

    // ── Step 3a: Shift Sumcheck (reduce shifted-column claims) ──────
    let shift_sumcheck_output = if !sig.shifts.is_empty() {
        let claims: Vec<ShiftClaim<PiopField>> = sig.shifts
            .iter()
            .enumerate()
            .map(|(i, spec)| ShiftClaim {
                source_col: i,
                shift_amount: spec.shift_amount,
                eval_point: cpr_state.evaluation_point.clone(),
                claimed_eval: cpr_proof.down_evals[i].clone(),
            })
            .collect();
        Some(shift_sumcheck_prove(
            &mut transcript,
            &claims,
            &shift_trace_columns,
            num_vars,
            &field_cfg,
        ))
    } else {
        None
    };

    // ── Step 3b: PIOP — Batched Decomposed LogUp (column typing) ────
    let t2b = Instant::now();
    let lookup_proof = if let Some((columns, raw_indices, remapped_specs, _reverse_index_map)) = lookup_precomputed {
        let (lk_proof, _lk_state) = prove_batched_lookup_with_indices(
            &mut transcript,
            &columns,
            &raw_indices,
            &remapped_specs,
            &projecting_element,
            &field_cfg,
        )
        .expect("Batched lookup prover failed");

        Some(LookupProofData::Classic(lk_proof))
    } else {
        None
    };
    let lookup_time = t2b.elapsed();

    // ── Step 4: PCS Prove (test + evaluate) ────────────────────────
    // r_PCS = r_CPR: pass the CPR evaluation point directly to the PCS.
    let t3 = Instant::now();
    let (eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PiopField, CHECK>(
            params,
            &pcs_trace,
            &cpr_state.evaluation_point,
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t3.elapsed();

    let total_time = total_start.elapsed();

    // Serialize proofs.
    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    // Serialize PIOP proof data.
    let ic_proof_values: Vec<Vec<u8>> = ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    let cpr_sumcheck_messages: Vec<Vec<u8>> = cpr_proof
        .sumcheck_proof
        .messages
        .iter()
        .map(|msg| msg.0.tail_evaluations.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let cpr_sumcheck_claimed_sum = field_to_bytes(&cpr_proof.sumcheck_proof.claimed_sum);
    // Only serialize non-public column evaluations — the verifier
    // computes public column MLE evaluations from the known public data.
    let cpr_up_evals: Vec<Vec<u8>> = cpr_proof.up_evals.iter()
        .enumerate()
        .filter(|(i, _)| !sig.is_public_column(*i))
        .map(|(_, v)| field_to_bytes(v))
        .collect();
    let cpr_down_evals: Vec<Vec<u8>> = cpr_proof.down_evals.iter().map(field_to_bytes).collect();
    let evaluation_point_bytes: Vec<Vec<u8>> = cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    let pcs_evals_bytes: Vec<Vec<u8>> = vec![field_to_bytes(&eval_f)];

    // Serialize shift sumcheck proof (if present).
    let shift_sumcheck = shift_sumcheck_output.map(|output| {
        SerializedShiftSumcheckProof {
            rounds: output.proof.rounds.iter().map(|rp| {
                rp.evals.iter().flat_map(|e| field_to_bytes(e)).collect()
            }).collect(),
            v_finals: output.v_finals.iter().enumerate()
                .filter(|(i, _)| !sig.is_public_shift(*i))
                .map(|(_, v)| field_to_bytes(v)).collect(),
        }
    });

    ZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        ic_proof_values,
        cpr_sumcheck_messages,
        cpr_sumcheck_claimed_sum,
        cpr_up_evals,
        cpr_down_evals,
        lookup_proof,
        shift_sumcheck,
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ideal_check_time,
            combined_poly_resolver: cpr_time,
            lookup: lookup_time,
            pcs_prove: pcs_prove_time,
            serialize: Duration::ZERO,
            total: total_time,
        },
    }
}

/// Run the full Zinc+ prover pipeline with **classic** batched decomposition
/// LogUp (chunks + inverses + sumcheck), instead of GKR LogUp.
///
/// Unlike [`prove`], this variant **batches the CPR and lookup sumchecks**
/// into a single multi-degree sumcheck via
/// [`MultiDegreeSumcheck::prove_as_subprotocol`]. Group 0 is the CPR
/// (degree `max_degree + 2`), groups 1..N are the lookup table groups
/// (degree 2 each). All groups share verifier challenges and produce
/// a common evaluation point.
#[allow(clippy::type_complexity)]
pub fn prove_classic_logup<U, Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_vars: usize,
    lookup_specs: &[LookupColumnSpec],
    affine_lookup_specs: &[AffineLookupSpec],
) -> ZincProof
where
    U: Uair<Scalar = BinaryPoly<D>>,
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D>>,
    BinaryPoly<D>: From<u32>,
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
{
    let total_start = Instant::now();
    let sig = U::signature();
    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    let t0 = Instant::now();
    let pcs_trace = private_trace(trace, &sig.pcs_excluded_columns());
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, &pcs_trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    let t1 = Instant::now();
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    let projected_scalars = project_scalars::<PiopField, U>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });

    let (ic_proof, ic_state) = if max_degree == 1 {
        IdealCheckProtocol::<PiopField>::prove_mle_first::<U, D>(
            &mut transcript,
            trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Ideal check prover failed")
    } else {
        let projected_trace = project_trace_coeffs::<PiopField, i64, i64, D>(
            trace, &[], &[], &field_cfg,
        );
        IdealCheckProtocol::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Ideal check prover failed")
    };
    let ideal_check_time = t1.elapsed();

    // ── Step 3: Batched CPR + Lookup via multi-degree sumcheck ──────
    let t2 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("scalar projection failed");
    let field_trace = project_trace_to_field::<PiopField, D>(
        trace, &[], &[], &projecting_element,
    );

    // Extract lookup columns before CPR consumes the field trace.
    let has_lookups = !lookup_specs.is_empty() || !affine_lookup_specs.is_empty();
    let lookup_precomputed = if has_lookups {
        let (mut columns, mut raw_indices, mut remapped_specs, reverse_index_map) =
            if !lookup_specs.is_empty() {
                extract_lookup_columns_from_field_trace(trace, &field_trace, lookup_specs, &field_cfg)
            } else {
                (vec![], vec![], vec![], vec![])
            };
        let ws_map = append_affine_virtual_columns::<D>(
            trace, affine_lookup_specs, &projecting_element, &field_cfg, num_vars,
            &mut columns, &mut raw_indices, &mut remapped_specs, &reverse_index_map,
        );
        Some((columns, raw_indices, remapped_specs, ws_map))
    } else {
        None
    };

    // ── CPR: build sumcheck group (does NOT run sumcheck) ───────────
    let mut cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U>(
        &mut transcript,
        field_trace,
        &ic_state.evaluation_point,
        &field_projected_scalars,
        num_constraints,
        num_vars,
        max_degree,
        &field_cfg,
    )
    .expect("CPR build_prover_group failed");
    let cpr_num_cols = cpr_group.num_cols;

    // ── Lookup: build sumcheck groups (does NOT run sumchecks) ───────
    let mut lookup_groups_data: Vec<(LookupSumcheckGroup<PiopField>, LookupGroupMeta)> = Vec::new();
    if let Some((ref columns, ref raw_indices, ref remapped_specs, ref ws_map)) = lookup_precomputed {
        let groups = group_lookup_specs(remapped_specs);
        for group in &groups {
            let instance = build_lookup_instance_from_indices_pub(
                columns, raw_indices, group, &projecting_element, &field_cfg,
            )
            .expect("lookup instance build failed");

            let witness_len = instance.witnesses[0].len();

            let lk_group = BatchedDecompLogupProtocol::<PiopField>::build_prover_group(
                &mut transcript,
                &instance,
                &field_cfg,
            )
            .expect("lookup build_prover_group failed");

            let meta = LookupGroupMeta {
                table_type: group.table_type.clone(),
                num_columns: group.column_indices.len(),
                witness_len,
                witness_sources: group.column_indices.iter()
                    .map(|&remapped| ws_map[remapped].clone())
                    .collect(),
            };
            lookup_groups_data.push((lk_group, meta));
        }
    }

    // Determine the shared num_vars across CPR and all lookup groups.
    // If the lookup subtable is larger than the witness, its num_vars
    // may exceed CPR's. We zero-pad the CPR's MLEs in that case.
    let shared_num_vars = {
        let max_lk = lookup_groups_data
            .iter()
            .map(|(g, _)| g.num_vars)
            .max()
            .unwrap_or(num_vars);
        num_vars.max(max_lk)
    };

    // Pad CPR MLEs to shared_num_vars if needed.
    if shared_num_vars > num_vars {
        let target_len = 1usize << shared_num_vars;
        for mle in &mut cpr_group.mles {
            mle.evaluations
                .resize(target_len, Default::default());
            mle.num_vars = shared_num_vars;
        }
    }

    // ── Assemble all groups and run multi-degree sumcheck ────────────
    let has_lookup = !lookup_groups_data.is_empty();
    let (batched_proof, evaluation_point, cpr_up_evals, cpr_down_evals, _lookup_proof) =
    if has_lookup {
        // Build the groups vector: CPR first, then lookup groups.
        let mut sumcheck_groups: Vec<(
            usize,
            Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>>,
            Box<dyn Fn(&[PiopField]) -> PiopField + Send + Sync>,
        )> = Vec::with_capacity(1 + lookup_groups_data.len());

        sumcheck_groups.push((cpr_group.degree, cpr_group.mles, cpr_group.comb_fn));

        let mut lookup_pre_data: Vec<(LookupSumcheckGroup<PiopField>, LookupGroupMeta)> = Vec::new();
        for (lk_group, meta) in lookup_groups_data {
            sumcheck_groups.push((lk_group.degree, lk_group.mles, lk_group.comb_fn));
            // We still need the ancillary data; reconstruct a shell.
            lookup_pre_data.push((LookupSumcheckGroup {
                degree: 0, // unused after this point
                mles: vec![], // already moved
                comb_fn: Box::new(|_: &[PiopField]| unreachable!()),
                num_vars: 0,
                aggregated_multiplicities: lk_group.aggregated_multiplicities,
                chunk_inverse_witnesses: lk_group.chunk_inverse_witnesses,
                inverse_table: lk_group.inverse_table,
            }, meta));
        }

        // Run the combined multi-degree sumcheck.
        let (md_proof, mut prover_states) =
            MultiDegreeSumcheck::<PiopField>::prove_as_subprotocol(
                &mut transcript,
                sumcheck_groups,
                shared_num_vars,
                &field_cfg,
            );

        // ── CPR finalize: extract up/down evals from group 0's state ──
        let cpr_prover_state = prover_states.remove(0);
        let (up_evals, down_evals, cpr_state_final) =
            CombinedPolyResolver::<PiopField>::finalize_prover(
                &mut transcript,
                cpr_prover_state,
                cpr_num_cols,
                &field_cfg,
            )
            .expect("CPR finalize_prover failed");

        // ── Lookup finalize: build per-group proofs from remaining states ──
        let mut lookup_group_proofs = Vec::new();
        let mut lookup_group_meta = Vec::new();
        for (i, (lk_pre, meta)) in lookup_pre_data.into_iter().enumerate() {
            // Extract this lookup group's sumcheck proof from the md_proof.
            let lk_sumcheck_proof = SumcheckProof {
                messages: md_proof.group_messages[i + 1]
                    .iter()
                    .cloned()
                    .collect(),
                claimed_sum: md_proof.claimed_sums[i + 1].clone(),
            };
            let lk_eval_point = prover_states.remove(0).randomness;
            let (lk_proof, _lk_state) =
                BatchedDecompLogupProtocol::<PiopField>::finalize_prover(
                    lk_pre, lk_sumcheck_proof, lk_eval_point,
                );
            lookup_group_proofs.push(lk_proof);
            lookup_group_meta.push(meta);
        }

        let batched = BatchedCprLookupProof {
            md_proof,
            // Only include non-public column evaluations.
            cpr_up_evals: {
                up_evals.iter()
                    .enumerate()
                    .filter(|(i, _)| !sig.is_public_column(*i))
                    .map(|(_, v)| v.clone())
                    .collect()
            },
            cpr_down_evals: down_evals.clone(),
            lookup_group_meta,
            lookup_group_proofs,
        };

        (
            Some(LookupProofData::BatchedClassic(batched)),
            cpr_state_final.evaluation_point,
            up_evals,
            down_evals,
            true,
        )
    } else {
        // No lookup — run CPR sumcheck alone (falls back to normal path).
        let groups = vec![(cpr_group.degree, cpr_group.mles, cpr_group.comb_fn)];
        let (_md_proof, mut prover_states) =
            MultiDegreeSumcheck::<PiopField>::prove_as_subprotocol(
                &mut transcript,
                groups,
                num_vars,
                &field_cfg,
            );

        let cpr_prover_state = prover_states.remove(0);
        let (up_evals, down_evals, cpr_state_final) =
            CombinedPolyResolver::<PiopField>::finalize_prover(
                &mut transcript,
                cpr_prover_state,
                cpr_num_cols,
                &field_cfg,
            )
            .expect("CPR finalize_prover failed");

        (None, cpr_state_final.evaluation_point, up_evals, down_evals, false)
    };

    let cpr_lookup_time = t2.elapsed();

    // ── Step 4: PCS Prove (test + evaluate) ────────────────────────
    // r_PCS = r_CPR: pass the CPR evaluation point directly to the PCS.
    // When lookup is present, the sumcheck may have shared_num_vars > num_vars;
    // truncate to num_vars since PCS polynomials only have num_vars variables.
    let t3 = Instant::now();
    let (eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PiopField, CHECK>(
            params,
            &pcs_trace,
            &evaluation_point[..num_vars],
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t3.elapsed();
    let total_time = total_start.elapsed();

    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };
    let ic_proof_values: Vec<Vec<u8>> = ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    // For the batched path, CPR sumcheck data is embedded in the
    // BatchedClassic variant; these fields are unused but populated
    // for the no-lookup fallback case.
    let cpr_sumcheck_messages = Vec::new();
    let cpr_sumcheck_claimed_sum = vec![0u8; <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES];
    // Only serialize non-public column evaluations — the verifier
    // computes public column MLE evaluations from the known public data.
    let cpr_up_evals_bytes: Vec<Vec<u8>> = cpr_up_evals.iter()
        .enumerate()
        .filter(|(i, _)| !sig.is_public_column(*i))
        .map(|(_, v)| field_to_bytes(v))
        .collect();
    let cpr_down_evals_bytes: Vec<Vec<u8>> = cpr_down_evals.iter().map(field_to_bytes).collect();
    let evaluation_point_bytes: Vec<Vec<u8>> = evaluation_point.iter().map(field_to_bytes).collect();
    let pcs_evals_bytes: Vec<Vec<u8>> = vec![field_to_bytes(&eval_f)];

    ZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        ic_proof_values,
        cpr_sumcheck_messages,
        cpr_sumcheck_claimed_sum,
        cpr_up_evals: cpr_up_evals_bytes,
        cpr_down_evals: cpr_down_evals_bytes,
        lookup_proof: batched_proof,
        shift_sumcheck: None, // classic logup batches CPR+lookup; shift sumcheck not yet integrated
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ideal_check_time,
            combined_poly_resolver: cpr_lookup_time,
            lookup: Duration::ZERO,  // included in combined_poly_resolver
            pcs_prove: pcs_prove_time,
            serialize: Duration::ZERO,
            total: total_time,
        },
    }
}

/// Run the Zinc+ prover pipeline for any ring `R` (generic single-ring variant).
///
/// Unlike `prove()` (which is hardcoded to `BinaryPoly<D>`), this function
/// works with any evaluation ring — `Int<4>`, `DensePolynomial<i64, D>`, etc.
/// The ring must implement `ProjectableToField<PiopField>` and
/// `PiopField: FromWithConfig<R>` (for projecting elements to the PIOP field).
///
/// This is the target pipeline for ECDSA with `Int<4>` throughout:
/// the same `Int<4>` type is used as PCS evaluation type AND PIOP constraint ring.
#[allow(clippy::type_complexity)]
pub fn prove_generic<U, R, Zt, Lc, PcsF, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<R>],
    num_vars: usize,
    lookup_specs: &[LookupColumnSpec],
) -> ZincProof
where
    R: ProjectableToField<PiopField> + crypto_primitives::Semiring + std::fmt::Debug + Clone + Send + Sync + 'static,
    U: Uair<Scalar = R>,
    Zt: ZipTypes<Eval = R>,
    Lc: LinearCode<Zt>,
    PiopField: FromPrimitiveWithConfig + FromWithConfig<R>,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    // PCS field (may be wider than PiopField when Zt::Fmod > 128 bits)
    PcsF: PrimeField
        + FromPrimitiveWithConfig
        + FromWithConfig<PiopUint>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a PcsF>
        + FromRef<PcsF>,
    PcsF::Inner: FromRef<Zt::Fmod> + ConstTranscribable,
    Zt::Eval: ProjectableToField<PcsF>,
{
    let total_start = Instant::now();
    let sig = U::signature();
    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let pcs_trace = private_trace(trace, &sig.pcs_excluded_columns());
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, &pcs_trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: PIOP — Ideal Check ──────────────────────────────────
    let t1 = Instant::now();
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    // Project trace: each R element becomes a constant DynamicPolynomialF.
    let projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<PiopField>>> = trace
        .iter()
        .map(|col_mle| {
            col_mle
                .iter()
                .map(|elem| DynamicPolynomialF {
                    coeffs: vec![PiopField::from_with_cfg(elem.clone(), &field_cfg)],
                })
                .collect()
        })
        .collect();

    let projected_scalars = project_scalars::<PiopField, U>(|scalar| {
        DynamicPolynomialF {
            coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &field_cfg)],
        }
    });

    let (ic_proof, ic_state) =
        IdealCheckProtocol::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Ideal check prover failed");
    let ideal_check_time = t1.elapsed();

    // ── Step 3: PIOP — Combined Poly Resolver ───────────────────────
    let t2 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("scalar projection failed");
    // For scalar types (Int<N>), evaluating a constant poly at any point
    // yields the constant. We can use project_trace_to_field with int slot.
    let field_trace = project_trace_to_field::<PiopField, 1>(
        &[], &[], &projected_trace, &projecting_element,
    );

    // Extract source columns needed for shift sumcheck before CPR consumes
    // field_trace. Only done when the UAIR declares explicit shift specs.
    let shift_trace_columns: Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>> =
        sig.shifts.iter().map(|spec| field_trace[spec.source_col].clone()).collect();

    let (cpr_proof, cpr_state) =
        CombinedPolyResolver::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            field_trace,
            &ic_state.evaluation_point,
            &field_projected_scalars,
            num_constraints,
            num_vars,
            max_degree,
            &field_cfg,
        )
        .expect("Combined poly resolver failed");
    let cpr_time = t2.elapsed();

    // ── Step 3a: Shift Sumcheck (reduce shifted-column claims) ──────
    let shift_sumcheck_output = if !sig.shifts.is_empty() {
        let claims: Vec<ShiftClaim<PiopField>> = sig.shifts
            .iter()
            .enumerate()
            .map(|(i, spec)| ShiftClaim {
                source_col: i,
                shift_amount: spec.shift_amount,
                eval_point: cpr_state.evaluation_point.clone(),
                claimed_eval: cpr_proof.down_evals[i].clone(),
            })
            .collect();
        Some(shift_sumcheck_prove(
            &mut transcript,
            &claims,
            &shift_trace_columns,
            num_vars,
            &field_cfg,
        ))
    } else {
        None
    };

    // ── Step 3b: PIOP — Batched Decomposed LogUp (column typing) ────
    let t2b = Instant::now();
    let lookup_proof = if !lookup_specs.is_empty() {
        // Derive a lookup-specific projecting element from the transcript.
        let lookup_projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

        // Collect unique column indices referenced by lookup specs.
        let mut needed: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for spec in lookup_specs {
            let next_id = needed.len();
            needed.entry(spec.column_index).or_insert(next_id);
        }

        // Project only the columns that have lookup constraints.
        let projection_fn = R::prepare_projection(&lookup_projecting_element);
        let mut columns: Vec<Vec<PiopField>> = Vec::with_capacity(needed.len());
        let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());
        for &orig_idx in needed.keys() {
            let col_mle = &trace[orig_idx];
            let mut col_f = Vec::with_capacity(col_mle.len());
            let mut col_idx = Vec::with_capacity(col_mle.len());
            for v in col_mle.iter() {
                let fv = projection_fn(v);
                let uint = fv.retrieve();
                let idx = uint.as_words()[0] as usize;
                col_f.push(fv);
                col_idx.push(idx);
            }
            columns.push(col_f);
            raw_indices.push(col_idx);
        }

        // Remap lookup specs to 0-based indices into the projected columns.
        let index_map: std::collections::BTreeMap<usize, usize> = needed
            .keys()
            .enumerate()
            .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
            .collect();
        let remapped_specs: Vec<LookupColumnSpec> = lookup_specs
            .iter()
            .map(|spec| LookupColumnSpec {
                column_index: index_map[&spec.column_index],
                table_type: spec.table_type.clone(),
            })
            .collect();

        let (lk_proof, _lk_state) = prove_batched_lookup_with_indices(
            &mut transcript,
            &columns,
            &raw_indices,
            &remapped_specs,
            &lookup_projecting_element,
            &field_cfg,
        )
        .expect("Batched lookup prover failed");

        Some(LookupProofData::Classic(lk_proof))
    } else {
        None
    };
    let lookup_time = t2b.elapsed();

    // ── Step 4: PCS Prove (test + evaluate) ────────────────────────
    // r_PCS = r_CPR: convert the CPR evaluation point to PcsF.
    let t3 = Instant::now();
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PcsF, Zt::Fmod, Zt::PrimeTest>();
    let point: Vec<PcsF> = piop_point_to_pcs_field(&cpr_state.evaluation_point, &pcs_field_cfg);
    let (_eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PcsF, CHECK>(
            params,
            &pcs_trace,
            &point,
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t3.elapsed();

    let total_time = total_start.elapsed();

    // Serialize proofs.
    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    let ic_proof_values: Vec<Vec<u8>> = ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    let cpr_sumcheck_messages: Vec<Vec<u8>> = cpr_proof
        .sumcheck_proof
        .messages
        .iter()
        .map(|msg| msg.0.tail_evaluations.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let cpr_sumcheck_claimed_sum = field_to_bytes(&cpr_proof.sumcheck_proof.claimed_sum);
    // Only serialize non-public column evaluations — the verifier
    // computes public column MLE evaluations from the known public data.
    let cpr_up_evals: Vec<Vec<u8>> = cpr_proof.up_evals.iter()
        .enumerate()
        .filter(|(i, _)| !sig.is_public_column(*i))
        .map(|(_, v)| field_to_bytes(v))
        .collect();
    let cpr_down_evals: Vec<Vec<u8>> = cpr_proof.down_evals.iter().map(field_to_bytes).collect();
    let evaluation_point_bytes: Vec<Vec<u8>> = cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    // PCS evals are in PcsF (potentially wider than PiopField); serialize
    // using ConstTranscribable for the verifier to deserialize.
    let pcs_evals_bytes: Vec<Vec<u8>> = {
        let mut buf = vec![0u8; <PcsF::Inner as ConstTranscribable>::NUM_BYTES];
        _eval_f.inner().write_transcription_bytes(&mut buf);
        vec![buf]
    };

    // Serialize shift sumcheck proof (if present).
    // Skip v_finals entries whose source column is public — the
    // verifier will recompute those MLE evaluations itself.
    let shift_sumcheck = shift_sumcheck_output.map(|output| {
        SerializedShiftSumcheckProof {
            rounds: output.proof.rounds.iter().map(|rp| {
                rp.evals.iter().flat_map(|e| field_to_bytes(e)).collect()
            }).collect(),
            v_finals: output.v_finals.iter()
                .enumerate()
                .filter(|(i, _)| !sig.is_public_shift(*i))
                .map(|(_, v)| field_to_bytes(v))
                .collect(),
        }
    });

    ZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        ic_proof_values,
        cpr_sumcheck_messages,
        cpr_sumcheck_claimed_sum,
        cpr_up_evals,
        cpr_down_evals,
        lookup_proof,
        shift_sumcheck,
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ideal_check_time,
            combined_poly_resolver: cpr_time,
            lookup: lookup_time,
            pcs_prove: pcs_prove_time,
            serialize: Duration::ZERO,
            total: total_time,
        },
    }
}

/// Run the Zinc+ verifier for any ring `R` (generic single-ring variant).
///
/// Corresponds to `prove_generic`. Verifies IC + CPR + PCS.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn verify_generic<U, R, Zt, Lc, PcsF, const CHECK: bool, IdealOverF, IdealOverFFromRef>(
    params: &ZipPlusParams<Zt, Lc>,
    zinc_proof: &ZincProof,
    num_vars: usize,
    ideal_over_f_from_ref: IdealOverFFromRef,
    public_column_data: &[DenseMultilinearExtension<R>],
) -> VerifyResult
where
    R: ProjectableToField<PiopField> + crypto_primitives::Semiring + std::fmt::Debug + Clone + Send + Sync + 'static,
    U: Uair<Scalar = R>,
    Zt: ZipTypes<Eval = R>,
    Lc: LinearCode<Zt>,
    PiopField: FromPrimitiveWithConfig + FromWithConfig<R>,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    // PCS field (may be wider than PiopField when Zt::Fmod > 128 bits)
    PcsF: PrimeField
        + FromPrimitiveWithConfig
        + FromWithConfig<PiopUint>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a PcsF>
        + FromRef<PcsF>,
    PcsF::Inner: FromRef<Zt::Fmod> + ConstTranscribable,
    Zt::Eval: ProjectableToField<PcsF>,
    Zt::Cw: ProjectableToField<PcsF>,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
{
    let total_start = Instant::now();

    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Reconstruct Fiat-Shamir transcript ──────────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    let field_elem_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Step 1: IC verify ───────────────────────────────────────────
    let t0 = Instant::now();

    let ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>> = zinc_proof
        .ic_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: ic_combined_mle_values,
    };

    let ic_subclaim = match IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<U, _, _>(
        &mut transcript,
        ic_proof,
        num_constraints,
        num_vars,
        ideal_over_f_from_ref,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: t0.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };
    let ic_verify_time = t0.elapsed();

    // ── Step 2: CPR verify ──────────────────────────────────────────
    let t1 = Instant::now();

    let cpr_sumcheck_messages: Vec<ProverMsg<PiopField>> = zinc_proof
        .cpr_sumcheck_messages
        .iter()
        .map(|bytes| {
            let num_evals = bytes.len() / field_elem_size;
            let tail_evaluations: Vec<PiopField> = (0..num_evals)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
        })
        .collect();

    let cpr_claimed_sum = field_from_bytes(&zinc_proof.cpr_sumcheck_claimed_sum, &field_cfg);
    let cpr_up_evals: Vec<PiopField> = zinc_proof.cpr_up_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let cpr_down_evals: Vec<PiopField> = zinc_proof.cpr_down_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let cpr_sumcheck_proof = SumcheckProof {
        messages: cpr_sumcheck_messages,
        claimed_sum: cpr_claimed_sum.clone(),
    };

    // Compute projecting element and projected scalars for CPR verify.
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let projected_scalars_coeffs = project_scalars::<PiopField, U>(|scalar| {
        DynamicPolynomialF {
            coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &field_cfg)],
        }
    });
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars_coeffs, &projecting_element)
            .expect("scalar projection failed");

    let sig = U::signature();

    // Split CPR verification: pre-sumcheck → sumcheck → reconstruct
    // public evals → finalize.  We need the sumcheck point before we
    // can evaluate the public column MLEs.
    let cpr_pre = match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U>(
        &mut transcript,
        &cpr_claimed_sum,
        num_constraints,
        &projecting_element,
        &field_projected_scalars,
        &ic_subclaim,
        &field_cfg,
    ) {
        Ok(pre) => pre,
        Err(e) => {
            eprintln!("CPR pre-sumcheck failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    let sumcheck_subclaim = match MLSumcheck::<PiopField>::verify_as_subprotocol(
        &mut transcript,
        num_vars,
        max_degree + 2,
        &cpr_sumcheck_proof,
        &field_cfg,
    ) {
        Ok(sc) => sc,
        Err(e) => {
            eprintln!("CPR sumcheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // Reconstruct full up_evals: insert public column evaluations
    // computed from known public data at the sumcheck challenge point.
    let full_up_evals = if sig.public_columns.is_empty() {
        cpr_up_evals
    } else {
        let public_evals: Vec<PiopField> = {
            #[cfg(feature = "parallel")]
            let iter = public_column_data.par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = public_column_data.iter();
            iter.map(|col| {
                let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                    col.iter()
                        .map(|val| PiopField::from_with_cfg(val.clone(), &field_cfg).into_inner())
                        .collect();
                mle.evaluate_with_config(&sumcheck_subclaim.point, &field_cfg)
                    .expect("public column MLE evaluation should succeed")
            })
            .collect()
        };
        reconstruct_up_evals(
            &cpr_up_evals,
            &public_evals,
            &sig.public_columns,
            sig.total_cols(),
        )
    };

    let cpr_subclaim = match CombinedPolyResolver::<PiopField>::finalize_verifier::<U>(
        &mut transcript,
        sumcheck_subclaim.point,
        sumcheck_subclaim.expected_evaluation,
        &cpr_pre,
        full_up_evals,
        cpr_down_evals,
        num_vars,
        &field_projected_scalars,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("CPR verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };
    let cpr_verify_time = t1.elapsed();

    // ── Step 2a: Shift Sumcheck verify ──────────────────────────────
    if let Some(ref ss_proof_data) = zinc_proof.shift_sumcheck {
        assert!(!sig.shifts.is_empty(), "shift_sumcheck present but UAIR has no shifts");

        // Reconstruct claims from signature + CPR down_evals + evaluation point.
        let ss_down_evals: Vec<PiopField> = zinc_proof.cpr_down_evals.iter()
            .map(|b| field_from_bytes(b, &field_cfg))
            .collect();
        let claims: Vec<ShiftClaim<PiopField>> = sig.shifts
            .iter()
            .enumerate()
            .map(|(i, spec)| ShiftClaim {
                source_col: i,
                shift_amount: spec.shift_amount,
                eval_point: cpr_subclaim.evaluation_point.clone(),
                claimed_eval: ss_down_evals[i].clone(),
            })
            .collect();

        // Deserialize proof rounds.
        let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_proof_data.rounds.iter().map(|bytes| {
            let n = bytes.len() / field_elem_size;
            assert_eq!(n, 3, "each shift round poly should have 3 evaluations");
            ShiftRoundPoly {
                evals: [
                    field_from_bytes(&bytes[0..field_elem_size], &field_cfg),
                    field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], &field_cfg),
                    field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], &field_cfg),
                ],
            }
        }).collect();
        let ss_proof = ShiftSumcheckProof { rounds };

        // Deserialize v_finals.
        let v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
            .map(|b| field_from_bytes(b, &field_cfg))
            .collect();

        if let Err(e) = shift_sumcheck_verify(
            &mut transcript,
            &ss_proof,
            &claims,
            &v_finals,
            num_vars,
            &field_cfg,
        ) {
            eprintln!("Shift sumcheck verification failed: {e}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: cpr_verify_time,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    }

    // ── Step 2b: Verify Batched Decomposed LogUp (column typing) ────
    let t1b = Instant::now();
    if let Some(ref lookup_data) = zinc_proof.lookup_proof {
        let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

        let result = match lookup_data {
            LookupProofData::Gkr(proof) => verify_gkr_batched_lookup(
                &mut transcript, proof, &projecting_element, &field_cfg,
            ).map(|_| ()),
            LookupProofData::Classic(proof) => verify_batched_lookup(
                &mut transcript, proof, &projecting_element, &field_cfg,
            ).map(|_| ()),
            LookupProofData::BatchedClassic(_) => {
                unimplemented!(
                    "BatchedClassic proofs should use `verify` (BinaryPoly), \
                     not `verify_generic`"
                );
            }
        };
        if let Err(e) = result {
            eprintln!("Lookup verification failed (generic): {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: cpr_verify_time,
                    lookup_verify: t1b.elapsed(),
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    }
    let lookup_verify_time = t1b.elapsed();

    // ── Step 3: PCS Verify ──────────────────────────────────────────
    let t2 = Instant::now();

    // Derive PcsF field config (same as what ZipPlus::verify will derive
    // internally from a fresh transcript).
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PcsF, Zt::Fmod, Zt::PrimeTest>();

    // r_PCS = r_CPR: convert the CPR evaluation point to PcsF.
    let point_f: Vec<PcsF> = piop_point_to_pcs_field(&cpr_subclaim.evaluation_point, &pcs_field_cfg);

    // Deserialize PcsF eval.
    let eval_inner = <PcsF::Inner as ConstTranscribable>::read_transcription_bytes(
        &zinc_proof.pcs_evals_bytes[0],
    );
    let eval_f: PcsF = PcsF::new_unchecked_with_cfg(eval_inner, &pcs_field_cfg);

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(zinc_proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: ZipPlusProof = pcs_transcript.into();

    let pcs_result = ZipPlus::<Zt, Lc>::verify::<PcsF, CHECK>(
        params,
        &zinc_proof.commitment,
        &point_f,
        &eval_f,
        &pcs_proof,
    );

    if let Err(ref e) = pcs_result {
        eprintln!("PCS verification failed (generic): {e:?}");
    }

    let pcs_verify_time = t2.elapsed();

    VerifyResult {
        accepted: pcs_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: ic_verify_time,
            combined_poly_resolver_verify: cpr_verify_time,
            lookup_verify: lookup_verify_time,
            pcs_verify: pcs_verify_time,
            total: total_start.elapsed(),
        },
    }
}

/// Run the PCS-only prover pipeline (commit + test + evaluate).
///
/// This benchmarks the dominant cost (the PCS) without the PIOP overhead,
/// matching the paper's Zip+ benchmark.
pub fn prove_pcs_only<Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_vars: usize,
) -> (ZincProof, Vec<PiopField>)
where
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D>>,
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
{
    let total_start = Instant::now();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: PCS Prove (test + evaluate) ────────────────────────
    let t1 = Instant::now();
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PiopField, Zt::Fmod, Zt::PrimeTest>();
    let point: Vec<PiopField> = vec![PiopField::one_with_cfg(&pcs_field_cfg); num_vars];
    let (eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PiopField, CHECK>(
            params,
            trace,
            &point,
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t1.elapsed();

    let total_time = total_start.elapsed();

    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    let point_f = point;

    (
        ZincProof {
            pcs_proof_bytes: proof_bytes,
            commitment,
            ic_proof_values: vec![],
            cpr_sumcheck_messages: vec![],
            cpr_sumcheck_claimed_sum: vec![],
            cpr_up_evals: vec![],
            cpr_down_evals: vec![],
            lookup_proof: None,
            shift_sumcheck: None, // PCS-only: no PIOP
            evaluation_point_bytes: vec![],
            pcs_evals_bytes: vec![field_to_bytes(&eval_f)],
            timing: TimingBreakdown {
                pcs_commit: pcs_commit_time,
                ideal_check: Duration::ZERO,
                combined_poly_resolver: Duration::ZERO,
                lookup: Duration::ZERO,
                pcs_prove: pcs_prove_time,
                serialize: Duration::ZERO,
                total: total_time,
            },
        },
        point_f,
    )
}

/// Run the PCS-only verifier.
pub fn verify_pcs_only<Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    commitment: &ZipPlusCommitment,
    point: &[PiopField],
    eval_bytes: &[u8],
    proof_bytes: &[u8],
) -> VerifyResult
where
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod> + ConstTranscribable,
{
    let total_start = Instant::now();

    // Derive PiopField config from a fresh PCS transcript.
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PiopField, Zt::Fmod, Zt::PrimeTest>();

    // Deserialize eval.
    let eval_f: PiopField = PiopField::new_unchecked_with_cfg(
        <PiopField as Field>::Inner::read_transcription_bytes(eval_bytes),
        &pcs_field_cfg,
    );

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(proof_bytes.to_vec()),
    };
    let proof: ZipPlusProof = pcs_transcript.into();

    let t0 = Instant::now();
    let result = ZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        commitment,
        point,
        &eval_f,
        &proof,
    );
    let pcs_verify_time = t0.elapsed();

    let total_time = total_start.elapsed();

    VerifyResult {
        accepted: result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: Duration::ZERO,
            combined_poly_resolver_verify: Duration::ZERO,
            lookup_verify: Duration::ZERO,
            pcs_verify: pcs_verify_time,
            total: total_time,
        },
    }
}

// ─── Full Verifier ──────────────────────────────────────────────────────────

/// Run the full Zinc+ verifier: PIOP verification (IdealCheck + CPR) then PCS verification.
///
/// This verifies the complete pipeline:
/// 1. Deserialize the PIOP proof (IdealCheck + CPR/Sumcheck)
/// 2. Run the IdealCheck verifier (checks combined polynomials are in ideals)
/// 3. Run the CPR verifier (verifies sumcheck + constraint evaluation)
/// 4. Check that CPR evaluation claims match PCS evaluations
/// 5. Run PCS verifier
///
/// # Type parameters
/// - `U`: the UAIR.
/// - `Zt`: Zip+ types.
/// - `Lc`: linear code.
/// - `D`: degree + 1 for BinaryPoly.
/// - `CHECK`: overflow checking flag.
/// - `IdealOverF`: the field-level ideal type for verification.
/// - `IdealOverFFromRef`: maps from `IdealOrZero<U::Ideal>` to `IdealOverF`.
///
/// # Public columns
///
/// `public_column_data` provides the raw BinaryPoly data for columns
/// declared as public in `U::signature().public_columns`. The verifier
/// projects these columns to the PIOP field and evaluates their MLEs at
/// the CPR subclaim point, rather than trusting the prover's claimed
/// evaluations. Pass `&[]` when the UAIR has no public columns.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn verify<U, Zt, Lc, const D: usize, const CHECK: bool, IdealOverF, IdealOverFFromRef>(
    params: &ZipPlusParams<Zt, Lc>,
    zinc_proof: &ZincProof,
    num_vars: usize,
    ideal_over_f_from_ref: IdealOverFFromRef,
    public_column_data: &[zinc_poly::mle::DenseMultilinearExtension<BinaryPoly<D>>],
) -> VerifyResult
where
    U: Uair<Scalar = BinaryPoly<D>>,
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    BinaryPoly<D>: From<u32>,
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
{
    let total_start = Instant::now();

    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Reconstruct Fiat-Shamir transcript (must match prover) ──────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    let field_elem_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Step 1: Deserialize and verify IdealCheck ───────────────────
    let t0 = Instant::now();

    // Reconstruct IC proof from serialized bytes.
    let ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>> = zinc_proof
        .ic_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: ic_combined_mle_values,
    };

    let ic_verify_result = IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<U, _, _>(
        &mut transcript,
        ic_proof,
        num_constraints,
        num_vars,
        ideal_over_f_from_ref,
        &field_cfg,
    );

    let ic_subclaim = match ic_verify_result {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: t0.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };
    let ic_verify_time = t0.elapsed();

    // ── Step 2+2b: CPR + Lookup verification ──────────────────────
    //
    // If the proof uses `BatchedClassic`, the CPR and lookup sumchecks
    // were batched into a single multi-degree sumcheck. Otherwise, we
    // fall back to the sequential (non-batched) verification.
    let is_batched = matches!(
        zinc_proof.lookup_proof,
        Some(LookupProofData::BatchedClassic(_))
    );

    let (cpr_subclaim, cpr_verify_time, lookup_verify_time) = if is_batched {
        let t1 = Instant::now();
        let batched_proof = match zinc_proof.lookup_proof {
            Some(LookupProofData::BatchedClassic(ref bp)) => bp,
            _ => unreachable!(),
        };

        // Draw projecting element + compute projected scalars (same as non-batched).
        let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
        let projected_scalars_coeffs = project_scalars::<PiopField, U>(|scalar| {
            let one = PiopField::one_with_cfg(&field_cfg);
            let zero = PiopField::zero_with_cfg(&field_cfg);
            DynamicPolynomialF::new(
                scalar.iter().map(|coeff| {
                    if coeff.into_inner() { one.clone() } else { zero.clone() }
                }).collect::<Vec<_>>()
            )
        });
        let field_projected_scalars =
            project_scalars_to_field(projected_scalars_coeffs, &projecting_element)
                .expect("scalar projection failed");

        // CPR pre-sumcheck: draw α, check claimed_sum against IC.
        let cpr_pre = match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U>(
            &mut transcript,
            &batched_proof.md_proof.claimed_sums[0],
            num_constraints,
            &projecting_element,
            &field_projected_scalars,
            &ic_subclaim,
            &field_cfg,
        ) {
            Ok(pre) => pre,
            Err(e) => {
                eprintln!("CPR build_verifier_pre_sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

        // Lookup pre-sumcheck: for each group, mirror transcript ops.
        let mut lookup_pres = Vec::new();
        for (group_proof, meta) in batched_proof.lookup_group_proofs.iter()
            .zip(batched_proof.lookup_group_meta.iter())
        {
            let (subtable, shifts) = generate_table_and_shifts(
                &meta.table_type,
                &projecting_element,
                &field_cfg,
            );
            match BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                &mut transcript,
                group_proof,
                &subtable,
                &shifts,
                meta.num_columns,
                meta.witness_len,
                &field_cfg,
            ) {
                Ok(pre) => lookup_pres.push(pre),
                Err(e) => {
                    eprintln!("Lookup build_verifier_pre_sumcheck failed: {e:?}");
                    return VerifyResult {
                        accepted: false,
                        timing: VerifyTimingBreakdown {
                            ideal_check_verify: ic_verify_time,
                            combined_poly_resolver_verify: t1.elapsed(),
                            lookup_verify: Duration::ZERO,
                            pcs_verify: Duration::ZERO,
                            total: total_start.elapsed(),
                        },
                    };
                }
            }
        }

        // Compute shared_num_vars: max of CPR num_vars and all lookup num_vars.
        let shared_num_vars = lookup_pres
            .iter()
            .map(|p| p.num_vars)
            .max()
            .map_or(num_vars, |max_lookup| max_lookup.max(num_vars));

        // Multi-degree sumcheck verify.
        let md_subclaims = match MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
            &mut transcript,
            shared_num_vars,
            &batched_proof.md_proof,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("Multi-degree sumcheck verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

        // CPR finalize: reconstruct full up_evals (inserting public column
        // evaluations computed by the verifier) and check the subclaim.
        let sig = U::signature();
        let batched_full_up_evals = if sig.public_columns.is_empty() {
            batched_proof.cpr_up_evals.clone()
        } else {
            let binary_poly_projection =
                BinaryPoly::<D>::prepare_projection(&projecting_element);
            let public_evals: Vec<PiopField> = {
                #[cfg(feature = "parallel")]
                let iter = public_column_data.par_iter();
                #[cfg(not(feature = "parallel"))]
                let iter = public_column_data.iter();
                iter.map(|col| {
                    let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                        col.iter()
                            .map(|bp| binary_poly_projection(bp).inner().clone())
                            .collect();
                    // Pad MLE to shared_num_vars so it matches the batched
                    // sumcheck point length (lookup may extend by 1 variable).
                    let target_nv = md_subclaims.point.len();
                    if target_nv > mle.num_vars {
                        mle.evaluations.resize(1 << target_nv, Default::default());
                        mle.num_vars = target_nv;
                    }
                    mle.evaluate_with_config(&md_subclaims.point, &field_cfg)
                        .expect("public column MLE evaluation should succeed")
                })
                .collect()
            };
            reconstruct_up_evals(
                &batched_proof.cpr_up_evals,
                &public_evals,
                &sig.public_columns,
                sig.total_cols(),
            )
        };
        let batched_full_up_evals_saved = batched_full_up_evals.clone();
        let cpr_sub = match CombinedPolyResolver::<PiopField>::finalize_verifier::<U>(
            &mut transcript,
            md_subclaims.point.clone(),
            md_subclaims.expected_evaluations[0].clone(),
            &cpr_pre,
            batched_full_up_evals,
            batched_proof.cpr_down_evals.clone(),
            num_vars,
            &field_projected_scalars,
            &field_cfg,
        ) {
            Ok(sub) => sub,
            Err(e) => {
                eprintln!("CPR finalize_verifier failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

        // Shift sumcheck verify (batched path): after CPR finalize, before lookup.
        let sig_batched = U::signature();
        if let Some(ref ss_proof_data) = zinc_proof.shift_sumcheck {
            assert!(!sig_batched.shifts.is_empty());
            let ss_down_evals: Vec<PiopField> = zinc_proof.cpr_down_evals.iter()
                .map(|b| field_from_bytes(b, &field_cfg))
                .collect();
            let claims: Vec<ShiftClaim<PiopField>> = sig_batched.shifts
                .iter()
                .enumerate()
                .map(|(i, spec)| ShiftClaim {
                    source_col: i,
                    shift_amount: spec.shift_amount,
                    eval_point: cpr_sub.evaluation_point.clone(),
                    claimed_eval: ss_down_evals[i].clone(),
                })
                .collect();
            let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_proof_data.rounds.iter().map(|bytes| {
                ShiftRoundPoly {
                    evals: [
                        field_from_bytes(&bytes[0..field_elem_size], &field_cfg),
                        field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], &field_cfg),
                        field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], &field_cfg),
                    ],
                }
            }).collect();
            let ss_proof = ShiftSumcheckProof { rounds };

            // Deserialize only the private (non-public) v_finals from the proof.
            let private_v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
                .map(|b| field_from_bytes(b, &field_cfg))
                .collect();

            let has_public_shifts = sig_batched.shifts.iter()
                .any(|spec| sig_batched.is_public_column(spec.source_col));
            if has_public_shifts {
                let ss_pre = match shift_sumcheck_verify_pre(
                    &mut transcript, &ss_proof, &claims, num_vars, &field_cfg,
                ) {
                    Ok(pre) => pre,
                    Err(e) => {
                        eprintln!("Shift sumcheck pre-verify failed (batched): {e}");
                        return VerifyResult {
                            accepted: false,
                            timing: VerifyTimingBreakdown {
                                ideal_check_verify: ic_verify_time,
                                combined_poly_resolver_verify: t1.elapsed(),
                                lookup_verify: Duration::ZERO,
                                pcs_verify: Duration::ZERO,
                                total: total_start.elapsed(),
                            },
                        };
                    }
                };

                // Compute MLE evaluations for public source columns at challenge point s.
                let binary_poly_projection =
                    BinaryPoly::<D>::prepare_projection(&projecting_element);
                // BE → LE for MLE evaluation.
                let challenge_point_le: Vec<PiopField> =
                    ss_pre.challenge_point.iter().rev().cloned().collect();
                let public_shift_specs: Vec<&zinc_uair::ShiftSpec> = sig_batched.shifts.iter()
                    .filter(|spec| sig_batched.is_public_column(spec.source_col))
                    .collect();
                let public_v_finals: Vec<PiopField> = {
                    #[cfg(feature = "parallel")]
                    let iter = public_shift_specs.par_iter();
                    #[cfg(not(feature = "parallel"))]
                    let iter = public_shift_specs.iter();
                    iter.map(|spec| {
                        let pcd_idx = sig_batched.public_columns.iter()
                            .position(|&c| c == spec.source_col)
                            .expect("public shift source_col not found in public_columns");
                        let col = &public_column_data[pcd_idx];
                        let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                            col.iter()
                                .map(|bp| binary_poly_projection(bp).inner().clone())
                                .collect();
                        mle.evaluate_with_config(&challenge_point_le, &field_cfg)
                            .expect("public shift MLE evaluation should succeed")
                    })
                    .collect()
                };

                let full_v_finals = reconstruct_shift_v_finals(
                    &private_v_finals,
                    &public_v_finals,
                    sig_batched.shifts.len(),
                    |i| sig_batched.is_public_shift(i),
                );

                if let Err(e) = shift_sumcheck_verify_finalize(
                    &mut transcript, &ss_pre, &claims, &full_v_finals, &field_cfg,
                ) {
                    eprintln!("Shift sumcheck finalize failed (batched): {e}");
                    return VerifyResult {
                        accepted: false,
                        timing: VerifyTimingBreakdown {
                            ideal_check_verify: ic_verify_time,
                            combined_poly_resolver_verify: t1.elapsed(),
                            lookup_verify: Duration::ZERO,
                            pcs_verify: Duration::ZERO,
                            total: total_start.elapsed(),
                        },
                    };
                }
            } else {
                // No public shifts: use monolithic verify.
                if let Err(e) = shift_sumcheck_verify(
                    &mut transcript, &ss_proof, &claims, &private_v_finals, num_vars, &field_cfg,
                ) {
                    eprintln!("Shift sumcheck verification failed (batched): {e}");
                    return VerifyResult {
                        accepted: false,
                        timing: VerifyTimingBreakdown {
                            ideal_check_verify: ic_verify_time,
                            combined_poly_resolver_verify: t1.elapsed(),
                            lookup_verify: Duration::ZERO,
                            pcs_verify: Duration::ZERO,
                            total: total_start.elapsed(),
                        },
                    };
                }
            }
        }

        // Lookup finalize: check subclaim[g+1] for each group.
        for (g, ((lk_pre, group_proof), meta)) in lookup_pres.iter()
            .zip(batched_proof.lookup_group_proofs.iter())
            .zip(batched_proof.lookup_group_meta.iter())
            .enumerate()
        {
            // Compute eq_sum_w = Σ_{j < witness_len} eq(j, x*).
            // When witness_len = 2^w_nv < 2^point_nv, this equals
            // Π_{i = w_nv}^{point_len-1} (1 - x*[i]).
            let eq_sum_w = {
                let one = PiopField::one_with_cfg(&field_cfg);
                let w_nv = zinc_utils::log2(meta.witness_len.next_power_of_two()) as usize;
                let mut prod = one.clone();
                for i in w_nv..md_subclaims.point.len() {
                    prod *= one.clone() - &md_subclaims.point[i];
                }
                prod
            };

            // Collect parent column evaluations from CPR up_evals using
            // the witness source mapping stored in the group metadata.
            let parent_evals: Vec<PiopField> = meta.witness_sources.iter()
                .map(|ws| match ws {
                    LookupWitnessSource::Column { column_index } =>
                        batched_full_up_evals_saved[*column_index].clone(),
                    LookupWitnessSource::Affine { terms, constant_offset_bits } =>
                        eval_affine_parent::<D>(
                            terms, *constant_offset_bits,
                            &batched_full_up_evals_saved,
                            &projecting_element, &eq_sum_w,
                            &field_cfg,
                        ),
                })
                .collect();
            if let Err(e) = BatchedDecompLogupProtocol::<PiopField>::finalize_verifier(
                lk_pre,
                group_proof,
                &md_subclaims.point,
                &md_subclaims.expected_evaluations[g + 1],
                &parent_evals,
                &field_cfg,
            ) {
                eprintln!("Lookup finalize_verifier failed (group {g}): {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        }

        let cpr_verify_time = t1.elapsed();
        // Lookup time is included in CPR time for batched path.
        (cpr_sub, cpr_verify_time, Duration::ZERO)
    } else {
        // ── Sequential (non-batched) CPR + Lookup verification ──────
        let t1 = Instant::now();

        // Reconstruct CPR sumcheck proof.
        let cpr_sumcheck_messages: Vec<ProverMsg<PiopField>> = zinc_proof
            .cpr_sumcheck_messages
            .iter()
            .map(|bytes| {
                let num_evals = bytes.len() / field_elem_size;
                let tail_evaluations: Vec<PiopField> = (0..num_evals)
                    .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                    .collect();
                ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
            })
            .collect();

        let cpr_claimed_sum = field_from_bytes(&zinc_proof.cpr_sumcheck_claimed_sum, &field_cfg);
        // Deserialize private (non-public) up_evals from the proof.
        let private_up_evals: Vec<PiopField> = zinc_proof.cpr_up_evals.iter()
            .map(|b| field_from_bytes(b, &field_cfg))
            .collect();
        let cpr_down_evals: Vec<PiopField> = zinc_proof.cpr_down_evals.iter()
            .map(|b| field_from_bytes(b, &field_cfg))
            .collect();

        let cpr_sumcheck_proof = SumcheckProof {
            messages: cpr_sumcheck_messages,
            claimed_sum: cpr_claimed_sum,
        };

        // Compute projecting element and projected scalars for CPR verify.
        let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
        let projected_scalars_coeffs = project_scalars::<PiopField, U>(|scalar| {
            let one = PiopField::one_with_cfg(&field_cfg);
            let zero = PiopField::zero_with_cfg(&field_cfg);
            DynamicPolynomialF::new(
                scalar.iter().map(|coeff| {
                    if coeff.into_inner() { one.clone() } else { zero.clone() }
                }).collect::<Vec<_>>()
            )
        });
        let field_projected_scalars =
            project_scalars_to_field(projected_scalars_coeffs, &projecting_element)
                .expect("scalar projection failed");

        // Use the split CPR API so we can inject public column MLE
        // evaluations between the sumcheck and the constraint check.

        // Phase 1: pre-sumcheck (draws α, checks claimed_sum vs IC).
        let cpr_pre = match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U>(
            &mut transcript,
            &cpr_sumcheck_proof.claimed_sum,
            num_constraints,
            &projecting_element,
            &field_projected_scalars,
            &ic_subclaim,
            &field_cfg,
        ) {
            Ok(pre) => pre,
            Err(e) => {
                eprintln!("CPR build_verifier_pre_sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

        // Phase 2: sumcheck verify → subclaim with evaluation point.
        let subclaim = match MLSumcheck::<PiopField>::verify_as_subprotocol(
            &mut transcript,
            num_vars,
            max_degree + 2,
            &cpr_sumcheck_proof,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("CPR sumcheck verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

        // Phase 2b: compute public column MLE evaluations at the
        // subclaim point and reconstruct full up_evals.
        let sig = U::signature();
        let full_up_evals = if sig.public_columns.is_empty() {
            private_up_evals
        } else {
            let binary_poly_projection =
                BinaryPoly::<D>::prepare_projection(&projecting_element);
            let public_evals: Vec<PiopField> = {
                #[cfg(feature = "parallel")]
                let iter = public_column_data.par_iter();
                #[cfg(not(feature = "parallel"))]
                let iter = public_column_data.iter();
                iter.map(|col| {
                    let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                        col.iter()
                            .map(|bp| binary_poly_projection(bp).inner().clone())
                            .collect();
                    mle.evaluate_with_config(&subclaim.point, &field_cfg)
                        .expect("public column MLE evaluation should succeed")
                })
                .collect()
            };
            reconstruct_up_evals(
                &private_up_evals,
                &public_evals,
                &sig.public_columns,
                sig.total_cols(),
            )
        };

        // Phase 3: finalize CPR verifier (checks constraint, absorbs evals).
        let cpr_subclaim = match CombinedPolyResolver::<PiopField>::finalize_verifier::<U>(
            &mut transcript,
            subclaim.point,
            subclaim.expected_evaluation,
            &cpr_pre,
            full_up_evals,
            cpr_down_evals,
            num_vars,
            &field_projected_scalars,
            &field_cfg,
        ) {
            Ok(sub) => sub,
            Err(e) => {
                eprintln!("CPR finalize_verifier failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };
        let cpr_verify_time = t1.elapsed();

        // ── Step 2a: Shift sumcheck verify (sequential path) ────────
        if let Some(ref ss_proof_data) = zinc_proof.shift_sumcheck {
            assert!(!sig.shifts.is_empty(), "shift_sumcheck present but UAIR has no shifts");

            let ss_down_evals: Vec<PiopField> = zinc_proof.cpr_down_evals.iter()
                .map(|b| field_from_bytes(b, &field_cfg))
                .collect();
            let claims: Vec<ShiftClaim<PiopField>> = sig.shifts
                .iter()
                .enumerate()
                .map(|(i, spec)| ShiftClaim {
                    source_col: i,
                    shift_amount: spec.shift_amount,
                    eval_point: cpr_subclaim.evaluation_point.clone(),
                    claimed_eval: ss_down_evals[i].clone(),
                })
                .collect();

            let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_proof_data.rounds.iter().map(|bytes| {
                ShiftRoundPoly {
                    evals: [
                        field_from_bytes(&bytes[0..field_elem_size], &field_cfg),
                        field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], &field_cfg),
                        field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], &field_cfg),
                    ],
                }
            }).collect();
            let ss_proof = ShiftSumcheckProof { rounds };

            // Deserialize only the private (non-public) v_finals from the proof.
            let private_v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
                .map(|b| field_from_bytes(b, &field_cfg))
                .collect();

            // Use split API: replay sumcheck to get challenge point s,
            // then compute public v_finals before finalizing.
            let has_public_shifts = sig.shifts.iter().any(|spec| sig.is_public_column(spec.source_col));
            if has_public_shifts {
                let ss_pre = match shift_sumcheck_verify_pre(
                    &mut transcript, &ss_proof, &claims, num_vars, &field_cfg,
                ) {
                    Ok(pre) => pre,
                    Err(e) => {
                        eprintln!("Shift sumcheck pre-verify failed: {e}");
                        return VerifyResult {
                            accepted: false,
                            timing: VerifyTimingBreakdown {
                                ideal_check_verify: ic_verify_time,
                                combined_poly_resolver_verify: cpr_verify_time,
                                lookup_verify: Duration::ZERO,
                                pcs_verify: Duration::ZERO,
                                total: total_start.elapsed(),
                            },
                        };
                    }
                };

                // Compute MLE evaluations for public source columns
                // at the shift sumcheck challenge point s.
                let binary_poly_projection =
                    BinaryPoly::<D>::prepare_projection(&projecting_element);

                // The challenge point is in BE (sumcheck convention);
                // MLE evaluate_with_config expects LE ordering.
                let challenge_point_le: Vec<PiopField> =
                    ss_pre.challenge_point.iter().rev().cloned().collect();

                // Collect the public v_finals in shift order.
                let public_shift_specs: Vec<&zinc_uair::ShiftSpec> = sig.shifts.iter()
                    .filter(|spec| sig.is_public_column(spec.source_col))
                    .collect();
                let public_v_finals: Vec<PiopField> = {
                    #[cfg(feature = "parallel")]
                    let iter = public_shift_specs.par_iter();
                    #[cfg(not(feature = "parallel"))]
                    let iter = public_shift_specs.iter();
                    iter.map(|spec| {
                        let pcd_idx = sig.public_columns.iter()
                            .position(|&c| c == spec.source_col)
                            .expect("public shift source_col not found in public_columns");
                        let col = &public_column_data[pcd_idx];
                        let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                            col.iter()
                                .map(|bp| binary_poly_projection(bp).inner().clone())
                                .collect();
                        mle.evaluate_with_config(&challenge_point_le, &field_cfg)
                            .expect("public shift MLE evaluation should succeed")
                    })
                    .collect()
                };

                let full_v_finals = reconstruct_shift_v_finals(
                    &private_v_finals,
                    &public_v_finals,
                    sig.shifts.len(),
                    |i| sig.is_public_shift(i),
                );

                if let Err(e) = shift_sumcheck_verify_finalize(
                    &mut transcript, &ss_pre, &claims, &full_v_finals, &field_cfg,
                ) {
                    eprintln!("Shift sumcheck finalize failed: {e}");
                    return VerifyResult {
                        accepted: false,
                        timing: VerifyTimingBreakdown {
                            ideal_check_verify: ic_verify_time,
                            combined_poly_resolver_verify: cpr_verify_time,
                            lookup_verify: Duration::ZERO,
                            pcs_verify: Duration::ZERO,
                            total: total_start.elapsed(),
                        },
                    };
                }
            } else {
                // No public shifts: use monolithic verify with all v_finals from proof.
                if let Err(e) = shift_sumcheck_verify(
                    &mut transcript, &ss_proof, &claims, &private_v_finals, num_vars, &field_cfg,
                ) {
                    eprintln!("Shift sumcheck verification failed: {e}");
                    return VerifyResult {
                        accepted: false,
                        timing: VerifyTimingBreakdown {
                            ideal_check_verify: ic_verify_time,
                            combined_poly_resolver_verify: cpr_verify_time,
                            lookup_verify: Duration::ZERO,
                            pcs_verify: Duration::ZERO,
                            total: total_start.elapsed(),
                        },
                    };
                }
            }
        }

        // ── Step 2b: Verify Batched Decomposed LogUp (column typing) ──
        let t1b = Instant::now();
        if let Some(ref lookup_data) = zinc_proof.lookup_proof {
            // Reuse the same projecting_element from the CPR step.
            let result = match lookup_data {
                LookupProofData::Gkr(proof) => verify_gkr_batched_lookup(
                    &mut transcript, proof, &projecting_element, &field_cfg,
                ).map(|_| ()),
                LookupProofData::Classic(proof) => verify_batched_lookup(
                    &mut transcript, proof, &projecting_element, &field_cfg,
                ).map(|_| ()),
                LookupProofData::BatchedClassic(_) => unreachable!("handled above"),
            };
            if let Err(e) = result {
                eprintln!("Lookup verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: cpr_verify_time,
                        lookup_verify: t1b.elapsed(),
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        }
        let lookup_verify_time = t1b.elapsed();

        (cpr_subclaim, cpr_verify_time, lookup_verify_time)
    };

    // ── Step 3: PCS Verify ──────────────────────────────────────────
    // The CPR subclaim gives us the evaluation point and expected evaluations.
    // Verify that the PCS proof is consistent with these claims.
    let t2 = Instant::now();

    // Derive PiopField config from a fresh PCS transcript.
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PiopField, Zt::Fmod, Zt::PrimeTest>();

    // Deserialize the claimed evaluation.
    let eval_f: PiopField = PiopField::new_unchecked_with_cfg(
        <PiopField as Field>::Inner::read_transcription_bytes(&zinc_proof.pcs_evals_bytes[0]),
        &pcs_field_cfg,
    );

    // r_PCS = r_CPR: use the CPR evaluation point directly.
    // Truncate to num_vars since the sumcheck may have shared_num_vars > num_vars when lookup is present.
    let point_f: Vec<PiopField> = cpr_subclaim.evaluation_point[..num_vars].to_vec();

    // Deserialize PCS proof.
    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(zinc_proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: ZipPlusProof = pcs_transcript.into();

    let pcs_result = ZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        &zinc_proof.commitment,
        &point_f,
        &eval_f,
        &pcs_proof,
    );

    if let Err(ref e) = pcs_result {
        eprintln!("PCS verification failed: {e:?}");
    }

    let pcs_verify_time = t2.elapsed();
    let total_time = total_start.elapsed();

    VerifyResult {
        accepted: pcs_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: ic_verify_time,
            combined_poly_resolver_verify: cpr_verify_time,
            lookup_verify: lookup_verify_time,
            pcs_verify: pcs_verify_time,
            total: total_time,
        },
    }
}

// ─── Dual-Ring Pipeline ─────────────────────────────────────────────────────

/// Full proof data produced by the dual-ring prover.
///
/// Both IC passes share the same evaluation point and projecting element.
/// Both CPR sumchecks are batched into a single multi-degree sumcheck,
/// producing one shared evaluation point.
#[derive(Clone, Debug)]
pub struct DualRingZincProof {
    /// Serialized PCS proof bytes.
    pub pcs_proof_bytes: Vec<u8>,
    /// PCS commitment.
    pub commitment: ZipPlusCommitment,

    // ── IC proof values (both passes share the same evaluation point) ──
    pub bp_ic_proof_values: Vec<Vec<u8>>,
    pub qx_ic_proof_values: Vec<Vec<u8>>,

    // ── Batched CPR (multi-degree sumcheck: group 0 = BP, group 1 = QX) ──
    /// Per-group round messages, serialized.
    /// `md_group_messages[group][round]` → serialized field elements.
    pub md_group_messages: Vec<Vec<Vec<u8>>>,
    /// Claimed sum per group.
    pub md_claimed_sums: Vec<Vec<u8>>,
    /// Degree per group.
    pub md_degrees: Vec<usize>,

    /// BP CPR up/down evaluations at the shared sumcheck point.
    pub bp_cpr_up_evals: Vec<Vec<u8>>,
    pub bp_cpr_down_evals: Vec<Vec<u8>>,
    /// QX CPR up/down evaluations at the shared sumcheck point.
    pub qx_cpr_up_evals: Vec<Vec<u8>>,
    pub qx_cpr_down_evals: Vec<Vec<u8>>,

    /// Shift sumcheck proof for Q[X] columns. `None` if Q[X] UAIR has no shifts.
    pub qx_shift_sumcheck: Option<SerializedShiftSumcheckProof>,

    // ── PCS evaluation data ─────────────────────────────────────────
    pub evaluation_point_bytes: Vec<Vec<u8>>,
    pub pcs_evals_bytes: Vec<Vec<u8>>,

    pub timing: TimingBreakdown,
}

/// Run the dual-ring Zinc+ prover pipeline with **shared challenges**.
///
/// Both PIOP passes (BinaryPoly and Q[X]) share:
/// - The same Fiat-Shamir prime / field configuration
/// - The same IC evaluation point
/// - The same F[X]→F projecting element
///
/// Both CPR sumchecks are batched into a single multi-degree sumcheck
/// (group 0 = BinaryPoly, group 1 = Q[X]), producing a common
/// evaluation point for PCS opening.
///
/// # Type parameters
/// - `U1`: UAIR for BinaryPoly constraints.
/// - `U2`: UAIR for Q[X] constraints.
/// - `D1`: degree+1 for BinaryPoly.
/// - `D2`: degree+1 for DensePolynomial.
/// - `ConvertFn`: converts the BinaryPoly trace to a DensePolynomial trace.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn prove_dual_ring<U1, U2, Zt, Lc, const D1: usize, const D2: usize, const CHECK: bool, ConvertFn>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<BinaryPoly<D1>>],
    num_vars: usize,
    convert_trace: ConvertFn,
) -> DualRingZincProof
where
    U1: Uair<Scalar = BinaryPoly<D1>>,
    U2: Uair<Scalar = DensePolynomial<i64, D2>>,
    Zt: ZipTypes<Eval = BinaryPoly<D1>>,
    Lc: LinearCode<Zt>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D1>>,
    PiopField: FromPrimitiveWithConfig + FromWithConfig<i64>,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
    ConvertFn: Fn(&[DenseMultilinearExtension<BinaryPoly<D1>>]) -> Vec<DenseMultilinearExtension<DensePolynomial<i64, D2>>>,

{
    let total_start = Instant::now();

    let sig_bp = U1::signature();
    let bp_num_constraints = count_constraints::<U1>();
    let bp_max_degree = count_max_degree::<U1>();
    let qx_num_constraints = count_constraints::<U2>();
    let qx_max_degree = count_max_degree::<U2>();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let pcs_trace = private_trace(trace, &sig_bp.pcs_excluded_columns());
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, &pcs_trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: Fiat-Shamir transcript + field config ───────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    // ── Step 3: Shared IC evaluation point ──────────────────────────
    let ic_evaluation_point: Vec<PiopField> =
        transcript.get_field_challenges(num_vars, &field_cfg);

    // ── Step 4: IC₁ (BinaryPoly) — uses shared evaluation point ────
    let t1 = Instant::now();
    let bp_projected_trace = project_trace_coeffs::<PiopField, i64, i64, D1>(
        trace, &[], &[], &field_cfg,
    );
    let bp_projected_scalars = project_scalars::<PiopField, U1>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });

    let (bp_ic_proof, _bp_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_at_point::<U1>(
            &mut transcript,
            &bp_projected_trace,
            &bp_projected_scalars,
            bp_num_constraints,
            &ic_evaluation_point,
            &field_cfg,
        )
        .expect("BinaryPoly ideal check prover failed");

    // ── Step 5: Convert trace to Q[X] and run IC₂ ──────────────────
    let qx_trace = convert_trace(trace);
    let qx_projected_trace = project_trace_coeffs::<PiopField, i64, i64, D2>(
        &[], &qx_trace, &[], &field_cfg,
    );
    let qx_projected_scalars = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| PiopField::from_with_cfg(*coeff, &field_cfg)).collect::<Vec<_>>()
        )
    });

    let (qx_ic_proof, _qx_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_at_point::<U2>(
            &mut transcript,
            &qx_projected_trace,
            &qx_projected_scalars,
            qx_num_constraints,
            &ic_evaluation_point,
            &field_cfg,
        )
        .expect("Q[X] ideal check prover failed");
    let ic_time = t1.elapsed();

    // ── Step 6: Shared projecting element + project traces to F_q ───
    let t2 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

    let bp_field_projected_scalars =
        project_scalars_to_field(bp_projected_scalars, &projecting_element)
            .expect("BP scalar projection failed");
    let bp_field_trace = project_trace_to_field::<PiopField, D1>(
        trace, &[], &[], &projecting_element,
    );

    let qx_field_projected_scalars =
        project_scalars_to_field(qx_projected_scalars, &projecting_element)
            .expect("QX scalar projection failed");
    let qx_field_trace = project_trace_to_field::<PiopField, D2>(
        &[], &qx_projected_trace, &[], &projecting_element,
    );

    // ── Step 7: Build CPR sumcheck groups (without running sumcheck) ──
    // Extract QX shift trace columns before build_prover_group consumes qx_field_trace.
    let qx_sig = U2::signature();
    let qx_shift_trace_columns: Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>> =
        qx_sig.shifts.iter().map(|spec| qx_field_trace[spec.source_col].clone()).collect();

    let bp_cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U1>(
        &mut transcript,
        bp_field_trace,
        &ic_evaluation_point,
        &bp_field_projected_scalars,
        bp_num_constraints,
        num_vars,
        bp_max_degree,
        &field_cfg,
    )
    .expect("BP CPR build_prover_group failed");
    let bp_num_cols = bp_cpr_group.num_cols;

    let qx_cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U2>(
        &mut transcript,
        qx_field_trace,
        &ic_evaluation_point,
        &qx_field_projected_scalars,
        qx_num_constraints,
        num_vars,
        qx_max_degree,
        &field_cfg,
    )
    .expect("QX CPR build_prover_group failed");
    let qx_num_cols = qx_cpr_group.num_cols;

    // ── Step 8: Run batched multi-degree sumcheck ───────────────────
    let sumcheck_groups: Vec<(
        usize,
        Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>>,
        Box<dyn Fn(&[PiopField]) -> PiopField + Send + Sync>,
    )> = vec![
        (bp_cpr_group.degree, bp_cpr_group.mles, bp_cpr_group.comb_fn),
        (qx_cpr_group.degree, qx_cpr_group.mles, qx_cpr_group.comb_fn),
    ];

    let (md_proof, mut prover_states) =
        MultiDegreeSumcheck::<PiopField>::prove_as_subprotocol(
            &mut transcript,
            sumcheck_groups,
            num_vars,
            &field_cfg,
        );

    // ── Step 9: Finalize both CPR groups ────────────────────────────
    let bp_prover_state = prover_states.remove(0);
    let (bp_up_evals, bp_down_evals, bp_cpr_state) =
        CombinedPolyResolver::<PiopField>::finalize_prover(
            &mut transcript,
            bp_prover_state,
            bp_num_cols,
            &field_cfg,
        )
        .expect("BP CPR finalize_prover failed");

    let qx_prover_state = prover_states.remove(0);
    let (qx_up_evals, qx_down_evals, _qx_cpr_state) =
        CombinedPolyResolver::<PiopField>::finalize_prover(
            &mut transcript,
            qx_prover_state,
            qx_num_cols,
            &field_cfg,
        )
        .expect("QX CPR finalize_prover failed");

    // ── Step 9a: Shift Sumcheck for QX (reduce shifted-column claims) ──
    let qx_shift_sumcheck_output = if !qx_sig.shifts.is_empty() {
        let claims: Vec<ShiftClaim<PiopField>> = qx_sig.shifts
            .iter()
            .enumerate()
            .map(|(i, spec)| ShiftClaim {
                source_col: i,
                shift_amount: spec.shift_amount,
                eval_point: bp_cpr_state.evaluation_point.clone(),
                claimed_eval: qx_down_evals[i].clone(),
            })
            .collect();
        Some(shift_sumcheck_prove(
            &mut transcript,
            &claims,
            &qx_shift_trace_columns,
            num_vars,
            &field_cfg,
        ))
    } else {
        None
    };

    let cpr_time = t2.elapsed();

    // ── Step 10: PCS Prove (test + evaluate) ──────────────────────
    // r_PCS = r_CPR: pass the CPR evaluation point directly to the PCS.
    // Truncate to num_vars since the sumcheck may have shared_num_vars > num_vars when lookup is present.
    let t3 = Instant::now();
    let (eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PiopField, CHECK>(
            params,
            &pcs_trace,
            &bp_cpr_state.evaluation_point[..num_vars],
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t3.elapsed();

    let total_time = total_start.elapsed();

    // ── Serialize ───────────────────────────────────────────────────
    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    // Serialize IC proofs
    let bp_ic_values: Vec<Vec<u8>> = bp_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let qx_ic_values: Vec<Vec<u8>> = qx_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    // Serialize multi-degree sumcheck proof
    let md_group_messages: Vec<Vec<Vec<u8>>> = md_proof
        .group_messages
        .iter()
        .map(|group_rounds| {
            group_rounds
                .iter()
                .map(|msg| msg.0.tail_evaluations.iter().flat_map(|c| field_to_bytes(c)).collect())
                .collect()
        })
        .collect();
    let md_claimed_sums: Vec<Vec<u8>> = md_proof
        .claimed_sums
        .iter()
        .map(field_to_bytes)
        .collect();

    // Serialize up/down evals
    let bp_cpr_ups: Vec<Vec<u8>> = bp_up_evals.iter().map(field_to_bytes).collect();
    let bp_cpr_downs: Vec<Vec<u8>> = bp_down_evals.iter().map(field_to_bytes).collect();
    let qx_cpr_ups: Vec<Vec<u8>> = qx_up_evals.iter().map(field_to_bytes).collect();
    let qx_cpr_downs: Vec<Vec<u8>> = qx_down_evals.iter().map(field_to_bytes).collect();

    // Serialize QX shift sumcheck
    // Skip v_finals entries whose source column is public — the
    // verifier will recompute those MLE evaluations itself.
    let qx_shift_sumcheck = qx_shift_sumcheck_output.map(|output| {
        SerializedShiftSumcheckProof {
            rounds: output.proof.rounds.iter().map(|rp| {
                rp.evals.iter().flat_map(|c| field_to_bytes(c)).collect()
            }).collect(),
            v_finals: output.v_finals.iter()
                .enumerate()
                .filter(|(i, _)| !qx_sig.is_public_shift(*i))
                .map(|(_, v)| field_to_bytes(v))
                .collect(),
        }
    });

    // Serialize evaluation data
    let evaluation_point_bytes: Vec<Vec<u8>> =
        bp_cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    let pcs_evals_bytes: Vec<Vec<u8>> = vec![field_to_bytes(&eval_f)];

    DualRingZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        bp_ic_proof_values: bp_ic_values,
        qx_ic_proof_values: qx_ic_values,
        md_group_messages,
        md_claimed_sums,
        md_degrees: md_proof.degrees,
        bp_cpr_up_evals: bp_cpr_ups,
        bp_cpr_down_evals: bp_cpr_downs,
        qx_cpr_up_evals: qx_cpr_ups,
        qx_cpr_down_evals: qx_cpr_downs,
        qx_shift_sumcheck,
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ic_time,
            combined_poly_resolver: cpr_time,
            lookup: Duration::ZERO,
            pcs_prove: pcs_prove_time,
            serialize: Duration::ZERO,
            total: total_time,
        },
    }
}

/// Run the dual-ring Zinc+ verifier with **shared challenges**.
///
/// Mirrors [`prove_dual_ring`]: draws one IC evaluation point, one
/// projecting element, and verifies both CPRs via a single multi-degree
/// sumcheck.
///
/// - The BinaryPoly pass uses `TrivialIdeal` (F₂ constraints don't lift).
/// - The Q[X] pass uses the caller-provided `qx_ideal_from_ref` closure
///   to map UAIR ideals to field-level ideals (e.g. `DegreeOneIdeal(2)`).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn verify_dual_ring<U1, U2, Zt, Lc, const D1: usize, const D2: usize, const CHECK: bool, QxIdealOverF, QxIdealFromRef>(
    params: &ZipPlusParams<Zt, Lc>,
    proof: &DualRingZincProof,
    num_vars: usize,
    qx_ideal_from_ref: QxIdealFromRef,
) -> VerifyResult
where
    U1: Uair<Scalar = BinaryPoly<D1>>,
    U2: Uair<Scalar = DensePolynomial<i64, D2>>,
    Zt: ZipTypes<Eval = BinaryPoly<D1>>,
    Lc: LinearCode<Zt>,
    PiopField: FromPrimitiveWithConfig + FromWithConfig<i64>,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
    QxIdealOverF: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    QxIdealFromRef: Fn(&IdealOrZero<U2::Ideal>) -> QxIdealOverF,
{
    let total_start = Instant::now();

    let bp_num_constraints = count_constraints::<U1>();
    let qx_num_constraints = count_constraints::<U2>();
    // Degrees are encoded in the proof (proof.md_degrees) for the verifier.

    // ── Reconstruct Fiat-Shamir transcript ──────────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();
    let field_elem_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Shared IC evaluation point ──────────────────────────────────
    let ic_evaluation_point: Vec<PiopField> =
        transcript.get_field_challenges(num_vars, &field_cfg);

    // ── IC₁ verify (BinaryPoly, TrivialIdeal) ──────────────────────
    let t0 = Instant::now();

    let bp_ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>> = proof
        .bp_ic_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let bp_ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: bp_ic_combined_mle_values,
    };

    let bp_ic_subclaim = match IdealCheckProtocol::<PiopField>::verify_at_point::<U1, _, _>(
        &mut transcript,
        bp_ic_proof,
        bp_num_constraints,
        ic_evaluation_point.clone(),
        |_: &IdealOrZero<U1::Ideal>| TrivialIdeal,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("BinaryPoly IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: t0.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── IC₂ verify (Q[X], real ideals) ──────────────────────────────
    let qx_ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>> = proof
        .qx_ic_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let qx_ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: qx_ic_combined_mle_values,
    };

    let qx_ic_subclaim = match IdealCheckProtocol::<PiopField>::verify_at_point::<U2, _, _>(
        &mut transcript,
        qx_ic_proof,
        qx_num_constraints,
        ic_evaluation_point.clone(),
        &qx_ideal_from_ref,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("Q[X] IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: t0.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };
    let ic_verify_time = t0.elapsed();

    // ── Shared projecting element ───────────────────────────────────
    let t1 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

    // ── CPR₁ pre-sumcheck (draws α₁ from transcript) ───────────────
    let bp_projected_scalars_coeffs = project_scalars::<PiopField, U1>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });
    let bp_field_projected_scalars =
        project_scalars_to_field(bp_projected_scalars_coeffs, &projecting_element)
            .expect("BP scalar projection failed");

    let bp_cpr_pre = match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U1>(
        &mut transcript,
        &field_from_bytes(&proof.md_claimed_sums[0], &field_cfg),
        bp_num_constraints,
        &projecting_element,
        &bp_field_projected_scalars,
        &bp_ic_subclaim,
        &field_cfg,
    ) {
        Ok(pre) => pre,
        Err(e) => {
            eprintln!("BP CPR pre-sumcheck failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── CPR₂ pre-sumcheck (draws α₂ from transcript) ───────────────
    let qx_projected_scalars_coeffs = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| PiopField::from_with_cfg(*coeff, &field_cfg)).collect::<Vec<_>>()
        )
    });
    let qx_field_projected_scalars =
        project_scalars_to_field(qx_projected_scalars_coeffs, &projecting_element)
            .expect("QX scalar projection failed");

    let qx_cpr_pre = match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U2>(
        &mut transcript,
        &field_from_bytes(&proof.md_claimed_sums[1], &field_cfg),
        qx_num_constraints,
        &projecting_element,
        &qx_field_projected_scalars,
        &qx_ic_subclaim,
        &field_cfg,
    ) {
        Ok(pre) => pre,
        Err(e) => {
            eprintln!("QX CPR pre-sumcheck failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Deserialize multi-degree sumcheck proof ─────────────────────
    let md_group_messages: Vec<Vec<ProverMsg<PiopField>>> = proof
        .md_group_messages
        .iter()
        .map(|group_rounds| {
            group_rounds
                .iter()
                .map(|bytes| {
                    let num_evals = bytes.len() / field_elem_size;
                    let tail_evaluations: Vec<PiopField> = (0..num_evals)
                        .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                        .collect();
                    ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
                })
                .collect()
        })
        .collect();

    let md_claimed_sums: Vec<PiopField> = proof
        .md_claimed_sums
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let md_proof = MultiDegreeSumcheckProof {
        group_messages: md_group_messages,
        claimed_sums: md_claimed_sums,
        degrees: proof.md_degrees.clone(),
    };

    // ── Verify multi-degree sumcheck ────────────────────────────────
    let md_subclaims = match MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
        &mut transcript,
        num_vars,
        &md_proof,
        &field_cfg,
    ) {
        Ok(subclaims) => subclaims,
        Err(e) => {
            eprintln!("Multi-degree sumcheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Finalize CPR₁ (BinaryPoly) ─────────────────────────────────
    let bp_up_evals: Vec<PiopField> = proof.bp_cpr_up_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let bp_down_evals: Vec<PiopField> = proof.bp_cpr_down_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let bp_cpr_subclaim = match CombinedPolyResolver::<PiopField>::finalize_verifier::<U1>(
        &mut transcript,
        md_subclaims.point.clone(),
        md_subclaims.expected_evaluations[0].clone(),
        &bp_cpr_pre,
        bp_up_evals,
        bp_down_evals,
        num_vars,
        &bp_field_projected_scalars,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("BP CPR finalize_verifier failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Finalize CPR₂ (Q[X]) ───────────────────────────────────────
    let qx_up_evals: Vec<PiopField> = proof.qx_cpr_up_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let qx_down_evals: Vec<PiopField> = proof.qx_cpr_down_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    if let Err(e) = CombinedPolyResolver::<PiopField>::finalize_verifier::<U2>(
        &mut transcript,
        md_subclaims.point.clone(),
        md_subclaims.expected_evaluations[1].clone(),
        &qx_cpr_pre,
        qx_up_evals,
        qx_down_evals,
        num_vars,
        &qx_field_projected_scalars,
        &field_cfg,
    ) {
        eprintln!("QX CPR finalize_verifier failed: {e:?}");
        return VerifyResult {
            accepted: false,
            timing: VerifyTimingBreakdown {
                ideal_check_verify: ic_verify_time,
                combined_poly_resolver_verify: t1.elapsed(),
                lookup_verify: Duration::ZERO,
                pcs_verify: Duration::ZERO,
                total: total_start.elapsed(),
            },
        };
    }
    let cpr_verify_time = t1.elapsed();

    // ── Step: Shift Sumcheck verify for QX ──────────────────────────
    let qx_sig = U2::signature();
    if let Some(ref ss_proof_data) = proof.qx_shift_sumcheck {
        assert!(!qx_sig.shifts.is_empty(), "qx_shift_sumcheck present but QX UAIR has no shifts");

        let ss_down_evals: Vec<PiopField> = proof.qx_cpr_down_evals.iter()
            .map(|b| field_from_bytes(b, &field_cfg))
            .collect();
        let claims: Vec<ShiftClaim<PiopField>> = qx_sig.shifts
            .iter()
            .enumerate()
            .map(|(i, spec)| ShiftClaim {
                source_col: i,
                shift_amount: spec.shift_amount,
                eval_point: bp_cpr_subclaim.evaluation_point.clone(),
                claimed_eval: ss_down_evals[i].clone(),
            })
            .collect();

        let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_proof_data.rounds.iter().map(|bytes| {
            let n = bytes.len() / field_elem_size;
            assert_eq!(n, 3, "each shift round poly should have 3 evaluations");
            ShiftRoundPoly {
                evals: [
                    field_from_bytes(&bytes[0..field_elem_size], &field_cfg),
                    field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], &field_cfg),
                    field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], &field_cfg),
                ],
            }
        }).collect();
        let ss_proof = ShiftSumcheckProof { rounds };

        let v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
            .map(|b| field_from_bytes(b, &field_cfg))
            .collect();

        if let Err(e) = shift_sumcheck_verify(
            &mut transcript,
            &ss_proof,
            &claims,
            &v_finals,
            num_vars,
            &field_cfg,
        ) {
            eprintln!("QX Shift sumcheck verification failed: {e}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: cpr_verify_time,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    }

    // ── PCS Verify ──────────────────────────────────────────────────
    let t2 = Instant::now();

    // Derive PiopField config from a fresh PCS transcript.
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PiopField, Zt::Fmod, Zt::PrimeTest>();

    // Deserialize the claimed evaluation.
    let eval_f: PiopField = PiopField::new_unchecked_with_cfg(
        <PiopField as Field>::Inner::read_transcription_bytes(&proof.pcs_evals_bytes[0]),
        &pcs_field_cfg,
    );

    // r_PCS = r_CPR: use the CPR evaluation point directly.
    // Truncate to num_vars since the sumcheck may have shared_num_vars > num_vars when lookup is present.
    let point_f: Vec<PiopField> = bp_cpr_subclaim.evaluation_point[..num_vars].to_vec();

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: ZipPlusProof = pcs_transcript.into();

    let pcs_result = ZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        &proof.commitment,
        &point_f,
        &eval_f,
        &pcs_proof,
    );
    let pcs_verify_time = t2.elapsed();

    VerifyResult {
        accepted: pcs_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: ic_verify_time,
            combined_poly_resolver_verify: cpr_verify_time,
            lookup_verify: Duration::ZERO,
            pcs_verify: pcs_verify_time,
            total: total_start.elapsed(),
        },
    }
}

// ─── Dual-Circuit Pipeline ──────────────────────────────────────────────────

/// Full proof data produced by the dual-circuit prover.
///
/// Two **separate** circuits (e.g. SHA-256 with `BinaryPoly<D>` and ECDSA
/// with `Int<4>`) share one Fiat-Shamir transcript, one IC evaluation
/// point, one projecting element, and one multi-degree sumcheck.  Each
/// has its own PCS commitment and proof.
///
/// Multi-degree sumcheck group ordering:
///   group 0       — CPR for circuit 1 (BinaryPoly)
///   group 1       — CPR for circuit 2 (generic ring)
///   groups 2..N+2 — lookup groups for circuit 1
#[derive(Clone, Debug)]
pub struct DualCircuitZincProof {
    // ── PCS (circuit 1 = BinaryPoly) ────────────────────────────────
    pub pcs1_proof_bytes: Vec<u8>,
    pub pcs1_commitment: ZipPlusCommitment,

    // ── PCS (circuit 2 = generic ring) ──────────────────────────────
    pub pcs2_proof_bytes: Vec<u8>,
    pub pcs2_commitment: ZipPlusCommitment,

    // ── IC proof values (both use shared eval point) ────────────────
    pub ic1_proof_values: Vec<Vec<u8>>,
    pub ic2_proof_values: Vec<Vec<u8>>,

    // ── Batched multi-degree sumcheck ───────────────────────────────
    pub md_group_messages: Vec<Vec<Vec<u8>>>,
    pub md_claimed_sums: Vec<Vec<u8>>,
    pub md_degrees: Vec<usize>,

    // ── CPR up/down evals ───────────────────────────────────────────
    pub cpr1_up_evals: Vec<Vec<u8>>,
    pub cpr1_down_evals: Vec<Vec<u8>>,
    pub cpr2_up_evals: Vec<Vec<u8>>,
    pub cpr2_down_evals: Vec<Vec<u8>>,

    // ── Shift sumcheck for circuit 2 (e.g. ECDSA) ──────────────────
    // ── Unified evaluation sumcheck (eq + shift claims) ─────────────
    /// Batched sumcheck that unifies all column evaluation claims
    /// (from both circuits) at a single random point.  Contains eq
    /// claims for every up-eval column and shift claims for every
    /// explicit-shift down-eval column.  `None` only when there are
    /// zero columns (should never happen in practice).
    pub unified_eval_sumcheck: Option<SerializedShiftSumcheckProof>,

    // ── Lookup (circuit 1 only) ─────────────────────────────────────
    pub lookup_group_meta: Vec<LookupGroupMeta>,
    pub lookup_group_proofs: Vec<zinc_piop::lookup::BatchedDecompLogupProof<PiopField>>,

    // ── Shared evaluation point + PCS evals ─────────────────────────
    pub evaluation_point_bytes: Vec<Vec<u8>>,
    pub pcs1_evals_bytes: Vec<Vec<u8>>,
    pub pcs2_evals_bytes: Vec<Vec<u8>>,

    pub timing: TimingBreakdown,
}

/// Run the dual-circuit Zinc+ prover with **shared challenges**.
///
/// Combines two independent circuits into a single proving pipeline:
/// - Circuit 1: `BinaryPoly<D>` (e.g. SHA-256) with optional lookup
/// - Circuit 2: generic ring `R2` (e.g. ECDSA with `Int<4>`)
///
/// Both circuits share:
/// - The same Fiat-Shamir prime / field configuration
/// - The same IC evaluation point
/// - The same F\[X\]→F projecting element
/// - A single multi-degree sumcheck (CPR₁ + CPR₂ + lookup groups)
///
/// Each circuit retains its own PCS (commit → test → evaluate).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn prove_dual_circuit<U1, U2, R2, Zt1, Lc1, Zt2, Lc2, PcsF2, const D: usize, const CHECK: bool>(
    params1: &ZipPlusParams<Zt1, Lc1>,
    trace1: &[DenseMultilinearExtension<BinaryPoly<D>>],
    params2: &ZipPlusParams<Zt2, Lc2>,
    trace2: &[DenseMultilinearExtension<R2>],
    num_vars: usize,
    lookup_specs: &[LookupColumnSpec],
    affine_lookup_specs: &[AffineLookupSpec],
) -> DualCircuitZincProof
where
    // Circuit 1 (BinaryPoly<D>):
    U1: Uair<Scalar = BinaryPoly<D>>,
    Zt1: ZipTypes<Eval = BinaryPoly<D>>,
    Lc1: LinearCode<Zt1>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D>>,
    BinaryPoly<D>: From<u32>,
    Zt1::Eval: ProjectableToField<PiopField>,
    Zt1::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt1::Chal> + for<'a> FromWithConfig<&'a Zt1::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt1::Fmod>,
    // Circuit 2 (generic ring R2):
    R2: ProjectableToField<PiopField>
        + crypto_primitives::Semiring
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + 'static,
    U2: Uair<Scalar = R2>,
    Zt2: ZipTypes<Eval = R2>,
    Lc2: LinearCode<Zt2>,
    PiopField: FromWithConfig<R2>,
    PcsF2: PrimeField
        + FromPrimitiveWithConfig
        + FromWithConfig<PiopUint>
        + for<'a> FromWithConfig<&'a Zt2::Chal>
        + for<'a> FromWithConfig<&'a Zt2::CombR>
        + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a PcsF2>
        + FromRef<PcsF2>,
    PcsF2::Inner: FromRef<Zt2::Fmod> + ConstTranscribable,
    Zt2::Eval: ProjectableToField<PcsF2>,
    // Shared:
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable
        + FromRef<<PiopField as Field>::Inner>
        + Send
        + Sync
        + Default
        + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
{
    let total_start = Instant::now();

    let sig1 = U1::signature();
    let sig2 = U2::signature();
    let c1_num_constraints = count_constraints::<U1>();
    let c1_max_degree = count_max_degree::<U1>();
    let c2_num_constraints = count_constraints::<U2>();
    let c2_max_degree = count_max_degree::<U2>();

    // ── Step 1: PCS Commit (both circuits) ──────────────────────────
    let t0 = Instant::now();
    let pcs_trace1 = private_trace(trace1, &sig1.pcs_excluded_columns());
    let pcs_trace2 = private_trace(trace2, &sig2.pcs_excluded_columns());
    let (hint1, commitment1) = ZipPlus::<Zt1, Lc1>::commit(params1, &pcs_trace1)
        .expect("PCS1 commit failed");
    let (hint2, commitment2) = ZipPlus::<Zt2, Lc2>::commit(params2, &pcs_trace2)
        .expect("PCS2 commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: Shared transcript + field config ────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    // ── Step 3: Shared IC evaluation point ──────────────────────────
    let ic_evaluation_point: Vec<PiopField> =
        transcript.get_field_challenges(num_vars, &field_cfg);

    // ── Step 4: IC₁ (BinaryPoly) at shared point ───────────────────
    let t1 = Instant::now();
    let c1_projected_trace = project_trace_coeffs::<PiopField, i64, i64, D>(
        trace1, &[], &[], &field_cfg,
    );
    let c1_projected_scalars = project_scalars::<PiopField, U1>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar
                .iter()
                .map(|coeff| {
                    if coeff.into_inner() {
                        one.clone()
                    } else {
                        zero.clone()
                    }
                })
                .collect::<Vec<_>>(),
        )
    });

    let (c1_ic_proof, _c1_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_at_point::<U1>(
            &mut transcript,
            &c1_projected_trace,
            &c1_projected_scalars,
            c1_num_constraints,
            &ic_evaluation_point,
            &field_cfg,
        )
        .expect("BinaryPoly ideal check prover failed");

    // ── Step 5: IC₂ (generic R) at shared point ────────────────────
    let c2_projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<PiopField>>> =
        trace2
            .iter()
            .map(|col_mle| {
                col_mle
                    .iter()
                    .map(|elem| DynamicPolynomialF {
                        coeffs: vec![PiopField::from_with_cfg(elem.clone(), &field_cfg)],
                    })
                    .collect()
            })
            .collect();

    let c2_projected_scalars = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF {
            coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &field_cfg)],
        }
    });

    let (c2_ic_proof, _c2_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_at_point::<U2>(
            &mut transcript,
            &c2_projected_trace,
            &c2_projected_scalars,
            c2_num_constraints,
            &ic_evaluation_point,
            &field_cfg,
        )
        .expect("Generic ring ideal check prover failed");
    let ic_time = t1.elapsed();

    // ── Step 6: Shared projecting element + project both to F_q ────
    let t2 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

    let c1_field_projected_scalars =
        project_scalars_to_field(c1_projected_scalars, &projecting_element)
            .expect("C1 scalar projection failed");
    let c1_field_trace = project_trace_to_field::<PiopField, D>(
        trace1, &[], &[], &projecting_element,
    );

    let c2_field_projected_scalars =
        project_scalars_to_field(c2_projected_scalars, &projecting_element)
            .expect("C2 scalar projection failed");
    let c2_field_trace = project_trace_to_field::<PiopField, 1>(
        &[], &[], &c2_projected_trace, &projecting_element,
    );

    // ── Step 7: Extract lookup columns from circuit 1's field trace ──
    let has_lookups = !lookup_specs.is_empty() || !affine_lookup_specs.is_empty();
    let lookup_precomputed = if has_lookups {
        let (mut columns, mut raw_indices, mut remapped_specs, reverse_index_map) =
            if !lookup_specs.is_empty() {
                extract_lookup_columns_from_field_trace(trace1, &c1_field_trace, lookup_specs, &field_cfg)
            } else {
                (vec![], vec![], vec![], vec![])
            };
        let ws_map = append_affine_virtual_columns::<D>(
            trace1, affine_lookup_specs, &projecting_element, &field_cfg, num_vars,
            &mut columns, &mut raw_indices, &mut remapped_specs, &reverse_index_map,
        );
        Some((columns, raw_indices, remapped_specs, ws_map))
    } else {
        None
    };

    // ── Step 8: Build CPR group for circuit 1 ───────────────────────
    // Clone the field traces before build_prover_group consumes them —
    // the unified evaluation sumcheck (step 14a) needs the raw MLE
    // columns to fold during its sumcheck rounds.
    let c1_field_trace_for_eval = c1_field_trace.clone();
    let mut c1_cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U1>(
        &mut transcript,
        c1_field_trace,
        &ic_evaluation_point,
        &c1_field_projected_scalars,
        c1_num_constraints,
        num_vars,
        c1_max_degree,
        &field_cfg,
    )
    .expect("C1 CPR build_prover_group failed");
    let c1_num_cols = c1_cpr_group.num_cols;

    // ── Step 9: Build lookup groups for circuit 1 ───────────────────
    let mut lookup_groups_data: Vec<(LookupSumcheckGroup<PiopField>, LookupGroupMeta)> =
        Vec::new();
    if let Some((ref columns, ref raw_indices, ref remapped_specs, ref ws_map)) = lookup_precomputed {
        let groups = group_lookup_specs(remapped_specs);
        for group in &groups {
            let instance = build_lookup_instance_from_indices_pub(
                columns,
                raw_indices,
                group,
                &projecting_element,
                &field_cfg,
            )
            .expect("lookup instance build failed");

            let witness_len = instance.witnesses[0].len();

            let lk_group = BatchedDecompLogupProtocol::<PiopField>::build_prover_group(
                &mut transcript,
                &instance,
                &field_cfg,
            )
            .expect("lookup build_prover_group failed");

            let meta = LookupGroupMeta {
                table_type: group.table_type.clone(),
                num_columns: group.column_indices.len(),
                witness_len,
                witness_sources: group.column_indices.iter()
                    .map(|&remapped| ws_map[remapped].clone())
                    .collect(),
            };
            lookup_groups_data.push((lk_group, meta));
        }
    }

    // ── Step 10: Build CPR group for circuit 2 ──────────────────────
    // Clone ALL c2 field trace columns for the unified eval sumcheck.
    let c2_field_trace_for_eval: Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>> =
        c2_field_trace.clone();

    let mut c2_cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U2>(
        &mut transcript,
        c2_field_trace,
        &ic_evaluation_point,
        &c2_field_projected_scalars,
        c2_num_constraints,
        num_vars,
        c2_max_degree,
        &field_cfg,
    )
    .expect("C2 CPR build_prover_group failed");
    let c2_num_cols = c2_cpr_group.num_cols;

    // ── Step 11: Compute shared_num_vars and pad ────────────────────
    let shared_num_vars = {
        let max_lk = lookup_groups_data
            .iter()
            .map(|(g, _)| g.num_vars)
            .max()
            .unwrap_or(num_vars);
        num_vars.max(max_lk)
    };

    if shared_num_vars > num_vars {
        let target_len = 1usize << shared_num_vars;
        for mle in &mut c1_cpr_group.mles {
            mle.evaluations.resize(target_len, Default::default());
            mle.num_vars = shared_num_vars;
        }
        for mle in &mut c2_cpr_group.mles {
            mle.evaluations.resize(target_len, Default::default());
            mle.num_vars = shared_num_vars;
        }
    }

    // ── Step 12: Assemble all groups and run multi-degree sumcheck ──
    let mut sumcheck_groups: Vec<(
        usize,
        Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>>,
        Box<dyn Fn(&[PiopField]) -> PiopField + Send + Sync>,
    )> = Vec::with_capacity(2 + lookup_groups_data.len());

    sumcheck_groups.push((c1_cpr_group.degree, c1_cpr_group.mles, c1_cpr_group.comb_fn));
    sumcheck_groups.push((c2_cpr_group.degree, c2_cpr_group.mles, c2_cpr_group.comb_fn));

    let mut lookup_pre_data: Vec<(LookupSumcheckGroup<PiopField>, LookupGroupMeta)> = Vec::new();
    for (lk_group, meta) in lookup_groups_data {
        sumcheck_groups.push((lk_group.degree, lk_group.mles, lk_group.comb_fn));
        lookup_pre_data.push((
            LookupSumcheckGroup {
                degree: 0,
                mles: vec![],
                comb_fn: Box::new(|_: &[PiopField]| unreachable!()),
                num_vars: 0,
                aggregated_multiplicities: lk_group.aggregated_multiplicities,
                chunk_inverse_witnesses: lk_group.chunk_inverse_witnesses,
                inverse_table: lk_group.inverse_table,
            },
            meta,
        ));
    }

    let (md_proof, mut prover_states) =
        MultiDegreeSumcheck::<PiopField>::prove_as_subprotocol(
            &mut transcript,
            sumcheck_groups,
            shared_num_vars,
            &field_cfg,
        );

    // ── Step 13: Finalize CPR₁ (circuit 1) ─────────────────────────
    let c1_prover_state = prover_states.remove(0);
    let (c1_up_evals, c1_down_evals, c1_cpr_state) =
        CombinedPolyResolver::<PiopField>::finalize_prover(
            &mut transcript,
            c1_prover_state,
            c1_num_cols,
            &field_cfg,
        )
        .expect("C1 CPR finalize_prover failed");

    // ── Step 14: Finalize CPR₂ (circuit 2) ─────────────────────────
    let c2_prover_state = prover_states.remove(0);
    let (c2_up_evals, c2_down_evals, _c2_cpr_state) =
        CombinedPolyResolver::<PiopField>::finalize_prover(
            &mut transcript,
            c2_prover_state,
            c2_num_cols,
            &field_cfg,
        )
        .expect("C2 CPR finalize_prover failed");

    // ── Step 14a: Unified evaluation sumcheck ───────────────────────
    //
    // Batch together:
    //   • eq-based MLE evaluation claims for EVERY up-eval column from
    //     both circuits  (shift_amount = 0),
    //   • genuine shift claims for circuit 2's explicit ShiftSpecs
    //     (shift_amount > 0).
    //
    // After the sumcheck the verifier obtains evaluation claims for all
    // column MLEs at a single fresh random point ("point unification").
    let c1_num_up = c1_up_evals.len();
    let c2_num_up = c2_up_evals.len();

    let mut unified_claims: Vec<ShiftClaim<PiopField>> =
        Vec::with_capacity(c1_num_up + c2_num_up + sig2.shifts.len());

    // Eq claims for circuit 1 columns.
    for i in 0..c1_num_up {
        unified_claims.push(ShiftClaim {
            source_col: i,
            shift_amount: 0,
            eval_point: c1_cpr_state.evaluation_point.clone(),
            claimed_eval: c1_up_evals[i].clone(),
        });
    }
    // Eq claims for circuit 2 columns.
    for j in 0..c2_num_up {
        unified_claims.push(ShiftClaim {
            source_col: c1_num_up + j,
            shift_amount: 0,
            eval_point: c1_cpr_state.evaluation_point.clone(),
            claimed_eval: c2_up_evals[j].clone(),
        });
    }
    // Shift claims for circuit 2's explicit ShiftSpecs.
    for (k, spec) in sig2.shifts.iter().enumerate() {
        unified_claims.push(ShiftClaim {
            source_col: c1_num_up + spec.source_col,
            shift_amount: spec.shift_amount,
            eval_point: c1_cpr_state.evaluation_point.clone(),
            claimed_eval: c2_down_evals[k].clone(),
        });
    }

    // Assemble the combined trace column array:
    //   [c1_col_0 … c1_col_{m1-1}, c2_col_0 … c2_col_{m2-1}]
    let mut unified_trace_columns: Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>> =
        Vec::with_capacity(c1_num_up + c2_num_up);
    unified_trace_columns.extend(c1_field_trace_for_eval);
    unified_trace_columns.extend(c2_field_trace_for_eval);

    let unified_eval_output = shift_sumcheck_prove(
        &mut transcript,
        &unified_claims,
        &unified_trace_columns,
        num_vars,
        &field_cfg,
    );

    // ── Step 15: Finalize lookup groups ─────────────────────────────
    let mut lookup_group_proofs = Vec::new();
    let mut lookup_group_meta = Vec::new();
    for (i, (lk_pre, meta)) in lookup_pre_data.into_iter().enumerate() {
        let lk_sumcheck_proof = SumcheckProof {
            messages: md_proof.group_messages[i + 2].iter().cloned().collect(),
            claimed_sum: md_proof.claimed_sums[i + 2].clone(),
        };
        let lk_eval_point = prover_states.remove(0).randomness;
        let (lk_proof, _lk_state) =
            BatchedDecompLogupProtocol::<PiopField>::finalize_prover(
                lk_pre,
                lk_sumcheck_proof,
                lk_eval_point,
            );
        lookup_group_proofs.push(lk_proof);
        lookup_group_meta.push(meta);
    }

    let cpr_lookup_time = t2.elapsed();

    // ── Step 16: PCS Prove (both circuits) ────────────────────────
    // r_PCS = r_CPR: pass the CPR evaluation point directly.
    // Circuit 1 uses PiopField; circuit 2 converts to PcsF2.
    // Truncate to num_vars since the sumcheck may have shared_num_vars > num_vars when lookup is present.
    let t3 = Instant::now();
    let (eval1_f, proof1) =
        ZipPlus::<Zt1, Lc1>::prove::<PiopField, CHECK>(
            params1,
            &pcs_trace1,
            &c1_cpr_state.evaluation_point[..num_vars],
            &hint1,
        )
        .expect("PCS1 prove failed");

    let pcs2_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PcsF2, Zt2::Fmod, Zt2::PrimeTest>();
    let point2: Vec<PcsF2> = piop_point_to_pcs_field(&c1_cpr_state.evaluation_point[..num_vars], &pcs2_field_cfg);
    let (eval2_f, proof2) =
        ZipPlus::<Zt2, Lc2>::prove::<PcsF2, CHECK>(
            params2,
            &pcs_trace2,
            &point2,
            &hint2,
        )
        .expect("PCS2 prove failed");
    let pcs_prove_time = t3.elapsed();

    let total_time = total_start.elapsed();

    // ── Serialize ───────────────────────────────────────────────────
    let t_ser = Instant::now();
    let proof1_bytes: Vec<u8> = {
        let t: zip_plus::pcs_transcript::PcsTranscript = proof1.into();
        t.stream.into_inner()
    };
    let proof2_bytes: Vec<u8> = {
        let t: zip_plus::pcs_transcript::PcsTranscript = proof2.into();
        t.stream.into_inner()
    };

    let ic1_values: Vec<Vec<u8>> = c1_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let ic2_values: Vec<Vec<u8>> = c2_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    let md_group_messages: Vec<Vec<Vec<u8>>> = md_proof
        .group_messages
        .iter()
        .map(|group_rounds| {
            group_rounds
                .iter()
                .map(|msg| {
                    msg.0
                        .tail_evaluations
                        .iter()
                        .flat_map(|c| field_to_bytes(c))
                        .collect()
                })
                .collect()
        })
        .collect();
    let md_claimed_sums: Vec<Vec<u8>> = md_proof
        .claimed_sums
        .iter()
        .map(field_to_bytes)
        .collect();

    // Only serialize non-public column evaluations — the verifier
    // computes public column MLE evaluations from the known public data.
    let cpr1_ups: Vec<Vec<u8>> = c1_up_evals.iter()
        .enumerate()
        .filter(|(i, _)| !sig1.is_public_column(*i))
        .map(|(_, v)| field_to_bytes(v))
        .collect();
    let cpr1_downs: Vec<Vec<u8>> = c1_down_evals.iter().map(field_to_bytes).collect();
    let cpr2_ups: Vec<Vec<u8>> = c2_up_evals.iter()
        .enumerate()
        .filter(|(i, _)| !sig2.is_public_column(*i))
        .map(|(_, v)| field_to_bytes(v))
        .collect();
    let cpr2_downs: Vec<Vec<u8>> = c2_down_evals.iter().map(field_to_bytes).collect();

    // Serialize unified evaluation sumcheck.
    // Filter out v_finals for public columns / public shift sources.
    // The indexing is: [c1_eq_0..c1_eq_{m1-1}, c2_eq_0..c2_eq_{m2-1}, c2_shift_0..]
    let is_public_unified_claim = |idx: usize| -> bool {
        if idx < c1_num_up {
            sig1.is_public_column(idx)
        } else if idx < c1_num_up + c2_num_up {
            sig2.is_public_column(idx - c1_num_up)
        } else {
            sig2.is_public_shift(idx - c1_num_up - c2_num_up)
        }
    };
    let unified_eval_sumcheck = Some(SerializedShiftSumcheckProof {
        rounds: unified_eval_output.proof.rounds.iter().map(|rp| {
            rp.evals.iter().flat_map(|c| field_to_bytes(c)).collect()
        }).collect(),
        v_finals: unified_eval_output.v_finals.iter()
            .enumerate()
            .filter(|(i, _)| !is_public_unified_claim(*i))
            .map(|(_, v)| field_to_bytes(v))
            .collect(),
    });

    let evaluation_point_bytes: Vec<Vec<u8>> =
        c1_cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    let pcs1_evals: Vec<Vec<u8>> = vec![field_to_bytes(&eval1_f)];
    // PCS2 evals are in PcsF2; serialize using ConstTranscribable.
    let pcs2_evals: Vec<Vec<u8>> = {
        let mut buf = vec![0u8; <PcsF2::Inner as ConstTranscribable>::NUM_BYTES];
        eval2_f.inner().write_transcription_bytes(&mut buf);
        vec![buf]
    };
    let serialize_time = t_ser.elapsed();

    DualCircuitZincProof {
        pcs1_proof_bytes: proof1_bytes,
        pcs1_commitment: commitment1,
        pcs2_proof_bytes: proof2_bytes,
        pcs2_commitment: commitment2,
        ic1_proof_values: ic1_values,
        ic2_proof_values: ic2_values,
        md_group_messages,
        md_claimed_sums,
        md_degrees: md_proof.degrees,
        cpr1_up_evals: cpr1_ups,
        cpr1_down_evals: cpr1_downs,
        cpr2_up_evals: cpr2_ups,
        cpr2_down_evals: cpr2_downs,
        unified_eval_sumcheck,
        lookup_group_meta,
        lookup_group_proofs,
        evaluation_point_bytes,
        pcs1_evals_bytes: pcs1_evals,
        pcs2_evals_bytes: pcs2_evals,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ic_time,
            combined_poly_resolver: cpr_lookup_time,
            lookup: Duration::ZERO, // included in combined_poly_resolver
            pcs_prove: pcs_prove_time,
            serialize: serialize_time,
            total: total_time,
        },
    }
}

/// Run the dual-circuit Zinc+ verifier with **shared challenges**.
///
/// Mirrors [`prove_dual_circuit`]: draws one IC evaluation point, one
/// projecting element, and verifies both CPRs + lookup via a single
/// multi-degree sumcheck.  Then verifies both PCS proofs independently.
///
/// # Type parameters
/// - `U1`: UAIR for circuit 1 (BinaryPoly<D> constraints).
/// - `U2`: UAIR for circuit 2 (generic ring R2 constraints).
/// - `R2`: ring for circuit 2 (e.g. `Int<4>`).
/// - `IdealOverF1/2`: field-level ideal types for verification.
/// - `IdealFromRef1/2`: closures mapping UAIR ideals to field ideals.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn verify_dual_circuit<
    U1, U2, R2,
    Zt1, Lc1, Zt2, Lc2, PcsF2,
    const D: usize, const CHECK: bool,
    IdealOverF1, IdealFromRef1,
    IdealOverF2, IdealFromRef2,
>(
    params1: &ZipPlusParams<Zt1, Lc1>,
    params2: &ZipPlusParams<Zt2, Lc2>,
    proof: &DualCircuitZincProof,
    num_vars: usize,
    ideal_from_ref1: IdealFromRef1,
    ideal_from_ref2: IdealFromRef2,
    c1_public_column_data: &[zinc_poly::mle::DenseMultilinearExtension<BinaryPoly<D>>],
    c2_public_column_data: &[zinc_poly::mle::DenseMultilinearExtension<R2>],
) -> VerifyResult
where
    // Circuit 1 (BinaryPoly<D>):
    U1: Uair<Scalar = BinaryPoly<D>>,
    Zt1: ZipTypes<Eval = BinaryPoly<D>>,
    Lc1: LinearCode<Zt1>,
    BinaryPoly<D>: From<u32>,
    Zt1::Eval: ProjectableToField<PiopField>,
    Zt1::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt1::Chal> + for<'a> FromWithConfig<&'a Zt1::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt1::Fmod>,
    IdealOverF1: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    IdealFromRef1: Fn(&IdealOrZero<U1::Ideal>) -> IdealOverF1,
    // Circuit 2 (generic ring R2):
    R2: ProjectableToField<PiopField>
        + crypto_primitives::Semiring
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + 'static,
    U2: Uair<Scalar = R2>,
    Zt2: ZipTypes<Eval = R2>,
    Lc2: LinearCode<Zt2>,
    PiopField: FromWithConfig<R2>,
    PcsF2: PrimeField
        + FromPrimitiveWithConfig
        + FromWithConfig<PiopUint>
        + for<'a> FromWithConfig<&'a Zt2::Chal>
        + for<'a> FromWithConfig<&'a Zt2::CombR>
        + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a PcsF2>
        + FromRef<PcsF2>,
    PcsF2::Inner: FromRef<Zt2::Fmod> + ConstTranscribable,
    Zt2::Eval: ProjectableToField<PcsF2>,
    Zt2::Cw: ProjectableToField<PcsF2>,
    IdealOverF2: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    IdealFromRef2: Fn(&IdealOrZero<U2::Ideal>) -> IdealOverF2,
    // Shared:
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable
        + FromRef<<PiopField as Field>::Inner>
        + Send
        + Sync
        + Default
        + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
{
    let total_start = Instant::now();

    let c1_num_constraints = count_constraints::<U1>();
    let c2_num_constraints = count_constraints::<U2>();

    // ── Reconstruct Fiat-Shamir transcript ──────────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();
    let field_elem_size =
        <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Shared IC evaluation point ──────────────────────────────────
    let ic_evaluation_point: Vec<PiopField> =
        transcript.get_field_challenges(num_vars, &field_cfg);

    // ── IC₁ verify (BinaryPoly) ────────────────────────────────────
    let t0 = Instant::now();

    let c1_ic_combined: Vec<DynamicPolynomialF<PiopField>> = proof
        .ic1_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| {
                    field_from_bytes(
                        &bytes[i * field_elem_size..(i + 1) * field_elem_size],
                        &field_cfg,
                    )
                })
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let c1_ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: c1_ic_combined,
    };

    let c1_ic_subclaim =
        match IdealCheckProtocol::<PiopField>::verify_at_point::<U1, _, _>(
            &mut transcript,
            c1_ic_proof,
            c1_num_constraints,
            ic_evaluation_point.clone(),
            &ideal_from_ref1,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("C1 IdealCheck verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: t0.elapsed(),
                        combined_poly_resolver_verify: Duration::ZERO,
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

    // ── IC₂ verify (generic R) ─────────────────────────────────────
    let c2_ic_combined: Vec<DynamicPolynomialF<PiopField>> = proof
        .ic2_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| {
                    field_from_bytes(
                        &bytes[i * field_elem_size..(i + 1) * field_elem_size],
                        &field_cfg,
                    )
                })
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let c2_ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: c2_ic_combined,
    };

    let c2_ic_subclaim =
        match IdealCheckProtocol::<PiopField>::verify_at_point::<U2, _, _>(
            &mut transcript,
            c2_ic_proof,
            c2_num_constraints,
            ic_evaluation_point.clone(),
            &ideal_from_ref2,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("C2 IdealCheck verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: t0.elapsed(),
                        combined_poly_resolver_verify: Duration::ZERO,
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };
    let ic_verify_time = t0.elapsed();

    // ── Shared projecting element ───────────────────────────────────
    let t1 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

    // ── CPR₁ pre-sumcheck ───────────────────────────────────────────
    let c1_projected_scalars_coeffs = project_scalars::<PiopField, U1>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar
                .iter()
                .map(|coeff| {
                    if coeff.into_inner() {
                        one.clone()
                    } else {
                        zero.clone()
                    }
                })
                .collect::<Vec<_>>(),
        )
    });
    let c1_field_projected_scalars =
        project_scalars_to_field(c1_projected_scalars_coeffs, &projecting_element)
            .expect("C1 scalar projection failed");

    let c1_cpr_pre =
        match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U1>(
            &mut transcript,
            &field_from_bytes(&proof.md_claimed_sums[0], &field_cfg),
            c1_num_constraints,
            &projecting_element,
            &c1_field_projected_scalars,
            &c1_ic_subclaim,
            &field_cfg,
        ) {
            Ok(pre) => pre,
            Err(e) => {
                eprintln!("C1 CPR pre-sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

    // ── Lookup pre-sumcheck (circuit 1 only) ────────────────────────
    let mut lookup_pres = Vec::new();
    for (group_proof, meta) in proof
        .lookup_group_proofs
        .iter()
        .zip(proof.lookup_group_meta.iter())
    {
        let (subtable, shifts) = generate_table_and_shifts(
            &meta.table_type,
            &projecting_element,
            &field_cfg,
        );
        match BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
            &mut transcript,
            group_proof,
            &subtable,
            &shifts,
            meta.num_columns,
            meta.witness_len,
            &field_cfg,
        ) {
            Ok(pre) => lookup_pres.push(pre),
            Err(e) => {
                eprintln!("Lookup pre-sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        }
    }

    // ── CPR₂ pre-sumcheck ───────────────────────────────────────────
    let c2_projected_scalars_coeffs = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF {
            coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &field_cfg)],
        }
    });
    let c2_field_projected_scalars =
        project_scalars_to_field(c2_projected_scalars_coeffs, &projecting_element)
            .expect("C2 scalar projection failed");

    let c2_cpr_pre =
        match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U2>(
            &mut transcript,
            &field_from_bytes(&proof.md_claimed_sums[1], &field_cfg),
            c2_num_constraints,
            &projecting_element,
            &c2_field_projected_scalars,
            &c2_ic_subclaim,
            &field_cfg,
        ) {
            Ok(pre) => pre,
            Err(e) => {
                eprintln!("C2 CPR pre-sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

    // ── Compute shared_num_vars ─────────────────────────────────────
    let shared_num_vars = lookup_pres
        .iter()
        .map(|p| p.num_vars)
        .max()
        .map_or(num_vars, |max_lk| max_lk.max(num_vars));

    // ── Deserialize multi-degree sumcheck proof ─────────────────────
    let md_group_messages: Vec<Vec<ProverMsg<PiopField>>> = proof
        .md_group_messages
        .iter()
        .map(|group_rounds| {
            group_rounds
                .iter()
                .map(|bytes| {
                    let num_evals = bytes.len() / field_elem_size;
                    let tail_evaluations: Vec<PiopField> = (0..num_evals)
                        .map(|i| {
                            field_from_bytes(
                                &bytes[i * field_elem_size..(i + 1) * field_elem_size],
                                &field_cfg,
                            )
                        })
                        .collect();
                    ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
                })
                .collect()
        })
        .collect();

    let md_claimed_sums: Vec<PiopField> = proof
        .md_claimed_sums
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let md_proof_deserialized = MultiDegreeSumcheckProof {
        group_messages: md_group_messages,
        claimed_sums: md_claimed_sums,
        degrees: proof.md_degrees.clone(),
    };

    // ── Verify multi-degree sumcheck ────────────────────────────────
    let md_subclaims = match MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
        &mut transcript,
        shared_num_vars,
        &md_proof_deserialized,
        &field_cfg,
    ) {
        Ok(sc) => sc,
        Err(e) => {
            eprintln!("Multi-degree sumcheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Finalize CPR₁ ───────────────────────────────────────────────
    // Deserialize private (non-public) up_evals from the proof;
    // compute public column evaluations from c1_public_column_data.
    let c1_private_up_evals: Vec<PiopField> = proof
        .cpr1_up_evals
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let c1_down_evals: Vec<PiopField> = proof
        .cpr1_down_evals
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let c1_sig = U1::signature();
    let c1_up_evals = if c1_sig.public_columns.is_empty() {
        c1_private_up_evals
    } else {
        let binary_poly_projection =
            BinaryPoly::<D>::prepare_projection(&projecting_element);
        let c1_public_evals: Vec<PiopField> = {
            #[cfg(feature = "parallel")]
            let iter = c1_public_column_data.par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = c1_public_column_data.iter();
            iter.map(|col| {
                let mut mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                    col.iter()
                        .map(|bp| binary_poly_projection(bp).inner().clone())
                        .collect();
                let target_nv = md_subclaims.point.len();
                if target_nv > mle.num_vars {
                    mle.evaluations.resize(1 << target_nv, Default::default());
                    mle.num_vars = target_nv;
                }
                mle.evaluate_with_config(&md_subclaims.point, &field_cfg)
                    .expect("C1 public column MLE evaluation should succeed")
            })
            .collect()
        };
        reconstruct_up_evals(
            &c1_private_up_evals,
            &c1_public_evals,
            &c1_sig.public_columns,
            c1_sig.total_cols(),
        )
    };
    // Save for unified eval sumcheck (finalize_verifier consumes the Vecs).
    let c1_up_evals_saved = c1_up_evals.clone();

    let c1_cpr_subclaim =
        match CombinedPolyResolver::<PiopField>::finalize_verifier::<U1>(
            &mut transcript,
            md_subclaims.point.clone(),
            md_subclaims.expected_evaluations[0].clone(),
            &c1_cpr_pre,
            c1_up_evals,
            c1_down_evals,
            num_vars,
            &c1_field_projected_scalars,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("C1 CPR finalize_verifier failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

    // ── Finalize CPR₂ ───────────────────────────────────────────────
    // Deserialize private (non-public) up_evals from the proof;
    // compute public column evaluations from c2_public_column_data.
    let c2_private_up_evals: Vec<PiopField> = proof
        .cpr2_up_evals
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let c2_down_evals: Vec<PiopField> = proof
        .cpr2_down_evals
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let c2_sig = U2::signature();
    let c2_up_evals = if c2_sig.public_columns.is_empty() {
        c2_private_up_evals
    } else {
        let c2_public_evals: Vec<PiopField> = {
            #[cfg(feature = "parallel")]
            let iter = c2_public_column_data.par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = c2_public_column_data.iter();
            iter.map(|col| {
                let projected_col: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                    col.iter()
                        .map(|v| PiopField::from_with_cfg(v.clone(), &field_cfg).inner().clone())
                        .collect();
                projected_col.evaluate_with_config(&md_subclaims.point, &field_cfg)
                    .expect("C2 public column MLE evaluation should succeed")
            })
            .collect()
        };
        reconstruct_up_evals(
            &c2_private_up_evals,
            &c2_public_evals,
            &c2_sig.public_columns,
            c2_sig.total_cols(),
        )
    };
    // Save for unified eval sumcheck.
    let c2_up_evals_saved = c2_up_evals.clone();
    let c2_down_evals_saved = c2_down_evals.clone();

    if let Err(e) = CombinedPolyResolver::<PiopField>::finalize_verifier::<U2>(
        &mut transcript,
        md_subclaims.point.clone(),
        md_subclaims.expected_evaluations[1].clone(),
        &c2_cpr_pre,
        c2_up_evals,
        c2_down_evals,
        num_vars,
        &c2_field_projected_scalars,
        &field_cfg,
    ) {
        eprintln!("C2 CPR finalize_verifier failed: {e:?}");
        return VerifyResult {
            accepted: false,
            timing: VerifyTimingBreakdown {
                ideal_check_verify: ic_verify_time,
                combined_poly_resolver_verify: t1.elapsed(),
                lookup_verify: Duration::ZERO,
                pcs_verify: Duration::ZERO,
                total: total_start.elapsed(),
            },
        };
    }

    // ── Unified evaluation sumcheck verify ──────────────────────────
    if let Some(ref ss_proof_data) = proof.unified_eval_sumcheck {
        let c1_num_up = c1_up_evals_saved.len();
        let c2_num_up = c2_up_evals_saved.len();

        let mut unified_claims: Vec<ShiftClaim<PiopField>> =
            Vec::with_capacity(c1_num_up + c2_num_up + c2_sig.shifts.len());

        // Eq claims for circuit 1 columns.
        for i in 0..c1_num_up {
            unified_claims.push(ShiftClaim {
                source_col: i,
                shift_amount: 0,
                eval_point: c1_cpr_subclaim.evaluation_point.clone(),
                claimed_eval: c1_up_evals_saved[i].clone(),
            });
        }
        // Eq claims for circuit 2 columns.
        for j in 0..c2_num_up {
            unified_claims.push(ShiftClaim {
                source_col: c1_num_up + j,
                shift_amount: 0,
                eval_point: c1_cpr_subclaim.evaluation_point.clone(),
                claimed_eval: c2_up_evals_saved[j].clone(),
            });
        }
        // Shift claims for circuit 2's explicit ShiftSpecs.
        for (k, spec) in c2_sig.shifts.iter().enumerate() {
            unified_claims.push(ShiftClaim {
                source_col: c1_num_up + spec.source_col,
                shift_amount: spec.shift_amount,
                eval_point: c1_cpr_subclaim.evaluation_point.clone(),
                claimed_eval: c2_down_evals_saved[k].clone(),
            });
        }

        // Deserialize proof rounds.
        let rounds: Vec<ShiftRoundPoly<PiopField>> = ss_proof_data.rounds.iter().map(|bytes| {
            let n = bytes.len() / field_elem_size;
            assert_eq!(n, 3, "each unified eval round poly should have 3 evaluations");
            ShiftRoundPoly {
                evals: [
                    field_from_bytes(&bytes[0..field_elem_size], &field_cfg),
                    field_from_bytes(&bytes[field_elem_size..2 * field_elem_size], &field_cfg),
                    field_from_bytes(&bytes[2 * field_elem_size..3 * field_elem_size], &field_cfg),
                ],
            }
        }).collect();
        let ss_proof = ShiftSumcheckProof { rounds };

        // Determine which unified claims are public so we can
        // identify verifier-computed v_finals.
        let is_public_unified = |idx: usize| -> bool {
            if idx < c1_num_up {
                c1_sig.is_public_column(idx)
            } else if idx < c1_num_up + c2_num_up {
                c2_sig.is_public_column(idx - c1_num_up)
            } else {
                c2_sig.is_public_shift(idx - c1_num_up - c2_num_up)
            }
        };

        let has_public = (0..unified_claims.len()).any(&is_public_unified);

        if has_public {
            // Use split API: replay sumcheck to get challenge point s,
            // then compute public v_finals before finalizing.
            let ss_pre = match shift_sumcheck_verify_pre(
                &mut transcript, &ss_proof, &unified_claims, num_vars, &field_cfg,
            ) {
                Ok(pre) => pre,
                Err(e) => {
                    eprintln!("Unified eval sumcheck pre-verify failed: {e}");
                    return VerifyResult {
                        accepted: false,
                        timing: VerifyTimingBreakdown {
                            ideal_check_verify: ic_verify_time,
                            combined_poly_resolver_verify: t1.elapsed(),
                            lookup_verify: Duration::ZERO,
                            pcs_verify: Duration::ZERO,
                            total: total_start.elapsed(),
                        },
                    };
                }
            };

            // The challenge point is in BE (sumcheck convention);
            // MLE evaluate_with_config expects LE ordering.
            let challenge_point_le: Vec<PiopField> =
                ss_pre.challenge_point.iter().rev().cloned().collect();

            // Prepare BinaryPoly projection for circuit 1 public columns.
            let binary_poly_projection =
                BinaryPoly::<D>::prepare_projection(&projecting_element);

            // Compute v_finals for public columns; interleave with private.
            let private_v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
                .map(|b| field_from_bytes(b, &field_cfg))
                .collect();

            let total_claims = unified_claims.len();

            // Identify public indices and evaluate their MLEs (parallelizable).
            let public_indices: Vec<usize> = (0..total_claims)
                .filter(|&idx| is_public_unified(idx))
                .collect();

            let public_evals: Vec<PiopField> = {
                #[cfg(feature = "parallel")]
                let iter = public_indices.par_iter();
                #[cfg(not(feature = "parallel"))]
                let iter = public_indices.iter();
                iter.map(|&idx| {
                    if idx < c1_num_up {
                        // C1 public column.
                        let pcd_idx = c1_sig.public_columns.iter()
                            .position(|&c| c == idx)
                            .expect("C1 public column not found");
                        let col = &c1_public_column_data[pcd_idx];
                        let mle: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                            col.iter()
                                .map(|bp| binary_poly_projection(bp).inner().clone())
                                .collect();
                        mle.evaluate_with_config(&challenge_point_le, &field_cfg)
                            .expect("C1 public v_final MLE eval failed")
                    } else if idx < c1_num_up + c2_num_up {
                        // C2 public column (eq claim).
                        let col_idx = idx - c1_num_up;
                        let pcd_idx = c2_sig.public_columns.iter()
                            .position(|&c| c == col_idx)
                            .expect("C2 public column not found");
                        let col = &c2_public_column_data[pcd_idx];
                        let projected_col: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                            col.iter()
                                .map(|v| PiopField::from_with_cfg(v.clone(), &field_cfg).inner().clone())
                                .collect();
                        projected_col.evaluate_with_config(&challenge_point_le, &field_cfg)
                            .expect("C2 public v_final MLE eval failed")
                    } else {
                        // C2 public shift source.
                        let shift_idx = idx - c1_num_up - c2_num_up;
                        let spec = &c2_sig.shifts[shift_idx];
                        let pcd_idx = c2_sig.public_columns.iter()
                            .position(|&c| c == spec.source_col)
                            .expect("C2 public shift source not found in public_columns");
                        let col = &c2_public_column_data[pcd_idx];
                        let projected_col: DenseMultilinearExtension<<PiopField as Field>::Inner> =
                            col.iter()
                                .map(|v| PiopField::from_with_cfg(v.clone(), &field_cfg).inner().clone())
                                .collect();
                        projected_col.evaluate_with_config(&challenge_point_le, &field_cfg)
                            .expect("C2 public shift v_final MLE eval failed")
                    }
                })
                .collect()
            };

            // Interleave public and private v_finals in order.
            let mut full_v_finals = Vec::with_capacity(total_claims);
            let mut priv_idx = 0usize;
            let mut pub_idx = 0usize;
            for idx in 0..total_claims {
                if is_public_unified(idx) {
                    full_v_finals.push(public_evals[pub_idx].clone());
                    pub_idx += 1;
                } else {
                    full_v_finals.push(private_v_finals[priv_idx].clone());
                    priv_idx += 1;
                }
            }
            debug_assert_eq!(priv_idx, private_v_finals.len());
            debug_assert_eq!(pub_idx, public_evals.len());

            if let Err(e) = shift_sumcheck_verify_finalize(
                &mut transcript, &ss_pre, &unified_claims, &full_v_finals, &field_cfg,
            ) {
                eprintln!("Unified eval sumcheck finalize failed: {e}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        } else {
            // No public claims: use monolithic verify with all v_finals from proof.
            let v_finals: Vec<PiopField> = ss_proof_data.v_finals.iter()
                .map(|b| field_from_bytes(b, &field_cfg))
                .collect();

            if let Err(e) = shift_sumcheck_verify(
                &mut transcript,
                &ss_proof,
                &unified_claims,
                &v_finals,
                num_vars,
                &field_cfg,
            ) {
                eprintln!("Unified evaluation sumcheck verification failed: {e}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        }
    }

    // ── Finalize lookup groups ──────────────────────────────────────
    for (g, ((lk_pre, group_proof), meta)) in lookup_pres
        .iter()
        .zip(proof.lookup_group_proofs.iter())
        .zip(proof.lookup_group_meta.iter())
        .enumerate()
    {
        // Compute eq_sum_w for domain-scaling of affine constant offsets.
        let eq_sum_w = {
            let one = PiopField::one_with_cfg(&field_cfg);
            let w_nv = zinc_utils::log2(meta.witness_len.next_power_of_two()) as usize;
            let mut prod = one.clone();
            for i in w_nv..md_subclaims.point.len() {
                prod *= one.clone() - &md_subclaims.point[i];
            }
            prod
        };

        // Collect parent column evaluations from C1 CPR up_evals.
        let parent_evals: Vec<PiopField> = meta.witness_sources.iter()
            .map(|ws| match ws {
                LookupWitnessSource::Column { column_index } =>
                    c1_up_evals_saved[*column_index].clone(),
                LookupWitnessSource::Affine { terms, constant_offset_bits } =>
                    eval_affine_parent::<D>(
                        terms, *constant_offset_bits,
                        &c1_up_evals_saved,
                        &projecting_element, &eq_sum_w,
                        &field_cfg,
                    ),
            })
            .collect();
        if let Err(e) = BatchedDecompLogupProtocol::<PiopField>::finalize_verifier(
            lk_pre,
            group_proof,
            &md_subclaims.point,
            &md_subclaims.expected_evaluations[g + 2],
            &parent_evals,
            &field_cfg,
        ) {
            eprintln!("Lookup finalize_verifier failed (group {g}): {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    }
    let cpr_verify_time = t1.elapsed();

    // ── PCS Verify (both circuits, parallelized) ─────────────────
    let t2 = Instant::now();

    // r_PCS = r_CPR: both circuits share the same evaluation point.
    let verify_pcs1 = || {
        let pcs1_field_cfg = KeccakTranscript::default()
            .get_random_field_cfg::<PiopField, Zt1::Fmod, Zt1::PrimeTest>();
        let point1_f: Vec<PiopField> = c1_cpr_subclaim.evaluation_point[..num_vars].to_vec();
        let eval1_f: PiopField = PiopField::new_unchecked_with_cfg(
            <PiopField as Field>::Inner::read_transcription_bytes(&proof.pcs1_evals_bytes[0]),
            &pcs1_field_cfg,
        );
        let pcs1_transcript = zip_plus::pcs_transcript::PcsTranscript {
            fs_transcript: KeccakTranscript::default(),
            stream: std::io::Cursor::new(proof.pcs1_proof_bytes.clone()),
        };
        let pcs1_proof: ZipPlusProof = pcs1_transcript.into();

        let result = ZipPlus::<Zt1, Lc1>::verify::<PiopField, CHECK>(
            params1,
            &proof.pcs1_commitment,
            &point1_f,
            &eval1_f,
            &pcs1_proof,
        );
        if let Err(ref e) = result {
            eprintln!("PCS1 verification failed: {e:?}");
        }
        result
    };

    let verify_pcs2 = || {
        let pcs2_field_cfg = KeccakTranscript::default()
            .get_random_field_cfg::<PcsF2, Zt2::Fmod, Zt2::PrimeTest>();
        let point2_f: Vec<PcsF2> = piop_point_to_pcs_field(&c1_cpr_subclaim.evaluation_point[..num_vars], &pcs2_field_cfg);
        let eval2_f: PcsF2 = PcsF2::new_unchecked_with_cfg(
            <PcsF2::Inner as ConstTranscribable>::read_transcription_bytes(&proof.pcs2_evals_bytes[0]),
            &pcs2_field_cfg,
        );
        let pcs2_transcript = zip_plus::pcs_transcript::PcsTranscript {
            fs_transcript: KeccakTranscript::default(),
            stream: std::io::Cursor::new(proof.pcs2_proof_bytes.clone()),
        };
        let pcs2_proof: ZipPlusProof = pcs2_transcript.into();

        let result = ZipPlus::<Zt2, Lc2>::verify::<PcsF2, CHECK>(
            params2,
            &proof.pcs2_commitment,
            &point2_f,
            &eval2_f,
            &pcs2_proof,
        );
        if let Err(ref e) = result {
            eprintln!("PCS2 verification failed: {e:?}");
        }
        result
    };

    #[cfg(feature = "parallel")]
    let (pcs1_result, pcs2_result) = rayon_join(verify_pcs1, verify_pcs2);
    #[cfg(not(feature = "parallel"))]
    let (pcs1_result, pcs2_result) = (verify_pcs1(), verify_pcs2());

    let pcs_verify_time = t2.elapsed();

    VerifyResult {
        accepted: pcs1_result.is_ok() && pcs2_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: ic_verify_time,
            combined_poly_resolver_verify: cpr_verify_time,
            lookup_verify: Duration::ZERO, // included in CPR
            pcs_verify: pcs_verify_time,
            total: total_start.elapsed(),
        },
    }
}