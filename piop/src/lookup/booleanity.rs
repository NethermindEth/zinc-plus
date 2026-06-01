//! Algebraic booleanity check for binary_poly columns.
//!
//! Proves that each entry of a `binary_poly` witness column `v` is a
//! binary polynomial of degree `< D` by writing `v = Σ_{i=0}^{D-1} X^i ·
//! v_i` for bit-slice MLEs `v_i` over `F`, and running a zerocheck on
//! `Σ_k α^k · v_k(b) · (v_k(b) - 1) · eq(r, b)` using a dedicated
//! degree-3 group inside the protocol's multi-degree sumcheck.
//!
//! Running this as a *separate* group (not folded into the CPR group)
//! avoids paying the CPR's higher per-variable degree (`max_degree + 2`)
//! on the booleanity term — the booleanity-only group is degree 3, so
//! its `comb_fn` is invoked at 4 evaluation points per round instead of
//! `max_degree + 3`. For SHA-style UAIRs (max_degree ≥ 6) with hundreds
//! of bit-slice MLEs, this is a 2–2.5× saving on step 4 alone.

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField, semiring::boolean::Boolean};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::slice;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::binary::BinaryPoly,
    utils::{ArithErrors, build_eq_x_r_inner, build_eq_x_r_vec, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    ShiftedBitSliceSpec, VirtualBinaryPolySource, VirtualBinaryPolySpec, VirtualBoolSource,
    VirtualBoolSpec,
};
use zinc_utils::{
    cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField, powers,
};

/// Build the F::Inner-valued shifted bit-slice MLEs for each
/// `ShiftedBitSliceSpec`, in flat layout `spec*D + bit`. Each MLE has
/// `evaluations[t] = bit_idx-th bit of trace.binary_poly[col][t + shift]`,
/// zero-padded past the trace tail (matching the protocol's shift
/// convention used by `ShiftSpec` / down columns).
#[allow(clippy::arithmetic_side_effects)]
pub fn build_shifted_bit_slice_mles<F, const D: usize>(
    trace_witness_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    shifted_specs: &[ShiftedBitSliceSpec],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: InnerTransparentField,
    F::Inner: Clone + Send + Sync,
    F::Config: Sync,
{
    let zero_inner = F::zero_with_cfg(field_cfg).into_inner();
    let one_inner = F::one_with_cfg(field_cfg).into_inner();

    cfg_iter!(shifted_specs)
        .flat_map(|spec| {
            let col = &trace_witness_binary_poly[spec.witness_col_idx];
            let shift = spec.shift_amount;
            let n = col.evaluations.len();
            let num_vars = col.num_vars;

            let mut bit_evals: Vec<Vec<F::Inner>> =
                (0..D).map(|_| vec![zero_inner.clone(); n]).collect();
            for t in 0..n {
                let src_t = t.checked_add(shift).filter(|&v| v < n);
                if let Some(s) = src_t {
                    let bp = &col.evaluations[s];
                    for (bit_idx, coeff) in bp.iter().enumerate() {
                        if coeff.into_inner() {
                            bit_evals[bit_idx][t] = one_inner.clone();
                        }
                    }
                }
            }
            bit_evals
                .into_iter()
                .map(move |evaluations| DenseMultilinearExtension {
                    evaluations,
                    num_vars,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Streaming evaluator for shifted bit-slice MLEs at the shared
/// sumcheck point `r*`. Equivalent to
/// `build_shifted_bit_slice_mles(...).iter().map(evaluate_at(r*))`,
/// but skips materializing the `num_shifted_specs · D` F::Inner-valued
/// MLE buffers (~`num_shifted · D · n` F::Inner allocations). Builds
/// the size-`n` `eq(r*, ·)` table once and accumulates per-bit sums
/// directly from the `BinaryPoly<D>` trace columns in a single pass
/// per spec (t outer, bits inner — avoids `iter().nth(bit_idx)`'s
/// linear cost on custom binary-poly iterators).
///
/// Used when the prover doesn't otherwise need the materialized bit-
/// slice MLEs (i.e. when no `VirtualBoolSpec` is registered — the
/// `VirtualBinaryPolySpec` path reads source binary_polys directly).
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_shifted_bit_slice_evals_streaming<F, const D: usize>(
    trace_witness_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    shifted_specs: &[ShiftedBitSliceSpec],
    point: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<F>, ArithErrors>
where
    F: PrimeField + Send + Sync,
    F::Config: Sync,
{
    if shifted_specs.is_empty() {
        return Ok(Vec::new());
    }
    // Single shared eq table — one O(n) pass + O(n) memory across all
    // (spec, bit) sums.
    let eq_table = build_eq_x_r_vec(point, field_cfg)?;

    let zero = F::zero_with_cfg(field_cfg);
    let out: Vec<F> = cfg_iter!(shifted_specs)
        .flat_map(|spec| {
            let col = &trace_witness_binary_poly[spec.witness_col_idx];
            let shift = spec.shift_amount;
            let n = col.evaluations.len();
            // Per-bit accumulators, one F-element each.
            let mut accs: Vec<F> = vec![zero.clone(); D];
            for t in 0..n {
                let src_t = t.checked_add(shift).filter(|&v| v < n);
                if let Some(s) = src_t {
                    let bp = &col.evaluations[s];
                    let eq_t = &eq_table[t];
                    // Walk bits in their stored order; the iterator
                    // visits each coefficient once in O(D).
                    for (bit_idx, coeff) in bp.iter().enumerate() {
                        if coeff.into_inner() {
                            accs[bit_idx] = accs[bit_idx].clone() + eq_t.clone();
                        }
                    }
                }
            }
            accs
        })
        .collect();
    Ok(out)
}

/// Compute `coeff * value` in F::Inner using only `add_inner` /
/// `sub_inner`. Coefficients are small in practice (Maj uses ±1, ±2).
///
/// Generic fallback used by [`compute_virtual_closing_overrides`] (one
/// call per spec, not per row, so the per-call overhead is fine). The
/// per-row hot path in [`build_virtual_booleanity_mles`] inlines the
/// ±1 / ±2 cases instead of going through this function.
#[allow(clippy::arithmetic_side_effects)]
fn apply_coeff_inner<F: InnerTransparentField>(
    coeff: i64,
    value: &F::Inner,
    field_cfg: &F::Config,
) -> F::Inner
where
    F::Inner: Clone,
{
    let zero = F::zero_with_cfg(field_cfg).into_inner();
    let abs = coeff.unsigned_abs();
    let mut acc = zero.clone();
    for _ in 0..abs {
        acc = F::add_inner(&acc, value, field_cfg);
    }
    if coeff < 0 {
        F::sub_inner(&zero, &acc, field_cfg)
    } else {
        acc
    }
}

/// Build F::Inner-valued virtual booleanity MLEs from `VirtualBoolSpec`s.
///
/// `self_bit_slices` is the flat (`witness_col*D + bit`) bit decomposition
/// of the witness binary_poly cols; `int_witness_cols` is the per-row
/// F::Inner values of witness int cols (length `num_witness_int`,
/// indexed by witness-relative col).
///
/// The returned MLEs are in spec order; each MLE has `num_vars` matching
/// the trace and `evaluations[t] = Σ_j coeff_j · v_j(t)` per the spec.
#[allow(clippy::arithmetic_side_effects)]
pub fn build_virtual_booleanity_mles<F, const D: usize>(
    self_bit_slices: &[DenseMultilinearExtension<F::Inner>],
    shifted_bit_slice_mles: &[DenseMultilinearExtension<F::Inner>],
    public_bit_slices: &[DenseMultilinearExtension<F::Inner>],
    int_witness_cols: &[DenseMultilinearExtension<F::Inner>],
    virtual_specs: &[VirtualBoolSpec],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: InnerTransparentField,
    F::Inner: Clone + Send + Sync,
    F::Config: Sync,
{
    if virtual_specs.is_empty() {
        return Vec::new();
    }
    let template = self_bit_slices
        .first()
        .or_else(|| shifted_bit_slice_mles.first())
        .or_else(|| public_bit_slices.first())
        .or_else(|| int_witness_cols.first())
        .expect("virtual booleanity needs at least one source MLE");
    let n = template.evaluations.len();
    let num_vars = template.num_vars;
    let zero = F::zero_with_cfg(field_cfg).into_inner();

    cfg_iter!(virtual_specs)
        .map(|spec| {
            let mut evals: Vec<F::Inner> = vec![zero.clone(); n];
            for (coeff, source) in &spec.terms {
                // Hoist source-slice resolution outside the t loop —
                // `source` is constant across rows.
                let src: &[F::Inner] = match source {
                    VirtualBoolSource::SelfBitSlice {
                        witness_col_idx,
                        bit_idx,
                    } => &self_bit_slices[*witness_col_idx * D + *bit_idx]
                        .evaluations,
                    VirtualBoolSource::ShiftedBitSlice {
                        shifted_spec_idx,
                        bit_idx,
                    } => &shifted_bit_slice_mles
                        [*shifted_spec_idx * D + *bit_idx]
                        .evaluations,
                    VirtualBoolSource::PublicBitSlice {
                        public_col_idx,
                        bit_idx,
                    } => &public_bit_slices[*public_col_idx * D + *bit_idx]
                        .evaluations,
                    VirtualBoolSource::IntCol { witness_col_idx } => {
                        &int_witness_cols[*witness_col_idx].evaluations
                    }
                };
                debug_assert_eq!(src.len(), n);
                // Specialize the inner loop on |coeff|. Maj uses only
                // ±1 / ±2; the generic fallback never fires for SHA but
                // keeps the function general.
                match *coeff {
                    1 => {
                        for t in 0..n {
                            evals[t] =
                                F::add_inner(&evals[t], &src[t], field_cfg);
                        }
                    }
                    -1 => {
                        for t in 0..n {
                            evals[t] =
                                F::sub_inner(&evals[t], &src[t], field_cfg);
                        }
                    }
                    2 => {
                        for t in 0..n {
                            evals[t] =
                                F::add_inner(&evals[t], &src[t], field_cfg);
                            evals[t] =
                                F::add_inner(&evals[t], &src[t], field_cfg);
                        }
                    }
                    -2 => {
                        for t in 0..n {
                            evals[t] =
                                F::sub_inner(&evals[t], &src[t], field_cfg);
                            evals[t] =
                                F::sub_inner(&evals[t], &src[t], field_cfg);
                        }
                    }
                    c => {
                        for t in 0..n {
                            let term =
                                apply_coeff_inner::<F>(c, &src[t], field_cfg);
                            evals[t] =
                                F::add_inner(&evals[t], &term, field_cfg);
                        }
                    }
                }
            }
            DenseMultilinearExtension { evaluations: evals, num_vars }
        })
        .collect()
}

/// Verifier-side reconstruction of each virtual booleanity MLE's eval
/// at `r*`. Each output[s] = `Σ_j coeff_j · v_j` where `v_j` is looked
/// up in the relevant eval slice based on the spec's source kind.
/// Used as the `closing_overrides_tail` suffix when calling
/// `finalize_booleanity_verifier`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_virtual_closing_overrides<F, const D: usize>(
    virtual_specs: &[VirtualBoolSpec],
    self_bit_slice_evals: &[F],
    shifted_bit_slice_evals: &[F],
    public_bit_slice_evals: &[F],
    int_witness_up_evals: &[F],
    field_cfg: &F::Config,
) -> Vec<F>
where
    F: PrimeField,
{
    let zero = F::zero_with_cfg(field_cfg);
    virtual_specs
        .iter()
        .map(|spec| {
            let mut acc = zero.clone();
            for (coeff, source) in &spec.terms {
                let v: &F = match source {
                    VirtualBoolSource::SelfBitSlice {
                        witness_col_idx,
                        bit_idx,
                    } => &self_bit_slice_evals[*witness_col_idx * D + *bit_idx],
                    VirtualBoolSource::ShiftedBitSlice {
                        shifted_spec_idx,
                        bit_idx,
                    } => &shifted_bit_slice_evals
                        [*shifted_spec_idx * D + *bit_idx],
                    VirtualBoolSource::PublicBitSlice {
                        public_col_idx,
                        bit_idx,
                    } => &public_bit_slice_evals
                        [*public_col_idx * D + *bit_idx],
                    VirtualBoolSource::IntCol { witness_col_idx } => {
                        &int_witness_up_evals[*witness_col_idx]
                    }
                };
                let abs = coeff.unsigned_abs();
                let mut term = zero.clone();
                for _ in 0..abs {
                    term = term + v.clone();
                }
                if *coeff < 0 {
                    acc = acc - term;
                } else {
                    acc = acc + term;
                }
            }
            acc
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Packed virtual binary_poly columns
// ---------------------------------------------------------------------------

/// Read the bit pattern of a `BinaryPoly<D>` as a `u64` (low D bits set
/// according to the polynomial's coefficients). Generic across the
/// `binary_ref` / `binary_u64` backends — both expose `.iter()` over
/// `Boolean` coefficients.
#[allow(clippy::arithmetic_side_effects)]
fn binary_poly_to_u64<const D: usize>(bp: &BinaryPoly<D>) -> u64 {
    let mut out: u64 = 0;
    for (i, b) in bp.iter().enumerate() {
        if b.into_inner() {
            out |= 1u64 << i;
        }
    }
    out
}

/// Look up a virtual-binary-poly source's row-`t` value as a `u64` bit
/// pattern (low `D` bits). Off-trace shifts return `0`, matching the
/// `ShiftSpec` / down-column convention.
#[allow(clippy::arithmetic_side_effects)]
fn source_u_at<const D: usize>(
    source: &VirtualBinaryPolySource,
    t: usize,
    self_witness_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    public_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    shifted_specs: &[ShiftedBitSliceSpec],
) -> u64 {
    match source {
        VirtualBinaryPolySource::SelfWitnessCol { witness_col_idx } => {
            let bp = &self_witness_binary_poly[*witness_col_idx].evaluations[t];
            binary_poly_to_u64(bp)
        }
        VirtualBinaryPolySource::ShiftedWitnessCol { shifted_spec_idx } => {
            let spec = &shifted_specs[*shifted_spec_idx];
            let col = &self_witness_binary_poly[spec.witness_col_idx];
            let n = col.evaluations.len();
            let src_t = t.checked_add(spec.shift_amount).filter(|&v| v < n);
            if let Some(s) = src_t {
                binary_poly_to_u64(&col.evaluations[s])
            } else {
                0
            }
        }
        VirtualBinaryPolySource::PublicCol { public_col_idx } => {
            let bp = &public_binary_poly[*public_col_idx].evaluations[t];
            binary_poly_to_u64(bp)
        }
    }
}

/// Build packed virtual binary_poly MLEs from `VirtualBinaryPolySpec`s.
///
/// Per row `t`, per bit `i ∈ [0, D)`, the output bit is the LSB of
/// `Σ_j coeff_j · source_j[t].bit(i)`. Coefficients with even absolute
/// value contribute zero mod 2; coefficients with odd absolute value
/// contribute their source's bit pattern. So per row, the packed
/// output is `XOR over odd-|coeff| sources` (per-bit XOR ≡ scalar
/// XOR over u64 bit patterns), giving us one tight per-row inner
/// loop instead of `D × terms` per-bit inner work.
///
/// **Soundness:** the materialization assumes the spec residual is
/// bit-valued at every (t, i). For dishonest provers where the residual
/// lands outside `{0, 1}`, the verifier's per-bit closing override
/// (computed via [`compute_virtual_binary_poly_closing_overrides`])
/// will not match the booleanity polynomial check on the override —
/// concretely, `(override)² − override ≠ 0` when the residual is not
/// in `{0, 1}` — so the booleanity check rejects.
#[allow(clippy::arithmetic_side_effects)]
pub fn build_virtual_binary_poly_mles<const D: usize>(
    self_witness_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    public_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    shifted_specs: &[ShiftedBitSliceSpec],
    virtual_specs: &[VirtualBinaryPolySpec],
) -> Vec<DenseMultilinearExtension<BinaryPoly<D>>>
where
    BinaryPoly<D>: Send + Sync + Default + Clone,
{
    if virtual_specs.is_empty() {
        return Vec::new();
    }
    let template = self_witness_binary_poly
        .first()
        .or_else(|| public_binary_poly.first())
        .expect("virtual binary_poly needs at least one source binary_poly");
    let n = template.evaluations.len();
    let num_vars = template.num_vars;
    debug_assert!(D <= 64, "build_virtual_binary_poly_mles assumes D <= 64");

    cfg_iter!(virtual_specs)
        .map(|spec| {
            let mut evaluations: Vec<BinaryPoly<D>> = Vec::with_capacity(n);
            // Reusable scratch for the per-row Boolean coeffs.
            let mut coeffs: Vec<Boolean> = vec![Boolean::default(); D];
            for t in 0..n {
                let mut packed_lsb: u64 = 0;
                for (coeff, source) in &spec.terms {
                    if coeff.unsigned_abs() % 2 == 1 {
                        let src_u = source_u_at::<D>(
                            source,
                            t,
                            self_witness_binary_poly,
                            public_binary_poly,
                            shifted_specs,
                        );
                        packed_lsb ^= src_u;
                    }
                }
                for (i, slot) in coeffs.iter_mut().enumerate() {
                    *slot = Boolean::new((packed_lsb >> i) & 1 == 1);
                }
                evaluations.push(BinaryPoly::<D>::new(coeffs.as_slice()));
            }
            DenseMultilinearExtension { evaluations, num_vars }
        })
        .collect()
}

/// Verifier-side per-bit closing overrides for virtual binary_poly cols.
///
/// Output layout: flat `spec * D + bit_idx`, i.e. `D` consecutive entries
/// per spec. Each entry equals `Σ_j coeff_j · source_bit_eval_j[bit_idx]`
/// (in `F`), reconstructed from already-bound per-bit eval slices.
///
/// Used as a *prefix* of `closing_overrides_tail` when calling
/// `finalize_booleanity_verifier`: virtual binary_poly per-bit slices
/// occupy positions `(K_genuine + virtual_idx) * D + bit_idx` of
/// `bit_slice_evals`, immediately preceding the existing
/// `extra_bit_cols` entries.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_virtual_binary_poly_closing_overrides<F, const D: usize>(
    virtual_specs: &[VirtualBinaryPolySpec],
    self_witness_bit_slice_evals: &[F],
    shifted_bit_slice_evals: &[F],
    public_bit_slice_evals: &[F],
    field_cfg: &F::Config,
) -> Vec<F>
where
    F: PrimeField,
{
    let zero = F::zero_with_cfg(field_cfg);
    let mut out = Vec::with_capacity(virtual_specs.len() * D);
    for spec in virtual_specs {
        for i in 0..D {
            let mut acc = zero.clone();
            for (coeff, source) in &spec.terms {
                let v: &F = match source {
                    VirtualBinaryPolySource::SelfWitnessCol { witness_col_idx } => {
                        &self_witness_bit_slice_evals[*witness_col_idx * D + i]
                    }
                    VirtualBinaryPolySource::ShiftedWitnessCol { shifted_spec_idx } => {
                        &shifted_bit_slice_evals[*shifted_spec_idx * D + i]
                    }
                    VirtualBinaryPolySource::PublicCol { public_col_idx } => {
                        &public_bit_slice_evals[*public_col_idx * D + i]
                    }
                };
                let abs = coeff.unsigned_abs();
                let mut term = zero.clone();
                for _ in 0..abs {
                    term = term + v.clone();
                }
                if *coeff < 0 {
                    acc = acc - term;
                } else {
                    acc = acc + term;
                }
            }
            out.push(acc);
        }
    }
    out
}

use crate::{
    CombFn,
    sumcheck::{
        multi_degree::{MultiDegreeSumcheckGroup, Round1FastPath, Round1Output},
        prover::ProverState as SumcheckProverState,
    },
};

/// Build bit-slice MLEs over `F::Inner` for every binary_poly column.
///
/// Output ordering is **flat, column-major-then-bit-major**: index
/// `col_idx * D + bit_idx` is `MLE<F::Inner>` whose evaluations are the
/// `bit_idx`-th bit of each row's `BinaryPoly<D>` cast to 0/1 in `F::Inner`.
///
/// Length: `trace_binary_poly.len() * D`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_bit_slices_flat<F, const D: usize>(
    trace_binary_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    field_cfg: &F::Config,
) -> Vec<DenseMultilinearExtension<F::Inner>>
where
    F: InnerTransparentField,
    F::Inner: Clone + Send + Sync,
{
    let zero_inner = F::zero_with_cfg(field_cfg).into_inner();
    let one_inner = F::one_with_cfg(field_cfg).into_inner();

    cfg_iter!(trace_binary_poly)
        .flat_map(|col| {
            let num_vars = col.num_vars;
            // Per-column transpose: bit_evals[bit_idx][row_idx] = bit
            let mut bit_evals: Vec<Vec<F::Inner>> = (0..D)
                .map(|_| Vec::with_capacity(col.evaluations.len()))
                .collect();
            for bp in &col.evaluations {
                for (bit_idx, coeff) in bp.iter().enumerate() {
                    bit_evals[bit_idx].push(if coeff.into_inner() {
                        one_inner.clone()
                    } else {
                        zero_inner.clone()
                    });
                }
            }
            bit_evals
                .into_iter()
                .map(move |evaluations| DenseMultilinearExtension {
                    evaluations,
                    num_vars,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Round-1 fast path
// ---------------------------------------------------------------------------

/// Round-1 fast path for the booleanity zerocheck.
///
/// Pre-fold, every bit-slice value `v_i^{(k)}(b)` lies in `{0, 1} ⊂ F`.
/// In all four `(A, B) ∈ {0, 1}²` cases, the linear interpolation
/// `v(X, b') = (1 − X) A + X B` satisfies
///
/// ```text
/// v(X, b') · (v(X, b') − 1) = (A ⊕ B) · X(X − 1)
/// ```
///
/// (Cases (0, 0) and (1, 1): the polynomial is identically zero. Cases
/// (0, 1) and (1, 0): both expand to `X(X − 1)`.) The eq factor splits
/// as `eq_r*(X, b') = e_0(X, ic_ep_0) · E_other(b')`, so the round-1
/// polynomial collapses to
///
/// ```text
/// p_1(X) = e_0(X, ic_ep_0) · X(X − 1) · T_1
/// T_1 = Σ_{b'} S(b') · E_other(b')
/// S(b') = Σ_{(k, i)} α^{k·D + i} · (A_{k,i,b'} ⊕ B_{k,i,b'})
/// ```
///
/// The prover sends `[p_1(1), p_1(2), p_1(3)]`. `p_1(1) = 0` because
/// `X(X − 1)` vanishes at X = 1. `p_1(0) = 0` (also vanishes), so
/// the asserted sum `p_1(0) + p_1(1) = 0`. Computing `T_1` is one pass
/// over `2^(num_vars − 1)` row pairs, doing `K · D` bit-XOR-tests per
/// row pair — no F-multiplications inside the inner loop.
pub struct BooleanityRound1FastPath<F: PrimeField, const D: usize> {
    /// Owned clones of the binary_poly trace columns. Read directly to
    /// avoid materializing F-valued bit-slice MLEs at full size.
    binary_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
    /// Owned F::Inner-valued MLEs for additional "single-bit" columns
    /// (e.g. int trace columns each holding a `{0, 1}` value per row).
    /// Each element is treated as one extra bit slice — booleanity
    /// asserts every entry lies in `{0, 1}`.
    extra_bit_cols: Vec<DenseMultilinearExtension<F::Inner>>,
    /// `[1, α, α², ..., α^{K·D + E − 1}]` where E = `extra_bit_cols.len()`.
    /// Layout: binary cols first (col-major-then-bit-major over D), then
    /// extra bit cols in their input order.
    alpha_powers: Vec<F>,
    /// `eq(b', ic_evaluation_point[1..])` for `b' ∈ {0, 1}^(num_vars − 1)`.
    eq_other_table: Vec<F::Inner>,
    /// `ic_evaluation_point[0]`.
    ic_ep_0: F,
    /// Number of variables of the parent sumcheck.
    num_vars: usize,
}

impl<F, const D: usize> Round1FastPath<F> for BooleanityRound1FastPath<F, D>
where
    F: InnerTransparentField + Send + Sync + 'static,
    F::Inner: Send + Sync + Zero + Default,
{
    fn round_1_message(&self, config: &F::Config) -> Round1Output<F> {
        let extra_offset = self.binary_cols.len() * D;
        debug_assert_eq!(
            self.alpha_powers.len(),
            extra_offset + self.extra_bit_cols.len(),
            "alpha_powers length must match K·D + E"
        );
        debug_assert_eq!(
            self.eq_other_table.len(),
            1usize << self.num_vars.saturating_sub(1),
            "eq_other_table size must be 2^(num_vars - 1)"
        );

        let zero = F::zero_with_cfg(config);
        let one = F::one_with_cfg(config);
        let half = 1usize << (self.num_vars - 1);

        // T_1 = Σ_{b'} S(b') · E_other(b'). Compute S(b') row pair by row
        // pair: for each binary_poly column, a single XOR of the two adjacent
        // BinaryPoly rows yields all D bit differences in one shot; iterate
        // the bits and conditionally fold α-powers in. For each extra bit
        // column (one bit per row, F::Inner-valued and assumed `{0, 1}`),
        // a single inequality test toggles its α power.
        let t1: F = cfg_into_iter!(0..half)
            .map(|b_prime| {
                let mut s_b: F = zero.clone();
                for (k, col) in self.binary_cols.iter().enumerate() {
                    let row_a = &col.evaluations[2 * b_prime];
                    let row_b = &col.evaluations[2 * b_prime + 1];
                    for (i, (a_bit, b_bit)) in row_a.iter().zip(row_b.iter()).enumerate() {
                        if a_bit.into_inner() != b_bit.into_inner() {
                            s_b = s_b + self.alpha_powers[k * D + i].clone();
                        }
                    }
                }
                for (j, col) in self.extra_bit_cols.iter().enumerate() {
                    let a = &col.evaluations[2 * b_prime];
                    let b = &col.evaluations[2 * b_prime + 1];
                    // `a` and `b` are in `{0, 1}` (F::Inner). XOR via
                    // `is_zero` distinguishes the four cases without
                    // needing F::Inner: PartialEq.
                    if a.is_zero() != b.is_zero() {
                        s_b = s_b + self.alpha_powers[extra_offset + j].clone();
                    }
                }
                let e_other =
                    F::new_unchecked_with_cfg(self.eq_other_table[b_prime].clone(), config);
                s_b * e_other
            })
            .reduce_with_or_default(zero.clone());

        // p_1(2) = 2 · (3·ic_ep_0 − 1) · T_1
        // p_1(3) = 6 · (5·ic_ep_0 − 2) · T_1
        let two = one.clone() + one.clone();
        let three = two.clone() + one.clone();
        let five = three.clone() + two.clone();
        let six = three.clone() + three.clone();

        let coeff_2 = three * self.ic_ep_0.clone() - one.clone();
        let coeff_3 = five * self.ic_ep_0.clone() - two.clone();

        let p1_at_1 = zero.clone();
        let p1_at_2 = (one.clone() + one.clone()) * coeff_2 * t1.clone();
        let p1_at_3 = six * coeff_3 * t1;

        Round1Output {
            asserted_sum: zero,
            tail_evaluations: vec![p1_at_1, p1_at_2, p1_at_3],
        }
    }

    fn fold_with_r1(
        self: Box<Self>,
        r_1: &F,
        config: &F::Config,
    ) -> Vec<DenseMultilinearExtension<F::Inner>> {
        let one = F::one_with_cfg(config);
        let one_minus_r1 = one.clone() - r_1.clone();
        let r1 = r_1.clone();
        let half = 1usize << (self.num_vars - 1);

        // eq_r_folded(b') = ((1 − r_1)(1 − ic_ep_0) + r_1 · ic_ep_0) · E_other(b').
        let eq_scalar = one_minus_r1.clone() * (one.clone() - self.ic_ep_0.clone())
            + r1.clone() * self.ic_ep_0.clone();
        let eq_folded_evals: Vec<F::Inner> = cfg_iter!(self.eq_other_table)
            .map(|e| {
                let lifted = F::new_unchecked_with_cfg(e.clone(), config);
                (eq_scalar.clone() * lifted).into_inner()
            })
            .collect();
        let eq_folded = DenseMultilinearExtension {
            num_vars: self.num_vars - 1,
            evaluations: eq_folded_evals,
        };

        // For each (k, i) bit-slice: fold over variable 0 with r_1.
        // (A, B) ∈ {0, 1}² → folded ∈ {0, r_1, 1 − r_1, 1} via lookup.
        let zero_inner = F::zero_with_cfg(config).into_inner();
        let one_inner = one.inner().clone();
        let r1_inner = r1.inner().clone();
        let one_minus_r1_inner = one_minus_r1.inner().clone();

        let mut mles: Vec<DenseMultilinearExtension<F::Inner>> = Vec::with_capacity(
            1 + self.binary_cols.len() * D + self.extra_bit_cols.len(),
        );
        mles.push(eq_folded);

        // BinaryPoly<D> does not impl Index<usize> on every backend
        // (the simd-feature `BinaryU64Poly` exposes only `iter()`).
        // Iterate via `iter().zip(...)` per row pair, distributing each
        // bit's folded value into a per-bit accumulator.
        for col in &self.binary_cols {
            let mut per_bit: Vec<Vec<F::Inner>> =
                (0..D).map(|_| Vec::with_capacity(half)).collect();
            for b_prime in 0..half {
                let row_a = &col.evaluations[2 * b_prime];
                let row_b = &col.evaluations[2 * b_prime + 1];
                for (bit_idx, (a_bit, b_bit)) in row_a.iter().zip(row_b.iter()).enumerate() {
                    let v = match (a_bit.into_inner(), b_bit.into_inner()) {
                        (false, false) => zero_inner.clone(),
                        (true, true) => one_inner.clone(),
                        (false, true) => r1_inner.clone(),
                        (true, false) => one_minus_r1_inner.clone(),
                    };
                    per_bit[bit_idx].push(v);
                }
            }
            for evals in per_bit {
                mles.push(DenseMultilinearExtension {
                    num_vars: self.num_vars - 1,
                    evaluations: evals,
                });
            }
        }

        // Extra bit columns: each contributes one folded MLE. Values are
        // F::Inner already and assumed `{0, 1}`-valued; the (A,B) ∈
        // {0,1}² → {0, r_1, 1−r_1, 1} lookup uses `is_zero()` to
        // distinguish the cases.
        for col in &self.extra_bit_cols {
            let mut evals: Vec<F::Inner> = Vec::with_capacity(half);
            for b_prime in 0..half {
                let a = &col.evaluations[2 * b_prime];
                let b = &col.evaluations[2 * b_prime + 1];
                let a_is_one = !a.is_zero();
                let b_is_one = !b.is_zero();
                let v = match (a_is_one, b_is_one) {
                    (false, false) => zero_inner.clone(),
                    (true, true) => one_inner.clone(),
                    (false, true) => r1_inner.clone(),
                    (true, false) => one_minus_r1_inner.clone(),
                };
                evals.push(v);
            }
            mles.push(DenseMultilinearExtension {
                num_vars: self.num_vars - 1,
                evaluations: evals,
            });
        }

        mles
    }
}

/// Helper: reduce_with on a parallel iterator, falling back to a default
/// when the iterator is empty. Provided here to keep the parallel and
/// sequential paths uniform.
trait ReduceWithOrDefault<T: Clone> {
    fn reduce_with_or_default(self, default: T) -> T;
}

#[cfg(feature = "parallel")]
impl<I, T> ReduceWithOrDefault<T> for I
where
    I: ParallelIterator<Item = T>,
    T: Clone + Send + Sync + std::ops::Add<Output = T>,
{
    fn reduce_with_or_default(self, default: T) -> T {
        self.reduce(|| default.clone(), |a, b| a + b)
    }
}

#[cfg(not(feature = "parallel"))]
impl<I, T> ReduceWithOrDefault<T> for I
where
    I: Iterator<Item = T>,
    T: Clone + std::ops::Add<Output = T>,
{
    fn reduce_with_or_default(self, default: T) -> T {
        self.fold(default, |a, b| a + b)
    }
}

// ---------------------------------------------------------------------------
// Sumcheck group prep / finalize (separate degree-3 group)
// ---------------------------------------------------------------------------

/// Ancillary data produced by [`prepare_booleanity_group`] and consumed
/// by [`finalize_booleanity_prover`]. Carries the bit-slice count needed
/// to extract the right slice of evals after sumcheck completes.
pub struct BooleanityProverAncillary {
    /// Number of bit-slice MLEs in the group (excludes the leading eq_r MLE).
    pub num_bit_slices: usize,
}

/// Ancillary data produced by [`prepare_booleanity_verifier`] and
/// consumed by [`finalize_booleanity_verifier`].
pub struct BooleanityVerifierAncillary<F: PrimeField> {
    /// Powers of the booleanity folding challenge: `[1, α, α², ..., α^{B-1}]`.
    pub folding_challenge_powers: Vec<F>,
    /// Evaluation point used to build `eq_r` (mirrors what the prover used).
    pub ic_evaluation_point: Vec<F>,
}

/// Build a degree-3 multi-degree sumcheck group for the booleanity
/// zerocheck. MLE layout (post round-1 fold):
/// `[eq_r, v_0, ..., v_{K·D − 1}, e_0, ..., e_{E − 1}]`
/// where `K · D` covers the binary_poly bit slices and `E =
/// extra_bit_cols.len()` covers per-row scalar bit columns.
///
/// Round 1 is supplied via [`BooleanityRound1FastPath`], which reads the
/// `binary_cols` and `extra_bit_cols` directly and never materializes
/// F-valued full-size bit-slice MLEs. Rounds 2..n use the standard
/// sumcheck path on the half-size folded MLEs the fast path emits.
///
/// Each entry of `extra_bit_cols` must be `{0, 1}`-valued in `F::Inner`;
/// booleanity makes that a soundness condition.
///
/// Returns `None` when both `binary_cols` and `extra_bit_cols` are
/// empty (no booleanity check needed; caller should skip pushing this
/// group).
#[allow(clippy::arithmetic_side_effects)]
pub fn prepare_booleanity_group<F, const D: usize>(
    transcript: &mut impl Transcript,
    binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    extra_bit_cols: &[DenseMultilinearExtension<F::Inner>],
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Option<(MultiDegreeSumcheckGroup<F>, BooleanityProverAncillary)>, BooleanityError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default + Clone,
    F::Modulus: ConstTranscribable,
{
    if binary_cols.is_empty() && extra_bit_cols.is_empty() {
        return Ok(None);
    }

    let num_bit_slices = binary_cols.len() * D + extra_bit_cols.len();
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let folding_challenge: F = transcript.get_field_challenge(field_cfg);
    let folding_challenge_powers: Vec<F> =
        powers(folding_challenge, one.clone(), num_bit_slices);

    // Pre-build E_other = eq(b', ic_evaluation_point[1..]) for the
    // round-1 fast path. The full-size eq_r is only needed for rounds
    // 2..n in folded form, which fold_with_r1 produces directly.
    assert!(
        !ic_evaluation_point.is_empty(),
        "ic_evaluation_point must be non-empty for booleanity (num_vars >= 1)"
    );
    let num_vars = ic_evaluation_point.len();
    let eq_other_table: Vec<F::Inner> = if num_vars >= 2 {
        build_eq_x_r_inner(&ic_evaluation_point[1..], field_cfg)?.evaluations
    } else {
        // num_vars == 1: the "other" subspace is empty, E_other is a single 1.
        vec![one.inner().clone()]
    };

    let fast_path: Box<dyn Round1FastPath<F>> = Box::new(BooleanityRound1FastPath::<F, D> {
        binary_cols: binary_cols.to_vec(),
        extra_bit_cols: extra_bit_cols.to_vec(),
        alpha_powers: folding_challenge_powers.clone(),
        eq_other_table,
        ic_ep_0: ic_evaluation_point[0].clone(),
        num_vars,
    });

    let comb_fn: CombFn<F> = Box::new(move |mle_values: &[F]| {
        let eq_r = &mle_values[0];
        let bits = &mle_values[1..];
        debug_assert_eq!(bits.len(), folding_challenge_powers.len());

        // Σ_k α^k · v_k · (v_k - 1) computed as Σ_k α^k · (v_k² - v_k) to
        // avoid a per-iteration `(v - one)` clone.
        let mut acc = zero.clone();
        for (v, coeff) in bits.iter().zip(folding_challenge_powers.iter()) {
            let v_sq = v.clone() * v.clone();
            acc = acc + coeff.clone() * (v_sq - v.clone());
        }
        acc * eq_r.clone()
    });

    // Empty `poly` — the fast path supplies post-round-1 MLEs.
    Ok(Some((
        MultiDegreeSumcheckGroup::with_round_1_fast(3, Vec::new(), comb_fn, fast_path),
        BooleanityProverAncillary { num_bit_slices },
    )))
}

/// Extract `bit_slice_evals` from the booleanity group's prover state
/// after the multi-degree sumcheck completes. The leading `eq_r` MLE
/// eval is dropped (verifier recomputes it).
#[allow(clippy::arithmetic_side_effects)]
pub fn finalize_booleanity_prover<F>(
    transcript: &mut impl Transcript,
    sumcheck_prover_state: SumcheckProverState<F>,
    ancillary: BooleanityProverAncillary,
    field_cfg: &F::Config,
) -> Result<Vec<F>, BooleanityError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    debug_assert!(
        sumcheck_prover_state
            .mles
            .iter()
            .all(|mle| mle.num_vars == 1)
    );

    let last_challenge = sumcheck_prover_state
        .randomness
        .last()
        .expect("sumcheck must have at least one round")
        .clone();

    let mut mles = sumcheck_prover_state.mles;
    // mles[0] is eq_r — drop it; the rest are the bit-slices in order.
    let _eq_r_mle = mles.remove(0);
    let bit_slice_evals: Vec<F> = mles
        .into_iter()
        .map(|m| m.evaluate_with_config(slice::from_ref(&last_challenge), field_cfg))
        .collect::<Result<Vec<_>, _>>()?;

    debug_assert_eq!(bit_slice_evals.len(), ancillary.num_bit_slices);

    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(&bit_slice_evals, &mut buf);

    Ok(bit_slice_evals)
}

/// Pre-sumcheck verifier half. Samples α (matching prover order),
/// validates that the booleanity group's claimed sum is zero (this is a
/// pure zerocheck), and stashes per-bit α-powers for the post-sumcheck
/// finalize.
///
/// Returns `None` when there are no binary_poly columns (mirrors the
/// prover's early-out).
#[allow(clippy::arithmetic_side_effects)]
pub fn prepare_booleanity_verifier<F>(
    transcript: &mut impl Transcript,
    claimed_sum: F,
    num_bit_slices: usize,
    ic_evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Option<BooleanityVerifierAncillary<F>>, BooleanityError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    if num_bit_slices == 0 {
        return Ok(None);
    }

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    if claimed_sum != zero {
        return Err(BooleanityError::NonZeroClaimedSum { got: claimed_sum });
    }

    let folding_challenge: F = transcript.get_field_challenge(field_cfg);
    let folding_challenge_powers: Vec<F> = powers(folding_challenge, one, num_bit_slices);

    Ok(Some(BooleanityVerifierAncillary {
        folding_challenge_powers,
        ic_evaluation_point: ic_evaluation_point.to_vec(),
    }))
}

/// Post-sumcheck verifier half. Validates that
/// `expected_evaluation == Σ_k α^k · v_k · (v_k - 1) · eq_r(ic_eval_point, r*)`
/// where `r*` is the multi-degree sumcheck's shared point. Absorbs
/// `bit_slice_evals` into the transcript.
///
/// `closing_overrides_tail` replaces the trailing positions of
/// `bit_slice_evals` for the closing computation only — used to
/// substitute a column's CPR `up_eval` (or a virtual MLE's reconstructed
/// linear-combo eval) for its bit-slice eval when both are bound to the
/// same MLE at the shared point. Absorption is always over the original
/// `bit_slice_evals`. Pass an empty slice to disable.
#[allow(clippy::arithmetic_side_effects)]
pub fn finalize_booleanity_verifier<F>(
    transcript: &mut impl Transcript,
    bit_slice_evals: &[F],
    closing_overrides_tail: &[F],
    shared_point: &[F],
    expected_evaluation: F,
    ancillary: BooleanityVerifierAncillary<F>,
    field_cfg: &F::Config,
) -> Result<(), BooleanityError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    if bit_slice_evals.len() != ancillary.folding_challenge_powers.len() {
        return Err(BooleanityError::WrongBitSliceEvalCount {
            got: bit_slice_evals.len(),
            expected: ancillary.folding_challenge_powers.len(),
        });
    }
    if closing_overrides_tail.len() > bit_slice_evals.len() {
        return Err(BooleanityError::WrongBitSliceEvalCount {
            got: closing_overrides_tail.len(),
            expected: bit_slice_evals.len(),
        });
    }

    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);

    let eq_r_value = eq_eval(shared_point, &ancillary.ic_evaluation_point, one.clone())?;

    let n_proof = bit_slice_evals.len() - closing_overrides_tail.len();
    let bool_folded = bit_slice_evals[..n_proof]
        .iter()
        .chain(closing_overrides_tail.iter())
        .zip(ancillary.folding_challenge_powers.iter())
        .fold(zero, |acc, (v, coeff)| {
            let v_sq = v.clone() * v.clone();
            acc + coeff.clone() * (v_sq - v.clone())
        });

    let recomputed = bool_folded * eq_r_value;

    if recomputed != expected_evaluation {
        return Err(BooleanityError::SumcheckClaimMismatch {
            got: expected_evaluation,
            expected: recomputed,
        });
    }

    let mut buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_random_field_slice(bit_slice_evals, &mut buf);

    Ok(())
}

// ---------------------------------------------------------------------------
// Lifting consistency check (between bit-slice evals and parent column)
// ---------------------------------------------------------------------------

/// Verifier check: each binary_poly column's projected MLE evaluation at
/// `r*` (`up_evals[col_idx]`) must equal `Σ_i a^i · bit_slice_evals[i]`,
/// where `a` is the field-projection element used to send `F[X] → F`.
///
/// In projected `F`-land, `ψ_a(MLE[v](r*)) = Σ_i a^i · MLE[v_i](r*)`. With
/// overwhelming probability over the random `a`, the equation pins down
/// each bit-slice eval against the true bit-decomposition of the
/// committed parent column.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_bit_decomposition_consistency<F: PrimeField>(
    parent_evals_per_col: &[F],
    bit_slice_evals: &[F],
    projecting_element: &F,
    bits_per_col: usize,
) -> Result<(), BooleanityError<F>> {
    if bit_slice_evals.len() != parent_evals_per_col.len() * bits_per_col {
        return Err(BooleanityError::WrongBitSliceEvalCount {
            got: bit_slice_evals.len(),
            expected: parent_evals_per_col.len() * bits_per_col,
        });
    }

    if bits_per_col == 0 {
        return Ok(());
    }

    let zero = F::zero_with_cfg(projecting_element.cfg());
    let one = F::one_with_cfg(projecting_element.cfg());

    // Powers [1, a, a^2, ..., a^{bits_per_col - 1}].
    let mut a_powers: Vec<F> = Vec::with_capacity(bits_per_col);
    let mut acc = one;
    for _ in 0..bits_per_col {
        a_powers.push(acc.clone());
        acc *= projecting_element;
    }

    for (col_idx, parent_eval) in parent_evals_per_col.iter().enumerate() {
        let base = col_idx * bits_per_col;
        let recombined =
            bit_slice_evals[base..base + bits_per_col]
                .iter()
                .zip(&a_powers)
                .fold(zero.clone(), |acc, (bit_eval, a_pow)| {
                    acc + bit_eval.clone() * a_pow
                });

        if &recombined != parent_eval {
            return Err(BooleanityError::ConsistencyMismatch {
                col_idx,
                got: recombined,
                expected: parent_eval.clone(),
            });
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
pub enum BooleanityError<F: PrimeField> {
    #[error(
        "wrong bit-slice evaluation count: got {got}, expected {expected}"
    )]
    WrongBitSliceEvalCount { got: usize, expected: usize },
    #[error(
        "bit-decomposition consistency mismatch on binary_poly column {col_idx}: got Σ a^i·bᵢ = {got:?}, expected parent eval {expected:?}"
    )]
    ConsistencyMismatch { col_idx: usize, got: F, expected: F },
    #[error("booleanity zerocheck claimed sum non-zero: {got:?}")]
    NonZeroClaimedSum { got: F },
    #[error("booleanity sumcheck claim mismatch: got {got:?}, expected {expected:?}")]
    SumcheckClaimMismatch { got: F, expected: F },
    #[error("eq_r evaluation failed: {0}")]
    EqEvalError(#[from] ArithErrors),
    #[error("MLE evaluation failed: {0}")]
    MleEvaluationError(#[from] EvaluationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{
        FromWithConfig, boolean::Boolean, crypto_bigint_monty::MontyField,
    };

    type F = MontyField<4>;

    fn test_cfg() -> <F as crypto_primitives::PrimeField>::Config {
        crate::test_utils::test_config()
    }

    fn col_from_u8s(patterns: &[u8]) -> DenseMultilinearExtension<BinaryPoly<8>> {
        use std::array;
        let evaluations: Vec<BinaryPoly<8>> = patterns
            .iter()
            .map(|&p| {
                let coeffs: [Boolean; 8] =
                    array::from_fn(|i| Boolean::new((p >> i) & 1 != 0));
                BinaryPoly::<8>::new(coeffs)
            })
            .collect();
        let num_vars = evaluations.len().next_power_of_two().trailing_zeros() as usize;
        DenseMultilinearExtension {
            num_vars,
            evaluations,
        }
    }

    #[test]
    fn bit_slices_round_trip_recovers_original_bits() {
        let cfg = test_cfg();
        let col = col_from_u8s(&[0b00000000, 0b11111111, 0b10101010, 0b01010101]);
        let bit_slices = compute_bit_slices_flat::<F, 8>(std::slice::from_ref(&col), &cfg);

        assert_eq!(bit_slices.len(), 8);
        let one = F::one_with_cfg(&cfg).into_inner();
        let zero = F::zero_with_cfg(&cfg).into_inner();
        for (row, p) in [0b00000000u8, 0b11111111, 0b10101010, 0b01010101]
            .iter()
            .enumerate()
        {
            for bit in 0..8 {
                let want = if (p >> bit) & 1 != 0 {
                    one.clone()
                } else {
                    zero.clone()
                };
                assert_eq!(bit_slices[bit].evaluations[row], want, "row {row} bit {bit}");
            }
        }
    }

    #[test]
    fn consistency_check_accepts_honest_decomposition() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);

        let bits: [u32; 8] = [1, 0, 1, 1, 0, 0, 0, 1];
        let bit_evals: Vec<F> = bits
            .iter()
            .map(|&b| if b == 1 { one.clone() } else { zero.clone() })
            .collect();
        let a = one.clone() + one.clone() + one.clone();

        let mut parent_eval = zero.clone();
        let mut a_pow = one.clone();
        for be in &bit_evals {
            parent_eval = parent_eval + be.clone() * a_pow.clone();
            a_pow = a_pow * a.clone();
        }

        verify_bit_decomposition_consistency(
            std::slice::from_ref(&parent_eval),
            &bit_evals,
            &a,
            8,
        )
        .expect("honest decomposition should satisfy consistency check");
    }

    #[test]
    fn consistency_check_rejects_tampered_bit() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);

        let bits: [u32; 4] = [1, 1, 1, 1];
        let bit_evals: Vec<F> = bits
            .iter()
            .map(|&b| if b == 1 { one.clone() } else { zero.clone() })
            .collect();
        let a = one.clone() + one.clone();

        let mut parent_eval = zero.clone();
        let mut a_pow = one.clone();
        for be in &bit_evals {
            parent_eval = parent_eval + be.clone() * a_pow.clone();
            a_pow = a_pow * a.clone();
        }

        let mut tampered = bit_evals.clone();
        tampered[0] = tampered[0].clone() + one;

        let res = verify_bit_decomposition_consistency(
            std::slice::from_ref(&parent_eval),
            &tampered,
            &a,
            4,
        );
        assert!(matches!(res, Err(BooleanityError::ConsistencyMismatch { .. })));
    }

    #[test]
    fn consistency_check_no_op_when_no_binary_poly_columns() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let parent_evals: Vec<F> = vec![];
        let bit_evals: Vec<F> = vec![];
        verify_bit_decomposition_consistency(&parent_evals, &bit_evals, &one, 8).unwrap();
    }

    /// Cross-validate the round-1 fast path against a faithful standard
    /// run of `ProverState::prove_round`. Both must produce the same
    /// tail evaluations and the same asserted sum (zero, since this is
    /// a zerocheck).
    #[test]
    fn fast_path_round_1_matches_standard_prove_round() {
        use crate::CombFn;
        use crate::sumcheck::prover::ProverState as SumcheckProverState;

        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);

        // 3-variable case (8 rows). Two binary_poly cols of width 8 each.
        // Mix of fully-zero, fully-one, and varying patterns to exercise all
        // four (A, B) cases for the XOR fold structure.
        let binary_cols = vec![
            col_from_u8s(&[0b00000000, 0b00010001, 0b00100010, 0b00110011, 0b01000100, 0b01010101, 0b01100110, 0b01110111]),
            col_from_u8s(&[0b11110000, 0b11100001, 0b11010010, 0b11000011, 0b10110100, 0b10100101, 0b10010110, 0b10000111]),
        ];
        let num_vars = 3;
        const D: usize = 8;
        let k = binary_cols.len();
        let num_bit_slices = k * D;

        let alpha = F::from_with_cfg(7u64, &cfg);
        let alpha_powers: Vec<F> = powers(alpha, one.clone(), num_bit_slices);

        let ic_ep: Vec<F> = vec![
            F::from_with_cfg(3u64, &cfg),
            F::from_with_cfg(5u64, &cfg),
            F::from_with_cfg(11u64, &cfg),
        ];

        // ---- Fast path ----
        let eq_other_table = build_eq_x_r_inner(&ic_ep[1..], &cfg).unwrap().evaluations;
        let fast_path = BooleanityRound1FastPath::<F, D> {
            binary_cols: binary_cols.clone(),
            extra_bit_cols: Vec::new(),
            alpha_powers: alpha_powers.clone(),
            eq_other_table,
            ic_ep_0: ic_ep[0].clone(),
            num_vars,
        };
        let fast_out = fast_path.round_1_message(&cfg);

        // ---- Standard path ----
        let bit_slices = compute_bit_slices_flat::<F, D>(&binary_cols, &cfg);
        let eq_r = build_eq_x_r_inner(&ic_ep, &cfg).unwrap();
        let mut mles: Vec<DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner>> =
            Vec::with_capacity(1 + bit_slices.len());
        mles.push(eq_r);
        mles.extend(bit_slices);

        let alpha_powers_for_comb = alpha_powers.clone();
        let zero_for_comb = zero.clone();
        let comb_fn: CombFn<F> = Box::new(move |mle_values: &[F]| {
            let eq_r = &mle_values[0];
            let bits = &mle_values[1..];
            let mut acc = zero_for_comb.clone();
            for (v, coeff) in bits.iter().zip(alpha_powers_for_comb.iter()) {
                let v_sq = v.clone() * v.clone();
                acc = acc + coeff.clone() * (v_sq - v.clone());
            }
            acc * eq_r.clone()
        });

        let mut state = SumcheckProverState::new(mles, num_vars, 3);
        let std_msg = state.prove_round(&None, comb_fn, &cfg);
        let std_asserted = state
            .asserted_sum
            .clone()
            .expect("asserted_sum recorded after first round");

        assert_eq!(
            fast_out.tail_evaluations.len(),
            std_msg.0.tail_evaluations.len(),
            "tail length mismatch (fast vs standard)"
        );
        assert_eq!(
            fast_out.tail_evaluations, std_msg.0.tail_evaluations,
            "fast-path round-1 tail must match standard path"
        );
        assert_eq!(
            fast_out.asserted_sum, std_asserted,
            "fast-path asserted_sum must match standard path (zerocheck → 0)"
        );
        assert_eq!(
            fast_out.asserted_sum, zero,
            "booleanity zerocheck asserted_sum must be 0"
        );
    }

    /// Cross-validate `fold_with_r1` against folding bit-slice MLEs via
    /// the standard `fix_variables` path. Both must produce the same
    /// post-round-1 MLE values for every (k, i) bit-slice.
    #[test]
    fn fast_path_fold_with_r1_matches_standard_fix_variables() {
        use std::slice;
        use zinc_poly::mle::MultilinearExtensionWithConfig;

        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);

        let binary_cols = vec![col_from_u8s(&[
            0b00000000, 0b11111111, 0b01010101, 0b10101010, 0b11001100, 0b00110011, 0b11110000,
            0b00001111,
        ])];
        let num_vars = 3;
        const D: usize = 8;
        let alpha = F::from_with_cfg(13u64, &cfg);
        let alpha_powers: Vec<F> = powers(alpha, one.clone(), D);
        let ic_ep: Vec<F> = vec![
            F::from_with_cfg(2u64, &cfg),
            F::from_with_cfg(4u64, &cfg),
            F::from_with_cfg(8u64, &cfg),
        ];
        let r_1 = F::from_with_cfg(17u64, &cfg);

        let eq_other_table = build_eq_x_r_inner(&ic_ep[1..], &cfg).unwrap().evaluations;
        let fast_path = Box::new(BooleanityRound1FastPath::<F, D> {
            binary_cols: binary_cols.clone(),
            extra_bit_cols: Vec::new(),
            alpha_powers,
            eq_other_table,
            ic_ep_0: ic_ep[0].clone(),
            num_vars,
        });

        let fast_mles = fast_path.fold_with_r1(&r_1, &cfg);
        // [eq_r_folded, bit_0, bit_1, ..., bit_{D-1}].
        assert_eq!(fast_mles.len(), 1 + D);

        // Standard-path comparison: build full-size bit-slice MLEs and
        // fold each with r_1. Build full-size eq_r and fold the same way.
        let bit_slices_full = compute_bit_slices_flat::<F, D>(&binary_cols, &cfg);
        let eq_r_full = build_eq_x_r_inner(&ic_ep, &cfg).unwrap();

        let mut std_eq_r = eq_r_full;
        std_eq_r.fix_variables_with_config(slice::from_ref(&r_1), &cfg);
        assert_eq!(
            fast_mles[0].num_vars, num_vars - 1,
            "fast-path eq_r_folded must have num_vars - 1 variables"
        );
        assert_eq!(
            fast_mles[0].evaluations, std_eq_r.evaluations,
            "fast-path eq_r_folded must match standard fix_variables"
        );

        for (idx, mut bit_mle) in bit_slices_full.into_iter().enumerate() {
            bit_mle.fix_variables_with_config(slice::from_ref(&r_1), &cfg);
            assert_eq!(
                fast_mles[1 + idx].evaluations, bit_mle.evaluations,
                "fast-path bit-slice {idx} folded value must match standard fix_variables"
            );
        }
    }

    #[test]
    fn consistency_check_rejects_wrong_eval_count() {
        let cfg = test_cfg();
        let one = F::one_with_cfg(&cfg);
        let parent_evals = vec![one.clone()];
        let bit_evals: Vec<F> = vec![one.clone(), one.clone()];
        let res = verify_bit_decomposition_consistency(&parent_evals, &bit_evals, &one, 8);
        assert!(matches!(res, Err(BooleanityError::WrongBitSliceEvalCount { .. })));
    }
}
