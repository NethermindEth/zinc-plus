//! Fold-first SumFold (V2): univariate-skip booleanity zerocheck over the
//! instance axis, run *before* IdealCheck and scalarization.
//!
//! The instance axis is packed into one univariate variable over a symmetric
//! integer node set. The skip-round polynomial
//!
//! q(Y) = Σ_{q,z} ω_{q,z} · D̂_q(Y,z)(D̂_q(Y,z) − 1),
//! ω_{q,z} = ρ^idx(q) · eq(r_ic, z),
//!
//! is computed through the Gram sufficient statistic (G, h) with
//! G[j,k] = Σ ω·d_j·d_k and h[j] = Σ ω·d_j, so the streaming pass over the
//! witness pays no per-interpolation-point work; every evaluation of q is an
//! O(N²) quadratic form afterwards. The prover transmits q in evaluation
//! basis over 2N−1 statement-fixed nodes; the verifier enforces the
//! γ-weighted zerocheck Σ_j γ^j·q(u_j) = 0, samples α, and derives the fold
//! weights θ_j = L_j(α) plus the relaxed booleanity residue B★ = q(α).
//!
//! See `documentation/fold-first-sumfold-doc/prover-algorithm.md` for the
//! protocol-level specification and the optimization catalog (instance-major
//! bit-packing and delayed-reduction accumulators are later optimizations;
//! this module is the correctness-first reference).

use crypto_primitives::{
    FromPrimitiveWithConfig, FromWithConfig, PrimeField, crypto_bigint_uint::Uint,
};
use num_traits::Zero;
use std::borrow::Borrow;
use thiserror::Error;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_poly::utils::build_eq_x_r_vec;
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_utils::{
    delayed_reduction::{
        BarrettDelayedReduction, DelayedFieldProductSum, DelayedModularReductionAlgorithm,
        MontgomeryLimbs,
    },
    inner_transparent_field::InnerTransparentField,
    powers,
};

use crate::neutron_nova::SumFoldError;
use crate::neutron_nova::projection_sha::{
    MleColumn, MleTable, NONZERO_SHA_FAMILIES, NUM_NONZERO_SHA_FAMILIES, NUM_SHA_RESIDUAL_FAMILIES,
    ProjectedPublic, ProjectedTrace, ProjectionFoldWitness, SHA_ROW_COUNT, SHA_ROW_VARS,
    SHA_WORD_BITS, ShaBinaryFoldField, ShaBooleanitySource, ShaProjectionError, ShaWordCol,
    VirtualChMajValues, booleanity_source_value_at_row_with_virtuals,
    build_folded_row_sumcheck_group, build_sha_ideal_values_at_point, fold_mle_tables,
    fold_optional_binary_mle_tables, fold_projected_traces_with_weights,
    folded_row_integrand_values, reconstruct_virtual_ch_maj_at_row_unchecked, scalarize_bit_slices,
    sources_need_virtuals, verify_folded_row_sumcheck_claim, verify_fresh_sha_ideal_polys,
};
use crate::sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof};

/// Symmetric-integer packing domain for the univariate skip round.
///
/// Stores `2N − 1` nodes: the `N` packing (domain) nodes first — consecutive
/// integers centred on zero — followed by `N − 1` off-domain evaluation
/// nodes continuing upward. All node coordinates are pairwise distinct.
#[derive(Clone, Debug)]
pub struct SkipDomain<F> {
    nodes: Vec<F>,
    /// Inverse Lagrange denominators over the first `N` nodes.
    inv_denoms_domain: Vec<F>,
    /// Inverse Lagrange denominators over all `2N − 1` nodes.
    inv_denoms_all: Vec<F>,
    n_instances: usize,
}

#[allow(clippy::arithmetic_side_effects)]
fn signed_field<F>(value: i64, field_cfg: &F::Config) -> F
where
    F: PrimeField + FromWithConfig<u64>,
{
    if value >= 0 {
        F::from_with_cfg(value.unsigned_abs(), field_cfg)
    } else {
        F::zero_with_cfg(field_cfg) - F::from_with_cfg(value.unsigned_abs(), field_cfg)
    }
}

/// Montgomery batch inversion: replaces `values[i]` with `values[i]⁻¹` using
/// one field inversion and `3(n−1)` multiplications. Panics on zero inputs
/// (never produced here: Lagrange denominators over distinct nodes).
#[allow(clippy::arithmetic_side_effects)]
fn batch_invert<F>(values: &mut [F], field_cfg: &F::Config)
where
    F: PrimeField,
{
    if values.is_empty() {
        return;
    }
    let one = F::one_with_cfg(field_cfg);
    let mut prefix = Vec::with_capacity(values.len());
    let mut acc = one.clone();
    for value in values.iter() {
        prefix.push(acc.clone());
        acc *= value.clone();
    }
    let mut inv = one / acc;
    for (value, pre) in values.iter_mut().zip(prefix).rev() {
        let next = inv.clone() * value.clone();
        *value = inv * pre;
        inv = next;
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn inverse_lagrange_denominators<F>(points: &[F], field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField,
{
    let one = F::one_with_cfg(field_cfg);
    let mut denoms: Vec<F> = points
        .iter()
        .enumerate()
        .map(|(j, u_j)| {
            let mut denom = one.clone();
            for (i, u_i) in points.iter().enumerate() {
                if i != j {
                    denom *= u_j.clone() - u_i;
                }
            }
            denom
        })
        .collect();
    batch_invert(&mut denoms, field_cfg);
    denoms
}

/// Lagrange basis values `{L_j(x)}_j` for the given interpolation points.
#[allow(clippy::arithmetic_side_effects)]
fn lagrange_basis_at<F>(points: &[F], inv_denoms: &[F], x: &F, field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField,
{
    let len = points.len();
    let one = F::one_with_cfg(field_cfg);
    // prefix[j] = Π_{i<j} (x − u_i), suffix[j] = Π_{i>j} (x − u_i)
    let mut prefix = vec![one.clone(); len];
    for j in 1..len {
        prefix[j] = prefix[j - 1].clone() * (x.clone() - &points[j - 1]);
    }
    let mut suffix = vec![one; len];
    for j in (0..len.saturating_sub(1)).rev() {
        suffix[j] = suffix[j + 1].clone() * (x.clone() - &points[j + 1]);
    }
    (0..len)
        .map(|j| prefix[j].clone() * &suffix[j] * &inv_denoms[j])
        .collect()
}

impl<F> SkipDomain<F>
where
    F: PrimeField + FromWithConfig<u64>,
{
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(n_instances: usize, field_cfg: &F::Config) -> Result<Self, SumFoldError> {
        if n_instances < 2 {
            return Err(SumFoldError::SkipRoundTooFewInstances { got: n_instances });
        }
        let half = i64::try_from(n_instances / 2)
            .map_err(|_| SumFoldError::DomainTooLarge { ell: n_instances })?;
        let count = i64::try_from(n_instances)
            .map_err(|_| SumFoldError::DomainTooLarge { ell: n_instances })?;
        // Domain nodes: j − ⌊N/2⌋ for j = 0..N−1; off-domain nodes continue
        // upward from ⌈N/2⌉ = N − ⌊N/2⌋.
        let mut nodes = Vec::with_capacity(2 * n_instances - 1);
        for j in 0..count {
            nodes.push(signed_field::<F>(j - half, field_cfg));
        }
        for s in 0..count - 1 {
            nodes.push(signed_field::<F>(count - half + s, field_cfg));
        }
        let inv_denoms_domain = inverse_lagrange_denominators(&nodes[..n_instances], field_cfg);
        let inv_denoms_all = inverse_lagrange_denominators(&nodes, field_cfg);
        Ok(Self {
            nodes,
            inv_denoms_domain,
            inv_denoms_all,
            n_instances,
        })
    }

    pub fn n_instances(&self) -> usize {
        self.n_instances
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn node_count(&self) -> usize {
        2 * self.n_instances - 1
    }

    /// All nodes: the `N` domain nodes first, then the off-domain nodes.
    pub fn nodes(&self) -> &[F] {
        &self.nodes
    }

    pub fn domain_nodes(&self) -> &[F] {
        &self.nodes[..self.n_instances]
    }

    /// Fold weights `θ_j = L_j(x)` over the `N` domain nodes.
    pub fn lagrange_at(&self, x: &F, field_cfg: &F::Config) -> Vec<F> {
        lagrange_basis_at(
            &self.nodes[..self.n_instances],
            &self.inv_denoms_domain,
            x,
            field_cfg,
        )
    }

    /// Evaluate the unique degree `≤ 2N−2` polynomial through
    /// `(nodes[i], node_values[i])` at `x`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn interpolate(
        &self,
        node_values: &[F],
        x: &F,
        field_cfg: &F::Config,
    ) -> Result<F, SumFoldError> {
        if node_values.len() != self.node_count() {
            return Err(SumFoldError::SkipRoundMessageLength {
                got: node_values.len(),
                expected: self.node_count(),
            });
        }
        let basis = lagrange_basis_at(&self.nodes, &self.inv_denoms_all, x, field_cfg);
        let mut acc = F::zero_with_cfg(field_cfg);
        for (value, weight) in node_values.iter().zip(&basis) {
            acc += value.clone() * weight;
        }
        Ok(acc)
    }
}

/// Upper-triangular index for `j ≤ k`.
#[allow(clippy::arithmetic_side_effects)]
fn tri(j: usize, k: usize) -> usize {
    debug_assert!(j <= k);
    k * (k + 1) / 2 + j
}

/// The Gram sufficient statistic of the booleanity zerocheck:
/// `G[j,k] = Σ_{q,z} ω_{q,z}·d_j·d_k` (upper triangle) and
/// `h[j] = Σ_{q,z} ω_{q,z}·d_j`, where `d_j = D_{j,q}(z)`.
#[derive(Clone, Debug)]
pub struct GramAccumulator<F> {
    g: Vec<F>,
    h: Vec<F>,
    n_instances: usize,
}

const MASK_LANES: usize = 128;

/// Instance-major bit-packed view of the real SHA booleanity sources
/// (O-10): for each `(word_col, bit, row)` slot, bit `j` of the mask is
/// instance `j`'s value. Slots where any instance holds a non-0/1 value are
/// flagged so the accumulator falls back to the general field path there.
pub struct ShaSourceMasks {
    /// Indexed `slot * SHA_ROW_COUNT + row`, `slot = col * SHA_WORD_BITS + bit`.
    real: Vec<u128>,
    non_binary: Vec<bool>,
    lane: u128,
}

impl ShaSourceMasks {
    /// Build the instance-major masks, or `None` when the batch does not fit
    /// the `u128` lanes (`n > 128`).
    #[allow(clippy::arithmetic_side_effects)]
    pub fn build<F, Trace>(traces: &[Trace], field_cfg: &F::Config) -> Option<Self>
    where
        F: PrimeField + Send + Sync,
        F::Config: Sync,
        Trace: Borrow<ProjectedTrace<F>> + Sync,
    {
        let n = traces.len();
        if n == 0 || n > MASK_LANES {
            return None;
        }
        let lane = if n == MASK_LANES {
            u128::MAX
        } else {
            (1u128 << n) - 1
        };
        let one = F::one_with_cfg(field_cfg);
        let slots = ShaWordCol::COUNT * SHA_WORD_BITS;
        let mut real = vec![0u128; slots * SHA_ROW_COUNT];
        let mut non_binary = vec![false; slots * SHA_ROW_COUNT];

        // `is_zero` avoids the full `PartialEq` (which also compares field
        // parameters); the `== one` compare only runs for set bits.
        let fill_slot = |slot: usize, real_rows: &mut [u128], nb_rows: &mut [bool]| {
            for (row, (mask_out, nb_out)) in
                real_rows.iter_mut().zip(nb_rows.iter_mut()).enumerate()
            {
                let mut mask = 0u128;
                let mut bad = false;
                for (j, trace) in traces.iter().enumerate() {
                    let value = &trace.borrow().bit_slices[slot].evaluations[row];
                    if F::is_zero(value) {
                        continue;
                    }
                    if *value == one {
                        mask |= 1u128 << j;
                    } else {
                        bad = true;
                        break;
                    }
                }
                *mask_out = mask;
                *nb_out = bad;
            }
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            real.par_chunks_mut(SHA_ROW_COUNT)
                .zip(non_binary.par_chunks_mut(SHA_ROW_COUNT))
                .enumerate()
                .for_each(|(slot, (real_rows, nb_rows))| fill_slot(slot, real_rows, nb_rows));
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (slot, (real_rows, nb_rows)) in real
                .chunks_mut(SHA_ROW_COUNT)
                .zip(non_binary.chunks_mut(SHA_ROW_COUNT))
                .enumerate()
            {
                fill_slot(slot, real_rows, nb_rows);
            }
        }

        Some(Self {
            real,
            non_binary,
            lane,
        })
    }

    /// Mask of a real slot at a (possibly out-of-range) shifted row.
    /// `None` when any instance's value there is non-binary.
    #[allow(clippy::arithmetic_side_effects)]
    fn real_at(&self, col: ShaWordCol, bit: usize, row: usize) -> Option<u128> {
        if row >= SHA_ROW_COUNT {
            return Some(0);
        }
        let idx = (col.index() * SHA_WORD_BITS + bit) * SHA_ROW_COUNT + row;
        if self.non_binary[idx] {
            None
        } else {
            Some(self.real[idx])
        }
    }

    /// Resolve a booleanity source at `row` into an instance mask, or `None`
    /// when the general path must handle it (non-binary inputs, or a virtual
    /// expression outside {0,1} for some instance). Virtual recipes mirror
    /// `reconstruct_virtual_ch_maj_at_row`:
    ///
    /// - ch1 = E[z+2] + E[z+1] − 2·Uef[z+2]  ∈ {0,1} iff Uef = E₂∧E₁, value E₂⊕E₁
    /// - ch2 = E[z+2] − E[z] + 2·UNegEg[z+2] + 2·Ch2Comp[z]
    ///   ∈ {0,1} iff UNegEg+Ch2Comp = ¬E₂∧E₀ (no double-count), value E₂⊕E₀
    /// - maj = A[z]+A[z+1]+A[z+2] − 2·Maj[z+2] − 2·MajComp[z]
    ///   ∈ {0,1} iff Maj+MajComp = majority(A₀,A₁,A₂), value A₀⊕A₁⊕A₂
    #[allow(clippy::arithmetic_side_effects)]
    fn resolve(&self, source: &ShaBooleanitySource, row: usize) -> Option<u128> {
        let lane = self.lane;
        match source {
            ShaBooleanitySource::WordBit { col, bit } => self.real_at(*col, *bit, row),
            ShaBooleanitySource::VirtualCh1 { bit } => {
                let e2 = self.real_at(ShaWordCol::E, *bit, row + 2)?;
                let e1 = self.real_at(ShaWordCol::E, *bit, row + 1)?;
                let u = self.real_at(ShaWordCol::Uef, *bit, row + 2)?;
                let valid = !(u ^ (e2 & e1)) & lane;
                if valid != lane {
                    return None;
                }
                Some(e2 ^ e1)
            }
            ShaBooleanitySource::VirtualCh2 { bit } => {
                let e2 = self.real_at(ShaWordCol::E, *bit, row + 2)?;
                let e0 = self.real_at(ShaWordCol::E, *bit, row)?;
                let u = self.real_at(ShaWordCol::UNegEg, *bit, row + 2)?;
                let c = self.real_at(ShaWordCol::Ch2Comp, *bit, row)?;
                let t = !e2 & e0 & lane;
                let one_of = (u ^ c) & !(u & c) & lane;
                let none_of = !(u | c) & lane;
                let valid = (t & one_of) | (!t & lane & none_of);
                if valid != lane {
                    return None;
                }
                Some(e2 ^ e0)
            }
            ShaBooleanitySource::VirtualMaj { bit } => {
                let a0 = self.real_at(ShaWordCol::A, *bit, row)?;
                let a1 = self.real_at(ShaWordCol::A, *bit, row + 1)?;
                let a2 = self.real_at(ShaWordCol::A, *bit, row + 2)?;
                let m = self.real_at(ShaWordCol::Maj, *bit, row + 2)?;
                let mc = self.real_at(ShaWordCol::MajComp, *bit, row)?;
                let maj3 = (a0 & a1) | (a0 & a2) | (a1 & a2);
                let one_of = (m ^ mc) & !(m & mc) & lane;
                let none_of = !(m | mc) & lane;
                let valid = (maj3 & one_of) | (!maj3 & lane & none_of);
                if valid != lane {
                    return None;
                }
                Some(a0 ^ a1 ^ a2)
            }
        }
    }
}

/// Fast-path worker (O-12/O-15): accumulate all mask-resolvable `(q, z)`
/// items in `rows` into thread-local `(g, h)`; items that need the general
/// field path are returned for the sequential fallback pass.
#[allow(clippy::arithmetic_side_effects)]
fn gram_fast_rows<F>(
    rows: std::ops::Range<usize>,
    masks: &ShaSourceMasks,
    row_weights: &[F],
    rho_powers: &[F],
    sources: &[ShaBooleanitySource],
    n: usize,
    reducer: &BarrettDelayedReduction<'_, F>,
    field_cfg: &F::Config,
) -> (Vec<F>, Vec<F>, Vec<(usize, usize)>)
where
    F: PrimeField + MontgomeryLimbs,
{
    // O-14: unreduced 5-limb accumulators with the reducer's flush contract
    // (DEFAULT_DMR_FLUSH_ADDS for 4-limb moduli — never hit at our sizes;
    // every add for small test moduli). Reduced once per entry at worker end.
    let entries = tri(n - 1, n - 1) + 1;
    let zero = F::zero_with_cfg(field_cfg);
    let flush_adds = reducer.flush_adds();
    let mut g_buckets = vec![Uint::<5>::zero(); entries];
    let mut g_pending = vec![0usize; entries];
    let mut g_acc = vec![zero.clone(); entries];
    let mut h_buckets = vec![Uint::<5>::zero(); n];
    let mut h_pending = vec![0usize; n];
    let mut h_acc = vec![zero; n];
    let mut fallback = Vec::new();

    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn bucket_add<F>(
        reducer: &BarrettDelayedReduction<'_, F>,
        flush_adds: usize,
        bucket: &mut Uint<5>,
        pending: &mut usize,
        acc: &mut F,
        omega: &F,
    ) where
        F: PrimeField + MontgomeryLimbs,
    {
        // Small-modulus configs (flush_adds == 1) reduce on every add; a
        // plain field add is strictly cheaper there. Real 4-limb moduli
        // (flush_adds = 2^20) never flush at our sizes.
        if flush_adds <= 1 {
            *acc += omega.clone();
            return;
        }
        reducer.add(bucket, omega);
        *pending += 1;
        if *pending >= flush_adds {
            let full = std::mem::replace(bucket, Uint::zero());
            *acc += reducer.reduce(full);
            *pending = 0;
        }
    }

    for row in rows {
        let row_weight = &row_weights[row];
        for (source_idx, source) in sources.iter().enumerate() {
            match masks.resolve(source, row) {
                Some(0) => {}
                Some(mask) => {
                    let omega = rho_powers[source_idx].clone() * row_weight;
                    let mut mk = mask;
                    while mk != 0 {
                        let k = mk.trailing_zeros() as usize;
                        bucket_add(
                            reducer,
                            flush_adds,
                            &mut h_buckets[k],
                            &mut h_pending[k],
                            &mut h_acc[k],
                            &omega,
                        );
                        // j ≤ k, j in mask (diagonal included).
                        let mut mj = mask & (((1u128 << k) - 1) | (1u128 << k));
                        while mj != 0 {
                            let j = mj.trailing_zeros() as usize;
                            let entry = tri(j, k);
                            bucket_add(
                                reducer,
                                flush_adds,
                                &mut g_buckets[entry],
                                &mut g_pending[entry],
                                &mut g_acc[entry],
                                &omega,
                            );
                            mj &= mj - 1;
                        }
                        mk &= mk - 1;
                    }
                }
                None => fallback.push((source_idx, row)),
            }
        }
    }

    for (bucket, acc) in g_buckets.into_iter().zip(g_acc.iter_mut()) {
        if !bucket.is_zero() {
            *acc += reducer.reduce(bucket);
        }
    }
    for (bucket, acc) in h_buckets.into_iter().zip(h_acc.iter_mut()) {
        if !bucket.is_zero() {
            *acc += reducer.reduce(bucket);
        }
    }
    (g_acc, h_acc, fallback)
}

/// One streaming pass over `(source, row)` items across all instances.
///
/// Dispatches to the mask-packed fast path (O-10/O-12/O-15) whenever every
/// instance fits a `u128` lane; per-item fallbacks handle non-binary values
/// and virtual expressions outside {0,1} exactly like the reference path.
///
/// `row_weights` must be the `eq(r_ic, ·)` table over the 128 SHA rows and
/// `rho_powers` must have one entry per booleanity source (`ρ^idx(q)`).
#[allow(clippy::arithmetic_side_effects)]
pub fn accumulate_booleanity_gram<F, Trace>(
    traces: &[Trace],
    row_weights: &[F],
    rho_powers: &[F],
    sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<GramAccumulator<F>, ShaProjectionError>
where
    F: PrimeField + MontgomeryLimbs + Send + Sync,
    F::Config: Sync,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
{
    let n = traces.len();
    if n == 0 {
        return accumulate_booleanity_gram_reference(
            traces,
            row_weights,
            rho_powers,
            sources,
            field_cfg,
        );
    }
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if rho_powers.len() != sources.len() {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "rho_powers",
            col: 0,
            got: rho_powers.len(),
            expected: sources.len(),
        });
    }
    let Some(masks) = ShaSourceMasks::build(traces, field_cfg) else {
        return accumulate_booleanity_gram_reference(
            traces,
            row_weights,
            rho_powers,
            sources,
            field_cfg,
        );
    };
    accumulate_booleanity_gram_with_masks(
        traces,
        &masks,
        row_weights,
        rho_powers,
        sources,
        field_cfg,
    )
}

/// Mask-packed Gram pass with caller-provided masks (build them once with
/// [`ShaSourceMasks::build`] and reuse for [`fold_projected_traces_with_theta_masks`]).
#[allow(clippy::arithmetic_side_effects)]
pub fn accumulate_booleanity_gram_with_masks<F, Trace>(
    traces: &[Trace],
    masks: &ShaSourceMasks,
    row_weights: &[F],
    rho_powers: &[F],
    sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<GramAccumulator<F>, ShaProjectionError>
where
    F: PrimeField + MontgomeryLimbs + Send + Sync,
    F::Config: Sync,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
{
    let n = traces.len();
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if rho_powers.len() != sources.len() {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "rho_powers",
            col: 0,
            got: rho_powers.len(),
            expected: sources.len(),
        });
    }
    let reducer = BarrettDelayedReduction::<F>::new(field_cfg);

    #[cfg(feature = "parallel")]
    let (mut g, mut h, fallback) = {
        use rayon::prelude::*;
        const ROW_CHUNK: usize = 4;
        let chunks: Vec<_> = (0..SHA_ROW_COUNT.div_ceil(ROW_CHUNK))
            .map(|c| c * ROW_CHUNK..((c + 1) * ROW_CHUNK).min(SHA_ROW_COUNT))
            .collect();
        chunks
            .into_par_iter()
            .map(|rows| {
                gram_fast_rows(
                    rows,
                    &masks,
                    row_weights,
                    rho_powers,
                    sources,
                    n,
                    &reducer,
                    field_cfg,
                )
            })
            .reduce(
                || {
                    let zero = F::zero_with_cfg(field_cfg);
                    (
                        vec![zero.clone(); tri(n - 1, n - 1) + 1],
                        vec![zero; n],
                        Vec::new(),
                    )
                },
                |(mut g_a, mut h_a, mut f_a), (g_b, h_b, f_b)| {
                    for (a, b) in g_a.iter_mut().zip(g_b) {
                        *a += b;
                    }
                    for (a, b) in h_a.iter_mut().zip(h_b) {
                        *a += b;
                    }
                    f_a.extend(f_b);
                    (g_a, h_a, f_a)
                },
            )
    };
    #[cfg(not(feature = "parallel"))]
    let (mut g, mut h, fallback) = gram_fast_rows(
        0..SHA_ROW_COUNT,
        &masks,
        row_weights,
        rho_powers,
        sources,
        n,
        &reducer,
        field_cfg,
    );

    // General field path for the few items the masks could not certify.
    if !fallback.is_empty() {
        let zero = F::zero_with_cfg(field_cfg);
        let one = F::one_with_cfg(field_cfg);
        let needs_virtuals = sources_need_virtuals(sources);
        let mut values = vec![zero.clone(); n];
        let mut virtuals_cache: std::collections::HashMap<usize, Vec<VirtualChMajValues<F>>> =
            std::collections::HashMap::new();
        for (source_idx, row) in fallback {
            let source = &sources[source_idx];
            let is_virtual = !matches!(source, ShaBooleanitySource::WordBit { .. });
            let virtuals_row = if needs_virtuals && is_virtual {
                Some(match virtuals_cache.entry(row) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let computed = traces
                            .iter()
                            .map(|trace| {
                                reconstruct_virtual_ch_maj_at_row_unchecked(
                                    trace.borrow(),
                                    row,
                                    field_cfg,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        entry.insert(computed)
                    }
                })
            } else {
                None
            };
            let mut any_nonzero = false;
            for (j, trace) in traces.iter().enumerate() {
                let value = booleanity_source_value_at_row_with_virtuals(
                    trace.borrow(),
                    row,
                    source,
                    virtuals_row.as_ref().map(|v| &v[j]),
                    field_cfg,
                )?;
                any_nonzero |= value != zero;
                values[j] = value;
            }
            if !any_nonzero {
                continue;
            }
            let omega = rho_powers[source_idx].clone() * &row_weights[row];
            accumulate_general_item(&values, &omega, &zero, &one, &mut g, &mut h);
        }
    }

    Ok(GramAccumulator {
        g,
        h,
        n_instances: n,
    })
}

/// Shared general-path pair accumulation for one `(q, z)` item.
#[allow(clippy::arithmetic_side_effects)]
fn accumulate_general_item<F>(values: &[F], omega: &F, zero: &F, one: &F, g: &mut [F], h: &mut [F])
where
    F: PrimeField,
{
    let n = values.len();
    for k in 0..n {
        if values[k] == *zero {
            continue;
        }
        let omega_k = if values[k] == *one {
            omega.clone()
        } else {
            omega.clone() * &values[k]
        };
        h[k] += omega_k.clone();
        for j in 0..=k {
            if values[j] == *zero {
                continue;
            }
            if values[j] == *one {
                g[tri(j, k)] += omega_k.clone();
            } else {
                g[tri(j, k)] += omega_k.clone() * &values[j];
            }
        }
    }
}

/// Reference (correctness-first) implementation of the Gram pass. Kept for
/// differential tests and as the fallback for `n > 128` instances.
///
/// `row_weights` must be the `eq(r_ic, ·)` table over the 128 SHA rows and
/// `rho_powers` must have one entry per booleanity source (`ρ^idx(q)`).
#[allow(clippy::arithmetic_side_effects)]
pub fn accumulate_booleanity_gram_reference<F, Trace>(
    traces: &[Trace],
    row_weights: &[F],
    rho_powers: &[F],
    sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<GramAccumulator<F>, ShaProjectionError>
where
    F: PrimeField,
    Trace: Borrow<ProjectedTrace<F>>,
{
    let n = traces.len();
    if row_weights.len() != SHA_ROW_COUNT {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "row_weights",
            col: 0,
            got: row_weights.len(),
            expected: SHA_ROW_COUNT,
        });
    }
    if rho_powers.len() != sources.len() {
        return Err(ShaProjectionError::ColumnRowCount {
            kind: "rho_powers",
            col: 0,
            got: rho_powers.len(),
            expected: sources.len(),
        });
    }
    let zero = F::zero_with_cfg(field_cfg);
    let one = F::one_with_cfg(field_cfg);
    let needs_virtuals = sources_need_virtuals(sources);
    let mut g = vec![zero.clone(); tri(n.saturating_sub(1), n.saturating_sub(1)) + 1];
    let mut h = vec![zero.clone(); n];
    let mut values = vec![zero.clone(); n];

    for (row, row_weight) in row_weights.iter().enumerate().take(SHA_ROW_COUNT) {
        let virtuals = if needs_virtuals {
            traces
                .iter()
                .map(|trace| {
                    reconstruct_virtual_ch_maj_at_row_unchecked(trace.borrow(), row, field_cfg)
                        .map(Some)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![None; n]
        };
        for (source_idx, source) in sources.iter().enumerate() {
            let mut any_nonzero = false;
            for (j, trace) in traces.iter().enumerate() {
                let value = booleanity_source_value_at_row_with_virtuals(
                    trace.borrow(),
                    row,
                    source,
                    virtuals[j].as_ref(),
                    field_cfg,
                )?;
                any_nonzero |= value != zero;
                values[j] = value;
            }
            if !any_nonzero {
                continue;
            }
            let omega = rho_powers[source_idx].clone() * row_weight;
            for k in 0..n {
                if values[k] == zero {
                    continue;
                }
                let omega_k = if values[k] == one {
                    omega.clone()
                } else {
                    omega.clone() * &values[k]
                };
                h[k] += omega_k.clone();
                for j in 0..=k {
                    if values[j] == zero {
                        continue;
                    }
                    if values[j] == one {
                        g[tri(j, k)] += omega_k.clone();
                    } else {
                        g[tri(j, k)] += omega_k.clone() * &values[j];
                    }
                }
            }
        }
    }

    Ok(GramAccumulator {
        g,
        h,
        n_instances: n,
    })
}

impl<F> GramAccumulator<F>
where
    F: PrimeField + FromWithConfig<u64>,
{
    pub fn n_instances(&self) -> usize {
        self.n_instances
    }

    /// Evaluate the skip-round polynomial `q` at every node of `domain`.
    ///
    /// Domain node `j` is exact for any witness:
    /// `q(u_j) = G[j,j] − h[j]` (zero iff instance `j`'s weighted booleanity
    /// residue vanishes). Off-domain nodes use the quadratic form
    /// `L(x)ᵀ·G·L(x) − hᵀ·L(x)`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn skip_node_values(
        &self,
        domain: &SkipDomain<F>,
        field_cfg: &F::Config,
    ) -> Result<Vec<F>, SumFoldError> {
        let n = self.n_instances;
        if domain.n_instances() != n {
            return Err(SumFoldError::InstanceCountMismatch {
                ell: n,
                got: domain.n_instances(),
                expected: n,
            });
        }
        let mut out = Vec::with_capacity(domain.node_count());
        for j in 0..n {
            out.push(self.g[tri(j, j)].clone() - &self.h[j]);
        }
        for node in &domain.nodes()[n..] {
            let basis = domain.lagrange_at(node, field_cfg);
            let mut acc = F::zero_with_cfg(field_cfg);
            for k in 0..n {
                // Diagonal term L_k²·G[k,k] − L_k·h[k].
                acc += basis[k].clone() * &basis[k] * &self.g[tri(k, k)];
                acc -= basis[k].clone() * &self.h[k];
                // Off-diagonal terms counted twice.
                for j in 0..k {
                    let cross = basis[j].clone() * &basis[k] * &self.g[tri(j, k)];
                    acc += cross.clone() + cross;
                }
            }
            out.push(acc);
        }
        Ok(out)
    }
}

/// The skip-round message: `q` in evaluation basis over the domain's nodes.
/// The first `N` slots are the domain values (all zero for an honest
/// prover); the remaining `N − 1` are the off-domain evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldFirstSkipRoundProof<F> {
    pub node_values: Vec<F>,
}

/// Outputs both parties derive from the accepted skip round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkipRoundVerdict<F> {
    /// Fold weights `θ_j = L_j(α)`.
    pub theta: Vec<F>,
    /// Relaxed folded booleanity residue `B★ = q(α)`.
    pub b_star: F,
    /// The packing challenge `α`.
    pub alpha: F,
}

/// Prover side of the skip round: emit the node values, absorb them, sample
/// `α`, and derive the fold weights and folded booleanity residue.
pub fn prove_skip_round<F>(
    gram: &GramAccumulator<F>,
    domain: &SkipDomain<F>,
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> Result<(FoldFirstSkipRoundProof<F>, SkipRoundVerdict<F>), SumFoldError>
where
    F: PrimeField + FromWithConfig<u64>,
    F::Inner: Transcribable + ConstTranscribable,
    F::Modulus: Transcribable,
{
    let node_values = gram.skip_node_values(domain, field_cfg)?;
    let verdict = bind_skip_round(&node_values, domain, transcript, field_cfg)?;
    Ok((FoldFirstSkipRoundProof { node_values }, verdict))
}

/// Verifier side of the skip round: check the message length and the
/// γ-weighted zerocheck over the domain slots, then mirror the prover's
/// transcript interaction.
///
/// The γ-weights are soundness-load-bearing: an unweighted sum over the
/// domain admits cross-instance cancellation of nonzero booleanity residues.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_skip_round<F>(
    proof: &FoldFirstSkipRoundProof<F>,
    gamma: &F,
    domain: &SkipDomain<F>,
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> Result<SkipRoundVerdict<F>, SumFoldError>
where
    F: PrimeField + FromWithConfig<u64>,
    F::Inner: Transcribable + ConstTranscribable,
    F::Modulus: Transcribable,
{
    if proof.node_values.len() != domain.node_count() {
        return Err(SumFoldError::SkipRoundMessageLength {
            got: proof.node_values.len(),
            expected: domain.node_count(),
        });
    }
    let n = domain.n_instances();
    let gamma_powers = powers(gamma.clone(), F::one_with_cfg(field_cfg), n);
    let mut weighted = F::zero_with_cfg(field_cfg);
    for (value, weight) in proof.node_values.iter().take(n).zip(&gamma_powers) {
        weighted += value.clone() * weight;
    }
    if weighted != F::zero_with_cfg(field_cfg) {
        return Err(SumFoldError::SkipRoundZeroCheckFailed);
    }
    bind_skip_round(&proof.node_values, domain, transcript, field_cfg)
}

fn bind_skip_round<F>(
    node_values: &[F],
    domain: &SkipDomain<F>,
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> Result<SkipRoundVerdict<F>, SumFoldError>
where
    F: PrimeField + FromWithConfig<u64>,
    F::Inner: Transcribable + ConstTranscribable,
    F::Modulus: Transcribable,
{
    transcript.absorb_random_field_slice_owned(node_values);
    let alpha: F = transcript.get_field_challenge(field_cfg);
    let theta = domain.lagrange_at(&alpha, field_cfg);
    let b_star = domain.interpolate(node_values, &alpha, field_cfg)?;
    Ok(SkipRoundVerdict {
        theta,
        b_star,
        alpha,
    })
}

/// Fold the binary bit-slice tables with `θ` through the instance masks:
/// `folded[slot][row] = Σ_{j: mask bit set} θ_j` (unreduced 5-limb subset
/// sums, one Barrett per output), with per-(slot, row) fallback to the naive
/// weighted fold for non-binary values.
#[allow(clippy::arithmetic_side_effects)]
fn fold_bit_slices_with_masks<F, Trace>(
    traces: &[Trace],
    masks: &ShaSourceMasks,
    theta: &[F],
    field_cfg: &F::Config,
) -> MleTable<F>
where
    F: PrimeField + MontgomeryLimbs + Send + Sync,
    F::Config: Sync,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
{
    let reducer = BarrettDelayedReduction::<F>::new(field_cfg);
    let slots = ShaWordCol::COUNT * SHA_WORD_BITS;
    let fold_slot = |slot: usize| -> MleColumn<F> {
        let mut evaluations = Vec::with_capacity(SHA_ROW_COUNT);
        for row in 0..SHA_ROW_COUNT {
            let idx = slot * SHA_ROW_COUNT + row;
            if masks.non_binary[idx] {
                let mut acc = F::zero_with_cfg(field_cfg);
                for (trace, weight) in traces.iter().zip(theta) {
                    let value = &trace.borrow().bit_slices[slot].evaluations[row];
                    if !F::is_zero(value) {
                        acc += weight.clone() * value;
                    }
                }
                evaluations.push(acc);
                continue;
            }
            let mut mask = masks.real[idx];
            if mask == 0 {
                evaluations.push(F::zero_with_cfg(field_cfg));
                continue;
            }
            let mut acc = F::zero_with_cfg(field_cfg);
            if reducer.flush_adds() <= 1 {
                while mask != 0 {
                    let j = mask.trailing_zeros() as usize;
                    acc += theta[j].clone();
                    mask &= mask - 1;
                }
            } else {
                let mut bucket = Uint::<5>::zero();
                while mask != 0 {
                    let j = mask.trailing_zeros() as usize;
                    reducer.add(&mut bucket, &theta[j]);
                    mask &= mask - 1;
                }
                if !bucket.is_zero() {
                    acc = reducer.reduce(bucket);
                }
            }
            evaluations.push(acc);
        }
        MleColumn {
            evaluations,
            num_vars: SHA_ROW_VARS,
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..slots).into_par_iter().map(fold_slot).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..slots).map(fold_slot).collect()
    }
}

/// [`fold_projected_traces_with_weights`] with the binary bit-slice fold
/// routed through pre-built instance masks (O-10 reuse: the same masks the
/// Gram pass consumes).
#[allow(clippy::arithmetic_side_effects)]
pub fn fold_projected_traces_with_theta_masks<F, Trace, Public>(
    traces: &[Trace],
    publics: &[Public],
    theta: &[F],
    masks: &ShaSourceMasks,
    field_cfg: &F::Config,
) -> Result<(ProjectionFoldWitness<F>, ProjectedPublic<F>), ShaProjectionError>
where
    F: ShaBinaryFoldField + MontgomeryLimbs,
    F::Config: Sync,
    Trace: Borrow<ProjectedTrace<F>> + Sync,
    Public: Borrow<ProjectedPublic<F>>,
{
    if traces.len() != publics.len() {
        return Err(ShaProjectionError::InstanceCountMismatch {
            got: publics.len(),
            expected: traces.len(),
        });
    }
    if theta.len() != traces.len() {
        return Err(ShaProjectionError::FoldingWeightCount {
            got: theta.len(),
            expected: traces.len(),
        });
    }

    let bit_slices = fold_bit_slices_with_masks(traces, masks, theta, field_cfg);
    let folded_public_columns = fold_mle_tables(
        "public.columns",
        publics.iter().map(|public| &public.borrow().columns),
        theta,
        field_cfg,
    )?;
    let folded_trace = ProjectedTrace {
        bit_slices,
        scalarized: fold_mle_tables(
            "scalarized",
            traces.iter().map(|trace| &trace.borrow().scalarized),
            theta,
            field_cfg,
        )?,
        int_columns: fold_mle_tables(
            "int_columns",
            traces.iter().map(|trace| &trace.borrow().int_columns),
            theta,
            field_cfg,
        )?,
        public_columns: folded_public_columns.clone(),
    };
    let folded_public = ProjectedPublic {
        columns: folded_public_columns,
        bit_slices: fold_optional_binary_mle_tables(
            "public.bit_slices",
            publics
                .iter()
                .map(|public| public.borrow().bit_slices.as_ref()),
            theta,
            field_cfg,
        )?,
    };

    Ok((
        ProjectionFoldWitness {
            trace: folded_trace,
        },
        folded_public,
    ))
}

/// Fold-first orchestration errors.
#[derive(Debug, Error)]
pub enum FoldFirstError {
    #[error(transparent)]
    Projection(#[from] ShaProjectionError),
    #[error(transparent)]
    SumFold(#[from] SumFoldError),
    #[error("fold-first row sumcheck verification failed")]
    RowSumcheck,
    #[error("fold-first row sumcheck claimed sum does not match the assembled target")]
    TargetMismatch,
}

/// `Σ_{f∈F≠0} λ^f · E'_f(a)` — the linear part of the assembled
/// row-sumcheck target, computed from the transmitted folded ideal
/// polynomials.
#[allow(clippy::arithmetic_side_effects)]
pub fn sha_nonzero_target_at<F>(
    ideal_polys: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    a: &F,
    lambda: &F,
    field_cfg: &F::Config,
) -> F
where
    F: PrimeField,
{
    let zero = F::zero_with_cfg(field_cfg);
    let lambda_powers = powers(
        lambda.clone(),
        F::one_with_cfg(field_cfg),
        NUM_SHA_RESIDUAL_FAMILIES,
    );
    let mut acc = zero.clone();
    for (slot, family) in NONZERO_SHA_FAMILIES.iter().enumerate() {
        let mut value = zero.clone();
        for coeff in ideal_polys[slot].coeffs.iter().rev() {
            value = value * a + coeff;
        }
        acc += lambda_powers[family.index()].clone() * value;
    }
    acc
}

/// The fold-first SumFold proof: skip-round message, folded ideal
/// polynomials, and the folded row sumcheck.
#[derive(Clone, Debug)]
pub struct FoldFirstSumFoldProof<F: PrimeField> {
    pub skip_round: FoldFirstSkipRoundProof<F>,
    pub folded_ideal_polys: [DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    pub row_sumcheck: MultiDegreeSumcheckProof<F>,
}

/// Prover-side outputs alongside the proof.
#[derive(Clone, Debug)]
pub struct FoldFirstProverArtifacts<F: PrimeField> {
    pub theta: Vec<F>,
    pub folded_witness: ProjectionFoldWitness<F>,
    pub folded_public: ProjectedPublic<F>,
    pub b_star: F,
    pub target: F,
}

/// Verifier-side outputs: the fold weights, the assembled target, and the
/// row-sumcheck endpoint claim to be discharged by the opening layer.
#[derive(Clone, Debug)]
pub struct FoldFirstVerifierClaims<F: PrimeField> {
    pub theta: Vec<F>,
    pub b_star: F,
    pub target: F,
    pub row_point: Vec<F>,
    pub expected_row_eval: F,
}

pub fn absorb_fold_first_ideal_polys<F>(
    ideal_polys: &[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES],
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) where
    F: PrimeField + FromWithConfig<u64>,
    F::Inner: Transcribable,
    F::Modulus: Transcribable,
{
    for poly in ideal_polys {
        let len = u64::try_from(poly.coeffs.len()).expect("coefficient count fits u64");
        transcript
            .absorb_random_field_owned(&<F as FromWithConfig<u64>>::from_with_cfg(len, field_cfg));
        transcript.absorb_random_field_slice_owned(&poly.coeffs);
    }
}

/// Prover side of the fold-first SumFold (V2) flow:
/// r_ic, ρ, γ → booleanity skip round → θ-fold → folded IdealCheck →
/// a, λ, ξ → folded row sumcheck with the assembled target
/// `T' = Σ λ^f E'_f(a) + ξ·B★`.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_fold_first_sha_sumfold<F>(
    traces: &[ProjectedTrace<F>],
    publics: &[ProjectedPublic<F>],
    sources: &[ShaBooleanitySource],
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> Result<(FoldFirstSumFoldProof<F>, FoldFirstProverArtifacts<F>), FoldFirstError>
where
    F: ShaBinaryFoldField
        + InnerTransparentField
        + DelayedFieldProductSum
        + MontgomeryLimbs
        + FromWithConfig<u64>
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Config: Sync,
    F::Inner: Transcribable + ConstTranscribable,
    F::Modulus: Transcribable,
{
    let domain = SkipDomain::new(traces.len(), field_cfg)?;
    let r_ic: [F; SHA_ROW_VARS] =
        std::array::from_fn(|_| transcript.get_field_challenge(field_cfg));
    let rho: F = transcript.get_field_challenge(field_cfg);
    // γ is the verifier's zerocheck batching challenge; the prover samples it
    // only to keep the transcripts aligned.
    let _gamma: F = transcript.get_field_challenge(field_cfg);

    let row_weights = build_eq_x_r_vec(&r_ic, field_cfg).map_err(ShaProjectionError::from)?;
    let rho_powers = powers(rho.clone(), F::one_with_cfg(field_cfg), sources.len());
    let masks = ShaSourceMasks::build(traces, field_cfg);
    let gram = match &masks {
        Some(masks) => accumulate_booleanity_gram_with_masks(
            traces,
            masks,
            &row_weights,
            &rho_powers,
            sources,
            field_cfg,
        )?,
        None => accumulate_booleanity_gram_reference(
            traces,
            &row_weights,
            &rho_powers,
            sources,
            field_cfg,
        )?,
    };
    let (skip_round, verdict) = prove_skip_round(&gram, &domain, transcript, field_cfg)?;

    let (mut folded_witness, folded_public) = match &masks {
        Some(masks) => fold_projected_traces_with_theta_masks(
            traces,
            publics,
            &verdict.theta,
            masks,
            field_cfg,
        )?,
        None => fold_projected_traces_with_weights(traces, publics, &verdict.theta, field_cfg)?,
    };
    let folded_ideal_polys =
        build_sha_ideal_values_at_point(&folded_witness.trace, &folded_public, &r_ic, field_cfg)?;
    absorb_fold_first_ideal_polys(&folded_ideal_polys, transcript, field_cfg);

    let a: F = transcript.get_field_challenge(field_cfg);
    let lambda: F = transcript.get_field_challenge(field_cfg);
    let xi: F = transcript.get_field_challenge(field_cfg);

    // Scalarization happens exactly once, on the folded trace, after `a`.
    folded_witness.trace.scalarized =
        scalarize_bit_slices(&folded_witness.trace.bit_slices, &a, field_cfg)?;

    let target = sha_nonzero_target_at(&folded_ideal_polys, &a, &lambda, field_cfg)
        + xi.clone() * &verdict.b_star;

    let integrand = folded_row_integrand_values(
        &folded_witness.trace,
        &folded_public,
        &r_ic,
        &a,
        &lambda,
        &rho,
        &xi,
        sources,
        field_cfg,
    )?;
    // Completeness identity (tested, not asserted, so tamper tests can drive
    // this prover with invalid witnesses): for traces whose zero-family
    // residuals vanish, `folded_row_integrand_sum(&integrand) == target`.
    let group = build_folded_row_sumcheck_group(&integrand, field_cfg)?;
    let (row_sumcheck, _) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], SHA_ROW_VARS, field_cfg);

    Ok((
        FoldFirstSumFoldProof {
            skip_round,
            folded_ideal_polys,
            row_sumcheck,
        },
        FoldFirstProverArtifacts {
            theta: verdict.theta,
            folded_witness,
            folded_public,
            b_star: verdict.b_star,
            target,
        },
    ))
}

/// Verifier side of the fold-first SumFold (V2) flow. Mirrors the prover's
/// transcript, enforces the γ-weighted zerocheck, the folded ideal
/// membership, and the row-sumcheck target `T' = Σ λ^f E'_f(a) + ξ·B★`, and
/// returns the endpoint claim for the opening layer.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_fold_first_sha_sumfold<F>(
    proof: &FoldFirstSumFoldProof<F>,
    n_instances: usize,
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> Result<FoldFirstVerifierClaims<F>, FoldFirstError>
where
    F: PrimeField
        + InnerTransparentField
        + FromWithConfig<u64>
        + FromPrimitiveWithConfig
        + Send
        + Sync,
    F::Inner: Transcribable + ConstTranscribable,
    F::Modulus: Transcribable,
{
    let domain = SkipDomain::new(n_instances, field_cfg)?;
    let _r_ic: [F; SHA_ROW_VARS] =
        std::array::from_fn(|_| transcript.get_field_challenge(field_cfg));
    let _rho: F = transcript.get_field_challenge(field_cfg);
    let gamma: F = transcript.get_field_challenge(field_cfg);

    let verdict = verify_skip_round(&proof.skip_round, &gamma, &domain, transcript, field_cfg)?;

    verify_fresh_sha_ideal_polys(std::slice::from_ref(&proof.folded_ideal_polys), field_cfg)?;
    absorb_fold_first_ideal_polys(&proof.folded_ideal_polys, transcript, field_cfg);

    let a: F = transcript.get_field_challenge(field_cfg);
    let lambda: F = transcript.get_field_challenge(field_cfg);
    let xi: F = transcript.get_field_challenge(field_cfg);

    let target = sha_nonzero_target_at(&proof.folded_ideal_polys, &a, &lambda, field_cfg)
        + xi * &verdict.b_star;

    let subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
        transcript,
        SHA_ROW_VARS,
        &proof.row_sumcheck,
        field_cfg,
    )
    .map_err(|_| FoldFirstError::RowSumcheck)?;
    let claimed_sums = proof.row_sumcheck.claimed_sums();
    if claimed_sums.len() != 1 {
        return Err(FoldFirstError::RowSumcheck);
    }
    verify_folded_row_sumcheck_claim(&claimed_sums[0], &target)
        .map_err(|_| FoldFirstError::TargetMismatch)?;

    Ok(FoldFirstVerifierClaims {
        theta: verdict.theta,
        b_star: verdict.b_star,
        target,
        row_point: subclaims.point().to_vec(),
        expected_row_eval: subclaims.expected_evaluations()[0].clone(),
    })
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::neutron_nova::projection_sha::{
        MleTable, ProjectedPublic, SHA_ROW_VARS, SHA_WORD_BITS, ShaIntCol, ShaPublicCol,
        ShaWordCol, bit_slice_index, scalarize_bit_slices,
    };
    use crate::test_utils::test_config;
    use crypto_primitives::crypto_bigint_monty::MontyField;
    use zinc_poly::mle::DenseMultilinearExtension;
    use zinc_poly::utils::build_eq_x_r_vec;
    use zinc_transcript::Blake3Transcript;

    type F = MontyField<4>;

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &test_config())
    }

    fn mle_table_from_columns(columns: Vec<Vec<F>>) -> MleTable<F> {
        columns
            .into_iter()
            .map(|evaluations| DenseMultilinearExtension {
                evaluations,
                num_vars: SHA_ROW_VARS,
            })
            .collect()
    }

    fn zero_table(cols: usize) -> MleTable<F> {
        let cfg = test_config();
        mle_table_from_columns(vec![vec![F::zero_with_cfg(&cfg); SHA_ROW_COUNT]; cols])
    }

    fn flatten_bits(bits: Vec<Vec<Vec<F>>>) -> MleTable<F> {
        let mut flattened = (0..bits.len() * SHA_WORD_BITS)
            .map(|_| Vec::new())
            .collect::<Vec<_>>();
        for (col_idx, rows) in bits.into_iter().enumerate() {
            for row in rows {
                for (bit_idx, value) in row.into_iter().enumerate() {
                    flattened[bit_slice_index(col_idx, bit_idx, SHA_WORD_BITS)].push(value);
                }
            }
        }
        mle_table_from_columns(flattened)
    }

    pub(crate) fn synthetic_boolean_trace(instance_idx: u64) -> ProjectedTrace<F> {
        let cfg = test_config();
        let zero = F::zero_with_cfg(&cfg);
        let mut bits =
            vec![vec![vec![zero.clone(); SHA_WORD_BITS]; SHA_ROW_COUNT]; ShaWordCol::COUNT];
        for (col_idx, col) in bits.iter_mut().enumerate() {
            for (row_idx, row) in col.iter_mut().enumerate() {
                for (bit_idx, bit) in row.iter_mut().enumerate() {
                    let selector = instance_idx
                        + u64::try_from(col_idx * 17 + row_idx * 3 + bit_idx)
                            .expect("selector fits u64");
                    if selector % 2 == 1 {
                        *bit = f(1);
                    }
                }
            }
        }
        let bit_slices = flatten_bits(bits);
        let scalarized = scalarize_bit_slices(&bit_slices, &f(3), &cfg).unwrap();
        ProjectedTrace {
            bit_slices,
            scalarized,
            int_columns: zero_table(ShaIntCol::COUNT),
            public_columns: zero_table(ShaPublicCol::COUNT),
        }
    }

    /// A trace whose bits are pseudo-random 0/1 AND whose auxiliary columns
    /// (`Uef`, `UNegEg`, `Ch2Comp`, `Maj`, `MajComp`) satisfy the defining
    /// relations that make the virtual Ch/Maj booleanity sources 0/1-valued:
    ///
    /// - `ch1(z) = E[z+2] ⊕ E[z+1]` via `Uef[r] = E[r]·E[r−1]`,
    /// - `ch2(z) = E[z+2] ⊕ E[z]` via `UNegEg[r] = (1−E[r])·E[r−2]`,
    /// - `maj(z) = (A[z]+A[z+1]+A[z+2]) mod 2` via `Maj[r] = majority`.
    ///
    /// The last two rows of `A` and `E` are zeroed so the no-wrap boundary
    /// reads stay boolean.
    pub(crate) fn boolean_virtuals_trace(instance_idx: u64) -> ProjectedTrace<F> {
        let cfg = test_config();
        let bit_of = |col: usize, row: usize, bit: usize| -> u64 {
            let selector =
                instance_idx + u64::try_from(col * 17 + row * 3 + bit).expect("selector fits u64");
            selector % 2
        };
        let mut bits = vec![vec![[0u64; SHA_WORD_BITS]; SHA_ROW_COUNT]; ShaWordCol::COUNT];
        for col_idx in 0..ShaWordCol::COUNT {
            let col = ShaWordCol::ALL[col_idx];
            if matches!(
                col,
                ShaWordCol::Uef
                    | ShaWordCol::UNegEg
                    | ShaWordCol::Ch2Comp
                    | ShaWordCol::Maj
                    | ShaWordCol::MajComp
            ) {
                continue;
            }
            for row in 0..SHA_ROW_COUNT {
                for bit in 0..SHA_WORD_BITS {
                    bits[col_idx][row][bit] = bit_of(col_idx, row, bit);
                }
            }
        }
        // Boundary hygiene: zero the last two rows of A and E.
        for row in [SHA_ROW_COUNT - 2, SHA_ROW_COUNT - 1] {
            bits[ShaWordCol::A.index()][row] = [0; SHA_WORD_BITS];
            bits[ShaWordCol::E.index()][row] = [0; SHA_WORD_BITS];
        }
        // Derived columns keeping the virtual sources boolean.
        let e_bits = bits[ShaWordCol::E.index()].clone();
        let a_bits = bits[ShaWordCol::A.index()].clone();
        for row in 0..SHA_ROW_COUNT {
            for bit in 0..SHA_WORD_BITS {
                let e = |r: usize| -> u64 { e_bits[r][bit] };
                let a = |r: usize| -> u64 { a_bits[r][bit] };
                let uef = if row >= 1 { e(row) * e(row - 1) } else { 0 };
                let uneg = if row >= 2 {
                    (1 - e(row)) * e(row - 2)
                } else {
                    0
                };
                let maj_sum = a(row)
                    + if row >= 1 { a(row - 1) } else { 0 }
                    + if row >= 2 { a(row - 2) } else { 0 };
                bits[ShaWordCol::Uef.index()][row][bit] = uef;
                bits[ShaWordCol::UNegEg.index()][row][bit] = uneg;
                bits[ShaWordCol::Maj.index()][row][bit] = maj_sum / 2;
                // Ch2Comp and MajComp stay zero.
            }
        }
        let field_bits = bits
            .into_iter()
            .map(|rows| {
                rows.into_iter()
                    .map(|row| row.into_iter().map(f).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let bit_slices = flatten_bits(field_bits);
        let scalarized = scalarize_bit_slices(&bit_slices, &f(3), &cfg).unwrap();
        ProjectedTrace {
            bit_slices,
            scalarized,
            int_columns: zero_table(ShaIntCol::COUNT),
            public_columns: zero_table(ShaPublicCol::COUNT),
        }
    }

    pub(crate) fn zero_trace() -> ProjectedTrace<F> {
        let cfg = test_config();
        let zero = F::zero_with_cfg(&cfg);
        let bits = vec![vec![vec![zero; SHA_WORD_BITS]; SHA_ROW_COUNT]; ShaWordCol::COUNT];
        let bit_slices = flatten_bits(bits);
        let scalarized = scalarize_bit_slices(&bit_slices, &f(3), &test_config()).unwrap();
        ProjectedTrace {
            bit_slices,
            scalarized,
            int_columns: zero_table(ShaIntCol::COUNT),
            public_columns: zero_table(ShaPublicCol::COUNT),
        }
    }

    pub(crate) fn zero_public() -> ProjectedPublic<F> {
        ProjectedPublic {
            columns: zero_table(ShaPublicCol::COUNT),
            bit_slices: None,
        }
    }

    fn small_sources() -> Vec<ShaBooleanitySource> {
        vec![
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::A,
                bit: 0,
            },
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::E,
                bit: 7,
            },
            ShaBooleanitySource::WordBit {
                col: ShaWordCol::W,
                bit: 13,
            },
            ShaBooleanitySource::VirtualCh1 { bit: 2 },
            ShaBooleanitySource::VirtualMaj { bit: 5 },
        ]
    }

    fn test_row_weights(cfg: &<F as PrimeField>::Config) -> Vec<F> {
        let r_ic: Vec<F> = (0..SHA_ROW_VARS as u64).map(|i| f(11 + 2 * i)).collect();
        build_eq_x_r_vec(&r_ic, cfg).unwrap()
    }

    /// Direct per-instance booleanity residue Σ_{q,z} ω·d(d−1) for instance j.
    fn direct_instance_residue(
        trace: &ProjectedTrace<F>,
        row_weights: &[F],
        rho_powers: &[F],
        sources: &[ShaBooleanitySource],
    ) -> F {
        let cfg = test_config();
        let one = F::one_with_cfg(&cfg);
        let mut acc = F::zero_with_cfg(&cfg);
        for (row, row_weight) in row_weights.iter().enumerate() {
            let virtuals = if sources_need_virtuals(sources) {
                Some(reconstruct_virtual_ch_maj_at_row_unchecked(trace, row, &cfg).unwrap())
            } else {
                None
            };
            for (idx, source) in sources.iter().enumerate() {
                let d = booleanity_source_value_at_row_with_virtuals(
                    trace,
                    row,
                    source,
                    virtuals.as_ref(),
                    &cfg,
                )
                .unwrap();
                let term = d.clone() * (d - &one);
                acc += rho_powers[idx].clone() * row_weight * term;
            }
        }
        acc
    }

    /// Naive oracle: extend each per-(q,z) instance-value vector to `x` via
    /// Lagrange over the domain nodes and sum ω·v(v−1).
    fn naive_q_at(
        traces: &[ProjectedTrace<F>],
        row_weights: &[F],
        rho_powers: &[F],
        sources: &[ShaBooleanitySource],
        domain: &SkipDomain<F>,
        x: &F,
    ) -> F {
        let cfg = test_config();
        let one = F::one_with_cfg(&cfg);
        let basis = domain.lagrange_at(x, &cfg);
        let mut acc = F::zero_with_cfg(&cfg);
        for (row, row_weight) in row_weights.iter().enumerate() {
            let virtuals: Vec<_> = traces
                .iter()
                .map(|trace| {
                    if sources_need_virtuals(sources) {
                        Some(reconstruct_virtual_ch_maj_at_row_unchecked(trace, row, &cfg).unwrap())
                    } else {
                        None
                    }
                })
                .collect();
            for (idx, source) in sources.iter().enumerate() {
                let mut extended = F::zero_with_cfg(&cfg);
                for (j, trace) in traces.iter().enumerate() {
                    let d = booleanity_source_value_at_row_with_virtuals(
                        trace,
                        row,
                        source,
                        virtuals[j].as_ref(),
                        &cfg,
                    )
                    .unwrap();
                    extended += basis[j].clone() * d;
                }
                let term = extended.clone() * (extended - &one);
                acc += rho_powers[idx].clone() * row_weight * term;
            }
        }
        acc
    }

    #[test]
    fn skip_domain_nodes_are_distinct_symmetric_integers() {
        let cfg = test_config();
        for n in [2usize, 3, 4, 5, 8] {
            let domain = SkipDomain::<F>::new(n, &cfg).unwrap();
            assert_eq!(domain.node_count(), 2 * n - 1);
            let nodes = domain.nodes();
            for i in 0..nodes.len() {
                for j in 0..i {
                    assert_ne!(nodes[i], nodes[j], "duplicate node at n={n}");
                }
            }
            // Domain is centred: contains 0.
            assert!(domain.domain_nodes().contains(&F::zero_with_cfg(&cfg)));
        }
        assert!(SkipDomain::<F>::new(1, &cfg).is_err());
    }

    #[test]
    fn lagrange_weights_are_indicators_on_domain_and_sum_to_one() {
        let cfg = test_config();
        let domain = SkipDomain::<F>::new(4, &cfg).unwrap();
        let one = F::one_with_cfg(&cfg);
        let zero = F::zero_with_cfg(&cfg);
        for (j, node) in domain.domain_nodes().to_vec().iter().enumerate() {
            let basis = domain.lagrange_at(node, &cfg);
            for (i, value) in basis.iter().enumerate() {
                assert_eq!(*value, if i == j { one.clone() } else { zero.clone() });
            }
        }
        // Partition of unity at an off-domain point.
        let x = f(97);
        let basis = domain.lagrange_at(&x, &cfg);
        let sum = basis
            .iter()
            .fold(zero.clone(), |acc, value| acc + value.clone());
        assert_eq!(sum, one);
    }

    #[test]
    fn interpolate_matches_polynomial_evaluation() {
        let cfg = test_config();
        let domain = SkipDomain::<F>::new(3, &cfg).unwrap();
        // p(Y) = 3Y² + 2Y + 7 (degree 2 ≤ 2N−2 = 4).
        let p = |y: &F| f(3) * y * y + f(2) * y + f(7);
        let node_values: Vec<F> = domain.nodes().iter().map(p).collect();
        let x = f(123_456);
        let expected = p(&x);
        let got = domain.interpolate(&node_values, &x, &cfg).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn honest_bits_give_zero_domain_values() {
        let cfg = test_config();
        let traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        let row_weights = test_row_weights(&cfg);
        let sources = small_sources();
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let domain = SkipDomain::<F>::new(traces.len(), &cfg).unwrap();
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();
        let values = gram.skip_node_values(&domain, &cfg).unwrap();
        let zero = F::zero_with_cfg(&cfg);
        for value in values.iter().take(traces.len()) {
            assert_eq!(*value, zero);
        }
    }

    #[test]
    fn gram_diagonal_equals_h_for_honest_bits() {
        let cfg = test_config();
        let traces: Vec<_> = (0..3).map(boolean_virtuals_trace).collect();
        let row_weights = test_row_weights(&cfg);
        let sources = small_sources();
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();
        for j in 0..traces.len() {
            assert_eq!(gram.g[tri(j, j)], gram.h[j]);
        }
    }

    #[test]
    fn gram_matches_naive_q_at_all_nodes() {
        let cfg = test_config();
        let traces: Vec<_> = (0..4).map(synthetic_boolean_trace).collect();
        let row_weights = test_row_weights(&cfg);
        let sources = small_sources();
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let domain = SkipDomain::<F>::new(traces.len(), &cfg).unwrap();
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();
        let values = gram.skip_node_values(&domain, &cfg).unwrap();
        for (node, value) in domain.nodes().to_vec().iter().zip(&values) {
            let expected = naive_q_at(&traces, &row_weights, &rho_powers, &sources, &domain, node);
            assert_eq!(*value, expected);
        }
        // Also check an arbitrary off-node point through interpolation.
        let x = f(777);
        let expected = naive_q_at(&traces, &row_weights, &rho_powers, &sources, &domain, &x);
        let got = domain.interpolate(&values, &x, &cfg).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn tampered_bit_gives_matching_nonzero_domain_value() {
        let cfg = test_config();
        let mut traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        // Set a bit participating in `small_sources` to a non-boolean value.
        let idx = bit_slice_index(ShaWordCol::A.index(), 0, SHA_WORD_BITS);
        traces[2].bit_slices[idx].evaluations[9] = f(2);
        let row_weights = test_row_weights(&cfg);
        let sources = small_sources();
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let domain = SkipDomain::<F>::new(traces.len(), &cfg).unwrap();
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();
        let values = gram.skip_node_values(&domain, &cfg).unwrap();
        let zero = F::zero_with_cfg(&cfg);
        let expected = direct_instance_residue(&traces[2], &row_weights, &rho_powers, &sources);
        assert_ne!(expected, zero);
        assert_eq!(values[2], expected);
        for j in [0usize, 1, 3] {
            assert_eq!(values[j], zero);
        }
    }

    #[test]
    fn b_star_equals_folded_booleanity_row_sum() {
        use crate::neutron_nova::projection_sha::fold_projected_traces_with_weights;
        let cfg = test_config();
        let traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        let publics: Vec<_> = (0..4).map(|_| zero_public()).collect();
        let row_weights = test_row_weights(&cfg);
        let sources = small_sources();
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let domain = SkipDomain::<F>::new(traces.len(), &cfg).unwrap();
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();

        let mut transcript = Blake3Transcript::new();
        let (_, verdict) = prove_skip_round(&gram, &domain, &mut transcript, &cfg).unwrap();

        // Fold with the Lagrange weights θ = L(α); the booleanity residue of
        // the folded trace must equal q(α): packing evaluation commutes with
        // folding because D̂_q(α, z) = Σ_j L_j(α)·D_{j,q}(z) = D'_q(z).
        let (folded, _folded_public) =
            fold_projected_traces_with_weights(&traces, &publics, &verdict.theta, &cfg).unwrap();
        let direct = direct_instance_residue(&folded.trace, &row_weights, &rho_powers, &sources);
        assert_eq!(verdict.b_star, direct);
    }

    #[test]
    fn skip_round_roundtrip_agrees() {
        let cfg = test_config();
        let traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        let row_weights = test_row_weights(&cfg);
        let sources = small_sources();
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let domain = SkipDomain::<F>::new(traces.len(), &cfg).unwrap();
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        let (proof, prover_verdict) =
            prove_skip_round(&gram, &domain, &mut prover_transcript, &cfg).unwrap();

        let gamma = f(41);
        let mut verifier_transcript = Blake3Transcript::new();
        let verifier_verdict =
            verify_skip_round(&proof, &gamma, &domain, &mut verifier_transcript, &cfg).unwrap();

        assert_eq!(prover_verdict, verifier_verdict);
        let one = F::one_with_cfg(&cfg);
        let sum = verifier_verdict
            .theta
            .iter()
            .fold(F::zero_with_cfg(&cfg), |acc, value| acc + value.clone());
        assert_eq!(sum, one);
    }

    #[test]
    fn gamma_check_rejects_tampered_witness() {
        let cfg = test_config();
        let mut traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        let idx = bit_slice_index(ShaWordCol::A.index(), 0, SHA_WORD_BITS);
        traces[1].bit_slices[idx].evaluations[4] = f(3);
        let row_weights = test_row_weights(&cfg);
        let sources = small_sources();
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let domain = SkipDomain::<F>::new(traces.len(), &cfg).unwrap();
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        let (proof, _) = prove_skip_round(&gram, &domain, &mut prover_transcript, &cfg).unwrap();

        let gamma = f(41);
        let mut verifier_transcript = Blake3Transcript::new();
        let err = verify_skip_round(&proof, &gamma, &domain, &mut verifier_transcript, &cfg)
            .expect_err("tampered witness must fail the gamma zero-check");
        assert!(matches!(err, SumFoldError::SkipRoundZeroCheckFailed));
    }

    // Prove-side microbenchmark comparing the V2 SumFold block against the V1
    // block it replaces, on identical synthetic traces with the full 640
    // production booleanity sources. Run with:
    //   cargo test -p zinc-piop --release fold_first::tests::bench_ -- \
    //       --ignored --nocapture --test-threads=1
    // NOTE: this V2 path is the correctness-first reference (Gram *structure*
    // O-1, but not the bit-packed O-10 / DMR O-14 arithmetic), so these
    // numbers are an upper bound on the design's eventual prover time.
    #[test]
    #[ignore = "microbenchmark; run explicitly in --release"]
    fn bench_v1_vs_v2_sumfold_block() {
        use crate::neutron_nova::projection_sha::{
            build_production_sha_sumfold_group_owned, production_sha_booleanity_sources,
        };
        use std::time::Instant;

        let cfg = test_config();
        let sources = production_sha_booleanity_sources();
        eprintln!("booleanity sources: {}", sources.len());
        eprintln!(
            "{:>4}  {:>12}  {:>12}  {:>12}",
            "N", "gram(ms)", "v2_prove(ms)", "v1_prove(ms)"
        );

        for log_n in 3u32..=5 {
            let n = 1usize << log_n;
            let traces: Vec<_> = (0..n as u64).map(boolean_virtuals_trace).collect();
            let publics: Vec<_> = (0..n).map(|_| zero_public()).collect();
            let r_ic: [F; SHA_ROW_VARS] = std::array::from_fn(|i| f(11 + 2 * i as u64));
            let row_weights = build_eq_x_r_vec(&r_ic, &cfg).unwrap();
            let rho = f(29);
            let rho_powers = powers(rho.clone(), F::one_with_cfg(&cfg), sources.len());

            let best = |mut run: Box<dyn FnMut()>, iters: usize| -> f64 {
                run(); // warm
                let mut best = f64::MAX;
                for _ in 0..iters {
                    let start = Instant::now();
                    run();
                    best = best.min(start.elapsed().as_secs_f64() * 1e3);
                }
                best
            };

            let gram_ms = {
                let traces = traces.clone();
                let row_weights = row_weights.clone();
                let rho_powers = rho_powers.clone();
                let sources = sources.clone();
                best(
                    Box::new(move || {
                        accumulate_booleanity_gram(
                            &traces,
                            &row_weights,
                            &rho_powers,
                            &sources,
                            &cfg,
                        )
                        .unwrap();
                    }),
                    5,
                )
            };

            let v2_ms = {
                let traces = traces.clone();
                let publics = publics.clone();
                let sources = sources.clone();
                best(
                    Box::new(move || {
                        let mut transcript = Blake3Transcript::new();
                        prove_fold_first_sha_sumfold(
                            &traces,
                            &publics,
                            &sources,
                            &mut transcript,
                            &cfg,
                        )
                        .unwrap();
                    }),
                    3,
                )
            };

            let v1_ms = {
                let traces = traces.clone();
                let publics = publics.clone();
                let sources = sources.clone();
                let rho = rho.clone();
                let beta: Vec<F> = (0..u64::from(log_n)).map(|i| f(5 + 2 * i)).collect();
                best(
                    Box::new(move || {
                        let mut transcript = Blake3Transcript::new();
                        let group = build_production_sha_sumfold_group_owned(
                            traces.clone().into_boxed_slice(),
                            &publics,
                            &beta,
                            &r_ic,
                            &f(3),
                            &f(23),
                            &rho,
                            &f(31),
                            &sources,
                            3,
                            &cfg,
                        )
                        .unwrap();
                        MultiDegreeSumcheck::prove_as_subprotocol(
                            &mut transcript,
                            vec![group],
                            log_n as usize,
                            &cfg,
                        );
                    }),
                    3,
                )
            };

            eprintln!("{n:>4}  {gram_ms:>12.2}  {v2_ms:>12.2}  {v1_ms:>12.2}");
        }
    }

    fn assert_gram_matches_reference(
        traces: &[ProjectedTrace<F>],
        sources: &[ShaBooleanitySource],
    ) {
        let cfg = test_config();
        let row_weights = test_row_weights(&cfg);
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let fast =
            accumulate_booleanity_gram(traces, &row_weights, &rho_powers, sources, &cfg).unwrap();
        let reference =
            accumulate_booleanity_gram_reference(traces, &row_weights, &rho_powers, sources, &cfg)
                .unwrap();
        assert_eq!(fast.g, reference.g);
        assert_eq!(fast.h, reference.h);
    }

    #[test]
    fn mask_fast_path_matches_reference_on_honest_virtuals() {
        use crate::neutron_nova::projection_sha::production_sha_booleanity_sources;
        let traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        assert_gram_matches_reference(&traces, &production_sha_booleanity_sources());
    }

    #[test]
    fn mask_fast_path_matches_reference_with_non_binary_fallback() {
        use crate::neutron_nova::projection_sha::production_sha_booleanity_sources;
        let mut traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        // Non-binary real bits force per-item fallbacks in real and virtual
        // sources that read these slots.
        let idx_a = bit_slice_index(ShaWordCol::A.index(), 5, SHA_WORD_BITS);
        traces[1].bit_slices[idx_a].evaluations[10] = f(7);
        let idx_e = bit_slice_index(ShaWordCol::E.index(), 9, SHA_WORD_BITS);
        traces[3].bit_slices[idx_e].evaluations[33] = f(2);
        assert_gram_matches_reference(&traces, &production_sha_booleanity_sources());
    }

    #[test]
    fn mask_fast_path_matches_reference_on_invalid_virtual_relations() {
        use crate::neutron_nova::projection_sha::production_sha_booleanity_sources;
        // Random bits: every bit is 0/1 but the Uef/UNegEg/Maj relations do
        // not hold, so virtual sources take the fallback path with values
        // outside {0,1}.
        let traces: Vec<_> = (0..4).map(synthetic_boolean_trace).collect();
        assert_gram_matches_reference(&traces, &production_sha_booleanity_sources());
    }

    #[test]
    fn masked_fold_matches_weighted_fold() {
        use crate::neutron_nova::projection_sha::fold_projected_traces_with_weights;
        let cfg = test_config();
        let mut traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        // Include a non-binary slot so the per-(slot,row) fallback is hit.
        let idx = bit_slice_index(ShaWordCol::W.index(), 11, SHA_WORD_BITS);
        traces[2].bit_slices[idx].evaluations[40] = f(5);
        let publics: Vec<_> = (0..4).map(|_| zero_public()).collect();
        let theta: Vec<F> = (0..4u64).map(|i| f(101 + 7 * i)).collect();
        let masks = ShaSourceMasks::build(&traces, &cfg).unwrap();

        let (masked, masked_public) =
            fold_projected_traces_with_theta_masks(&traces, &publics, &theta, &masks, &cfg)
                .unwrap();
        let (reference, reference_public) =
            fold_projected_traces_with_weights(&traces, &publics, &theta, &cfg).unwrap();
        assert_eq!(masked.trace, reference.trace);
        assert_eq!(masked_public, reference_public);
    }

    #[test]
    fn batch_invert_matches_individual_inversion() {
        let cfg = test_config();
        let one = F::one_with_cfg(&cfg);
        let mut values: Vec<F> = (2u64..12).map(f).collect();
        let expected: Vec<F> = values.iter().map(|v| one.clone() / v.clone()).collect();
        batch_invert(&mut values, &cfg);
        assert_eq!(values, expected);
    }

    #[test]
    fn zero_trace_folded_ideal_polys_pass_membership() {
        let cfg = test_config();
        let trace = zero_trace();
        let public = zero_public();
        let r_ic: [F; SHA_ROW_VARS] = std::array::from_fn(|i| f(11 + 2 * i as u64));
        let polys = build_sha_ideal_values_at_point(&trace, &public, &r_ic, &cfg).unwrap();
        verify_fresh_sha_ideal_polys(std::slice::from_ref(&polys), &cfg).unwrap();
    }

    #[test]
    fn synthetic_trace_ideal_polys_fail_membership() {
        let cfg = test_config();
        let trace = boolean_virtuals_trace(0);
        let public = zero_public();
        let r_ic: [F; SHA_ROW_VARS] = std::array::from_fn(|i| f(11 + 2 * i as u64));
        let polys = build_sha_ideal_values_at_point(&trace, &public, &r_ic, &cfg).unwrap();
        assert!(verify_fresh_sha_ideal_polys(std::slice::from_ref(&polys), &cfg).is_err());
    }

    #[test]
    fn nonzero_target_matches_manual_lambda_sum() {
        use crate::neutron_nova::projection_sha::NUM_SHA_RESIDUAL_FAMILIES;
        let cfg = test_config();
        let trace = boolean_virtuals_trace(1);
        let public = zero_public();
        let r_ic: [F; SHA_ROW_VARS] = std::array::from_fn(|i| f(11 + 2 * i as u64));
        let polys = build_sha_ideal_values_at_point(&trace, &public, &r_ic, &cfg).unwrap();
        let a = f(19);
        let lambda = f(23);
        let lambda_powers = powers(
            lambda.clone(),
            F::one_with_cfg(&cfg),
            NUM_SHA_RESIDUAL_FAMILIES,
        );
        let mut expected = F::zero_with_cfg(&cfg);
        for (slot, family) in NONZERO_SHA_FAMILIES.iter().enumerate() {
            let mut value = F::zero_with_cfg(&cfg);
            let a_powers = powers(a.clone(), F::one_with_cfg(&cfg), polys[slot].coeffs.len());
            for (coeff, power) in polys[slot].coeffs.iter().zip(&a_powers) {
                value += coeff.clone() * power;
            }
            expected += lambda_powers[family.index()].clone() * value;
        }
        assert_eq!(sha_nonzero_target_at(&polys, &a, &lambda, &cfg), expected);
    }

    #[test]
    fn fold_first_zero_traces_prove_and_verify() {
        let cfg = test_config();
        let traces: Vec<_> = (0..4).map(|_| zero_trace()).collect();
        let publics: Vec<_> = (0..4).map(|_| zero_public()).collect();
        let sources = small_sources();

        let mut prover_transcript = Blake3Transcript::new();
        let (proof, artifacts) =
            prove_fold_first_sha_sumfold(&traces, &publics, &sources, &mut prover_transcript, &cfg)
                .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        let claims =
            verify_fold_first_sha_sumfold(&proof, traces.len(), &mut verifier_transcript, &cfg)
                .unwrap();

        let zero = F::zero_with_cfg(&cfg);
        assert_eq!(claims.theta, artifacts.theta);
        assert_eq!(claims.b_star, zero);
        assert_eq!(claims.target, zero);
        assert_eq!(claims.target, artifacts.target);
        // All-zero folded integrand: the endpoint claim is zero as well.
        assert_eq!(claims.expected_row_eval, zero);
        assert_eq!(claims.row_point.len(), SHA_ROW_VARS);
    }

    #[test]
    fn fold_first_algebraic_identity_on_synthetic_traces() {
        use crate::neutron_nova::projection_sha::{
            NUM_SHA_RESIDUAL_FAMILIES, ShaResidualFamily, folded_row_integrand_sum,
            folded_row_integrand_values, residual_polys_at_row,
        };
        let cfg = test_config();
        let traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        let publics: Vec<_> = (0..4).map(|_| zero_public()).collect();
        let sources = small_sources();
        let r_ic: [F; SHA_ROW_VARS] = std::array::from_fn(|i| f(11 + 2 * i as u64));
        let rho = f(29);
        let row_weights = build_eq_x_r_vec(&r_ic, &cfg).unwrap();
        let rho_powers = powers(rho.clone(), F::one_with_cfg(&cfg), sources.len());
        let domain = SkipDomain::<F>::new(traces.len(), &cfg).unwrap();
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();
        let mut transcript = Blake3Transcript::new();
        let (_, verdict) = prove_skip_round(&gram, &domain, &mut transcript, &cfg).unwrap();

        let (mut folded, folded_public) =
            crate::neutron_nova::projection_sha::fold_projected_traces_with_weights(
                &traces,
                &publics,
                &verdict.theta,
                &cfg,
            )
            .unwrap();
        let polys =
            build_sha_ideal_values_at_point(&folded.trace, &folded_public, &r_ic, &cfg).unwrap();

        let a = f(19);
        let lambda = f(23);
        let xi = f(31);
        folded.trace.scalarized = scalarize_bit_slices(&folded.trace.bit_slices, &a, &cfg).unwrap();

        let integrand = folded_row_integrand_values(
            &folded.trace,
            &folded_public,
            &r_ic,
            &a,
            &lambda,
            &rho,
            &xi,
            &sources,
            &cfg,
        )
        .unwrap();
        let total = folded_row_integrand_sum(&integrand, &cfg).unwrap();

        // Zero-family correction: synthetic traces have nonzero zero-family
        // residuals; honest SHA traces zero this term out.
        let lambda_powers = powers(
            lambda.clone(),
            F::one_with_cfg(&cfg),
            NUM_SHA_RESIDUAL_FAMILIES,
        );
        let mut zero_family_term = F::zero_with_cfg(&cfg);
        for (row, row_weight) in row_weights.iter().enumerate() {
            let residuals =
                residual_polys_at_row(&folded.trace, &folded_public, row, &cfg).unwrap();
            for (family_idx, poly) in residuals.iter().enumerate() {
                if ShaResidualFamily::ALL[family_idx].is_nonzero_ideal() {
                    continue;
                }
                let a_powers = powers(a.clone(), F::one_with_cfg(&cfg), poly.coeffs.len());
                let mut value = F::zero_with_cfg(&cfg);
                for (coeff, power) in poly.coeffs.iter().zip(&a_powers) {
                    value += coeff.clone() * power;
                }
                zero_family_term += row_weight.clone() * &lambda_powers[family_idx] * value;
            }
        }

        let assembled = sha_nonzero_target_at(&polys, &a, &lambda, &cfg)
            + xi.clone() * &verdict.b_star
            + zero_family_term;
        assert_eq!(total, assembled);
    }

    #[test]
    fn fold_first_rejects_tampered_bit() {
        let cfg = test_config();
        let mut traces: Vec<_> = (0..4).map(|_| zero_trace()).collect();
        // A single set W bit breaks the (X−2) message-schedule membership of
        // the folded ideal polys while keeping booleanity honest.
        let idx = bit_slice_index(ShaWordCol::W.index(), 3, SHA_WORD_BITS);
        traces[1].bit_slices[idx].evaluations[20] = f(1);
        let publics: Vec<_> = (0..4).map(|_| zero_public()).collect();
        let sources = small_sources();

        let mut prover_transcript = Blake3Transcript::new();
        let (proof, _) =
            prove_fold_first_sha_sumfold(&traces, &publics, &sources, &mut prover_transcript, &cfg)
                .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        assert!(
            verify_fold_first_sha_sumfold(&proof, traces.len(), &mut verifier_transcript, &cfg)
                .is_err()
        );
    }

    #[test]
    fn fold_first_rejects_non_boolean_bit() {
        let cfg = test_config();
        let mut traces: Vec<_> = (0..4).map(|_| zero_trace()).collect();
        let idx = bit_slice_index(ShaWordCol::A.index(), 0, SHA_WORD_BITS);
        traces[2].bit_slices[idx].evaluations[7] = f(2);
        let publics: Vec<_> = (0..4).map(|_| zero_public()).collect();
        let sources = small_sources();

        let mut prover_transcript = Blake3Transcript::new();
        let (proof, _) =
            prove_fold_first_sha_sumfold(&traces, &publics, &sources, &mut prover_transcript, &cfg)
                .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        let err =
            verify_fold_first_sha_sumfold(&proof, traces.len(), &mut verifier_transcript, &cfg)
                .expect_err("non-boolean bit must be rejected");
        assert!(matches!(
            err,
            FoldFirstError::SumFold(SumFoldError::SkipRoundZeroCheckFailed)
        ));
    }

    #[test]
    fn fold_first_rejects_wrong_ideal_poly() {
        let cfg = test_config();
        let traces: Vec<_> = (0..4).map(|_| zero_trace()).collect();
        let publics: Vec<_> = (0..4).map(|_| zero_public()).collect();
        let sources = small_sources();

        let mut prover_transcript = Blake3Transcript::new();
        let (mut proof, _) =
            prove_fold_first_sha_sumfold(&traces, &publics, &sources, &mut prover_transcript, &cfg)
                .unwrap();
        proof.folded_ideal_polys[0].coeffs = vec![f(1)];

        let mut verifier_transcript = Blake3Transcript::new();
        assert!(
            verify_fold_first_sha_sumfold(&proof, traces.len(), &mut verifier_transcript, &cfg)
                .is_err()
        );
    }

    #[test]
    fn zeroed_domain_message_passes_gamma_but_shifts_b_star() {
        let cfg = test_config();
        let mut traces: Vec<_> = (0..4).map(boolean_virtuals_trace).collect();
        let idx = bit_slice_index(ShaWordCol::A.index(), 0, SHA_WORD_BITS);
        traces[1].bit_slices[idx].evaluations[4] = f(3);
        let row_weights = test_row_weights(&cfg);
        let sources = small_sources();
        let rho_powers = powers(f(29), F::one_with_cfg(&cfg), sources.len());
        let domain = SkipDomain::<F>::new(traces.len(), &cfg).unwrap();
        let gram =
            accumulate_booleanity_gram(&traces, &row_weights, &rho_powers, &sources, &cfg).unwrap();

        let mut forged = gram.skip_node_values(&domain, &cfg).unwrap();
        let zero = F::zero_with_cfg(&cfg);
        for value in forged.iter_mut().take(domain.n_instances()) {
            *value = zero.clone();
        }
        let forged_proof = FoldFirstSkipRoundProof {
            node_values: forged,
        };

        let gamma = f(41);
        let mut verifier_transcript = Blake3Transcript::new();
        let verdict = verify_skip_round(
            &forged_proof,
            &gamma,
            &domain,
            &mut verifier_transcript,
            &cfg,
        )
        .expect("zeroed domain slots pass the gamma check");

        // But B★ no longer matches the true folded booleanity residue, so
        // the downstream row-sumcheck/opening chain rejects. Compute the true
        // residue directly at α via the naive oracle.
        let true_b_star = naive_q_at(
            &traces,
            &row_weights,
            &rho_powers,
            &sources,
            &domain,
            &verdict.alpha,
        );
        assert_ne!(verdict.b_star, true_b_star);
    }
}
