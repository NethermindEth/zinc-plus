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

use crypto_primitives::{FromWithConfig, PrimeField};
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_utils::powers;

use crate::neutron_nova::SumFoldError;
use crate::neutron_nova::projection_sha::{
    ProjectedTrace, SHA_ROW_COUNT, ShaBooleanitySource, ShaProjectionError,
    booleanity_source_value_at_row_with_virtuals, reconstruct_virtual_ch_maj_at_row_unchecked,
    sources_need_virtuals,
};

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

#[allow(clippy::arithmetic_side_effects)]
fn inverse_lagrange_denominators<F>(points: &[F], field_cfg: &F::Config) -> Vec<F>
where
    F: PrimeField,
{
    let one = F::one_with_cfg(field_cfg);
    points
        .iter()
        .enumerate()
        .map(|(j, u_j)| {
            let mut denom = one.clone();
            for (i, u_i) in points.iter().enumerate() {
                if i != j {
                    denom = denom * (u_j.clone() - u_i);
                }
            }
            one.clone() / denom
        })
        .collect()
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

/// One streaming pass over `(source, row)` items across all instances.
///
/// `row_weights` must be the `eq(r_ic, ·)` table over the 128 SHA rows and
/// `rho_powers` must have one entry per booleanity source (`ρ^idx(q)`).
#[allow(clippy::arithmetic_side_effects)]
pub fn accumulate_booleanity_gram<F>(
    traces: &[ProjectedTrace<F>],
    row_weights: &[F],
    rho_powers: &[F],
    sources: &[ShaBooleanitySource],
    field_cfg: &F::Config,
) -> Result<GramAccumulator<F>, ShaProjectionError>
where
    F: PrimeField,
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
                    reconstruct_virtual_ch_maj_at_row_unchecked(trace, row, field_cfg).map(Some)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![None; n]
        };
        for (source_idx, source) in sources.iter().enumerate() {
            let mut any_nonzero = false;
            for (j, trace) in traces.iter().enumerate() {
                let value = booleanity_source_value_at_row_with_virtuals(
                    trace,
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
                acc = acc - basis[k].clone() * &self.h[k];
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

#[cfg(test)]
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
            let selector = instance_idx
                + u64::try_from(col * 17 + row * 3 + bit).expect("selector fits u64");
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
                        Some(
                            reconstruct_virtual_ch_maj_at_row_unchecked(trace, row, &cfg).unwrap(),
                        )
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
