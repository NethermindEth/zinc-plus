//! The **merged (inline) Hadamard discharge** — the Lagrange-reinterpretation
//! construction. Instead of the standalone oblong zerocheck at its own point
//! `[z, γ]` (`f2_oblong_hadamard`), the AND relations ride the MAIN pipeline.
//!
//! ## The construction, in one paragraph
//!
//! Reinterpret each cell's bits as **subspace-Lagrange coefficients** over the
//! oblong base domain `H` (`L(v) = Σ_b v_b·L_b(X)`). Bitwise AND `u ⊙ v = w`
//! then becomes the ring congruence `L(u)·L(v) ≡ L(w) mod Z_H` — an
//! ideal-membership claim with generator `Z_H = Π_{h∈H}(X−h)` — so the
//! discharge is the standard ideal-check + evaluation-projection + sumcheck
//! pipeline:
//!
//! 1. **Phase-1 message** (the `⟨Z_H⟩` ideal check, before α): the prover
//!    ships the `D` extension-domain evaluations of
//!    `R₀(X) = Σ_k γ_h^k · Σ_rows eq(row; r_had)·(L(U_k)·L(V_k) − L(W_k))(row)`
//!    — a degree ≤ 2(D−1) polynomial **vanishing on `H`** for honest data (the
//!    membership in `⟨Z_H⟩`), so the base half never travels.
//! 2. **Projection at α**: `ψ_α ∘ L = ψ_z|_{z=α}` — the operand columns are
//!    folded with the base-Lagrange weights `L_b(α)`, and the product term
//!    joins the main Step-4 `MultiDegreeSumcheck` as a second group (degree 3
//!    with its eq factor) whose claimed sum the verifier checks against the
//!    reconstruction `R₀(α)` ([`r0_at`]).
//! 3. **End claims at `r*`**: the group's closing operand evals derive from
//!    per-pair evals `ψ_{L(α)}(col↓Δ)(r*)` (the weight-generic
//!    [`derive_operand_parents`](crate::f2_hadamard::derive_operand_parents)
//!    machinery), which fold into the SAME multipoint as the α-claims and
//!    bind to the open's `a'` via one extra weight-vector projection.
//!
//! ## The GF(2⁸) fast lane (the production configuration)
//!
//! Phase 1 runs the **byte-lookup `GF(2⁸)` NTT with the eq-split**
//! ([`Gf8Scheme`], the production oblong's speed lever): the base domain `H`
//! is the `embed(H₈)` subspace, and the Hadamard family's zerocheck
//! randomness is the **hybrid point**
//! `r_had = [s₀, s₁, s₂] ++ r_IC[3..]` ([`merged_eq_point`]) — the scheme's
//! three deterministic small-field challenges on the LOW row variables (their
//! `GF(2⁸)` eq tensor is `F_2`-linearly independent, so per-8-row-block
//! violation patterns survive; the same soundness shape as the production
//! oblong) and the shared IC point on the rest. The kernel's per-row weight
//! `embed(eq_small[x&7])·eq_big[x≫3]` equals the protocol's
//! `build_eq_x_r` table at `r_had` exactly (both little-endian, `embed` is a
//! field hom — pinned by the `hybrid_eq_table_matches_kernel_convention`
//! canary), so the same table serves the Phase-1 kernel and the sumcheck
//! group, and the verifier's closing factor is `eq(r*; r_had)`. All relations
//! run as **one stacked kernel pass** ([`merged_round_message`]): the
//! relation-batching `γ_h^k` folds into relation `k`'s per-chunk big-eq
//! weights, so the kernel parallelises over the full `K·2^ν` row range.
//!
//! Vs the standalone oblong discharge this deletes Phase 2 (the separate row
//! MLE-check at `[z, γ_word]`), the second transcript point `r_H ≡ γ_word`,
//! the all-witness-cols `z_up_evals` pass, and the doubled multipoint trace
//! (only AND-referenced cols are appended); soundness is the oblong's
//! verbatim — the Phase-1 messages are in bijection (`R₀` off-subspace values
//! ↔ the quotient `R₀/Z_H`). Adder relations may ride along with **trusted**
//! parents (the same posture as the oblong/fused arms). The `GF(2⁸)` scheme
//! pins the cell width to `D = WORD_BITS = 32` on this branch. Design
//! ledger: `documentation/f2x-sha-todo.md`, "Merged (inline ⟨Z_H⟩) Hadamard
//! discharge" (2026-06-12).

use crypto_primitives::{Field, FromPrimitiveWithConfig, PrimeField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheckGroup;
use zinc_piop::sumcheck::prover::RoundPolyEvaluator;
use zinc_utils::cfg_into_iter;
use zinc_utils::inner_transparent_field::InnerTransparentField;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;
use zinc_poly::univariate::binary_subspace::extrapolate_over_subspace;
use zinc_poly::univariate::oblong_and::{OblongScheme, WORD_BITS, eq_indicator};
use zinc_poly::univariate::oblong_and_gf8::Gf8Scheme;

use crate::f2_hadamard::{
    F2AdderSpec, F2HadamardSpec, build_adder_operand_columns, build_operand_column, cell_mask,
};

type Gf = BinaryFieldGF128;
type Inner = <BinaryFieldGF128 as Field>::Inner;

/// Build the packed operand columns of the merged discharge, relation-major
/// (`U_0, V_0, W_0, U_1, …`): AND specs first (XOR/row-shift/complement
/// operands via [`build_operand_column`]), then adders (Binius carry-AND
/// operands via [`build_adder_operand_columns`] — these double as the
/// trusted-parent source).
#[allow(clippy::arithmetic_side_effects)]
pub fn merged_operand_columns<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    and_specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<BinaryPoly<D>>> {
    let mut out = Vec::with_capacity((and_specs.len() + adder_specs.len()) * 3);
    for spec in and_specs {
        for op in [&spec.u, &spec.v, &spec.w] {
            out.push(build_operand_column::<D>(columns, op, num_vars));
        }
    }
    for adder in adder_specs {
        out.extend(build_adder_operand_columns::<D>(columns, adder, num_vars));
    }
    out
}

/// The Hadamard family's **hybrid zerocheck point**
/// `r_had = [s₀, s₁, s₂] ++ ic_point[3..]`: the scheme's deterministic
/// small-field challenges on the low row variables (the eq-split enabler) +
/// the shared IC point on the rest. Both the Phase-1 kernel's row weighting
/// and the sumcheck group's eq table are `eq(·; r_had)`; the verifier's
/// closing factor is `eq(r*; r_had)`.
pub fn merged_eq_point(scheme: &Gf8Scheme, ic_point: &[Gf]) -> Vec<Gf> {
    let small = scheme.small_challenges();
    assert!(
        ic_point.len() >= small.len(),
        "merged Hadamard discharge needs num_vars ≥ {} (the eq-split width)",
        small.len(),
    );
    let mut r_had = Vec::with_capacity(ic_point.len());
    r_had.extend_from_slice(small);
    r_had.extend_from_slice(&ic_point[small.len()..]);
    r_had
}

/// The γ_h-batched Phase-1 message on the **GF(2⁸) fast lane**: the
/// `WORD_BITS` `embed(H₈)`-extension-domain evaluations of
/// `R₀(X) = Σ_k γ_h^k · Σ_rows eq(row; r_had)·(Ũ_k·Ṽ_k − W̃_k)(X, row)`.
///
/// All relations run as **one stacked eq-split kernel pass**
/// ([`Gf8Scheme::round_message_with_eq_big`]): the operand word columns are
/// concatenated relation-major and the relation-batching weight `γ_h^k` is
/// folded into relation `k`'s per-chunk big-eq weights
/// (`eq_big_stacked[(k, chunk)] = γ_h^k · eq_big(chunk; ic_point[3..])`), so
/// the kernel parallelises over the full `K·2^ν` row range. For honest data
/// `R₀` vanishes on the base domain, so these values determine it.
///
/// The `GF(2⁸)` scheme works at the fixed word width — requires
/// `D == WORD_BITS`.
#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::cast_possible_truncation)]
pub fn merged_round_message<const D: usize>(
    scheme: &Gf8Scheme,
    operand_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    ic_point: &[Gf],
    gamma_h: Gf,
) -> Vec<Gf> {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    assert_eq!(D, WORD_BITS, "the GF(2⁸) merged lane is fixed at D = WORD_BITS");
    debug_assert_eq!(operand_cols.len() % 3, 0, "operands come 3 per relation");
    let k_rel = operand_cols.len() / 3;
    let n_small = scheme.small_challenges().len();
    let n = operand_cols
        .first()
        .map(|c| c.evaluations.len())
        .unwrap_or(0);
    if k_rel == 0 {
        return vec![Gf::zero(); WORD_BITS];
    }

    // Stacked operand words, relation-major (parallel over the 3·K columns).
    let stack = |off: usize| -> Vec<u32> {
        let cols: Vec<Vec<u32>> = zinc_utils::cfg_into_iter!(0..k_rel)
            .map(|k| {
                operand_cols[3 * k + off]
                    .evaluations
                    .iter()
                    .map(|x| cell_mask::<D>(x) as u32)
                    .collect()
            })
            .collect();
        cols.concat()
    };
    let (a, b, c) = (stack(0), stack(1), stack(2));

    // Per-chunk big-eq weights with γ_h^k baked in per relation block.
    let eq_big_base = eq_indicator(&ic_point[n_small..]);
    debug_assert_eq!(eq_big_base.len() * (1 << n_small), n, "eq must cover the rows");
    let mut eq_big_stacked = Vec::with_capacity(k_rel * eq_big_base.len());
    let mut w = Gf::one();
    for _ in 0..k_rel {
        eq_big_stacked.extend(eq_big_base.iter().map(|&e| w * e));
        w *= &gamma_h;
    }

    scheme
        .round_message_with_eq_big(&a, &b, &c, &eq_big_stacked)
        .to_vec()
}

/// Evaluate the Phase-1 polynomial at `x`: `R₀` is the unique degree
/// `< 2·WORD_BITS` polynomial with base-domain values **zero** and
/// extension-domain values `msg` over the scheme's `embed(H₈)` full subspace;
/// extrapolate it to `x`. The merged group's claimed sumcheck total must
/// equal `r0_at(msg, α)` — shipping only `msg` is what makes the `⟨Z_H⟩`
/// membership free (the reconstruction vanishes on `H` by construction).
#[allow(clippy::arithmetic_side_effects)]
pub fn r0_at(scheme: &Gf8Scheme, msg: &[Gf], x: Gf) -> Gf {
    debug_assert_eq!(
        msg.len(),
        WORD_BITS,
        "Phase-1 message is the WORD_BITS extension evals",
    );
    let mut coeffs = vec![Gf::zero(); 2 * WORD_BITS];
    coeffs[WORD_BITS..].copy_from_slice(msg);
    extrapolate_over_subspace(scheme.full_subspace(), &coeffs, x)
}

/// The base-Lagrange weights `L_b(α)` over the scheme's `embed(H₈)` base
/// subspace, folding a cell's `WORD_BITS` bits at the (shared)
/// evaluation-projection point — the merged path's analogue of the monomial
/// `α^b` powers: `ψ_α(L(v)) = Σ_b v_b·L_b(α) = ψ_z(v)|_{z=α}`.
pub fn lagrange_alpha(scheme: &Gf8Scheme, alpha: Gf) -> Vec<Gf> {
    scheme.base_lagrange(alpha).to_vec()
}

/// Build the merged-Hadamard sumcheck group: degree 3, MLEs
/// `[eq(·; r_had), ZU_0, ZV_0, ZW_0, …]` with `Z• = ψ_{L(α)}(•)` the
/// Lagrange-folded operand columns (storage in `Inner` form, as the framework
/// expects), comb `eq · Σ_k γ_h^k·(ZU_k·ZV_k + ZW_k)` (char 2: `− = +`). Its
/// claimed sum is `R₀(α)`.
#[allow(clippy::arithmetic_side_effects)]
/// Fused round-polynomial evaluator for the merged degree-3 Hadamard group,
/// used for **every round** in place of the generic per-point value-array
/// gather (see [`RoundPolyEvaluator`]). The generic path rebuilds a
/// `1 + 3·k_rel`-element value array per hypercube point — a scatter across
/// all operand MLEs that dominates the main-pipeline sumcheck. This evaluator
/// instead loops **relation-outer, point-inner**, streaming each relation's
/// three folded operands `ZU_k, ZV_k, ZW_k` in coefficient space (Karatsuba
/// for `ZU·ZV`), weights by `γ_h^k`, then applies the common `eq` factor in a
/// second pass. Mirrors the standalone discharge's `HadamardRoundEvaluator`
/// but over folded operands (no per-bit inner loop). Same arithmetic,
/// exact-field order-independent sum ⇒ the round message is **byte-identical**
/// to the comb below.
struct MergedHadamardRoundEvaluator<F: PrimeField> {
    k_rel: usize,
    /// `gamma_pows[k] = γ_h^k`.
    gamma_pows: Vec<F>,
}

impl<F> RoundPolyEvaluator<F> for MergedHadamardRoundEvaluator<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: Send + Sync + Zero + Clone,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn round_evals(
        &self,
        mles: &[DenseMultilinearExtension<F::Inner>],
        round: usize,
        num_vars: usize,
        degree: usize,
        config: &F::Config,
    ) -> Vec<F> {
        debug_assert_eq!(degree, 3, "merged Hadamard group is degree 3");
        debug_assert_eq!(mles.len(), 1 + self.k_rel * 3);
        let zero = F::zero_with_cfg(config);
        let half = 1usize << (num_vars - round); // number of point-pairs

        let eq = &mles[0].evaluations;
        let ops = &mles[1..]; // 3·k_rel folded operands, relation-major
        let weights = &self.gamma_pows;

        const CHUNK: usize = 512;
        let n_chunks = half.div_ceil(CHUNK).max(1);

        // Each chunk returns its partial of M's coeffs [m0, m1, m2, m3]; the
        // round poly is cubic (linear `eq` × degree-2 `Σ_k γ^k(ZU·ZV+ZW)`).
        let partials: Vec<[F; 4]> = cfg_into_iter!(0..n_chunks)
            .map(|c| {
                let lo = c * CHUNK;
                let hi = ((c + 1) * CHUNK).min(half);
                if lo >= hi {
                    return [zero.clone(), zero.clone(), zero.clone(), zero.clone()];
                }
                let clen = hi - lo;
                // Pass 1: G_pt coeffs (g0, g1, g2) in three contiguous bands.
                let mut g = vec![zero.clone(); 3 * clen];
                for k in 0..self.k_rel {
                    let w = &weights[k];
                    let u = &ops[3 * k].evaluations;
                    let v = &ops[3 * k + 1].evaluations;
                    let ws = &ops[3 * k + 2].evaluations;
                    for pt in lo..hi {
                        let li = pt - lo;
                        let u0 = F::new_unchecked_with_cfg(u[2 * pt].clone(), config);
                        let u1 = F::new_unchecked_with_cfg(u[2 * pt + 1].clone(), config);
                        let v0 = F::new_unchecked_with_cfg(v[2 * pt].clone(), config);
                        let v1 = F::new_unchecked_with_cfg(v[2 * pt + 1].clone(), config);
                        let w0 = F::new_unchecked_with_cfg(ws[2 * pt].clone(), config);
                        let dw = F::new_unchecked_with_cfg(ws[2 * pt + 1].clone(), config)
                            - w0.clone();
                        // (ZU·ZV) coeffs via Karatsuba (3 muls): X² coeff Δu·Δv,
                        // X⁰ coeff u0·v0, X¹ coeff u1·v1 − p0 − p2.
                        let p0 = u0.clone() * v0.clone();
                        let p2 = (u1.clone() - u0) * (v1.clone() - v0);
                        let p1 = u1 * v1 - p0.clone() - p2.clone();
                        // term = ZU·ZV + ZW (char 2: + ≡ −); W = w0 + Δw·X.
                        let t0 = p0 + w0;
                        let t1 = p1 + dw;
                        let t2 = p2;
                        g[li] += w.clone() * t0;
                        g[clen + li] += w.clone() * t1;
                        g[2 * clen + li] += w.clone() * t2;
                    }
                }
                // Pass 2: M_pt = eq_pt·G_pt; accumulate M's coeffs over the chunk.
                let mut m = [zero.clone(), zero.clone(), zero.clone(), zero.clone()];
                for pt in lo..hi {
                    let li = pt - lo;
                    let e0 = F::new_unchecked_with_cfg(eq[2 * pt].clone(), config);
                    let de = F::new_unchecked_with_cfg(eq[2 * pt + 1].clone(), config) - e0.clone();
                    let g0 = g[li].clone();
                    let g1 = g[clen + li].clone();
                    let g2 = g[2 * clen + li].clone();
                    m[0] += e0.clone() * g0.clone();
                    m[1] += e0.clone() * g1.clone() + de.clone() * g0;
                    m[2] += e0.clone() * g2.clone() + de.clone() * g1;
                    m[3] += de * g2;
                }
                m
            })
            .collect();

        let mut m = [zero.clone(), zero.clone(), zero.clone(), zero.clone()];
        for p in &partials {
            for (x, mc) in m.iter_mut().enumerate() {
                *mc += p[x].clone();
            }
        }

        // Evaluate the cubic M at {0, 1, X, X+1} — same boundary convention as
        // the generic path ⇒ byte-identical round message.
        let horner = |t: &F| -> F {
            m[0].clone()
                + t.clone() * (m[1].clone() + t.clone() * (m[2].clone() + t.clone() * m[3].clone()))
        };
        let bp2 = F::from_with_cfg(2, config);
        let bp3 = F::from_with_cfg(3, config);
        vec![
            m[0].clone(),
            m[0].clone() + m[1].clone() + m[2].clone() + m[3].clone(),
            horner(&bp2),
            horner(&bp3),
        ]
    }
}

pub fn merged_hadamard_group(
    eq_mle: DenseMultilinearExtension<Inner>,
    folded_operands: Vec<DenseMultilinearExtension<Inner>>,
    gamma_h: Gf,
) -> MultiDegreeSumcheckGroup<Gf> {
    debug_assert_eq!(folded_operands.len() % 3, 0, "operands come 3 per relation");
    let k_rel = folded_operands.len() / 3;
    let mut polys = Vec::with_capacity(1 + folded_operands.len());
    polys.push(eq_mle);
    polys.extend(folded_operands);
    let mut gamma_pows = Vec::with_capacity(k_rel);
    let mut w = Gf::one();
    for _ in 0..k_rel {
        gamma_pows.push(w);
        w *= &gamma_h;
    }
    // Fused rounds evaluator (byte-identical to the comb below; streams the 3
    // folded operands per relation in coefficient space). The generic path
    // scatter-gathers all 1+3·k_rel operand MLEs per hypercube point, which is
    // the bulk of the main-pipeline sumcheck cost.
    let evaluator: Box<dyn RoundPolyEvaluator<Gf>> = Box::new(MergedHadamardRoundEvaluator {
        k_rel,
        gamma_pows: gamma_pows.clone(),
    });
    MultiDegreeSumcheckGroup::new(
        3,
        polys,
        Box::new(move |v: &[Gf]| {
            let mut s = Gf::zero();
            for (k, gw) in gamma_pows.iter().enumerate() {
                // char 2: ZU·ZV − ZW = ZU·ZV + ZW.
                s += *gw * (v[1 + 3 * k] * v[2 + 3 * k] + v[3 + 3 * k]);
            }
            v[0] * s
        }),
    )
    .with_round_evaluator(evaluator)
}

/// The verifier's expected merged-group evaluation at `r*`:
/// `eq(r*; r_had) · Σ_k γ_h^k·(P_{3k}·P_{3k+1} + P_{3k+2})` over the derived
/// operand parents (AND parents from the bound pair evals, then the trusted
/// adder parents), in relation-major order. Compared against the sumcheck's
/// `expected_evaluations()[1]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn merged_expected_evaluation(eq_at_rstar_rhad: Gf, operand_parents: &[Gf], gamma_h: Gf) -> Gf {
    debug_assert_eq!(operand_parents.len() % 3, 0, "operands come 3 per relation");
    let mut s = Gf::zero();
    let mut w = Gf::one();
    for rel in operand_parents.chunks_exact(3) {
        s += w * (rel[0] * rel[1] + rel[2]);
        w *= &gamma_h;
    }
    eq_at_rstar_rhad * s
}

/// The distinct referenced witness columns of the AND specs' pairs, sorted
/// (the multipoint append set: each gets one Lagrange-α-projected MLE).
pub fn distinct_pair_cols(pairs: &[(usize, usize)]) -> Vec<usize> {
    let mut cols: Vec<usize> = pairs.iter().map(|&(c, _)| c).collect();
    cols.sort_unstable();
    cols.dedup();
    cols
}

/// Pair evals `ψ_{L(α)}(col↓Δ)(r*)` with **per-distinct-col projection**:
/// each referenced column is Lagrange-folded once and reused across its
/// `(col, Δ)` pairs (the shared `pair_alpha_evals` re-projects per pair).
/// Output order matches `pairs` (the wire order).
#[allow(clippy::arithmetic_side_effects)]
pub fn pair_evals_dedup<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    pairs: &[(usize, usize)],
    weights: &[Gf],
    r_star: &[Gf],
) -> Vec<Gf> {
    use zinc_poly::univariate::binary_gf128::project_column_with_powers;
    use zinc_utils::cfg_iter;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    let dcols = distinct_pair_cols(pairs);
    let zero_inner = Gf::zero().into_inner();
    // Project each distinct col once (parallel over cols).
    let projected: Vec<Vec<Inner>> = cfg_iter!(dcols)
        .map(|&col| {
            project_column_with_powers::<D>(&columns[col].evaluations, weights)
                .iter()
                .map(|x| *x.inner())
                .collect()
        })
        .collect();
    // Per pair: row-shift the projected vector (zero past the tail), eval at r*.
    cfg_iter!(pairs)
        .map(|&(col, shift)| {
            let pos = dcols.binary_search(&col).expect("col in distinct set");
            let proj = &projected[pos];
            let n = proj.len();
            let shifted: Vec<Inner> = (0..n)
                .map(|t| match t.checked_add(shift).filter(|&v| v < n) {
                    Some(s) => proj[s].clone(),
                    None => zero_inner.clone(),
                })
                .collect();
            let mle = DenseMultilinearExtension::from_evaluations_vec(
                columns[col].num_vars,
                shifted,
                zero_inner.clone(),
            );
            <DenseMultilinearExtension<_> as zinc_poly::mle::MultilinearExtensionWithConfig<
                Gf,
            >>::evaluate_with_config(mle, r_star, &())
            .expect("shifted Lagrange eval at r* should succeed")
        })
        .collect()
}

/// Evaluate a sumcheck group's residual MLEs at the full point: after the
/// last round the framework leaves each MLE folded through all but the final
/// challenge (length-2 tables); one more fold by `r_last` yields the exact
/// multilinear evaluation at `r*`. Used to read the merged group's operand
/// evals (the trusted adder parents) off the prover state for free.
#[allow(clippy::arithmetic_side_effects)]
pub fn residual_evals(mles: &[DenseMultilinearExtension<Inner>], r_last: Gf) -> Vec<Gf> {
    mles.iter()
        .map(|m| {
            debug_assert_eq!(
                m.evaluations.len(),
                2,
                "residual sumcheck MLE must be a length-2 table",
            );
            let lo = <Gf as PrimeField>::new_unchecked_with_cfg(m.evaluations[0].clone(), &());
            let hi = <Gf as PrimeField>::new_unchecked_with_cfg(m.evaluations[1].clone(), &());
            lo + r_last * (hi - lo)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::boolean::Boolean;
    use zinc_poly::univariate::binary_subspace::BinarySubspace;
    use zinc_poly::univariate::oblong_and::SKIPPED_VARS;

    const D: usize = 32;

    fn col_from_u32s(patterns: &[u32]) -> DenseMultilinearExtension<BinaryPoly<D>> {
        use std::array;
        let evaluations: Vec<BinaryPoly<D>> = patterns
            .iter()
            .map(|&p| {
                let coeffs: [Boolean; D] =
                    array::from_fn(|i| Boolean::new((p >> i) & 1 != 0));
                BinaryPoly::<D>::new(coeffs)
            })
            .collect();
        let num_vars = patterns.len().next_power_of_two().trailing_zeros() as usize;
        DenseMultilinearExtension {
            num_vars,
            evaluations,
        }
    }

    fn sample(seed: u64) -> Gf {
        let hi = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(29)
            ^ 0x1234_5678_9ABC_DEF0;
        Gf::from_words([seed ^ 0xA5A5_5A5A_0F0F_F0F0, hi])
    }

    const U: [u32; 8] = [
        0xDEAD_BEEF,
        0x0BAD_F00D,
        0x1234_5678,
        0x9ABC_DEF0,
        0xFFFF_0000,
        0x00FF_00FF,
        0xA5A5_A5A5,
        0x5A5A_5A5A,
    ];
    const V: [u32; 8] = [
        0xCAFE_BABE,
        0xFEED_FACE,
        0x8765_4321,
        0x0FED_CBA9,
        0x0F0F_F0F0,
        0xFF00_FF00,
        0x3333_CCCC,
        0xCCCC_3333,
    ];

    /// Convention canary: the gf8 kernel's per-row weighting
    /// (`embed(eq_small[x&7])·eq_big[x≫3]`, both little-endian) must equal the
    /// protocol's `build_eq_x_r` table at the hybrid point `r_had`. If this
    /// ever fails, the kernel and the sumcheck group disagree on `eq` and the
    /// merged claimed-sum check would break.
    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn hybrid_eq_table_matches_kernel_convention() {
        let scheme = Gf8Scheme::new();
        let num_vars = 5usize;
        let ic_point: Vec<Gf> = (0..num_vars as u64).map(|i| sample(900 + i)).collect();
        let r_had = merged_eq_point(&scheme, &ic_point);

        // Kernel view: eq_small (the embedded small tensor, low 3 bits) ⊗
        // eq_big over ic_point[3..] (eq_indicator, little-endian).
        let eq_small = eq_indicator(scheme.small_challenges());
        let eq_big = eq_indicator(&ic_point[3..]);
        // Protocol view: the build_eq_x_r table at r_had.
        let table = zinc_poly::utils::build_eq_x_r_vec(&r_had, &())
            .expect("eq table build");
        for x in 0..(1usize << num_vars) {
            assert_eq!(
                table[x],
                eq_small[x & 7] * eq_big[x >> 3],
                "eq convention mismatch at row {x}"
            );
        }
    }

    /// Completeness of the Phase-1 ↔ group identity on the gf8 lane: for
    /// honest data, the merged group's true sum at α (brute force with the
    /// hybrid eq + embed(H₈) Lagrange weights) equals `R₀(α)` reconstructed
    /// from the shipped extension evals; the reconstruction vanishes on the
    /// embed base domain; and a corrupted bit registers there.
    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn round_message_reconstruction_matches_group_sum() {
        let scheme = Gf8Scheme::new();
        let w0: Vec<u32> = U.iter().zip(&V).map(|(a, b)| a & b).collect();
        // Second relation with a row-shift + complement to exercise operands.
        let n = 8usize;
        let w1: Vec<u32> = (0..n)
            .map(|t| {
                let shifted = if t + 1 < n { U[t + 1] } else { 0 };
                (!U[t]) & shifted
            })
            .collect();
        let columns = [
            col_from_u32s(&U),
            col_from_u32s(&V),
            col_from_u32s(&w0),
            col_from_u32s(&w1),
        ];
        let specs = [
            F2HadamardSpec::plain(0, 1, 2),
            F2HadamardSpec {
                u: crate::f2_hadamard::F2Operand::col(0).complemented(),
                v: crate::f2_hadamard::F2Operand::shifted(0, 1),
                w: crate::f2_hadamard::F2Operand::col(3),
            },
        ];
        let num_vars = 3;
        let ic_point: Vec<Gf> = (0..num_vars as u64).map(|i| sample(100 + i)).collect();
        let r_had = merged_eq_point(&scheme, &ic_point);
        let eq = zinc_poly::utils::build_eq_x_r_vec(&r_had, &()).expect("eq build");
        let gamma_h = sample(7);
        let alpha = sample(42);

        let ops = merged_operand_columns::<D>(&columns, &specs, &[], num_vars);
        let msg = merged_round_message::<D>(&scheme, &ops, &ic_point, gamma_h);

        // Brute-force group sum at α: fold operands with L_b(α) over the
        // embed(H₈) base, accumulate with the hybrid eq.
        let lag = lagrange_alpha(&scheme, alpha);
        let fold = |c: &DenseMultilinearExtension<BinaryPoly<D>>, row: usize| -> Gf {
            let mut acc = Gf::zero();
            let mask = cell_mask::<D>(&c.evaluations[row]);
            for (b, w) in lag.iter().enumerate() {
                if (mask >> b) & 1 == 1 {
                    acc += *w;
                }
            }
            acc
        };
        let mut total = Gf::zero();
        for row in 0..(1usize << num_vars) {
            let mut s = Gf::zero();
            let mut gw = Gf::one();
            for rel in ops.chunks_exact(3) {
                s += gw * (fold(&rel[0], row) * fold(&rel[1], row) + fold(&rel[2], row));
                gw *= &gamma_h;
            }
            total += eq[row] * s;
        }

        assert_eq!(
            total,
            r0_at(&scheme, &msg, alpha),
            "honest group sum at α must equal the reconstructed R₀(α)"
        );

        // On the embed base domain, R₀ must genuinely vanish for honest data.
        let full = scheme.full_subspace();
        let mut coeffs = vec![Gf::zero(); 2 * WORD_BITS];
        coeffs[WORD_BITS..].copy_from_slice(&msg);
        for b in 0..WORD_BITS {
            assert_eq!(
                extrapolate_over_subspace(full, &coeffs, full.get(b)),
                Gf::zero(),
                "honest R₀ must vanish on base point {b}"
            );
        }
        // Sanity: the monomial subspace would NOT reconstruct this message
        // (different basis) — pin that the scheme's subspace is the embed one.
        let monomial = BinarySubspace::with_dim(SKIPPED_VARS + 1);
        assert_ne!(
            monomial.get(WORD_BITS),
            full.get(WORD_BITS),
            "embed(H₈) must differ from the monomial subspace"
        );

        // A corrupted bit must register on the base domain (the violation
        // indicator weighted by the hybrid eq is nonzero).
        let mut w_bad = w0.clone();
        w_bad[3] ^= 1 << 9;
        let columns_bad = [
            col_from_u32s(&U),
            col_from_u32s(&V),
            col_from_u32s(&w_bad),
            col_from_u32s(&w1),
        ];
        let ops_bad = merged_operand_columns::<D>(&columns_bad, &specs, &[], num_vars);
        let words = |c: &DenseMultilinearExtension<BinaryPoly<D>>| -> Vec<u64> {
            c.evaluations.iter().map(|x| cell_mask::<D>(x)).collect()
        };
        let (a0, b0, c0) = (words(&ops_bad[0]), words(&ops_bad[1]), words(&ops_bad[2]));
        let mut base_eval_at_bit9 = Gf::zero();
        for row in 0..(1usize << num_vars) {
            let bit = |w: &[u64]| (w[row] >> 9) & 1;
            let viol = bit(&a0) & bit(&b0) ^ bit(&c0);
            if viol == 1 {
                base_eval_at_bit9 += eq[row];
            }
        }
        assert_ne!(
            base_eval_at_bit9,
            Gf::zero(),
            "the corrupted bit must register on the base domain"
        );
    }
}
