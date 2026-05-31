//! Wiring-R Hadamard-discharge phase for the F_2 SHA-256 prover.
//!
//! Runs the coefficient-wise Hadamard zerocheck
//! ([`zinc_piop::lookup::hadamard`]) over the committed binary_poly
//! columns *before* the ψ_α projection is sampled, so α can later double
//! as the recombination element that pins the per-slice evals to the
//! column openings (see `protocol/src/f2_hadamard_plan.md`, Wiring R).
//!
//! ## Operands (Phase B)
//!
//! Each relation `W = U ⊙ V` is between three **operands**. An operand is
//! a bitwise expression over committed columns: the XOR of one or more
//! `(column, ↓Δ row-shift)` terms, optionally bitwise-complemented (`1 −`,
//! i.e. NOT). This covers the SHA AND relations C12–C14 (Appendix A of the
//! plan): `W_E ⊙ W_E^↓1`, `(1−W_E) ⊙ W_E^↓2`, and the Maj combo
//! `(W_A + W_A^↓2) ⊙ (W_A^↓1 + W_A^↓2) = (W_MAJ + W_A^↓2)`.
//!
//! **`+`/`1−` are F_2[X] addition (bitwise XOR), not F-linear sums** — so
//! the operand bit-slices are built by XOR-ing the source bits at the bit
//! level (a {0,1}-valued F MLE), *not* via the integer prover's
//! `build_virtual_booleanity_mles` (which does F-addition and would yield
//! `2` for `1+1`). Because ψ_α is an F_2-algebra hom,
//! `ψ_α(A ⊕ B) = ψ_α(A) + ψ_α(B)`, so an operand's parent eval at `r*_H`
//! derives *linearly* from the per-`(col, Δ)` evals plus the complement
//! constant `Σ_b α^b`. The `Δ = 0` pair evals are bound to the commitment
//! by the second-mp discharge; the `Δ ≠ 0` (row-shifted) pair evals are
//! prover-supplied / **trusted** for now — MLE evaluation does not commute
//! with row shifts, so binding them needs the row-shift discharge (shared
//! with the ledger's Issue 1; honest-prover-first per plan §6).

use crypto_primitives::Field;
use crypto_primitives::boolean::Boolean;
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
type Inner = <BinaryFieldGF128 as Field>::Inner;

/// One additive (XOR) term of a Hadamard operand: a committed column,
/// optionally row-shifted by `↓row_shift` (row `t` reads `col[t + Δ]`,
/// zero past the tail — the `↓Δ` convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F2OperandTerm {
    pub col: usize,
    pub row_shift: usize,
}

/// A Hadamard operand: the bitwise XOR of `terms`, optionally
/// bitwise-complemented (`complement = true` ⇒ the operand is
/// `1 ⊕ (⊕ terms)`, i.e. NOT). All at the bit level (`{0,1}`-valued).
#[derive(Clone, Debug)]
pub struct F2Operand {
    pub terms: Vec<F2OperandTerm>,
    pub complement: bool,
}

impl F2Operand {
    /// The plain committed column `col` (no shift, no complement).
    pub fn col(col: usize) -> Self {
        Self {
            terms: vec![F2OperandTerm { col, row_shift: 0 }],
            complement: false,
        }
    }

    /// The row-shifted column `col^↓shift`.
    pub fn shifted(col: usize, row_shift: usize) -> Self {
        Self {
            terms: vec![F2OperandTerm { col, row_shift }],
            complement: false,
        }
    }

    /// The XOR of the given `(col, row_shift)` terms.
    pub fn xor(terms: Vec<F2OperandTerm>) -> Self {
        Self {
            terms,
            complement: false,
        }
    }

    /// Mark this operand bitwise-complemented (`1 −`).
    pub fn complemented(mut self) -> Self {
        self.complement = true;
        self
    }
}

/// A coefficient-wise Hadamard relation `W = U ⊙ V` between three operands.
#[derive(Clone, Debug)]
pub struct F2HadamardSpec {
    pub u: F2Operand,
    pub v: F2Operand,
    pub w: F2Operand,
}

impl F2HadamardSpec {
    /// The Phase-A shape: three plain committed columns, no shifts.
    pub fn plain(u_col: usize, v_col: usize, w_col: usize) -> Self {
        Self {
            u: F2Operand::col(u_col),
            v: F2Operand::col(v_col),
            w: F2Operand::col(w_col),
        }
    }
}

/// A SHA-256 **adder** relation, verifying `t = x + y (mod 2^D)` via the
/// Binius carry identity `(x + X·c) ⊙ (y + X·c) = c + X·c`, where `c` is
/// the carry word of the addition. `t`, `x`, `y` are committed columns
/// (each optionally row-shifted). The carry is *defined* from them:
/// `c[b] = (t ⊕ x ⊕ y)[b+1]` for `b < D−1` (`= SHR¹(t⊕x⊕y)`), and the top
/// bit `c[D−1] = β = Maj(x[D−1], y[D−1], (t⊕x⊕y)[D−1])` (the overflow
/// carry, computed). The three Hadamard operands `U = x⊕X·c`, `V = y⊕X·c`,
/// `W = c⊕X·c` are built per row and checked by the zerocheck — meaningful
/// because `c` is defined from `t⊕x⊕y`, so the relation holds iff the carry
/// recurrence (hence `t = x+y`) does.
///
/// **Honest-prover-first**: the operands are derived from the committed
/// columns but their parents at `r*_H` are shipped **trusted** (the `β`
/// bit and the bit-shifts break the simple pair-eval algebra). Binding
/// them — committing `W_β` and a sound row/bit-shift discharge — is the
/// follow-up (`f2_hadamard_plan.md` §4.5 / §6, ledger Issue 1).
#[derive(Clone, Copy, Debug)]
pub struct F2AdderSpec {
    pub t: F2OperandTerm,
    pub x: F2OperandTerm,
    pub y: F2OperandTerm,
}

/// Read column `term.col` at row `r + term.row_shift` as a `u64` bitmask
/// (low `D` bits), zero past the trace tail (the `↓Δ` convention).
#[inline]
fn shifted_cell<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    term: F2OperandTerm,
    r: usize,
    n: usize,
) -> u64 {
    match r.checked_add(term.row_shift).filter(|&v| v < n) {
        Some(s) => cell_mask::<D>(&columns[term.col].evaluations[s]),
        None => 0,
    }
}

/// Build a `BinaryPoly<D>` cell from the low `D` bits of `m`.
#[inline]
fn cell_from_mask<const D: usize>(m: u64) -> BinaryPoly<D> {
    let coeffs: [Boolean; D] = std::array::from_fn(|i| Boolean::new((m >> i) & 1 != 0));
    BinaryPoly::<D>::new(coeffs)
}

/// Build the three packed operand columns `[U, V, W]` of an adder relation
/// (column-major over rows). See [`F2AdderSpec`] for the construction.
#[allow(clippy::arithmetic_side_effects)]
fn build_adder_operand_columns<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    spec: &F2AdderSpec,
    num_vars: usize,
) -> [DenseMultilinearExtension<BinaryPoly<D>>; 3] {
    let n = 1usize << num_vars;
    let mask = all_ones_mask::<D>();
    let top = D - 1;
    let mut u = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for r in 0..n {
        let tv = shifted_cell::<D>(columns, spec.t, r, n);
        let xv = shifted_cell::<D>(columns, spec.x, r, n);
        let yv = shifted_cell::<D>(columns, spec.y, r, n);
        let z = (tv ^ xv ^ yv) & mask;
        // β = Maj of the three top bits (the overflow carry).
        let xb = (xv >> top) & 1;
        let yb = (yv >> top) & 1;
        let zb = (z >> top) & 1;
        let beta = (xb & yb) | (xb & zb) | (yb & zb);
        // c = SHR¹(z) on bits 0..D−2, c[D−1] = β.
        let c = ((z >> 1) | (beta << top)) & mask;
        let xc = (c << 1) & mask; // X·c (bit D−1 drops, bit 0 = 0)
        u.push(cell_from_mask::<D>((xv ^ xc) & mask));
        v.push(cell_from_mask::<D>((yv ^ xc) & mask));
        w.push(cell_from_mask::<D>((c ^ xc) & mask));
    }
    let mk = |evals: Vec<BinaryPoly<D>>| DenseMultilinearExtension {
        evaluations: evals,
        num_vars,
    };
    [mk(u), mk(v), mk(w)]
}

/// α-projected MLE eval at `r_star` of each built adder operand column —
/// the **trusted** parent the recombination ties the operand slices to.
/// `alpha_pows` must be `[1, α, …, α^{D-1}]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn adder_operand_alpha_evals<const D: usize>(
    adder_operand_columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    alpha_pows: &[Gf],
    r_star: &[Gf],
) -> Vec<Gf> {
    let zero_inner = Gf::zero().into_inner();
    adder_operand_columns
        .iter()
        .map(|col| {
            let proj = project_column_with_powers::<D>(&col.evaluations, alpha_pows);
            let mle = DenseMultilinearExtension::from_evaluations_vec(
                col.num_vars,
                proj.iter().map(|x| *x.inner()).collect(),
                zero_inner.clone(),
            );
            mle.evaluate_with_config(r_star, &())
                .expect("adder operand α-eval at r*_H should succeed")
        })
        .collect()
}

/// Proof data produced by the Hadamard phase: the zerocheck proof plus the
/// per-slice evals at `r*_H` (column-major over the operands, in the order
/// `U_0, V_0, W_0, U_1, …`; `D` slices per operand).
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

/// Iterate the three operands of a relation in zerocheck order (U, V, W).
fn operands(spec: &F2HadamardSpec) -> [&F2Operand; 3] {
    [&spec.u, &spec.v, &spec.w]
}

/// The distinct `(col, row_shift)` pairs referenced by any operand term,
/// sorted. Both prover and verifier derive this identically from the
/// specs, so it need not travel in the proof. The `Δ = 0` pairs are bound
/// to the commitment; the `Δ ≠ 0` pairs carry trusted shifted evals.
pub fn distinct_pairs(specs: &[F2HadamardSpec]) -> Vec<(usize, usize)> {
    let mut pairs: Vec<(usize, usize)> = specs
        .iter()
        .flat_map(|s| operands(s))
        .flat_map(|op| op.terms.iter().map(|t| (t.col, t.row_shift)))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn pair_index(pairs: &[(usize, usize)], col: usize, row_shift: usize) -> usize {
    pairs
        .binary_search(&(col, row_shift))
        .expect("operand term must be in the distinct-pair set")
}

/// Read a `BinaryPoly<D>` cell as a `u64` bitmask (low `D` bits).
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn cell_mask<const D: usize>(bp: &BinaryPoly<D>) -> u64 {
    let mut out: u64 = 0;
    for (i, c) in bp.iter().enumerate() {
        if c.into_inner() {
            out |= 1u64 << i;
        }
    }
    out
}

/// All-ones mask over the low `D` bits (the bitwise-complement constant).
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn all_ones_mask<const D: usize>() -> u64 {
    if D >= 64 {
        u64::MAX
    } else {
        (1u64 << D) - 1
    }
}

/// Build the `D` bit-slice MLEs of one operand (column-major, `D` slices),
/// computing each bit as the bitwise XOR of the source bits at the right
/// row-shift, plus the complement constant. The result is `{0,1}`-valued.
#[allow(clippy::arithmetic_side_effects)]
fn build_operand_slices<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    operand: &F2Operand,
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<Inner>> {
    let n = 1usize << num_vars;
    let one_inner = *Gf::one().inner();
    let zero_inner = *Gf::zero().inner();

    // Per-row operand bitmask: complement ⊕ XOR of (shifted) source masks.
    let const_mask = if operand.complement {
        all_ones_mask::<D>()
    } else {
        0
    };
    let mut op_mask: Vec<u64> = vec![const_mask; n];
    for term in &operand.terms {
        let col = &columns[term.col];
        let shift = term.row_shift;
        for (t, slot) in op_mask.iter_mut().enumerate() {
            if let Some(s) = t.checked_add(shift).filter(|&v| v < n) {
                *slot ^= cell_mask::<D>(&col.evaluations[s]);
            }
        }
    }

    (0..D)
        .map(|b| {
            let evals: Vec<Inner> = op_mask
                .iter()
                .map(|m| {
                    if (m >> b) & 1 == 1 {
                        one_inner.clone()
                    } else {
                        zero_inner.clone()
                    }
                })
                .collect();
            DenseMultilinearExtension {
                evaluations: evals,
                num_vars,
            }
        })
        .collect()
}

/// Build the full operand-slice MLE set for the zerocheck: the AND specs'
/// operand slices first (`U_k, V_k, W_k` per spec), then the adder specs'
/// (`U_j, V_j, W_j` per adder, built via [`build_adder_operand_columns`] +
/// [`compute_bit_slices_flat`]). Operand index `3k+i` owns slices
/// `[(3k+i)·D, (3k+i+1)·D)`. Also returns the built adder operand packed
/// columns (3 per adder, in order) — the caller projects them at α after
/// it is sampled to get the trusted adder parents.
fn build_all_operand_slices<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
) -> (
    Vec<DenseMultilinearExtension<Inner>>,
    Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
) {
    let cfg = ();
    let mut out = Vec::new();
    for spec in specs {
        for operand in operands(spec) {
            out.extend(build_operand_slices::<D>(columns, operand, num_vars));
        }
    }
    let mut adder_cols = Vec::with_capacity(adder_specs.len().saturating_mul(3));
    for spec in adder_specs {
        let uvw = build_adder_operand_columns::<D>(columns, spec, num_vars);
        out.extend(compute_bit_slices_flat::<Gf, D>(&uvw, &cfg));
        adder_cols.extend(uvw);
    }
    (out, adder_cols)
}

/// One `HadamardTriple` per relation, indexing the operand-slice groups
/// laid out by [`build_all_operand_slices`] (`U_k = 3k`, `V_k = 3k+1`,
/// `W_k = 3k+2`).
#[allow(clippy::arithmetic_side_effects)]
fn relations_into_triples(num_relations: usize) -> Vec<HadamardTriple> {
    (0..num_relations)
        .map(|k| HadamardTriple {
            u_col: 3 * k,
            v_col: 3 * k + 1,
            w_col: 3 * k + 2,
        })
        .collect()
}

/// Derive each operand's parent eval `ψ_α(operand)(r*_H)` from the
/// per-pair evals and the complement constant `Σ_{b<D} α^b`. Output is in
/// operand order (`U_0, V_0, W_0, U_1, …`) to match the zerocheck slices
/// and feed [`verify_bit_decomposition_consistency`].
///
/// `pair_evals[i] = ψ_α(pairs[i].col ^↓ pairs[i].1)(r*_H)`. Since
/// `ψ_α(A ⊕ B) = ψ_α(A) + ψ_α(B)` (char-2 additivity), an operand's parent
/// is the F-sum of its terms' pair evals plus (if complemented) `Σ α^b`.
#[allow(clippy::arithmetic_side_effects)]
pub fn derive_operand_parents(
    specs: &[F2HadamardSpec],
    pairs: &[(usize, usize)],
    pair_evals: &[Gf],
    alpha_pows: &[Gf],
) -> Vec<Gf> {
    // Σ_{b<D} α^b — the bitwise-complement constant.
    let complement_const: Gf = alpha_pows.iter().fold(Gf::zero(), |acc, p| acc + *p);
    let mut out = Vec::new();
    for spec in specs {
        for operand in operands(spec) {
            let mut parent = if operand.complement {
                complement_const
            } else {
                Gf::zero()
            };
            for term in &operand.terms {
                parent += pair_evals[pair_index(pairs, term.col, term.row_shift)];
            }
            out.push(parent);
        }
    }
    out
}

/// Compute, for each distinct pair `(col, Δ)`, the α-projected MLE eval of
/// the row-shifted column at `r*`: `ψ_α(col ^↓Δ)(r*)`. For `Δ = 0` this is
/// `ψ_α(col)(r*)` (later bound to the commitment); for `Δ ≠ 0` it is a
/// shifted eval the verifier trusts (the row-shift gap). `alpha_pows` must
/// be `[1, α, …, α^{D-1}]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn pair_alpha_evals<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    pairs: &[(usize, usize)],
    alpha_pows: &[Gf],
    r_star: &[Gf],
) -> Vec<Gf> {
    let zero_inner = Gf::zero().into_inner();
    pairs
        .iter()
        .map(|&(col, shift)| {
            let column = &columns[col];
            let n = column.evaluations.len();
            // ψ_α(col) per row, then row-shift by Δ (zero past the tail).
            let proj = project_column_with_powers::<D>(&column.evaluations, alpha_pows);
            let shifted: Vec<Inner> = (0..n)
                .map(|t| match t.checked_add(shift).filter(|&v| v < n) {
                    Some(s) => *proj[s].inner(),
                    None => zero_inner.clone(),
                })
                .collect();
            let mle =
                DenseMultilinearExtension::from_evaluations_vec(column.num_vars, shifted, zero_inner.clone());
            mle.evaluate_with_config(r_star, &())
                .expect("shifted α-eval at r*_H should succeed")
        })
        .collect()
}

/// Prover half of the Wiring-R Hadamard phase. Returns `None` when there
/// are no relations. On success returns the proof, the distinct
/// `(col, row_shift)` pair list (for the AND specs), the Hadamard sumcheck
/// point `r*_H`, and the built adder operand packed columns (3 per adder,
/// in order) — the caller needs all of these after α: the pairs to compute
/// the AND pair-evals, and the adder columns to compute the trusted adder
/// operand parents.
pub fn prove_f2_hadamard_phase<const D: usize>(
    transcript: &mut impl Transcript,
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
    specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    ic_evaluation_point: &[Gf],
    num_vars: usize,
) -> Option<(
    F2HadamardProof,
    Vec<(usize, usize)>,
    Vec<Gf>,
    Vec<DenseMultilinearExtension<BinaryPoly<D>>>,
)> {
    if specs.is_empty() && adder_specs.is_empty() {
        return None;
    }
    let cfg = ();
    let pairs = distinct_pairs(specs);
    let (bit_slice_mles, adder_cols) =
        build_all_operand_slices::<D>(columns, specs, adder_specs, num_vars);
    let num_relations = specs.len().saturating_add(adder_specs.len());
    let relations = relations_into_triples(num_relations);

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
        pairs,
        r_star_h,
        adder_cols,
    ))
}

/// Verifier half: checks the zerocheck (claimed sum zero + round
/// consistency + closing eval) and returns `(distinct_pairs, r*_H)`. The
/// caller then derives the AND operand parents via [`derive_operand_parents`]
/// (from the per-pair evals), appends the trusted adder operand parents, and
/// runs
/// [`zinc_piop::lookup::booleanity::verify_bit_decomposition_consistency`]
/// against `proof.bit_slice_evals` to pin the operand slices.
pub fn verify_f2_hadamard_phase<const D: usize>(
    transcript: &mut impl Transcript,
    proof: &F2HadamardProof,
    specs: &[F2HadamardSpec],
    adder_specs: &[F2AdderSpec],
    ic_evaluation_point: &[Gf],
    num_vars: usize,
) -> Result<(Vec<(usize, usize)>, Vec<Gf>), F2HadamardVerifyError> {
    let cfg = ();
    let pairs = distinct_pairs(specs);
    let num_relations = specs.len().saturating_add(adder_specs.len());
    let relations = relations_into_triples(num_relations);

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

    Ok((pairs, r_star_h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::Field;
    use crypto_primitives::boolean::Boolean;
    use zinc_piop::lookup::booleanity::verify_bit_decomposition_consistency;
    use zinc_poly::univariate::binary_gf128::alpha_powers;
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

    /// Drive an honest prove → verify → recombination round-trip for a set
    /// of AND + adder specs over the given columns, at a test α. Panics if
    /// any stage fails. Returns nothing — the asserts are the test.
    fn round_trip<const D: usize>(
        columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
        specs: &[F2HadamardSpec],
        adder_specs: &[F2AdderSpec],
        num_vars: usize,
    ) {
        let ic = ic_point();
        let mut pt = Blake3Transcript::new();
        let (proof, pairs_p, rstar_p, adder_cols) =
            prove_f2_hadamard_phase::<D>(&mut pt, columns, specs, adder_specs, &ic, num_vars)
                .unwrap();

        let mut vt = Blake3Transcript::new();
        let (pairs_v, rstar_v) =
            verify_f2_hadamard_phase::<D>(&mut vt, &proof, specs, adder_specs, &ic, num_vars)
                .unwrap();
        assert_eq!(pairs_p, pairs_v);
        assert_eq!(rstar_p, rstar_v);

        // Recombination at a test α: AND operand parents derived from the
        // pair evals, adder operand parents from the built columns (trusted);
        // then check the sent operand slices recombine to them.
        let alpha = Gf::from_words([0x9E37_79B9, 0x1234_5678]);
        let pows = alpha_powers(&alpha, D);
        let pair_evals = pair_alpha_evals::<D>(columns, &pairs_v, &pows, &rstar_v);
        let mut parents = derive_operand_parents(specs, &pairs_v, &pair_evals, &pows);
        parents.extend(adder_operand_alpha_evals::<D>(&adder_cols, &pows, &rstar_v));
        verify_bit_decomposition_consistency(&parents, &proof.bit_slice_evals, &alpha, D)
            .expect("operand recombination");
    }

    #[test]
    fn plain_and_round_trips() {
        const D: usize = 4;
        let num_vars = 3;
        let u = [0b1011u32, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let v = [0b1101u32, 0b1010, 0b0111, 0b1001, 0b0110, 0b1111, 0b1000, 0b0101];
        let w: Vec<u32> = u.iter().zip(&v).map(|(a, b)| a & b).collect();
        let columns = [
            col_from_u32s::<D>(&u),
            col_from_u32s::<D>(&v),
            col_from_u32s::<D>(&w),
        ];
        round_trip::<D>(&columns, &[F2HadamardSpec::plain(0, 1, 2)], &[], num_vars);
    }

    #[test]
    fn row_shift_and_round_trips() {
        const D: usize = 4;
        let num_vars = 3;
        let n = 1usize << num_vars;
        // W = A ⊙ A^↓1  (row t: A[t] & A[t+1], with A[n]=0).
        let a = [0b1011u32, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let w: Vec<u32> = (0..n)
            .map(|t| if t + 1 < n { a[t] & a[t + 1] } else { 0 })
            .collect();
        let columns = [col_from_u32s::<D>(&a), col_from_u32s::<D>(&w)];
        let spec = F2HadamardSpec {
            u: F2Operand::col(0),
            v: F2Operand::shifted(0, 1),
            w: F2Operand::col(1),
        };
        round_trip::<D>(&columns, &[spec], &[], num_vars);
    }

    #[test]
    fn complement_and_round_trips() {
        const D: usize = 4;
        let num_vars = 3;
        let mask = (1u32 << D) - 1;
        // W = (1 − A) ⊙ B  (row-wise bitwise NOT of A, AND B).
        let a = [0b1011u32, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let b = [0b1101u32, 0b1010, 0b0111, 0b1001, 0b0110, 0b1111, 0b1000, 0b0101];
        let w: Vec<u32> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| ((!x) & mask) & y)
            .collect();
        let columns = [
            col_from_u32s::<D>(&a),
            col_from_u32s::<D>(&b),
            col_from_u32s::<D>(&w),
        ];
        let spec = F2HadamardSpec {
            u: F2Operand::col(0).complemented(),
            v: F2Operand::col(1),
            w: F2Operand::col(2),
        };
        round_trip::<D>(&columns, &[spec], &[], num_vars);
    }

    #[test]
    fn maj_combo_round_trips() {
        // C14 shape: (A + A^↓2) ⊙ (A^↓1 + A^↓2) = (Maj + A^↓2), where
        // Maj = (A + A^↓2) & (A^↓1 + A^↓2) − A^↓2 ... we just fill W to
        // satisfy the relation directly.
        const D: usize = 4;
        let num_vars = 3;
        let n = 1usize << num_vars;
        let a = [0b1011u32, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let sh = |arr: &[u32], d: usize, t: usize| if t + d < n { arr[t + d] } else { 0 };
        // U[t] = A[t] ^ A[t+2], V[t] = A[t+1] ^ A[t+2], W[t] = U[t] & V[t].
        // We store W as a committed column equal to (U⊙V) ^ A^↓2 ^ A^↓2 = U⊙V,
        // but the relation's W operand is (W_col ^ A^↓2); pick W_col = (U⊙V) ^ A^↓2.
        let w_col: Vec<u32> = (0..n)
            .map(|t| {
                let u = a[t] ^ sh(&a, 2, t);
                let v = sh(&a, 1, t) ^ sh(&a, 2, t);
                (u & v) ^ sh(&a, 2, t)
            })
            .collect();
        let columns = [col_from_u32s::<D>(&a), col_from_u32s::<D>(&w_col)];
        // U = A + A^↓2 ; V = A^↓1 + A^↓2 ; W = W_col + A^↓2.
        let spec = F2HadamardSpec {
            u: F2Operand::xor(vec![
                F2OperandTerm { col: 0, row_shift: 0 },
                F2OperandTerm { col: 0, row_shift: 2 },
            ]),
            v: F2Operand::xor(vec![
                F2OperandTerm { col: 0, row_shift: 1 },
                F2OperandTerm { col: 0, row_shift: 2 },
            ]),
            w: F2Operand::xor(vec![
                F2OperandTerm { col: 1, row_shift: 0 },
                F2OperandTerm { col: 0, row_shift: 2 },
            ]),
        };
        round_trip::<D>(&columns, &[spec], &[], num_vars);
    }

    #[test]
    fn corrupt_w_is_rejected() {
        const D: usize = 4;
        let num_vars = 3;
        let u = [0b1011u32, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let v = [0b1101u32, 0b1010, 0b0111, 0b1001, 0b0110, 0b1111, 0b1000, 0b0101];
        let mut w: Vec<u32> = u.iter().zip(&v).map(|(a, b)| a & b).collect();
        w[0] ^= 1; // corrupt
        let columns = [
            col_from_u32s::<D>(&u),
            col_from_u32s::<D>(&v),
            col_from_u32s::<D>(&w),
        ];
        let specs = [F2HadamardSpec::plain(0, 1, 2)];
        let ic = ic_point();
        let mut pt = Blake3Transcript::new();
        let (proof, _, _, _) =
            prove_f2_hadamard_phase::<D>(&mut pt, &columns, &specs, &[], &ic, num_vars).unwrap();
        let mut vt = Blake3Transcript::new();
        let rejected = verify_f2_hadamard_phase::<D>(&mut vt, &proof, &specs, &[], &ic, num_vars);
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
        assert!(
            prove_f2_hadamard_phase::<D>(&mut pt, &cols, &[], &[], &ic_point(), 3).is_none()
        );
    }

    #[test]
    fn adder_round_trips() {
        // Honest t = x + y (mod 2^D): the Binius carry relation holds, so
        // the zerocheck sum is 0 and the recombination ties the (trusted)
        // operand parents.
        const D: usize = 4;
        let num_vars = 3;
        let mask: u32 = (1 << D) - 1;
        let x = [0b1011u32, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let y = [0b1101u32, 0b1010, 0b0111, 0b1001, 0b0110, 0b1111, 0b1000, 0b0101];
        let t: Vec<u32> = x.iter().zip(&y).map(|(a, b)| (a.wrapping_add(*b)) & mask).collect();
        let columns = [
            col_from_u32s::<D>(&t),
            col_from_u32s::<D>(&x),
            col_from_u32s::<D>(&y),
        ];
        let adder = F2AdderSpec {
            t: F2OperandTerm { col: 0, row_shift: 0 },
            x: F2OperandTerm { col: 1, row_shift: 0 },
            y: F2OperandTerm { col: 2, row_shift: 0 },
        };
        round_trip::<D>(&columns, &[], &[adder], num_vars);
    }

    #[test]
    fn adder_rejects_wrong_sum() {
        // t ≠ x + y ⇒ the carry recurrence fails ⇒ non-zero claimed sum.
        const D: usize = 4;
        let num_vars = 3;
        let mask: u32 = (1 << D) - 1;
        let x = [0b1011u32, 0b0110, 0b1111, 0b0001, 0b1010, 0b0101, 0b1100, 0b0011];
        let y = [0b1101u32, 0b1010, 0b0111, 0b1001, 0b0110, 0b1111, 0b1000, 0b0101];
        let mut t: Vec<u32> = x.iter().zip(&y).map(|(a, b)| (a.wrapping_add(*b)) & mask).collect();
        t[0] ^= 0b10; // break t = x + y at a non-LSB bit (so a carry differs)
        let columns = [
            col_from_u32s::<D>(&t),
            col_from_u32s::<D>(&x),
            col_from_u32s::<D>(&y),
        ];
        let adder = [F2AdderSpec {
            t: F2OperandTerm { col: 0, row_shift: 0 },
            x: F2OperandTerm { col: 1, row_shift: 0 },
            y: F2OperandTerm { col: 2, row_shift: 0 },
        }];
        let ic = ic_point();
        let mut pt = Blake3Transcript::new();
        let (proof, _, _, _) =
            prove_f2_hadamard_phase::<D>(&mut pt, &columns, &[], &adder, &ic, num_vars).unwrap();
        let mut vt = Blake3Transcript::new();
        let rejected = verify_f2_hadamard_phase::<D>(&mut vt, &proof, &[], &adder, &ic, num_vars);
        assert!(matches!(
            rejected,
            Err(F2HadamardVerifyError::Hadamard(
                HadamardError::NonZeroClaimedSum { .. }
            ))
        ));
    }
}
