//! **Witness-side binding for the F_2 lookup adder** (closes Issue-1).
//!
//! The grand-product GKR ([`zinc_piop::lookup::gkr_lookup`]) reduces the
//! witness side to one claim: `leafMLE(r_w) = eval_w`, for the tensor leaf
//! layout of [`super::f2_prove`] (`leaf[p·n+r] = 1 + mask_a[r]·(1+δ+fp_{a,i}[r])`,
//! rows = low vars, pair index `p = a·nl+i` = high vars). Splitting
//! `r_w = (r_row, r_pair)` and using `Σ_x eq(q,x) = 1`:
//!
//! ```text
//!   eval_w = PubMLE + Σ_r eq(r_row,r) · Σ_a mask_a[r] · Q_a[r]
//!   PubMLE = 1 + (1+δ)·Σ_p eq(r_pair,p)·maskMLE_a(r_row)            (public)
//!   Q_a[r] = Σ_i eq(r_pair, a·nl+i)·fp_{a,i}[r]
//! ```
//!
//! `fp_{a,i}[r]` is **F_2-linear in the committed bits** (word limbs +
//! committed carry bits), so `Q_a` is a public-weight projection of the
//! committed columns: `Q_a[r] = Σ_{(col,Δ)} Σ_β W_a[(col,Δ)][β]·bit_β(col[r+Δ])`.
//!
//! This module proves the masked row-sum `B := eval_w + PubMLE` with **one
//! degree-3 sumcheck** (`comb = eq·Σ_a mask_a·Q_a`), reducing it to
//! **bit-slice shifted evals** `bitslice_β(col↓Δ)-MLE(r★)` at the sumcheck
//! point `r★`. The verifier recombines those with the public weights
//! (`Q̃_a(r★)`), evaluates `eq` (closed form) and the public masks, and
//! checks the sumcheck's final evaluation.
//!
//! The shipped bit-slice evals are themselves bound to the commitment by the
//! ψ_α recombination `Σ_β α^β·bit_eval_β = ψ_α(col↓Δ)(r★)` — a pointed-shift
//! claim on the already-bound α-block, folded into the existing
//! multipoint-eval (wired in the prove pipeline, not here).

use crypto_primitives::Field;
use zinc_piop::sumcheck::{MLSumcheck, SumCheckError, SumcheckProof};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;
use zinc_transcript::traits::Transcript;
use zinc_utils::cfg_iter;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::f2_hadamard::{F2AdderSpec, F2OperandTerm};

type Gf = BinaryFieldGF128;
type Inner = <Gf as Field>::Inner;

/// `X^b` as a GF(2^128) element (`b < 64`): the bit-pattern `1 << b`.
#[inline]
fn x_pow(b: usize) -> Gf {
    Gf::from_words([1u64 << b, 0])
}

/// `eq(point, j)` for an integer index `j` (point coordinates LSB-first).
#[allow(clippy::arithmetic_side_effects)]
pub fn eq_at_index(point: &[Gf], j: usize) -> Gf {
    let one = Gf::one();
    point.iter().enumerate().fold(one, |acc, (b, c)| {
        acc * if (j >> b) & 1 == 1 { *c } else { one + *c }
    })
}

/// Per-relation public weight map: `(col, row_shift) → w ∈ Gf^D` with
/// `Q_a[r] = Σ_{(col,Δ)} Σ_β w[β]·bit_β(col[r+Δ])`. Mirrors the fingerprint
/// `fp = emb(x_i) + γ·emb(y_i⊕y2_i) + γ²·cin_i + γ³·emb(t_i) + γ⁴·cout_i`
/// with `cin_i = carry.bit_{i−1}` (`cin_0 = 0`), `cout_i = carry.bit_i`,
/// each weighted by `eq(r_pair, a·nl+i)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn lookup_binding_weights<const D: usize>(
    adder_specs: &[F2AdderSpec],
    carry_col_base: usize,
    r_pair: &[Gf],
    gamma: &Gf,
    limb_bits: usize,
) -> Vec<Vec<((usize, usize), Vec<Gf>)>> {
    let nl = D / limb_bits;
    let g2 = *gamma * *gamma;
    let g3 = g2 * *gamma;
    let g4 = g3 * *gamma;
    // Find-or-insert the (col, Δ) slot, returning its index (NLL-friendly).
    fn slot<const D: usize>(
        acc: &mut Vec<((usize, usize), Vec<Gf>)>,
        key: (usize, usize),
    ) -> usize {
        match acc.iter().position(|(k, _)| *k == key) {
            Some(idx) => idx,
            None => {
                acc.push((key, vec![Gf::zero(); D]));
                acc.len() - 1
            }
        }
    }
    adder_specs
        .iter()
        .enumerate()
        .map(|(a, spec)| {
            // (col, Δ) → weight vector, insertion-ordered.
            let mut acc: Vec<((usize, usize), Vec<Gf>)> = Vec::new();
            for i in 0..nl {
                let eqp = eq_at_index(r_pair, a * nl + i);
                let mut add = |acc: &mut Vec<((usize, usize), Vec<Gf>)>,
                               term: &F2OperandTerm,
                               role: Gf| {
                    let idx = slot::<D>(acc, (term.col, term.row_shift));
                    let w = &mut acc[idx].1;
                    for b in 0..limb_bits {
                        w[limb_bits * i + b] = w[limb_bits * i + b] + eqp * role * x_pow(b);
                    }
                };
                add(&mut acc, &spec.x, Gf::one());
                add(&mut acc, &spec.y, *gamma);
                if let Some(y2) = &spec.y2 {
                    add(&mut acc, y2, *gamma);
                }
                add(&mut acc, &spec.t, g3);
                // Carry reads (bit-granular, not limb-shaped).
                let idx = slot::<D>(&mut acc, (carry_col_base + a, 0usize));
                let w = &mut acc[idx].1;
                if i > 0 {
                    w[i - 1] = w[i - 1] + eqp * g2; // cin_i = carry.bit_{i−1}
                }
                w[i] = w[i] + eqp * g4; // cout_i = carry.bit_i
            }
            acc
        })
        .collect()
}

/// The distinct `(col, Δ)` pairs referenced by the weight maps, in first-use
/// order — the bit-slice eval groups the prover ships and the fold binds.
pub fn binding_distinct_pairs(
    weights: &[Vec<((usize, usize), Vec<Gf>)>],
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for per_rel in weights {
        for (key, _) in per_rel {
            if !pairs.contains(key) {
                pairs.push(*key);
            }
        }
    }
    pairs
}

/// Proof of the binding sumcheck: the sumcheck itself plus the bit-slice
/// shifted evals `bitslice_β(col↓Δ)-MLE(r★)` per distinct pair (row-major:
/// `bit_evals[pair][β]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupBindingProof<const D: usize> {
    pub sumcheck: SumcheckProof<Gf>,
    pub bit_evals: Vec<Vec<Gf>>,
}

/// Failure modes of [`verify_lookup_binding`].
#[derive(Debug, thiserror::Error)]
pub enum LookupBindingError {
    #[error("binding sumcheck error: {0}")]
    Sumcheck(#[from] SumCheckError<Gf>),
    #[error("binding claimed sum != eval_w + PubMLE")]
    ClaimedSumMismatch,
    #[error("binding final evaluation mismatch (eq·Σ mask·Q != sumcheck eval)")]
    FinalEvalMismatch,
    #[error("binding bit-eval shape mismatch")]
    Shape,
}

/// Public per-relation active-row mask as a 0/1 `Gf` vector (empty spec mask
/// = all-active).
fn mask_vec(spec: &F2AdderSpec, n: usize) -> Vec<Gf> {
    let one = Gf::one();
    (0..n)
        .map(|r| {
            let active =
                spec.active_rows.is_empty() || spec.active_rows.get(r).copied().unwrap_or(false);
            if active { one } else { Gf::zero() }
        })
        .collect()
}

/// MLE eval at `point` of a `Gf` row vector.
fn row_mle_eval(v: &[Gf], num_vars: usize, point: &[Gf]) -> Gf {
    let zero_inner = Gf::zero().into_inner();
    let mle = DenseMultilinearExtension::from_evaluations_vec(
        num_vars,
        v.iter().map(|x| *x.inner()).collect(),
        zero_inner,
    );
    <DenseMultilinearExtension<Inner> as MultilinearExtensionWithConfig<Gf>>::evaluate_with_config(
        mle, &point.to_vec(), &(),
    )
    .expect("row MLE eval")
}

/// `PubMLE(r_w)` — the public part of the leaf-MLE claim:
/// `1 + (1+δ)·Σ_{a,i} eq(r_pair, a·nl+i)·maskMLE_a(r_row)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn lookup_binding_public_part<const D: usize>(
    adder_specs: &[F2AdderSpec],
    num_vars: usize,
    limb_bits: usize,
    r_row: &[Gf],
    r_pair: &[Gf],
    delta: &Gf,
) -> Gf {
    let n = 1usize << num_vars;
    let nl = D / limb_bits;
    let one = Gf::one();
    let mut out = one;
    for (a, spec) in adder_specs.iter().enumerate() {
        let m_eval = row_mle_eval(&mask_vec(spec, n), num_vars, r_row);
        let eq_sum: Gf = (0..nl)
            .map(|i| eq_at_index(r_pair, a * nl + i))
            .fold(Gf::zero(), |acc, e| acc + e);
        out = out + (one + *delta) * eq_sum * m_eval;
    }
    out
}

/// Prover: one degree-3 sumcheck for `B = Σ_r eq(r_row,r)·Σ_a mask_a[r]·Q_a[r]`,
/// then the bit-slice shifted evals at the sumcheck point `r★`.
/// Returns `(proof, r★)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_lookup_binding<const D: usize>(
    transcript: &mut impl Transcript,
    cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    adder_specs: &[F2AdderSpec],
    weights: &[Vec<((usize, usize), Vec<Gf>)>],
    num_vars: usize,
    r_row: &[Gf],
) -> (LookupBindingProof<D>, Vec<Gf>) {
    let n = 1usize << num_vars;
    let zero_inner = Gf::zero().into_inner();
    let num_rel = adder_specs.len();

    // Shifted cell read (zero past the tail) as a u64 bit mask.
    let cell_at = |col: usize, shift: usize, r: usize| -> u64 {
        match r.checked_add(shift).filter(|&i| i < n) {
            Some(i) => crate::f2_hadamard::cell_mask::<D>(&cols[col].evaluations[i]),
            None => 0,
        }
    };

    // Q_a[r] = Σ_{(col,Δ)} Σ_{β set} w[β].
    let q_rows: Vec<Vec<Gf>> = cfg_iter!(weights)
        .map(|per_rel| {
            (0..n)
                .map(|r| {
                    let mut acc = Gf::zero();
                    for ((col, shift), w) in per_rel {
                        let mut m = cell_at(*col, *shift, r);
                        while m != 0 {
                            let b = m.trailing_zeros() as usize;
                            acc = acc + w[b];
                            m &= m - 1;
                        }
                    }
                    acc
                })
                .collect()
        })
        .collect();

    let mk = |v: Vec<Gf>| {
        DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            v.iter().map(|x| *x.inner()).collect(),
            zero_inner,
        )
    };
    let eq_mle = zinc_poly::utils::build_eq_x_r_inner(r_row, &()).expect("eq table");
    let mut mles = Vec::with_capacity(1 + 2 * num_rel);
    mles.push(eq_mle);
    for spec in adder_specs {
        mles.push(mk(mask_vec(spec, n)));
    }
    for q in &q_rows {
        mles.push(mk(q.clone()));
    }

    // comb = eq · Σ_a mask_a · Q_a   (degree 3).
    let comb = move |vals: &[Gf]| -> Gf {
        let eq = vals[0];
        let mut s = Gf::zero();
        for a in 0..num_rel {
            s = s + vals[1 + a] * vals[1 + num_rel + a];
        }
        eq * s
    };
    let (sumcheck, state) =
        MLSumcheck::prove_as_subprotocol(transcript, mles, num_vars, 3, comb, &());
    let r_star = state.randomness.clone();

    // Bit-slice shifted evals at r★ per distinct (col, Δ).
    let pairs = binding_distinct_pairs(weights);
    let bit_evals: Vec<Vec<Gf>> = cfg_iter!(pairs)
        .map(|&(col, shift)| {
            (0..D)
                .map(|beta| {
                    let v: Vec<Gf> = (0..n)
                        .map(|r| {
                            if (cell_at(col, shift, r) >> beta) & 1 == 1 {
                                Gf::one()
                            } else {
                                Gf::zero()
                            }
                        })
                        .collect();
                    row_mle_eval(&v, num_vars, &r_star)
                })
                .collect()
        })
        .collect();

    // Absorb the bit evals (they precede any challenge that binds them).
    let mut buf = vec![0u8; <Inner as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
    for group in &bit_evals {
        for v in group {
            transcript.absorb_random_field(v, &mut buf);
        }
    }

    (LookupBindingProof { sumcheck, bit_evals }, r_star)
}

/// Verifier: checks the sumcheck against `claimed_b = eval_w + PubMLE(r_w)`,
/// recombines `Q̃_a(r★)` from the shipped bit-slice evals with the public
/// weights, and checks the final evaluation. Returns `r★` (the point at
/// which the bit evals must be bound to the commitment by the caller).
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_lookup_binding<const D: usize>(
    transcript: &mut impl Transcript,
    proof: &LookupBindingProof<D>,
    claimed_b: &Gf,
    adder_specs: &[F2AdderSpec],
    weights: &[Vec<((usize, usize), Vec<Gf>)>],
    num_vars: usize,
    r_row: &[Gf],
) -> Result<Vec<Gf>, LookupBindingError> {
    if proof.sumcheck.claimed_sum != *claimed_b {
        return Err(LookupBindingError::ClaimedSumMismatch);
    }
    let subclaim =
        MLSumcheck::<Gf>::verify_as_subprotocol(transcript, num_vars, 3, &proof.sumcheck, &())?;
    let r_star = subclaim.point.clone();

    let pairs = binding_distinct_pairs(weights);
    if proof.bit_evals.len() != pairs.len() || proof.bit_evals.iter().any(|g| g.len() != D) {
        return Err(LookupBindingError::Shape);
    }
    let mut buf = vec![0u8; <Inner as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
    for group in &proof.bit_evals {
        for v in group {
            transcript.absorb_random_field(v, &mut buf);
        }
    }

    let n = 1usize << num_vars;
    let one = Gf::one();
    let eq_val = zinc_poly::utils::eq_eval(&r_star, &r_row.to_vec(), one)
        .map_err(|_| LookupBindingError::FinalEvalMismatch)?;

    // Σ_a mask̃_a(r★)·Q̃_a(r★), with Q̃ recombined from the bit evals.
    let mut s = Gf::zero();
    for (a, spec) in adder_specs.iter().enumerate() {
        let m_eval = row_mle_eval(&mask_vec(spec, n), num_vars, &r_star);
        let mut q = Gf::zero();
        for (key, w) in &weights[a] {
            let gi = pairs.iter().position(|p| p == key).expect("pair from weights");
            for beta in 0..D {
                q = q + w[beta] * proof.bit_evals[gi][beta];
            }
        }
        s = s + m_eval * q;
    }
    if eq_val * s != subclaim.expected_evaluation {
        return Err(LookupBindingError::FinalEvalMismatch);
    }
    Ok(r_star)
}
