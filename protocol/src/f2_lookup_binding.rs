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
//! Every fingerprint term is a **fixed-family read** of a committed column.
//! Two family kinds (`nf = 2·nl − 1` families total, indexed `f`):
//!
//! - **Limb families** (`f = i < nl`): `L_i(col)[r] = Σ_{b<ℓ}
//!   bit_{ℓi+b}(col[r])·X^b` — the operand/result limb coordinates.
//! - **Boundary-bit families** (`f = nl+j−1`, `j ∈ 1..nl`): `e_{ℓj}(col)[r]
//!   = bit_{ℓj}(col[r])` — the **virtualized carries**. Carry-into-bit-p of
//!   an add `t = x+y` satisfies `C[p] = (x⊕y⊕t)[p]`, so `cin_i = cout_{i−1}
//!   = z[ℓi] = Σ_op e_{ℓi}(op↓Δ)` is a Gf-linear parity of single-bit reads
//!   of the ALREADY-COMMITTED operands — no carry columns exist. `cin_0 = 0`
//!   is a constant; the LAST limb is marginalised (no cout coordinate, the
//!   `γ⁴·emb(2)` tag constant instead — see `add_lookup::marginal_tag`).
//!
//! So
//!
//! ```text
//!   Q_a[r] = const_a + Σ_{(col,Δ,f)} coeff_{a,(col,Δ,f)} · F_f(col↓Δ)[r]
//! ```
//!
//! with public coefficients `coeff = eq(r_pair,·)·γ^ρ` and the public tag
//! constant `const_a`. This module proves the masked row-sum `B := eval_w +
//! PubMLE` with **one degree-3 sumcheck** (`comb = eq·Σ_a mask_a·Q_a`),
//! reducing it to the **family shifted evals** `F̃_f(col↓Δ)(r★)` at the
//! sumcheck point — shipped per distinct `(col,Δ)` (all `nf` families). The
//! verifier recombines `Q̃_a(r★)` from them with the public coefficients and
//! checks the final evaluation.
//!
//! **Why fixed families (and not per-bit evals recombined under α):** each
//! shipped eval must be bound to the commitment. A fixed family admits the
//! z-block pattern — pointed-shift claims into the multipoint fold against
//! family-block columns, whose `r_0`-evals are checked against the opened
//! `a′` via `Σ_g γ_g·F_{f,g}(r_0) = w_f(a′)` with the **fresh** post-absorb
//! `γ_open` (SZ binds each eval individually). Per-bit evals recombined
//! under the *already-known* α would NOT bind (a cheater shifts mass between
//! slices keeping the α-combination). Note the boundary-bit families are NOT
//! that unsound pattern: each `e_{ℓj}` is its own fixed family with its own
//! block columns and its own `a′` consistency under the fresh `γ_open`. The
//! fold wiring lives in the prove pipeline; this module's contract ends at
//! the shipped family evals.

use crypto_primitives::Field;
use zinc_piop::sumcheck::{MLSumcheck, SumCheckError, SumcheckProof};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;
use zinc_transcript::traits::Transcript;
use zinc_utils::cfg_iter;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::f2_hadamard::F2AdderSpec;

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

/// Number of binding families for `ℓ`-bit limbs of a `D`-bit word:
/// `nl` limb families plus `nl−1` boundary-bit families (the virtualized
/// carries at bits `ℓ, 2ℓ, …, ℓ(nl−1)`).
pub fn num_families<const D: usize>(limb_bits: usize) -> usize {
    let nl = D / limb_bits;
    2 * nl - 1
}

/// The bit position read by family `f` when `f ≥ nl` (a boundary-bit
/// family): `p = ℓ·(f − nl + 1)`.
#[inline]
fn boundary_bit(family: usize, nl: usize, limb_bits: usize) -> usize {
    limb_bits * (family - nl + 1)
}

/// The `a′`-consistency weight vector of family `f`: for a limb family,
/// `w[ℓf+b] = X^b`; for a boundary-bit family, `w[p] = 1` at its single bit.
/// `gf128poly_project(a′, w_f)` must equal `Σ_g γ_open,g·F_{f,g}(r_0)`.
pub fn family_weights<const D: usize>(family: usize, limb_bits: usize) -> Vec<Gf> {
    let nl = D / limb_bits;
    let mut w = vec![Gf::zero(); D];
    if family < nl {
        for (b, slot) in w.iter_mut().skip(family * limb_bits).take(limb_bits).enumerate() {
            *slot = x_pow(b);
        }
    } else {
        w[boundary_bit(family, nl, limb_bits)] = Gf::one();
    }
    w
}

/// The η-combined per-bit weight table `W[p] = Σ_f η^f·w_f[p]` — i.e.
/// `η^{⌊p/ℓ⌋}·X^{p mod ℓ}`, plus `η^{nl+j−1}` at the boundary bits `p = ℓj`
/// (`j ≥ 1`). One table serves both consumers of the η-batched binding:
/// the combined block column is the per-row read `Σ_{set bits p} W[p]`
/// ([`eta_block_rows`]), and the single `a′` consistency equation projects
/// `a′` against `W` directly (`Σ_f η^f·w_f(a′) = proj(a′, W)`).
#[allow(clippy::arithmetic_side_effects)]
pub fn eta_combined_weights<const D: usize>(eta: &Gf, limb_bits: usize) -> Vec<Gf> {
    let nl = D / limb_bits;
    let nf = num_families::<D>(limb_bits);
    let mut eta_pows = Vec::with_capacity(nf);
    let mut acc = Gf::one();
    for _ in 0..nf {
        eta_pows.push(acc);
        acc = acc * *eta;
    }
    let mut w = vec![Gf::zero(); D];
    for (p, slot) in w.iter_mut().enumerate() {
        let (f, b) = (p / limb_bits, p % limb_bits);
        *slot = eta_pows[f] * x_pow(b);
        if b == 0 && f >= 1 {
            *slot = *slot + eta_pows[nl + f - 1]; // boundary-bit family e_{ℓf}
        }
    }
    w
}

/// The η-combined family block of one column: row `r` is
/// `Σ_f η^f·F_f(col)[r] = Σ_{set bits p of the cell} W[p]` (zero past the
/// tail). The single fold source per witness column under the η-batched
/// binding.
#[allow(clippy::arithmetic_side_effects)]
pub fn eta_block_rows<const D: usize>(
    cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    col: usize,
    weights: &[Gf],
    n: usize,
) -> Vec<Gf> {
    (0..n)
        .map(|r| {
            let mut m = crate::f2_hadamard::cell_mask::<D>(&cols[col].evaluations[r]);
            let mut acc = Gf::zero();
            while m != 0 {
                let p = m.trailing_zeros() as usize;
                acc = acc + weights[p];
                m &= m - 1;
            }
            acc
        })
        .collect()
}

/// The eq tensor of `point` as `Gf` rows (`eq[r] = eq(point, r)`, indices
/// LSB-first — the same convention as [`eq_at_index`] and the MLE
/// evaluator).
pub fn eq_table_gf(point: &[Gf]) -> Vec<Gf> {
    let mle = zinc_poly::utils::build_eq_x_r_inner(point, &()).expect("eq table");
    mle.evaluations
        .iter()
        .map(|u| {
            let w = u.as_words();
            Gf::from_words([w[0], w[1]])
        })
        .collect()
}

/// Per-bit-position eq-weighted bucket sums of a row-shifted column:
/// `S_p = Σ_{r : bit p of col[r+Δ] set} eq[r]` — one add-only pass. By
/// linearity, EVERY family MLE eval of the `(col, Δ)` pair at the eq
/// tensor's point is a public weighting of the buckets:
/// `F̃_f(col↓Δ)(point) = Σ_r eq[r]·F_f(col↓Δ)[r] = Σ_p w_f[p]·S_p`.
#[allow(clippy::arithmetic_side_effects)]
fn bit_bucket_sums<const D: usize>(
    cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    col: usize,
    shift: usize,
    eq: &[Gf],
    n: usize,
) -> Vec<Gf> {
    let mut s = vec![Gf::zero(); D];
    for r in 0..n {
        let mut m = match r.checked_add(shift).filter(|&i| i < n) {
            Some(i) => crate::f2_hadamard::cell_mask::<D>(&cols[col].evaluations[i]),
            None => 0,
        };
        while m != 0 {
            let p = m.trailing_zeros() as usize;
            s[p] = s[p] + eq[r];
            m &= m - 1;
        }
    }
    s
}

/// All `nf` family evals from the bucket sums: limb family `f < nl` is
/// `Σ_b X^b·S_{ℓf+b}`, boundary family `f ≥ nl` is `S_{ℓ(f−nl+1)}`.
#[allow(clippy::arithmetic_side_effects)]
fn family_evals_from_buckets<const D: usize>(s: &[Gf], limb_bits: usize) -> Vec<Gf> {
    let nl = D / limb_bits;
    let nf = num_families::<D>(limb_bits);
    (0..nf)
        .map(|f| {
            if f < nl {
                let mut acc = Gf::zero();
                for b in 0..limb_bits {
                    acc = acc + x_pow(b) * s[f * limb_bits + b];
                }
                acc
            } else {
                s[boundary_bit(f, nl, limb_bits)]
            }
        })
        .collect()
}

/// All `nf` family MLE evals of a (possibly shifted) column at `point` —
/// the verifier's public-column recompute, one bucket pass instead of `nf`
/// separate MLE evaluations. The caller supplies the eq tensor of `point`
/// ([`eq_table_gf`]) so it is built once across columns.
pub fn family_proj_evals_bucketed<const D: usize>(
    cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    col: usize,
    shift: usize,
    limb_bits: usize,
    eq: &[Gf],
    n: usize,
) -> Vec<Gf> {
    let s = bit_bucket_sums::<D>(cols, col, shift, eq, n);
    family_evals_from_buckets::<D>(&s, limb_bits)
}

/// Per-relation public family-read coefficients:
/// `Q_a[r] = const_a + Σ coeff·F_f(col↓Δ)[r]`, keyed `((col, Δ), family)`.
/// Mirrors `fp = emb(x_i) + γ·emb(y_i⊕y2_i) + γ²·cin_i + γ³·emb(t_i) +
/// γ⁴·cout_i` with the carries VIRTUALIZED: `cin_i = cout_{i−1} = z[ℓi]`
/// (`z = x⊕y⊕y2⊕t`) is the boundary-bit parity `Σ_op e_{ℓi}(op↓Δ)`;
/// `cin_0 = 0`; the last limb is marginalised, contributing the public
/// `γ⁴·emb(2)` tag to `const_a` instead of a cout read.
#[derive(Clone, Debug)]
pub struct BindingCoeffs {
    /// `((col, row_shift), family) → coeff`, insertion-ordered.
    pub terms: Vec<(((usize, usize), usize), Gf)>,
    /// Public constant addend of `Q_a` (the marginal-tag term).
    pub constant: Gf,
}

impl BindingCoeffs {
    #[allow(clippy::arithmetic_side_effects)]
    fn add(&mut self, key: ((usize, usize), usize), c: Gf) {
        match self.terms.iter_mut().find(|(k, _)| *k == key) {
            Some((_, acc)) => *acc = *acc + c,
            None => self.terms.push((key, c)),
        }
    }
}

/// Build the per-relation coefficient maps.
#[allow(clippy::arithmetic_side_effects)]
pub fn lookup_binding_coeffs<const D: usize>(
    adder_specs: &[F2AdderSpec],
    r_pair: &[Gf],
    gamma: &Gf,
    limb_bits: usize,
) -> Vec<BindingCoeffs> {
    let nl = D / limb_bits;
    let g2 = *gamma * *gamma;
    let g3 = g2 * *gamma;
    let g4 = g3 * *gamma;
    // γ⁴·emb(2): the marginalised last limb's tag (matches
    // `add_lookup::marginal_tag` = the polynomial X).
    let tag = g4 * Gf::from_words([2, 0]);
    adder_specs
        .iter()
        .enumerate()
        .map(|(a, spec)| {
            let mut acc = BindingCoeffs { terms: Vec::new(), constant: Gf::zero() };
            // The z-read operand set of this relation: the carry parity
            // reads every term of z = x ⊕ y (⊕ y2) ⊕ t.
            let mut zread_ops: Vec<(usize, usize)> = vec![
                (spec.x.col, spec.x.row_shift),
                (spec.y.col, spec.y.row_shift),
                (spec.t.col, spec.t.row_shift),
            ];
            if let Some(y2) = &spec.y2 {
                zread_ops.push((y2.col, y2.row_shift));
            }
            for i in 0..nl {
                let eqp = eq_at_index(r_pair, a * nl + i);
                acc.add(((spec.x.col, spec.x.row_shift), i), eqp);
                acc.add(((spec.y.col, spec.y.row_shift), i), eqp * *gamma);
                if let Some(y2) = &spec.y2 {
                    acc.add(((y2.col, y2.row_shift), i), eqp * *gamma);
                }
                acc.add(((spec.t.col, spec.t.row_shift), i), eqp * g3);
                if i > 0 {
                    // cin_i = z[ℓi]: boundary family e_{ℓi} = nl + (i−1).
                    for &key in &zread_ops {
                        acc.add((key, nl + (i - 1)), eqp * g2);
                    }
                }
                if i + 1 < nl {
                    // cout_i = z[ℓ(i+1)]: boundary family e_{ℓ(i+1)} = nl + i
                    // — the SAME functional as cin_{i+1} (structural chain).
                    for &key in &zread_ops {
                        acc.add((key, nl + i), eqp * g4);
                    }
                } else {
                    // Last limb: marginalised — the public tag constant.
                    acc.constant = acc.constant + eqp * tag;
                }
            }
            acc
        })
        .collect()
}

/// The distinct `(col, Δ)` pairs referenced, in first-use order — the
/// limb-eval groups the prover ships and the fold binds.
pub fn binding_distinct_pairs(coeffs: &[BindingCoeffs]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for per_rel in coeffs {
        for ((key, _), _) in &per_rel.terms {
            if !pairs.contains(key) {
                pairs.push(*key);
            }
        }
    }
    pairs
}

/// Proof of the binding sumcheck: the sumcheck itself plus the family
/// shifted evals `F̃_f(col↓Δ)-MLE(r★)`, `limb_evals[pair][f]` per distinct
/// `(col, Δ)` pair (all `nf = 2·nl−1` families each: `nl` limb packs then
/// `nl−1` boundary bits).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupBindingProof {
    pub sumcheck: SumcheckProof<Gf>,
    pub limb_evals: Vec<Vec<Gf>>,
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
    #[error("binding limb-eval shape mismatch")]
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

/// MLE eval of the (0/1) active-row mask at the eq tensor's point:
/// `m̃(point) = Σ_{r active} eq[r]` — add-only (equal to the dense MLE
/// evaluation by linearity).
#[allow(clippy::arithmetic_side_effects)]
fn mask_eval_from_eq(spec: &F2AdderSpec, eq: &[Gf]) -> Gf {
    let mut acc = Gf::zero();
    for (r, e) in eq.iter().enumerate() {
        let active =
            spec.active_rows.is_empty() || spec.active_rows.get(r).copied().unwrap_or(false);
        if active {
            acc = acc + *e;
        }
    }
    acc
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
    let _ = num_vars;
    let nl = D / limb_bits;
    let one = Gf::one();
    let eq_row = eq_table_gf(r_row);
    let mut out = one;
    for (a, spec) in adder_specs.iter().enumerate() {
        let m_eval = mask_eval_from_eq(spec, &eq_row);
        let eq_sum: Gf = (0..nl)
            .map(|i| eq_at_index(r_pair, a * nl + i))
            .fold(Gf::zero(), |acc, e| acc + e);
        out = out + (one + *delta) * eq_sum * m_eval;
    }
    out
}


/// Prover: one degree-3 sumcheck for `B = Σ_r eq(r_row,r)·Σ_a mask_a[r]·Q_a[r]`,
/// then the family shifted evals at the sumcheck point `r★`.
/// Returns `(proof, r★)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_lookup_binding<const D: usize>(
    transcript: &mut impl Transcript,
    cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    adder_specs: &[F2AdderSpec],
    coeffs: &[BindingCoeffs],
    num_vars: usize,
    limb_bits: usize,
    r_row: &[Gf],
) -> (LookupBindingProof, Vec<Gf>) {
    let n = 1usize << num_vars;
    let nl = D / limb_bits;
    let nf = num_families::<D>(limb_bits);
    let zero_inner = Gf::zero().into_inner();
    let num_rel = adder_specs.len();

    // Q_a[r] = const_a + Σ coeff·F_f(col↓Δ)[r]. Precombine each relation's
    // coefficients into per-bit weights per (col, Δ) —
    // `W[p] = Σ_f coeff_f·w_f[p]` — so every row is pure adds over the
    // cell's set bits (the `eta_block_rows` trick with per-relation
    // weights); equal to the term-by-term form by linearity.
    let q_rows: Vec<Vec<Gf>> = cfg_iter!(coeffs)
        .map(|per_rel| {
            let mut weights: Vec<((usize, usize), Vec<Gf>)> = Vec::new();
            for (((col, shift), family), c) in &per_rel.terms {
                let key = (*col, *shift);
                let idx = match weights.iter().position(|(k, _)| *k == key) {
                    Some(i) => i,
                    None => {
                        weights.push((key, vec![Gf::zero(); D]));
                        weights.len() - 1
                    }
                };
                let w = &mut weights[idx].1;
                if *family < nl {
                    for (b, slot) in
                        w.iter_mut().skip(family * limb_bits).take(limb_bits).enumerate()
                    {
                        *slot = *slot + *c * x_pow(b);
                    }
                } else {
                    let p = boundary_bit(*family, nl, limb_bits);
                    w[p] = w[p] + *c;
                }
            }
            let mut q = vec![per_rel.constant; n];
            for ((col, shift), w) in &weights {
                for (r, qr) in q.iter_mut().enumerate() {
                    let mut m = match r.checked_add(*shift).filter(|&i| i < n) {
                        Some(i) => crate::f2_hadamard::cell_mask::<D>(&cols[*col].evaluations[i]),
                        None => 0,
                    };
                    while m != 0 {
                        let p = m.trailing_zeros() as usize;
                        *qr = *qr + w[p];
                        m &= m - 1;
                    }
                }
            }
            q
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

    // Family shifted evals at r★ per distinct (col, Δ) — all nf families
    // from one add-only bucket pass per pair (`Σ_r eq·F_f = Σ_p w_f[p]·S_p`).
    let pairs = binding_distinct_pairs(coeffs);
    let eq_star = eq_table_gf(&r_star);
    let limb_evals: Vec<Vec<Gf>> = cfg_iter!(pairs)
        .map(|&(col, shift)| {
            family_proj_evals_bucketed::<D>(cols, col, shift, limb_bits, &eq_star, n)
        })
        .collect();

    // Absorb the limb evals (they precede the challenges that bind them).
    let mut buf = vec![0u8; <Inner as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
    for group in &limb_evals {
        for v in group {
            transcript.absorb_random_field(v, &mut buf);
        }
    }

    (LookupBindingProof { sumcheck, limb_evals }, r_star)
}

/// Verifier: checks the sumcheck against `claimed_b = eval_w + PubMLE(r_w)`,
/// recombines `Q̃_a(r★)` from the shipped family evals with the public
/// coefficients, and checks the final evaluation. Returns `r★`; the caller
/// must bind each shipped `F̃_f(col↓Δ)(r★)` to the commitment (pointed-shift
/// claims on the family blocks + the fresh-`γ_open` `a′` consistency).
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_lookup_binding<const D: usize>(
    transcript: &mut impl Transcript,
    proof: &LookupBindingProof,
    claimed_b: &Gf,
    adder_specs: &[F2AdderSpec],
    coeffs: &[BindingCoeffs],
    num_vars: usize,
    limb_bits: usize,
    r_row: &[Gf],
) -> Result<Vec<Gf>, LookupBindingError> {
    if proof.sumcheck.claimed_sum != *claimed_b {
        return Err(LookupBindingError::ClaimedSumMismatch);
    }
    let subclaim =
        MLSumcheck::<Gf>::verify_as_subprotocol(transcript, num_vars, 3, &proof.sumcheck, &())?;
    let r_star = subclaim.point.clone();

    let nf = num_families::<D>(limb_bits);
    let pairs = binding_distinct_pairs(coeffs);
    if proof.limb_evals.len() != pairs.len() || proof.limb_evals.iter().any(|g| g.len() != nf) {
        return Err(LookupBindingError::Shape);
    }
    let mut buf = vec![0u8; <Inner as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
    for group in &proof.limb_evals {
        for v in group {
            transcript.absorb_random_field(v, &mut buf);
        }
    }

    let one = Gf::one();
    let eq_val = zinc_poly::utils::eq_eval(&r_star, &r_row.to_vec(), one)
        .map_err(|_| LookupBindingError::FinalEvalMismatch)?;

    // Σ_a mask̃_a(r★)·Q̃_a(r★), with Q̃ recombined from the family evals
    // (the public constant evaluates to itself — the all-ones MLE is 1
    // at any point) and the mask evals as add-only eq sums.
    let eq_star = eq_table_gf(&r_star);
    let mut s = Gf::zero();
    for (a, spec) in adder_specs.iter().enumerate() {
        let m_eval = mask_eval_from_eq(spec, &eq_star);
        let mut q = coeffs[a].constant;
        for ((key, family), c) in &coeffs[a].terms {
            let gi = pairs.iter().position(|p| p == key).expect("pair from coeffs");
            q = q + *c * proof.limb_evals[gi][*family];
        }
        s = s + m_eval * q;
    }
    if eq_val * s != subclaim.expected_evaluation {
        return Err(LookupBindingError::FinalEvalMismatch);
    }
    Ok(r_star)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fused η-table must equal the η-combination of the per-family
    /// reference weights — pins the boundary-family indexing inside
    /// [`eta_combined_weights`] (the table mixes limb `X^b` weights with
    /// boundary-bit `1`s at `p = ℓj`; an index slip there would silently
    /// unbind a family).
    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn eta_table_matches_per_family_weights() {
        const D: usize = 32;
        let eta = Gf::from_words([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
        for limb_bits in [4usize, 8] {
            let nf = num_families::<D>(limb_bits);
            let mut expected = vec![Gf::zero(); D];
            let mut pow = Gf::one();
            for f in 0..nf {
                for (slot, w) in expected.iter_mut().zip(family_weights::<D>(f, limb_bits)) {
                    *slot = *slot + pow * w;
                }
                pow = pow * eta;
            }
            assert_eq!(
                eta_combined_weights::<D>(&eta, limb_bits),
                expected,
                "ℓ = {limb_bits}"
            );
        }
    }
}
