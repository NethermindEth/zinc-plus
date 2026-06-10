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
//! Every fingerprint term is a **limb-family read** of a committed column:
//! `L_i(col)[r] = Σ_{b<ℓ} bit_{ℓi+b}(col[r])·X^b` (a FIXED weight family per
//! limb `i` — the carry columns are laid out with carry `j` at bit `ℓ·j`
//! precisely so the carry reads are limb reads too). So
//!
//! ```text
//!   Q_a[r] = Σ_{(col,Δ,i)} coeff_{a,(col,Δ,i)} · L_i(col↓Δ)[r]
//! ```
//!
//! with public coefficients `coeff = eq(r_pair,·)·γ^ρ`. This module proves
//! the masked row-sum `B := eval_w + PubMLE` with **one degree-3 sumcheck**
//! (`comb = eq·Σ_a mask_a·Q_a`), reducing it to the **limb-family shifted
//! evals** `L̃_i(col↓Δ)(r★)` at the sumcheck point — shipped per distinct
//! `(col,Δ)` (all `nl` limbs). The verifier recombines `Q̃_a(r★)` from them
//! with the public coefficients and checks the final evaluation.
//!
//! **Why limb families (and not per-bit evals):** each shipped eval must be
//! bound to the commitment. A fixed family `L_i` admits the z-block pattern —
//! pointed-shift claims into the multipoint fold against `L_i`-block columns,
//! whose `r_0`-evals are checked against the opened `a′` via
//! `Σ_g γ_g·L_{i,g}(r_0) = w_i(a′)` with the **fresh** post-absorb `γ_open`
//! (SZ binds each eval individually). Per-bit evals recombined under the
//! *already-known* α would NOT bind (a cheater shifts mass between slices
//! keeping the α-combination). The fold wiring lives in the prove pipeline;
//! this module's contract ends at the shipped `L̃` evals.

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

/// Per-relation public limb-read coefficients:
/// `Q_a[r] = Σ coeff·L_limb(col↓Δ)[r]`, keyed `((col, Δ), limb)`.
/// Mirrors `fp = emb(x_i) + γ·emb(y_i⊕y2_i) + γ²·cin_i + γ³·emb(t_i) + γ⁴·cout_i`
/// with the carry column laid out so `cout_j` sits at bit `ℓ·j` (limb-`j`
/// read): `cin_i = L_{i−1}(carry)` (`cin_0 = 0`), `cout_i = L_i(carry)`.
#[derive(Clone, Debug, Default)]
pub struct BindingCoeffs {
    /// `((col, row_shift), limb) → coeff`, insertion-ordered.
    pub terms: Vec<(((usize, usize), usize), Gf)>,
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
    carry_col_base: usize,
    r_pair: &[Gf],
    gamma: &Gf,
    limb_bits: usize,
) -> Vec<BindingCoeffs> {
    let nl = D / limb_bits;
    let g2 = *gamma * *gamma;
    let g3 = g2 * *gamma;
    let g4 = g3 * *gamma;
    adder_specs
        .iter()
        .enumerate()
        .map(|(a, spec)| {
            let mut acc = BindingCoeffs::default();
            for i in 0..nl {
                let eqp = eq_at_index(r_pair, a * nl + i);
                acc.add((((spec.x.col, spec.x.row_shift), i)), eqp);
                acc.add((((spec.y.col, spec.y.row_shift), i)), eqp * *gamma);
                if let Some(y2) = &spec.y2 {
                    acc.add(((y2.col, y2.row_shift), i), eqp * *gamma);
                }
                acc.add(((spec.t.col, spec.t.row_shift), i), eqp * g3);
                let carry = (carry_col_base + a, 0usize);
                if i > 0 {
                    acc.add((carry, i - 1), eqp * g2); // cin_i = L_{i−1}(carry)
                }
                acc.add((carry, i), eqp * g4); // cout_i = L_i(carry)
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

/// Proof of the binding sumcheck: the sumcheck itself plus the limb-family
/// shifted evals `L̃_i(col↓Δ)-MLE(r★)`, `limb_evals[pair][i]` per distinct
/// `(col, Δ)` pair (all `nl` limbs each).
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

/// The limb-family projected, row-shifted vector `L_i(col↓Δ)` as `Gf` rows
/// (zero past the tail, mirroring the operand-read convention).
#[allow(clippy::arithmetic_side_effects)]
fn limb_proj_rows<const D: usize>(
    cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    col: usize,
    shift: usize,
    limb: usize,
    limb_bits: usize,
    n: usize,
) -> Vec<Gf> {
    (0..n)
        .map(|r| {
            let m = match r.checked_add(shift).filter(|&i| i < n) {
                Some(i) => crate::f2_hadamard::cell_mask::<D>(&cols[col].evaluations[i]),
                None => 0,
            };
            let mut acc = Gf::zero();
            let mut bits = (m >> (limb * limb_bits)) & ((1u64 << limb_bits) - 1);
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                acc = acc + x_pow(b);
                bits &= bits - 1;
            }
            acc
        })
        .collect()
}

/// Prover: one degree-3 sumcheck for `B = Σ_r eq(r_row,r)·Σ_a mask_a[r]·Q_a[r]`,
/// then the limb-family shifted evals at the sumcheck point `r★`.
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
    let zero_inner = Gf::zero().into_inner();
    let num_rel = adder_specs.len();

    // Q_a[r] = Σ coeff·L_limb(col↓Δ)[r].
    let q_rows: Vec<Vec<Gf>> = cfg_iter!(coeffs)
        .map(|per_rel| {
            let mut q = vec![Gf::zero(); n];
            for (((col, shift), limb), c) in &per_rel.terms {
                let proj = limb_proj_rows::<D>(cols, *col, *shift, *limb, limb_bits, n);
                for (qr, p) in q.iter_mut().zip(proj) {
                    *qr = *qr + *c * p;
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

    // Limb-family shifted evals at r★ per distinct (col, Δ) — all nl limbs.
    let pairs = binding_distinct_pairs(coeffs);
    let limb_evals: Vec<Vec<Gf>> = cfg_iter!(pairs)
        .map(|&(col, shift)| {
            (0..nl)
                .map(|i| {
                    let v = limb_proj_rows::<D>(cols, col, shift, i, limb_bits, n);
                    row_mle_eval(&v, num_vars, &r_star)
                })
                .collect()
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
/// recombines `Q̃_a(r★)` from the shipped limb evals with the public
/// coefficients, and checks the final evaluation. Returns `r★`; the caller
/// must bind each shipped `L̃_i(col↓Δ)(r★)` to the commitment (pointed-shift
/// claims on the limb-family blocks + the fresh-`γ_open` `a′` consistency).
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

    let nl = D / limb_bits;
    let pairs = binding_distinct_pairs(coeffs);
    if proof.limb_evals.len() != pairs.len() || proof.limb_evals.iter().any(|g| g.len() != nl) {
        return Err(LookupBindingError::Shape);
    }
    let mut buf = vec![0u8; <Inner as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
    for group in &proof.limb_evals {
        for v in group {
            transcript.absorb_random_field(v, &mut buf);
        }
    }

    let n = 1usize << num_vars;
    let one = Gf::one();
    let eq_val = zinc_poly::utils::eq_eval(&r_star, &r_row.to_vec(), one)
        .map_err(|_| LookupBindingError::FinalEvalMismatch)?;

    // Σ_a mask̃_a(r★)·Q̃_a(r★), with Q̃ recombined from the limb evals.
    let mut s = Gf::zero();
    for (a, spec) in adder_specs.iter().enumerate() {
        let m_eval = row_mle_eval(&mask_vec(spec, n), num_vars, &r_star);
        let mut q = Gf::zero();
        for ((key, limb), c) in &coeffs[a].terms {
            let gi = pairs.iter().position(|p| p == key).expect("pair from coeffs");
            q = q + *c * proof.limb_evals[gi][*limb];
        }
        s = s + m_eval * q;
    }
    if eq_val * s != subclaim.expected_evaluation {
        return Err(LookupBindingError::FinalEvalMismatch);
    }
    Ok(r_star)
}
