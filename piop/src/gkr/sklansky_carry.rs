//! Sklansky parallel-prefix carry tree as a layered F_2 circuit, proven
//! via a stack of [multi-degree sumchecks](crate::sumcheck::multi_degree)
//! over GF(2^192).
//!
//! # Scope (Phase 1)
//!
//! Specialised to a **single 32-bit add**: input multilinear extensions
//! `a, b: {0,1}^5 → GF(2^192)` (where bit-position `i` is encoded as
//! Boolean point `i_4 i_3 i_2 i_1 i_0` in standard little-endian-bit
//! order), output the carry MLE `c`. No row dimension yet — that comes
//! in Phase 4 via "extra" leading variables that simply pass through
//! every sumcheck unchanged. Specialised to `BinaryFieldGF192`; the
//! trait abstraction will emerge in Phase 2 once integration shape is
//! clearer.
//!
//! # Protocol overview
//!
//! Let `(G_k, P_k): {0,1}^5 → GF(2^192)` be the per-bit generate /
//! propagate words after Sklansky stage `k`, for `k = 0,…,5`. Layer 0
//! is the input `(a·b, a⊕b)`, layer 5 is the final
//! `(carry-out, full-propagate)`. Stage `k` produces layer `k+1` from
//! layer `k` by the wire-update equations
//!
//! ```text
//!   G_{k+1}(x) = G_k(x) + T_k(x) · P_k(x) · G_k(σ_k(x))
//!   P_{k+1}(x) = P_k(x) · (1 + T_k(x) · (1 + P_k(σ_k(x))))
//! ```
//!
//! where `T_k(x) = x_k` (the indicator "x is a stage-k target" — every
//! position with the `k`-th bit set is a target in Sklansky) and
//! `σ_k(x)` zeroes bit `k` and sets bits `0..k-1` to 1, leaving the
//! higher bits alone. These identities hold *on the Boolean hypercube*;
//! they're used as the inner expression of one sumcheck per stage. The
//! identity is **not** a polynomial identity (the RHS has per-variable
//! degree up to 2 while the LHS is multilinear), but sumcheck only
//! requires the sum-over-the-Boolean-hypercube to match.
//!
//! # Two-point → one-point collapse
//!
//! Each stage's sumcheck reduces a claim `λ_G·G_{k+1}(r) + λ_P·P_{k+1}(r)`
//! at one point `r` to four claims at *two* points `x*, σ_k(x*)`:
//! `G_k(x*), P_k(x*), G_k(σ_k(x*)), P_k(σ_k(x*))`. To feed the next
//! stage we collapse to one point via:
//!
//! 1. Sample `γ` from the transcript; combine `H := G_k + γ·P_k`.
//! 2. Restrict `H` to the affine line `ℓ(t) = (1-t)·x* + t·σ_k(x*)`.
//!    `H(ℓ(t))` is a univariate poly in `t` of degree ≤ `k+1` (line
//!    varies in `k+1` coords). Prover sends 6 evaluations
//!    `H(ℓ(0)), …, H(ℓ(5))` — enough for any stage.
//! 3. Sample `t*`; new point `ℓ(t*)`, new claim `H(ℓ(t*))`. Next
//!    stage's `λ_G' = 1`, `λ_P' = γ`.
//!
//! # Bottom claim
//!
//! After 5 stages, the bottom claim is on `H_bot = G_0 + γ_bot · P_0`
//! at point `r_bot`. Since `G_0(y) = a(y)·b(y)` and `P_0(y) = a(y)+b(y)`
//! (over GF(2^192)), the verifier can check the bottom claim by
//! opening `a, b` at `r_bot`. In the Phase-2 integration that opening
//! is what the existing F_2 MLE-opening lane provides.

use crypto_primitives::{Field, FromWithConfig, PrimeField};
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::binary_gf192::BinaryFieldGF192,
    utils::build_eq_x_r_inner,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};

use crate::{
    CombFn,
    sumcheck::{
        SumCheckError,
        multi_degree::{
            MultiDegreeSumcheck, MultiDegreeSumcheckGroup, MultiDegreeSumcheckProof,
        },
    },
};

// ---------------------------------------------------------------------------
// Field-specialised aliases.
// ---------------------------------------------------------------------------

/// Width of the carry word in bits — and number of sumcheck variables
/// for a single 32-bit add.
pub const WIDTH_LOG: usize = 5;

/// Number of Sklansky stages, equal to [`WIDTH_LOG`].
pub const NUM_STAGES: usize = WIDTH_LOG;

/// Degree-5 max line-restriction polynomial uses 6 evaluations.
const LINE_EVAL_COUNT: usize = NUM_STAGES + 1;

/// Per-variable degree of the per-stage sumcheck (eq · degree-2 body).
const STAGE_DEGREE: usize = 3;

type F = BinaryFieldGF192;
type FInner = <F as Field>::Inner;
type FCfg = <F as PrimeField>::Config;

// ---------------------------------------------------------------------------
// Public types.
// ---------------------------------------------------------------------------

/// A two-MLE claim `λ_G · G(point) + λ_P · P(point) = value` at one
/// point of `GF(2^192)^WIDTH_LOG`.
#[derive(Clone, Debug)]
pub struct CombinedClaim {
    pub point: Vec<F>,
    pub lambda_g: F,
    pub lambda_p: F,
    pub value: F,
}

/// Per-stage transition proof: the inner sumcheck plus the prover-
/// supplied MLE evaluations needed by the verifier to recompute the
/// sumcheck's expected value, plus the line-restriction evaluations.
#[derive(Clone, Debug)]
pub struct StageProof {
    pub sumcheck: MultiDegreeSumcheckProof<F>,
    pub g_at_x: F,
    pub p_at_x: F,
    pub g_src_at_x: F,
    pub p_src_at_x: F,
    pub line_evals: [F; LINE_EVAL_COUNT],
}

/// Full GKR proof for the Sklansky carry tree.
#[derive(Clone, Debug)]
pub struct SklanskyCarryProof {
    /// Top-down ordering: index 0 proves stage 4 (top transition);
    /// index 4 proves stage 0 (bottom transition).
    pub stages: [StageProof; NUM_STAGES],
}

#[derive(Debug, thiserror::Error)]
pub enum SklanskyError {
    #[error("inner sumcheck failed: {0}")]
    Sumcheck(#[from] SumCheckError<F>),
    #[error("stage {stage}: claimed sumcheck sum {got:?} differs from outer claim {want:?}")]
    ClaimMismatch { stage: usize, got: F, want: F },
    #[error(
        "stage {stage}: prover's sumcheck-expected evaluation {got:?} \
         doesn't match recomputation from sent openings {want:?}"
    )]
    OpeningMismatch { stage: usize, got: F, want: F },
    #[error(
        "stage {stage}: line-restriction self-consistency violated \
         (line_evals[0]={got_zero:?} vs g+γp at x*={want_zero:?}, or \
         line_evals[1]={got_one:?} vs g_src+γp_src={want_one:?})"
    )]
    LineSelfCheck {
        stage: usize,
        got_zero: F,
        want_zero: F,
        got_one: F,
        want_one: F,
    },
}

// ---------------------------------------------------------------------------
// Wiring helpers.
// ---------------------------------------------------------------------------

/// Source index for Sklansky stage `k` at target Boolean position `tgt`
/// (assumes `tgt` has bit `k` set; for `tgt`s with bit `k` clear the
/// formula still produces a well-defined value, which is what the MLE
/// extension uses).
#[inline]
fn sigma_index(stage: usize, tgt: usize) -> usize {
    debug_assert!(stage < WIDTH_LOG);
    let half = 1usize << stage; // 2^k
    let full = half << 1; // 2^(k+1)
    (tgt & !(full - 1)) | (half - 1)
}

/// MLE of "x is a stage-`k` target" = "bit `k` of x is 1". Over the
/// 5-Boolean hypercube its evaluations are simply `(i >> k) & 1`,
/// and its multilinear extension to GF(2^192)^5 is `x_k` itself
/// (degree 1 in `x_k`, 0 in the other variables).
fn build_t_mle(stage: usize) -> DenseMultilinearExtension<FInner> {
    let zero = *F::zero().inner();
    let one = *F::one().inner();
    let mut evals = Vec::with_capacity(1 << WIDTH_LOG);
    for i in 0..(1usize << WIDTH_LOG) {
        evals.push(if (i >> stage) & 1 == 1 { one } else { zero });
    }
    DenseMultilinearExtension::from_evaluations_vec(WIDTH_LOG, evals, zero)
}

/// The "source" MLE: take `layer` (an MLE over 5 vars) and produce
/// a 5-var MLE whose Boolean-hypercube values are `layer[σ_k(i)]`.
/// On Boolean inputs, `src_mle[x] = layer[σ_k(x)]`; multilinearity
/// extends that to GF(2^192)^5.
fn build_src_mle(
    layer: &DenseMultilinearExtension<FInner>,
    stage: usize,
) -> DenseMultilinearExtension<FInner> {
    debug_assert_eq!(layer.num_vars, WIDTH_LOG);
    let zero = *F::zero().inner();
    let mut evals = Vec::with_capacity(1 << WIDTH_LOG);
    for i in 0..(1usize << WIDTH_LOG) {
        evals.push(layer.evaluations[sigma_index(stage, i)]);
    }
    DenseMultilinearExtension::from_evaluations_vec(WIDTH_LOG, evals, zero)
}

/// Compute `σ_k(x*)` as a point in `GF(2^192)^5`. `σ_k` differs from
/// `x*` only in coords `0..=k`: bit `k` → 0, bits `0..k-1` → 1, higher
/// bits unchanged.
fn sigma_point(stage: usize, x: &[F]) -> Vec<F> {
    debug_assert_eq!(x.len(), WIDTH_LOG);
    let mut out = x.to_vec();
    for i in 0..stage {
        out[i] = F::one();
    }
    out[stage] = F::zero();
    out
}

/// Affine line `ℓ(t) = (1-t)·x + t·σ`. Each coord is computed
/// independently: `ℓ(t)_i = x_i + t·(σ_i - x_i)` (subtraction in char-2
/// is addition).
fn line_at(x: &[F], sigma: &[F], t: F) -> Vec<F> {
    debug_assert_eq!(x.len(), sigma.len());
    x.iter()
        .zip(sigma.iter())
        .map(|(xi, si)| *xi + t * (*si + *xi))
        .collect()
}

/// Naive Lagrange interpolation: given `values[j] = f(j as field)` for
/// `j = 0..values.len()`, evaluate the unique interpolating polynomial
/// at `t`.
fn lagrange_eval(values: &[F], t: F, cfg: &FCfg) -> F {
    let n = values.len();
    let xs: Vec<F> = (0..n)
        .map(|j| F::from_with_cfg(j as u64, cfg))
        .collect();
    let mut acc = F::zero();
    for i in 0..n {
        let mut num = F::one();
        let mut den = F::one();
        for j in 0..n {
            if i != j {
                num = num * (t + xs[j]); // (t - xs[j]) in char 2
                den = den * (xs[i] + xs[j]);
            }
        }
        acc = acc + values[i] * num * den.inverse();
    }
    acc
}

/// Build the GF(2^192) MLE of the input bit-decomposition of a u32.
/// Bit `i` of `bits` lands at hypercube index `i` (LSB-first), value 0
/// or 1.
pub fn u32_to_mle(bits: u32) -> DenseMultilinearExtension<FInner> {
    let zero = *F::zero().inner();
    let one = *F::one().inner();
    let evals: Vec<FInner> = (0..32)
        .map(|i| if (bits >> i) & 1 == 1 { one } else { zero })
        .collect();
    DenseMultilinearExtension::from_evaluations_vec(WIDTH_LOG, evals, zero)
}

/// Convert an `Inner` value to an `F` (192-bit "raw" reinterpretation).
#[inline]
fn lift(inner: FInner, cfg: &FCfg) -> F {
    F::new_unchecked_with_cfg(inner, cfg)
}

// ---------------------------------------------------------------------------
// Stage comb_fn: shared by prover (closure) and verifier (recompute).
// ---------------------------------------------------------------------------

/// The per-stage sumcheck integrand, evaluated from MLE values
/// `[eq, g, p, t, g_src, p_src]` at a point.
///
/// On the Boolean hypercube this equals
/// `eq(r,x) · (λ_G · G_{k+1}(x) + λ_P · P_{k+1}(x))`.
#[inline]
fn stage_integrand(values: &[F], lambda_g: F, lambda_p: F) -> F {
    debug_assert_eq!(values.len(), 6);
    let eq = values[0];
    let g = values[1];
    let p = values[2];
    let t = values[3];
    let g_src = values[4];
    let p_src = values[5];

    // λ_G · G_{k+1} = λ_G · (G_k + T · P_k · G_src)
    let g_term = lambda_g * (g + t * p * g_src);
    // λ_P · P_{k+1} = λ_P · (P_k · (1 + T · (1 + P_src)))
    //              = λ_P · (P_k + T · P_k + T · P_k · P_src)  in char-2
    let p_term = lambda_p * (p + t * p + t * p * p_src);
    eq * (g_term + p_term)
}

fn make_stage_comb_fn(lambda_g: F, lambda_p: F) -> CombFn<F> {
    Box::new(move |v: &[F]| stage_integrand(v, lambda_g, lambda_p))
}

// ---------------------------------------------------------------------------
// Prover.
// ---------------------------------------------------------------------------

/// Prove the Sklansky-carry GKR end-to-end. Reduces the top claim
/// `λ_G · G_top(r) + λ_P · P_top(r) = top_value` to a bottom claim on
/// `(G_0, P_0)` at a derived point.
pub fn prove(
    transcript: &mut impl Transcript,
    g_layers: &[DenseMultilinearExtension<FInner>; NUM_STAGES + 1],
    p_layers: &[DenseMultilinearExtension<FInner>; NUM_STAGES + 1],
    top: &CombinedClaim,
    cfg: &FCfg,
) -> (SklanskyCarryProof, CombinedClaim) {
    assert_eq!(top.point.len(), WIDTH_LOG);

    let mut current = top.clone();
    let mut stage_proofs: Vec<StageProof> = Vec::with_capacity(NUM_STAGES);
    let mut buf = vec![0u8; FInner::NUM_BYTES];

    // Process stages top-down: stage k transition uses layer-k as "old"
    // and layer-(k+1) as "new". We start at the top (k = NUM_STAGES - 1).
    for stage in (0..NUM_STAGES).rev() {
        let g_old = g_layers[stage].clone();
        let p_old = p_layers[stage].clone();
        let g_src = build_src_mle(&g_old, stage);
        let p_src = build_src_mle(&p_old, stage);
        let t_mle = build_t_mle(stage);
        let eq_r = build_eq_x_r_inner::<F>(&current.point, cfg)
            .expect("eq table build for valid point");

        // Cache cloned MLEs so we can re-evaluate after the sumcheck
        // consumes the originals.
        let polys: Vec<DenseMultilinearExtension<FInner>> = vec![
            eq_r.clone(),
            g_old.clone(),
            p_old.clone(),
            t_mle.clone(),
            g_src.clone(),
            p_src.clone(),
        ];

        let group = MultiDegreeSumcheckGroup::new(
            STAGE_DEGREE,
            polys,
            make_stage_comb_fn(current.lambda_g, current.lambda_p),
        );

        let (proof, prover_states) =
            MultiDegreeSumcheck::<F>::prove_as_subprotocol(
                transcript,
                vec![group],
                WIDTH_LOG,
                cfg,
            );

        // The sumcheck auto-derives the claim — sanity-check it
        // matches the outer claim we promised.
        debug_assert_eq!(
            proof.claimed_sums()[0],
            current.value,
            "stage {stage} prover sumcheck claim ({:?}) != outer claim ({:?})",
            proof.claimed_sums()[0],
            current.value
        );

        let x_star: Vec<F> = prover_states[0].randomness.clone();

        // Re-evaluate MLEs at x_star (the sumcheck consumed the originals
        // inside the prover state, so use our clones).
        let g_at_x = g_old.clone().evaluate_with_config(&x_star, cfg)
            .expect("g_old eval at x*");
        let p_at_x = p_old.clone().evaluate_with_config(&x_star, cfg)
            .expect("p_old eval at x*");
        let g_src_at_x = g_src.clone().evaluate_with_config(&x_star, cfg)
            .expect("g_src eval at x*");
        let p_src_at_x = p_src.clone().evaluate_with_config(&x_star, cfg)
            .expect("p_src eval at x*");

        // Absorb the four opened MLE evaluations into the transcript
        // BEFORE squeezing γ for the line restriction. (eq(r, x*) and
        // T(x*) are independently recomputable by the verifier.)
        transcript.absorb_random_field(&g_at_x, &mut buf);
        transcript.absorb_random_field(&p_at_x, &mut buf);
        transcript.absorb_random_field(&g_src_at_x, &mut buf);
        transcript.absorb_random_field(&p_src_at_x, &mut buf);

        // Line-restriction γ.
        let gamma: F = transcript.get_field_challenge(cfg);

        // ℓ(t) = (1-t) x* + t σ(x*); H = G_old + γ · P_old.
        let sigma = sigma_point(stage, &x_star);
        let mut line_evals = [F::zero(); LINE_EVAL_COUNT];
        // ℓ(0) = x*, ℓ(1) = σ(x*) — exact known values.
        line_evals[0] = g_at_x + gamma * p_at_x;
        line_evals[1] = g_src_at_x + gamma * p_src_at_x;
        for j in 2..LINE_EVAL_COUNT {
            let t = F::from_with_cfg(j as u64, cfg);
            let l_t = line_at(&x_star, &sigma, t);
            let g_lt = g_old.clone().evaluate_with_config(&l_t, cfg)
                .expect("g_old eval at ℓ(t)");
            let p_lt = p_old.clone().evaluate_with_config(&l_t, cfg)
                .expect("p_old eval at ℓ(t)");
            line_evals[j] = g_lt + gamma * p_lt;
        }

        transcript.absorb_random_field_slice(&line_evals, &mut buf);
        let t_star: F = transcript.get_field_challenge(cfg);

        let r_next = line_at(&x_star, &sigma, t_star);
        let value_next = lagrange_eval(&line_evals, t_star, cfg);

        stage_proofs.push(StageProof {
            sumcheck: proof,
            g_at_x,
            p_at_x,
            g_src_at_x,
            p_src_at_x,
            line_evals,
        });

        current = CombinedClaim {
            point: r_next,
            lambda_g: F::one(),
            lambda_p: gamma,
            value: value_next,
        };
    }

    (
        SklanskyCarryProof {
            stages: stage_proofs.try_into().expect("NUM_STAGES proofs"),
        },
        current,
    )
}

// ---------------------------------------------------------------------------
// Verifier.
// ---------------------------------------------------------------------------

/// Verify the GKR proof; returns the derived bottom claim.
pub fn verify(
    transcript: &mut impl Transcript,
    top: &CombinedClaim,
    proof: &SklanskyCarryProof,
    cfg: &FCfg,
) -> Result<CombinedClaim, SklanskyError> {
    assert_eq!(top.point.len(), WIDTH_LOG);

    let mut current = top.clone();
    let mut buf = vec![0u8; FInner::NUM_BYTES];

    for (idx, stage_proof) in proof.stages.iter().enumerate() {
        let stage = NUM_STAGES - 1 - idx;

        // 1. Outer claim equals sumcheck's claimed sum.
        let claimed = stage_proof.sumcheck.claimed_sums()[0];
        if claimed != current.value {
            return Err(SklanskyError::ClaimMismatch {
                stage,
                got: claimed,
                want: current.value,
            });
        }

        // 2. Replay sumcheck.
        let subclaims = MultiDegreeSumcheck::<F>::verify_as_subprotocol(
            transcript,
            WIDTH_LOG,
            &stage_proof.sumcheck,
            cfg,
        )?;
        let x_star = subclaims.point().to_vec();
        let expected = subclaims.expected_evaluations()[0];

        // 3. Recompute the integrand at x* using prover-supplied openings
        //    plus locally-computed eq(r, x*) and T(x*) = x*_stage.
        let eq_at_x = zinc_poly::utils::eq_eval(&current.point, &x_star, F::one())
            .expect("eq_eval at x*");
        let t_at_x = x_star[stage];
        let recomputed = stage_integrand(
            &[
                eq_at_x,
                stage_proof.g_at_x,
                stage_proof.p_at_x,
                t_at_x,
                stage_proof.g_src_at_x,
                stage_proof.p_src_at_x,
            ],
            current.lambda_g,
            current.lambda_p,
        );
        if recomputed != expected {
            return Err(SklanskyError::OpeningMismatch {
                stage,
                got: expected,
                want: recomputed,
            });
        }

        // 4. Absorb openings + γ.
        transcript.absorb_random_field(&stage_proof.g_at_x, &mut buf);
        transcript.absorb_random_field(&stage_proof.p_at_x, &mut buf);
        transcript.absorb_random_field(&stage_proof.g_src_at_x, &mut buf);
        transcript.absorb_random_field(&stage_proof.p_src_at_x, &mut buf);
        let gamma: F = transcript.get_field_challenge(cfg);

        // 5. Line-restriction self-consistency: line_evals[0/1] must
        //    match what γ · the openings produce at ℓ(0)=x* and ℓ(1)=σ(x*).
        let want_zero = stage_proof.g_at_x + gamma * stage_proof.p_at_x;
        let want_one = stage_proof.g_src_at_x + gamma * stage_proof.p_src_at_x;
        if stage_proof.line_evals[0] != want_zero
            || stage_proof.line_evals[1] != want_one
        {
            return Err(SklanskyError::LineSelfCheck {
                stage,
                got_zero: stage_proof.line_evals[0],
                want_zero,
                got_one: stage_proof.line_evals[1],
                want_one,
            });
        }

        transcript.absorb_random_field_slice(&stage_proof.line_evals, &mut buf);
        let t_star: F = transcript.get_field_challenge(cfg);

        let sigma = sigma_point(stage, &x_star);
        let r_next = line_at(&x_star, &sigma, t_star);
        let value_next = lagrange_eval(&stage_proof.line_evals, t_star, cfg);

        current = CombinedClaim {
            point: r_next,
            lambda_g: F::one(),
            lambda_p: gamma,
            value: value_next,
        };
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use zinc_test_uair::sha256_f2_prefix::prefix_carry_trace;
    use zinc_transcript::Blake3Transcript;

    /// Lay out the Sklansky carry computation as 6 paired `(G, P)` layers
    /// over GF(2^192) for the test. Production code consumes layers
    /// supplied by the surrounding prover, not derived here.
    fn build_layers(
        a: u32,
        b: u32,
    ) -> ([DenseMultilinearExtension<FInner>; NUM_STAGES + 1],
          [DenseMultilinearExtension<FInner>; NUM_STAGES + 1])
    {
        let trace = prefix_carry_trace(a, b);
        let mut gs: Vec<DenseMultilinearExtension<FInner>> = Vec::with_capacity(NUM_STAGES + 1);
        let mut ps: Vec<DenseMultilinearExtension<FInner>> = Vec::with_capacity(NUM_STAGES + 1);
        for (g_word, p_word) in trace {
            gs.push(u32_to_mle(g_word));
            ps.push(u32_to_mle(p_word));
        }
        (
            gs.try_into().expect("layer count = NUM_STAGES + 1"),
            ps.try_into().expect("layer count = NUM_STAGES + 1"),
        )
    }

    /// Take a random `(a, b) ∈ u32 × u32`, build the Sklansky layers,
    /// pick a random top claim, prove it, verify it, and confirm the
    /// bottom claim matches what we'd compute directly from `a, b`'s
    /// MLEs.
    #[test]
    fn end_to_end_single_add() {
        let cfg = &();
        let mut rng = StdRng::seed_from_u64(0xA5A5_F00D_DEAD_BEEF);

        for trial in 0..32 {
            let a: u32 = rng.random();
            let b: u32 = rng.random();

            let (gs, ps) = build_layers(a, b);

            // Verifier-sampled top claim.
            let r: Vec<F> = (0..WIDTH_LOG)
                .map(|_| F::from_words([rng.random(), rng.random(), rng.random()]))
                .collect();
            let lambda_g = F::from_words([rng.random(), rng.random(), rng.random()]);
            let lambda_p = F::from_words([rng.random(), rng.random(), rng.random()]);

            let g_top_at_r = gs[NUM_STAGES]
                .clone()
                .evaluate_with_config(&r, cfg)
                .unwrap();
            let p_top_at_r = ps[NUM_STAGES]
                .clone()
                .evaluate_with_config(&r, cfg)
                .unwrap();
            let top_value = lambda_g * g_top_at_r + lambda_p * p_top_at_r;

            let top = CombinedClaim {
                point: r,
                lambda_g,
                lambda_p,
                value: top_value,
            };

            // Prove.
            let mut p_tx = Blake3Transcript::new();
            let (proof, bottom_p) = prove(&mut p_tx, &gs, &ps, &top, cfg);

            // Verify (fresh transcript — must reproduce exactly).
            let mut v_tx = Blake3Transcript::new();
            let bottom_v = verify(&mut v_tx, &top, &proof, cfg)
                .unwrap_or_else(|e| panic!("trial {trial}: verify failed: {e}"));

            // Prover and verifier must agree on the derived bottom claim.
            assert_eq!(
                bottom_p.point, bottom_v.point,
                "trial {trial}: bottom point divergence"
            );
            assert_eq!(
                bottom_p.value, bottom_v.value,
                "trial {trial}: bottom value divergence"
            );
            assert_eq!(bottom_p.lambda_g, bottom_v.lambda_g);
            assert_eq!(bottom_p.lambda_p, bottom_v.lambda_p);

            // Bottom claim sanity: `λ_G · G_0(r_bot) + λ_P · P_0(r_bot)`
            // computed directly from the input layer must match.
            let g0_at_rbot = gs[0]
                .clone()
                .evaluate_with_config(&bottom_v.point, cfg)
                .unwrap();
            let p0_at_rbot = ps[0]
                .clone()
                .evaluate_with_config(&bottom_v.point, cfg)
                .unwrap();
            let want = bottom_v.lambda_g * g0_at_rbot + bottom_v.lambda_p * p0_at_rbot;
            assert_eq!(
                bottom_v.value, want,
                "trial {trial}: bottom claim doesn't match direct evaluation of layer 0",
            );

            // Bridge to `a, b` directly:
            //   `P_0(y) = a(y) ⊕ b(y)` over the Boolean cube, and `⊕`
            //   is linear, so `P_0_MLE(r) = a_MLE(r) + b_MLE(r)` at
            //   *any* point.
            //   `G_0(y) = a(y) ∧ b(y)`, and AND is NOT linear — so
            //   `G_0_MLE(r) ≠ a_MLE(r) · b_MLE(r)` at non-Boolean `r`.
            //   The product expression equals `G_0_MLE(r)` only after
            //   one further sumcheck pass
            //   (`Σ_y eq(r, y) · a(y) · b(y) = G_0_MLE(r)`); that
            //   bridge is the extra layer Phase 2's integration adds
            //   on top of this GKR, *not* part of the GKR proper. We
            //   verify the cheap (P) half here as a smoke test on the
            //   bottom-layer correspondence.
            let a_mle = u32_to_mle(a);
            let b_mle = u32_to_mle(b);
            let a_at_rbot = a_mle.evaluate_with_config(&bottom_v.point, cfg).unwrap();
            let b_at_rbot = b_mle.evaluate_with_config(&bottom_v.point, cfg).unwrap();
            let p0_direct = a_at_rbot + b_at_rbot;
            assert_eq!(
                p0_at_rbot, p0_direct,
                "trial {trial}: P_0_MLE(r_bot) should equal a_MLE(r_bot) + b_MLE(r_bot)",
            );
        }
    }

    /// Independent check on the wiring helpers: `sigma_index(stage, tgt)`
    /// must agree with [`zinc_test_uair::sha256_f2_prefix::sklansky_edges`]
    /// for every target in every stage.
    #[test]
    fn sigma_index_matches_sklansky_schedule() {
        use zinc_test_uair::sha256_f2_prefix::sklansky_edges;
        for stage in 0..NUM_STAGES {
            for edge in sklansky_edges(stage) {
                assert_eq!(
                    sigma_index(stage, edge.tgt as usize),
                    edge.src as usize,
                    "stage {stage}, tgt {}",
                    edge.tgt
                );
            }
        }
    }

    /// The `build_src_mle` MLE at a Boolean position `i` should equal
    /// the source-position evaluation, i.e. `layer[σ_k(i)]`.
    #[test]
    fn build_src_mle_agrees_on_hypercube() {
        let cfg = &();
        let a: u32 = 0xDEAD_BEEF;
        let b: u32 = 0xCAFE_F00D;
        let (gs, _) = build_layers(a, b);

        for stage in 0..NUM_STAGES {
            let src_mle = build_src_mle(&gs[stage], stage);
            for i in 0..(1 << WIDTH_LOG) {
                let want = gs[stage].evaluations[sigma_index(stage, i)];
                let got = src_mle.evaluations[i];
                assert_eq!(got, want, "stage {stage}, i {i}");
            }
            // Also spot-check evaluation at a non-Boolean point yields
            // the expected MLE-extension behaviour: at i=0 (Boolean
            // origin) the source should be `layer[σ_k(0)]`. (Sanity.)
            let origin: Vec<F> = (0..WIDTH_LOG).map(|_| F::zero()).collect();
            let eval = src_mle.clone().evaluate_with_config(&origin, cfg).unwrap();
            let want = lift(gs[stage].evaluations[sigma_index(stage, 0)], cfg);
            assert_eq!(eval, want, "stage {stage}: src MLE at origin");
        }
    }
}
