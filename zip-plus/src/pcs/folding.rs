//! Column-splitting and folding protocol for BinaryPoly columns.
//!
//! # Overview
//!
//! Given a column `v` with entries in `{0,1}^{<D}[X]` (i.e., `BinaryPoly<D>`),
//! each entry can be split as `v[i] = u[i] + X^{D/2} * w[i]`, where both
//! `u[i]` and `w[i]` live in `{0,1}^{<D/2}[X]` (i.e., `BinaryPoly<{D/2}>`).
//!
//! Instead of committing to the length-`n` column `v` with `BinaryPoly<D>`
//! entries, we commit to `v' = u || w`, a length-`2n` column with
//! `BinaryPoly<{D/2}>` entries. The first `n` entries of `v'` are the low
//! halves `u[0..n]`, and the last `n` entries are the high halves `w[0..n]`.
//!
//! Since the codeword elements are half the size, Zip+ column openings
//! become cheaper, reducing total proof size.
//!
//! # Folding protocol
//!
//! After the PIOP produces an evaluation claim `MLE[v](r) = c` in the field
//! (projected via the projecting element `α`), the folding protocol reduces
//! it to an evaluation claim on `v'`:
//!
//! 1. **Prover** provides `c₁ = MLE[v'](r ‖ 0)` and `c₂ = MLE[v'](r ‖ 1)`,
//!    which are `MLE[u_proj](r)` and `MLE[w_proj](r)` respectively, where
//!    `u_proj[i] = u[i](α)` and `w_proj[i] = w[i](α)`.
//!
//! 2. **Verifier** checks: `c₁ + α^{D/2} · c₂ = c mod q`.
//!
//! 3. **Verifier** sends random `β ∈ F_q` (via Fiat-Shamir).
//!
//! 4. Both compute `h(Y) = (1 − Y + β·Y) · ((1 − Y)·c₁ + Y·c₂)`,
//!    which is fully determined by `c₁`, `c₂`, `β`.
//!
//! 5. **Verifier** sends random `γ ∈ F_q` (via Fiat-Shamir).
//!
//! 6. **New claim**: `MLE[v'](r ‖ γ) = h(γ) / (1 − γ + β·γ)`.
//!    Since `h(γ)/(1−γ+βγ) = (1−γ)·c₁ + γ·c₂`, the new claimed evaluation
//!    is simply the linear interpolation of `c₁` and `c₂` at `γ`.
//!
//! 7. All such claims (one per committed column) are batched and proved
//!    with a single Zip+ invocation at the extended point `(r ‖ γ)`.

use crate::ZipError;
use crypto_primitives::PrimeField;
use crypto_primitives::semiring::boolean::Boolean;
use num_traits::Zero;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::mul_by_scalar::MulByScalar;
use zinc_utils::projectable_to_field::ProjectableToField;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Split a column of `BinaryPoly<D>` entries into a concatenated column
/// of `BinaryPoly<HALF_D>` entries.
///
/// Each entry `v[i]` with `D` binary coefficients is split into:
/// - `u[i]` = low `HALF_D` coefficients  (coefficients `0..HALF_D`)
/// - `w[i]` = high `HALF_D` coefficients (coefficients `HALF_D..D`)
///
/// so that `v[i] = u[i] + X^HALF_D · w[i]`.
///
/// Returns a column of length `2n` where:
/// - `v'[0..n]   = u[0..n]`  (low halves)
/// - `v'[n..2n]  = w[0..n]`  (high halves)
///
/// The returned MLE has `num_vars + 1` variables, with the last variable
/// selecting between the low half (0) and high half (1).
///
/// # Panics
/// Panics if `D != 2 * HALF_D`.
pub fn split_column<const D: usize, const HALF_D: usize>(
    column: &DenseMultilinearExtension<BinaryPoly<D>>,
) -> DenseMultilinearExtension<BinaryPoly<HALF_D>> {
    assert_eq!(
        D,
        2 * HALF_D,
        "split_column: D ({D}) must equal 2 * HALF_D ({HALF_D})"
    );

    let n = column.evaluations.len();
    let mut lo_evals = Vec::with_capacity(n);
    let mut hi_evals = Vec::with_capacity(n);

    for entry in &column.evaluations {
        let coeffs: Vec<Boolean> = entry.iter().map(|b| b.to_owned()).collect();
        let lo_arr: [Boolean; HALF_D] = std::array::from_fn(|i| coeffs[i]);
        let hi_arr: [Boolean; HALF_D] = std::array::from_fn(|i| coeffs[HALF_D + i]);
        let lo = BinaryPoly::<HALF_D>::new(lo_arr);
        let hi = BinaryPoly::<HALF_D>::new(hi_arr);
        lo_evals.push(lo);
        hi_evals.push(hi);
    }

    // Concatenate: v' = u || w
    lo_evals.extend(hi_evals);

    DenseMultilinearExtension::from_evaluations_vec(
        column.num_vars + 1,
        lo_evals,
        BinaryPoly::zero(),
    )
}

/// Split all columns of `BinaryPoly<D>` entries into `BinaryPoly<HALF_D>`.
///
/// See [`split_column`] for details.
#[allow(unreachable_code)]
pub fn split_columns<const D: usize, const HALF_D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
) -> Vec<DenseMultilinearExtension<BinaryPoly<HALF_D>>> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        return columns.par_iter().map(|col| split_column::<D, HALF_D>(col)).collect();
    }
    columns.iter().map(|col| split_column::<D, HALF_D>(col)).collect()
}

/// Split a column of `BinaryPoly<D>` entries directly to `BinaryPoly<1>`.
///
/// Instead of chaining `log₂(D)` intermediate split_column calls (D→D/2→…→1),
/// this function extracts all D individual bits of each entry in a single pass,
/// producing a column of length `D·n` with `BinaryPoly<1>` entries.
///
/// The layout is: for each successive halving round, the "low" and "high"
/// halves are concatenated. The final ordering matches what `log₂(D)` repeated
/// applications of [`split_column`] would produce:
///
/// For D=32 (5 splits): bit order in the final `32·n` array is determined by
/// the recursive halving — bit `b` of entry `i` ends up at index
/// `i + n · reverse_bit_order(b)`, where `reverse_bit_order` reverses the 5-bit
/// index represented by which half is chosen at each split round.
///
/// # Panics
/// Panics if `D` is not a power of two or if `FINAL != 1`.
pub fn split_column_to_final<const D: usize>(
    column: &DenseMultilinearExtension<BinaryPoly<D>>,
) -> DenseMultilinearExtension<BinaryPoly<1>> {
    assert!(D.is_power_of_two(), "D ({D}) must be a power of two");
    assert!(D >= 2, "D ({D}) must be at least 2");

    let n = column.evaluations.len();
    let log_d = D.trailing_zeros() as usize; // number of splits
    let total_len = n * D; // = n << log_d

    // We'll store results in the exact order that repeated split_column produces.
    // split_column takes [lo_bits || hi_bits] at each step, so the recursive
    // structure means: at the final level, bit `b` at entry `i` maps to
    // position `i + n * bit_reverse(b, log_d)`, where bit_reverse reverses
    // the log_d-bit representation of b.
    //
    // The bit-reversal comes from the fact that split_column puts lo (bit 0) in
    // the first half and hi (bit 1) in the second half at each level.

    let mut result = vec![BinaryPoly::<1>::zero(); total_len];

    let one = BinaryPoly::<1>::new([Boolean::new(true)]);

    for (i, entry) in column.evaluations.iter().enumerate() {
        for b in 0..D {
            if entry.iter().nth(b).map_or(false, |c| c.into_inner()) {
                // Compute the position in the final split array.
                // bit_reverse(b, log_d): reverse the log_d least-significant bits of b.
                let reversed = bit_reverse(b, log_d);
                result[i + n * reversed] = one.clone();
            }
        }
    }

    DenseMultilinearExtension::from_evaluations_vec(
        column.num_vars + log_d,
        result,
        BinaryPoly::zero(),
    )
}

/// Split all columns of `BinaryPoly<D>` entries directly to `BinaryPoly<1>`.
///
/// See [`split_column_to_final`] for details.
#[allow(unreachable_code)]
pub fn split_columns_to_final<const D: usize>(
    columns: &[DenseMultilinearExtension<BinaryPoly<D>>],
) -> Vec<DenseMultilinearExtension<BinaryPoly<1>>> {
    #[cfg(feature = "parallel")]
    {
        return columns.par_iter().map(|col| split_column_to_final::<D>(col)).collect();
    }
    columns.iter().map(|col| split_column_to_final::<D>(col)).collect()
}

/// Reverse the lowest `num_bits` of `val`.
///
/// E.g. `bit_reverse(0b10110, 5)` = `0b01101`.
#[inline]
fn bit_reverse(val: usize, num_bits: usize) -> usize {
    let mut result = 0;
    let mut v = val;
    for _ in 0..num_bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Prover-side: compute the folding evaluations `c₁` and `c₂` for each
/// split column.
///
/// For each split column `v' = u || w`:
/// - `c₁[j] = MLE[u_j](r)` projected to `F` via `α`
/// - `c₂[j] = MLE[w_j](r)` projected to `F` via `α`
///
/// where `u_j = v'_j[0..n]` (first half) and `w_j = v'_j[n..2n]` (second half).
///
/// # Arguments
/// - `split_polys`: the split `BinaryPoly<HALF_D>` columns (length `2n` each)
/// - `point`: the PIOP evaluation point `r` (length `num_vars`)
/// - `projecting_element`: the PIOP projecting element `α`
/// - `field_cfg`: field configuration for `F`
#[allow(unreachable_code)]
pub fn compute_folding_evals<F, const HALF_D: usize>(
    split_polys: &[DenseMultilinearExtension<BinaryPoly<HALF_D>>],
    point: &[F],
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<(Vec<F>, Vec<F>), ZipError>
where
    F: PrimeField + zinc_utils::from_ref::FromRef<F> + for<'a> MulByScalar<&'a F> + 'static,
    BinaryPoly<HALF_D>: ProjectableToField<F>,
{
    // ── Parallel path ────────────────────────────────────────────────────────
    // Each column's projections and MLE evaluations are independent, so we
    // compute them in parallel across columns when the `parallel` feature is on.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let pairs: Vec<Result<(F, F), ZipError>> = split_polys
            .par_iter()
            .map(|poly| {
                // Each thread gets its own projection closure (cheap to build).
                let project = BinaryPoly::<HALF_D>::prepare_projection(projecting_element);
                let zero_f = F::zero_with_cfg(field_cfg);

                let n = poly.evaluations.len() / 2;
                assert!(
                    n.is_power_of_two(),
                    "Each split column must have a power-of-two half-length, got {n}"
                );
                let half_num_vars = poly.num_vars - 1;
                assert_eq!(
                    half_num_vars,
                    point.len(),
                    "Point length ({}) must match half_num_vars ({half_num_vars})",
                    point.len(),
                );

                let u_proj: Vec<F> =
                    poly.evaluations[..n].iter().map(|bp| project(bp)).collect();
                let w_proj: Vec<F> =
                    poly.evaluations[n..].iter().map(|bp| project(bp)).collect();

                let u_mle = DenseMultilinearExtension::from_evaluations_vec(
                    half_num_vars,
                    u_proj,
                    zero_f.clone(),
                );
                let w_mle = DenseMultilinearExtension::from_evaluations_vec(
                    half_num_vars,
                    w_proj,
                    zero_f.clone(),
                );

                let c1 = u_mle
                    .evaluate(point, zero_f.clone())
                    .map_err(ZipError::PolynomialEvaluationError)?;
                let c2 = w_mle
                    .evaluate(point, zero_f.clone())
                    .map_err(ZipError::PolynomialEvaluationError)?;

                Ok((c1, c2))
            })
            .collect();

        let pairs: Vec<(F, F)> = pairs.into_iter().collect::<Result<_, _>>()?;
        let (c1s, c2s) = pairs.into_iter().unzip();
        return Ok((c1s, c2s));
    }

    // ── Sequential fallback ──────────────────────────────────────────────────
    let project = BinaryPoly::<HALF_D>::prepare_projection(projecting_element);
    let zero_f = F::zero_with_cfg(field_cfg);

    let mut c1s = Vec::with_capacity(split_polys.len());
    let mut c2s = Vec::with_capacity(split_polys.len());

    for poly in split_polys {
        let n = poly.evaluations.len() / 2;
        assert!(
            n.is_power_of_two(),
            "Each split column must have a power-of-two half-length, got {n}"
        );
        let half_num_vars = poly.num_vars - 1;
        assert_eq!(
            half_num_vars,
            point.len(),
            "Point length ({}) must match half_num_vars ({half_num_vars})",
            point.len(),
        );

        // Project the first half (u) and second half (w) to field elements.
        let u_proj: Vec<F> = poly.evaluations[..n]
            .iter()
            .map(|bp| project(bp))
            .collect();
        let w_proj: Vec<F> = poly.evaluations[n..]
            .iter()
            .map(|bp| project(bp))
            .collect();

        // Build MLEs of F elements and evaluate at point r.
        let u_mle = DenseMultilinearExtension::from_evaluations_vec(
            half_num_vars,
            u_proj,
            zero_f.clone(),
        );
        let w_mle = DenseMultilinearExtension::from_evaluations_vec(
            half_num_vars,
            w_proj,
            zero_f.clone(),
        );

        let c1 = u_mle
            .evaluate(point, zero_f.clone())
            .map_err(ZipError::PolynomialEvaluationError)?;
        let c2 = w_mle
            .evaluate(point, zero_f.clone())
            .map_err(ZipError::PolynomialEvaluationError)?;

        c1s.push(c1);
        c2s.push(c2);
    }

    Ok((c1s, c2s))
}

/// Prover-side: execute the full folding protocol.
///
/// 1. Computes `c₁[j]`, `c₂[j]` for each column
/// 2. Absorbs them into the transcript
/// 3. Squeezes `β` and `γ`
/// 4. Returns the new evaluation point `(r ‖ γ)` and the new claimed
///    evaluations `d[j] = (1−γ)·c₁[j] + γ·c₂[j]`.
///
/// The `c₁` and `c₂` values are also returned so they can be serialized
/// in the proof for the verifier to read.
///
/// # Arguments
/// - `transcript`: mutable reference to the shared PIOP Fiat-Shamir transcript
/// - `split_polys`: the split `BinaryPoly<HALF_D>` columns (length `2n` each)
/// - `point`: the PIOP evaluation point `r` (length `num_vars`)
/// - `projecting_element`: the PIOP projecting element `α`
/// - `field_cfg`: field configuration
///
/// # Returns
/// `(c1s, c2s, new_point, new_evals)` where:
/// - `c1s[j]`, `c2s[j]` are the folding evaluations per column
/// - `new_point = (r₀, …, r_{nv−1}, γ)` — the PCS evaluation point
/// - `new_evals[j] = (1−γ)·c1s[j] + γ·c2s[j]` — not used directly by PCS
///   (PCS computes its own aggregate eval), but useful for verification
#[allow(clippy::type_complexity)]
pub fn fold_claims_prove<F, T, const HALF_D: usize>(
    transcript: &mut T,
    split_polys: &[DenseMultilinearExtension<BinaryPoly<HALF_D>>],
    point: &[F],
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Result<FoldingProverOutput<F>, ZipError>
where
    F: PrimeField + zinc_utils::from_ref::FromRef<F> + for<'a> MulByScalar<&'a F> + 'static,
    F::Inner: ConstTranscribable,
    BinaryPoly<HALF_D>: ProjectableToField<F>,
    T: Transcript,
{
    // Step 1: compute c₁, c₂ for each column
    let (c1s, c2s) = compute_folding_evals::<F, HALF_D>(
        split_polys,
        point,
        projecting_element,
        field_cfg,
    )?;

    // Step 2: absorb c₁, c₂ into transcript
    let mut buf = vec![0u8; <F::Inner as ConstTranscribable>::NUM_BYTES];
    for c in &c1s {
        transcript.absorb_random_field(c, &mut buf);
    }
    for c in &c2s {
        transcript.absorb_random_field(c, &mut buf);
    }

    // Step 3: squeeze β
    let _beta: F = transcript.get_field_challenge(field_cfg);

    // Step 4: squeeze γ
    let gamma: F = transcript.get_field_challenge(field_cfg);

    // Step 5: compute new point (r ‖ γ)
    let mut new_point = point.to_vec();
    new_point.push(gamma.clone());

    // Step 6: compute new claimed evals: d[j] = (1 − γ)·c₁[j] + γ·c₂[j]
    let one = F::one_with_cfg(field_cfg);
    let one_minus_gamma = one - gamma.clone();
    let new_evals: Vec<F> = c1s
        .iter()
        .zip(&c2s)
        .map(|(c1, c2)| {
            let mut d = one_minus_gamma.clone();
            d *= c1.clone();
            let mut g_c2 = gamma.clone();
            g_c2 *= c2.clone();
            d += g_c2;
            d
        })
        .collect();

    Ok(FoldingProverOutput {
        c1s,
        c2s,
        new_point,
        new_evals,
    })
}

/// Output of the prover-side folding protocol.
pub struct FoldingProverOutput<F> {
    /// `c₁[j] = MLE[u_j](r)` for each committed column `j`.
    pub c1s: Vec<F>,
    /// `c₂[j] = MLE[w_j](r)` for each committed column `j`.
    pub c2s: Vec<F>,
    /// The extended PCS evaluation point `(r₀, …, r_{nv−1}, γ)`.
    pub new_point: Vec<F>,
    /// The folded per-column claims `d[j] = (1−γ)·c₁[j] + γ·c₂[j]`.
    pub new_evals: Vec<F>,
}

/// Verifier-side: execute the folding protocol.
///
/// Reads `c₁[j]`, `c₂[j]` from the proof data. Checks the consistency
/// relation `c₁[j] + α^{HALF_D} · c₂[j] = original_eval[j]` for every
/// committed column `j`. Then squeezes `β`, `γ` from the transcript and
/// returns the new evaluation point and new claimed evals.
///
/// # Arguments
/// - `transcript`: mutable reference to the shared PIOP Fiat-Shamir transcript
/// - `c1s`: prover-provided `c₁` values (one per committed column)
/// - `c2s`: prover-provided `c₂` values (one per committed column)
/// - `original_evals`: the PIOP's per-column projected evaluations at `r`
/// - `alpha_power`: `α^{HALF_D}` precomputed by the verifier
/// - `point`: the PIOP evaluation point `r` (length `num_vars`)
/// - `field_cfg`: field configuration
///
/// # Returns
/// `Ok((new_point, new_evals))` on success, or an error if the consistency
/// check fails.
#[allow(clippy::type_complexity)]
pub fn fold_claims_verify<F, T>(
    transcript: &mut T,
    c1s: &[F],
    c2s: &[F],
    original_evals: &[F],
    alpha_power: &F,
    point: &[F],
    field_cfg: &F::Config,
) -> Result<FoldingVerifierOutput<F>, ZipError>
where
    F: PrimeField + zinc_utils::from_ref::FromRef<F>,
    F::Inner: ConstTranscribable,
    T: Transcript,
{
    let num_cols = c1s.len();
    assert_eq!(
        num_cols,
        c2s.len(),
        "c1s and c2s must have the same length"
    );
    assert_eq!(
        num_cols,
        original_evals.len(),
        "c1s and original_evals must have the same length"
    );

    // Step 1: verify consistency: c₁[j] + α^{HALF_D} · c₂[j] = original_eval[j]
    for j in 0..num_cols {
        let mut expected = alpha_power.clone();
        expected *= c2s[j].clone();
        expected += c1s[j].clone();

        if expected != original_evals[j] {
            return Err(ZipError::InvalidPcsOpen(format!(
                "Folding consistency check failed for column {j}: \
                 c₁ + α^half_d · c₂ = {expected:?}, expected {eval:?}",
                eval = original_evals[j],
            )));
        }
    }

    // Step 2: absorb c₁, c₂ into transcript (same order as prover)
    let mut buf = vec![0u8; <F::Inner as ConstTranscribable>::NUM_BYTES];
    for c in c1s {
        transcript.absorb_random_field(c, &mut buf);
    }
    for c in c2s {
        transcript.absorb_random_field(c, &mut buf);
    }

    // Step 3: squeeze β (must match prover's transcript)
    let _beta: F = transcript.get_field_challenge(field_cfg);

    // Step 4: squeeze γ
    let gamma: F = transcript.get_field_challenge(field_cfg);

    // Step 5: compute new point (r ‖ γ)
    let mut new_point = point.to_vec();
    new_point.push(gamma.clone());

    // Step 6: compute new claimed evals: d[j] = (1 − γ)·c₁[j] + γ·c₂[j]
    let one = F::one_with_cfg(field_cfg);
    let one_minus_gamma = one - gamma.clone();
    let new_evals: Vec<F> = c1s
        .iter()
        .zip(c2s)
        .map(|(c1, c2)| {
            let mut d = one_minus_gamma.clone();
            d *= c1.clone();
            let mut g_c2 = gamma.clone();
            g_c2 *= c2.clone();
            d += g_c2;
            d
        })
        .collect();

    Ok(FoldingVerifierOutput {
        new_point,
        new_evals,
    })
}

/// Output of the verifier-side folding protocol.
pub struct FoldingVerifierOutput<F> {
    /// The extended PCS evaluation point `(r₀, …, r_{nv−1}, γ)`.
    pub new_point: Vec<F>,
    /// The folded per-column claims `d[j] = (1−γ)·c₁[j] + γ·c₂[j]`.
    pub new_evals: Vec<F>,
}

/// Compute `α^{HALF_D}` by repeated squaring.
///
/// Used by the verifier to check the folding consistency relation.
pub fn compute_alpha_power<F: PrimeField>(alpha: &F, half_d: usize) -> F {
    let mut result = alpha.clone();
    // Compute alpha^half_d via repeated squaring
    // half_d is a power of 2 in typical usage (e.g. 16), so this is efficient.
    let mut base = alpha.clone();
    result = F::one_with_cfg(result.cfg());
    let mut exp = half_d;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base.clone();
        }
        base *= base.clone();
        exp >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::semiring::boolean::Boolean;
    use num_traits::One;

    /// Helper: create a `BinaryPoly<D>` from a u32.
    fn bp32(val: u32) -> BinaryPoly<32> {
        BinaryPoly::from(val)
    }

    #[test]
    fn split_column_basic() {
        // Create a small column with 4 entries of BinaryPoly<32>
        let polys: Vec<BinaryPoly<32>> = vec![
            bp32(0x0001_0002), // lo = 0x0002 (bits 0..15), hi = 0x0001 (bits 16..31)
            bp32(0x0003_0004),
            bp32(0xFFFF_0000),
            bp32(0x0000_FFFF),
        ];

        let col = DenseMultilinearExtension::from_evaluations_vec(
            2,
            polys,
            BinaryPoly::zero(),
        );

        let split = split_column::<32, 16>(&col);

        assert_eq!(split.num_vars, 3);
        assert_eq!(split.evaluations.len(), 8);

        // First 4 entries are low halves (bits 0..15)
        // Last 4 entries are high halves (bits 16..31)
        // For 0x0001_0002: lo = 0x0002, hi = 0x0001
        let lo_0: Vec<Boolean> = split.evaluations[0].iter().collect();
        let hi_0: Vec<Boolean> = split.evaluations[4].iter().collect();

        // bit 1 of lo should be set (0x0002 = ...0010)
        assert!(lo_0[1].into_inner());
        assert!(!lo_0[0].into_inner());

        // bit 0 of hi should be set (0x0001 = ...0001)
        assert!(hi_0[0].into_inner());
        assert!(!hi_0[1].into_inner());
    }

    #[test]
    fn split_preserves_reconstruction() {
        // v[i] = u[i] + X^16 * w[i]
        // When we evaluate v[i] at X=2:
        //   v[i](2) = u[i](2) + 2^16 * w[i](2)
        let val: u32 = 0xABCD_1234;
        let poly = bp32(val);

        let col = DenseMultilinearExtension::from_evaluations_vec(
            0,
            vec![poly],
            BinaryPoly::zero(),
        );

        let split = split_column::<32, 16>(&col);
        assert_eq!(split.evaluations.len(), 2);

        let lo_val: u32 = val & 0xFFFF;
        let hi_val: u32 = val >> 16;

        // Evaluate lo and hi at X=2
        use zinc_poly::EvaluatablePolynomial;
        let lo_at_2: i64 = split.evaluations[0].evaluate_at_point(&2i64).unwrap();
        let hi_at_2: i64 = split.evaluations[1].evaluate_at_point(&2i64).unwrap();

        assert_eq!(lo_at_2 as u32, lo_val);
        assert_eq!(hi_at_2 as u32, hi_val);
        assert_eq!(
            lo_at_2 + (1i64 << 16) * hi_at_2,
            bp32(val).evaluate_at_point(&2i64).unwrap()
        );
    }

    #[test]
    fn split_columns_batch() {
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![bp32(0x0001_0002), bp32(0x0003_0004)],
            BinaryPoly::zero(),
        );
        let col2 = DenseMultilinearExtension::from_evaluations_vec(
            1,
            vec![bp32(0x0005_0006), bp32(0x0007_0008)],
            BinaryPoly::zero(),
        );

        let split = split_columns::<32, 16>(&[col1, col2]);

        assert_eq!(split.len(), 2);
        assert_eq!(split[0].num_vars, 2);
        assert_eq!(split[0].evaluations.len(), 4);
        assert_eq!(split[1].num_vars, 2);
        assert_eq!(split[1].evaluations.len(), 4);
    }
}
