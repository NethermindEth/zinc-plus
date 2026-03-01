//! Shift predicate evaluation.
//!
//! Provides functions for:
//! 1. Building the shift table `h[j] = eq*(r, bin(j + c))` for the prover.
//! 2. Evaluating `S_c(x, y)` at arbitrary field points for the verifier.

use crypto_primitives::PrimeField;
use zinc_poly::utils::build_eq_x_r_inner;
use zinc_poly::mle::DenseMultilinearExtension;

/// Build the shift table for the sumcheck prover.
///
/// Returns `h` of length `n = 2^m` where:
///   `h[j] = eq*(r, bin(j + c))`  if `j + c < n`,
///   `h[j] = 0`                   otherwise.
///
/// Identity: `MLE[shift_c(v)](r) = sum_{j=0}^{n-1} h[j] * v[j]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn build_shift_table<F: PrimeField>(
    r: &[F],
    c: usize,
    field_cfg: &F::Config,
) -> DenseMultilinearExtension<F::Inner>
where
    F::Inner: num_traits::Zero,
{
    let m = r.len();
    let n = 1usize << m;
    assert!(c > 0 && c < n, "shift c must satisfy 0 < c < 2^m");

    let eq_table = build_eq_x_r_inner(r, field_cfg)
        .expect("build_eq_x_r_inner should succeed");

    let zero = F::zero_with_cfg(field_cfg).inner().clone();
    let mut shift_evals = vec![zero; n];
    for j in 0..(n - c) {
        shift_evals[j] = eq_table.evaluations[j + c].clone();
    }

    DenseMultilinearExtension {
        num_vars: m,
        evaluations: shift_evals,
    }
}

/// Evaluate the shift predicate `S_c(x, y)` at arbitrary field points.
///
/// Uses the decomposition:
///
///   S_c(x, y) = L_0^{(c)}(x^lo, y^lo) · eq*(x^hi, y^hi)
///             + L_1^{(c)}(x^lo, y^lo) · Next*_{m'}(x^hi, y^hi)
///
/// where k = ceil(log2(2c)), m' = m − k.
///
/// Cost: O(m + c · log c) field operations.
#[allow(clippy::arithmetic_side_effects)]
pub fn eval_shift_predicate<F: PrimeField>(
    x: &[F],
    y: &[F],
    c: usize,
) -> F {
    let m = x.len();
    assert_eq!(y.len(), m);
    assert!(c > 0 && c < (1usize << m));

    let k = compute_k(c);

    if k >= m {
        return eval_shift_small(x, y, c, m);
    }

    let m_hi = m - k;
    let (x_hi, x_lo) = x.split_at(m_hi);
    let (y_hi, y_lo) = y.split_at(m_hi);

    let l0 = eval_l0(x_lo, y_lo, c, k);
    let l1 = eval_l1(x_lo, y_lo, c, k);
    let eq_hi = eval_eq_poly(x_hi, y_hi);
    let next_hi = eval_next(x_hi, y_hi);

    l0 * &eq_hi + &(l1 * &next_hi)
}

/// k = ceil(log2(2c)).
fn compute_k(c: usize) -> usize {
    assert!(c > 0);
    let two_c = 2 * c;
    if two_c <= 1 {
        return 0;
    }
    let mut k = 0;
    let mut val = 1usize;
    while val < two_c {
        k += 1;
        val <<= 1;
    }
    k
}

/// eq*(u, v) = prod_i (u_i * v_i + (1 − u_i)(1 − v_i)).
#[allow(clippy::arithmetic_side_effects)]
fn eval_eq_poly<F: PrimeField>(u: &[F], v: &[F]) -> F {
    assert!(!u.is_empty());
    let one = F::one_with_cfg(&u[0].cfg().clone());
    u.iter()
        .zip(v.iter())
        .map(|(ui, vi)| {
            ui.clone() * vi + &((one.clone() - ui) * &(one.clone() - vi))
        })
        .fold(one.clone(), |acc, next| acc * &next)
}

/// delta_{bin_k(a)}(u) = eq*(u, bin_k(a)).
#[allow(clippy::arithmetic_side_effects)]
fn eval_delta<F: PrimeField>(u: &[F], a: usize, k: usize) -> F {
    assert!(!u.is_empty());
    let cfg = u[0].cfg().clone();
    let one = F::one_with_cfg(&cfg);
    let zero = F::zero_with_cfg(&cfg);
    let mut result = one.clone();
    for i in 0..k {
        let bit = ((a >> (k - 1 - i)) & 1) as u64;
        let b = if bit == 1 { one.clone() } else { zero.clone() };
        result = result * &(u[i].clone() * &b + &((one.clone() - &u[i]) * &(one.clone() - &b)));
    }
    result
}

/// L_0^{(c)}(x^lo, y^lo): no-carry component.
#[allow(clippy::arithmetic_side_effects)]
fn eval_l0<F: PrimeField>(x_lo: &[F], y_lo: &[F], c: usize, k: usize) -> F {
    let cfg = x_lo[0].cfg().clone();
    let upper = (1usize << k) - c;
    let mut sum = F::zero_with_cfg(&cfg);
    for a in 0..upper {
        sum = sum + &(eval_delta(x_lo, a, k) * &eval_delta(y_lo, a + c, k));
    }
    sum
}

/// L_1^{(c)}(x^lo, y^lo): carry component.
#[allow(clippy::arithmetic_side_effects)]
fn eval_l1<F: PrimeField>(x_lo: &[F], y_lo: &[F], c: usize, k: usize) -> F {
    let cfg = x_lo[0].cfg().clone();
    let two_k = 1usize << k;
    let mut sum = F::zero_with_cfg(&cfg);
    for a in (two_k - c)..two_k {
        sum = sum + &(eval_delta(x_lo, a, k) * &eval_delta(y_lo, a + c - two_k, k));
    }
    sum
}

/// Next*_m(u, v): the MLE of the successor predicate.
///
/// Next*(u, v) = 1 iff Val(v) = Val(u) + 1 and Val(u) < 2^m − 1.
///
/// Evaluated in O(m) using prefix/suffix products.
#[allow(clippy::arithmetic_side_effects)]
fn eval_next<F: PrimeField>(u: &[F], v: &[F]) -> F {
    let m = u.len();
    if m == 0 {
        // shouldn't happen, but return zero
        return F::zero_with_cfg(&u[0].cfg().clone());
    }
    let cfg = u[0].cfg().clone();
    let one = F::one_with_cfg(&cfg);

    // prefix_eq[j] = prod_{i=0}^{j-1} eq*(u_i, v_i)
    let mut prefix_eq = vec![one.clone(); m + 1];
    for i in 0..m {
        let eq_i = u[i].clone() * &v[i] + &((one.clone() - &u[i]) * &(one.clone() - &v[i]));
        prefix_eq[i + 1] = prefix_eq[i].clone() * &eq_i;
    }

    // suffix_carry[j] = prod_{i=j}^{m-1} u_i · (1 − v_i)
    let mut suffix_carry = vec![one.clone(); m + 1];
    for i in (0..m).rev() {
        suffix_carry[i] = suffix_carry[i + 1].clone() * &(u[i].clone() * &(one.clone() - &v[i]));
    }

    let mut result = F::zero_with_cfg(&cfg);
    for j in 0..m {
        result = result + &(prefix_eq[j].clone() * &(one.clone() - &u[j]) * &v[j] * &suffix_carry[j + 1]);
    }

    result
}

/// Special case when k >= m: the entire vector is "low bits" (no high block).
#[allow(clippy::arithmetic_side_effects)]
fn eval_shift_small<F: PrimeField>(x: &[F], y: &[F], c: usize, m: usize) -> F {
    let cfg = x[0].cfg().clone();
    let n = 1usize << m;
    let mut sum = F::zero_with_cfg(&cfg);
    for a in 0..n.saturating_sub(c) {
        sum = sum + &(eval_delta(x, a, m) * &eval_delta(y, a + c, m));
    }
    sum
}

// ── Left-shift (look-ahead) variants ────────────────────────────────

/// Build the **left-shift** table for the sumcheck prover.
///
/// Returns `h` of length `n = 2^m` where:
///   `h[j] = eq*(r, bin(j − c))`  if `j ≥ c`,
///   `h[j] = 0`                   otherwise.
///
/// Identity: `MLE[left_c(v)](r) = sum_{j=0}^{n-1} h[j] * v[j]`,
/// where `left_c(v)[i] = v[i + c]` for `i < n − c`, 0 otherwise.
#[allow(clippy::arithmetic_side_effects)]
pub fn build_left_shift_table<F: PrimeField>(
    r: &[F],
    c: usize,
    field_cfg: &F::Config,
) -> DenseMultilinearExtension<F::Inner>
where
    F::Inner: num_traits::Zero,
{
    let m = r.len();
    let n = 1usize << m;

    // shift_amount = 0: the "shift" is the identity, so the table is
    // just eq*(r, ·).  This allows MLE evaluation claims  sum_b MLE[v](b)·eq(b,r)=v(r)
    // to participate in the same batched sumcheck as genuine shift claims.
    if c == 0 {
        return build_eq_x_r_inner(r, field_cfg)
            .expect("build_eq_x_r_inner should succeed");
    }

    assert!(c < n, "shift c must satisfy c < 2^m");

    let eq_table = build_eq_x_r_inner(r, field_cfg)
        .expect("build_eq_x_r_inner should succeed");

    let zero = F::zero_with_cfg(field_cfg).inner().clone();
    let mut shift_evals = vec![zero; n];
    for j in c..n {
        shift_evals[j] = eq_table.evaluations[j - c].clone();
    }

    DenseMultilinearExtension {
        num_vars: m,
        evaluations: shift_evals,
    }
}

/// Evaluate the **left-shift** predicate at arbitrary field points.
///
/// `eval_left_shift_predicate(x, y, c)` returns the MLE of the indicator
/// `{ (a, b) : val(a) = val(b) + c }` evaluated at `(x, y)`.
///
/// This equals `eval_shift_predicate(y, x, c)` — i.e. the right-shift
/// predicate with arguments swapped.
#[allow(clippy::arithmetic_side_effects)]
pub fn eval_left_shift_predicate<F: PrimeField>(
    x: &[F],
    y: &[F],
    c: usize,
) -> F {
    // shift_amount = 0: the predicate is simply eq*(x, y).
    if c == 0 {
        return eval_eq_poly(x, y);
    }
    // L_c(x, y) = S_c(y, x)
    eval_shift_predicate(y, x, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{LIMBS, test_config};
    use crypto_primitives::crypto_bigint_monty::MontyField;

    type F = MontyField<LIMBS>;

    #[test]
    fn shift_table_correctness() {
        use crypto_primitives::FromWithConfig;
        use rand::Rng;
        let cfg = test_config();
        let m = 4;
        let n = 1usize << m;
        let mut rng = rand::rng();

        for c in [1, 2, 3, 7] {
            // Random witness
            let v: Vec<F> = (0..n)
                .map(|_| F::from_with_cfg((rng.random::<u64>() % 1000) as i128, &cfg))
                .collect();
            let r: Vec<F> = (0..m)
                .map(|_| F::from_with_cfg((rng.random::<u64>() % 1000) as i128, &cfg))
                .collect();

            // Ground truth: right-shift vector and eval MLE
            let mut shifted = vec![F::zero_with_cfg(&cfg); n];
            for i in c..n {
                shifted[i] = v[i - c].clone();
            }

            let eq_table: DenseMultilinearExtension<<F as crypto_primitives::Field>::Inner> =
                build_eq_x_r_inner(&r, &cfg).unwrap();
            let expected: F = shifted
                .iter()
                .zip(eq_table.evaluations.iter())
                .map(|(vi, ei)| {
                    let vi_f = vi.clone();
                    let ei_f = F::new_unchecked_with_cfg(ei.clone(), &cfg);
                    vi_f * &ei_f
                })
                .fold(F::zero_with_cfg(&cfg), |a, b| a + &b);

            // Via right-shift table
            let h = build_shift_table(&r, c, &cfg);
            let got: F = h
                .evaluations
                .iter()
                .zip(v.iter())
                .map(|(hi, vi)| {
                    let hi_f = F::new_unchecked_with_cfg(hi.clone(), &cfg);
                    hi_f * vi
                })
                .fold(F::zero_with_cfg(&cfg), |a, b| a + &b);

            assert_eq!(got, expected, "right-shift table mismatch for c={c}");

            // Ground truth: left-shift vector and eval MLE
            let mut left_shifted = vec![F::zero_with_cfg(&cfg); n];
            for i in 0..n.saturating_sub(c) {
                left_shifted[i] = v[i + c].clone();
            }

            let expected_left: F = left_shifted
                .iter()
                .zip(eq_table.evaluations.iter())
                .map(|(vi, ei)| {
                    let vi_f = vi.clone();
                    let ei_f = F::new_unchecked_with_cfg(ei.clone(), &cfg);
                    vi_f * &ei_f
                })
                .fold(F::zero_with_cfg(&cfg), |a, b| a + &b);

            // Via left-shift table
            let h_left = build_left_shift_table(&r, c, &cfg);
            let got_left: F = h_left
                .evaluations
                .iter()
                .zip(v.iter())
                .map(|(hi, vi)| {
                    let hi_f = F::new_unchecked_with_cfg(hi.clone(), &cfg);
                    hi_f * vi
                })
                .fold(F::zero_with_cfg(&cfg), |a, b| a + &b);

            assert_eq!(got_left, expected_left, "left-shift table mismatch for c={c}");
        }
    }

    /// Verifies that eval_left_shift_predicate matches the table-based
    /// computation after sumcheck-style folding.
    #[test]
    fn eval_predicate_matches_table_folding() {
        use crypto_primitives::FromWithConfig;
        use rand::Rng;
        let cfg = test_config();
        let one = F::one_with_cfg(&cfg);
        let mut rng = rand::rng();

        for m in [3, 4, 5, 7] {
            let n = 1usize << m;
            for c in [1, 2, 3] {
                if c >= n { continue; }

                // Random eval_point r (LE convention, from build_eq_x_r_inner)
                let r: Vec<F> = (0..m)
                    .map(|_| F::from_with_cfg((rng.random::<u64>() % 1000) as i128, &cfg))
                    .collect();

                // Build left-shift table h[j] = eq*(r, bin(j-c)) for j >= c
                let h = build_left_shift_table(&r, c, &cfg);

                // Simulate sumcheck folding with random challenges
                let mut table: Vec<F> = h.evaluations.iter()
                    .map(|e| F::new_unchecked_with_cfg(e.clone(), &cfg))
                    .collect();

                let mut challenges = Vec::with_capacity(m);
                for _round in 0..m {
                    let half = table.len() / 2;
                    let s: F = F::from_with_cfg((rng.random::<u64>() % 1000) as i128, &cfg);
                    let one_s = one.clone() - &s;
                    let mut new_table = Vec::with_capacity(half);
                    for j in 0..half {
                        new_table.push(
                            table[j].clone() * &one_s + &(table[j + half].clone() * &s),
                        );
                    }
                    table = new_table;
                    challenges.push(s);
                }
                let h_final = table[0].clone();

                // Now try eval_left_shift_predicate with different argument orderings
                let r_rev: Vec<F> = r.iter().rev().cloned().collect();
                let ch_rev: Vec<F> = challenges.iter().rev().cloned().collect();

                let v1 = eval_left_shift_predicate(&challenges, &r, c);
                let v2 = eval_left_shift_predicate(&challenges, &r_rev, c);
                let v3 = eval_left_shift_predicate(&ch_rev, &r, c);
                let v4 = eval_left_shift_predicate(&ch_rev, &r_rev, c);

                // Find which one matches
                let matched = if h_final == v1 { "challenges, r" }
                    else if h_final == v2 { "challenges, r_rev" }
                    else if h_final == v3 { "ch_rev, r" }
                    else if h_final == v4 { "ch_rev, r_rev" }
                    else { "NONE" };

                assert_ne!(matched, "NONE",
                    "m={m}, c={c}: no argument ordering matched h_final.\n\
                     h_final={h_final:?}\nv1(ch,r)={v1:?}\nv2(ch,r_rev)={v2:?}\n\
                     v3(ch_rev,r)={v3:?}\nv4(ch_rev,r_rev)={v4:?}");

                // We expect a consistent pattern
                eprintln!("m={m}, c={c}: matched with ({matched})");
            }
        }
    }
}
