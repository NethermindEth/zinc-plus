//! Shift predicate evaluation.
//!
//! Evaluates `S_c(x, y)` — the multilinear extension of the shift-by-c
//! indicator — at arbitrary field points.

use crypto_primitives::PrimeField;
use zinc_poly::utils::next_mle_eval;

/// Evaluate the shift predicate `S_c(x, y)` at arbitrary field points.
///
/// Uses the high/low decomposition:
///   `S_c(x, y) = L_0(x_lo, y_lo) · eq(x_hi, y_hi)
///              + L_1(x_lo, y_lo) · next_mle(x_hi, y_hi)`
///
/// where `k = ceil(log2(2c))` determines the split point.
///
/// Cost: O(m + c · log c) field operations.
#[allow(clippy::arithmetic_side_effects)]
pub fn eval_shift_predicate<F: PrimeField>(x: &[F], y: &[F], c: usize, cfg: &F::Config) -> F {
    let m = x.len();
    assert_eq!(y.len(), m);
    let zero = F::zero_with_cfg(cfg);
    let one = F::one_with_cfg(cfg);

    // S_0(x, y) = eq(x, y): identity shift.
    if c == 0 {
        return eval_eq_poly(x, y, &one);
    }

    // S_1(x, y) = next_mle(x, y): the successor predicate is exactly shift-by-1.
    if c == 1 {
        return next_mle_eval(x, y, zero, one);
    }

    assert!(c < (1usize << m), "shift c must satisfy c < 2^m");
    // k = ceil(log2(2*c))
    let k = (2 * c).next_power_of_two().trailing_zeros() as usize;
    if k >= m {
        return eval_shift_small(x, y, c, m, &zero, &one);
    }

    // LE convention: x[0..k] are the low bits, x[k..] are the high bits.
    let (x_lo, x_hi) = x.split_at(k);
    let (y_lo, y_hi) = y.split_at(k);

    let l0 = eval_l0(x_lo, y_lo, c, k, &zero, &one);
    let l1 = eval_l1(x_lo, y_lo, c, k, &zero, &one);
    let eq = eval_eq_poly(x_hi, y_hi, &one);
    let next = next_mle_eval(x_hi, y_hi, zero, one);
    l0 * eq + l1 * next
}

/// `eq(u, v) = prod_i (u_i * v_i + (1 - u_i)(1 - v_i))`
///
/// Evaluates the Multilinear polynomial for eq polynomial
pub(crate) fn eval_eq_poly<F: PrimeField>(u: &[F], v: &[F], one: &F) -> F {
    u.iter()
        .zip(v.iter())
        .map(|(u_i, v_i)| u_i.clone() * v_i + (one.clone() - u_i) * (one.clone() - v_i))
        .fold(one.clone(), |acc, term| acc * term)
}

/// `delta_{bin_k(a)}(u) = eq(u, bin_k(a))`.
///
/// Evaluates the Lagrange basis polynomial for the binary encoding of `a`
/// with `k` bits at the point `u`.
///
/// LE convention: `u[i]` corresponds to bit `i` (LSB = index 0).
pub(crate) fn eval_delta<F: PrimeField>(u: &[F], a: usize, k: usize, one: &F) -> F {
    let mut result = one.clone();
    for (i, u) in u.iter().take(k).enumerate() {
        let bit = (a >> i) & 1;
        if bit == 1 {
            result *= u;
        } else {
            result *= one.clone() - u
        }
    }
    result
}

/// `L_0^{(c)}(x_lo, y_lo)` — no-carry component.
///
/// `sum_{a=0}^{2^k - 1 - c} delta(x_lo, a) * delta(y_lo, a + c)`
///
/// On Booleans: 1 iff `Val(y_lo) = Val(x_lo) + c` with no carry into the high
/// block.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn eval_l0<F: PrimeField>(
    x_lo: &[F],
    y_lo: &[F],
    c: usize,
    k: usize,
    zero: &F,
    one: &F,
) -> F {
    let upper = (1 << k) - c;
    (0..upper).fold(zero.clone(), |acc, a| {
        acc + eval_delta(x_lo, a, k, one) * eval_delta(y_lo, a + c, k, one)
    })
}

/// `L_1^{(c)}(x_lo, y_lo)` — carry component.
///
/// `sum_{a=2^k-c}^{2^k-1} delta(x_lo, a) * delta(y_lo, a + c - 2^k)`
///
/// On Booleans: 1 iff the addition carries into the high block.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn eval_l1<F: PrimeField>(
    x_lo: &[F],
    y_lo: &[F],
    c: usize,
    k: usize,
    zero: &F,
    one: &F,
) -> F {
    let two_k = 1 << k;
    ((two_k - c)..two_k).fold(zero.clone(), |acc, a| {
        acc + eval_delta(x_lo, a, k, one) * eval_delta(y_lo, a + c - two_k, k, one)
    })
}

/// Special case when `k >= m`: no high block, direct evaluation.
///
/// `sum_{a=0}^{n-1-c} delta(x, a, m) * delta(y, a+c, m)`
#[allow(clippy::arithmetic_side_effects)]
fn eval_shift_small<F: PrimeField>(x: &[F], y: &[F], c: usize, m: usize, zero: &F, one: &F) -> F {
    let upper = (1 << m) - c;
    (0..upper).fold(zero.clone(), |acc, a| {
        acc + eval_delta(x, a, m, one) * eval_delta(y, a + c, m, one)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_config;
    use crypto_primitives::{Field, FromWithConfig, crypto_bigint_monty::MontyField};
    use rand::Rng;
    use zinc_poly::utils::{build_eq_x_r_inner, build_next_c_r_mle};

    type F = MontyField<4>;

    /// LE convention: to_bin(val, i) = bit i of val (LSB = index 0).
    fn to_bin(val: usize, bit: usize, cfg: &<F as PrimeField>::Config) -> F {
        if (val >> bit) & 1 == 1 {
            F::one_with_cfg(cfg)
        } else {
            F::zero_with_cfg(cfg)
        }
    }

    /// Convert F::Inner back to F.
    fn from_inner(inner: <F as Field>::Inner, cfg: &<F as PrimeField>::Config) -> F {
        let mut f = F::zero_with_cfg(cfg);
        *f.inner_mut() = inner;
        f
    }

    fn rand_field(rng: &mut impl Rng, cfg: &<F as PrimeField>::Config) -> F {
        F::from_with_cfg(rng.random::<u32>(), cfg)
    }

    /// Check S_c on Boolean inputs: S_c(bin(a), bin(a+c)) = 1,
    /// and S_c(bin(a), bin(b)) = 0 for b != a+c.
    #[test]
    fn test_shift_predicate_boolean() {
        let cfg = test_config();
        let m = 4;
        let n = 1usize << m;

        for c in [1, 2, 5] {
            for a in 0..n {
                for b in 0..n {
                    let x: Vec<F> = (0..m).map(|i| to_bin(a, i, &cfg)).collect();
                    let y: Vec<F> = (0..m).map(|i| to_bin(b, i, &cfg)).collect();
                    let val = eval_shift_predicate(&x, &y, c, &cfg);

                    if b == a + c && a + c < n {
                        assert_eq!(
                            val,
                            F::one_with_cfg(&cfg),
                            "S_c({a},{b}) should be 1 for c={c}"
                        );
                    } else {
                        assert_eq!(
                            val,
                            F::zero_with_cfg(&cfg),
                            "S_c({a},{b}) should be 0 for c={c}"
                        );
                    }
                }
            }
        }
    }

    /// Verify the next_mle on all Boolean inputs.
    #[test]
    fn test_next_boolean() {
        let cfg = test_config();
        let m = 4;
        let n = 1usize << m;
        for a in 0..n {
            for b in 0..n {
                let u: Vec<F> = (0..m).map(|i| to_bin(a, i, &cfg)).collect();
                let v: Vec<F> = (0..m).map(|i| to_bin(b, i, &cfg)).collect();
                let val = next_mle_eval(&u, &v, F::zero_with_cfg(&cfg), F::one_with_cfg(&cfg));

                if b == a + 1 && a + 1 < n {
                    assert_eq!(val, F::one_with_cfg(&cfg), "Next({a},{b}) should be 1");
                } else {
                    assert_eq!(val, F::zero_with_cfg(&cfg), "Next({a},{b}) should be 0");
                }
            }
        }
    }

    /// Check verifier (`eval_shift_predicate`) against prover
    /// (`build_next_c_r_mle`) at Boolean points:
    ///   eval_shift_predicate(r, bin(b), c) == build_next_c_r_mle(r, c)[b]
    #[test]
    fn test_shift_predicate_vs_prover_mle() {
        let cfg = test_config();
        let mut rng = rand::rng();
        let m = 4;
        let n = 1usize << m;
        let c = 3;

        let r: Vec<F> = (0..m).map(|_| rand_field(&mut rng, &cfg)).collect();
        let next_c = build_next_c_r_mle(&r, c, &cfg).unwrap();

        for b in 0..n {
            let b_bin: Vec<F> = (0..m).map(|i| to_bin(b, i, &cfg)).collect();
            let val = eval_shift_predicate(&r, &b_bin, c, &cfg);
            let expected = from_inner(next_c.evaluations[b], &cfg);
            assert_eq!(val, expected, "S_{c}(r, bin({b})) mismatch with prover MLE");
        }
    }

    /// Check at random field points via MLE summation:
    ///   eval_shift_predicate(r, y, c) == sum_b build_next_c_r_mle(r, c)[b] *
    /// eq(b, y)
    #[test]
    fn test_shift_predicate_random_points() {
        let cfg = test_config();
        let mut rng = rand::rng();
        let m = 4;
        let c = 3;

        for _ in 0..8 {
            let r: Vec<F> = (0..m).map(|_| rand_field(&mut rng, &cfg)).collect();
            let y: Vec<F> = (0..m).map(|_| rand_field(&mut rng, &cfg)).collect();

            let next_c = build_next_c_r_mle(&r, c, &cfg).unwrap();
            let eq_y = build_eq_x_r_inner(&y, &cfg).unwrap();
            let zero = F::zero_with_cfg(&cfg);
            let rhs = next_c
                .evaluations
                .iter()
                .zip(eq_y.evaluations.iter())
                .fold(zero, |acc, (ni, ei)| {
                    acc + from_inner(*ni, &cfg) * from_inner(*ei, &cfg)
                });
            let lhs = eval_shift_predicate(&r, &y, c, &cfg);

            assert_eq!(lhs, rhs, "random-point MLE mismatch");
        }
    }

    /// Test c=0 (identity) and c=1 (successor) fast paths at random points,
    /// and verify predicate vs prover MLE consistency across multiple shift
    /// amounts.
    #[test]
    fn test_fast_paths_and_multi_c() {
        let cfg = test_config();
        let mut rng = rand::rng();
        let m = 4;
        let n = 1usize << m;

        for c in [0, 1, 2, 5, 7] {
            let r: Vec<F> = (0..m).map(|_| rand_field(&mut rng, &cfg)).collect();
            let next_c = build_next_c_r_mle(&r, c, &cfg).unwrap();

            // Predicate vs prover MLE at Boolean y
            for b in 0..n {
                let b_bin: Vec<F> = (0..m).map(|i| to_bin(b, i, &cfg)).collect();
                let val = eval_shift_predicate(&r, &b_bin, c, &cfg);
                let expected = from_inner(next_c.evaluations[b], &cfg);
                assert_eq!(val, expected, "S_{c}(r, bin({b})) mismatch with prover MLE");
            }

            // Predicate vs prover MLE at random y (MLE consistency)
            for _ in 0..4 {
                let y: Vec<F> = (0..m).map(|_| rand_field(&mut rng, &cfg)).collect();
                let eq_y = build_eq_x_r_inner(&y, &cfg).unwrap();
                let zero = F::zero_with_cfg(&cfg);
                let rhs = next_c
                    .evaluations
                    .iter()
                    .zip(eq_y.evaluations.iter())
                    .fold(zero.clone(), |acc, (ni, ei)| {
                        acc + from_inner(*ni, &cfg) * from_inner(*ei, &cfg)
                    });
                let lhs = eval_shift_predicate(&r, &y, c, &cfg);
                assert_eq!(lhs, rhs, "random-point MLE mismatch for c={c}");
            }
        }
    }

    /// Boundary test: large c values where most rows shift beyond the domain.
    #[test]
    fn test_shift_predicate_boundary() {
        let cfg = test_config();
        let m = 3;
        let n = 1usize << m; // 8

        for c in [n / 2, n - 1] {
            // Boolean correctness: S_c(bin(a), bin(b)) = 1 iff b == a+c < n
            for a in 0..n {
                for b in 0..n {
                    let x: Vec<F> = (0..m).map(|i| to_bin(a, i, &cfg)).collect();
                    let y: Vec<F> = (0..m).map(|i| to_bin(b, i, &cfg)).collect();
                    let val = eval_shift_predicate(&x, &y, c, &cfg);

                    if b == a + c && a + c < n {
                        assert_eq!(
                            val,
                            F::one_with_cfg(&cfg),
                            "S_{c}(bin({a}), bin({b})) should be 1"
                        );
                    } else {
                        assert_eq!(
                            val,
                            F::zero_with_cfg(&cfg),
                            "S_{c}(bin({a}), bin({b})) should be 0"
                        );
                    }
                }
            }

            // Prover MLE: first c entries zero, rest match eq(r, b-c)
            let mut rng = rand::rng();
            let r: Vec<F> = (0..m).map(|_| rand_field(&mut rng, &cfg)).collect();
            let next_c = build_next_c_r_mle(&r, c, &cfg).unwrap();
            let zero_inner = *F::zero_with_cfg(&cfg).inner();

            // First c entries must be zero
            for b in 0..c {
                assert_eq!(
                    next_c.evaluations[b], zero_inner,
                    "next_c[{b}] should be zero for c={c}"
                );
            }
            // Remaining entries should be nonzero (with overwhelming probability)
            let nonzero_count = next_c.evaluations[c..]
                .iter()
                .filter(|e| **e != zero_inner)
                .count();
            assert_eq!(
                nonzero_count,
                n - c,
                "expected {} nonzero entries for c={c}",
                n - c
            );
        }
    }

    /// Check that build_next_c_r_mle correctly reproduces MLE[shift_c(v)](r)
    /// via inner product: sum_b next_c(b) * v[b] == sum_b eq(r, b-c) * v[b].
    #[test]
    fn test_prover_mle_inner_product() {
        let cfg = test_config();
        let mut rng = rand::rng();
        let m = 4;
        let n = 1usize << m;

        for c in [1, 2, 3, 7] {
            let v: Vec<F> = (0..n).map(|_| rand_field(&mut rng, &cfg)).collect();
            let r: Vec<F> = (0..m).map(|_| rand_field(&mut rng, &cfg)).collect();

            // Ground truth: sum_{b>=c} eq(r, b-c) * v[b]
            let eq_r = build_eq_x_r_inner(&r, &cfg).unwrap();
            let zero = F::zero_with_cfg(&cfg);
            let expected = (c..n).fold(zero.clone(), |acc, b| {
                acc + from_inner(eq_r.evaluations[b - c], &cfg) * &v[b]
            });

            // Via prover MLE: sum_b next_c[b] * v[b]
            let next_c = build_next_c_r_mle(&r, c, &cfg).unwrap();
            let got = next_c
                .evaluations
                .iter()
                .zip(v.iter())
                .fold(zero, |acc, (ni, vi)| {
                    acc + vi.clone() * from_inner(*ni, &cfg)
                });

            assert_eq!(got, expected, "prover MLE inner product mismatch for c={c}");
        }
    }
}
