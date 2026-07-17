//! Shift predicate evaluation.
//!
//! Evaluates `S_c(x, y)` — the multilinear extension of the shift-by-c
//! indicator — at arbitrary field points.

use crypto_primitives::SemiringConfig;
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
pub fn eval_shift_predicate<C: SemiringConfig>(
    cfg: &C,
    x: &[C::Element],
    y: &[C::Element],
    c: usize,
) -> C::Element {
    let m = x.len();
    assert_eq!(y.len(), m);

    // S_0(x, y) = eq(x, y): identity shift.
    if c == 0 {
        return eval_eq_poly(cfg, x, y);
    }

    // S_1(x, y) = next_mle(x, y): the successor predicate is exactly shift-by-1.
    if c == 1 {
        return next_mle_eval(cfg, x, y);
    }

    assert!(c < (1usize << m), "shift c must satisfy c < 2^m");
    // k = ceil(log2(2*c))
    let k = (2 * c).next_power_of_two().trailing_zeros() as usize;
    if k >= m {
        return eval_shift_small(cfg, x, y, c, m);
    }

    // LE convention: x[0..k] are the low bits, x[k..] are the high bits.
    let (x_lo, x_hi) = x.split_at(k);
    let (y_lo, y_hi) = y.split_at(k);

    let l0 = eval_l0(cfg, x_lo, y_lo, c, k);
    let l1 = eval_l1(cfg, x_lo, y_lo, c, k);
    let eq = eval_eq_poly(cfg, x_hi, y_hi);
    let next = next_mle_eval(cfg, x_hi, y_hi);

    cfg.add(&cfg.mul(&l0, &eq), &cfg.mul(&l1, &next))
}

/// `eq(u, v) = prod_i (u_i * v_i + (1 - u_i)(1 - v_i))`
///
/// Evaluates the Multilinear polynomial for eq polynomial
pub(crate) fn eval_eq_poly<C: SemiringConfig>(
    cfg: &C,
    u: &[C::Element],
    v: &[C::Element],
) -> C::Element {
    let one = cfg.one();
    u.iter()
        .zip(v.iter())
        .map(|(u_i, v_i)| {
            cfg.add(
                &cfg.mul(u_i, v_i),
                &cfg.mul(&cfg.sub(&one, u_i), &cfg.sub(&one, v_i)),
            )
        })
        .fold(one.clone(), |acc, term| cfg.mul(&acc, &term))
}

/// `delta_{bin_k(a)}(u) = eq(u, bin_k(a))`.
///
/// Evaluates the Lagrange basis polynomial for the binary encoding of `a`
/// with `k` bits at the point `u`.
///
/// LE convention: `u[i]` corresponds to bit `i` (LSB = index 0).
pub(crate) fn eval_delta<C: SemiringConfig>(
    cfg: &C,
    u: &[C::Element],
    a: usize,
    k: usize,
) -> C::Element {
    let one = cfg.one();
    let mut result = one.clone();
    for (i, u) in u.iter().take(k).enumerate() {
        let bit = (a >> i) & 1;
        if bit == 1 {
            cfg.mul_assign(&mut result, u);
        } else {
            let term = cfg.sub(&one, u);
            cfg.mul_assign(&mut result, &term);
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
pub(crate) fn eval_l0<C: SemiringConfig>(
    cfg: &C,
    x_lo: &[C::Element],
    y_lo: &[C::Element],
    c: usize,
    k: usize,
) -> C::Element {
    let upper = (1 << k) - c;
    (0..upper).fold(cfg.zero(), |acc, a| {
        cfg.add(
            &acc,
            &cfg.mul(
                &eval_delta(cfg, x_lo, a, k),
                &eval_delta(cfg, y_lo, a + c, k),
            ),
        )
    })
}

/// `L_1^{(c)}(x_lo, y_lo)` — carry component.
///
/// `sum_{a=2^k-c}^{2^k-1} delta(x_lo, a) * delta(y_lo, a + c - 2^k)`
///
/// On Booleans: 1 iff the addition carries into the high block.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn eval_l1<C: SemiringConfig>(
    cfg: &C,
    x_lo: &[C::Element],
    y_lo: &[C::Element],
    c: usize,
    k: usize,
) -> C::Element {
    let two_k = 1 << k;
    ((two_k - c)..two_k).fold(cfg.zero(), |acc, a| {
        cfg.add(
            &acc,
            &cfg.mul(
                &eval_delta(cfg, x_lo, a, k),
                &eval_delta(cfg, y_lo, a + c - two_k, k),
            ),
        )
    })
}

/// Special case when `k >= m`: no high block, direct evaluation.
///
/// `sum_{a=0}^{n-1-c} delta(x, a, m) * delta(y, a+c, m)`
#[allow(clippy::arithmetic_side_effects)]
fn eval_shift_small<C: SemiringConfig>(
    cfg: &C,
    x: &[C::Element],
    y: &[C::Element],
    c: usize,
    m: usize,
) -> C::Element {
    let upper = (1 << m) - c;
    (0..upper).fold(cfg.zero(), |acc, a| {
        cfg.add(
            &acc,
            &cfg.mul(&eval_delta(cfg, x, a, m), &eval_delta(cfg, y, a + c, m)),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_config;
    use crypto_primitives::{
        ProjectElementWithConfig,
        crypto_bigint_monty::{MontyField, MontyFieldElement},
    };
    use rand::prelude::*;
    use zinc_poly::utils::{build_eq_x_r, build_next_c_r_mle};

    type F = MontyField<4>;
    type E = MontyFieldElement<4>;

    /// LE convention: to_bin(val, i) = bit i of val (LSB = index 0).
    fn to_bin(cfg: &F, val: usize, bit: usize) -> E {
        if (val >> bit) & 1 == 1 {
            cfg.one()
        } else {
            cfg.zero()
        }
    }

    fn rand_field(cfg: &F, rng: &mut impl Rng) -> E {
        cfg.project(&rng.random::<u32>())
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
                    let x: Vec<E> = (0..m).map(|i| to_bin(&cfg, a, i)).collect();
                    let y: Vec<E> = (0..m).map(|i| to_bin(&cfg, b, i)).collect();
                    let val = eval_shift_predicate(&cfg, &x, &y, c);

                    if b == a + c && a + c < n {
                        assert_eq!(val, cfg.one(), "S_c({a},{b}) should be 1 for c={c}");
                    } else {
                        assert_eq!(val, cfg.zero(), "S_c({a},{b}) should be 0 for c={c}");
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
                let u: Vec<E> = (0..m).map(|i| to_bin(&cfg, a, i)).collect();
                let v: Vec<E> = (0..m).map(|i| to_bin(&cfg, b, i)).collect();
                let val = next_mle_eval(&cfg, &u, &v);

                if b == a + 1 && a + 1 < n {
                    assert_eq!(val, cfg.one(), "Next({a},{b}) should be 1");
                } else {
                    assert_eq!(val, cfg.zero(), "Next({a},{b}) should be 0");
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

        let r: Vec<E> = (0..m).map(|_| rand_field(&cfg, &mut rng)).collect();
        let next_c = build_next_c_r_mle(&cfg, &r, c).unwrap();

        for b in 0..n {
            let b_bin: Vec<E> = (0..m).map(|i| to_bin(&cfg, b, i)).collect();
            let val = eval_shift_predicate(&cfg, &r, &b_bin, c);
            assert_eq!(
                val, next_c.evaluations[b],
                "S_{c}(r, bin({b})) mismatch with prover MLE"
            );
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
            let r: Vec<E> = (0..m).map(|_| rand_field(&cfg, &mut rng)).collect();
            let y: Vec<E> = (0..m).map(|_| rand_field(&cfg, &mut rng)).collect();

            let next_c = build_next_c_r_mle(&cfg, &r, c).unwrap();
            let eq_y = build_eq_x_r(&cfg, &y).unwrap();
            let rhs = next_c
                .evaluations
                .iter()
                .zip(eq_y.evaluations.iter())
                .fold(cfg.zero(), |acc, (ni, ei)| cfg.add(&acc, &cfg.mul(ni, ei)));
            let lhs = eval_shift_predicate(&cfg, &r, &y, c);

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
            let r: Vec<E> = (0..m).map(|_| rand_field(&cfg, &mut rng)).collect();
            let next_c = build_next_c_r_mle(&cfg, &r, c).unwrap();

            // Predicate vs prover MLE at Boolean y
            for b in 0..n {
                let b_bin: Vec<E> = (0..m).map(|i| to_bin(&cfg, b, i)).collect();
                let val = eval_shift_predicate(&cfg, &r, &b_bin, c);
                assert_eq!(
                    val, next_c.evaluations[b],
                    "S_{c}(r, bin({b})) mismatch with prover MLE"
                );
            }

            // Predicate vs prover MLE at random y (MLE consistency)
            for _ in 0..4 {
                let y: Vec<E> = (0..m).map(|_| rand_field(&cfg, &mut rng)).collect();
                let eq_y = build_eq_x_r(&cfg, &y).unwrap();
                let rhs = next_c
                    .evaluations
                    .iter()
                    .zip(eq_y.evaluations.iter())
                    .fold(cfg.zero(), |acc, (ni, ei)| cfg.add(&acc, &cfg.mul(ni, ei)));
                let lhs = eval_shift_predicate(&cfg, &r, &y, c);
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
                    let x: Vec<E> = (0..m).map(|i| to_bin(&cfg, a, i)).collect();
                    let y: Vec<E> = (0..m).map(|i| to_bin(&cfg, b, i)).collect();
                    let val = eval_shift_predicate(&cfg, &x, &y, c);

                    if b == a + c && a + c < n {
                        assert_eq!(val, cfg.one(), "S_{c}(bin({a}), bin({b})) should be 1");
                    } else {
                        assert_eq!(val, cfg.zero(), "S_{c}(bin({a}), bin({b})) should be 0");
                    }
                }
            }

            // Prover MLE: first c entries zero, rest match eq(r, b-c)
            let mut rng = rand::rng();
            let r: Vec<E> = (0..m).map(|_| rand_field(&cfg, &mut rng)).collect();
            let next_c = build_next_c_r_mle(&cfg, &r, c).unwrap();
            let zero = cfg.zero();

            // First c entries must be zero
            for b in 0..c {
                assert_eq!(
                    next_c.evaluations[b], zero,
                    "next_c[{b}] should be zero for c={c}"
                );
            }
            // Remaining entries should be nonzero (with overwhelming probability)
            let nonzero_count = next_c.evaluations[c..]
                .iter()
                .filter(|e| **e != zero)
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
            let v: Vec<E> = (0..n).map(|_| rand_field(&cfg, &mut rng)).collect();
            let r: Vec<E> = (0..m).map(|_| rand_field(&cfg, &mut rng)).collect();

            // Ground truth: sum_{b>=c} eq(r, b-c) * v[b]
            let eq_r = build_eq_x_r(&cfg, &r).unwrap();
            let expected = (c..n).fold(cfg.zero(), |acc, b| {
                cfg.add(&acc, &cfg.mul(&eq_r.evaluations[b - c], &v[b]))
            });

            // Via prover MLE: sum_b next_c[b] * v[b]
            let next_c = build_next_c_r_mle(&cfg, &r, c).unwrap();
            let got = next_c
                .evaluations
                .iter()
                .zip(v.iter())
                .fold(cfg.zero(), |acc, (ni, vi)| cfg.add(&acc, &cfg.mul(vi, ni)));

            assert_eq!(got, expected, "prover MLE inner product mismatch for c={c}");
        }
    }
}
