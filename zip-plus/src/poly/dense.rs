use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_std::log2;
use crypto_bigint::Random;
use num_traits::Zero;
use rand_core::RngCore;
use crate::poly::mle::{MultilinearExtension, MultilinearExtensionRand};
use crypto_primitives::{Matrix, Ring};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearExtension<F> {
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: Vec<F>,
    /// Number of variables
    pub num_vars: usize,
}

impl<R: Ring> DenseMultilinearExtension<R> {
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &[R]) -> Self {
        Self::from_evaluations_vec(num_vars, evaluations.to_vec())
    }

    pub fn evaluate(&self, point: &[R]) -> Option<R> {
        if point.len() == self.num_vars {
            Some(self.fixed_variables(point)[0].clone())
        } else {
            None
        }
    }

    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<R>) -> Self {
        // assert that the number of variables matches the size of evaluations
        assert!(
            evaluations.len() <= 1 << num_vars,
            "The size of evaluations should not exceed 2^num_vars. \n eval len: {:?}. num vars: {num_vars}",
            evaluations.len()
        );

        if evaluations.len() != 1 << num_vars {
            let mut evaluations = evaluations;
            evaluations.resize(1 << num_vars, R::zero());
            return Self {
                num_vars,
                evaluations,
            };
        }

        Self {
            num_vars,
            evaluations,
        }
    }

    /// Returns the dense MLE from the given matrix, without modifying the
    /// original matrix.
    pub fn from_matrix<M: Matrix<R>>(matrix: &M) -> Self {
        let n_vars: usize = (log2(matrix.num_rows()) + log2(matrix.num_cols())) as usize; // n_vars = s + s'

        // Matrices might need to get padded before turned into an MLE
        let padded_rows = matrix.num_rows().next_power_of_two();
        let padded_cols = matrix.num_cols().next_power_of_two();

        // build dense vector representing the sparse padded matrix
        let mut v = vec![R::zero(); padded_rows * padded_cols];

        for (row_i, row) in matrix.rows().enumerate() {
            for (col_i, val) in row {
                v[(padded_cols * row_i) + col_i] = val.clone();
            }
        }

        // convert the dense vector into a mle
        Self::from_slice(n_vars, &v)
    }

    /// Takes n_vars and a dense slice and returns its dense MLE.
    pub fn from_slice(n_vars: usize, v: &[R]) -> Self {
        let v_padded: Vec<R> = if v.len() != (1 << n_vars) {
            // pad to 2^n_vars
            [
                v.to_owned(),
                ark_std::iter::repeat_n(R::zero(), (1 << n_vars) - v.len()).collect(),
            ]
            .concat()
        } else {
            v.to_owned()
        };
        DenseMultilinearExtension::from_evaluations_vec(n_vars, v_padded)
    }

    fn unary<G>(&mut self, f: G)
    where
        G: FnMut(&mut R),
    {
        self.evaluations.iter_mut().for_each(f);
    }

    fn binary<G>(&mut self, other: &Self, mut f: G)
    where
        G: FnMut(&mut R, &R),
    {
        self.evaluations
            .iter_mut()
            .zip(other.evaluations.iter())
            .for_each(|(a, b)| f(a, b));
    }
}

impl<R> MultilinearExtension<R> for DenseMultilinearExtension<R>
where
    R: Ring,
{
    fn fix_variables(&mut self, partial_point: &[R]) {
        assert!(
            partial_point.len() <= self.num_vars,
            "too many partial points"
        );

        let poly = &mut self.evaluations;
        let nv = self.num_vars;
        let dim = partial_point.len();

        for i in 1..dim + 1 {
            let r = partial_point[i - 1].clone();
            for b in 0..1 << (nv - i) {
                let left = poly[b << 1].clone();
                let right = poly[(b << 1) + 1].clone();
                let a = right - left.clone();
                if !a.is_zero() {
                    poly[b] = left + r.clone() * a;
                } else {
                    poly[b] = left;
                };
            }
        }

        self.evaluations.truncate(1 << (nv - dim));
        self.num_vars = nv - dim;
    }

    fn fixed_variables(&self, partial_point: &[R]) -> Self {
        let mut res = self.clone();
        res.fix_variables(partial_point);
        res
    }
}

impl<R> MultilinearExtensionRand<R> for DenseMultilinearExtension<R>
where
    R: Ring + Random
{
    fn rand<Rng: RngCore + ?Sized>(num_vars: usize, rng: &mut Rng) -> Self {
        Self::from_evaluations_vec(
            num_vars,
            (0..1 << num_vars).map(|_| R::random(rng)).collect(),
        )
    }
}

impl<T> Index<usize> for DenseMultilinearExtension<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<T> IndexMut<usize> for DenseMultilinearExtension<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.evaluations[index]
    }
}

impl<R: Ring> Add for DenseMultilinearExtension<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<R: Ring> Add<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a += b);
        self
    }
}

impl<R: Ring> Mul<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a *= b);
        self
    }
}

impl<R: Ring> Sub<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a -= b);
        self
    }
}

impl<R: Ring> Neg for DenseMultilinearExtension<R> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.unary(|v| *v = v.checked_neg().expect("Negation overflow"));
        self
    }
}

impl<R: Ring> Mul<R> for DenseMultilinearExtension<R> {
    type Output = Self;

    fn mul(mut self, rhs: R) -> Self::Output {
        self.unary(|v| *v *= &rhs);
        self
    }
}

impl<R: Ring> AddAssign<&Self> for DenseMultilinearExtension<R> {
    fn add_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a += b);
    }
}

impl<R: Ring> SubAssign<&Self> for DenseMultilinearExtension<R> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a -= b);
    }
}

impl<R: Ring> MulAssign<&Self> for DenseMultilinearExtension<R> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a *= b);
    }
}

impl<R: Ring> AddAssign<(R, &Self)> for DenseMultilinearExtension<R> {
    fn add_assign(&mut self, rhs: (R, &Self)) {
        let coeff = rhs.0;
        self.binary(rhs.1, |a, b| *a += b.clone() * &coeff);
    }
}

impl<R: Ring> Zero for DenseMultilinearExtension<R> {
    fn zero() -> Self {
        Self {
            num_vars: 0,
            evaluations: vec![R::zero()],
        }
    }

    fn is_zero(&self) -> bool {
        self.num_vars == 0 && self.evaluations[0].is_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::utils::{build_eq_x_r, build_eq_x_r_vec};
    use crate::{field::ConstMontyField, poly::mle::MultilinearExtension};
    use crypto_bigint::{const_monty_params, U128};
    use crypto_primitives::DenseRowMatrix;
    use num_traits::{ConstZero, One, Zero};
    use proptest::prelude::*;

    const_monty_params!(ModP, U128, "0076F668F4274572E39A3EA8285319B5");
    type F = ConstMontyField<ModP, { U128::LIMBS }>;

    fn any_f() -> impl Strategy<Value = F> {
        any::<u128>().prop_map(F::from)
    }

    fn any_dme() -> impl Strategy<Value = DenseMultilinearExtension<F>> {
        (0usize..=5).prop_flat_map(|n| {
            let len = 1usize << n;
            prop::collection::vec(any_f(), len)
                .prop_map(move |evals| DenseMultilinearExtension::from_evaluations_vec(n, evals))
        })
    }

    #[test]
    fn test_build_eq_x_r_vec_basic() {
        let r = [F::from(3u64)];
        let evals = build_eq_x_r_vec(&r).unwrap();
        assert_eq!(evals, vec![F::one() - r[0], r[0]]);
    }

    #[test]
    fn test_build_eq_x_r_vec_two_vars() {
        let r = [F::from(2u64), F::from(5u64)];
        let evals = build_eq_x_r_vec(&r).unwrap();
        let e00 = (F::one() - r[0]) * (F::one() - r[1]);
        let e01 = r[0] * (F::one() - r[1]);
        let e10 = (F::one() - r[0]) * r[1];
        let e11 = r[0] * r[1];
        assert_eq!(evals, vec![e00, e01, e10, e11]);
    }

    #[test]
    fn test_build_eq_x_r_error_on_empty() {
        let r: [F; 0] = [];
        let err = build_eq_x_r_vec(&r).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Invalid parameters"));
    }

    #[test]
    fn test_build_eq_x_r_mle_properties() {
        let r = [F::from(7u64), F::from(11u64), F::from(13u64)];
        let mle = build_eq_x_r(&r).unwrap();
        assert_eq!(mle.num_vars, r.len());
        let evals = mle.evaluations;
        let direct = build_eq_x_r_vec(&r).unwrap();
        assert_eq!(evals, direct);
    }

    #[test]
    fn test_dense_from_slice_and_indexing() {
        let n_vars = 3usize;
        let v = vec![F::from(1u64), F::from(2u64), F::from(3u64)];
        let dense = DenseMultilinearExtension::from_slice(n_vars, &v);
        assert_eq!(dense.num_vars, n_vars);
        let mut expected = v.clone();
        expected.resize(1 << n_vars, F::zero());
        assert_eq!(dense.evaluations, expected);
        assert_eq!(dense[0], F::from(1u64));
        let mut d2 = dense.clone();
        d2[1] = F::from(99u64);
        assert_eq!(d2[1], F::from(99u64));
    }

    #[test]
    fn test_fix_variables_and_evaluate() {
        let evals = vec![
            F::from(10u64),
            F::from(20u64),
            F::from(30u64),
            F::from(40u64),
        ];
        let mle = DenseMultilinearExtension::from_evaluations_vec(2, evals.clone());
        for (idx, &(x0, x1)) in [
            (F::zero(), F::zero()),
            (F::one(), F::zero()),
            (F::zero(), F::one()),
            (F::one(), F::one()),
        ]
        .iter()
        .enumerate()
        {
            let val = mle.evaluate(&[x0, x1]).unwrap();
            assert_eq!(val, evals[idx]);
        }
        let mut m2 = mle.clone();
        m2.fix_variables(&[F::one()]);
        assert_eq!(m2.num_vars, 1);
        assert_eq!(m2.evaluations, vec![F::from(20u64), F::from(40u64)]);
    }

    #[test]
    fn test_from_matrix_padding_and_conversion() {
        let m: DenseRowMatrix<F> = DenseRowMatrix::from(vec![
            vec![F::from(5u64), F::zero()],
            vec![F::zero(), F::zero()],
            vec![F::zero(), F::from(7u64)],
        ]);
        let dense = DenseMultilinearExtension::from_matrix(&m);
        assert_eq!(dense.num_vars, 3);
        let evals = dense.evaluations;
        assert_eq!(evals[0], F::from(5u64));
        assert_eq!(evals[5], F::from(7u64));
        assert!(evals.iter().enumerate().all(|(i, v)| if i == 0 || i == 5 {
            true
        } else {
            v.is_zero().into()
        }));
    }

    #[test]
    fn test_from_evaluations_vec_padding_branch_and_slice() {
        // len < 2^n triggers padding branch
        let evals = vec![F::from(1u64), F::from(2u64)];
        let n = 2usize; // 4 expected
        let d1 = DenseMultilinearExtension::from_evaluations_vec(n, evals.clone());
        let mut expected = evals.clone();
        expected.resize(1 << n, F::ZERO);
        assert_eq!(d1.evaluations, expected);
        let d2 = DenseMultilinearExtension::from_evaluations_slice(n, &evals);
        assert_eq!(d2.evaluations, expected);
    }

    #[test]
    fn test_fix_variables_edge_cases_and_full_truncate() {
        let d = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![F::from(1), F::from(2), F::from(3), F::from(4)],
        );
        let d_fixed = d.fixed_variables(&[]);
        assert_eq!(d_fixed.num_vars, 2);
        assert_eq!(
            d_fixed.evaluations,
            vec![F::from(1), F::from(2), F::from(3), F::from(4)]
        );
        let mut d2 = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![F::from(10), F::from(20), F::from(30), F::from(40)],
        );
        d2.fix_variables(&[F::one(), F::zero()]);
        assert_eq!(d2.num_vars, 0);
        assert_eq!(d2.evaluations, vec![F::from(20)]);
    }

    #[test]
    fn test_evaluate_length_mismatch_returns_none() {
        let d = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![F::from(1), F::from(2), F::from(3), F::from(4)],
        );
        assert!(d.evaluate(&[F::one()]).is_none());
        assert!(d.evaluate(&[F::one(), F::one(), F::zero()]).is_none());
    }

    #[test]
    fn test_zero_impl_for_dense_mle() {
        let z: DenseMultilinearExtension<F> = Zero::zero();
        assert_eq!(z.num_vars, 0);
        assert_eq!(z.evaluations, vec![F::ZERO]);
        assert!(z.is_zero());
    }

    #[test]
    fn test_arithmetic_ops_elementwise_add_sub_mul_and_neg() {
        let a = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![F::from(1), F::from(2), F::from(3), F::from(4)],
        );
        let b = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![F::from(5), F::from(6), F::from(7), F::from(8)],
        );

        let sum = a.clone() + &b;
        assert_eq!(
            sum.evaluations,
            vec![F::from(6), F::from(8), F::from(10), F::from(12)]
        );

        let diff = b.clone() - &a;
        assert_eq!(
            diff.evaluations,
            vec![F::from(4), F::from(4), F::from(4), F::from(4)]
        );

        let prod = a.clone() * &b;
        assert_eq!(
            prod.evaluations,
            vec![F::from(5), F::from(12), F::from(21), F::from(32)]
        );

        // Neg
        let neg_a = -a.clone();
        let mut expected = vec![];
        for v in a.evaluations {
            expected.push(-v);
        }
        assert_eq!(neg_a.evaluations, expected);
    }

    #[test]
    fn test_scalar_mul_and_assign_variants() {
        let a = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![F::from(1), F::from(2), F::from(3), F::from(4)],
        );
        let b = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![F::from(10), F::from(20), F::from(30), F::from(40)],
        );

        let three = F::from(3u64);
        let scaled = a.clone() * three;
        assert_eq!(
            scaled.evaluations,
            vec![F::from(3), F::from(6), F::from(9), F::from(12)]
        );

        let mut c = a.clone();
        c += &b;
        assert_eq!(
            c.evaluations,
            vec![F::from(11), F::from(22), F::from(33), F::from(44)]
        );

        c -= &b;
        assert_eq!(c.evaluations, a.evaluations);

        let mut d = a.clone();
        d *= &b;
        assert_eq!(
            d.evaluations,
            vec![F::from(10), F::from(40), F::from(90), F::from(160)]
        );

        let mut e = a.clone();
        let two = F::from(2u64);
        e += (two, &b);
        assert_eq!(
            e.evaluations,
            vec![F::from(21), F::from(42), F::from(63), F::from(84)]
        );
    }

    fn any_aligned_pair_with_point() -> impl Strategy<
        Value = (
            DenseMultilinearExtension<F>,
            DenseMultilinearExtension<F>,
            Vec<F>,
        ),
    > {
        (0usize..=5).prop_flat_map(|n| {
            let len = 1usize << n;
            prop::collection::vec(any_f(), len).prop_flat_map(move |e1| {
                let n2 = n;
                prop::collection::vec(any_f(), len).prop_flat_map(move |e2| {
                    let n3 = n2;
                    point_n(n3).prop_map({
                        let e1v = e1.clone();
                        let e2v = e2.clone();
                        move |r| {
                            (
                                DenseMultilinearExtension::from_evaluations_vec(n3, e1v.clone()),
                                DenseMultilinearExtension::from_evaluations_vec(n3, e2v.clone()),
                                r,
                            )
                        }
                    })
                })
            })
        })
    }
    fn point_n(n: usize) -> impl Strategy<Value = Vec<F>> {
        prop::collection::vec(any_f(), n)
    }

    proptest! {
        #[test]
        fn prop_eval_add_is_linear((p1, p2, r) in any_aligned_pair_with_point()) {
            let lhs = (p1.clone() + &p2).evaluate(&r).unwrap();
            let rhs = p1.evaluate(&r).unwrap() + p2.evaluate(&r).unwrap();
            prop_assert_eq!(lhs, rhs);
        }

        #[test]
        fn prop_fix_vars_commutes_with_eval((p, r, k) in any_dme().prop_flat_map(|p| {
            let n = p.num_vars;
            let point = point_n(n);
            let ks = 0usize..=n;
            (Just(p), point, ks)
        })) {
            let mut pfixed = p.clone();
            pfixed.fix_variables(&r[..k]);
            let lhs = pfixed.evaluate(&r[k..]).unwrap();
            let rhs = p.evaluate(&r).unwrap();
            prop_assert_eq!(lhs, rhs);
        }

        #[test]
        fn prop_fix_vars_is_idempotent((p, k1, k2) in any_dme().prop_flat_map(|p| {
            let n = p.num_vars;
            let ks1 = 0usize..=n;
            (Just(p), ks1).prop_flat_map(move |(p, k1)| {
                let ks2 = 0usize..=n.saturating_sub(k1);
                (Just(p), Just(k1), ks2)
            })
        }), r1 in prop::collection::vec(any_f(), 0..=8usize), r2 in prop::collection::vec(any_f(), 0..=8usize)) {
            let mut p_step = p.clone();
            p_step.fix_variables(&r1[..k1.min(r1.len())]);
            p_step.fix_variables(&r2[..k2.min(r2.len())]);

            let mut p_once = p.clone();
            let mut concat = r1[..k1.min(r1.len())].to_vec();
            concat.extend_from_slice(&r2[..k2.min(r2.len())]);
            p_once.fix_variables(&concat);

            prop_assert_eq!(p_step.evaluations, p_once.evaluations);
            prop_assert_eq!(p_step.num_vars, p_once.num_vars);
        }
    }
}
