use core::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    EvaluationError,
    mle::{MultilinearExtension, MultilinearExtensionRand},
};
use ark_std::{cfg_into_iter, log2};
use crypto_primitives::{Matrix, PrimeField, Ring, Semiring};
use rand::{distr::StandardUniform, prelude::*};
use rand_core::RngCore;
use zinc_utils::{
    add, inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField, sub,
};

use super::MultilinearExtensionWithConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearExtension<T> {
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: Vec<T>,
    /// Number of variables
    pub num_vars: usize,
}

impl<R> DenseMultilinearExtension<R> {
    pub fn zero_vars(evaluation: R) -> Self {
        Self {
            evaluations: vec![evaluation],
            num_vars: 0,
        }
    }
}

impl<R: Clone> DenseMultilinearExtension<R> {
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &[R], zero: R) -> Self {
        Self::from_evaluations_vec(num_vars, evaluations.to_vec(), zero)
    }

    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<R>, zero: R) -> Self {
        // assert that the number of variables matches the size of evaluations
        assert!(
            evaluations.len() <= 1 << num_vars,
            "The size of evaluations should not exceed 2^num_vars. \n eval len: {:?}. num vars: {num_vars}",
            evaluations.len()
        );

        if evaluations.len() != 1 << num_vars {
            let mut evaluations = evaluations;
            evaluations.resize(1 << num_vars, zero);
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
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_matrix<M: Matrix<R>>(matrix: &M, zero: R) -> Self {
        let n_vars: usize = (log2(matrix.num_rows()) + log2(matrix.num_cols())) as usize; // n_vars = s + s'

        // Matrices might need to get padded before turned into an MLE
        let padded_rows = matrix.num_rows().next_power_of_two();
        let padded_cols = matrix.num_cols().next_power_of_two();

        // build dense vector representing the sparse padded matrix
        let mut v = vec![zero.clone(); padded_rows * padded_cols];

        for (row_i, row) in matrix.cells().enumerate() {
            for (col_i, val) in row {
                v[(padded_cols * row_i) + col_i] = val.clone();
            }
        }

        // convert the dense vector into a mle
        Self::from_evaluations_slice(n_vars, &v, zero)
    }
}

impl<R: Semiring> DenseMultilinearExtension<R> {
    pub fn evaluate<S>(&self, point: &[S], zero: R) -> Result<R, EvaluationError>
    where
        R: for<'a> MulByScalar<&'a S>,
    {
        if point.len() == self.num_vars {
            Ok(self
                .fixed_variables(point, zero)
                .evaluations
                .into_iter()
                .next()
                .expect("Evaluations should not be empty"))
        } else {
            Err(EvaluationError::WrongPointWidth {
                expected: self.num_vars,
                actual: point.len(),
            })
        }
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

impl<F> MultilinearExtensionWithConfig<F> for DenseMultilinearExtension<F::Inner>
where
    F: InnerTransparentField,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn fix_variables_with_config(
        &mut self,
        partial_point: &[F],
        config: &<F as PrimeField>::Config,
    ) {
        assert!(
            partial_point.len() <= self.num_vars,
            "too many partial points"
        );

        if partial_point.len().is_zero() {
            return;
        }

        let poly = &mut self.evaluations;
        let nv = self.num_vars;
        let dim = partial_point.len();

        let mut r = partial_point[0].clone();
        for i in 1..dim + 1 {
            for b in 0..1 << (nv - i) {
                *r.inner_mut() = partial_point[i - 1].inner().clone();
                if poly[2 * b + 1] != poly[2 * b] {
                    // a = f(1) - f(0)
                    let a = F::sub_inner(&poly[2 * b + 1], &poly[2 * b], config);

                    // poly[b] = f(0) + r * a
                    r.mul_assign_by_inner(&a);
                    poly[b] = F::add_inner(&poly[2 * b], r.inner(), config);
                } else {
                    poly[b] = poly[2 * b].clone();
                };
            }
        }

        self.evaluations.truncate(1 << (nv - dim));
        self.num_vars = sub!(nv, dim);
    }

    fn fixed_variables_with_config(
        &self,
        partial_point: &[F],
        config: &<F as PrimeField>::Config,
    ) -> Self {
        let mut res = self.clone();
        res.fix_variables_with_config(partial_point, config);
        res
    }

    fn evaluate_with_config(&self, point: &[F], config: &<F as PrimeField>::Config) -> Option<F> {
        if point.len() == self.num_vars {
            Some(F::new_unchecked_with_cfg(
                self.fixed_variables_with_config(point, config)
                    .evaluations
                    .into_iter()
                    .next()
                    .expect("Evaluations should not be empty"),
                config,
            ))
        } else {
            None
        }
    }
}

impl<R> MultilinearExtension<R> for DenseMultilinearExtension<R>
where
    R: Semiring,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn fix_variables<S>(&mut self, partial_point: &[S], zero: R)
    where
        R: for<'a> MulByScalar<&'a S>,
    {
        assert!(
            partial_point.len() <= self.num_vars,
            "too many partial points"
        );

        let poly = &mut self.evaluations;
        let nv = self.num_vars;
        let dim = partial_point.len();

        for i in 1..dim + 1 {
            let r = &partial_point[i - 1];
            for b in 0..1 << (nv - i) {
                let left = &poly[2 * b];
                let right = &poly[2 * b + 1];
                // a = f(1) - f(0)
                let a = sub!(right, &left);
                if a != zero {
                    // poly[b] = f(0) + r * a
                    let ar = a.mul_by_scalar(r).expect("Multiplication overflow");
                    poly[b] = add!(left, &ar);
                } else {
                    poly[b] = left.clone();
                };
            }
        }

        self.evaluations.truncate(1 << (nv - dim));
        self.num_vars = sub!(nv, dim);
    }

    fn fixed_variables<S>(&self, partial_point: &[S], zero: R) -> Self
    where
        R: for<'a> MulByScalar<&'a S>,
    {
        let mut res = self.clone();
        res.fix_variables(partial_point, zero);
        res
    }
}

impl<R> MultilinearExtensionRand<R> for DenseMultilinearExtension<R>
where
    R: Clone,
    StandardUniform: Distribution<R>,
{
    fn rand<Rng: RngCore + ?Sized>(num_vars: usize, rng: &mut Rng) -> Self {
        Self {
            num_vars,
            evaluations: (0..1 << num_vars).map(|_| rng.random::<R>()).collect(),
        }
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

impl<R: Ring> Neg for DenseMultilinearExtension<R> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.unary(|v| *v = v.checked_neg().expect("Negation overflow"));
        self
    }
}

impl<R: Semiring> Add for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<R: Semiring> Add<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a += b);
        self
    }
}

impl<R: Semiring> Sub<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a -= b);
        self
    }
}

impl<R: Semiring> Mul<&Self> for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.binary(rhs, |a, b| *a *= b);
        self
    }
}

impl<R: Semiring> Mul<R> for DenseMultilinearExtension<R> {
    type Output = Self;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul(mut self, rhs: R) -> Self::Output {
        self.unary(|v| *v *= &rhs);
        self
    }
}

impl<R: Semiring> AddAssign<&Self> for DenseMultilinearExtension<R> {
    #[allow(clippy::arithmetic_side_effects)]
    fn add_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a += b);
    }
}

impl<R: Semiring> SubAssign<&Self> for DenseMultilinearExtension<R> {
    #[allow(clippy::arithmetic_side_effects)]
    fn sub_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a -= b);
    }
}

impl<R: Semiring> MulAssign<&Self> for DenseMultilinearExtension<R> {
    #[allow(clippy::arithmetic_side_effects)]
    fn mul_assign(&mut self, rhs: &Self) {
        self.binary(rhs, |a, b| *a *= b);
    }
}

impl<R: Semiring> AddAssign<(R, &Self)> for DenseMultilinearExtension<R> {
    #[allow(clippy::arithmetic_side_effects)]
    fn add_assign(&mut self, rhs: (R, &Self)) {
        let coeff = rhs.0;
        self.binary(rhs.1, |a, b| *a += b.clone() * &coeff);
    }
}

pub fn project_coeffs<F: PrimeField, R: ProjectableToField<F> + Send + Sync>(
    mle: DenseMultilinearExtension<R>,
    sampled_value: &F,
) -> DenseMultilinearExtension<F::Inner> {
    let projection = R::prepare_projection(sampled_value);

    DenseMultilinearExtension {
        evaluations: cfg_into_iter!(mle.evaluations)
            .map(|x| projection(&x).inner().clone())
            .collect(),
        num_vars: mle.num_vars,
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use crate::utils::{build_eq_x_r, build_eq_x_r_vec};

    use super::*;

    use crypto_primitives::{
        DenseRowMatrix, IntoWithConfig, PrimeField, crypto_bigint_monty::MontyField,
        crypto_bigint_uint::Uint,
    };
    use proptest::prelude::*;

    const LIMBS: usize = 4;

    fn get_dyn_config(hex_modulus: &str) -> <MontyField<LIMBS> as PrimeField>::Config {
        let modulus = Uint::new(
            crypto_bigint::Uint::from_str_radix_vartime(hex_modulus, 16)
                .expect("Invalid modulus hex string"),
        );
        MontyField::make_cfg(&modulus).expect("Failed to create field config")
    }

    const MODULUS: &str = "0076F668F4274572E39A3EA8285319B5";
    type F = MontyField<LIMBS>;

    fn any_f(cfg: <F as PrimeField>::Config) -> impl Strategy<Value = F> + 'static {
        any::<u128>().prop_map(move |v| v.into_with_cfg(&cfg))
    }

    fn any_dme() -> impl Strategy<Value = DenseMultilinearExtension<F>> {
        let cfg = get_dyn_config(MODULUS);
        (0usize..=5).prop_flat_map(move |n| {
            let len = 1usize << n;
            let cfg = cfg;
            prop::collection::vec(any_f(cfg), len).prop_map(move |evals| {
                DenseMultilinearExtension::from_evaluations_vec(n, evals, F::zero_with_cfg(&cfg))
            })
        })
    }

    #[test]
    fn test_build_eq_x_r_vec_basic() {
        let cfg = get_dyn_config(MODULUS);
        let r: [F; _] = [3_u64.into_with_cfg(&cfg)];
        let evals = build_eq_x_r_vec(&r, &cfg).unwrap();
        assert_eq!(
            evals,
            vec![F::one_with_cfg(&cfg) - r[0].clone(), r[0].clone()]
        );
    }

    #[test]
    fn test_build_eq_x_r_vec_two_vars() {
        let cfg = get_dyn_config(MODULUS);
        let r: [F; _] = [2u64.into_with_cfg(&cfg), 5u64.into_with_cfg(&cfg)];
        let evals = build_eq_x_r_vec(&r, &cfg).unwrap();
        let e00 = (F::one_with_cfg(&cfg) - r[0].clone()) * (F::one_with_cfg(&cfg) - r[1].clone());
        let e01 = r[0].clone() * (F::one_with_cfg(&cfg) - r[1].clone());
        let e10 = (F::one_with_cfg(&cfg) - r[0].clone()) * r[1].clone();
        let e11 = r[0].clone() * r[1].clone();
        assert_eq!(evals, vec![e00, e01, e10, e11]);
    }

    #[test]
    fn test_build_eq_x_r_error_on_empty() {
        let cfg = get_dyn_config(MODULUS);
        let r: [F; 0] = [];
        let err = build_eq_x_r_vec(&r, &cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Invalid parameters"));
    }

    #[test]
    fn test_build_eq_x_r_mle_properties() {
        let cfg = get_dyn_config(MODULUS);
        let r: [F; _] = [
            7u64.into_with_cfg(&cfg),
            11u64.into_with_cfg(&cfg),
            13u64.into_with_cfg(&cfg),
        ];
        let mle = build_eq_x_r(&r, &cfg).unwrap();
        assert_eq!(mle.num_vars, r.len());
        let evals = mle.evaluations;
        let direct = build_eq_x_r_vec(&r, &cfg).unwrap();
        assert_eq!(evals, direct);
    }

    #[test]
    fn test_dense_from_slice_and_indexing() {
        let cfg = get_dyn_config(MODULUS);
        let n_vars = 3usize;
        let v = vec![
            1u64.into_with_cfg(&cfg),
            2u64.into_with_cfg(&cfg),
            3u64.into_with_cfg(&cfg),
        ];
        let dense =
            DenseMultilinearExtension::from_evaluations_slice(n_vars, &v, F::zero_with_cfg(&cfg));
        assert_eq!(dense.num_vars, n_vars);
        let mut expected = v.clone();
        expected.resize(1 << n_vars, F::zero_with_cfg(&cfg));
        assert_eq!(dense.evaluations, expected);
        assert_eq!(dense[0], 1u64.into_with_cfg(&cfg));
        let mut d2 = dense.clone();
        d2[1] = 99u64.into_with_cfg(&cfg);
        assert_eq!(d2[1], 99u64.into_with_cfg(&cfg));
    }

    #[test]
    fn test_fix_variables_and_evaluate() {
        let cfg = get_dyn_config(MODULUS);
        let evals = vec![
            10u64.into_with_cfg(&cfg),
            20u64.into_with_cfg(&cfg),
            30u64.into_with_cfg(&cfg),
            40u64.into_with_cfg(&cfg),
        ];
        let mle = DenseMultilinearExtension::from_evaluations_vec(
            2,
            evals.clone(),
            F::zero_with_cfg(&cfg),
        );
        for (idx, (x0, x1)) in [
            (F::zero_with_cfg(&cfg), F::zero_with_cfg(&cfg)),
            (F::one_with_cfg(&cfg), F::zero_with_cfg(&cfg)),
            (F::zero_with_cfg(&cfg), F::one_with_cfg(&cfg)),
            (F::one_with_cfg(&cfg), F::one_with_cfg(&cfg)),
        ]
        .iter()
        .enumerate()
        {
            let val = mle
                .evaluate(&[x0.clone(), x1.clone()], F::zero_with_cfg(&cfg))
                .unwrap();
            assert_eq!(val, evals[idx]);
        }
        let mut m2 = mle.clone();
        m2.fix_variables(&[F::one_with_cfg(&cfg)], F::zero_with_cfg(&cfg));
        assert_eq!(m2.num_vars, 1);
        assert_eq!(
            m2.evaluations,
            vec![20u64.into_with_cfg(&cfg), 40u64.into_with_cfg(&cfg)]
        );
    }

    #[test]
    fn test_from_matrix_padding_and_conversion() {
        let cfg = get_dyn_config(MODULUS);
        let m: DenseRowMatrix<F> = DenseRowMatrix::from(vec![
            vec![5u64.into_with_cfg(&cfg), F::zero_with_cfg(&cfg)],
            vec![F::zero_with_cfg(&cfg), F::zero_with_cfg(&cfg)],
            vec![F::zero_with_cfg(&cfg), 7u64.into_with_cfg(&cfg)],
        ]);
        let dense = DenseMultilinearExtension::from_matrix(&m, F::zero_with_cfg(&cfg));
        assert_eq!(dense.num_vars, 3);
        let evals = dense.evaluations;
        assert_eq!(evals[0], 5u64.into_with_cfg(&cfg));
        assert_eq!(evals[5], 7u64.into_with_cfg(&cfg));
        assert!(evals.iter().enumerate().all(|(i, v)| if i == 0 || i == 5 {
            true
        } else {
            v.is_zero_with_cfg(&cfg)
        }));
    }

    #[test]
    fn test_from_evaluations_vec_padding_branch_and_slice() {
        let cfg = get_dyn_config(MODULUS);
        // len < 2^n triggers padding branch
        let evals = vec![1u64.into_with_cfg(&cfg), 2u64.into_with_cfg(&cfg)];
        let n = 2usize; // 4 expected
        let d1 = DenseMultilinearExtension::from_evaluations_vec(
            n,
            evals.clone(),
            F::zero_with_cfg(&cfg),
        );
        let mut expected = evals.clone();
        expected.resize(1 << n, F::zero_with_cfg(&cfg));
        assert_eq!(d1.evaluations, expected);
        let d2 =
            DenseMultilinearExtension::from_evaluations_slice(n, &evals, F::zero_with_cfg(&cfg));
        assert_eq!(d2.evaluations, expected);
    }

    #[test]
    fn test_fix_variables_edge_cases_and_full_truncate() {
        let cfg = get_dyn_config(MODULUS);
        let d = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                1.into_with_cfg(&cfg),
                2.into_with_cfg(&cfg),
                3.into_with_cfg(&cfg),
                4.into_with_cfg(&cfg),
            ],
            F::zero_with_cfg(&cfg),
        );
        let d_fixed = d.fixed_variables(&[], F::zero_with_cfg(&cfg));
        assert_eq!(d_fixed.num_vars, 2);
        assert_eq!(
            d_fixed.evaluations,
            vec![
                1.into_with_cfg(&cfg),
                2.into_with_cfg(&cfg),
                3.into_with_cfg(&cfg),
                4.into_with_cfg(&cfg)
            ]
        );
        let mut d2 = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                10.into_with_cfg(&cfg),
                20.into_with_cfg(&cfg),
                30.into_with_cfg(&cfg),
                40.into_with_cfg(&cfg),
            ],
            F::zero_with_cfg(&cfg),
        );
        d2.fix_variables(
            &[F::one_with_cfg(&cfg), F::zero_with_cfg(&cfg)],
            F::zero_with_cfg(&cfg),
        );
        assert_eq!(d2.num_vars, 0);
        assert_eq!(d2.evaluations, vec![20.into_with_cfg(&cfg)]);
    }

    #[test]
    fn test_evaluate_length_mismatch_returns_error() {
        let cfg = get_dyn_config(MODULUS);
        let d = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                1.into_with_cfg(&cfg),
                2.into_with_cfg(&cfg),
                3.into_with_cfg(&cfg),
                4.into_with_cfg(&cfg),
            ],
            F::zero_with_cfg(&cfg),
        );
        assert!(
            d.evaluate(&[F::one_with_cfg(&cfg)], F::zero_with_cfg(&cfg))
                .is_err()
        );
        assert!(
            d.evaluate(
                &[
                    F::one_with_cfg(&cfg),
                    F::one_with_cfg(&cfg),
                    F::zero_with_cfg(&cfg)
                ],
                F::zero_with_cfg(&cfg)
            )
            .is_err()
        );
    }

    #[test]
    fn test_zero_impl_for_dense_mle() {
        let cfg = get_dyn_config(MODULUS);
        let z: DenseMultilinearExtension<F> =
            DenseMultilinearExtension::zero_vars(F::zero_with_cfg(&cfg));
        assert_eq!(z.num_vars, 0);
        assert_eq!(z.evaluations, vec![F::zero_with_cfg(&cfg)]);
    }

    #[test]
    fn test_arithmetic_ops_elementwise_add_sub_mul_and_neg() {
        let cfg = get_dyn_config(MODULUS);
        let a = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                1.into_with_cfg(&cfg),
                2.into_with_cfg(&cfg),
                3.into_with_cfg(&cfg),
                4.into_with_cfg(&cfg),
            ],
            F::zero_with_cfg(&cfg),
        );
        let b = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                5.into_with_cfg(&cfg),
                6.into_with_cfg(&cfg),
                7.into_with_cfg(&cfg),
                8.into_with_cfg(&cfg),
            ],
            F::zero_with_cfg(&cfg),
        );

        let sum = a.clone() + &b;
        assert_eq!(
            sum.evaluations,
            vec![
                6.into_with_cfg(&cfg),
                8.into_with_cfg(&cfg),
                10.into_with_cfg(&cfg),
                12.into_with_cfg(&cfg)
            ]
        );

        let diff = b.clone() - &a;
        assert_eq!(
            diff.evaluations,
            vec![
                4.into_with_cfg(&cfg),
                4.into_with_cfg(&cfg),
                4.into_with_cfg(&cfg),
                4.into_with_cfg(&cfg)
            ]
        );

        let prod = a.clone() * &b;
        assert_eq!(
            prod.evaluations,
            vec![
                5.into_with_cfg(&cfg),
                12.into_with_cfg(&cfg),
                21.into_with_cfg(&cfg),
                32.into_with_cfg(&cfg)
            ]
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
        let cfg = get_dyn_config(MODULUS);
        let a = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                1.into_with_cfg(&cfg),
                2.into_with_cfg(&cfg),
                3.into_with_cfg(&cfg),
                4.into_with_cfg(&cfg),
            ],
            F::zero_with_cfg(&cfg),
        );
        let b = DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![
                10.into_with_cfg(&cfg),
                20.into_with_cfg(&cfg),
                30.into_with_cfg(&cfg),
                40.into_with_cfg(&cfg),
            ],
            F::zero_with_cfg(&cfg),
        );

        let three: F = 3u64.into_with_cfg(&cfg);
        let scaled = a.clone() * three;
        assert_eq!(
            scaled.evaluations,
            vec![
                3.into_with_cfg(&cfg),
                6.into_with_cfg(&cfg),
                9.into_with_cfg(&cfg),
                12.into_with_cfg(&cfg)
            ]
        );

        let mut c = a.clone();
        c += &b;
        assert_eq!(
            c.evaluations,
            vec![
                11.into_with_cfg(&cfg),
                22.into_with_cfg(&cfg),
                33.into_with_cfg(&cfg),
                44.into_with_cfg(&cfg)
            ]
        );

        c -= &b;
        assert_eq!(c.evaluations, a.evaluations);

        let mut d = a.clone();
        d *= &b;
        assert_eq!(
            d.evaluations,
            vec![
                10.into_with_cfg(&cfg),
                40.into_with_cfg(&cfg),
                90.into_with_cfg(&cfg),
                160.into_with_cfg(&cfg)
            ]
        );

        let mut e = a.clone();
        let two = 2u64.into_with_cfg(&cfg);
        e += (two, &b);
        assert_eq!(
            e.evaluations,
            vec![
                21.into_with_cfg(&cfg),
                42.into_with_cfg(&cfg),
                63.into_with_cfg(&cfg),
                84.into_with_cfg(&cfg)
            ]
        );
    }

    fn any_aligned_pair_with_point() -> impl Strategy<
        Value = (
            DenseMultilinearExtension<F>,
            DenseMultilinearExtension<F>,
            Vec<F>,
        ),
    > {
        let cfg = get_dyn_config(MODULUS);
        (0usize..=5).prop_flat_map(move |n| {
            let cfg = cfg;
            let len = 1usize << n;
            prop::collection::vec(any_f(cfg), len).prop_flat_map(move |e1| {
                let cfg = cfg;
                let n2 = n;
                prop::collection::vec(any_f(cfg), len).prop_flat_map(move |e2| {
                    let cfg = cfg;
                    let n3 = n2;
                    point_n(n3).prop_map({
                        let e1v = e1.clone();
                        let e2v = e2.clone();
                        move |r| {
                            (
                                DenseMultilinearExtension::from_evaluations_vec(
                                    n3,
                                    e1v.clone(),
                                    F::zero_with_cfg(&cfg),
                                ),
                                DenseMultilinearExtension::from_evaluations_vec(
                                    n3,
                                    e2v.clone(),
                                    F::zero_with_cfg(&cfg),
                                ),
                                r,
                            )
                        }
                    })
                })
            })
        })
    }
    fn point_n(n: usize) -> impl Strategy<Value = Vec<F>> {
        prop::collection::vec(any_f(get_dyn_config(MODULUS)), n)
    }

    proptest! {
        #[test]
        fn prop_eval_add_is_linear((p1, p2, r) in any_aligned_pair_with_point()) {
            let cfg = get_dyn_config(MODULUS);
            let lhs = (p1.clone() + &p2).evaluate(&r, F::zero_with_cfg(&cfg)).unwrap();
            let rhs = p1.evaluate(&r, F::zero_with_cfg(&cfg)).unwrap() + p2.evaluate(&r, F::zero_with_cfg(&cfg)).unwrap();
            prop_assert_eq!(lhs, rhs);
        }

        #[test]
        fn prop_fix_vars_commutes_with_eval((p, r, k) in any_dme().prop_flat_map(|p| {
            let n = p.num_vars;
            let point = point_n(n);
            let ks = 0usize..=n;
            (Just(p), point, ks)
        })) {
            let cfg = get_dyn_config(MODULUS);
            let mut pfixed = p.clone();
            pfixed.fix_variables(&r[..k], F::zero_with_cfg(&cfg));
            let lhs = pfixed.evaluate(&r[k..], F::zero_with_cfg(&cfg)).unwrap();
            let rhs = p.evaluate(&r, F::zero_with_cfg(&cfg)).unwrap();
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
        }), r1 in prop::collection::vec(any_f(get_dyn_config(MODULUS)), 0..=8usize), r2 in prop::collection::vec(any_f(get_dyn_config(MODULUS)), 0..=8usize)) {
            let cfg = get_dyn_config(MODULUS);
            let mut p_step = p.clone();
            p_step.fix_variables(&r1[..k1.min(r1.len())], F::zero_with_cfg(&cfg));
            p_step.fix_variables(&r2[..k2.min(r2.len())], F::zero_with_cfg(&cfg));

            let mut p_once = p.clone();
            let mut concat = r1[..k1.min(r1.len())].to_vec();
            concat.extend_from_slice(&r2[..k2.min(r2.len())]);
            p_once.fix_variables(&concat, F::zero_with_cfg(&cfg));

            prop_assert_eq!(p_step.evaluations, p_once.evaluations);
            prop_assert_eq!(p_step.num_vars, p_once.num_vars);
        }

        #[test]
        fn prop_mle_eval_eq_eval_with_config((p, r) in any_dme().prop_flat_map(|p| {
            let n = p.num_vars;
            let point = point_n(n);
            (Just(p), point)
        })) {
            let cfg = get_dyn_config(MODULUS);

            let p_inner = DenseMultilinearExtension {
                num_vars: p.num_vars,
                evaluations: p.evaluations.iter().map(|x| *x.inner()).collect()
            };

            let lhs = p.evaluate(&r, F::zero_with_cfg(&cfg)).unwrap();
            let rhs = p_inner.evaluate_with_config(&r, &cfg).unwrap();
            prop_assert_eq!(lhs, rhs);
        }
    }
}
