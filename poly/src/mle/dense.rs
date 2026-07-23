mod try_collect_dense_mle;

use crate::{
    EvaluationError,
    mle::{MultilinearExtension, MultilinearExtensionRand},
};
use core::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use crypto_primitives::{FieldConfig, Matrix, Ring, Semiring, SemiringConfig, SetElement};
use rand::{distr::StandardUniform, prelude::*};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    slice::SliceIndex,
};
use zinc_utils::{cfg_into_iter, projectable_to_field::ProjectableToField, sub};

pub use try_collect_dense_mle::*;

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
        let n_vars: usize = // n_vars = s + s'
            (zinc_utils::log2(matrix.num_rows()) + zinc_utils::log2(matrix.num_cols())) as usize;

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

impl<R: Default> DenseMultilinearExtension<R> {
    pub fn from_evaluations_vec_pad(mut evaluations: Vec<R>) -> Self {
        let len = evaluations.len();

        evaluations.resize_with(len.next_power_of_two(), Default::default);

        let num_vars = zinc_utils::log2(evaluations.len()) as usize;

        Self {
            evaluations,
            num_vars,
        }
    }
}

impl<R: Clone> DenseMultilinearExtension<R> {
    pub fn from_evaluations_vec_pad_with_zero(mut evaluations: Vec<R>, zero: &R) -> Self {
        let len = evaluations.len();

        evaluations.resize(len.next_power_of_two(), zero.clone());

        let num_vars = zinc_utils::log2(evaluations.len()) as usize;

        Self {
            evaluations,
            num_vars,
        }
    }
}

// Keeping Send bound here to match the FromParallelIterator impl
impl<R: Send + Default> FromIterator<R> for DenseMultilinearExtension<R> {
    fn from_iter<T: IntoIterator<Item = R>>(iter: T) -> Self {
        Self::from_evaluations_vec_pad(iter.into_iter().collect())
    }
}

impl<R> Deref for DenseMultilinearExtension<R> {
    type Target = [R];

    fn deref(&self) -> &Self::Target {
        &self.evaluations
    }
}

impl<R> DerefMut for DenseMultilinearExtension<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evaluations
    }
}

impl<R> IntoIterator for DenseMultilinearExtension<R> {
    type Item = R;

    type IntoIter = std::vec::IntoIter<R>;

    fn into_iter(self) -> Self::IntoIter {
        self.evaluations.into_iter()
    }
}

#[cfg(feature = "parallel")]
impl<R: Send + Default> FromParallelIterator<R> for DenseMultilinearExtension<R> {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = R>,
    {
        Self::from_evaluations_vec_pad(par_iter.into_par_iter().collect())
    }
}

#[cfg(feature = "parallel")]
impl<R: Send + Sync> IntoParallelIterator for DenseMultilinearExtension<R> {
    type Iter = rayon::vec::IntoIter<R>;

    type Item = R;

    fn into_par_iter(self) -> Self::Iter {
        self.evaluations.into_par_iter()
    }
}

#[cfg(feature = "parallel")]
impl<'data, R: Send + Sync> IntoParallelRefIterator<'data> for &'data DenseMultilinearExtension<R> {
    type Iter = rayon::slice::Iter<'data, R>;

    type Item = &'data R;

    fn par_iter(&'data self) -> Self::Iter {
        self.evaluations.par_iter()
    }
}

#[cfg(feature = "parallel")]
impl<'data, R: Send + Sync> IntoParallelRefMutIterator<'data>
    for &'data mut DenseMultilinearExtension<R>
{
    type Iter = rayon::slice::IterMut<'data, R>;

    type Item = &'data mut R;

    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.evaluations.par_iter_mut()
    }
}

impl<R: SetElement> DenseMultilinearExtension<R> {
    pub fn evaluate<C>(&self, cfg: &C, point: &[R]) -> Result<R, EvaluationError>
    where
        C: SemiringConfig<Element = R>,
    {
        if point.len() == self.num_vars {
            Ok(self
                .fixed_variables(cfg, point)
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
        self.iter_mut().for_each(f);
    }

    fn binary<G>(&mut self, other: &Self, mut f: G)
    where
        G: FnMut(&mut R, &R),
    {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| f(a, b));
    }
}

impl<C: SemiringConfig> MultilinearExtension<C> for DenseMultilinearExtension<C::Element> {
    #[allow(clippy::arithmetic_side_effects)]
    fn fix_variables(&mut self, cfg: &C, partial_point: &[C::Element]) {
        assert!(
            partial_point.len() <= self.num_vars,
            "too many partial points"
        );

        let nv = self.num_vars;
        let dim = partial_point.len();

        for i in 1..dim + 1 {
            let r = &partial_point[i - 1];
            for b in 0..1 << (nv - i) {
                let left = &self[2 * b];
                let right = &self[2 * b + 1];
                // a = f(1) - f(0)
                let a = cfg.checked_sub(right, left).expect("Subtraction overflow");
                if !cfg.is_zero(&a) {
                    // self[b] = f(0) + r * a
                    let ar = cfg.checked_mul(&a, r).expect("Multiplication overflow");
                    self[b] = cfg.checked_add(left, &ar).expect("Addition overflow");
                } else {
                    self[b] = left.clone();
                };
            }
        }

        self.evaluations.truncate(1 << (nv - dim));
        self.num_vars = sub!(nv, dim);
    }

    fn fixed_variables(&self, cfg: &C, partial_point: &[C::Element]) -> Self {
        let mut res = self.clone();
        res.fix_variables(cfg, partial_point);
        res
    }
}

impl<R> MultilinearExtensionRand<R> for DenseMultilinearExtension<R>
where
    R: Send + Clone + Default,
    StandardUniform: Distribution<R>,
{
    fn rand<G: Rng + ?Sized>(num_vars: usize, rng: &mut G) -> Self {
        (0..1 << num_vars).map(|_| rng.random::<R>()).collect()
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for DenseMultilinearExtension<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for DenseMultilinearExtension<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
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

pub fn project_coeffs<C, R>(
    cfg: &C,
    mle: DenseMultilinearExtension<R>,
    sampled_value: &C::Element,
) -> DenseMultilinearExtension<C::Element>
where
    C: FieldConfig,
    R: ProjectableToField<C> + Send + Sync + 'static,
{
    let projection = R::prepare_projection(cfg, sampled_value);

    DenseMultilinearExtension {
        evaluations: cfg_into_iter!(mle.evaluations)
            .map(|x| projection(&x))
            .collect(),
        num_vars: mle.num_vars,
    }
}

pub trait CollectDenseMleWithZero: Iterator {
    fn collect_dense_mle_with_zero(
        self,
        zero: &Self::Item,
    ) -> DenseMultilinearExtension<Self::Item>;
}

impl<T> CollectDenseMleWithZero for T
where
    T: Iterator,
    T::Item: Clone,
{
    fn collect_dense_mle_with_zero(
        self,
        zero: &Self::Item,
    ) -> DenseMultilinearExtension<Self::Item> {
        let evaluations = self.collect();

        DenseMultilinearExtension::from_evaluations_vec_pad_with_zero(evaluations, zero)
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::redundant_clone,
    clippy::clone_on_copy
)]
mod tests {
    use super::*;
    use std::str::FromStr;

    use crypto_bigint::{U256, const_monty_params};
    use crypto_primitives::{
        BaseFieldConfig, DenseRowMatrix, FixedConfig, LiftElementWithConfig,
        ProjectElementWithConfig, SetConfig, crypto_bigint_const_monty::F256,
        crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use proptest::prelude::*;

    const LIMBS: usize = 4;

    fn get_dyn_config(hex_modulus: &str) -> MontyField<LIMBS> {
        let modulus =
            Uint::from_str(&format!("0x{hex_modulus}")).expect("Invalid modulus hex string");
        MontyField::new(&modulus).expect("Failed to create field config")
    }

    const MODULUS: &str = "0076F668F4274572E39A3EA8285319B5";
    type F = MontyField<LIMBS>;
    type E = <F as SetConfig>::Element;

    fn any_e(cfg: F) -> impl Strategy<Value = E> + 'static {
        any::<u128>().prop_map(move |v| cfg.project(&v))
    }

    fn any_dme() -> impl Strategy<Value = DenseMultilinearExtension<E>> {
        let cfg = get_dyn_config(MODULUS);
        (0usize..=5).prop_flat_map(move |n| {
            let len = 1usize << n;
            prop::collection::vec(any_e(cfg), len).prop_map(move |evals| {
                DenseMultilinearExtension::from_evaluations_vec(n, evals, cfg.zero())
            })
        })
    }

    #[test]
    fn test_dense_from_slice_and_indexing() {
        let cfg = get_dyn_config(MODULUS);
        let n_vars = 3usize;
        let v = vec![cfg.project(&1u64), cfg.project(&2u64), cfg.project(&3u64)];
        let dense = DenseMultilinearExtension::from_evaluations_slice(n_vars, &v, cfg.zero());
        assert_eq!(dense.num_vars, n_vars);
        let mut expected = v.clone();
        expected.resize(1 << n_vars, cfg.zero());
        assert_eq!(dense.evaluations, expected);
        assert_eq!(dense[0], cfg.project(&1u64));
        let mut d2 = dense.clone();
        d2[1] = cfg.project(&99u64);
        assert_eq!(d2[1], cfg.project(&99u64));
    }

    #[test]
    fn test_fix_variables_and_evaluate() {
        let cfg = get_dyn_config(MODULUS);
        let evals: Vec<E> = [10u64, 20, 30, 40].iter().map(|v| cfg.project(v)).collect();
        let mle = DenseMultilinearExtension::from_evaluations_vec(2, evals.clone(), cfg.zero());
        for (idx, (x0, x1)) in [
            (cfg.zero(), cfg.zero()),
            (cfg.one(), cfg.zero()),
            (cfg.zero(), cfg.one()),
            (cfg.one(), cfg.one()),
        ]
        .iter()
        .enumerate()
        {
            let val = mle
                .clone()
                .evaluate(&cfg, &[x0.clone(), x1.clone()])
                .unwrap();
            assert_eq!(val, evals[idx]);
        }
        let mut m2 = mle.clone();
        m2.fix_variables(&cfg, &[cfg.one()]);
        assert_eq!(m2.num_vars, 1);
        assert_eq!(
            m2.evaluations,
            vec![cfg.project(&20u64), cfg.project(&40u64)]
        );
    }

    #[test]
    fn test_from_matrix_padding_and_conversion() {
        let cfg = get_dyn_config(MODULUS);
        let m: DenseRowMatrix<E> = DenseRowMatrix::from(vec![
            vec![cfg.project(&5u64), cfg.zero()],
            vec![cfg.zero(), cfg.zero()],
            vec![cfg.zero(), cfg.project(&7u64)],
        ]);
        let dense = DenseMultilinearExtension::from_matrix(&m, cfg.zero());
        assert_eq!(dense.num_vars, 3);
        assert_eq!(dense[0], cfg.project(&5u64));
        assert_eq!(dense[5], cfg.project(&7u64));
        assert!(dense.iter().enumerate().all(|(i, v)| if i == 0 || i == 5 {
            true
        } else {
            cfg.is_zero(v)
        }));
    }

    #[test]
    fn test_from_evaluations_vec_padding_branch_and_slice() {
        let cfg = get_dyn_config(MODULUS);
        // len < 2^n triggers padding branch
        let evals = vec![cfg.project(&1u64), cfg.project(&2u64)];
        let n = 2usize; // 4 expected
        let d1 = DenseMultilinearExtension::from_evaluations_vec(n, evals.clone(), cfg.zero());
        let mut expected = evals.clone();
        expected.resize(1 << n, cfg.zero());
        assert_eq!(d1.evaluations, expected);
        let d2 = DenseMultilinearExtension::from_evaluations_slice(n, &evals, cfg.zero());
        assert_eq!(d2.evaluations, expected);
    }

    #[test]
    fn test_fix_variables_edge_cases_and_full_truncate() {
        let cfg = get_dyn_config(MODULUS);
        let evals: Vec<E> = [1u64, 2, 3, 4].iter().map(|v| cfg.project(v)).collect();
        let d = DenseMultilinearExtension::from_evaluations_vec(2, evals.clone(), cfg.zero());
        let d_fixed = d.fixed_variables(&cfg, &[]);
        assert_eq!(d_fixed.num_vars, 2);
        assert_eq!(d_fixed.evaluations, evals);

        let evals: Vec<E> = [10u64, 20, 30, 40].iter().map(|v| cfg.project(v)).collect();
        let mut d2 = DenseMultilinearExtension::from_evaluations_vec(2, evals, cfg.zero());
        d2.fix_variables(&cfg, &[cfg.one(), cfg.zero()]);
        assert_eq!(d2.num_vars, 0);
        assert_eq!(d2.evaluations, vec![cfg.project(&20u64)]);
    }

    #[test]
    fn test_evaluate_length_mismatch_returns_error() {
        let cfg = get_dyn_config(MODULUS);
        let evals: Vec<E> = [1u64, 2, 3, 4].iter().map(|v| cfg.project(v)).collect();
        let d = DenseMultilinearExtension::from_evaluations_vec(2, evals, cfg.zero());
        assert!(d.clone().evaluate(&cfg, &[cfg.one()]).is_err());
        assert!(
            d.evaluate(&cfg, &[cfg.one(), cfg.one(), cfg.zero()])
                .is_err()
        );
    }

    #[test]
    fn test_zero_impl_for_dense_mle() {
        let cfg = get_dyn_config(MODULUS);
        let z: DenseMultilinearExtension<E> = DenseMultilinearExtension::zero_vars(cfg.zero());
        assert_eq!(z.num_vars, 0);
        assert_eq!(z.evaluations, vec![cfg.zero()]);
    }

    #[test]
    fn test_arithmetic_ops_elementwise_add_sub_mul_and_neg() {
        let a: DenseMultilinearExtension<i128> =
            DenseMultilinearExtension::from_evaluations_vec(2, vec![1, 2, 3, 4], 0);
        let b: DenseMultilinearExtension<i128> =
            DenseMultilinearExtension::from_evaluations_vec(2, vec![5, 6, 7, 8], 0);

        let sum = a.clone() + &b;
        assert_eq!(sum.evaluations, vec![6, 8, 10, 12]);

        let diff = b.clone() - &a;
        assert_eq!(diff.evaluations, vec![4, 4, 4, 4]);

        let prod = a.clone() * &b;
        assert_eq!(prod.evaluations, vec![5, 12, 21, 32]);

        // Neg
        let neg_a = -a.clone();
        assert_eq!(neg_a.evaluations, vec![-1, -2, -3, -4]);
    }

    #[test]
    fn test_scalar_mul_and_assign_variants() {
        let a: DenseMultilinearExtension<i128> =
            DenseMultilinearExtension::from_evaluations_vec(2, vec![1, 2, 3, 4], 0);
        let b: DenseMultilinearExtension<i128> =
            DenseMultilinearExtension::from_evaluations_vec(2, vec![10, 20, 30, 40], 0);

        let scaled = a.clone() * 3;
        assert_eq!(scaled.evaluations, vec![3, 6, 9, 12]);

        let mut c = a.clone();
        c += &b;
        assert_eq!(c.evaluations, vec![11, 22, 33, 44]);

        c -= &b;
        assert_eq!(c.evaluations, a.evaluations);

        let mut d = a.clone();
        d *= &b;
        assert_eq!(d.evaluations, vec![10, 40, 90, 160]);

        let mut e = a.clone();
        e += (2, &b);
        assert_eq!(e.evaluations, vec![21, 42, 63, 84]);
    }

    fn any_aligned_pair_with_point() -> impl Strategy<
        Value = (
            DenseMultilinearExtension<E>,
            DenseMultilinearExtension<E>,
            Vec<E>,
        ),
    > {
        let cfg = get_dyn_config(MODULUS);
        (0usize..=5).prop_flat_map(move |n| {
            let len = 1usize << n;
            prop::collection::vec(any_e(cfg), len).prop_flat_map(move |e1| {
                let n2 = n;
                prop::collection::vec(any_e(cfg), len).prop_flat_map(move |e2| {
                    let n3 = n2;
                    point_n(n3).prop_map({
                        let e1v = e1.clone();
                        let e2v = e2.clone();
                        move |r| {
                            (
                                DenseMultilinearExtension::from_evaluations_vec(
                                    n3,
                                    e1v.clone(),
                                    cfg.zero(),
                                ),
                                DenseMultilinearExtension::from_evaluations_vec(
                                    n3,
                                    e2v.clone(),
                                    cfg.zero(),
                                ),
                                r,
                            )
                        }
                    })
                })
            })
        })
    }
    fn point_n(n: usize) -> impl Strategy<Value = Vec<E>> {
        prop::collection::vec(any_e(get_dyn_config(MODULUS)), n)
    }

    proptest! {
        #[test]
        #[cfg_attr(miri, ignore)] // long running
        fn prop_eval_add_is_linear((p1, p2, r) in any_aligned_pair_with_point()) {
            let cfg = get_dyn_config(MODULUS);
            // Elementwise sum via the config (raw elements have no std ops).
            let sum = DenseMultilinearExtension {
                num_vars: p1.num_vars,
                evaluations: p1
                    .evaluations
                    .iter()
                    .zip(p2.evaluations.iter())
                    .map(|(a, b)| cfg.add(a, b))
                    .collect(),
            };
            let lhs = sum.evaluate(&cfg, &r).unwrap();
            let rhs = cfg.add(
                &p1.evaluate(&cfg, &r).unwrap(),
                &p2.evaluate(&cfg, &r).unwrap(),
            );
            prop_assert_eq!(lhs, rhs);
        }

        #[test]
        #[cfg_attr(miri, ignore)] // long running
        fn prop_fix_vars_commutes_with_eval((p, r, k) in any_dme().prop_flat_map(|p| {
            let n = p.num_vars;
            let point = point_n(n);
            let ks = 0usize..=n;
            (Just(p), point, ks)
        })) {
            let cfg = get_dyn_config(MODULUS);
            let mut pfixed = p.clone();
            pfixed.fix_variables(&cfg, &r[..k]);
            let lhs = pfixed.evaluate(&cfg, &r[k..]).unwrap();
            let rhs = p.evaluate(&cfg, &r).unwrap();
            prop_assert_eq!(lhs, rhs);
        }

        #[test]
        #[cfg_attr(miri, ignore)] // long running
        fn prop_fix_vars_is_idempotent((p, k1, k2) in any_dme().prop_flat_map(|p| {
            let n = p.num_vars;
            let ks1 = 0usize..=n;
            (Just(p), ks1).prop_flat_map(move |(p, k1)| {
                let ks2 = 0usize..=n.saturating_sub(k1);
                (Just(p), Just(k1), ks2)
            })
        }), r1 in prop::collection::vec(any_e(get_dyn_config(MODULUS)), 0..=8usize), r2 in prop::collection::vec(any_e(get_dyn_config(MODULUS)), 0..=8usize)) {
            let cfg = get_dyn_config(MODULUS);
            let mut p_step = p.clone();
            p_step.fix_variables(&cfg, &r1[..k1.min(r1.len())]);
            p_step.fix_variables(&cfg, &r2[..k2.min(r2.len())]);

            let mut p_once = p.clone();
            let mut concat = r1[..k1.min(r1.len())].to_vec();
            concat.extend_from_slice(&r2[..k2.min(r2.len())]);
            p_once.fix_variables(&cfg, &concat);

            prop_assert_eq!(p_step.evaluations, p_once.evaluations);
            prop_assert_eq!(p_step.num_vars, p_once.num_vars);
        }
    }

    // Equivalence of the fixed-config path (`FixedConfig<ConstMontyField>`)
    // and the dynamic-config path (`MontyField`) over the same modulus.
    const_monty_params!(
        ModQ,
        U256,
        "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27"
    );
    type CF = F256<ModQ>;
    const MODULUS_Q: &str = "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27";

    proptest! {
        #[test]
        #[cfg_attr(miri, ignore)] // long running
        fn prop_mle_eval_fixed_cfg_matches_dyn_cfg((evals, point) in (0usize..=5).prop_flat_map(|n| {
            (
                prop::collection::vec(any::<u128>(), 1usize << n),
                prop::collection::vec(any::<u128>(), n),
            )
        })) {
            let n = point.len();
            let fixed_cfg = FixedConfig::<CF>::default();
            let dyn_cfg = get_dyn_config(MODULUS_Q);

            let p_fixed = DenseMultilinearExtension::from_evaluations_vec(
                n,
                evals.iter().map(|v| CF::from(*v)).collect(),
                fixed_cfg.zero(),
            );
            let p_dyn = DenseMultilinearExtension::from_evaluations_vec(
                n,
                evals.iter().map(|v| dyn_cfg.project(v)).collect(),
                dyn_cfg.zero(),
            );

            let r_fixed: Vec<CF> = point.iter().map(|v| CF::from(*v)).collect();
            let r_dyn: Vec<E> = point.iter().map(|v| dyn_cfg.project(v)).collect();

            let lhs = fixed_cfg.lift(&p_fixed.evaluate(&fixed_cfg, &r_fixed).unwrap());
            let rhs = dyn_cfg.lift(&p_dyn.evaluate(&dyn_cfg, &r_dyn).unwrap());
            prop_assert_eq!(lhs, rhs);
        }
    }
}
