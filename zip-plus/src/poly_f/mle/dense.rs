use core::ops::IndexMut;

use ark_ff::Zero;
use ark_std::{
    borrow::ToOwned,
    cfg_iter, cfg_iter_mut, log2,
    ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign},
    vec,
    vec::Vec,
};
#[cfg(feature = "parallel")]
use rayon::iter::*;

use super::{MultilinearExtension, swap_bits};
use crypto_primitives::{Matrix, PrimeField};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: Vec<F>,
    /// Number of variables
    pub num_vars: usize,
}

impl<F> DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &[F]) -> Self {
        Self::from_evaluations_vec(num_vars, evaluations.to_vec())
    }

    pub fn evaluate(&self, point: &[F]) -> Option<F> {
        if point.len() == self.num_vars {
            Some(self.fixed_variables(point)[0].clone())
        } else {
            None
        }
    }

    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<F>) -> Self {
        // assert that the number of variables matches the size of evaluations
        assert!(
            evaluations.len() <= 1 << num_vars,
            "The size of evaluations should not exceed 2^num_vars. \n eval len: {:?}. num vars: {num_vars}",
            evaluations.len()
        );

        if evaluations.len() != 1 << num_vars {
            let mut evaluations = evaluations;
            evaluations.resize(1 << num_vars, F::zero());
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
    pub fn from_matrix<M: Matrix<F>>(matrix: &M) -> Self {
        let n_vars: usize = (log2(matrix.num_rows()) + log2(matrix.num_cols())) as usize; // n_vars = s + s'

        // Matrices might need to get padded before turned into an MLE
        let padded_rows = matrix.num_rows().next_power_of_two();
        let padded_cols = matrix.num_cols().next_power_of_two();

        // build dense vector representing the sparse padded matrix
        let mut v = vec![F::zero(); padded_rows * padded_cols];

        for (row_i, row) in matrix.rows().enumerate() {
            for (col_i, val) in row {
                v[(padded_rows * *col_i) + row_i] = val.clone();
            }
        }

        // convert the dense vector into a mle
        Self::from_slice(n_vars, &v)
    }

    /// Takes n_vars and a dense slice and returns its dense MLE.
    pub fn from_slice(n_vars: usize, v: &[F]) -> Self {
        let v_padded: Vec<F> = if v.len() != (1 << n_vars) {
            // pad to 2^n_vars
            [
                v.to_owned(),
                ark_std::iter::repeat_n(F::zero(), (1 << n_vars) - v.len()).collect(),
            ]
            .concat()
        } else {
            v.to_owned()
        };
        Self::from_evaluations_vec(n_vars, v_padded)
    }

    pub fn relabel_in_place(&mut self, mut a: usize, mut b: usize, k: usize) {
        // enforce order of a and b
        if a > b {
            ark_std::mem::swap(&mut a, &mut b);
        }
        if a == b || k == 0 {
            return;
        }
        assert!(b + k <= self.num_vars, "invalid relabel argument");
        assert!(a + k <= b, "overlapped swap window is not allowed");
        for i in 0..self.evaluations.len() {
            let j = swap_bits(i, a, b, k);
            if i < j {
                self.evaluations.swap(i, j);
            }
        }
    }
}

impl<F> MultilinearExtension<F> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn fix_variables(&mut self, partial_point: &[F]) {
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

    fn fixed_variables(&self, partial_point: &[F]) -> Self {
        let mut res = self.clone();
        res.fix_variables(partial_point);
        res
    }
}

impl<F> Zero for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn zero() -> Self {
        Self {
            num_vars: 0,
            evaluations: vec![F::zero()],
        }
    }

    fn is_zero(&self) -> bool {
        self.num_vars == 0 && self.evaluations[0].is_zero()
    }
}

impl<F> Add for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl<F> Add for &DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = DenseMultilinearExtension<F>;

    fn add(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return self.clone();
        }

        if self.is_zero() {
            return rhs.clone();
        }

        assert_eq!(
            self.num_vars, rhs.num_vars,
            "trying to add two dense MLEs with different numbers of variables"
        );

        let result = cfg_iter!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Self::Output::from_evaluations_vec(self.num_vars, result)
    }
}

impl<F> AddAssign for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<F> AddAssign<&Self> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn add_assign(&mut self, other: &Self) {
        if self.is_zero() {
            *self = other.clone();
            return;
        }

        if other.is_zero() {
            return;
        }

        assert_eq!(
            self.num_vars, other.num_vars,
            "trying to add two dense MLEs with different numbers of variables"
        );

        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(other.evaluations))
            .for_each(|(a, b)| a.add_assign(b));
    }
}

impl<F> AddAssign<(F, &Self)> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn add_assign(&mut self, (r, other): (F, &Self)) {
        if self.is_zero() {
            *self = other.clone();

            cfg_iter_mut!(self.evaluations).for_each(|a| a.mul_assign(&r));

            return;
        }

        if other.is_zero() {
            return;
        }

        assert_eq!(
            self.num_vars, other.num_vars,
            "trying to add two dense MLEs with different numbers of variables"
        );

        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(other.evaluations))
            .for_each(|(a, b)| a.add_assign(&(r.clone() * b)));
    }
}

impl<F> Neg for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        cfg_iter_mut!(self.evaluations).for_each(|a| *a = a.clone().neg());
        self
    }
}

impl<F> Sub for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        &self - &other
    }
}

impl<F> Sub for &DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = DenseMultilinearExtension<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return self.clone();
        }

        if self.is_zero() {
            return rhs.clone().neg();
        }

        assert_eq!(
            self.num_vars, rhs.num_vars,
            "trying to subtract two dense MLEs with different numbers of variables"
        );
        let result = cfg_iter!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Self::Output::from_evaluations_vec(self.num_vars, result)
    }
}

impl<F> SubAssign for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign(&other);
    }
}

impl<F> SubAssign<&Self> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn sub_assign(&mut self, rhs: &Self) {
        if self.is_zero() {
            *self = rhs.clone().neg();
            return;
        }

        if rhs.is_zero() {
            return;
        }

        assert_eq!(
            self.num_vars, rhs.num_vars,
            "trying to subtract two dense MLEs with different numbers of variables"
        );

        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .for_each(|(a, b)| a.sub_assign(b.clone()));
    }
}

impl<F> Mul<F> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self.evaluations.iter_mut().for_each(|x| *x *= &rhs);

        self
    }
}

impl<F> MulAssign<F> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn mul_assign(&mut self, rhs: F) {
        self.evaluations.iter_mut().for_each(|x| *x *= &rhs);
    }
}

impl<F> Sub<F> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = Self;

    fn sub(mut self, rhs: F) -> Self::Output {
        self.evaluations.iter_mut().for_each(|x| *x -= rhs.clone());

        self
    }
}

impl<F> Add<F> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = Self;

    fn add(mut self, rhs: F) -> Self::Output {
        self.evaluations.iter_mut().for_each(|x| *x += &rhs);

        self
    }
}

impl<F> Index<usize> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<F> IndexMut<usize> for DenseMultilinearExtension<F>
where
    F: PrimeField,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.evaluations[index]
    }
}

unsafe impl<F> Send for DenseMultilinearExtension<F>
where
    F: PrimeField,
{}

unsafe impl<F> Sync for DenseMultilinearExtension<F>
where
    F: PrimeField,
{}
