use core::ops::IndexMut;

use ark_ff::Zero;
use ark_std::{
    borrow::ToOwned,
    cfg_iter, cfg_iter_mut, log2,
    ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign},
    rand, vec,
    vec::Vec,
};
#[cfg(feature = "parallel")]
use rayon::iter::*;

use super::{MultilinearExtension, swap_bits};
use crate::traits::Integer;
use crypto_primitives::Matrix;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearExtension<I> {
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: Vec<I>,
    /// Number of variables
    pub num_vars: usize,
}

impl<I: Integer> DenseMultilinearExtension<I> {
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &[I]) -> Self {
        Self::from_evaluations_vec(num_vars, evaluations.to_vec())
    }

    pub fn evaluate(&self, point: &[I]) -> Option<I> {
        if point.len() == self.num_vars {
            Some(self.fixed_variables(point)[0].clone())
        } else {
            None
        }
    }

    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<I>) -> Self {
        // assert that the number of variables matches the size of evaluations
        assert!(
            evaluations.len() <= 1 << num_vars,
            "The size of evaluations should not exceed 2^num_vars. \n eval len: {:?}. num vars: {num_vars}",
            evaluations.len()
        );

        if evaluations.len() != 1 << num_vars {
            let mut evaluations = evaluations;
            evaluations.resize(1 << num_vars, I::ZERO);
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
    pub fn from_matrix<M: Matrix<I>>(matrix: &M) -> Self {
        let n_vars: usize = (log2(matrix.num_rows()) + log2(matrix.num_cols())) as usize; // n_vars = s + s'

        // Matrices might need to get padded before turned into an MLE
        let padded_rows = matrix.num_rows().next_power_of_two();
        let padded_cols = matrix.num_cols().next_power_of_two();

        // build dense vector representing the sparse padded matrix
        let mut v = vec![I::ZERO; padded_rows * padded_cols];

        for (row_i, row) in matrix.rows().enumerate() {
            for (col_i, val) in row {
                v[(padded_cols * row_i) + *col_i] = val.clone();
            }
        }

        // convert the dense vector into a mle
        Self::from_slice(n_vars, &v)
    }

    /// Takes n_vars and a dense slice and returns its dense MLE.
    pub fn from_slice(n_vars: usize, v: &[I]) -> Self {
        let v_padded: Vec<I> = if v.len() != (1 << n_vars) {
            // pad to 2^n_vars
            [
                v.to_owned(),
                ark_std::iter::repeat_n(I::ZERO, (1 << n_vars) - v.len()).collect(),
            ]
            .concat()
        } else {
            v.to_owned()
        };
        DenseMultilinearExtension::from_evaluations_vec(n_vars, v_padded)
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

impl<I: Integer> MultilinearExtension<I> for DenseMultilinearExtension<I> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn rand<Rn: rand::Rng>(num_vars: usize, rng: &mut Rn) -> Self {
        Self::from_evaluations_vec(
            num_vars,
            (0..1 << num_vars).map(|_| I::random(rng)).collect(),
        )
    }

    fn relabel(&self, a: usize, b: usize, k: usize) -> Self {
        let mut copy = self.clone();
        copy.relabel_in_place(a, b, k);
        copy
    }

    fn fix_variables(&mut self, partial_point: &[I]) {
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
                let a = right - &left;
                if !Zero::is_zero(&a) {
                    poly[b] = left + r.clone() * a;
                } else {
                    poly[b] = left;
                };
            }
        }

        self.evaluations.truncate(1 << (nv - dim));
        self.num_vars = nv - dim;
    }

    fn fixed_variables(&self, partial_point: &[I]) -> Self {
        let mut res = self.clone();
        res.fix_variables(partial_point);
        res
    }

    fn to_evaluations(&self) -> Vec<I> {
        self.evaluations.to_vec()
    }
}

impl<I: Integer> Zero for DenseMultilinearExtension<I> {
    fn zero() -> Self {
        Self {
            num_vars: 0,
            evaluations: vec![Zero::zero()],
        }
    }

    fn is_zero(&self) -> bool {
        self.num_vars == 0 && Zero::is_zero(&self.evaluations[0])
    }
}

impl<I: Integer> Add for DenseMultilinearExtension<I> {
    type Output = DenseMultilinearExtension<I>;

    fn add(self, other: DenseMultilinearExtension<I>) -> Self {
        &self + &other
    }
}

impl<'a, I: Integer> Add<&'a DenseMultilinearExtension<I>> for &DenseMultilinearExtension<I> {
    type Output = DenseMultilinearExtension<I>;

    fn add(self, rhs: &'a DenseMultilinearExtension<I>) -> Self::Output {
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
            .map(|(a, b)| a.clone() + b)
            .collect();

        Self::Output::from_evaluations_vec(self.num_vars, result)
    }
}

impl<I: Integer> AddAssign for DenseMultilinearExtension<I> {
    fn add_assign(&mut self, rhs: DenseMultilinearExtension<I>) {
        self.add_assign(&rhs);
    }
}

impl<'a, I: Integer> AddAssign<&'a DenseMultilinearExtension<I>> for DenseMultilinearExtension<I> {
    fn add_assign(&mut self, other: &'a DenseMultilinearExtension<I>) {
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

impl<I: Integer> AddAssign<(I, &DenseMultilinearExtension<I>)> for DenseMultilinearExtension<I> {
    fn add_assign(&mut self, (r, other): (I, &DenseMultilinearExtension<I>)) {
        if self.is_zero() {
            *self = other.clone();

            cfg_iter_mut!(self.evaluations).for_each(|a| *a = a.clone() * &r);

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

impl<I: Integer> Neg for DenseMultilinearExtension<I> {
    type Output = DenseMultilinearExtension<I>;

    fn neg(mut self) -> Self {
        cfg_iter_mut!(self.evaluations).for_each(|a| *a = I::ZERO - a);

        self
    }
}

impl<I: Integer> Sub for DenseMultilinearExtension<I> {
    type Output = Self;

    fn sub(self, other: DenseMultilinearExtension<I>) -> Self {
        &self - &other
    }
}

impl<'a, I: Integer> Sub<&'a DenseMultilinearExtension<I>> for &DenseMultilinearExtension<I> {
    type Output = DenseMultilinearExtension<I>;

    fn sub(self, rhs: &'a DenseMultilinearExtension<I>) -> Self::Output {
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
            .map(|(a, b)| a.clone() - b)
            .collect();

        Self::Output::from_evaluations_vec(self.num_vars, result)
    }
}

impl<I: Integer> SubAssign for DenseMultilinearExtension<I> {
    fn sub_assign(&mut self, other: DenseMultilinearExtension<I>) {
        self.sub_assign(&other);
    }
}

impl<'a, I: Integer> SubAssign<&'a DenseMultilinearExtension<I>> for DenseMultilinearExtension<I> {
    fn sub_assign(&mut self, rhs: &'a DenseMultilinearExtension<I>) {
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
            .for_each(|(a, b)| *a = a.clone() - b);
    }
}

impl<I: Integer> Mul<I> for DenseMultilinearExtension<I> {
    type Output = DenseMultilinearExtension<I>;

    fn mul(mut self, rhs: I) -> DenseMultilinearExtension<I> {
        self.evaluations
            .iter_mut()
            .for_each(|x| *x = x.clone() * &rhs);

        self
    }
}

impl<I: Integer> MulAssign<I> for DenseMultilinearExtension<I> {
    fn mul_assign(&mut self, rhs: I) {
        self.evaluations
            .iter_mut()
            .for_each(|x| *x = x.clone() * &rhs);
    }
}

impl<I: Integer> Sub<I> for DenseMultilinearExtension<I> {
    type Output = DenseMultilinearExtension<I>;

    fn sub(mut self, rhs: I) -> DenseMultilinearExtension<I> {
        self.evaluations
            .iter_mut()
            .for_each(|x| *x = x.clone() - &rhs);

        self
    }
}

impl<I: Integer> Add<I> for DenseMultilinearExtension<I> {
    type Output = DenseMultilinearExtension<I>;

    fn add(mut self, rhs: I) -> DenseMultilinearExtension<I> {
        self.evaluations.iter_mut().for_each(|x| *x += &rhs);

        self
    }
}

impl<I: Integer> Index<usize> for DenseMultilinearExtension<I> {
    type Output = I;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<I: Integer> IndexMut<usize> for DenseMultilinearExtension<I> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.evaluations[index]
    }
}

unsafe impl<I> Send for DenseMultilinearExtension<I> {}

unsafe impl<I> Sync for DenseMultilinearExtension<I> {}
