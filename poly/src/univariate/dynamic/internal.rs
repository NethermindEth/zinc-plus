use std::{
    fmt::Display,
    ops::{AddAssign, Neg, SubAssign},
};

use crypto_primitives::Semiring;
use itertools::Itertools;

use crate::univariate::dynamic::multiplication::mul_schoolbook;

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn new_coeffs_trimmed<R: Clone>(coeffs: &[R], is_zero: impl Fn(&R) -> bool) -> Vec<R> {
    if let Some((non_zero, _)) = coeffs.iter().rev().find_position(|&coeff| !is_zero(coeff)) {
        let deg_plus_one = coeffs.len() - non_zero;

        coeffs.iter().take(deg_plus_one).cloned().collect()
    } else {
        Vec::new()
    }
}

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn degree<R>(coeffs: &[R], is_zero: impl Fn(&R) -> bool) -> Option<usize> {
    coeffs
        .iter()
        .rev()
        .find_position(|coeff| !is_zero(coeff))
        .map(|(non_zero, _)| coeffs.len() - non_zero - 1)
}

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn trim<R>(coeffs: &mut Vec<R>, is_zero: impl Fn(&R) -> bool) {
    coeffs.truncate(degree(coeffs, is_zero).map_or(0, |degree| degree + 1))
}

pub(crate) fn display<R: Display>(
    coeffs: &[R],
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    write!(f, "[")?;
    let mut first = true;

    for coeff in coeffs.iter() {
        if first {
            first = false;
        } else {
            write!(f, ", ")?;
        }
        write!(f, "{}", coeff)?;
    }

    write!(f, "]")
}

pub(crate) fn is_zero<R>(coeffs: &[R], is_zero: impl Fn(&R) -> bool) -> bool {
    coeffs.iter().all(is_zero)
}

pub(crate) fn neg<R: Neg<Output = R>>(coeffs: Vec<R>) -> Vec<R> {
    coeffs.into_iter().map(|coeff| coeff.neg()).collect()
}

macro_rules! impl_assign_op {
    ($name:ident, $trait:ident) => {
        #[allow(clippy::arithmetic_side_effects)] // by definition
        pub(crate) fn $name<R: for<'a> $trait<&'a R> + Clone>(
            lhs: &mut Vec<R>,
            rhs: &[R],
            zero_from_elem: impl Fn(&R) -> R,
        ) {
            if lhs.len() < rhs.len() {
                lhs.resize(rhs.len(), zero_from_elem(&rhs[0]));
            }

            lhs.iter_mut().zip(rhs).for_each(|(lhs, rhs)| {
                lhs.$name(rhs);
            });
        }
    };
}

impl_assign_op!(add_assign, AddAssign);
impl_assign_op!(sub_assign, SubAssign);

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn mul<R: Semiring, const CHECK: bool>(
    lhs: &[R],
    rhs: &[R],
    is_zero_elem: impl Fn(&R) -> bool,
) -> Option<Vec<R>> {
    if is_zero(lhs, &is_zero_elem) || is_zero(rhs, is_zero_elem) {
        return Some(Vec::new());
    }

    let degree = (lhs.len() - 1) + (rhs.len() - 1);
    let mut coeffs = Vec::with_capacity(degree + 1);

    mul_schoolbook::<_, CHECK>(lhs, rhs, coeffs.spare_capacity_mut())?;

    // Safety: the multiplication algorithm should fill in the entire spare
    // capacity.
    unsafe {
        coeffs.set_len(degree + 1);
    }

    Some(coeffs)
}
