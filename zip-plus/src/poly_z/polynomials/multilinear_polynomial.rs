// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// Adapted for rings by Nethermind

use ark_std::{end_timer, rand::RngCore, start_timer, vec, vec::Vec};

pub use crate::poly_z::mle::{DenseMultilinearExtension, MultilinearExtension};
use crate::{
    poly::{get_batched_nv, ArithErrors, RefCounter},
    traits::Integer,
};

pub fn evaluate_no_par<I: Integer>(poly: &DenseMultilinearExtension<I>, point: &[I]) -> I {
    assert_eq!(poly.num_vars, point.len());
    fix_variables_no_par(poly, point).evaluations[0].clone()
}

fn fix_variables_no_par<I: Integer>(
    poly: &DenseMultilinearExtension<I>,
    partial_point: &[I],
) -> DenseMultilinearExtension<I> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for i in 1..dim + 1 {
        let r = partial_point[i - 1].clone();
        for b in 0..1 << (nv - i) {
            poly[b] = poly[b << 1].clone() + (poly[(b << 1) + 1].clone() - &poly[b << 1]) * &r;
        }
    }
    DenseMultilinearExtension::from_evaluations_slice(nv - dim, &poly[..1 << (nv - dim)])
}

/// merge a set of polynomials. Returns an error if the
/// polynomials do not share a same number of nvs.
pub fn merge_polynomials<I: Integer>(
    polynomials: &[RefCounter<DenseMultilinearExtension<I>>],
) -> Result<RefCounter<DenseMultilinearExtension<I>>, ArithErrors> {
    let nv = polynomials[0].num_vars();
    for poly in polynomials.iter() {
        if nv != poly.num_vars() {
            return Err(ArithErrors::InvalidParameters(
                "num_vars do not match for polynomials".into(),
            ));
        }
    }

    let merged_nv = get_batched_nv(nv, polynomials.len());
    let mut scalars = vec![];
    for poly in polynomials.iter() {
        scalars.extend_from_slice(poly.to_evaluations().as_slice());
    }
    scalars.extend_from_slice(vec![I::ZERO; (1 << merged_nv) - scalars.len()].as_ref());
    Ok(RefCounter::new(
        DenseMultilinearExtension::from_evaluations_vec(merged_nv, scalars),
    ))
}

pub fn fix_last_variables_no_par<I: Integer>(
    poly: &DenseMultilinearExtension<I>,
    partial_point: &[I],
) -> DenseMultilinearExtension<I> {
    let mut res = fix_last_variable_no_par(poly, partial_point.last().unwrap());
    for p in partial_point.iter().rev().skip(1) {
        res = fix_last_variable_no_par(&res, p);
    }
    res
}

fn fix_last_variable_no_par<I: Integer>(
    poly: &DenseMultilinearExtension<I>,
    partial_point: &I,
) -> DenseMultilinearExtension<I> {
    let nv = poly.num_vars();
    let half_len = 1 << (nv - 1);
    let mut res = vec![I::ZERO; half_len];
    for (i, e) in res.iter_mut().enumerate().take(half_len) {
        *e = poly.evaluations[i].clone()
            + partial_point.clone()
                * (poly.evaluations[i + half_len].clone() - &poly.evaluations[i]);
    }
    DenseMultilinearExtension::from_evaluations_vec(nv - 1, res)
}
