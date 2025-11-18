use ark_std::{end_timer, start_timer, vec::Vec};
use crypto_bigint::Random;
use crypto_primitives::PrimeField;
use itertools::Itertools;
use rand::{Rng, RngCore};
use std::ops::Range;
use std::sync::Arc;
use zinc_poly::{mle::DenseMultilinearExtension, utils::ArithErrors};

#[allow(clippy::arithmetic_side_effects)]
#[allow(clippy::type_complexity)]
pub(crate) fn rand_poly<F: PrimeField + Random, Rn: RngCore>(
    nv: usize,
    num_multiplicands_range: Range<usize>,
    num_products: usize,
    rng: &mut Rn,
    config: &F::Config,
) -> Result<
    (
        (Vec<DenseMultilinearExtension<F>>, usize),
        Vec<(F, Vec<usize>)>,
        F,
    ),
    ArithErrors,
> {
    let mut sum = F::zero_with_cfg(config);
    let mut mles = vec![];
    let mut products = Vec::with_capacity(num_products);
    let mut degree = 0;
    let mut current_mle_index = 0;
    for _ in 0..num_products {
        let num_multiplicands = rng.random_range(num_multiplicands_range.clone());
        degree = num_multiplicands.max(degree);
        let (product, product_sum) = random_mle_list(nv, num_multiplicands, rng, config.clone());

        let coefficient = F::try_random(rng).expect("sampling coefficient failed");
        mles.extend(product);
        sum += &(product_sum * &coefficient);

        let indices: Vec<usize> =
            (current_mle_index..current_mle_index + num_multiplicands).collect();
        products.push((coefficient, indices));
        current_mle_index += num_multiplicands;
    }

    Ok(((mles, degree), products, sum))
}

pub fn rand_poly_comb_fn<F: PrimeField>(
    vals: &[F],
    products: &[(F, Vec<usize>)],
    config: F::Config,
) -> F {
    let mut result = F::zero_with_cfg(&config);
    for (coef, indices) in products {
        let term = coef.clone() * indices.iter().map(|&i| &vals[i]).product::<F>();
        result += &term;
    }

    result
}

/// Sample a random list of multilinear polynomials.
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube.
pub fn random_mle_list<F: PrimeField + Random, Rn: RngCore>(
    nv: usize,
    degree: usize,
    rng: &mut Rn,
    config: F::Config,
) -> (Vec<DenseMultilinearExtension<F>>, F) {
    let start = start_timer!(|| "sample random mle list");
    let mut multiplicands = (0..degree)
        .map(|_| Vec::with_capacity(1 << nv))
        .collect_vec();
    let mut sum = F::zero_with_cfg(&config);

    for _ in 0..1 << nv {
        let mut product = F::one_with_cfg(&config);

        for e in multiplicands.iter_mut() {
            let val = F::try_random(rng).expect("sampling random value failed");
            e.push(val.clone());
            product *= &val;
        }
        sum += &product;
    }

    let list = multiplicands
        .into_iter()
        .map(|x| DenseMultilinearExtension::from_evaluations_vec(nv, x, F::zero_with_cfg(&config)))
        .collect();

    end_timer!(start);
    (list, sum)
}
