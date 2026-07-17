use std::ops::Range;

use crypto_primitives::{FieldConfig, ProjectElementWithConfig};
use itertools::Itertools;
use rand::prelude::*;
use zinc_poly::{mle::DenseMultilinearExtension, utils::ArithErrors};

#[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
pub(crate) fn rand_poly<C, G: Rng>(
    nv: usize,
    num_multiplicands_range: Range<usize>,
    num_products: usize,
    rng: &mut G,
    config: &C,
) -> Result<
    (
        (Vec<DenseMultilinearExtension<C::Element>>, usize),
        Vec<(C::Element, Vec<usize>)>,
        C::Element,
    ),
    ArithErrors,
>
where
    C: FieldConfig + ProjectElementWithConfig<u128>,
{
    let mut sum = config.zero();
    let mut mles: Vec<DenseMultilinearExtension<C::Element>> = vec![];
    let mut products = Vec::with_capacity(num_products);
    let mut degree = 0;
    let mut current_mle_index = 0;
    for _ in 0..num_products {
        let num_multiplicands = rng.random_range(num_multiplicands_range.clone());
        degree = num_multiplicands.max(degree);
        let (product, product_sum) = random_mle_list(nv, num_multiplicands, rng, config);

        let coefficient = config.project(&rng.random::<u128>());
        mles.extend(product);
        let term = config.mul(&product_sum, &coefficient);
        config.add_assign(&mut sum, &term);

        let indices: Vec<usize> =
            (current_mle_index..current_mle_index + num_multiplicands).collect();
        products.push((coefficient, indices));
        current_mle_index += num_multiplicands;
    }

    Ok(((mles, degree), products, sum))
}

pub fn rand_poly_comb_fn<C: FieldConfig>(
    config: &C,
    vals: &[C::Element],
    products: &[(C::Element, Vec<usize>)],
) -> C::Element {
    let mut result = config.zero();
    for (coef, indices) in products {
        let product = indices
            .iter()
            .map(|&i| &vals[i])
            .fold(config.one(), |mut acc, next| {
                config.mul_assign(&mut acc, next);
                acc
            });
        let term = config.mul(coef, &product);
        config.add_assign(&mut result, &term);
    }

    result
}

/// Sample a random list of multilinear polynomials.
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube.
pub fn random_mle_list<C, G: Rng>(
    nv: usize,
    degree: usize,
    rng: &mut G,
    config: &C,
) -> (Vec<DenseMultilinearExtension<C::Element>>, C::Element)
where
    C: FieldConfig + ProjectElementWithConfig<u128>,
{
    let mut multiplicands = (0..degree)
        .map(|_| Vec::with_capacity(1 << nv))
        .collect_vec();
    let mut sum = config.zero();

    for _ in 0..1 << nv {
        let mut product = config.one();

        for e in multiplicands.iter_mut() {
            let val = config.project(&rng.random::<u128>());
            e.push(val.clone());
            config.mul_assign(&mut product, &val);
        }
        config.add_assign(&mut sum, &product);
    }

    let list = multiplicands
        .into_iter()
        .map(|x| DenseMultilinearExtension::from_evaluations_vec(nv, x, config.zero()))
        .collect();

    (list, sum)
}
