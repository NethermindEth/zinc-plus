use crypto_primitives::PrimeField;
use num_traits::Zero;
use thiserror::Error;
use zinc_utils::cfg_iter_mut;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::mle::DenseMultilinearExtension;

/// Returns ceil(log2(x)).
/// Copied from ark-std.
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
pub const fn log2(x: usize) -> u32 {
    if x == 0 {
        0
    } else if x.is_power_of_two() {
        1usize.leading_zeros() - x.leading_zeros()
    } else {
        0usize.leading_zeros() - x.leading_zeros()
    }
}

/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(displaydoc::Display, Debug, Error)]
pub enum ArithErrors {
    /// Invalid parameters: {0}
    InvalidParameters(String),
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r<F>(
    r: &[F],
    cfg: &F::Config,
) -> Result<DenseMultilinearExtension<F>, ArithErrors>
where
    F: PrimeField,
{
    let evals = build_eq_x_r_vec(r, cfg)?;
    let mle =
        DenseMultilinearExtension::from_evaluations_vec(r.len(), evals, F::zero_with_cfg(cfg));

    Ok(mle)
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub(crate) fn build_eq_x_r_vec<F>(r: &[F], cfg: &F::Config) -> Result<Vec<F>, ArithErrors>
where
    F: PrimeField,
{
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    let mut eval = Vec::new();
    build_eq_x_r_helper(r, &mut eval, cfg)?;

    Ok(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_helper<F>(r: &[F], buf: &mut Vec<F>, cfg: &F::Config) -> Result<(), ArithErrors>
where
    F: PrimeField,
{
    if r.is_empty() {
        return Err(ArithErrors::InvalidParameters("r length is 0".into()));
    } else if r.len() == 1 {
        // initializing the buffer with [1-r_0, r_0]
        buf.push(F::one_with_cfg(cfg) - &r[0]);
        buf.push(r[0].clone());
    } else {
        build_eq_x_r_helper(&r[1..], buf, cfg)?;

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]

        let mut res = vec![F::zero_with_cfg(cfg); buf.len() << 1];
        cfg_iter_mut!(res).enumerate().for_each(|(i, val)| {
            let bi = buf[i >> 1].clone();
            let tmp = r[0].clone() * &bi;
            if (i & 1) == 0 {
                *val = bi - tmp;
            } else {
                *val = tmp;
            }
        });
        *buf = res;
    }

    Ok(())
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_inner<F>(
    r: &[F],
    cfg: &F::Config,
) -> Result<DenseMultilinearExtension<F::Inner>, ArithErrors>
where
    F: PrimeField,
    F::Inner: Zero,
{
    let evals = build_eq_x_r_inner_vec(r, cfg)?;
    let mle = DenseMultilinearExtension {
        num_vars: r.len(),
        evaluations: evals,
    };

    Ok(mle)
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
fn build_eq_x_r_inner_vec<F>(r: &[F], cfg: &F::Config) -> Result<Vec<F::Inner>, ArithErrors>
where
    F: PrimeField,
    F::Inner: Zero,
{
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    let mut eval = Vec::new();
    build_eq_x_r_inner_helper(r, &mut eval, cfg)?;

    Ok(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_inner_helper<F>(
    r: &[F],
    buf: &mut Vec<F::Inner>,
    cfg: &F::Config,
) -> Result<(), ArithErrors>
where
    F: PrimeField,
    F::Inner: Zero,
{
    let one = F::one_with_cfg(cfg);
    if r.is_empty() {
        return Err(ArithErrors::InvalidParameters("r length is 0".into()));
    } else if r.len() == 1 {
        // initializing the buffer with [1-r_0, r_0]
        buf.push((one - &r[0]).inner().clone());
        buf.push(r[0].inner().clone());
    } else {
        build_eq_x_r_inner_helper(&r[1..], buf, cfg)?;

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]

        let mut bi = F::zero_with_cfg(cfg);

        *buf = (0..(buf.len() << 1))
            .map(|i| {
                *bi.inner_mut() = buf[i >> 1].clone();
                let tmp = r[0].clone() * &bi;
                if (i & 1) == 0 {
                    (bi.clone() - &tmp).inner().clone()
                } else {
                    tmp.inner().clone()
                }
            })
            .collect();
    }

    Ok(())
}
