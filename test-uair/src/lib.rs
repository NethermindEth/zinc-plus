#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_witness;

use crypto_primitives::{FixedSemiring, Semiring, boolean::Boolean, crypto_bigint_int::Int};
use rand::{
    Rng, RngCore,
    distr::{Distribution, StandardUniform},
};
use zinc_poly::{
    Polynomial,
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial,
        dynamic::over_fixed_semiring::DynamicPolynomialFS, ideal::DegreeOneIdeal,
    },
};
use zinc_uair::{ConstraintBuilder, Uair};
use zinc_utils::from_ref::FromRef;

pub use generate_witness::*;
use zinc_uair::ideal::ImpossibleIdeal;

pub struct TestUairSimpleMultiplication;

impl<R: Semiring + 'static> Uair<R> for TestUairSimpleMultiplication {
    type Ideal = ImpossibleIdeal; // Not used

    fn num_cols() -> usize {
        3
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        down: &[B::Expr],
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        b.assert_zero(up[0].clone() * &up[1] - &down[0]);
        b.assert_zero(up[1].clone() * &up[2] - &down[1]);
        b.assert_zero(up[0].clone() * &up[2] - &down[2]);
    }
}

impl<R, const DEGREE_PLUS_ONE: usize> GenerateWitness<DensePolynomial<R, DEGREE_PLUS_ONE>>
    for TestUairSimpleMultiplication
where
    R: FixedSemiring + 'static,
    StandardUniform: Distribution<R>,
{
    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<R, DEGREE_PLUS_ONE>>> {
        let mut a: Vec<DynamicPolynomialFS<R>> =
            vec![DynamicPolynomialFS::new(vec![rng.random::<R>()])];
        let mut b: Vec<DynamicPolynomialFS<R>> =
            vec![DynamicPolynomialFS::new(vec![R::zero(), rng.random::<R>()])];
        let mut c: Vec<DynamicPolynomialFS<R>> =
            vec![DynamicPolynomialFS::new(vec![R::zero(), rng.random::<R>()])];

        for i in 1..1 << num_vars {
            let prev_a = a[i - 1].clone();
            let prev_b = b[i - 1].clone();
            let prev_c = c[i - 1].clone();

            a.push(prev_a.clone() * &prev_b);
            b.push(prev_b * &prev_c);
            c.push(prev_a * prev_c);
        }

        vec![
            a.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(DEGREE_PLUS_ONE),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None")
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
            b.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(DEGREE_PLUS_ONE),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
            c.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(DEGREE_PLUS_ONE),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
        ]
    }
}

pub struct TestAirBinary;

impl Uair<DensePolynomial<Boolean, 32>> for TestAirBinary {
    type Ideal = ImpossibleIdeal;

    fn num_cols() -> usize {
        2
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        _down: &[B::Expr],
        from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&DensePolynomial<Boolean, 32>) -> B::Expr,
    {
        // X is the polynomial with coefficients [0, 1, 0, ..., 0]
        let x_poly = {
            let mut coeffs = [Boolean::new(false); 32];
            coeffs[1] = Boolean::new(true);
            DensePolynomial::new(coeffs)
        };
        // Constraint: up[0] * X - up[1] = 0
        b.assert_zero(up[0].clone() * &from_ref(&x_poly) - &up[1])
    }
}

impl GenerateWitness<DensePolynomial<Boolean, 32>> for TestAirBinary {
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<DensePolynomial<Boolean, 32>>> {
        // Generate random binary polynomials for column `a`
        let a: DenseMultilinearExtension<DensePolynomial<Boolean, 32>> =
            DenseMultilinearExtension::<u32>::rand(num_vars, rng)
                .into_iter()
                .map(|x| {
                    // Mask to 31 bits to ensure the highest degree is at most 30
                    // (since we need to shift left for a * X)
                    DensePolynomial::from_ref(&BinaryPoly::<32>::from(x & 0x7FFF_FFFF))
                })
                .collect();

        // Compute a * X by shifting coefficients left by one position
        // (coeff[0] <- 0, coeff[i+1] <- coeff[i])
        let a_times_x: DenseMultilinearExtension<DensePolynomial<Boolean, 32>> = a
            .iter()
            .map(|poly| {
                let mut new_coeffs = [Boolean::new(false); 32];
                // Shift coefficients: coeff[i] -> coeff[i+1]
                for i in 0..31 {
                    new_coeffs[i + 1] = poly.coeffs[i];
                }
                DensePolynomial::new(new_coeffs)
            })
            .collect();

        vec![a, a_times_x]
    }
}

#[cfg(test)]
mod tests {
    use zinc_uair::{collect_scalars::collect_scalars, constraint_counter::count_constraints};

    use super::*;

    const LIMBS: usize = 4;

    #[test]
    fn test_uair_simple_multiplication_correct_constraints_number() {
        assert_eq!(
            count_constraints::<DensePolynomial<Int<LIMBS>, 32>, TestUairSimpleMultiplication>(),
            3
        );
    }

    #[test]
    fn test_air_no_multiplication_correct_constraints_number() {
        assert_eq!(
            count_constraints::<DensePolynomial<Int<LIMBS>, 32>, TestAirNoMultiplication>(),
            1
        );
    }

    #[test]
    fn test_air_scalar_multiplications_correct_collect_scalars() {
        assert_eq!(
            collect_scalars::<DensePolynomial<Int<LIMBS>, 32>, TestAirScalarMultiplications>(),
            (vec![
                DensePolynomial::new([Int::from_i8(-1), Int::from_i8(0), Int::from_i8(1)]),
                DensePolynomial::new([
                    Int::from_i8(1),
                    Int::from_i8(2),
                    Int::from_i8(3),
                    Int::from_i8(4),
                ])
            ]
            .into_iter()
            .collect())
        );
    }
}
