#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
mod generate_witness;

pub use generate_witness::*;

use crypto_primitives::{Semiring, boolean::Boolean};
use rand::prelude::*;
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::{binary::BinaryPoly, dynamic::over_fixed_semiring::DynamicPolynomialFS},
};
use zinc_uair::{ConstraintBuilder, Uair, ideal::ImpossibleIdeal};

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

impl<const DEGREE_PLUS_ONE: usize> GenerateWitness<BinaryPoly<DEGREE_PLUS_ONE>>
    for TestUairSimpleMultiplication
{
    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>> {
        let mut a: Vec<DynamicPolynomialFS<Boolean>> =
            vec![DynamicPolynomialFS::new(vec![rng.random()])];
        let mut b: Vec<DynamicPolynomialFS<Boolean>> =
            vec![DynamicPolynomialFS::new(vec![Boolean::FALSE, rng.random()])];
        let mut c: Vec<DynamicPolynomialFS<Boolean>> =
            vec![DynamicPolynomialFS::new(vec![Boolean::FALSE, rng.random()])];

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
                    BinaryPoly::new(x.coeffs)
                })
                .collect(),
            b.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(DEGREE_PLUS_ONE),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    BinaryPoly::new(x.coeffs)
                })
                .collect(),
            c.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(DEGREE_PLUS_ONE),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    BinaryPoly::new(x.coeffs)
                })
                .collect(),
        ]
    }
}

pub struct TestAirBinary;

impl Uair<BinaryPoly<32>> for TestAirBinary {
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
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
    {
        // X is the polynomial with coefficients [0, 1, 0, ..., 0] (i.e., bit 1 is set)
        let x_poly = BinaryPoly::<32>::from(2_u32);
        // Constraint: up[0] * X - up[1] = 0
        b.assert_zero(up[0].clone() * &from_ref(&x_poly) - &up[1])
    }
}

impl GenerateWitness<BinaryPoly<32>> for TestAirBinary {
    #[allow(clippy::cast_possible_truncation)] // Intentional to fit within 32 bits
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
        // Generate random binary polynomials for column `a`
        let a: DenseMultilinearExtension<BinaryPoly<32>> =
            DenseMultilinearExtension::<u32>::rand(num_vars, rng)
                .into_iter()
                .map(|x| {
                    // Mask to 31 bits to ensure the highest degree is at most 30
                    // (since we need to shift left for a * X)
                    BinaryPoly::<32>::from(x & 0x7FFF_FFFF)
                })
                .collect();

        // Compute a * X by shifting coefficients left by one position
        // (coeff[0] <- 0, coeff[i+1] <- coeff[i])
        let a_times_x: DenseMultilinearExtension<BinaryPoly<32>> = a
            .iter()
            .map(|poly| {
                let shifted = (poly.to_u64() << 1) as u32;
                BinaryPoly::<32>::from(shifted)
            })
            .collect();

        vec![a, a_times_x]
    }
}

#[cfg(test)]
mod tests {
    use zinc_uair::{
        collect_scalars::collect_scalars, constraint_counter::count_constraints,
        degree_counter::count_max_degree,
    };

    use super::*;

    #[test]
    fn test_uair_simple_multiplication_correct_constraints_number() {
        assert_eq!(
            count_constraints::<BinaryPoly<32>, TestUairSimpleMultiplication>(),
            3
        );
    }

    #[test]
    fn test_air_binary_correct_constraints_number() {
        assert_eq!(count_constraints::<BinaryPoly<32>, TestAirBinary>(), 1);
    }

    #[test]
    fn test_uair_simple_multiplication_correct_max_degree() {
        assert_eq!(
            count_max_degree::<DensePolynomial<Int<LIMBS>, 32>, TestUairSimpleMultiplication>(),
            2
        );
    }

    #[test]
    fn test_air_no_multiplication_correct_max_degree() {
        assert_eq!(
            count_max_degree::<DensePolynomial<Int<LIMBS>, 32>, TestAirNoMultiplication>(),
            1
        );
    }

    #[test]
    fn test_air_binary_correct_collect_scalars() {
        assert_eq!(
            collect_scalars::<BinaryPoly<32>, TestAirBinary>(),
            (vec![BinaryPoly::from(2_u32)].into_iter().collect())
        );
    }
}
