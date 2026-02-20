#![allow(clippy::arithmetic_side_effects)]
//! A SHA-256-like UAIR with multiple column types for benchmarking the
//! end-to-end SNARK.
//!
//! The UAIR models a simplified SHA-256 compression step with:
//! - 10 binary polynomial columns (representing 32-bit words)
//! - 5 integer columns (representing sums / intermediate values)
//!
//! Constraints (all degree-1, modular):
//!   * For i in 0..5: `binary_poly[i] - int[i] ∈ (X - 2)`
//!     (the first 5 binary polys are decompositions of the 5 integers)
//!   * For i in 0..5: `binary_poly[i+5] - down.int[i] ∈ (X - 2)`
//!     (the next 5 binary polys are decompositions of the _shifted_ integers)

use crate::GenerateMultiTypeWitness;
use crypto_primitives::crypto_bigint_int::Int;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dense::DensePolynomial},
};
use zinc_uair::{
    ConstraintBuilder, TraceRow, Uair, UairSignature,
    ideal::degree_one::DegreeOneIdeal,
};

/// A multi-type UAIR that resembles a SHA-256 compression trace.
pub struct Sha256CompressionUair<const LIMBS: usize>;

impl<const LIMBS: usize> Uair for Sha256CompressionUair<LIMBS> {
    type Ideal = DegreeOneIdeal<Int<LIMBS>>;
    type Scalar = DensePolynomial<Int<LIMBS>, 64>;

    fn signature() -> UairSignature {
        UairSignature {
            binary_poly_cols: 10,
            arbitrary_poly_cols: 0,
            int_cols: 5,
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let modulus = ideal_from_ref(&DegreeOneIdeal::new(Int::from(2)));

        // binary_poly[i] is the binary decomposition of int[i]
        for i in 0..5 {
            b.assert_in_ideal(
                up.binary_poly[i].clone() - &up.int[i],
                &modulus,
            );
        }

        // binary_poly[i+5] is the binary decomposition of the next row's int[i]
        for i in 0..5 {
            b.assert_in_ideal(
                up.binary_poly[i + 5].clone() - &down.int[i],
                &modulus,
            );
        }
    }
}

impl<const LIMBS: usize> GenerateMultiTypeWitness for Sha256CompressionUair<LIMBS> {
    type PolyCoeff = u32;
    type Int = u32;

    fn generate_witness<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> (
        Vec<DenseMultilinearExtension<BinaryPoly<32>>>,
        Vec<DenseMultilinearExtension<DensePolynomial<Self::PolyCoeff, 32>>>,
        Vec<DenseMultilinearExtension<Self::Int>>,
    ) {
        use zinc_poly::mle::MultilinearExtensionRand;

        let size: usize = 1 << num_vars;

        // Generate 5 random integer columns
        let int_cols: Vec<DenseMultilinearExtension<u32>> = (0..5)
            .map(|_| DenseMultilinearExtension::rand(num_vars, rng))
            .collect();

        let mut binary_cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
            Vec::with_capacity(10);

        // First 5 binary poly columns: decomposition of int[i]
        for col in &int_cols {
            binary_cols.push(col.iter().map(|val| BinaryPoly::from(*val)).collect());
        }

        // Next 5 binary poly columns: decomposition of shifted int[i]
        // (shifted means: the value at the *next* row)
        for col in &int_cols {
            let shifted: DenseMultilinearExtension<BinaryPoly<32>> = (0..size)
                .map(|j| {
                    let next_val = if j + 1 < size { col[j + 1] } else { 0 };
                    BinaryPoly::from(next_val)
                })
                .collect();
            binary_cols.push(shifted);
        }

        (binary_cols, vec![], int_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zinc_poly::EvaluatablePolynomial;
    use zinc_uair::{
        collect_scalars::collect_scalars,
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    const LIMBS: usize = 1;

    #[test]
    fn sha256_uair_shape() {
        let sig = Sha256CompressionUair::<LIMBS>::signature();
        assert_eq!(sig.binary_poly_cols, 10);
        assert_eq!(sig.arbitrary_poly_cols, 0);
        assert_eq!(sig.int_cols, 5);
    }

    #[test]
    fn sha256_uair_constraint_degrees() {
        assert_eq!(count_constraints::<Sha256CompressionUair<LIMBS>>(), 10);
        assert_eq!(count_constraint_degrees::<Sha256CompressionUair<LIMBS>>(), vec![1; 10]);
        assert_eq!(count_max_degree::<Sha256CompressionUair<LIMBS>>(), 1);
    }

    #[test]
    fn sha256_uair_no_scalars() {
        let scalars = collect_scalars::<Sha256CompressionUair<LIMBS>>();
        assert!(scalars.is_empty());
    }

    #[test]
    fn sha256_uair_generates_valid_witness() {
        let mut rng = rand::rng();
        let num_vars = 4;
        let (binary, arb, ints) =
            Sha256CompressionUair::<LIMBS>::generate_witness(num_vars, &mut rng);
        assert_eq!(binary.len(), 10);
        assert!(arb.is_empty());
        assert_eq!(ints.len(), 5);

        // Check first 5 binary columns match ints
        for i in 0..5 {
            for j in 0..(1 << num_vars) {
                let bp_val = binary[i][j]
                    .evaluate_at_point(&2u32)
                    .expect("evaluation should succeed");
                assert_eq!(bp_val, ints[i][j], "mismatch at col {i}, row {j}");
            }
        }
    }
}

