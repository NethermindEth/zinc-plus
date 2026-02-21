#![allow(clippy::arithmetic_side_effects)]
//! A SHA-256-like UAIR with multiple column types for benchmarking the
//! end-to-end SNARK.
//!
//! The UAIR models a simplified SHA-256 compression step with:
//! - 7 binary polynomial columns (representing 32-bit words)
//! - 27 integer columns (representing sums / intermediate values)
//!
//! Constraints:
//!   * For i in 0..3: `binary_poly[i] - int[i] \u2208 (X - 2)`  (degree 1)
//!     (the first 3 binary polys are decompositions of the first 3 integers)
//!   * For i in 0..3: `binary_poly[i+3] - down.int[i] \u2208 (X - 2)`  (degree 1)
//!     (the next 3 binary polys are decompositions of the _shifted_ first 3 integers)
//!   * For i in 0..6: `int[3+3i] + int[4+3i] - int[5+3i] == 0`  (degree 1)
//!     (linear sum constraints across 6 triples in int[3..21])
//!   * For i in 0..2: `int[21+3i] * int[22+3i] - int[23+3i] == 0`  (degree 2)
//!     (quadratic product constraints across 2 triples in int[21..27])
//!   * `binary_poly[6]` is an unconstrained auxiliary column.

use crate::GenerateMultiTypeWitness;
use crypto_primitives::crypto_bigint_int::Int;
use rand::Rng;
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
            binary_poly_cols: 7,
            arbitrary_poly_cols: 0,
            int_cols: 27,
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
        for i in 0..3 {
            b.assert_in_ideal(
                up.binary_poly[i].clone() - &up.int[i],
                &modulus,
            );
        }

        // binary_poly[i+3] is the binary decomposition of the next row's int[i]
        for i in 0..3 {
            b.assert_in_ideal(
                up.binary_poly[i + 3].clone() - &down.int[i],
                &modulus,
            );
        }

        // Linear sum constraints: int[3+3i] + int[4+3i] - int[5+3i] == 0
        for i in 0..6 {
            let a = up.int[3 + 3 * i].clone();
            let b_expr = up.int[4 + 3 * i].clone();
            let c = up.int[5 + 3 * i].clone();
            b.assert_zero(a + &b_expr - &c);
        }

        // Quadratic product constraints: int[21+3i] * int[22+3i] - int[23+3i] == 0
        for i in 0..2 {
            let a = up.int[21 + 3 * i].clone();
            let b_expr = up.int[22 + 3 * i].clone();
            let c = up.int[23 + 3 * i].clone();
            b.assert_zero(a * &b_expr - &c);
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

        // Generate first 3 random integer columns (constrained by binary decomposition)
        let mut int_cols: Vec<DenseMultilinearExtension<u32>> = (0..3)
            .map(|_| DenseMultilinearExtension::rand(num_vars, rng))
            .collect();

        // Generate int[3..21] in triples satisfying int[3+3i] + int[4+3i] == int[5+3i]
        for _ in 0..6 {
            let a: DenseMultilinearExtension<u32> =
                DenseMultilinearExtension::rand(num_vars, rng);
            let b: DenseMultilinearExtension<u32> =
                DenseMultilinearExtension::rand(num_vars, rng);
            let c: DenseMultilinearExtension<u32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| x.wrapping_add(y))
                .collect();
            int_cols.push(a);
            int_cols.push(b);
            int_cols.push(c);
        }

        // Generate int[21..27] in triples satisfying int[21+3i] * int[22+3i] == int[23+3i]
        // Use u16-range values so the product fits in u32.
        for _ in 0..2 {
            let a: DenseMultilinearExtension<u32> = (0..size)
                .map(|_| (rng.random::<u16>()) as u32)
                .collect();
            let b: DenseMultilinearExtension<u32> = (0..size)
                .map(|_| (rng.random::<u16>()) as u32)
                .collect();
            let c: DenseMultilinearExtension<u32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| x * y)
                .collect();
            int_cols.push(a);
            int_cols.push(b);
            int_cols.push(c);
        }

        let mut binary_cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
            Vec::with_capacity(7);

        // First 3 binary poly columns: decomposition of int[0..3]
        for col in &int_cols[..3] {
            binary_cols.push(col.iter().map(|val| BinaryPoly::from(*val)).collect());
        }

        // Next 3 binary poly columns: decomposition of shifted int[0..3]
        // (shifted means: the value at the *next* row)
        for col in &int_cols[..3] {
            let shifted: DenseMultilinearExtension<BinaryPoly<32>> = (0..size)
                .map(|j| {
                    let next_val = if j + 1 < size { col[j + 1] } else { 0 };
                    BinaryPoly::from(next_val)
                })
                .collect();
            binary_cols.push(shifted);
        }

        // 7th binary poly column: unconstrained auxiliary (random)
        binary_cols.push(DenseMultilinearExtension::rand(num_vars, rng));

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
        assert_eq!(sig.binary_poly_cols, 7);
        assert_eq!(sig.arbitrary_poly_cols, 0);
        assert_eq!(sig.int_cols, 27);
    }

    #[test]
    fn sha256_uair_constraint_degrees() {
        assert_eq!(count_constraints::<Sha256CompressionUair<LIMBS>>(), 14);
        let mut expected_degrees = vec![1; 12];
        expected_degrees.extend([2, 2]);
        assert_eq!(count_constraint_degrees::<Sha256CompressionUair<LIMBS>>(), expected_degrees);
        assert_eq!(count_max_degree::<Sha256CompressionUair<LIMBS>>(), 2);
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
        assert_eq!(binary.len(), 7);
        assert!(arb.is_empty());
        assert_eq!(ints.len(), 27);

        // Check first 3 binary columns match ints
        for i in 0..3 {
            for j in 0..(1 << num_vars) {
                let bp_val = binary[i][j]
                    .evaluate_at_point(&2u32)
                    .expect("evaluation should succeed");
                assert_eq!(bp_val, ints[i][j], "mismatch at col {i}, row {j}");
            }
        }
    }
}

