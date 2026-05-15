//! Cantor basis construction for the additive FFT over
//! `GF(2^16) = F_2[X] / f` with `f(X) = X^16 + X^5 + X^3 + X^2 + 1`
//! (a standard primitive polynomial), and lifts of `GF(2^16)`
//! elements into `Z[X] / f̃` (same monomials, `{0,1} ⊂ Z` coefficients,
//! no mod-2 reduction).
//!
//! The Cantor basis `{β_0, …, β_15}` is defined recursively by
//! `β_0 = 1` and `β_i² + β_i = β_{i-1}` over `GF(2^16)`. It exists
//! whenever the extension degree is a power of two (here `16 = 2^4`),
//! because the trace map satisfies `Tr(β_{i-1}) = 0` at every step.
//! The subspaces `V_i = span_{F_2}{β_0, …, β_{i-1}}` form a flag, and
//! the associated subspace polynomials `s_i(X) = ∏_{α ∈ V_i}(X + α)`
//! are `F_2`-linearised — i.e. only monomials of degree `2^j` appear —
//! which keeps the lifted twiddle coefficients in `{0, 1} ⊂ Z`.
//!
//! Mirrors `binary_add_fft_8/basis.rs` (D = 8) but for D = 16.

use crypto_primitives::crypto_bigint_int::Int;
use num_traits::{ConstOne, ConstZero};

/// Non-leading monomial degrees of `f̃` (degrees `< 16` with
/// coefficient `1`). For `X^16 + X^5 + X^3 + X^2 + 1` these are
/// `{0, 2, 3, 5}`.
pub const F_TILDE_LOWER_DEGREES_16: [usize; 4] = [0, 2, 3, 5];

/// Extension degree of the binary field used by this code: `16`.
pub const D_16: usize = 16;

/// An element of `GF(2^16) = F_2[X] / f`, packed in a `u16`
/// (bit `i` is the coefficient of `X^i`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Gf2_16(pub u16);

impl Gf2_16 {
    pub const ZERO: Self = Gf2_16(0);
    pub const ONE: Self = Gf2_16(1);

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Gf2_16(self.0 ^ other.0)
    }

    /// Carry-less multiply followed by reduction mod `f`.
    pub fn mul(self, other: Self) -> Self {
        let a = self.0 as u64;
        let b = other.0 as u64;
        let mut prod: u64 = 0;
        for i in 0..D_16 {
            if (b >> i) & 1 != 0 {
                prod ^= a << i;
            }
        }
        // Fold each high monomial X^j (j ≥ 16) using
        // X^16 ≡ X^5 + X^3 + X^2 + 1 (mod f).
        for j in (D_16..(2 * D_16 - 1)).rev() {
            if (prod >> j) & 1 != 0 {
                let base = j - D_16;
                prod ^= 1u64 << j;
                for &shift in &F_TILDE_LOWER_DEGREES_16 {
                    prod ^= 1u64 << (base + shift);
                }
            }
        }
        Gf2_16(prod as u16)
    }

    #[inline]
    pub fn square(self) -> Self {
        self.mul(self)
    }

    /// Trace map `Tr: GF(2^16) → F_2`,
    /// `Tr(a) = a + a^2 + a^4 + ⋯ + a^(2^15)`.
    pub fn trace(self) -> bool {
        let mut acc = self;
        let mut sum = self;
        for _ in 1..D_16 {
            acc = acc.square();
            sum = sum.add(acc);
        }
        debug_assert!(
            sum.0 <= 1,
            "trace failed to collapse to F_2: got {:#06x}",
            sum.0
        );
        sum.0 & 1 != 0
    }

    /// Solve `Y² + Y = self` over `GF(2^16)`. Requires `Tr(self) = 0`;
    /// returns one of the two roots (the other is `Y + 1`).
    pub fn solve_quadratic(self) -> Option<Self> {
        if self.trace() {
            return None;
        }
        // `Y² + Y` is `F_2`-linear in `Y`; build its matrix and solve.
        let mut cols = [0u16; D_16];
        for (j, col) in cols.iter_mut().enumerate() {
            let ej = Gf2_16(1u16 << j);
            *col = ej.square().0 ^ ej.0;
        }
        gaussian_solve(&cols, self.0).map(Gf2_16)
    }

    #[inline]
    pub fn coeff(self, i: usize) -> bool {
        ((self.0 >> i) & 1) != 0
    }
}

/// Solve `M y = b` over `F_2` where `M` is given column-major as
/// `cols[j]` being the `j`-th column packed in a `u16`. Returns one
/// solution if the system is consistent.
fn gaussian_solve(cols: &[u16; D_16], b: u16) -> Option<u16> {
    // Build the augmented matrix row-major: low 16 bits = M[r][*],
    // bit 16 = b[r]. We use u32 to hold the augmented bit.
    let mut rows = [0u32; D_16];
    for (j, &col) in cols.iter().enumerate() {
        for r in 0..D_16 {
            if (col >> r) & 1 != 0 {
                rows[r] |= 1u32 << j;
            }
        }
    }
    for r in 0..D_16 {
        if (b >> r) & 1 != 0 {
            rows[r] |= 1u32 << D_16;
        }
    }

    let mut pivot_col = [None; D_16];
    let mut row = 0usize;
    let mut col = 0usize;
    while col < D_16 && row < D_16 {
        let pivot = (row..D_16).find(|&r| (rows[r] >> col) & 1 != 0);
        if let Some(p) = pivot {
            rows.swap(row, p);
            for r in 0..D_16 {
                if r != row && (rows[r] >> col) & 1 != 0 {
                    rows[r] ^= rows[row];
                }
            }
            pivot_col[row] = Some(col);
            row += 1;
        }
        col += 1;
    }

    // Inconsistency: a zero row with a nonzero augmented bit.
    for r in row..D_16 {
        if (rows[r] >> D_16) & 1 != 0 {
            return None;
        }
    }

    let mut y = 0u16;
    for r in 0..row {
        if let Some(c) = pivot_col[r] {
            if (rows[r] >> D_16) & 1 != 0 {
                y |= 1u16 << c;
            }
        }
    }
    Some(y)
}

/// Compute the Cantor basis: `β_0 = 1`, `β_i` is a root of
/// `Y² + Y = β_{i-1}`. Panics if the recursion ever lands on an
/// element with nonzero trace (would indicate a bug in `solve_quadratic`).
pub fn cantor_basis_16() -> [Gf2_16; D_16] {
    let mut basis = [Gf2_16::ZERO; D_16];
    basis[0] = Gf2_16::ONE;
    for i in 1..D_16 {
        basis[i] = basis[i - 1]
            .solve_quadratic()
            .expect("Cantor recursion: β_{i-1} should have zero trace");
    }
    basis
}

/// Lift a `Gf2_16` element to a `Z[X] / f̃` representation
/// `[Int<T>; 16]`, mapping bit `i` of the `u16` to `Int::ONE` or
/// `Int::ZERO` at coefficient `i`.
pub fn lift_to_int_array_16<const T: usize>(g: Gf2_16) -> [Int<T>; D_16] {
    let mut out = [Int::<T>::ZERO; D_16];
    for (i, slot) in out.iter_mut().enumerate() {
        if g.coeff(i) {
            *slot = Int::<T>::ONE;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_one_identities() {
        let a = Gf2_16(0x5A3C);
        assert_eq!(a.add(Gf2_16::ZERO), a);
        assert_eq!(a.mul(Gf2_16::ONE), a);
        assert_eq!(a.mul(Gf2_16::ZERO), Gf2_16::ZERO);
        assert_eq!(Gf2_16::ZERO.add(Gf2_16::ZERO), Gf2_16::ZERO);
    }

    #[test]
    fn addition_is_xor() {
        assert_eq!(Gf2_16(0b1010).add(Gf2_16(0b1100)), Gf2_16(0b0110));
        assert_eq!(Gf2_16(0xADCA).add(Gf2_16(0xADCA)), Gf2_16::ZERO);
    }

    #[test]
    fn x_times_x15_reduces_via_f() {
        // X · X^15 = X^16 ≡ X^5 + X^3 + X^2 + 1 (mod f).
        let x = Gf2_16(1 << 1);
        let x15 = Gf2_16(1 << 15);
        let expected = Gf2_16((1 << 5) | (1 << 3) | (1 << 2) | 1);
        assert_eq!(x.mul(x15), expected);
    }

    #[test]
    fn multiplication_is_commutative() {
        let a = Gf2_16(0xCA1F);
        let b = Gf2_16(0xDEAD);
        assert_eq!(a.mul(b), b.mul(a));
    }

    #[test]
    fn multiplication_is_associative() {
        let a = Gf2_16(0x1234);
        let b = Gf2_16(0x9abc);
        let c = Gf2_16(0xfeed);
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
    }

    #[test]
    fn frobenius_squares_are_consistent() {
        // (a + b)^2 = a^2 + b^2 in characteristic 2.
        let a = Gf2_16(0xa5b2);
        let b = Gf2_16(0xb017);
        assert_eq!(a.add(b).square(), a.square().add(b.square()));
    }

    #[test]
    fn trace_of_one_is_zero_for_even_degree() {
        // For GF(2^k) with k even, Tr(1) = k · 1 mod 2 = 0.
        assert!(!Gf2_16::ONE.trace());
    }

    #[test]
    fn solve_quadratic_satisfies_equation() {
        let b = Gf2_16::ONE;
        let y = b.solve_quadratic().expect("Tr(1) = 0 in GF(2^16)");
        assert_eq!(y.square().add(y), b);
    }

    #[test]
    fn solve_quadratic_rejects_nonzero_trace() {
        // Scan a sample of nonzero-trace inputs (uniformly across the
        // 16-bit space); Tr is surjective so half have Tr=1. Verify
        // that the solver refuses them.
        let mut found = false;
        for raw in (0u32..(1 << 16)).step_by(7) {
            let candidate = Gf2_16(raw as u16);
            if candidate.trace() {
                assert!(
                    candidate.solve_quadratic().is_none(),
                    "solve_quadratic accepted Tr≠0 input {:#06x}",
                    raw
                );
                found = true;
            }
        }
        assert!(found, "no nonzero-trace element found");
    }

    #[test]
    fn cantor_basis_recursion_holds() {
        let basis = cantor_basis_16();
        assert_eq!(basis[0], Gf2_16::ONE);
        for i in 1..D_16 {
            let lhs = basis[i].square().add(basis[i]);
            assert_eq!(lhs, basis[i - 1], "Cantor relation broken at i={i}");
        }
    }

    #[test]
    fn cantor_basis_is_linearly_independent() {
        let basis = cantor_basis_16();
        let mut rows: [u16; D_16] = basis.map(|b| b.0);
        let mut rank = 0;
        let mut col = 0;
        while col < D_16 && rank < D_16 {
            if let Some(p) = (rank..D_16).find(|&r| (rows[r] >> col) & 1 != 0) {
                rows.swap(rank, p);
                for r in 0..D_16 {
                    if r != rank && (rows[r] >> col) & 1 != 0 {
                        rows[r] ^= rows[rank];
                    }
                }
                rank += 1;
            }
            col += 1;
        }
        assert_eq!(rank, D_16, "Cantor basis is not linearly independent");
    }

    #[test]
    fn lift_preserves_bit_pattern() {
        let g = Gf2_16(0b1011_0101_1100_1001);
        let arr: [Int<2>; D_16] = lift_to_int_array_16(g);
        for i in 0..D_16 {
            let expected = if g.coeff(i) { Int::<2>::ONE } else { Int::<2>::ZERO };
            assert_eq!(arr[i], expected, "mismatch at bit {i}");
        }
    }
}
