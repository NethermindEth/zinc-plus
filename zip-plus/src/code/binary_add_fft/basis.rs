//! Cantor basis construction for the additive FFT over
//! `GF(2^32) = F_2[X] / f` with `f(X) = X^32 + X^7 + X^3 + X^2 + 1`,
//! and lifts of `GF(2^32)` elements into `Z[X] / f̃` (same monomials,
//! `{0,1} ⊂ Z` coefficients, no mod-2 reduction).
//!
//! The Cantor basis `{β_0, …, β_31}` is defined recursively by
//! `β_0 = 1` and `β_i² + β_i = β_{i-1}` over `GF(2^32)`. It exists
//! whenever the extension degree is a power of two (here `32 = 2^5`),
//! because the trace map satisfies `Tr(β_{i-1}) = 0` at every step.
//! The subspaces `V_i = span_{F_2}{β_0, …, β_{i-1}}` form a flag, and
//! the associated subspace polynomials `s_i(X) = ∏_{α ∈ V_i}(X + α)`
//! are `F_2`-linearised — i.e. only monomials of degree `2^j` appear —
//! which keeps the lifted twiddle coefficients in `{0, 1} ⊂ Z`.

use crypto_primitives::crypto_bigint_int::Int;
use num_traits::{ConstOne, ConstZero};

/// Non-leading monomial degrees of `f̃` (degrees `< 32` with
/// coefficient `1`). For `X^32 + X^7 + X^3 + X^2 + 1` these are
/// `{0, 2, 3, 7}`.
pub const F_TILDE_LOWER_DEGREES: [usize; 4] = [0, 2, 3, 7];

/// Extension degree of the binary field used by this code: `32`.
pub const D: usize = 32;

/// An element of `GF(2^32) = F_2[X] / f`, packed in a `u32`
/// (bit `i` is the coefficient of `X^i`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Gf2_32(pub u32);

impl Gf2_32 {
    pub const ZERO: Self = Gf2_32(0);
    pub const ONE: Self = Gf2_32(1);

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Gf2_32(self.0 ^ other.0)
    }

    /// Carry-less multiply followed by reduction mod `f`.
    pub fn mul(self, other: Self) -> Self {
        let a = self.0 as u64;
        let b = other.0 as u64;
        let mut prod: u64 = 0;
        for i in 0..D {
            if (b >> i) & 1 != 0 {
                prod ^= a << i;
            }
        }
        // Fold each high monomial X^j (j ≥ 32) using
        // X^32 ≡ X^7 + X^3 + X^2 + 1 (mod f).
        for j in (D..63).rev() {
            if (prod >> j) & 1 != 0 {
                let base = j - D;
                prod ^= 1u64 << j;
                for &shift in &F_TILDE_LOWER_DEGREES {
                    prod ^= 1u64 << (base + shift);
                }
            }
        }
        Gf2_32(prod as u32)
    }

    #[inline]
    pub fn square(self) -> Self {
        self.mul(self)
    }

    /// Trace map `Tr: GF(2^32) → F_2`,
    /// `Tr(a) = a + a^2 + a^4 + ⋯ + a^(2^31)`.
    pub fn trace(self) -> bool {
        let mut acc = self;
        let mut sum = self;
        for _ in 1..D {
            acc = acc.square();
            sum = sum.add(acc);
        }
        debug_assert!(
            sum.0 <= 1,
            "trace failed to collapse to F_2: got {:#010x}",
            sum.0
        );
        sum.0 & 1 != 0
    }

    /// Solve `Y² + Y = self` over `GF(2^32)`. Requires `Tr(self) = 0`;
    /// returns one of the two roots (the other is `Y + 1`).
    pub fn solve_quadratic(self) -> Option<Self> {
        if self.trace() {
            return None;
        }
        // `Y² + Y` is `F_2`-linear in `Y`; build its matrix and solve.
        let mut cols = [0u32; D];
        for (j, col) in cols.iter_mut().enumerate() {
            let ej = Gf2_32(1u32 << j);
            *col = ej.square().0 ^ ej.0;
        }
        gaussian_solve(&cols, self.0).map(Gf2_32)
    }

    #[inline]
    pub fn coeff(self, i: usize) -> bool {
        ((self.0 >> i) & 1) != 0
    }
}

/// Solve `M y = b` over `F_2` where `M` is given column-major as
/// `cols[j]` being the `j`-th column packed in a `u32`. Returns one
/// solution if the system is consistent.
fn gaussian_solve(cols: &[u32; D], b: u32) -> Option<u32> {
    // Build the augmented matrix row-major: low 32 bits = M[r][*],
    // bit 32 = b[r].
    let mut rows = [0u64; D];
    for (j, &col) in cols.iter().enumerate() {
        for r in 0..D {
            if (col >> r) & 1 != 0 {
                rows[r] |= 1u64 << j;
            }
        }
    }
    for r in 0..D {
        if (b >> r) & 1 != 0 {
            rows[r] |= 1u64 << D;
        }
    }

    let mut pivot_col = [None; D];
    let mut row = 0usize;
    let mut col = 0usize;
    while col < D && row < D {
        let pivot = (row..D).find(|&r| (rows[r] >> col) & 1 != 0);
        if let Some(p) = pivot {
            rows.swap(row, p);
            for r in 0..D {
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
    for r in row..D {
        if (rows[r] >> D) & 1 != 0 {
            return None;
        }
    }

    let mut y = 0u32;
    for r in 0..row {
        if let Some(c) = pivot_col[r] {
            if (rows[r] >> D) & 1 != 0 {
                y |= 1u32 << c;
            }
        }
    }
    Some(y)
}

/// Compute the Cantor basis: `β_0 = 1`, `β_i` is a root of
/// `Y² + Y = β_{i-1}`. Panics if the recursion ever lands on an
/// element with nonzero trace (would indicate a bug in `solve_quadratic`).
pub fn cantor_basis() -> [Gf2_32; D] {
    let mut basis = [Gf2_32::ZERO; D];
    basis[0] = Gf2_32::ONE;
    for i in 1..D {
        basis[i] = basis[i - 1]
            .solve_quadratic()
            .expect("Cantor recursion: β_{i-1} should have zero trace");
    }
    basis
}

/// Lift a `Gf2_32` element to a `Z[X] / f̃` representation
/// `[Int<T>; 32]`, mapping bit `i` of the `u32` to `Int::ONE` or
/// `Int::ZERO` at coefficient `i`.
pub fn lift_to_int_array<const T: usize>(g: Gf2_32) -> [Int<T>; D] {
    let mut out = [Int::<T>::ZERO; D];
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
        let a = Gf2_32(0x1234_5678);
        assert_eq!(a.add(Gf2_32::ZERO), a);
        assert_eq!(a.mul(Gf2_32::ONE), a);
        assert_eq!(a.mul(Gf2_32::ZERO), Gf2_32::ZERO);
        assert_eq!(Gf2_32::ZERO.add(Gf2_32::ZERO), Gf2_32::ZERO);
    }

    #[test]
    fn addition_is_xor() {
        assert_eq!(Gf2_32(0b1010).add(Gf2_32(0b1100)), Gf2_32(0b0110));
        assert_eq!(Gf2_32(0xdead).add(Gf2_32(0xdead)), Gf2_32::ZERO);
    }

    #[test]
    fn x_times_x31_reduces_via_f() {
        // X · X^31 = X^32 ≡ X^7 + X^3 + X^2 + 1 (mod f).
        let x = Gf2_32(1 << 1);
        let x31 = Gf2_32(1 << 31);
        let expected = Gf2_32((1 << 7) | (1 << 3) | (1 << 2) | 1);
        assert_eq!(x.mul(x31), expected);
    }

    #[test]
    fn multiplication_is_commutative() {
        let a = Gf2_32(0xCAFE_BABE);
        let b = Gf2_32(0xDEAD_BEEF);
        assert_eq!(a.mul(b), b.mul(a));
    }

    #[test]
    fn multiplication_is_associative() {
        let a = Gf2_32(0x1234_5678);
        let b = Gf2_32(0x9abc_def0);
        let c = Gf2_32(0xfeed_face);
        assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
    }

    #[test]
    fn frobenius_squares_are_consistent() {
        // (a + b)^2 = a^2 + b^2 in characteristic 2.
        let a = Gf2_32(0xa55a_a55a);
        let b = Gf2_32(0xb00b_face);
        assert_eq!(a.add(b).square(), a.square().add(b.square()));
    }

    #[test]
    fn trace_of_one_is_zero_for_even_degree() {
        // For GF(2^k) with k even, Tr(1) = k · 1 mod 2 = 0.
        assert!(!Gf2_32::ONE.trace());
    }

    #[test]
    fn solve_quadratic_satisfies_equation() {
        let b = Gf2_32::ONE;
        let y = b.solve_quadratic().expect("Tr(1) = 0 in GF(2^32)");
        assert_eq!(y.square().add(y), b);
    }

    #[test]
    fn solve_quadratic_rejects_nonzero_trace() {
        // Trace is a surjective F_2-linear functional, but the
        // distribution among small u32 values is uneven enough that
        // sequential scanning is unreliable. Sample pseudorandomly
        // (deterministic seed) and break as soon as a Tr=1 element
        // is found; verify the solver refuses it.
        let mut state: u32 = 0xc0ff_ee01;
        let mut found = false;
        for _ in 0..(1u32 << 17) {
            // xorshift32
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let candidate = Gf2_32(state);
            if candidate.trace() {
                assert!(
                    candidate.solve_quadratic().is_none(),
                    "solve_quadratic accepted Tr≠0 input {:#010x}",
                    state
                );
                found = true;
                break;
            }
        }
        assert!(found, "no nonzero-trace element found in 2^17 samples");
    }

    #[test]
    fn cantor_basis_recursion_holds() {
        let basis = cantor_basis();
        assert_eq!(basis[0], Gf2_32::ONE);
        for i in 1..D {
            let lhs = basis[i].square().add(basis[i]);
            assert_eq!(lhs, basis[i - 1], "Cantor relation broken at i={i}");
        }
    }

    #[test]
    fn cantor_basis_is_linearly_independent() {
        let basis = cantor_basis();
        let mut rows: [u32; D] = basis.map(|b| b.0);
        let mut rank = 0;
        let mut col = 0;
        while col < D && rank < D {
            if let Some(p) = (rank..D).find(|&r| (rows[r] >> col) & 1 != 0) {
                rows.swap(rank, p);
                for r in 0..D {
                    if r != rank && (rows[r] >> col) & 1 != 0 {
                        rows[r] ^= rows[rank];
                    }
                }
                rank += 1;
            }
            col += 1;
        }
        assert_eq!(rank, D, "Cantor basis is not linearly independent");
    }

    #[test]
    fn lift_preserves_bit_pattern() {
        let g = Gf2_32(0b1011_0101);
        let arr: [Int<2>; D] = lift_to_int_array(g);
        for i in 0..D {
            let expected = if g.coeff(i) { Int::<2>::ONE } else { Int::<2>::ZERO };
            assert_eq!(arr[i], expected, "mismatch at bit {i}");
        }
    }
}
