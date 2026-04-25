//! ECDSA Jacobian → affine conversion UAIR slice (F_p / EC ops —
//! third increment, standalone variant).
//!
//! Spec source: standard Jacobian → affine conversion `(X, Y, Z) →
//! (X·Z⁻², Y·Z⁻³)`.
//!
//! ## What this slice covers
//!
//! Per row, an **independent** Jacobian → affine conversion: given a
//! Jacobian point `(X, Y, Z) ∈ F_p^3` with `Z ≠ 0`, produce affine
//! `(X_aff, Y_aff)` and the inverse witness:
//!
//!     Z_inv  · Z       = 1                    (mod p)
//!     Z_inv_sq         = Z_inv²               (mod p)
//!     Z_inv_cube       = Z_inv · Z_inv_sq     (mod p)
//!     X_aff            = X · Z_inv_sq         (mod p)
//!     Y_aff            = Y · Z_inv_cube       (mod p)
//!
//! No row-to-row chaining. Used at the end-of-trace boundary in the
//! composed Shamir UAIR to extract the affine x-coordinate `R_x` for
//! the final ECDSA signature check (which the verifier does
//! off-protocol mod n).
//!
//! ## Direct F_p constraints (no quotient witnesses)
//!
//! The proving field is the secp256k1 base prime, so each F_p identity
//! is a direct constraint. `Z_inv` is a non-deterministic witness
//! whose validity is enforced by `Z · Z_inv = 1`.
//!
//! ## Soundness caveats
//!
//! - When `Z = 0` (point at infinity), no `Z_inv` exists; the
//!   constraint `Z · Z_inv = 1` is unsatisfiable, so a valid proof
//!   implies `Z ≠ 0`. The composed UAIR's accumulator should reach a
//!   non-identity final state in honest executions; if it ends at
//!   infinity (which can only happen for adversarial inputs in valid
//!   ECDSA), the proof simply fails — that's the correct behavior.
//! - `X_aff`, `Y_aff` are not range-checked into `[0, p)` — same
//!   "free representative" stance as the other F_p slices.

use core::marker::PhantomData;

use crypto_bigint::{NonZero, Odd, Uint as CbUint};
use crypto_primitives::{ConstSemiring, crypto_bigint_int::Int};
use rand::RngCore;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, TotalColumnLayout, TraceRow, Uair, UairSignature,
    UairTrace,
    ideal::ImpossibleIdeal,
};

use crate::GenerateRandomTrace;
use crate::ecdsa_doubling::{EC_FP_INT_LIMBS, EcdsaFpRing, SECP256K1_P_UINT};

// ---------------------------------------------------------------------------
// Column layout.
// ---------------------------------------------------------------------------

pub mod cols {
    /// Public column: row activator.
    pub const S_ACTIVE: usize = 0;
    pub const NUM_INT_PUB: usize = 1;

    // Witness: Jacobian input.
    pub const W_X: usize = 1;
    pub const W_Y: usize = 2;
    pub const W_Z: usize = 3;

    // Witness: inverse + powers.
    pub const W_Z_INV: usize = 4;
    pub const W_Z_INV_SQ: usize = 5;
    pub const W_Z_INV_CUBE: usize = 6;

    // Witness: affine output.
    pub const W_X_AFF: usize = 7;
    pub const W_Y_AFF: usize = 8;

    pub const NUM_INT: usize = 9;
}

/// The affine-conversion UAIR. One independent conversion per row.
#[derive(Clone, Debug)]
pub struct AffineConversionUair<R>(PhantomData<R>);

impl<R> Uair for AffineConversionUair<R>
where
    R: EcdsaFpRing,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(0, 0, cols::NUM_INT_PUB);
        UairSignature::new(total, public, vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int = up.int;
        let s_active = &int[cols::S_ACTIVE];
        let x = &int[cols::W_X];
        let y = &int[cols::W_Y];
        let z = &int[cols::W_Z];
        let z_inv = &int[cols::W_Z_INV];
        let z_inv_sq = &int[cols::W_Z_INV_SQ];
        let z_inv_cube = &int[cols::W_Z_INV_CUBE];
        let x_aff = &int[cols::W_X_AFF];
        let y_aff = &int[cols::W_Y_AFF];

        // Constant `1` as a B::Expr (for the Z·Z_inv − 1 = 0 constraint).
        let one_expr = from_ref(&const_scalar::<R>(R::ONE));

        // C1: Z·Z_inv − 1 = 0   (mod p)
        let z_zinv = z.clone() * z_inv;
        let c1_inner = z_zinv - &one_expr;
        b.assert_zero(s_active.clone() * &c1_inner);

        // C2: Z_inv_sq − Z_inv·Z_inv = 0
        let c2_inner = z_inv_sq.clone() - &(z_inv.clone() * z_inv);
        b.assert_zero(s_active.clone() * &c2_inner);

        // C3: Z_inv_cube − Z_inv·Z_inv_sq = 0
        let c3_inner = z_inv_cube.clone() - &(z_inv.clone() * z_inv_sq);
        b.assert_zero(s_active.clone() * &c3_inner);

        // C4: X_aff − X·Z_inv_sq = 0
        let c4_inner = x_aff.clone() - &(x.clone() * z_inv_sq);
        b.assert_zero(s_active.clone() * &c4_inner);

        // C5: Y_aff − Y·Z_inv_cube = 0
        let c5_inner = y_aff.clone() - &(y.clone() * z_inv_cube);
        b.assert_zero(s_active.clone() * &c5_inner);
    }
}

/// Build a constant-polynomial (degree 0) `c` as a `DensePolynomial<R, 32>`.
fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// Witness generator.
// ---------------------------------------------------------------------------

impl<R> GenerateRandomTrace<32> for AffineConversionUair<R>
where
    R: EcdsaFpRing + From<Int<EC_FP_INT_LIMBS>>,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n_rows = 1usize << num_vars;
        let mk_col = || vec![R::ZERO; n_rows];

        let s_active_col: Vec<R> = vec![R::ONE; n_rows];
        let mut x_col: Vec<R> = mk_col();
        let mut y_col: Vec<R> = mk_col();
        let mut z_col: Vec<R> = mk_col();
        let mut z_inv_col: Vec<R> = mk_col();
        let mut z_inv_sq_col: Vec<R> = mk_col();
        let mut z_inv_cube_col: Vec<R> = mk_col();
        let mut x_aff_col: Vec<R> = mk_col();
        let mut y_aff_col: Vec<R> = mk_col();

        for row in 0..n_rows {
            let x_in = rand_fp(rng);
            let y_in = rand_fp(rng);
            let z_in = rand_nonzero_fp(rng);

            let w = compute_affine_witness(&x_in, &y_in, &z_in);

            x_col[row] = R::from(uint_to_int(x_in));
            y_col[row] = R::from(uint_to_int(y_in));
            z_col[row] = R::from(uint_to_int(z_in));
            z_inv_col[row] = R::from(w.z_inv);
            z_inv_sq_col[row] = R::from(w.z_inv_sq);
            z_inv_cube_col[row] = R::from(w.z_inv_cube);
            x_aff_col[row] = R::from(w.x_aff);
            y_aff_col[row] = R::from(w.y_aff);
        }

        let to_mle = |col: Vec<R>| -> DenseMultilinearExtension<R> { col.into_iter().collect() };

        let int = vec![
            to_mle(s_active_col),
            to_mle(x_col),
            to_mle(y_col),
            to_mle(z_col),
            to_mle(z_inv_col),
            to_mle(z_inv_sq_col),
            to_mle(z_inv_cube_col),
            to_mle(x_aff_col),
            to_mle(y_aff_col),
        ];

        UairTrace {
            int: int.into(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// F_p arithmetic helpers.
// ---------------------------------------------------------------------------

fn rand_fp<Rng: RngCore + ?Sized>(rng: &mut Rng) -> CbUint<EC_FP_INT_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    let mut limbs = [0u64; EC_FP_INT_LIMBS];
    for limb in &mut limbs {
        *limb = rng.next_u64();
    }
    limbs[EC_FP_INT_LIMBS - 1] = 0;
    let raw = CbUint::<EC_FP_INT_LIMBS>::from_words(limbs);
    raw.rem_vartime(&p_nz)
}

/// Sample a non-zero F_p element. Loops until non-zero (probability of
/// hitting zero is negligible).
fn rand_nonzero_fp<Rng: RngCore + ?Sized>(rng: &mut Rng) -> CbUint<EC_FP_INT_LIMBS> {
    use crypto_bigint::Zero as _;
    loop {
        let candidate = rand_fp(rng);
        if !bool::from(candidate.is_zero()) {
            return candidate;
        }
    }
}

fn mul_mod_p(
    a: &CbUint<EC_FP_INT_LIMBS>,
    b: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = a.widening_mul(b).into();
    let p_wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = SECP256K1_P_UINT.resize();
    let p_wide_nz = NonZero::new(p_wide).expect("p is nonzero");
    let (_, rem) = wide.div_rem_vartime(&p_wide_nz);
    rem.resize()
}

fn inv_mod_p(a: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    let p_odd = Odd::new(SECP256K1_P_UINT).expect("p is odd");
    let inv_opt = a.invert_odd_mod(&p_odd);
    inv_opt.expect("a has no inverse mod p (a == 0?)")
}

fn uint_to_int(u: CbUint<EC_FP_INT_LIMBS>) -> Int<EC_FP_INT_LIMBS> {
    debug_assert!(
        u.bits() <= 64 * EC_FP_INT_LIMBS as u32 - 1,
        "uint top bit must be 0 to reinterpret as signed"
    );
    Int::new(*u.as_int())
}

// ---------------------------------------------------------------------------
// Affine-conversion witness.
// ---------------------------------------------------------------------------

struct AffineWitness {
    z_inv: Int<EC_FP_INT_LIMBS>,
    z_inv_sq: Int<EC_FP_INT_LIMBS>,
    z_inv_cube: Int<EC_FP_INT_LIMBS>,
    x_aff: Int<EC_FP_INT_LIMBS>,
    y_aff: Int<EC_FP_INT_LIMBS>,
}

fn compute_affine_witness(
    x: &CbUint<EC_FP_INT_LIMBS>,
    y: &CbUint<EC_FP_INT_LIMBS>,
    z: &CbUint<EC_FP_INT_LIMBS>,
) -> AffineWitness {
    let z_inv = inv_mod_p(z);
    let z_inv_sq = mul_mod_p(&z_inv, &z_inv);
    let z_inv_cube = mul_mod_p(&z_inv, &z_inv_sq);
    let x_aff = mul_mod_p(x, &z_inv_sq);
    let y_aff = mul_mod_p(y, &z_inv_cube);

    AffineWitness {
        z_inv: uint_to_int(z_inv),
        z_inv_sq: uint_to_int(z_inv_sq),
        z_inv_cube: uint_to_int(z_inv_cube),
        x_aff: uint_to_int(x_aff),
        y_aff: uint_to_int(y_aff),
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::ConstOne;
    use rand::rng;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    /// Sanity: 5 constraints, all at degree 3.
    #[test]
    fn affine_constraint_shape() {
        type U = AffineConversionUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 5);
        let degrees = count_constraint_degrees::<U>();
        assert_eq!(degrees, vec![3, 3, 3, 3, 3]);
        assert_eq!(count_max_degree::<U>(), 3);
    }

    /// The witness generator produces a trace where every constraint
    /// vanishes mod p per row.
    #[test]
    fn witness_satisfies_constraints_mod_p() {
        let num_vars = 2;
        let mut r = rng();
        let trace = <AffineConversionUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::
            generate_random_trace(num_vars, &mut r);
        let n_rows = 1 << num_vars;
        assert_eq!(trace.int.len(), cols::NUM_INT);

        let int_to_uint = |v: &Int<EC_FP_INT_LIMBS>| -> CbUint<EC_FP_INT_LIMBS> {
            *v.inner().as_uint()
        };

        for row in 0..n_rows {
            let s_active = trace.int[cols::S_ACTIVE][row].clone();
            assert_eq!(s_active, Int::ONE);

            let read = |c: usize| int_to_uint(&trace.int[c][row]);
            let x = read(cols::W_X);
            let y = read(cols::W_Y);
            let z = read(cols::W_Z);
            let z_inv = read(cols::W_Z_INV);
            let z_inv_sq = read(cols::W_Z_INV_SQ);
            let z_inv_cube = read(cols::W_Z_INV_CUBE);
            let x_aff = read(cols::W_X_AFF);
            let y_aff = read(cols::W_Y_AFF);

            // C1: Z · Z_inv = 1
            let one_uint: CbUint<EC_FP_INT_LIMBS> = CbUint::ONE;
            assert_eq!(mul_mod_p(&z, &z_inv), one_uint, "C1 (Z·Z_inv=1) at row {row}");

            // C2: Z_inv_sq = Z_inv²
            assert_eq!(z_inv_sq, mul_mod_p(&z_inv, &z_inv), "C2 (Z_inv²) at row {row}");

            // C3: Z_inv_cube = Z_inv · Z_inv_sq
            assert_eq!(
                z_inv_cube,
                mul_mod_p(&z_inv, &z_inv_sq),
                "C3 (Z_inv³) at row {row}",
            );

            // C4: X_aff = X · Z_inv_sq
            assert_eq!(x_aff, mul_mod_p(&x, &z_inv_sq), "C4 (X_aff) at row {row}");

            // C5: Y_aff = Y · Z_inv_cube
            assert_eq!(y_aff, mul_mod_p(&y, &z_inv_cube), "C5 (Y_aff) at row {row}");
        }
    }
}
