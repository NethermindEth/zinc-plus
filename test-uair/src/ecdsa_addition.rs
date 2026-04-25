//! ECDSA Jacobian + affine point-addition UAIR slice (F_p / EC ops —
//! second increment, standalone variant).
//!
//! Spec source: standard mixed Jacobian + affine addition formulas.
//!
//! ## What this slice covers
//!
//! Per row, an **independent** mixed Jacobian + affine addition on
//! secp256k1: given a Jacobian accumulator `(X1, Y1, Z1) ∈ F_p^3` and a
//! public affine addend `(X2, Y2) ∈ F_p^2` (treated as Z2 = 1), produce
//! the Jacobian sum `(X3, Y3, Z3)`:
//!
//!     A = X2·Z1²              C = A − X1
//!     B = Y2·Z1³              D = B − Y1
//!     E = C²
//!     F = C·E   (= C³ = H³)
//!     G = X1·E  (= X1·H²)
//!     X3 = D² − F − 2·G
//!     Y3 = D·(G − X3) − Y1·F
//!     Z3 = Z1·C
//!
//! No row-to-row chaining, no Shamir loop, no doubling — those merge
//! together in the composed UAIR (next increment). This slice exists
//! to land the addition column / constraint layout in isolation.
//!
//! ## Direct F_p constraints (no quotient witnesses)
//!
//! Same approach as `ecdsa_doubling`: the proving field is the
//! secp256k1 base prime, so each F_p identity is a direct constraint
//! over the proving field (no `+ q · p` term needed).
//!
//! ## Column-vs-degree trade-off
//!
//! We introduce 7 intermediate witness columns (Z1², Z1³, C, D, E, F,
//! G) so every constraint stays at degree ≤ 3 (1 selector × 1 product
//! of 2 trace MLEs). The fully-inlined version would push degrees to
//! ~9, which is hostile to the IC sumcheck.
//!
//! ## Soundness caveats
//!
//! - The output `(X3, Y3, Z3)` is not range-checked into `[0, p)` —
//!   same "free representative" stance as doubling.
//! - The addend `(X2, Y2)` is assumed to lie on the curve. The
//!   composed UAIR will commit to a fixed set of public addends
//!   (`O, G, Q, G+Q`), all curve-checked by the verifier off-protocol.
//! - Curve membership of the input accumulator is preserved
//!   inductively from the initial state in the composed UAIR.

use core::marker::PhantomData;

use crypto_bigint::{CheckedSub, NonZero, Uint as CbUint};
use crypto_primitives::{ConstSemiring, crypto_bigint_int::Int};
use rand::RngCore;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, TotalColumnLayout, TraceRow, Uair, UairSignature,
    UairTrace,
    ideal::ImpossibleIdeal,
};

use crate::GenerateRandomTrace;
use crate::ecdsa_doubling::{
    EC_FP_INT_LIMBS, EcdsaFpRing, SECP256K1_P_UINT,
};

// ---------------------------------------------------------------------------
// Column layout.
// ---------------------------------------------------------------------------

pub mod cols {
    /// Public column: row activator.
    pub const S_ACTIVE: usize = 0;
    /// Public addend X coordinate (verifier-supplied).
    pub const PA_X2: usize = 1;
    /// Public addend Y coordinate (verifier-supplied).
    pub const PA_Y2: usize = 2;
    pub const NUM_INT_PUB: usize = 3;

    // Witness: Jacobian accumulator input.
    pub const W_X1: usize = 3;
    pub const W_Y1: usize = 4;
    pub const W_Z1: usize = 5;

    // Witness: Jacobian output of the addition.
    pub const W_X3: usize = 6;
    pub const W_Y3: usize = 7;
    pub const W_Z3: usize = 8;

    // Witness: intermediates (Z1², Z1³, C=H, D=r, E=H², F=H³, G=X1·H²).
    pub const W_Z1_SQ: usize = 9;
    pub const W_Z1_CUBE: usize = 10;
    pub const W_C: usize = 11; // = X2·Z1² − X1 (= H)
    pub const W_D: usize = 12; // = Y2·Z1³ − Y1 (= r)
    pub const W_E: usize = 13; // = C²
    pub const W_F: usize = 14; // = C·E
    pub const W_G: usize = 15; // = X1·E

    pub const NUM_INT: usize = 16;
}

/// The addition UAIR. One independent Jacobian + affine addition per row.
#[derive(Clone, Debug)]
pub struct JacobianAdditionUair<R>(PhantomData<R>);

impl<R> Uair for JacobianAdditionUair<R>
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
        _from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int = up.int;
        let s_active = &int[cols::S_ACTIVE];
        let pa_x2 = &int[cols::PA_X2];
        let pa_y2 = &int[cols::PA_Y2];
        let x1 = &int[cols::W_X1];
        let y1 = &int[cols::W_Y1];
        let z1 = &int[cols::W_Z1];
        let x3 = &int[cols::W_X3];
        let y3 = &int[cols::W_Y3];
        let z3 = &int[cols::W_Z3];
        let z1_sq = &int[cols::W_Z1_SQ];
        let z1_cube = &int[cols::W_Z1_CUBE];
        let c = &int[cols::W_C];
        let d = &int[cols::W_D];
        let e = &int[cols::W_E];
        let f = &int[cols::W_F];
        let g = &int[cols::W_G];

        let two_scalar = const_scalar::<R>(R::from(2_u32));

        // C1: Z1_sq − Z1·Z1 = 0
        let c1_inner = z1_sq.clone() - &(z1.clone() * z1);
        b.assert_zero(s_active.clone() * &c1_inner);

        // C2: Z1_cube − Z1·Z1_sq = 0
        let c2_inner = z1_cube.clone() - &(z1.clone() * z1_sq);
        b.assert_zero(s_active.clone() * &c2_inner);

        // C3: C − (X2·Z1_sq − X1) = 0   (i.e. C + X1 − X2·Z1_sq = 0)
        let c3_inner = c.clone() + x1 - &(pa_x2.clone() * z1_sq);
        b.assert_zero(s_active.clone() * &c3_inner);

        // C4: D − (Y2·Z1_cube − Y1) = 0
        let c4_inner = d.clone() + y1 - &(pa_y2.clone() * z1_cube);
        b.assert_zero(s_active.clone() * &c4_inner);

        // C5: E − C·C = 0
        let c5_inner = e.clone() - &(c.clone() * c);
        b.assert_zero(s_active.clone() * &c5_inner);

        // C6: F − C·E = 0
        let c6_inner = f.clone() - &(c.clone() * e);
        b.assert_zero(s_active.clone() * &c6_inner);

        // C7: G − X1·E = 0
        let c7_inner = g.clone() - &(x1.clone() * e);
        b.assert_zero(s_active.clone() * &c7_inner);

        // C8: X3 − D·D + F + 2·G = 0
        let d_sq = d.clone() * d;
        let two_g = mbs(g, &two_scalar).expect("2·G overflow");
        let c8_inner = x3.clone() - &d_sq + f + &two_g;
        b.assert_zero(s_active.clone() * &c8_inner);

        // C9: Y3 − D·(G − X3) + Y1·F = 0
        let g_minus_x3 = g.clone() - x3;
        let d_times_g_minus_x3 = d.clone() * &g_minus_x3;
        let y1_f = y1.clone() * f;
        let c9_inner = y3.clone() - &d_times_g_minus_x3 + &y1_f;
        b.assert_zero(s_active.clone() * &c9_inner);

        // C10: Z3 − Z1·C = 0
        let c10_inner = z3.clone() - &(z1.clone() * c);
        b.assert_zero(s_active.clone() * &c10_inner);
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

impl<R> GenerateRandomTrace<32> for JacobianAdditionUair<R>
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

        let zero_r = R::ZERO;
        let mk_col = || vec![zero_r.clone(); n_rows];

        let mut s_active_col: Vec<R> = vec![R::ONE; n_rows];
        let mut x2_col: Vec<R> = mk_col();
        let mut y2_col: Vec<R> = mk_col();
        let mut x1_col: Vec<R> = mk_col();
        let mut y1_col: Vec<R> = mk_col();
        let mut z1_col: Vec<R> = mk_col();
        let mut x3_col: Vec<R> = mk_col();
        let mut y3_col: Vec<R> = mk_col();
        let mut z3_col: Vec<R> = mk_col();
        let mut z1_sq_col: Vec<R> = mk_col();
        let mut z1_cube_col: Vec<R> = mk_col();
        let mut c_col: Vec<R> = mk_col();
        let mut d_col: Vec<R> = mk_col();
        let mut e_col: Vec<R> = mk_col();
        let mut f_col: Vec<R> = mk_col();
        let mut g_col: Vec<R> = mk_col();

        // Suppress unused warning at the cost of having one in s_active_col
        // (we already initialized it via vec! above).
        let _ = &mut s_active_col;

        for row in 0..n_rows {
            let x1_in = rand_fp(rng);
            let y1_in = rand_fp(rng);
            let z1_in = rand_fp(rng);
            let x2_in = rand_fp(rng);
            let y2_in = rand_fp(rng);

            let w = compute_addition_witness(&x1_in, &y1_in, &z1_in, &x2_in, &y2_in);

            x1_col[row] = R::from(uint_to_int(x1_in));
            y1_col[row] = R::from(uint_to_int(y1_in));
            z1_col[row] = R::from(uint_to_int(z1_in));
            x2_col[row] = R::from(uint_to_int(x2_in));
            y2_col[row] = R::from(uint_to_int(y2_in));
            x3_col[row] = R::from(w.x3);
            y3_col[row] = R::from(w.y3);
            z3_col[row] = R::from(w.z3);
            z1_sq_col[row] = R::from(w.z1_sq);
            z1_cube_col[row] = R::from(w.z1_cube);
            c_col[row] = R::from(w.c);
            d_col[row] = R::from(w.d);
            e_col[row] = R::from(w.e);
            f_col[row] = R::from(w.f);
            g_col[row] = R::from(w.g);
        }

        let to_mle = |col: Vec<R>| -> DenseMultilinearExtension<R> { col.into_iter().collect() };

        let int = vec![
            to_mle(s_active_col),
            to_mle(x2_col),
            to_mle(y2_col),
            to_mle(x1_col),
            to_mle(y1_col),
            to_mle(z1_col),
            to_mle(x3_col),
            to_mle(y3_col),
            to_mle(z3_col),
            to_mle(z1_sq_col),
            to_mle(z1_cube_col),
            to_mle(c_col),
            to_mle(d_col),
            to_mle(e_col),
            to_mle(f_col),
            to_mle(g_col),
        ];

        UairTrace {
            int: int.into(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// F_p arithmetic helpers (re-using the doubling slice's primitives).
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

fn small_mul_mod_p(a: &CbUint<EC_FP_INT_LIMBS>, k: u32) -> CbUint<EC_FP_INT_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    let mut acc = CbUint::<EC_FP_INT_LIMBS>::ZERO;
    for _ in 0..k {
        acc = acc.wrapping_add(a);
        if p_geq(&acc) {
            acc = acc.rem_vartime(&p_nz);
        }
    }
    acc
}

#[inline]
fn p_geq(a: &CbUint<EC_FP_INT_LIMBS>) -> bool {
    a.checked_sub(&SECP256K1_P_UINT).is_some().into()
}

fn sub_mod_p(
    a: &CbUint<EC_FP_INT_LIMBS>,
    b: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    if a.checked_sub(b).is_some().into() {
        a.wrapping_sub(b).rem_vartime(&p_nz)
    } else {
        let a_plus_p = a.wrapping_add(&SECP256K1_P_UINT);
        a_plus_p.wrapping_sub(b).rem_vartime(&p_nz)
    }
}

fn uint_to_int(u: CbUint<EC_FP_INT_LIMBS>) -> Int<EC_FP_INT_LIMBS> {
    debug_assert!(
        u.bits() <= 64 * EC_FP_INT_LIMBS as u32 - 1,
        "uint top bit must be 0 to reinterpret as signed"
    );
    Int::new(*u.as_int())
}

// ---------------------------------------------------------------------------
// Addition witness.
// ---------------------------------------------------------------------------

struct AdditionWitness {
    x3: Int<EC_FP_INT_LIMBS>,
    y3: Int<EC_FP_INT_LIMBS>,
    z3: Int<EC_FP_INT_LIMBS>,
    z1_sq: Int<EC_FP_INT_LIMBS>,
    z1_cube: Int<EC_FP_INT_LIMBS>,
    c: Int<EC_FP_INT_LIMBS>,
    d: Int<EC_FP_INT_LIMBS>,
    e: Int<EC_FP_INT_LIMBS>,
    f: Int<EC_FP_INT_LIMBS>,
    g: Int<EC_FP_INT_LIMBS>,
}

fn compute_addition_witness(
    x1: &CbUint<EC_FP_INT_LIMBS>,
    y1: &CbUint<EC_FP_INT_LIMBS>,
    z1: &CbUint<EC_FP_INT_LIMBS>,
    x2: &CbUint<EC_FP_INT_LIMBS>,
    y2: &CbUint<EC_FP_INT_LIMBS>,
) -> AdditionWitness {
    let z1_sq = mul_mod_p(z1, z1);
    let z1_cube = mul_mod_p(z1, &z1_sq);
    let a = mul_mod_p(x2, &z1_sq);
    let b = mul_mod_p(y2, &z1_cube);
    let c = sub_mod_p(&a, x1);
    let d = sub_mod_p(&b, y1);
    let e = mul_mod_p(&c, &c);
    let f = mul_mod_p(&c, &e);
    let g = mul_mod_p(x1, &e);

    // X3 = D² − F − 2·G
    let d_sq = mul_mod_p(&d, &d);
    let two_g = small_mul_mod_p(&g, 2);
    let x3 = sub_mod_p(&sub_mod_p(&d_sq, &f), &two_g);

    // Y3 = D·(G − X3) − Y1·F
    let g_minus_x3 = sub_mod_p(&g, &x3);
    let d_times = mul_mod_p(&d, &g_minus_x3);
    let y1_f = mul_mod_p(y1, &f);
    let y3 = sub_mod_p(&d_times, &y1_f);

    // Z3 = Z1·C
    let z3 = mul_mod_p(z1, &c);

    AdditionWitness {
        x3: uint_to_int(x3),
        y3: uint_to_int(y3),
        z3: uint_to_int(z3),
        z1_sq: uint_to_int(z1_sq),
        z1_cube: uint_to_int(z1_cube),
        c: uint_to_int(c),
        d: uint_to_int(d),
        e: uint_to_int(e),
        f: uint_to_int(f),
        g: uint_to_int(g),
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

    /// Sanity: 10 constraints, all at degree 3.
    #[test]
    fn addition_constraint_shape() {
        type U = JacobianAdditionUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 10);
        let degrees = count_constraint_degrees::<U>();
        assert_eq!(degrees, vec![3, 3, 3, 3, 3, 3, 3, 3, 3, 3]);
        assert_eq!(count_max_degree::<U>(), 3);
    }

    /// The witness generator produces a trace where every constraint
    /// vanishes mod p per row.
    #[test]
    fn witness_satisfies_constraints_mod_p() {
        let num_vars = 2;
        let mut r = rng();
        let trace = <JacobianAdditionUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::
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
            let x2 = read(cols::PA_X2);
            let y2 = read(cols::PA_Y2);
            let x1 = read(cols::W_X1);
            let y1 = read(cols::W_Y1);
            let z1 = read(cols::W_Z1);
            let x3 = read(cols::W_X3);
            let y3 = read(cols::W_Y3);
            let z3 = read(cols::W_Z3);
            let z1_sq = read(cols::W_Z1_SQ);
            let z1_cube = read(cols::W_Z1_CUBE);
            let c = read(cols::W_C);
            let d = read(cols::W_D);
            let e = read(cols::W_E);
            let f = read(cols::W_F);
            let g = read(cols::W_G);

            // Intermediate identities.
            assert_eq!(z1_sq, mul_mod_p(&z1, &z1), "Z1² at row {row}");
            assert_eq!(z1_cube, mul_mod_p(&z1, &z1_sq), "Z1³ at row {row}");

            let a_val = mul_mod_p(&x2, &z1_sq);
            assert_eq!(c, sub_mod_p(&a_val, &x1), "C = A−X1 at row {row}");

            let b_val = mul_mod_p(&y2, &z1_cube);
            assert_eq!(d, sub_mod_p(&b_val, &y1), "D = B−Y1 at row {row}");

            assert_eq!(e, mul_mod_p(&c, &c), "E = C² at row {row}");
            assert_eq!(f, mul_mod_p(&c, &e), "F = C·E at row {row}");
            assert_eq!(g, mul_mod_p(&x1, &e), "G = X1·E at row {row}");

            // Output identities.
            let d_sq = mul_mod_p(&d, &d);
            let two_g = small_mul_mod_p(&g, 2);
            let expected_x3 = sub_mod_p(&sub_mod_p(&d_sq, &f), &two_g);
            assert_eq!(x3, expected_x3, "X3 at row {row}");

            let g_minus_x3 = sub_mod_p(&g, &x3);
            let d_times = mul_mod_p(&d, &g_minus_x3);
            let y1_f = mul_mod_p(&y1, &f);
            let expected_y3 = sub_mod_p(&d_times, &y1_f);
            assert_eq!(y3, expected_y3, "Y3 at row {row}");

            assert_eq!(z3, mul_mod_p(&z1, &c), "Z3 = Z1·C at row {row}");
        }
    }
}
