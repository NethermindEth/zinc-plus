use crypto_primitives::PrimeField;
use zinc_utils::from_ref::FromRef;

pub mod binary_f2_wide;
pub mod binary_ref;
pub mod binary_u64;
pub mod dense;
pub mod dynamic;
pub mod nat_evaluation;

/// In-place `F_2` addition for `F_2[X]`-typed values. Unlike `AddAssign`
/// on the binary-poly types — which models the coefficients as the
/// integers `{0, 1}` and panics in debug mode on `1 + 1` because the
/// sum overflows `{0, 1}` — `F2AddAssign` performs the modular
/// `F_2`-style XOR with no overflow notion (`1 + 1 = 0`).
///
/// Use this trait wherever the intended arithmetic is `F_2` (e.g. an
/// RAA encoder operating over `F_2[X]/<X^D>`).
pub trait F2AddAssign {
    fn f2_add_assign(&mut self, rhs: &Self);
}

/// Shared projection helper for binary polynomials.
/// `get_coeff` should return true if the i-th coefficient is 1.
fn prepare_projection<F, P, GetCoeff, const N: usize>(
    sampled_value: &F,
    get_coeff: GetCoeff,
) -> impl Fn(&P) -> F + 'static
where
    F: PrimeField + FromRef<F> + 'static,
    GetCoeff: Fn(&P, usize) -> bool + 'static,
{
    let field_cfg = sampled_value.cfg().clone();
    let r_powers = {
        let mut r_powers = Vec::with_capacity(N);
        let mut curr = F::one_with_cfg(&field_cfg);
        r_powers.push(curr.clone());
        for _ in 1..N {
            curr *= sampled_value;
            r_powers.push(curr.clone());
        }
        r_powers
    };
    move |poly: &P| {
        let mut acc = F::zero_with_cfg(&field_cfg);
        for (i, r) in r_powers.iter().enumerate() {
            if get_coeff(poly, i) {
                acc += r.clone();
            }
        }
        acc
    }
}
pub mod binary {
    #[cfg(not(feature = "simd"))]
    pub use crate::univariate::binary_ref::{
        BinaryRefPoly as BinaryPoly, BinaryRefPolyInnerProduct as BinaryPolyInnerProduct,
    };
    #[cfg(feature = "simd")]
    pub use crate::univariate::binary_u64::{
        BinaryU64Poly as BinaryPoly, BinaryU64PolyInnerProduct as BinaryPolyInnerProduct,
    };
}
