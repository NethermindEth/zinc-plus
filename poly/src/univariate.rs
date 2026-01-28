use crypto_primitives::PrimeField;
use zinc_utils::from_ref::FromRef;

pub mod binary_ref;
pub mod binary_u64;
pub mod dense;
pub mod nat_evaluation;

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
        BinaryRefPolyWideningMulByScalar as BinaryPolyWideningMulByScalar,
    };
    #[cfg(feature = "simd")]
    pub use crate::univariate::binary_u64::{
        BinaryU64Poly as BinaryPoly, BinaryU64PolyInnerProduct as BinaryPolyInnerProduct,
        BinaryU64PolyWideningMulByScalar as BinaryPolyWideningMulByScalar,
    };
}
