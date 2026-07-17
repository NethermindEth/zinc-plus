pub mod binary_ref;
pub mod binary_u64;
pub mod dense;
pub mod dynamic;
pub mod nat_evaluation;

use crypto_primitives::FieldConfig;

/// Shared projection helper for binary polynomials.
/// `get_coeff` should return true if the i-th coefficient is 1.
fn prepare_projection<C, P, GetCoeff, const N: usize>(
    cfg: &C,
    sampled_value: &C::Element,
    get_coeff: GetCoeff,
) -> impl Fn(&P) -> C::Element
where
    C: FieldConfig + 'static,
    GetCoeff: Fn(&P, usize) -> bool + Send + Sync + 'static,
{
    let cfg = cfg.clone();
    let r_powers = {
        let mut r_powers = Vec::with_capacity(N);
        let mut curr = cfg.one();
        r_powers.push(curr.clone());
        for _ in 1..N {
            cfg.mul_assign(&mut curr, sampled_value);
            r_powers.push(curr.clone());
        }
        r_powers
    };
    move |poly: &P| {
        let mut acc = cfg.zero();
        for (i, r) in r_powers.iter().enumerate() {
            if get_coeff(poly, i) {
                cfg.add_assign(&mut acc, r);
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
