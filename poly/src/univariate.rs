use crypto_primitives::PrimeField;
use zinc_utils::from_ref::FromRef;

pub mod binary_ref;
pub mod binary_u64;
pub mod dense;
pub mod dynamic;
pub mod ideal;
pub mod nat_evaluation;

/// Chunk size (in bits) for table-based binary projection.
pub(crate) const PROJ_CHUNK_BITS: usize = 8;

/// Build the lookup tables for chunked binary-poly projection.
///
/// Returns `num_chunks` tables, each of size `2^chunk_width`, where
/// `tables[k][b] = Σ_{j ∈ set-bits(b)} α^{k·CHUNK + j}`.
/// Uses the recursive structure `T[n + 2^j] = T[n] + α^{start+j}`.
pub(crate) fn build_projection_tables<F: PrimeField + FromRef<F>, const N: usize>(
    sampled_value: &F,
) -> Vec<Vec<F>> {
    let field_cfg = sampled_value.cfg().clone();

    // Precompute powers α^0, α^1, …, α^{N-1}.
    let mut powers = Vec::with_capacity(N);
    let mut curr = F::one_with_cfg(&field_cfg);
    powers.push(curr.clone());
    for _ in 1..N {
        curr *= sampled_value;
        powers.push(curr.clone());
    }

    let num_chunks = (N + PROJ_CHUNK_BITS - 1) / PROJ_CHUNK_BITS;
    (0..num_chunks)
        .map(|chunk| {
            let start = chunk * PROJ_CHUNK_BITS;
            let chunk_width = std::cmp::min(PROJ_CHUNK_BITS, N - start);
            let chunk_size = 1usize << chunk_width;
            let mut table = vec![F::zero_with_cfg(&field_cfg); chunk_size];
            // Recursive build: T[n + 2^k] = T[n] + α^{start+k}
            for k in 0..chunk_width {
                let step = 1usize << k;
                for n in 0..step {
                    table[n + step] = table[n].clone() + &powers[start + k];
                }
            }
            table
        })
        .collect()
}

/// Shared projection helper for binary polynomials (table-based).
///
/// Splits the N coefficients into chunks of [`PROJ_CHUNK_BITS`] bits,
/// builds a 2^CHUNK lookup table per chunk, then evaluates each poly
/// with `num_chunks` table lookups + `num_chunks − 1` field additions
/// (instead of N conditional additions).
///
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
    let tables = build_projection_tables::<F, N>(sampled_value);

    move |poly: &P| {
        let mut acc = F::zero_with_cfg(&field_cfg);
        for (chunk_idx, table) in tables.iter().enumerate() {
            let start = chunk_idx * PROJ_CHUNK_BITS;
            let chunk_width = std::cmp::min(PROJ_CHUNK_BITS, N - start);
            let mut byte_val = 0usize;
            for bit in 0..chunk_width {
                if get_coeff(poly, start + bit) {
                    byte_val |= 1 << bit;
                }
            }
            acc += table[byte_val].clone();
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
