//! Lookup table generation for projected BitPoly and Word tables.
//!
//! These functions generate the projected lookup tables over the base
//! prime field F_q, after the Zinc+ projection pipeline has mapped
//! the original algebraic types to field elements.

use std::collections::HashMap;

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_utils::cfg_into_iter;

/// A precomputed table index mapping byte representations of table entries
/// to their position. Build once with [`build_table_index`] and reuse
/// across multiple [`compute_multiplicities_with_index`] calls.
pub type TableIndex = HashMap<Vec<u8>, usize>;

/// Generate the projected `BitPoly(w)` table over F_q.
///
/// For `BitPoly(w)`, the table has `2^w` entries. Each entry corresponds
/// to a binary polynomial `b_{w-1} X^{w-1} + … + b_0` evaluated at the
/// projecting element `a`:
///
/// ```text
/// T[n] = Σ_{k=0}^{w-1} bit_k(n) · a^k
/// ```
///
/// Uses the recursive structure `T[n + 2^k] = T[n] + a^k` for
/// efficient computation.
#[allow(clippy::arithmetic_side_effects)]
pub fn generate_bitpoly_table<F: PrimeField + FromPrimitiveWithConfig>(
    width: usize,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Vec<F> {
    let size = 1usize << width;
    let mut table = vec![F::zero_with_cfg(field_cfg); size];

    // Precompute powers of `a`: a^0, a^1, ..., a^{w-1}.
    let one = F::one_with_cfg(field_cfg);
    let mut powers_of_a = Vec::with_capacity(width);
    let mut current = one;
    for _ in 0..width {
        powers_of_a.push(current.clone());
        current *= projecting_element;
    }

    // Build the table recursively: T[n + 2^k] = T[n] + a^k.
    // Start with T[0] = 0.
    // For each bit position k, the entries with bit k set are
    // obtained by adding a^k to the entries without bit k set.
    for (k, power) in powers_of_a.iter().enumerate() {
        let step = 1usize << k;
        for n in 0..step {
            table[n + step] = table[n].clone() + power;
        }
    }

    table
}

/// Generate the projected `Word(w)` table over F_q.
///
/// For `Word(w)`, the table has `2^w` entries:
/// `{0, 1, …, 2^w − 1} mod q`.
#[allow(clippy::arithmetic_side_effects)]
pub fn generate_word_table<F: PrimeField + FromPrimitiveWithConfig + Send + Sync>(
    width: usize,
    field_cfg: &F::Config,
) -> Vec<F>
where
    F::Config: Sync,
{
    let size = 1usize << width;
    cfg_into_iter!(0..size)
        .map(|i| F::from_with_cfg(i as u64, field_cfg))
        .collect()
}

/// Generate the projected `BitPoly(half_width)` sub-table for
/// decomposition.
///
/// This is the same as `generate_bitpoly_table(half_width, …)`.
/// Provided as a convenience for clarity.
pub fn generate_bitpoly_subtable<F: PrimeField + FromPrimitiveWithConfig>(
    half_width: usize,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Vec<F> {
    generate_bitpoly_table(half_width, projecting_element, field_cfg)
}

/// Generate the projected `Word(half_width)` sub-table for
/// decomposition.
///
/// This is just `{0, 1, …, 2^{half_width} − 1} mod q`.
pub fn generate_word_subtable<F: PrimeField + FromPrimitiveWithConfig>(
    half_width: usize,
    field_cfg: &F::Config,
) -> Vec<F> {
    generate_word_table(half_width, field_cfg)
}

/// Compute the shift factor for BitPoly decomposition.
///
/// For `BitPoly(2k)` projected at `a`, the shift is `a^k`.
#[allow(clippy::arithmetic_side_effects)]
pub fn bitpoly_shift<F: PrimeField>(
    half_width: usize,
    projecting_element: &F,
) -> F {
    let mut result = F::one_with_cfg(projecting_element.cfg());
    for _ in 0..half_width {
        result *= projecting_element;
    }
    result
}

/// Compute the shift factor for Word decomposition.
///
/// For `Word(2k)`, the shift is `2^k mod q`.
#[allow(clippy::arithmetic_side_effects)]
pub fn word_shift<F: PrimeField + FromPrimitiveWithConfig>(
    half_width: usize,
    field_cfg: &F::Config,
) -> F {
    F::from_with_cfg(1u64 << half_width, field_cfg)
}

/// Compute batch multiplicative inverses using Montgomery's trick.
///
/// Given `values = [v_0, v_1, …, v_{n-1}]`, returns
/// `[v_0^{-1}, v_1^{-1}, …, v_{n-1}^{-1}]`.
///
/// Uses `O(n)` multiplications and a single inversion.
///
/// # Panics
///
/// Panics if any element is zero (non-invertible).
#[allow(clippy::arithmetic_side_effects)]
pub fn batch_inverse<F: PrimeField + Send + Sync>(values: &[F]) -> Vec<F>
where
    F::Config: Sync,
{
    if values.is_empty() {
        return Vec::new();
    }

    let n = values.len();
    let cfg = values[0].cfg();
    let one = F::one_with_cfg(cfg);

    // For small inputs, use the simple sequential algorithm.
    #[cfg(feature = "parallel")]
    const PARALLEL_THRESHOLD: usize = 4096;
    #[cfg(feature = "parallel")]
    if n >= PARALLEL_THRESHOLD {
        return batch_inverse_parallel(values);
    }

    // Sequential Montgomery's trick.
    let mut prefix = Vec::with_capacity(n);
    prefix.push(values[0].clone());
    for i in 1..n {
        let prev = prefix[i - 1].clone();
        prefix.push(prev * &values[i]);
    }

    let mut inv_acc = one / &prefix[n - 1];

    let mut result = vec![F::zero_with_cfg(cfg); n];
    for i in (1..n).rev() {
        result[i] = inv_acc.clone() * &prefix[i - 1];
        inv_acc *= &values[i];
    }
    result[0] = inv_acc;

    result
}

/// Parallel batch inverse using chunked Montgomery's trick.
///
/// Splits the input into chunks, inverts each chunk independently in
/// parallel. Each chunk does one field inversion + O(chunk_size) muls.
#[cfg(feature = "parallel")]
#[allow(clippy::arithmetic_side_effects)]
fn batch_inverse_parallel<F: PrimeField + Send + Sync>(values: &[F]) -> Vec<F>
where
    F::Config: Sync,
{
    let n = values.len();
    let cfg = values[0].cfg();
    let one = F::one_with_cfg(cfg);
    let num_threads = rayon::current_num_threads();
    let chunk_size = n.div_ceil(num_threads);

    let mut result = vec![F::zero_with_cfg(cfg); n];
    result
        .par_chunks_mut(chunk_size)
        .zip(values.par_chunks(chunk_size))
        .for_each(|(res_chunk, val_chunk)| {
            let cn = val_chunk.len();
            if cn == 0 {
                return;
            }
            // Local prefix products
            let mut prefix = Vec::with_capacity(cn);
            prefix.push(val_chunk[0].clone());
            for i in 1..cn {
                let prev = prefix[i - 1].clone();
                prefix.push(prev * &val_chunk[i]);
            }
            // Single inversion per chunk
            let mut inv_acc = one.clone() / &prefix[cn - 1];
            // Backward pass
            for i in (1..cn).rev() {
                res_chunk[i] = inv_acc.clone() * &prefix[i - 1];
                inv_acc *= &val_chunk[i];
            }
            res_chunk[0] = inv_acc;
        });

    result
}

/// Build a reusable table index mapping each table entry's byte
/// representation to its position.
///
/// Use with [`compute_multiplicities_with_index`] to avoid rebuilding
/// the hash map when computing multiplicities for multiple witnesses
/// against the same table.
pub fn build_table_index<F: PrimeField>(table: &[F]) -> TableIndex {
    let elem_size = std::mem::size_of::<F::Inner>();
    table
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    t.inner() as *const F::Inner as *const u8,
                    elem_size,
                )
            };
            (bytes.to_vec(), i)
        })
        .collect()
}

/// Compute multiplicity vector using a prebuilt [`TableIndex`].
///
/// Returns `m` such that `m[j]` is the number of occurrences of `T[j]`
/// in `witness`.
///
/// # Errors
///
/// Returns `None` if any witness entry is not found in the table.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_multiplicities_with_index<F: PrimeField + FromPrimitiveWithConfig + Send + Sync>(
    witness: &[F],
    table_index: &TableIndex,
    table_len: usize,
    field_cfg: &F::Config,
) -> Option<Vec<F>>
where
    F::Config: Sync,
{
    let elem_size = std::mem::size_of::<F::Inner>();

    #[cfg(feature = "parallel")]
    {
        use std::sync::atomic::{AtomicU64, Ordering};

        let counts: Vec<AtomicU64> = (0..table_len)
            .map(|_| AtomicU64::new(0))
            .collect();

        let all_valid = witness.par_iter().all(|w| {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    w.inner() as *const F::Inner as *const u8,
                    elem_size,
                )
            };
            if let Some(&idx) = table_index.get(bytes) {
                counts[idx].fetch_add(1, Ordering::Relaxed);
                true
            } else {
                false
            }
        });

        if !all_valid {
            return None;
        }

        Some(
            cfg_into_iter!(counts)
                .map(|c| F::from_with_cfg(c.into_inner(), field_cfg))
                .collect(),
        )
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut counts = vec![0u64; table_len];

        for w in witness {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    w.inner() as *const F::Inner as *const u8,
                    elem_size,
                )
            };
            let idx = table_index.get(bytes)?;
            counts[*idx] += 1;
        }

        Some(
            counts
                .into_iter()
                .map(|c| F::from_with_cfg(c, field_cfg))
                .collect(),
        )
    }
}

/// Compute multiplicity vector: for each table entry, count how many
/// times it appears in the witness.
///
/// Returns `m` such that `m[j]` is the number of occurrences of `T[j]`
/// in `witness`.
///
/// # Errors
///
/// Returns `None` if any witness entry is not found in the table.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_multiplicities<F: PrimeField + FromPrimitiveWithConfig + Send + Sync>(
    witness: &[F],
    table: &[F],
    field_cfg: &F::Config,
) -> Option<Vec<F>>
where
    F::Config: Sync,
{
    let index = build_table_index(table);
    compute_multiplicities_with_index(witness, &index, table.len(), field_cfg)
}

/// Compute `[1/(β − v_0), 1/(β − v_1), …, 1/(β − v_{n-1})]` without
/// materializing the intermediate `β − v_i` vector.
///
/// This fuses the subtraction and Montgomery batch-inverse into a
/// single pass, halving memory traffic compared to the two-step
/// `collect(β − v_i)` then `batch_inverse` approach.
///
/// # Panics
///
/// Panics if any `β − v_i` is zero.
#[allow(clippy::arithmetic_side_effects)]
pub fn batch_inverse_shifted<F: PrimeField + Send + Sync>(beta: &F, values: &[F]) -> Vec<F>
where
    F::Config: Sync,
{
    if values.is_empty() {
        return Vec::new();
    }

    let n = values.len();
    let cfg = beta.cfg();
    let one = F::one_with_cfg(cfg);

    #[cfg(feature = "parallel")]
    const PARALLEL_THRESHOLD: usize = 4096;
    #[cfg(feature = "parallel")]
    if n >= PARALLEL_THRESHOLD {
        return batch_inverse_shifted_parallel(beta, values);
    }

    // Sequential: build prefix products of (β − v_i) in-place.
    let mut prefix = Vec::with_capacity(n);
    prefix.push(beta.clone() - &values[0]);
    for i in 1..n {
        let diff = beta.clone() - &values[i];
        let prev = prefix[i - 1].clone();
        prefix.push(prev * &diff);
    }

    let mut inv_acc = one / &prefix[n - 1];

    let mut result = vec![F::zero_with_cfg(cfg); n];
    for i in (1..n).rev() {
        result[i] = inv_acc.clone() * &prefix[i - 1];
        let diff = beta.clone() - &values[i];
        inv_acc *= &diff;
    }
    result[0] = inv_acc;

    result
}

// ── Column decomposition helpers ────────────────────────────────────────────

/// Decompose a BitPoly column into K chunks of `chunk_width` bits.
///
/// Given a column of projected field elements (each being the evaluation
/// of a binary polynomial of degree < `total_width` at projecting element
/// `a`), splits each entry into K sub-polynomial evaluations such that:
///
/// ```text
/// witness[i] = Σ_{k=0}^{K-1} shifts[k] · chunks[k][i]
/// ```
///
/// where `shifts[k] = a^{k·chunk_width}` and `chunks[k][i]` is the
/// evaluation of the k-th chunk of the binary polynomial.
///
/// # Arguments
///
/// - `witness`: projected column entries (field elements).
/// - `total_width`: the full width (e.g. 32 for `BitPoly(32)`).
/// - `chunk_width`: the width of each sub-table (e.g. 16).
/// - `projecting_element`: the element `a` used for projection.
/// - `subtable`: the projected BitPoly(chunk_width) sub-table.
///
/// # Returns
///
/// A vector of K chunk vectors, where K = total_width / chunk_width.
/// Each chunk[k][i] is the projected evaluation of the k-th sub-polynomial.
///
/// Returns `None` if any witness entry is not found in the full
/// BitPoly(total_width) table.
#[allow(clippy::arithmetic_side_effects)]
pub fn decompose_bitpoly_column<F: PrimeField + FromPrimitiveWithConfig + Send + Sync>(
    witness: &[F],
    total_width: usize,
    chunk_width: usize,
    projecting_element: &F,
    subtable: &[F],
) -> Option<Vec<Vec<F>>>
where
    F::Config: Sync,
{
    let num_chunks = total_width / chunk_width;
    let mask = (1usize << chunk_width) - 1;

    // Build the full BitPoly table and index to map field elements → indices.
    let full_table = generate_bitpoly_table(total_width, projecting_element, projecting_element.cfg());
    let full_index = build_table_index(&full_table);

    let elem_size = std::mem::size_of::<F::Inner>();
    let mut chunks = vec![Vec::with_capacity(witness.len()); num_chunks];

    for w in witness {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                w.inner() as *const F::Inner as *const u8,
                elem_size,
            )
        };
        let &n = full_index.get(bytes)?;

        for k in 0..num_chunks {
            let sub_idx = (n >> (k * chunk_width)) & mask;
            chunks[k].push(subtable[sub_idx].clone());
        }
    }

    Some(chunks)
}

/// Decompose a column using precomputed raw integer indices.
///
/// Each entry's full-table index is provided directly in `raw_indices`,
/// avoiding the need to build the full `2^total_width` table for
/// reverse-mapping.  This is critical for large widths (e.g. 32) where
/// the full table would be `2^32` entries ≈ 128 GB.
///
/// # Arguments
///
/// - `raw_indices`: for each row, the integer index in `[0, 2^total_width)`.
/// - `total_width`: the full lookup width (e.g. 32).
/// - `chunk_width`: the width of each sub-table chunk (e.g. 8).
/// - `subtable`: the projected sub-table of size `2^chunk_width`.
///
/// # Returns
///
/// A vector of K = `total_width / chunk_width` chunk vectors.
#[allow(clippy::arithmetic_side_effects)]
pub fn decompose_raw_indices_to_chunks<F: Clone + Send + Sync>(
    raw_indices: &[usize],
    total_width: usize,
    chunk_width: usize,
    subtable: &[F],
) -> Vec<Vec<F>> {
    let num_chunks = total_width / chunk_width;
    let mask = (1usize << chunk_width) - 1;
    // Build each chunk independently — enables per-chunk parallelism.
    cfg_into_iter!(0..num_chunks)
        .map(|k| {
            raw_indices
                .iter()
                .map(|&idx| {
                    let sub_idx = (idx >> (k * chunk_width)) & mask;
                    subtable[sub_idx].clone()
                })
                .collect()
        })
        .collect()
}

/// Decompose a Word column into K chunks of `chunk_width` bits.
///
/// Given a column of projected field elements (each being an integer in
/// [0, 2^total_width)), splits each entry into K chunks such that:
///
/// ```text
/// witness[i] = Σ_{k=0}^{K-1} 2^{k·chunk_width} · chunks[k][i]
/// ```
///
/// where `chunks[k][i] = (witness_int >> k*chunk_width) & (2^chunk_width - 1)`.
///
/// # Implementation
///
/// Since `PrimeField` doesn't expose a direct `to_u64()`, we reverse-look up
/// each witness entry in the Word(total_width) full table to find its integer
/// index, then extract chunks from that index.
///
/// # Returns
///
/// A vector of K chunk vectors. Returns `None` if any witness entry is not
/// a valid Word(total_width) (i.e. not in [0, 2^total_width)).
#[allow(clippy::arithmetic_side_effects)]
pub fn decompose_word_column<F: PrimeField + FromPrimitiveWithConfig + Send + Sync>(
    witness: &[F],
    total_width: usize,
    chunk_width: usize,
    field_cfg: &F::Config,
) -> Option<Vec<Vec<F>>>
where
    F::Config: Sync,
{
    let num_chunks = total_width / chunk_width;
    let mask = (1usize << chunk_width) - 1;

    // Build the full Word table and its index to map field elements → integers.
    let full_table: Vec<F> = generate_word_table(total_width, field_cfg);
    let full_index = build_table_index(&full_table);

    let elem_size = std::mem::size_of::<F::Inner>();
    let mut chunks = vec![Vec::with_capacity(witness.len()); num_chunks];

    for w in witness {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                w.inner() as *const F::Inner as *const u8,
                elem_size,
            )
        };
        let &n = full_index.get(bytes)?;

        for k in 0..num_chunks {
            let chunk_val = (n >> (k * chunk_width)) & mask;
            chunks[k].push(F::from_with_cfg(chunk_val as u64, field_cfg));
        }
    }

    Some(chunks)
}

/// Parallel variant of [`batch_inverse_shifted`].
#[cfg(feature = "parallel")]
#[allow(clippy::arithmetic_side_effects)]
fn batch_inverse_shifted_parallel<F: PrimeField + Send + Sync>(beta: &F, values: &[F]) -> Vec<F>
where
    F::Config: Sync,
{
    let n = values.len();
    let cfg = beta.cfg();
    let one = F::one_with_cfg(cfg);
    let num_threads = rayon::current_num_threads();
    let chunk_size = n.div_ceil(num_threads);

    let mut result = vec![F::zero_with_cfg(cfg); n];
    result
        .par_chunks_mut(chunk_size)
        .zip(values.par_chunks(chunk_size))
        .for_each(|(res_chunk, val_chunk)| {
            let cn = val_chunk.len();
            if cn == 0 {
                return;
            }
            let mut prefix = Vec::with_capacity(cn);
            prefix.push(beta.clone() - &val_chunk[0]);
            for i in 1..cn {
                let diff = beta.clone() - &val_chunk[i];
                let prev = prefix[i - 1].clone();
                prefix.push(prev * &diff);
            }
            let mut inv_acc = one.clone() / &prefix[cn - 1];
            for i in (1..cn).rev() {
                res_chunk[i] = inv_acc.clone() * &prefix[i - 1];
                let diff = beta.clone() - &val_chunk[i];
                inv_acc *= &diff;
            }
            res_chunk[0] = inv_acc;
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    #[test]
    fn bitpoly_table_size_and_values() {
        let a = F::from(3u32);
        let table = generate_bitpoly_table(4, &a, &());

        assert_eq!(table.len(), 16); // 2^4

        // T[0] = 0
        assert_eq!(table[0], F::from(0u32));
        // T[1] = a^0 = 1
        assert_eq!(table[1], F::from(1u32));
        // T[2] = a^1 = 3
        assert_eq!(table[2], F::from(3u32));
        // T[3] = a^0 + a^1 = 1 + 3 = 4
        assert_eq!(table[3], F::from(4u32));
        // T[5] = a^0 + a^2 = 1 + 9 = 10
        assert_eq!(table[5], F::from(10u32));
    }

    #[test]
    fn word_table_values() {
        let table = generate_word_table::<F>(4, &());

        assert_eq!(table.len(), 16);
        for i in 0..16u32 {
            assert_eq!(table[i as usize], F::from(i));
        }
    }

    #[test]
    fn batch_inverse_correctness() {
        let values: Vec<F> = (1..=5u32).map(F::from).collect();
        let inverses = batch_inverse(&values);

        let one = F::from(1u32);
        for (v, inv) in values.iter().zip(inverses.iter()) {
            assert_eq!(v.clone() * inv, one);
        }
    }

    #[test]
    fn multiplicity_computation() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        // witness = [0, 1, 1, 3, 0]
        let witness: Vec<F> = vec![0u32, 1, 1, 3, 0]
            .into_iter()
            .map(F::from)
            .collect();
        let m = compute_multiplicities(&witness, &table, &()).unwrap();
        // m[0]=2, m[1]=2, m[2]=0, m[3]=1
        assert_eq!(m[0], F::from(2u32));
        assert_eq!(m[1], F::from(2u32));
        assert_eq!(m[2], F::from(0u32));
        assert_eq!(m[3], F::from(1u32));
    }
}
