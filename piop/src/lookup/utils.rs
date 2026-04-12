//! Lookup utilities: projected BitPoly/Word table generation,
//! Montgomery batch inversion, and multiplicity computation.
//!
//! Table generators produce the projected lookup tables over the base
//! prime field F_q. Batch inversion uses a two-phase hierarchical
//! Montgomery trick — exactly one field inversion regardless of thread
//! count, with parallel forward/backward sweeps across chunks.

use std::collections::HashMap;

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_utils::{cfg_chunk_size, cfg_chunks, cfg_chunks_mut, cfg_into_iter};

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

/// Compute batch multiplicative inverses using a two-phase Montgomery trick.
///
/// Given `values = [v_0, v_1, …, v_{n-1}]`, returns
/// `[v_0^{-1}, v_1^{-1}, …, v_{n-1}^{-1}]`.
///
/// Performs exactly one field inversion and `O(n)` multiplications,
/// even when parallelised across multiple chunks. The three phases:
///
///   1. **Forward (parallel):** each chunk computes local prefix products.
///   2. **Reduce (sequential, O(K) muls):** Montgomery trick on the K chunk
///      products → one inversion → per-chunk starting accumulators.
///   3. **Backward (parallel):** each chunk sweeps to produce inverses.
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
    let chunk_size = cfg_chunk_size!(n, 4096);

    // Phase 1: per-chunk prefix products (parallel).
    let prefixes: Vec<Vec<F>> = cfg_chunks!(values, chunk_size)
        .map(|chunk| {
            let mut prefix = Vec::with_capacity(chunk.len());
            prefix.push(chunk[0].clone());
            for i in 1..chunk.len() {
                let prev = prefix[i - 1].clone();
                prefix.push(prev * &chunk[i]);
            }
            prefix
        })
        .collect();

    // Phase 2: single global inversion over chunk products (sequential).
    let chunk_inv_accs = reduce_chunk_products(&prefixes, &one);

    // Phase 3: per-chunk backward sweep (parallel).
    let mut result = vec![F::zero_with_cfg(cfg); n];
    cfg_chunks_mut!(result, chunk_size)
        .zip(cfg_chunks!(values, chunk_size))
        .enumerate()
        .for_each(|(k, (res_chunk, val_chunk))| {
            let cn = val_chunk.len();
            let prefix = &prefixes[k];
            let mut inv_acc = chunk_inv_accs[k].clone();
            for i in (1..cn).rev() {
                res_chunk[i] = inv_acc.clone() * &prefix[i - 1];
                inv_acc *= &val_chunk[i];
            }
            res_chunk[0] = inv_acc;
        });

    result
}

/// Montgomery trick on K chunk products: one inversion, returns the
/// starting `inv_acc` for each chunk's backward sweep.
#[allow(clippy::arithmetic_side_effects)]
fn reduce_chunk_products<F: PrimeField>(prefixes: &[Vec<F>], one: &F) -> Vec<F> {
    let k = prefixes.len();
    let last = |i: usize| -> &F {
        prefixes[i]
            .last()
            .expect("chunk prefix is non-empty by construction")
    };

    if k == 1 {
        // Panics if the sole chunk product is zero (i.e. a zero input element).
        return vec![one.clone() / last(0)];
    }

    let mut global_prefix = Vec::with_capacity(k);
    global_prefix.push(last(0).clone());
    for i in 1..k {
        let prev = global_prefix[i - 1].clone();
        global_prefix.push(prev * last(i));
    }

    // Single field inversion — panics here if any input to
    // `batch_inverse` / `batch_inverse_shifted` was zero (or β − v_i = 0),
    // since the global prefix product will be zero.
    let mut inv_acc = one.clone() / &global_prefix[k - 1];
    let mut accs = vec![F::zero_with_cfg(one.cfg()); k];
    for i in (1..k).rev() {
        accs[i] = inv_acc.clone() * &global_prefix[i - 1];
        inv_acc *= last(i);
    }
    accs[0] = inv_acc;
    accs
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
                std::slice::from_raw_parts(t.inner() as *const F::Inner as *const u8, elem_size)
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
    if witness.is_empty() {
        return Some(
            (0..table_len)
                .map(|_| F::from_with_cfg(0u64, field_cfg))
                .collect(),
        );
    }

    let elem_size = std::mem::size_of::<F::Inner>();
    let chunk_size = cfg_chunk_size!(witness.len(), 4096);

    let local_counts: Vec<Vec<u64>> = cfg_chunks!(witness, chunk_size)
        .map(|chunk| {
            let mut counts = vec![0u64; table_len];
            for w in chunk {
                let bytes = unsafe {
                    std::slice::from_raw_parts(w.inner() as *const F::Inner as *const u8, elem_size)
                };
                match table_index.get(bytes) {
                    Some(&idx) => counts[idx] += 1,
                    // Empty vec: valid counts always have
                    // length table_len ≥ 1, so is_empty() detects this below.
                    None => return vec![],
                }
            }
            counts
        })
        .collect();

    let mut counts = vec![0u64; table_len];
    for local in &local_counts {
        if local.is_empty() {
            return None;
        }
        for (i, &c) in local.iter().enumerate() {
            counts[i] += c;
        }
    }

    Some(
        counts
            .into_iter()
            .map(|c| F::from_with_cfg(c, field_cfg))
            .collect(),
    )
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

/// Compute `[1/(β − v_0), 1/(β − v_1), …, 1/(β − v_{n-1})]` with
/// fused subtraction — avoids materializing a separate `β − v_i` vector.
///
/// Uses the same two-phase Montgomery trick as [`batch_inverse`]:
/// exactly one field inversion, `O(n)` multiplications, parallel
/// forward/backward sweeps.
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
    let chunk_size = cfg_chunk_size!(n, 4096);

    // Phase 1: per-chunk prefix products of (β − v_i) (parallel).
    let prefixes: Vec<Vec<F>> = cfg_chunks!(values, chunk_size)
        .map(|chunk| {
            let mut prefix = Vec::with_capacity(chunk.len());
            prefix.push(beta.clone() - &chunk[0]);
            for i in 1..chunk.len() {
                let diff = beta.clone() - &chunk[i];
                let prev = prefix[i - 1].clone();
                prefix.push(prev * &diff);
            }
            prefix
        })
        .collect();

    // Phase 2: single global inversion (sequential).
    let chunk_inv_accs = reduce_chunk_products(&prefixes, &one);

    // Phase 3: per-chunk backward sweep (parallel).
    let mut result = vec![F::zero_with_cfg(cfg); n];
    cfg_chunks_mut!(result, chunk_size)
        .zip(cfg_chunks!(values, chunk_size))
        .enumerate()
        .for_each(|(k, (res_chunk, val_chunk))| {
            let cn = val_chunk.len();
            let prefix = &prefixes[k];
            let mut inv_acc = chunk_inv_accs[k].clone();
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
            assert_eq!(v * inv, one);
        }
    }

    #[test]
    fn multiplicity_computation() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![0u32, 1, 1, 3, 0].into_iter().map(F::from).collect();
        let m = compute_multiplicities(&witness, &table, &()).unwrap();
        assert_eq!(m[0], F::from(2u32));
        assert_eq!(m[1], F::from(2u32));
        assert_eq!(m[2], F::from(0u32));
        assert_eq!(m[3], F::from(1u32));
    }

    #[test]
    fn multiplicity_rejects_invalid_witness() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![0u32, 5].into_iter().map(F::from).collect();
        assert!(compute_multiplicities(&witness, &table, &()).is_none());
    }

    #[test]
    fn multiplicity_rejects_single_invalid_entry() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![0u32, 1, 2, 3, 99].into_iter().map(F::from).collect();
        assert!(compute_multiplicities(&witness, &table, &()).is_none());
    }

    #[test]
    fn multiplicity_empty_witness() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![];
        let m = compute_multiplicities(&witness, &table, &()).unwrap();
        assert!(m.iter().all(|x| *x == F::from(0u32)));
    }

    #[test]
    fn multiplicity_all_same_entry() {
        let table: Vec<F> = (0..4u32).map(F::from).collect();
        let witness: Vec<F> = vec![2u32; 8].into_iter().map(F::from).collect();
        let m = compute_multiplicities(&witness, &table, &()).unwrap();
        assert_eq!(m[0], F::from(0u32));
        assert_eq!(m[1], F::from(0u32));
        assert_eq!(m[2], F::from(8u32));
        assert_eq!(m[3], F::from(0u32));
    }

    #[test]
    fn batch_inverse_shifted_correctness() {
        let beta = F::from(100u32);
        let values: Vec<F> = (0..5u32).map(F::from).collect();
        let result = batch_inverse_shifted(&beta, &values);

        let one = F::from(1u32);
        for (v, inv) in values.iter().zip(result.iter()) {
            assert_eq!((beta - v) * inv, one);
        }
    }

    #[test]
    fn batch_inverse_shifted_empty() {
        let beta = F::from(100u32);
        let result = batch_inverse_shifted(&beta, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn batch_inverse_empty() {
        let result = batch_inverse::<F>(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn batch_inverse_large() {
        let values: Vec<F> = (1..=8192u32).map(F::from).collect();
        let inverses = batch_inverse(&values);
        let one = F::from(1u32);
        for (v, inv) in values.iter().zip(inverses.iter()) {
            assert_eq!(v * inv, one);
        }
    }

    #[test]
    fn batch_inverse_shifted_large() {
        let beta = F::from(100_000u32);
        let values: Vec<F> = (0..8192u32).map(F::from).collect();
        let result = batch_inverse_shifted(&beta, &values);
        let one = F::from(1u32);
        for (v, inv) in values.iter().zip(result.iter()) {
            assert_eq!((beta - v) * inv, one);
        }
    }
}
