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
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};
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

/// Generate the projected `BitPoly(half_width)` sub-table for
/// decomposition.
///
/// This is the same as `generate_bitpoly_table(chunk_width, …)`.
/// Provided as a convenience for clarity.
pub fn generate_bitpoly_subtable<F: PrimeField + FromPrimitiveWithConfig>(
    chunk_width: usize,
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Vec<F> {
    generate_bitpoly_table(chunk_width, projecting_element, field_cfg)
}

/// Generate the projected `Word(chunk_width)` sub-table for
/// decomposition.
///
/// This is just `{0, 1, …, 2^{chunk_width} − 1} mod q`.
pub fn generate_word_subtable<F: PrimeField + FromPrimitiveWithConfig>(
    chunk_width: usize,
    field_cfg: &F::Config,
) -> Vec<F> {
    generate_word_table(chunk_width, field_cfg)
}

/// Compute the decomposition base for BitPoly.
///
/// For `BitPoly(w)` with `chunk_width = c` projected at `a`, the base
/// is `a^c`. The K positional bases are `[1, base, base^2, …, base^{K-1}]`
#[allow(clippy::arithmetic_side_effects)]
pub fn bitpoly_decomp_base<F: PrimeField>(chunk_width: usize, projecting_element: &F) -> F {
    let mut result = F::one_with_cfg(projecting_element.cfg());
    for _ in 0..chunk_width {
        result *= projecting_element;
    }
    result
}

/// Compute the decomposition base for Word.
///
/// For `Word(w)` with `chunk_width = c`, the base is `2^c mod q`.
/// The K positional bases are `[1, 2^c, 2^{2c}, …]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn word_decomp_base<F: PrimeField + FromPrimitiveWithConfig>(
    chunk_width: usize,
    field_cfg: &F::Config,
) -> F {
    F::from_with_cfg(1u64 << chunk_width, field_cfg)
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
pub fn build_table_index<F: PrimeField>(table: &[F]) -> TableIndex
where
    F::Inner: ConstTranscribable,
{
    table
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut bytes = vec![0u8; F::Inner::NUM_BYTES];
            t.inner().write_transcription_bytes_exact(&mut bytes);
            (bytes, i)
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
    F::Inner: ConstTranscribable,
{
    if witness.is_empty() {
        return Some(
            (0..table_len)
                .map(|_| F::from_with_cfg(0u64, field_cfg))
                .collect(),
        );
    }

    let chunk_size = cfg_chunk_size!(witness.len(), 4096);

    let local_counts: Vec<Vec<u64>> = cfg_chunks!(witness, chunk_size)
        .map(|chunk| {
            let mut counts = vec![0u64; table_len];
            let mut bytes = vec![0u8; F::Inner::NUM_BYTES];
            for w in chunk {
                w.inner().write_transcription_bytes_exact(&mut bytes);
                match table_index.get(&bytes) {
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
    F::Inner: ConstTranscribable,
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

// ── Column decomposition helpers ────────────────────────────────────────────

/// Decompose a BitPoly column into K chunks of `chunk_width` bits.
///
/// Given a column of projected field elements (each being the evaluation
/// of a binary polynomial of degree < `total_width` at projecting element
/// `a`), splits each entry into K sub-polynomial evaluations such that:
///
/// ```text
/// witness[i] = Σ_{k=0}^{K-1} decomp_bases[k] · chunks[k][i]
/// ```
///
/// where `decomp_bases[k] = a^{k·chunk_width}` and `chunks[k][i]` is
/// the evaluation of the k-th chunk of the binary polynomial.
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
    F::Inner: ConstTranscribable,
{
    let num_chunks = total_width / chunk_width;
    let mask = (1usize << chunk_width) - 1;

    // Build the full BitPoly table and index to map field elements → indices.
    let full_table =
        generate_bitpoly_table(total_width, projecting_element, projecting_element.cfg());
    let full_index = build_table_index(&full_table);

    let elem_size = std::mem::size_of::<F::Inner>();
    let mut chunks = vec![Vec::with_capacity(witness.len()); num_chunks];

    for w in witness {
        let bytes = unsafe {
            std::slice::from_raw_parts(w.inner() as *const F::Inner as *const u8, elem_size)
        };
        let &n = full_index.get(bytes)?;

        for (k, chunk) in chunks.iter_mut().take(num_chunks).enumerate() {
            let sub_idx = (n >> (k * chunk_width)) & mask;
            chunk.push(subtable[sub_idx].clone());
        }
    }

    Some(chunks)
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
    F::Inner: ConstTranscribable,
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
            std::slice::from_raw_parts(w.inner() as *const F::Inner as *const u8, elem_size)
        };
        let &n = full_index.get(bytes)?;

        for (k, chunk) in chunks.iter_mut().take(num_chunks).enumerate() {
            let chunk_val = (n >> (k * chunk_width)) & mask;
            chunk.push(F::from_with_cfg(chunk_val as u64, field_cfg));
        }
    }

    Some(chunks)
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

    // ── Subtable / decomp_base tests ────────────────────────────────

    #[test]
    fn bitpoly_subtable_matches_table() {
        let a = F::from(3u32);
        let sub = generate_bitpoly_subtable(4, &a, &());
        let full = generate_bitpoly_table(4, &a, &());
        assert_eq!(sub, full);
    }

    #[test]
    fn word_subtable_matches_table() {
        let sub = generate_word_subtable::<F>(4, &());
        let full = generate_word_table::<F>(4, &());
        assert_eq!(sub, full);
    }

    #[test]
    fn bitpoly_decomp_base_value() {
        let a = F::from(3u32);
        // chunk_width=4 → base = a^4 = 3^4 = 81
        let base = bitpoly_decomp_base(4, &a);
        assert_eq!(base, F::from(81u32));
    }

    #[test]
    fn word_decomp_base_value() {
        // chunk_width=4 → base = 2^4 = 16
        let base = word_decomp_base::<F>(4, &());
        assert_eq!(base, F::from(16u32));
    }

    // ── Decomposition tests ─────────────────────────────────────────

    /// Verify decomposition: reconstruction matches witness, chunks in
    /// subtable.
    #[allow(clippy::arithmetic_side_effects)]
    fn assert_decompose_roundtrip(witness: &[F], chunks: &[Vec<F>], bases: &[F], subtable: &[F]) {
        assert_eq!(chunks.len(), bases.len());
        assert!(chunks.iter().all(|c| c.len() == witness.len()));

        for (i, w) in witness.iter().enumerate() {
            let mut reconstructed = F::from(0u32);
            for (k, base) in bases.iter().enumerate() {
                reconstructed += &(base * chunks[k][i]);
            }
            assert_eq!(reconstructed, *w, "reconstruction failed at index {i}");
        }
        for chunk_col in chunks {
            for entry in chunk_col {
                assert!(subtable.contains(entry), "chunk entry not in subtable");
            }
        }
    }

    #[test]
    fn decompose_bitpoly_k2() {
        let a = F::from(3u32);
        let subtable = generate_bitpoly_subtable(4, &a, &());
        let full_table = generate_bitpoly_table(8, &a, &());
        let witness = vec![
            full_table[0],
            full_table[5],
            full_table[170],
            full_table[255],
        ];

        let chunks = decompose_bitpoly_column(&witness, 8, 4, &a, &subtable)
            .expect("decomposition should succeed");
        let base = bitpoly_decomp_base(4, &a);
        let bases = zinc_utils::powers(base, F::from(1u32), 2);
        assert_decompose_roundtrip(&witness, &chunks, &bases, &subtable);
    }

    #[test]
    fn decompose_bitpoly_k3() {
        let a = F::from(3u32);
        let subtable = generate_bitpoly_subtable(3, &a, &());
        let full_table = generate_bitpoly_table(9, &a, &());
        let witness = vec![
            full_table[0],
            full_table[7],
            full_table[73],
            full_table[256],
            full_table[511],
        ];

        let chunks = decompose_bitpoly_column(&witness, 9, 3, &a, &subtable)
            .expect("decomposition should succeed");
        let base = bitpoly_decomp_base(3, &a);
        let bases = zinc_utils::powers(base, F::from(1u32), 3);
        assert_decompose_roundtrip(&witness, &chunks, &bases, &subtable);
    }

    #[test]
    fn decompose_word_k2() {
        let subtable = generate_word_subtable::<F>(4, &());
        let witness: Vec<F> = [0u32, 1, 15, 16, 42, 170, 255]
            .iter()
            .map(|&v| F::from(v))
            .collect();

        let chunks =
            decompose_word_column(&witness, 8, 4, &()).expect("decomposition should succeed");
        let base = word_decomp_base::<F>(4, &());
        let bases = zinc_utils::powers(base, F::from(1u32), 2);
        assert_decompose_roundtrip(&witness, &chunks, &bases, &subtable);
    }

    #[test]
    fn decompose_word_k4() {
        let subtable = generate_word_subtable::<F>(2, &());
        let witness: Vec<F> = [0u32, 1, 63, 170, 255]
            .iter()
            .map(|&v| F::from(v))
            .collect();

        let chunks =
            decompose_word_column(&witness, 8, 2, &()).expect("decomposition should succeed");
        let base = word_decomp_base::<F>(2, &());
        let bases = zinc_utils::powers(base, F::from(1u32), 4);
        assert_decompose_roundtrip(&witness, &chunks, &bases, &subtable);
    }

    #[test]
    fn decompose_bitpoly_rejects_invalid() {
        let a = F::from(3u32);
        let subtable = generate_bitpoly_subtable(4, &a, &());
        let full_table = generate_bitpoly_table(8, &a, &());
        let mut bad = F::from(2u32);
        while full_table.contains(&bad) {
            bad += &F::from(1u32);
        }
        assert!(decompose_bitpoly_column(&[bad], 8, 4, &a, &subtable).is_none());
    }

    #[test]
    fn decompose_word_rejects_invalid() {
        assert!(decompose_word_column(&[F::from(300u32)], 8, 4, &()).is_none());
    }
}
