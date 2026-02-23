# Verifier Optimization — Agent Setup

## Problem

The PCS verifier is 3–4× over the 5 ms target for the 8×SHA-256+ECDSA benchmark:

| Config (NUM_COLUMN_OPENINGS) | Verifier time | Target |
|------------------------------|---------------|--------|
| 147 openings                 | 14.4–15.0 ms  | < 5 ms |
| 64 openings                  | 17.9–19.2 ms  | < 5 ms |

The benchmark runs **two sequential** `BatchedZipPlus::verify` calls (SHA-256 + ECDSA). With the split-SHA optimization, it's **three** (SHA BinaryPoly + SHA Int + ECDSA). All must complete in < 5 ms combined.

## Workspace

`/Users/albertgarretafontelles/dev/zinc-plus-new/`

## Architecture

```
BatchedZipPlus::verify()          ← entry point
  ├─ verify_testing()             ← testing phase
  │    ├─ sample challenges       (FS transcript, cheap)
  │    ├─ read combined_row       (deserialize, cheap)
  │    ├─ ★ encode_wide()         (PNTT over CombR integers)
  │    └─ for each of NUM_COLUMN_OPENINGS columns:
  │         ├─ verify_batched_column_testing()  (inner products)
  │         └─ ★ MerkleProof::verify()          (blake3 hash walk)
  │
  └─ verify_evaluation()          ← evaluation phase
       ├─ compute tensor (q_0, q_1)    (cheap)
       ├─ read batched_row + evals     (deserialize)
       ├─ inner_product(batched_row, q_1) check  (cheap)
       ├─ ★ encode_f()                 (PNTT over PrimeField elements)
       └─ for each of NUM_COLUMN_OPENINGS columns:
            └─ β-weighted projected column sum vs encoded row  (field muls)
```

The three starred (★) operations dominate verifier time.

## Cost Breakdown for 8×SHA-256 (PnttConfigF2_16R4B64<1>)

### Encoding (2 calls per verify)

| Operation | row_len | cw_len | Twiddle muls | Per-mul cost | Total work |
|-----------|---------|--------|--------------|-------------|------------|
| `encode_wide` (CombR=Int<6>=384-bit) | 512 | 2,048 | 143,360 | 6 limb muls | ~860K limb muls |
| `encode_f` (MontyField<4>=256-bit) | 512 | 2,048 | 143,360 | 1 Montgomery convert + 1 field mul | ~287K field muls |

**PNTT breakdown:** BASE_LEN=64, BASE_DIM=256, DEPTH=1
- Base layer: 2,048 output elements × 63 twiddle muls each = 129,024 (90% of total)
- Butterfly: 7 × 2,048 × 1 = 14,336 (10% of total)

### Column openings (147 per batch)

Each opening requires:
- `verify_batched_column_testing`: alpha-project, row-combine, sum across polys (inner products)
- `MerkleProof::verify`: `hash_column()` (serialize + blake3) + walk siblings up tree

For 20 SHA columns × 1 row × 256 B/cw = 5,120 bytes hashed per column leaf.
Merkle tree depth ≈ log₂(2,048) = 11 levels → 11 blake3 merges per proof.

### Total per verify (SHA-256 batch)

| Component | Est. time |
|-----------|-----------|
| `encode_wide` (Int<6>, 143K twiddle muls) | ~2–3 ms |
| `encode_f` (MontyField<4>, 143K field muls) | ~3–4 ms |
| 147 column openings (Merkle + inner products) | ~2–3 ms |
| Transcript deserialization + FS challenges | ~0.5 ms |
| **Total** | **~8–10 ms** |

The ECDSA batch adds another ~4–6 ms on top.

## Key Files

| File | What it does |
|------|-------------|
| `zip-plus/src/batched_pcs/phase_verify.rs` | Batched PCS verifier — the entry point and all verification logic |
| `zip-plus/src/pcs/phase_verify.rs` | Single-poly PCS verifier (same structure, not used in benchmarks) |
| `zip-plus/src/code/iprs.rs` | IPRS linear code: `encode`, `encode_wide`, `encode_f` |
| `zip-plus/src/code/iprs/pntt/radix8.rs` | Radix-8 pseudo-NTT: `pntt`, `base_multiply_into_output`, `combine_stages` |
| `zip-plus/src/code/iprs/pntt/radix8/params.rs` | PNTT configs: `PnttConfigF2_16R4B64`, `PnttConfigF2_16R4B16`, etc. |
| `zip-plus/src/code/iprs/pntt/radix8/butterfly.rs` | Radix-8 butterfly: `apply_radix_8_butterflies` |
| `zip-plus/src/code/iprs/pntt/radix8/mul_by_twiddle.rs` | Twiddle multiplication variants |
| `zip-plus/src/merkle.rs` | Merkle tree: `commit`, `open`, `verify`, `hash_column` |
| `zip-plus/src/pcs_transcript.rs` | Proof serialization/deserialization |
| `zip-plus/src/batched_pcs/structs.rs` | `BatchedZipPlus` struct, `ZipPlusParams` |
| `zip-plus/src/pcs/structs.rs` | `ZipTypes` trait, all associated types |
| `zip-plus/src/code.rs` | `LinearCode` trait |
| `utils/src/parallel.rs` | `cfg_iter!`, `cfg_into_iter!`, `cfg_chunks_mut!` macros |
| `snark/benches/e2e_sha256.rs` | Benchmarks |
| `zip-plus/Cargo.toml` | Feature flags (`parallel`, `simd`) |

## Optimization Strategies

### Strategy 1: Reduce encoding cost — the biggest lever

The verifier calls `encode_wide` and `encode_f` on the **full row** (512 elements for 8×SHA). These two calls alone account for ~60% of verifier time.

**Option 1a: Spot-check encoding instead of full encode.**
Instead of encoding the entire row and checking all opened column positions, only compute the encoding at the opened positions. The PNTT is a structured linear map — each output element is a dot product of the input row with a single row of the encoding matrix. For `O` column openings (147), compute only those `O` output values instead of all 2,048.

Cost reduction: from `cw_len × BASE_LEN` to `O × BASE_LEN` twiddle muls.
- Current: 2,048 × 63 = 129,024 (base) + 14,336 (butterfly) = 143,360
- Spot-check: 147 × 63 = 9,261 twiddle muls → **15× cheaper**

This requires extracting rows of the PNTT's encoding matrix at specific column indices. The PNTT is the composition of the base Vandermonde matrix with DEPTH butterfly stages — computing a single output element from the input requires applying the inverse butterfly path from that output position back to the base layer, then a dot product with the appropriate base matrix row. Alternatively, pre-compute and cache the full encoding matrix for the 147 opened positions.

**Option 1b: Leverage code structure for faster encoding.**
The PNTT base layer is a dense Vandermonde multiplication. For the verifier, only the evaluation at queried positions matters. This is equivalent to polynomial multi-point evaluation — there may be subquadratic algorithms.

**Option 1c: Cache the encoding matrix.**
Pre-compute the `O × row_len` submatrix of the full encoding matrix and store it in `ZipPlusParams`. Then the verifier just does `O` inner products of length `row_len` instead of a full PNTT. This trades memory for computation: 147 × 512 × 8 B ≈ 590 KB per batch (acceptable).

### Strategy 2: Reduce column-opening verification cost

**Option 2a: Batch Merkle verification.**
Currently each of the 147 column openings independently hashes the leaf and walks up the tree. Many openings share Merkle path prefixes. Batching the verification to share work on common path segments could save ~30% of Merkle hashing.

**Option 2b: Parallelize column openings more effectively.**
The column openings are already parallelized via `cfg_into_iter!` / rayon. But with only 147 items and complex per-item work, rayon's overhead may hurt. Consider:
- Using a thread pool with larger chunk sizes
- Manual batching into 4–8 chunks for the M4's 10 cores
- Removing rayon for small batches and using `std::thread::scope` directly

**Option 2c: Optimize `hash_column` serialization.**
`hash_column` allocates a fresh buffer and serializes all values for every column opening. For `BinaryPoly<32>` with 1 row, that's 20 × 256 = 5,120 bytes. Pre-allocating a reusable buffer or using `blake3::Hasher::update` incrementally without a buffer could help.

### Strategy 3: Reduce verification batches

**Option 3a: Merge SHA BinaryPoly and SHA Int into one PCS batch.**
Currently the split-SHA benchmark runs 3 verify calls. If the two SHA batches could be merged (committing Int columns as BinaryPoly<32> zero-padded), verification would be 2 calls instead of 3. This loses the proof-size benefit but speeds up the verifier.

**Option 3b: Interleave / parallelize the two (or three) verify calls.**
The SHA and ECDSA verify calls are independent. They could run in parallel using `rayon::join` or `std::thread::scope`, cutting wall-clock time by ~50% if cores are available.

### Strategy 4: Algorithmic improvements to the inner products

**Option 4a: SIMD-accelerate inner products.**
The `MBSInnerProduct` and `ScalarProduct` implementations may not use SIMD. For `MontyField<4>` (256-bit), vectorized Montgomery multiplication could help.

**Option 4b: Precompute beta powers only once.**
`beta_powers` is already precomputed in `verify_evaluation`. Ensure this pattern is used everywhere (it is).

### Strategy 5: Reduce NUM_COLUMN_OPENINGS

Fewer openings = less work per verify. But this is a security parameter — reducing it weakens soundness. Already explored: 64 vs 147 — the 64-opening config was actually *slower* (paradoxical, likely due to larger proof size causing more deserialization, or different code config). This needs careful analysis.

## Recommended Approach (Priority Order)

1. **Strategy 1c (Cache encoding matrix) or 1a (Spot-check encoding)** — biggest impact, ~50% reduction in verify time. Start here.
2. **Strategy 3b (Parallelize verify calls)** — easy 30–50% reduction for multi-batch case.
3. **Strategy 2a/2b (Batch/optimize Merkle)** — moderate impact, ~10–20%.
4. **Strategy 4a (SIMD inner products)** — moderate impact, depends on current vectorization.

With strategies 1+3, the target of < 5 ms for two batches is achievable:
- Current: ~8 ms (SHA) + ~5 ms (ECDSA) = ~13 ms sequential
- Spot-check encoding: ~3 ms (SHA) + ~2 ms (ECDSA) = ~5 ms sequential
- Parallelize batches: max(3, 2) ≈ ~3 ms

## Constraints

- Do NOT change the proof format or prover code (backward compatibility).
- Do NOT weaken security (keep NUM_COLUMN_OPENINGS unchanged).
- All existing tests must pass: `cargo test -p zip-plus`.
- Benchmark must still compile: `cargo check -p zinc-snark --bench e2e_sha256`.
- Feature flags `parallel` and `simd` must continue to work.

## Running Benchmarks

```bash
# Full benchmark suite
cargo bench -p zinc-snark --bench e2e_sha256 --features=zinc-snark/parallel,zinc-snark/simd -- "8xSHA256"

# Just the verifier
cargo bench -p zinc-snark --bench e2e_sha256 --features=zinc-snark/parallel,zinc-snark/simd -- "Verifier"

# Tests
cargo test -p zip-plus
cargo test -p zinc-snark
```

## Profiling

To profile the verifier hot path:

```bash
# Build benchmarks with debug symbols
cargo bench -p zinc-snark --bench e2e_sha256 --features=zinc-snark/parallel,zinc-snark/simd --no-run

# Find the binary
ls -la target/release/deps/e2e_sha256-*

# Run with Instruments (macOS)
xcrun xctrace record --template "Time Profiler" --launch -- target/release/deps/e2e_sha256-HASH --bench "8xSHA256+ECDSA/Combined/PCS/Verifier"
```

## Current Types Summary

### SHA-256 batch (BinaryPoly<32>)
| Type | Value | Size |
|------|-------|------|
| Eval | `BinaryPoly<32>` | 32 bytes (32 × 1-bit coefficients stored as u64) |
| Cw | `DensePolynomial<i64, 32>` | 256 bytes (32 × 8B i64 coefficients) |
| CombR | `Int<6>` | 48 bytes (384-bit) |
| Comb | `DensePolynomial<Int<6>, 32>` | 1,536 bytes |
| PcsF | `MontyField<4>` | 32 bytes (256-bit Montgomery field) |
| Fmod | `Uint<4>` | 32 bytes |

### ECDSA batch (Int<4>)
| Type | Value | Size |
|------|-------|------|
| Eval | `Int<4>` | 32 bytes (256-bit) |
| Cw | `Int<5>` | 40 bytes (320-bit) |
| CombR | `Int<8>` | 64 bytes (512-bit) |
| Comb | `Int<8>` | 64 bytes |
| PcsF | `MontyField<8>` | 64 bytes (512-bit Montgomery field) |
| Fmod | `Uint<8>` | 64 bytes |

### SHA-256 Int batch (Int<1>)
| Type | Value | Size |
|------|-------|------|
| Eval | `Int<1>` | 8 bytes (64-bit) |
| Cw | `Int<2>` | 16 bytes (128-bit) |
| CombR | `Int<4>` | 32 bytes (256-bit) |
| Comb | `Int<4>` | 32 bytes |
| PcsF | `MontyField<4>` | 32 bytes (256-bit Montgomery field) |
| Fmod | `Uint<4>` | 32 bytes |

### IPRS Code Config for 8× benchmarks: PnttConfigF2_16R4B64<1>
| Parameter | Value |
|-----------|-------|
| Field | F_{65537} |
| BASE_LEN | 64 |
| BASE_DIM | 256 |
| DEPTH | 1 |
| INPUT_LEN (row_len) | 512 |
| OUTPUT_LEN (cw_len) | 2,048 |
| Rate | 1/4 |
