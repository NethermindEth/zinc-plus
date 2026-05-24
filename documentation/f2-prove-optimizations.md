# F_2 Prove optimizations (`claude/gpu-merkle` branch)

Performance work on the SHA-256 F_2 prover, measured against
`cargo bench -p zinc-protocol --bench f2_sha256 --features
parallel,simd,unchecked,metal_gpu -- "Zinc\+ F_2 SHA-256/"`
on an **Apple M4** (10 perf cores, 10 GPU cores).

## TL;DR

| metric (nvars=16) | before branch | after branch | speedup |
|---|---:|---:|---:|
| `Commit` (full headline) | 29.4 ms | 18.2 ms | **1.61×** |
| `UAIR-b-AlphaProject` (micro) | 24.9 ms | 1.5 ms | **16×** |
| `UAIR-d-ColEvalsAtRstar` | 3.85 ms | 1.43 ms | **2.7×** |
| `Open-d-CombinedRow` (micro) | 8.5 ms | 5.2 ms | **1.63×** |
| **e2e `Prove`** (best of 3) | **~44 ms** | **~34.8 ms** | **~1.27×** |

Two commits on the branch:

- **d7d650e** — Metal GPU Blake3 commit + AlphaProject/Open-d speedups
- **8c27941** — UAIR-d parallelisation

## Methodology

The bench layout has three Criterion groups (driven by
[`protocol/benches/f2_sha256.rs`](../protocol/benches/f2_sha256.rs)):

- **`Zinc+ F_2 SHA-256`** — end-to-end `WitnessGen` / `Prove` / `Verify`
  at `nvars ∈ {9, 16, 20}`. The default invocation runs these.
- **`Zinc+ F_2 SHA-256 Steps`** — top-level prover/verifier step breakdown
  (`1-Commit` / `2-UAIR` / `3-Open`) at `nvars=9`.
- **`Zinc+ F_2 SHA-256 Micro`** — fine-grained breakdown of each prover
  sub-step at `nvars=16`. This is where we read off the per-step cost of
  the work the headline `Prove` does.

For each sub-step the Micro group has a dedicated entry. The Micro sum at
`nvars=16` should reconcile with the headline `Prove` time (and after
this branch's work, it does — sum 35.5 ms vs Prove ~35 ms median).

---

## 1. Metal GPU Blake3 leaf-hash for the commit Merkle

### Bottleneck

Before any work, `Micro/Commit-Fused` at `nvars=16` was **35.0 ms**, of
which roughly 2 ms is the encode pass and **~33 ms** is the row-major
fused leaf-hash + tree build. The leaf-hash itself is the Blake3 kernel
running over `num_leaves × leaf_bytes` of codeword data — 4 096 leaves
× ~21 KB each. CPU Blake3 on Apple Silicon hits ~3 GB/s, so for 86 MB
of leaf data we'd expect ~30 ms — exactly what we measured.

The bolt-rs repo (`albertgarreta/bolt-rs`) already ships a Metal Blake3
kernel for this exact shape. Importing it under a feature gate is the
biggest lever.

### What changed

**New files:**

- [`zip-plus/src/metal_gpu/mod.rs`](../zip-plus/src/metal_gpu/mod.rs) —
  `MetalContext` singleton + `hash_columns_gpu(GpuHashKind::Blake3, base,
  num_cols, col_byte_len)` dispatch. Adapted from
  `bolt-rs/src/metal_gpu/mod.rs`, stripped to just the Blake3 pipeline
  (the other hash families and the expander/RAA/FFT kernels are not
  needed). Persistent scratch buffers + warmup carried over.
- [`zip-plus/src/metal_gpu/shaders/hash_kernels.metal`](../zip-plus/src/metal_gpu/shaders/hash_kernels.metal)
  — Blake3-only kernel (~250 lines). Single- and multi-chunk paths up
  to 64 KB leaves; CPU fallback handles anything larger.

**Modified:**

- [`zip-plus/Cargo.toml`](../zip-plus/Cargo.toml) — new `metal_gpu`
  feature gating optional `metal`/`objc`/`block` deps (macOS-only).
- [`zip-plus/src/lib.rs`](../zip-plus/src/lib.rs) — `pub mod metal_gpu`
  under `#[cfg(all(feature = "metal_gpu", target_os = "macos"))]`.
- [`zip-plus/src/merkle.rs`](../zip-plus/src/merkle.rs) — two new
  builders:
  - `MerkleTree::new_from_row_major_grouped_gpu` — packs the canonical
    leaf-byte layout into a contiguous slab via rayon (one leaf per
    chunk), then dispatches one Metal Blake3 call over the whole slab.
    Produces the same Merkle root as the CPU path (verified by a
    round-trip test).
  - `MerkleTree::new_from_packed_slab_gpu` — same but takes a
    pre-packed slab (used by the inline-pack path below).
- [`zip-plus/src/pcs/phase_commit.rs`](../zip-plus/src/pcs/phase_commit.rs)
  — `commit_grouped` dispatches to the GPU builder under two guards:
  - `num_leaves >= 256` — below this the GPU launch overhead
    (~500 µs warm) dominates the CPU work. At `nvars=9` we have
    32 leaves and CPU finishes in 190 µs total, vs 610 µs for GPU.
  - `leaf_bytes <= 64 KB` — kernel multi-chunk path limit.

### Inline-pack refinement

The first GPU build did the pack as a dedicated pass over `cw_matrices`
(reading every cell a second time after the encode). That was ~5 ms of
the GPU path. The fix: scatter into the slab **inline inside the encode
par_iter**, so each rayon worker writes its matrix's contribution while
the data is still hot in its core's L1/L2. Different `m_idx` values
write to disjoint byte sub-ranges within every leaf, so concurrent
scatter is race-free under the `GpuSlabPtr` contract.

### Performance

| `Micro` sub-bench | time |
|---|---:|
| `Commit-Fused` (CPU baseline) | 35.0 ms |
| `Commit-Fused-GPU` | 10.0 ms |
| `Commit-Fused-GPU-Inline` | **7.55 ms** |

`Micro/Commit` (the full F_2 protocol commit including transcript
absorb): **29.4 → 18.2 ms** (1.61×).

### Correctness

Two new tests pin the GPU path against the CPU reference:

- `zip_plus::metal_gpu::tests::gpu_blake3_matches_cpu_for_various_sizes`
  — GPU leaf hash matches `blake3::hash` byte-for-byte across leaf sizes
  spanning the single-chunk and multi-chunk kernel branches.
- `zip_plus::merkle::tests::gpu_row_major_grouped_root_matches_cpu`
  — full GPU-built Merkle root matches the CPU root for four
  representative shapes including one similar to the SHA-256 F_2
  paired batch count.
- `zip_plus::merkle::tests::gpu_inline_packed_root_matches_cpu`
  — same parity for the inline-pack variant.

---

## 2. AlphaProject: per-cell `eval_f2_poly_d_at_with_powers`

### Bottleneck

`Micro/UAIR-b-AlphaProject` was **24.9 ms** at `nvars=16` — *bigger
than the commit Merkle phase*. The hot path is at
[`protocol/src/f2_prove.rs:830-858`](../protocol/src/f2_prove.rs#L830-L858):

```rust
let alpha_pows = alpha_powers(&alpha, D);            // D = 32
let projected_trace = cfg_iter!(extended_trace.binary_poly)
    .map(|col| col.evaluations.iter()
        .map(|cell| eval_f2_poly_d_at_with_powers::<D>(cell, &alpha_pows))
        .collect::<Vec<BinaryFieldGF192>>())
    ...
```

The outer loop (over columns) is already parallel via `cfg_iter!`. The
inner per-cell loop ran the textbook form:

```rust
for (i, c) in p.iter().enumerate() {     // D=32 iterations
    if c.inner() {                       // branch per bit
        acc += &alpha_powers[i];         // AddAssign rebuilds Uint<3>
    }
}
```

Per cell: 32 iterations of `BinaryU64PolyIter::next` + extract bit +
branch + (maybe) `AddAssign` on `BinaryFieldGF192` (which read both
operand words, XOR'd, and rebuilt a `Uint<3>` each time). At
~3.3M cells × 32 iter, that's ~100M branch + add operations.

### What changed

[`poly/src/univariate/binary_gf192.rs:972-1001`](../poly/src/univariate/binary_gf192.rs#L972-L1001)
— rewrite `eval_f2_poly_d_at_with_powers` to (a) read the cell via
`F2PackU64::pack_u64()` in one shot, (b) walk only the set bits via
`trailing_zeros` + `bits &= bits - 1`, and (c) accumulate into a raw
`[u64; 3]` (no `BinaryFieldGF192`/`Uint<3>` rebuild per add):

```rust
let mut bits = p.pack_u64();
let mut acc = [0u64; 3];
while bits != 0 {
    let i = bits.trailing_zeros() as usize;
    let pw = alpha_powers[i].words();
    acc[0] ^= pw[0];
    acc[1] ^= pw[1];
    acc[2] ^= pw[2];
    bits &= bits - 1;
}
BinaryFieldGF192::from_words(acc)
```

The same pattern is already used by `BinaryU64PolyInnerProduct`'s
`inner_product` at [`poly/src/univariate/binary_u64.rs:469-481`](../poly/src/univariate/binary_u64.rs#L469-L481);
this just brings the eval helper in line.

### Performance

| benchmark | before | after | speedup |
|---|---:|---:|---:|
| `Micro/UAIR-b-AlphaProject` | 24.9 ms | 1.5 ms | **16×** |

The bench iterates *sequentially* over columns (`.iter()`), so its time
reflects the pure per-cell speedup. The real
`prove_f2_uair_with_groups` runs α-projection in parallel
(`cfg_iter!`), so its share of the win is ~2-3 ms in absolute terms.

### Correctness

New parity test
[`poly/src/univariate/binary_gf192.rs::tests::eval_with_powers_matches_branchless`](../poly/src/univariate/binary_gf192.rs#L1721-L1755)
pins the new form against the existing branchless reference across
nine bit patterns. All 34 protocol e2e tests also still pass.

---

## 3. Open-d-CombinedRow: lift + inner-product fuse

### Bottleneck

`Micro/Open-d-CombinedRow` was **8.49 ms** at `nvars=16`. Inner loop
per `(col, j)` at [`protocol/src/f2_prove.rs:1818-1849`](../protocol/src/f2_prove.rs#L1818-L1849)
(real prover) and mirrored in the bench:

```rust
let column_j_lifted: Vec<BinaryF2Poly<1>> = (0..num_rows)
    .map(|i| lift_bp_to_f2_poly_1::<D>(&col.evaluations[i * row_len + j]))
    .collect();
let per_col_entry: BinaryF2Poly<4> = f2_inner_product::<1, 3, 4>(&column_j_lifted, &coeffs);
let scaled: BinaryF2Poly<7> = f2_poly_mul::<3, 4, 7>(&gamma[g], &per_col_entry);
```

Two issues:

1. **`lift_bp_to_f2_poly_1`** ([`poly/src/univariate/binary_gf192.rs:1232`](../poly/src/univariate/binary_gf192.rs#L1232))
   ran the same per-bit-branch textbook loop as the old `eval_f2_poly_d_at_with_powers`:

   ```rust
   for (i, c) in p.iter().enumerate() {
       if c.inner() { bits |= 1u64 << i; }
   }
   ```

   For `BinaryU64Poly<D>` (which is literally a `u64` under the hood
   when `simd` is enabled), this is reading what's already there
   bit-by-bit and re-packing.

2. The per-`(col, j)` `Vec<BinaryF2Poly<1>>` allocation: at
   `nvars=16` with 50 cols × 8 192 `j` values, that's 410 K small
   allocs per Open-d run — ~2 ms parallel.

### What changed

**Lift fix** at [`poly/src/univariate/binary_gf192.rs:1232-1248`](../poly/src/univariate/binary_gf192.rs#L1232-L1248):
one `pack_u64()` + a const-fold-able mask. ~10× faster per call.

**Inline fuse** at [`protocol/src/f2_prove.rs:1818-1858`](../protocol/src/f2_prove.rs#L1818-L1858)
(and mirrored in the bench at [`protocol/benches/f2_sha256.rs:1175-1201`](../protocol/benches/f2_sha256.rs#L1175-L1201)):
absorb the inner-product directly into a stack `BinaryF2Poly<4>` and
skip the intermediate `Vec` materialisation:

```rust
let mut per_col_entry = BinaryF2Poly::<4>::zero();
for i in 0..num_rows {
    let cell = lift_bp_to_f2_poly_1::<D>(&col.evaluations[i * row_len + j]);
    let prod: BinaryF2Poly<4> = f2_poly_mul::<1, 3, 4>(&cell, &coeffs[i]);
    per_col_entry += prod;
}
let scaled: BinaryF2Poly<7> = f2_poly_mul::<3, 4, 7>(&gamma[g], &per_col_entry);
```

The lift fix also benefits the *other* call site (step 7.1 of
`prove_f2_open` at [`protocol/src/f2_prove.rs:1759-1764`](../protocol/src/f2_prove.rs#L1759-L1764))
— another ~3.3M lift calls per Open run that no longer pay the
per-bit-branch tax.

### Performance

| Open-d state | time |
|---|---:|
| Baseline | 8.49 ms |
| Lift optimisation only | 7.13 ms |
| **+ inline inner-product** | **5.23 ms** (−38%) |

---

## 4. UAIR-d-ColEvalsAtRstar: parallelise the outer loop

### Bottleneck

`Micro/UAIR-d-ColEvalsAtRstar` was **3.85 ms** at `nvars=16`. The real
prove at [`protocol/src/f2_prove.rs:913-929`](../protocol/src/f2_prove.rs#L913-L929)
builds per-column MLE evals at the sumcheck point `r*`:

```rust
let all_col_evals: Vec<BinaryFieldGF192> = projected_trace
    .iter()                              // <-- sequential
    .map(|col| {
        let inner_mle = DenseMultilinearExtension::from_evaluations_vec(
            col.num_vars,
            col.evaluations.iter().map(|x| *x.inner()).collect(),
            zero_inner,
        );
        evaluate_with_config(inner_mle, &sumcheck_point, &field_cfg).expect(...)
    })
    .collect();
```

Each eval is `O(2^num_vars)` GF(2^192) ops over `num_cols`
independent columns — but the outer loop was `.iter()`, *sequential*,
even though every other heavy outer loop in `prove_f2_uair_with_groups`
uses `cfg_iter!`. Easy fix.

### What changed

[`protocol/src/f2_prove.rs:914`](../protocol/src/f2_prove.rs#L914)
— swap `.iter()` for `cfg_iter!`. Bench mirrored.

### Performance

| benchmark | before | after | speedup |
|---|---:|---:|---:|
| `Micro/UAIR-d-ColEvalsAtRstar` | 3.85 ms | 1.43 ms | **2.7×** |

A separate possible follow-up — zero-copy reinterpret of
`Vec<BinaryFieldGF192>` as `Vec<Uint<3>>` via the `#[repr(transparent)]`
guarantee — would save another ~0.3 ms (the per-col 1.5 MB clone). Not
pursued: the dominant cost in the 1.43 ms is the actual MLE-eval, not
the clone.

---

## 5. Bench bug fix: `Open-e-MerkleOpens` panic

While building the per-step breakdown, the `Open-e-MerkleOpens` micro
panicked at `nvars=16` with `Merkle prove: InvalidLeafIndex(5039)`.

The bench was calling

```rust
merkle_tree.prove(column_idx)
```

where `column_idx ∈ [0, codeword_len)`. The real
`prove_f2_open` at [`protocol/src/f2_prove.rs:1924-1926`](../protocol/src/f2_prove.rs#L1924-L1926)
maps the sampled column index to its leaf via

```rust
let group_idx = column_idx / LEAF_GROUP_SIZE;
merkle_tree.prove(group_idx)
```

because `commit_grouped` groups `LEAF_GROUP_SIZE = 8` columns per leaf.
The bench now mirrors that ([`protocol/benches/f2_sha256.rs:1198-1228`](../protocol/benches/f2_sha256.rs#L1198-L1228)).
Pre-existing bug; would have triggered at any non-tiny `nvars`.

---

## Final per-step nvars=16 breakdown

After all branch work:

| step | time | % of Prove |
|---|---:|---:|
| `Commit` (GPU + inline pack) | 18.23 ms | ~52% |
| `UAIR-a-F2NativeIC` | 1.20 ms | 3% |
| `UAIR-b-AlphaProject` | 1.52 ms | 4% |
| `UAIR-c-Sumcheck` | 1.49 ms | 4% |
| `UAIR-d-ColEvalsAtRstar` | 1.43 ms | 4% |
| `Open-a-AlphaBasis` | 0.18 ms | <1% |
| `Open-b-LiftedEqTensor` | 2.83 ms | 8% |
| `Open-c-Folds` | 0.43 ms | 1% |
| `Open-d-CombinedRow` | 5.23 ms | 15% |
| `Open-e-MerkleOpens` | 0.27 ms | <1% |
| `Open-f-GammaCoeffsLift` | 0.02 ms | <1% |
| `Open-g-AssembleOpened` | 0.23 ms | <1% |
| **Micro sum** | **33.06 ms** | — |
| **e2e Prove (best)** | **~34.8 ms** | reconciles |

## Remaining levers (in order of impact)

1. **`Open-b-LiftedEqTensor` (2.83 ms, ~8%)** — builds the (q₀, q₁)
   tensors via `AlphaPolyBasis::lift`. Likely a function-call-heavy
   GF(2^192) → F_2[X]<3> basis-change.
2. **`Open-c-Folds` and step 7.1 in `prove_f2_open`** — the *other*
   lift-and-inner-product site. Already benefits from the lift fix; its
   inner-product over `row_len` cells could be inlined the same way
   Open-d was.
3. **`Open-d` further** — switch to an on-GPU gather Blake3 kernel
   (`bolt-rs` has the building blocks); would save the ~2 ms CPU pack
   step in the commit phase too. Designed but not prototyped on this
   branch.
4. **`UAIR-d` zero-copy `Uint<3>` reinterpret** — ~0.3 ms.
5. **`commit_grouped` allocator reuse** — the 86 MB slab is freshly
   allocated each commit; a persistent pool would halve allocator
   churn but the headline savings are small.

## How to reproduce

```bash
# Headline e2e (with GPU acceleration enabled)
cargo bench -p zinc-protocol --bench f2_sha256 \
    --features parallel,simd,unchecked,metal_gpu -- "Zinc\+ F_2 SHA-256/"

# Per-step Micro breakdown at nvars=16
cargo bench -p zinc-protocol --bench f2_sha256 \
    --features parallel,simd,unchecked,metal_gpu -- "Zinc\+ F_2 SHA-256 Micro/"

# Without GPU (sanity check the CPU path still works + small workloads)
cargo bench -p zinc-protocol --bench f2_sha256 \
    --features parallel,simd,unchecked -- "Zinc\+ F_2 SHA-256/"
```

The `metal_gpu` feature is macOS-only (depends on the `metal` crate);
on Linux/Windows the CPU `Commit-Fused` path remains the default and
all other optimisations on this branch are platform-independent.
