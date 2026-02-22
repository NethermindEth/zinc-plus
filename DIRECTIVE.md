# DIRECTIVE: End-to-End Benchmarks for SHA-256 + ECDSA in Zinc+

**From:** Lead Author, Zinc+ Paper  
**To:** Implementation Engineer  
**Date:** 2026-02-22  
**Priority:** Critical  
**Deliverable:** Working, benchmarked, end-to-end SHA-256 + ECDSA SNARK pipeline

---

## 0. Preamble

I have reviewed the current state of the codebase at `/Users/albertgarretafontelles/dev/zinc-plus`. The foundation — UAIR trait system, polynomial infrastructure (`zinc-poly`), PIOP subprotocols (sumcheck, ideal check, combined poly resolver), Fiat-Shamir transcript, and Zip+ PCS with IPRS codes + Merkle trees — is solid. The existing test UAIRs (`TestUairSimpleMultiplication`, `TestAirBinary`) demonstrate the trait machinery works.

**What is missing is everything that actually matters for the paper's claims.** There is no SHA-256 arithmetization, no ECDSA arithmetization, no top-level prover/verifier composition, no witness generation for real inputs, and no end-to-end benchmarks. These are the numbers we publish. They must be correct and they must hit the targets.

I expect working, tested, reviewed-quality code — not scaffolding. Every module must have unit tests. Every constraint must be verified against known test vectors. Every benchmark must be reproducible on an M4 MacBook Air.

---

## 1. Implementation Order (strict priority)

You will implement in exactly this order. Do not skip ahead. Each phase must compile, pass all tests, and be committed before moving to the next.

### Phase 1: SHA-256 UAIR Arithmetization (`zinc-sha256-uair`)

This is the hardest piece and the one most likely to have subtle bugs. Start here.

### Phase 2: ECDSA UAIR Arithmetization (`zinc-ecdsa-uair`)

### Phase 3: Witness Generation for both UAIRs

### Phase 4: Top-Level SNARK Compositor (`zinc-snark`)

### Phase 5: End-to-End Benchmarks

---

## 2. Phase 1: SHA-256 UAIR — Exact Specification

Create a new crate `sha256-uair/` and add it to the workspace.

### 2.1 Trace Layout

The trace has **64 rows** (one per SHA-256 compression round) and **19 witness columns**, partitioned as follows:

| Columns | Count | Ring | Description |
|---------|-------|------|-------------|
| `a[0..9]` | 10 | `BitPoly_32` ⊂ `Z^{<32}[X]` | Working registers and intermediate bit-decomposed values |
| `q[0..4]` | 5 | `{0,...,2^32-1}` ⊂ `Q` | Integer representations for modular addition checks |
| `f[0..3]` | 4 | `F_2^{<32}[X]` | Binary polynomial temporaries for XOR/AND decomposition |

The columns represent:
- `a[0]` = register `a` (current round)
- `a[1]` = register `e` (current round)
- Remaining `a[2..9]` = intermediate values for Σ₀, Σ₁, Ch, Maj decomposition
- `q[0..4]` = integer witnesses for the two modular additions per round (T₁ and T₂ computation), plus carry bits
- `f[0..3]` = XOR/AND decomposition temporaries

**Shift-register optimization (CRITICAL):** Only registers `a` and `e` are stored as explicit columns. Registers `b,c,d` are recovered from rows `i-1, i-2, i-3` of the `a` column. Registers `f,g,h` are recovered from rows `i-1, i-2, i-3` of the `e` column. This means:

```
b[i] = a[i-1],  c[i] = a[i-2],  d[i] = a[i-3]
f[i] = e[i-1],  g[i] = e[i-2],  h[i] = e[i-3]
```

For `constrain_general`, you therefore need access not just to `up`/`down` but to rows at offsets -1, -2, -3. Review how the existing `Uair` trait exposes `up` (row i) and `down` (row i+1). **You will likely need to extend the trait or use a multi-row window.** Discuss this with me if the current trait is insufficient, but my expectation is that you encode the shift-register unwinding into the constraint polynomials using the existing two-row interface by expressing the relationships across consecutive constrain calls. If that is provably impossible, propose a minimal trait extension — do not over-engineer.

### 2.2 Constraint Monomials — Exact Counts

**53 total constraint monomials**, broken down as:

| Count | Ideal | Ring | Description |
|-------|-------|------|-------------|
| 10 | `(X^32 - 1)` in `F_2[X]` | Binary | Bitwise rotation constraints: `ROTR^r(u) - X^{32-r}·u ∈ (X^32 - 1)` for each rotation in Σ₀, Σ₁ |
| 6 | `(0)` in `F_2[X]` | Binary | XOR decomposition: `u + v = (u ⊕ v) + 2(u ∧ v)`, i.e., the carry polynomial must be zero in `F_2[X]` |
| 27 | `(X - 2)` in `Q[X]` | Rational | Modular addition constraints: evaluation at X=2 collapses polynomial to integer, enforcing mod 2^32 arithmetic |
| 10 | — | — | (accounted for in the rotation set above) |

**Rotation encoding:** For `ROTR^r` on a 32-bit word represented as `u ∈ F_2[X]/(X^32-1)`:
```
ROTR^r(u) = X^{32-r} · u  mod (X^32 - 1)
```
The constraint is: `ROTR^r(u) - X^{32-r} · u ∈ (X^32 - 1)`.

The specific rotations needed for SHA-256:
- Σ₀(a): ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a)
- Σ₁(e): ROTR⁶(e) ⊕ ROTR¹¹(e) ⊕ ROTR²⁵(e)
- σ₀(w): ROTR⁷(w) ⊕ ROTR¹⁸(w) ⊕ SHR³(w)
- σ₁(w): ROTR¹⁷(w) ⊕ ROTR¹⁹(w) ⊕ SHR¹⁰(w)

**XOR encoding:** `u ⊕ v` is computed via the identity:
```
u + v = (u ⊕ v) + 2·(u ∧ v)
```
In `F_2[X]`, the carry term `u ∧ v` is checked to be zero (since in F_2, 2=0), giving `u + v = u ⊕ v`.

**Modular addition encoding:** Evaluate polynomial representation at `X = 2` to recover integer value. The ideal `(X - 2)` in `Q[X]` means: `f(X) ∈ (X-2)` iff `f(2) = 0`. Use this to enforce `a + b - c - carry·2^32 ∈ (X-2)` where the carry is a single bit.

### 2.3 Ideal Types

You must implement concrete ideal types. The existing `Ideal` trait and `IdealCheck` in `zinc-uair` are the interface. You need:

1. **`CyclotomicIdeal32`** — represents `(X^32 - 1)` in `F_2[X]`. `contains(&self, poly)` checks that `poly mod (X^32-1) == 0`.
2. **`ZeroIdeal`** — represents `(0)` in `F_2[X]`. `contains(&self, poly)` checks `poly == 0`. (This may already be expressible via `assert_zero`, but make it explicit as an `Ideal` so the ideal-check PIOP handles it uniformly.)
3. **`EvalAt2Ideal`** — represents `(X-2)` in `Q[X]`. `contains(&self, poly)` checks `poly(2) == 0`.

These must implement `Ideal + IdealCheck<BinaryPoly<32>>` (or the appropriate expression type). The `IdealCheck` implementation for the cyclotomic ideal requires polynomial reduction mod `X^32-1`; implement this efficiently (it's just folding coefficients).

### 2.4 Implementation Details

```rust
pub struct Sha256Uair;

impl Uair<BinaryPoly<32>> for Sha256Uair {
    type Ideal = Sha256Ideal; // enum { Cyclotomic32, Zero, EvalAt2 }
    
    fn num_cols() -> usize { 19 }
    
    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: &[B::Expr],
        down: &[B::Expr],
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        // ...
    {
        // IMPLEMENT ALL 53 CONSTRAINTS HERE
    }
}
```

### 2.5 Testing Requirements

1. **Known-answer test:** Take the SHA-256 of `"abc"` (digest: `ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad`). Generate the full 64-round witness. Verify every constraint is satisfied.
2. **NIST test vectors:** At minimum, test against 3 NIST SHAVS test vectors (empty string, "abc", 448-bit message).
3. **Constraint counting:** Write a test that uses the existing `ConstraintCounter` builder from `zinc-uair` to verify exactly 53 constraint monomials are generated.
4. **Ideal counting:** Use `IdealCollector` to verify the ideal distribution matches the paper (10 cyclotomic, 6 zero, 27 eval-at-2).

---

## 3. Phase 2: ECDSA UAIR — Exact Specification

Create a new crate `ecdsa-uair/`.

### 3.1 Trace Layout

**258 rows × 14 columns:**

| Columns | Count | Ring | Description |
|---------|-------|------|-------------|
| `b[0..2]` | 3 | `{0,1}` ⊂ `Q` | Scalar bits for double-scalar multiplication (bit of s⁻¹·z and s⁻¹·r) and a control bit |
| `p[0..8]` | 9 | `F_p` (secp256k1 base field) | Point coordinates in Jacobian form: (X, Y, Z) for accumulator, and intermediate values |
| `n[0..1]` | 2 | `F_n` (secp256k1 scalar field) | Scalar reconstruction witnesses |

Where:
- `p` = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
- `n` = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

### 3.2 Shamir's Trick

The 258 rows encode a double-scalar multiplication `[u₁]G + [u₂]Q` using Shamir's trick:
- Process one bit of each scalar per row
- Each row performs: point doubling + conditional addition of G, Q, or G+Q based on the two scalar bits
- 256 rows for the bit processing + 2 rows for setup/finalization

### 3.3 Jacobian Coordinates

All point arithmetic is in Jacobian coordinates `(X:Y:Z)` where the affine point is `(X/Z², Y/Z³)`.

**Point doubling** (when `b[2]=0`):
```
M = 3·X₁²  (a=0 for secp256k1)
S = 4·X₁·Y₁²
X₃ = M² - 2·S
Y₃ = M·(S - X₃) - 8·Y₁⁴
Z₃ = 2·Y₁·Z₁
```

**Point addition** `(X₁:Y₁:Z₁) + (X₂:Y₂:1)` (affine second operand):
Standard mixed addition formulas.

### 3.4 Constraints

**9 constraints over F_p** (degree up to 6):
These enforce the Jacobian doubling and addition formulas. Each constraint is a polynomial identity that must hold modulo `p`. The degree-6 comes from the `Y₁⁴` and `Y₁²·Z₁` terms.

**2 constraints over F_n:**
- Scalar bit decomposition: accumulated scalar matches expected value
- Bit validity: `b[i] · (1 - b[i]) = 0` for each bit column

### 3.5 Field Arithmetic

secp256k1's base field `F_p` is a 256-bit prime field. You need:
- An `F_p` implementation that wraps `crypto-bigint` (the workspace already depends on `crypto_bigint = 0.7.0-rc.9`).
- Efficient modular multiplication (Montgomery form).
- The `ConstMonty` or `BoxedMonty` types in `zinc-utils/src/field/` should be usable — check if secp256k1's `p` fits their API.

### 3.6 Testing Requirements

1. **Known-answer test:** Take a known secp256k1 ECDSA signature (use a Bitcoin test vector), generate the full 258-row witness, verify all constraints.
2. **Bit decomposition test:** Verify scalar bits reconstruct to the correct scalar.
3. **Point arithmetic test:** Verify intermediate Jacobian coordinates against a reference implementation (use the `k256` crate or hand-compute a few steps).
4. **Constraint counting:** Verify exactly 11 constraints (9 over F_p, 2 over F_n).

---

## 4. Phase 3: Witness Generation

### 4.1 SHA-256 Witness Generator

```rust
pub fn sha256_witness(message: &[u8]) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> 
```

- Implement the full SHA-256 compression function, but instead of just computing the hash, record every intermediate value into the 19-column trace.
- For messages requiring multiple blocks: generate one 64-row trace per block. For the benchmark target, we need **8× SHA-256 blocks** (i.e., 8 × 64 = 512 rows).
- The MLE representation: 512 rows requires `num_vars = 9` (since 2⁹ = 512). Each column becomes a `DenseMultilinearExtension<BinaryPoly<32>>` with 512 evaluations.
- **Pad with valid dummy rows** if the number of SHA-256 blocks is not a power of 2. The dummy rows must still satisfy all constraints (use identity rounds or repeat the last state).

### 4.2 ECDSA Witness Generator

```rust
pub fn ecdsa_witness(
    msg_hash: &[u8; 32],
    signature: &EcdsaSignature,
    public_key: &PublicKey,
) -> Vec<DenseMultilinearExtension</* appropriate type */>>
```

- 258 rows requires `num_vars = 9` (pad to 512) or use 258 directly if the system supports non-power-of-2 (check the sumcheck prover — it likely requires power-of-2).
- If padding is needed: pad to 512 rows with dummy constraint-satisfying rows.

### 4.3 Testing

- Witness must round-trip: `sha256_witness("abc")` fed through the UAIR constraints produces zero residuals.
- Performance: witness generation itself should be < 1ms for SHA-256, < 5ms for ECDSA.

---

## 5. Phase 4: Top-Level SNARK Compositor (`zinc-snark`)

This is the glue. Create crate `zinc-snark/`.

### 5.1 Architecture

The SNARK is a composition of:
1. **Ideal Check** (from `zinc-piop`) — processes the UAIR constraints
2. **Combined Poly Resolver** (from `zinc-piop`) — reduces to sumcheck claims
3. **Sumcheck** (from `zinc-piop`) — reduces to polynomial evaluation claims
4. **Batched Zip+ PCS** (from `zip-plus`) — commits to and opens the witness polynomials

The data flow is:

```
Witness (trace matrix of MLEs)
    │
    ▼
[Batched PCS Commit] ──► commitment (Merkle root)
    │
    ▼
[Ideal Check Prover] ──► combined polynomial evaluations
    │
    ▼
[Combined Poly Resolver Prover] ──► sumcheck instance
    │
    ▼
[Sumcheck Prover] ──► evaluation claims on witness MLEs
    │
    ▼
[Batched PCS Evaluate + Test] ──► opening proofs
    │
    ▼
Proof = (commitment, ideal_check_proof, resolver_proof, sumcheck_proof, pcs_proof)
```

### 5.2 Prover API

```rust
pub struct ZincPlusProver;

impl ZincPlusProver {
    pub fn prove<U: Uair<BinaryPoly<DEGREE_PLUS_ONE>>, const DEGREE_PLUS_ONE: usize>(
        params: &ZincPlusParams,
        witness: &[DenseMultilinearExtension<BinaryPoly<DEGREE_PLUS_ONE>>],
    ) -> Result<ZincPlusProof, ZincPlusError> {
        // 1. Commit witness via BatchedZipPlus::commit
        // 2. Run IdealCheckProtocol::prove_as_subprotocol::<U>
        // 3. Run CombinedPolyResolver::prove_as_subprotocol::<_, U>
        // 4. Run MLSumcheck::prove_as_subprotocol
        // 5. Run BatchedZipPlus::evaluate (for claimed evaluations)
        // 6. Run BatchedZipPlus::test (for Merkle opening proofs)
        // 7. Serialize and return proof
    }
}
```

### 5.3 Verifier API

```rust
pub struct ZincPlusVerifier;

impl ZincPlusVerifier {
    pub fn verify<U: Uair<BinaryPoly<DEGREE_PLUS_ONE>>, const DEGREE_PLUS_ONE: usize>(
        params: &ZincPlusParams,
        proof: &ZincPlusProof,
        public_input: &PublicInput,
    ) -> Result<bool, ZincPlusError> {
        // 1. Deserialize proof components
        // 2. Verify ideal check
        // 3. Verify combined poly resolver
        // 4. Verify sumcheck
        // 5. Verify PCS opening (BatchedZipPlus::verify)
        // 6. Return accept/reject
    }
}
```

### 5.4 Parameters

For the benchmarks matching the paper:
- Base field: `p = 2^16 + 1 = 65537` (F_{65537})
- IPRS code rate: `1/4`
- Radix-8 FFT (already implemented in `zip-plus/src/code/iprs/pntt/radix8/`)
- Number of column openings: calibrated for 100-bit security
- `BinaryPoly<32>` for SHA-256 witness columns (since we need `F_2[X]/(X^32-1)`)

### 5.5 Transcript Threading

A single `KeccakTranscript` must be threaded through all subprotocols in order. The Fiat-Shamir security depends on this. Do NOT create separate transcripts for different phases.

### 5.6 Testing

1. **Round-trip test:** `prove` then `verify` must accept for valid witnesses.
2. **Soundness smoke test:** Corrupt one witness value, verify that `verify` rejects.
3. **Determinism test:** Same input produces same proof bytes.

---

## 6. Phase 5: End-to-End Benchmarks

Create `benches/e2e_benchmark.rs` using the `criterion` crate (already in workspace dependencies).

### 6.1 The Primary Benchmark (MUST MATCH PAPER)

**Scenario:** 8× SHA-256 compression blocks + 1× ECDSA verification

This is the paper's headline number. The witness consists of:
- SHA-256: 8 blocks × 64 rows = 512 rows × 19 columns
- ECDSA: 258 rows × 14 columns (padded to 512)
- Total: composed into a single SNARK instance

**Prover benchmark:**
```
zinc_plus_e2e/prove_8sha256_ecdsa    time: [21.xxx ms ... 23.xxx ms]
```
**Target: < 30ms.** Paper claims ~21-23ms.

**Verifier benchmark:**
```
zinc_plus_e2e/verify_8sha256_ecdsa   time: [2.xxx ms]
```
**Target: < 5ms.** Paper claims 2.30ms.

**Proof size:**
```
zinc_plus_e2e/proof_size_8sha256_ecdsa   211 KB (± 10%)
```
**Target: 200-300KB.** Paper claims 211KB.

### 6.2 Micro-Benchmarks

Also produce:
- `sha256_witness_generation/1_block` — witness gen time for 1 SHA-256 block
- `sha256_witness_generation/8_blocks` — witness gen time for 8 SHA-256 blocks
- `ecdsa_witness_generation` — witness gen time for ECDSA
- `sha256_prove_only` — SNARK prover for SHA-256 alone (no ECDSA)
- `ecdsa_prove_only` — SNARK prover for ECDSA alone (no SHA-256)
- `pcs_commit` — commitment phase only
- `pcs_test_verify` — PCS testing + verification phase only
- `ideal_check` — ideal check subprotocol only
- `sumcheck` — sumcheck subprotocol only

### 6.3 Benchmark Configuration

```rust
// All benchmarks must use:
// - Multi-threaded (rayon, use all cores)
// - Release mode with LTO (already configured in Cargo.toml)
// - Warm-up: 5 iterations
// - Measurement: 20 iterations minimum
// - Report: mean, std dev, throughput
```

### 6.4 Benchmark Output

Produce a summary table in the benchmark output (or as a separate script) that matches the paper's Table format:

```
┌─────────────────────────────┬──────────┬──────────┬────────────┐
│ Operation                   │ Time     │ Std Dev  │ Proof Size │
├─────────────────────────────┼──────────┼──────────┼────────────┤
│ Prover (8×SHA-256 + ECDSA)  │ XX.XX ms │ X.XX ms  │ XXX KB     │
│ Verifier                    │ X.XX ms  │ X.XX ms  │ —          │
│   ├─ Ideal Check            │ X.XX ms  │          │            │
│   ├─ Combined Poly Resolver │ X.XX ms  │          │            │
│   ├─ Sumcheck               │ X.XX ms  │          │            │
│   └─ PCS Verify             │ X.XX ms  │          │            │
└─────────────────────────────┴──────────┴──────────┴────────────┘
```

---

## 7. Code Quality Standards

### 7.1 Non-Negotiable

- **Zero `unwrap()` in library code.** Use `expect()` with descriptive messages only in test/bench code. All errors must propagate via `Result`.
- **Respect the existing clippy configuration:** `arithmetic_side_effects = "deny"`, `cast_possible_truncation = "deny"`, etc. The SHA-256 UAIR crate may need `#![allow(clippy::arithmetic_side_effects)]` at the crate level (as `test-uair` does), but document why.
- **All public items documented.** Every `pub fn`, `pub struct`, `pub enum` gets a doc comment explaining what it does, its invariants, and its relationship to the paper.
- **Reference the paper.** In doc comments for constraint implementations, cite the specific section/equation of the paper. Example: `/// Constraint (3.7): Σ₀ rotation check via ideal (X^32-1).`
- **No dead code.** No commented-out blocks. No `todo!()` in shipped code (use `unimplemented!()` with an issue reference if truly needed).

### 7.2 Performance

- Use `rayon` parallelism wherever beneficial. The existing codebase uses `cfg_iter!` macro from `zinc-utils` — use it consistently.
- SHA-256 witness generation must not allocate unnecessarily. Pre-allocate the trace matrix.
- Polynomial arithmetic in the constraint evaluation hot path must not box or heap-allocate per-operation.

### 7.3 Testing

- Every crate must have `cargo test` passing with zero failures.
- Every UAIR must have at least one known-answer test with a real cryptographic test vector.
- Property tests (using `proptest`, already in dependencies) for:
  - Witness generation produces constraint-satisfying traces for random inputs
  - Proof verification accepts valid proofs
  - Proof verification rejects corrupted proofs (flip one bit in proof, one bit in witness)

### 7.4 CI Compatibility

- `cargo clippy --workspace --all-targets -- -D warnings` must pass.
- `cargo test --workspace` must pass.
- `cargo bench --workspace --no-run` must compile.

---

## 8. Crate Structure Summary

After implementation, the workspace should have:

```
zinc-plus/
├── Cargo.toml (workspace)
├── uair/                    # UAIR trait (existing)
├── poly/                    # Polynomial types (existing)
├── piop/                    # Sumcheck, Ideal Check, Combined Poly Resolver (existing)
├── transcript/              # Keccak transcript (existing)
├── zip-plus/                # Zip+ PCS (existing)
├── utils/                   # Utilities (existing)
├── primality/               # Miller-Rabin (existing)
├── test-uair/               # Toy test UAIRs (existing)
├── sha256-uair/             # NEW: SHA-256 UAIR + witness generation
│   ├── src/
│   │   ├── lib.rs           # Sha256Uair impl
│   │   ├── ideals.rs        # CyclotomicIdeal32, EvalAt2Ideal
│   │   ├── witness.rs       # sha256_witness()
│   │   ├── constants.rs     # SHA-256 round constants K[], initial hash H[]
│   │   └── tests.rs         # Known-answer tests, constraint counting
│   └── Cargo.toml
├── ecdsa-uair/              # NEW: ECDSA UAIR + witness generation
│   ├── src/
│   │   ├── lib.rs           # EcdsaUair impl
│   │   ├── secp256k1.rs     # Field definitions, curve constants
│   │   ├── jacobian.rs      # Jacobian point arithmetic
│   │   ├── witness.rs       # ecdsa_witness()
│   │   └── tests.rs         # Known-answer tests
│   └── Cargo.toml
├── snark/                   # NEW: Top-level SNARK compositor
│   ├── src/
│   │   ├── lib.rs
│   │   ├── prover.rs        # ZincPlusProver
│   │   ├── verifier.rs      # ZincPlusVerifier
│   │   ├── params.rs        # ZincPlusParams
│   │   ├── proof.rs         # ZincPlusProof serialization
│   │   └── error.rs         # ZincPlusError
│   ├── benches/
│   │   └── e2e_benchmark.rs # THE benchmark
│   └── Cargo.toml
└── ...
```

---

## 9. What I Will Check

When you deliver, I will personally verify:

1. `cargo test --workspace` — all green, zero warnings.
2. `cargo clippy --workspace --all-targets` — clean.
3. `cargo bench --bench e2e_benchmark` — numbers within 50% of paper targets (I understand first-pass may need optimization, but it must be in the right ballpark; if prover is > 60ms, something is architecturally wrong).
4. SHA-256 known-answer: digest of `"abc"` matches `ba7816bf...`.
5. Constraint counts: exactly 53 for SHA-256, exactly 11 for ECDSA.
6. Proof size: measure and report exact bytes.
7. Proof round-trip: prove → verify → accept for valid; reject for corrupted.
8. Code review: I will read every line of the UAIR implementations. The constraints must match the paper's equations exactly. If I find a constraint that doesn't correspond to a specific equation in the paper, I will reject the PR.

---

## 10. Timeline Expectation

- **Phase 1 (SHA-256 UAIR):** 3-4 days. This is the most complex arithmetization.
- **Phase 2 (ECDSA UAIR):** 2-3 days. Field arithmetic is standard but the Jacobian formulas need care.
- **Phase 3 (Witness gen):** 1-2 days. Straightforward once UAIRs are correct.
- **Phase 4 (SNARK compositor):** 2-3 days. Mostly plumbing but the transcript threading is subtle.
- **Phase 5 (Benchmarks):** 1 day. Setup is mechanical once Phase 4 works.

**Total: ~10-13 working days.**

Do not gold-plate. Do not refactor existing crates unless strictly necessary for integration. Do not add features not specified here. Hit the spec, hit the tests, hit the benchmarks.

If you encounter a design decision not covered here, document the options, pick the simplest one that doesn't compromise correctness or performance, and move on. I'll review in the PR.

Start now. Deliver Phase 1 first.

— **Lead Author**
