# Value-sized field elements for the runtime ("random") prime

Branch: `runtime-const-field` (off `main-beta`).
Code: `utils/src/field/runtime_monty.rs`, bench `utils/examples/runtime_monty_bench.rs`.

> Scope note: this concerns the **integer prover path** (the projected prime
> field `F_q`), which is *out of scope* for `documentation/f2x-sha-todo.md` (that
> ledger is for the F_2 SHA-256 path). Hence this standalone doc.

## 1. The waste

On `main-beta` the projected / "random" field is

```rust
type F = crypto_primitives::crypto_bigint_monty::MontyField<LIMBS>;   // LIMBS = 4 (256-bit)
```

`MontyField<LIMBS>` wraps `crypto_bigint::modular::MontyForm<LIMBS>`, which is

```rust
struct MontyForm<const LIMBS: usize> {
    montgomery_form: Uint<LIMBS>,     // the actual value        (32 B for LIMBS=4)
    params: MontyParams<LIMBS>,       // modulus, R, R², mod_inv  (~112 B for LIMBS=4)
}
```

So **every field element stores the full field configuration inline**:

| type                         | bytes / element (LIMBS=4) | overhead |
|------------------------------|---------------------------|----------|
| `MontyForm<4>` / `MontyField<4>` | **144**               | 4.5×     |
| `ConstMontyForm<_, 4>`       | 32                        | 1×       |
| `Fp<_, 4>` (this branch)     | **32**                    | 1×       |

The decisive observation: **there is exactly one modulus per proof.** It is
built once —

```rust
// protocol/src/prover.rs
let field_cfg = crate::fixed_prime::secp256k1_field_cfg::<F, Zt::Fmod>();   // one MontyParams
```

— and then *cloned into every one of the millions of field elements* that make
up witness columns, MLE tables, eq(x,r) tables, codeword openings, sumcheck
state, etc. We are paying ~112 bytes per element to store, redundantly, a single
ambient constant. The cost is paid three times over:

* **memory** — vectors are 4.5× larger than the data they hold;
* **bandwidth / cache** — streaming a 4.5×-bloated vector thrashes cache; and
* **copies** — every `clone`/move of a field-element vector shuffles 144 B/elt
  instead of 32 B/elt (and this codebase clones field vectors constantly:
  projections, transposes, sumcheck-state construction…).

## 2. Why the obvious fix doesn't apply

`ConstMontyForm<MOD, LIMBS>` is already value-sized (the modulus is a
compile-time `ConstMontyParams` *type*, stored as `PhantomData`). But the
projecting prime is drawn from the Fiat–Shamir transcript at **runtime**, so it
cannot be a `const` type parameter. That is the whole reason `main-beta` reaches
for the dynamic `MontyField` and inherits the per-element config.

## 3. What the codebase does today (partial workaround)

`main-beta` is mid-migration to a *structure-of-arrays* workaround: store the
bare `F::Inner` (the `Uint`, 32 B) in the big vectors and thread the config
`&F::Config` alongside, doing arithmetic through a bespoke trait:

```rust
// utils/src/inner_transparent_field.rs
trait InnerTransparentField: PrimeField {
    fn add_inner(lhs: &Self::Inner, rhs: &Self::Inner, config: &Self::Config) -> Self::Inner;
    fn sub_inner(...);
    fn mul_assign_by_inner(...);
}
```

This works (e.g. `ProverState.mles: Vec<DenseMultilinearExtension<F::Inner>>`,
`build_eq_x_r_inner_vec`), but it is:

* **incomplete** — many bulk `Vec<F>` remain (e.g. `poly/src/utils.rs`,
  `piop/src/sumcheck/multi_degree.rs`, `piop/src/lookup/booleanity.rs`,
  `piop/src/combined_poly_resolver/*`, `zip-plus` `encode_f`);
* **duplicative** — every hot function exists twice (`build_eq_x_r_vec` →
  `Vec<F>` *and* `build_eq_x_r_inner_vec` → `Vec<F::Inner>`); and
* **leaky** — every call site must thread `&cfg` and remember to pick the
  `_inner` variant; you lose ordinary `a + b` / `a * b` ergonomics.

## 4. The better approach: a runtime-installed-modulus field

Make the element value-sized like `ConstMontyForm`, but install the (runtime)
modulus **once**, in a process-global cell selected by a zero-sized *slot* type:

```rust
pub struct Fp<S: Modulus<LIMBS>, const LIMBS: usize> {
    mont: Uint<LIMBS>,        // value in Montgomery form — 32 B, nothing else
    _slot: PhantomData<S>,    // zero-sized: which modulus
}

pub trait Modulus<const LIMBS: usize> {
    fn cell() -> &'static OnceLock<Params<LIMBS>>;   // one cell per slot
    // params()/install()/install_modulus() provided by default
}

define_modulus!(pub Secp256k1Base, 4);               // declares the ZST + its cell
```

* The modulus (plus the cached Montgomery `mod_neg_inv`) lives **once** in the
  slot's `OnceLock`, installed at proof start (`Secp256k1Base::install_modulus(p)`
  or `install_monty(field_cfg)`).
* Arithmetic reads the ambient params through the slot — a lock-free atomic load
  after the one-time install — so **no element ever carries the config**.
* Ordinary operators (`a + b`, `a * b`) work, because the modulus is ambient.
  No `&cfg` threading, no `_inner` duplication, no `InnerTransparentField`.

Add/sub/neg operate directly on the Montgomery-form limbs
(`Uint::add_mod`/`sub_mod`/`neg_mod`, exact because Montgomery form is additively
linear). Multiply is a faithful port of crypto-primitives' CIOS Montgomery
multiply (the original is `pub(crate)`, hence un-callable across the crate
boundary), taking the modulus by reference plus the cached `mod_neg_inv` — so it
matches `MontyForm`'s arithmetic cost while moving 4.5× fewer bytes. The port is
verified element-for-element against `MontyForm` by property tests.

### Correctness

`utils/src/field/runtime_monty.rs` proptests (400 cases each) assert
`Fp == MontyForm` (canonical form) for add, sub, mul, neg, `new`/`retrieve`
round-trip, `inv`, `pow`, and the compound-assign ops. All pass.

## 5. Measurements

`cargo run --offline --release -p zinc-utils --example runtime_monty_bench 22`
(secp256k1 base prime, 2^22 ≈ 4.2M elements, Apple Silicon, best-of-7):

```
element size : MontyForm = 144 B   Fp = 32 B   (4.50x smaller)   [exact]
vec alloc    : MontyForm = 576 MiB Fp = 128 MiB (4.50x smaller)  [exact]

kernel timings — best of 7 (ratio > 1.0 ⇒ Fp faster):
build_eq(x,r)          MontyForm  99.5 ms   Fp  66.7 ms   (1.49x)
clone vec              MontyForm  14.0 ms   Fp   3.1 ms   (4.47x)   ← pure bandwidth
hadamard a*b           MontyForm  78.4 ms   Fp  62.0 ms   (1.26x)
dot product            MontyForm  80.7 ms   Fp  70.7 ms   (1.14x)   ← most compute-bound
sumcheck fold          MontyForm  98.5 ms   Fp  80.6 ms   (1.22x)
```

* **Memory: 4.5× smaller, deterministically.** This is the headline.
* **Pure data movement (clone): 4.5× faster** — exactly the size ratio. Every
  projection / transpose / state copy in the prover benefits by this factor.
* **Arithmetic kernels: 1.1–1.5× faster** — from better cache density (smaller
  inputs) plus an equal-cost multiply.
* Under multicore contention (the realistic case for a prover saturating all
  cores) the memory advantage widens: in loaded runs these ratios reached
  2–3.3×, because the 4.5×-larger `MontyForm` vectors suffer disproportionately
  on a shared memory subsystem.

There is **no kernel where `Fp` is slower.** (An earlier revision that
recomputed `mod_neg_inv` per multiply lost ~10% on the compute-bound `dot`;
caching it in the slot fixed that.)

### End-to-end (the real workload)

With `Fp` wired in as the integer-path field `F` (see §7), the **full SHA+ECDSA
prover** peak RSS (debug, `test_e2e_sha_ecdsa_proof_shape`, `/usr/bin/time -l`):

```
MontyField (before) : 140.7 MiB
Fp         (after)  :  70.7 MiB     →  ~2.0x less total prover memory
```

The per-element config was pervasive enough that removing it nearly **halves the
whole prover's memory footprint** — not just the field vectors in isolation.

## 6. Tradeoffs / constraints

* **One modulus per slot per process.** A slot is set once (`OnceLock`):
  re-installing the *same* modulus is a no-op; a *different* one panics with a
  clear message. `main-beta`'s prover uses a single fixed prime, so one slot
  suffices. Code needing several moduli in one process declares several slots
  (distinct ZSTs ⇒ distinct cells). This is the price of operator ergonomics:
  a value-sized element with no `&cfg` parameter *must* read the modulus from
  somewhere ambient.
* **Install-before-use.** Arithmetic before `install` panics (clear message).
  This maps naturally onto "draw the prime, then prove."
* **Global state.** It is set-once and immutable thereafter, hence rayon-safe
  (all workers read the same frozen cell). For a future multi-prime-per-process
  need, the slot could be backed by a generation counter or a thread-local
  installed via `rayon::broadcast`; not needed for the current prover.

## 7. Drop-in integration (DONE)

`Fp` is now the integer-path field `F` — the full migration is implemented and
validated, not just prototyped:

1. **Trait surface implemented** on `Fp` (`utils/src/field/runtime_monty.rs`):
   `Semiring`, `Ring`, `Field`, `PrimeField`, plus `CheckedAdd/Sub/Mul/Neg`,
   `Pow<u32>`, `Div`/`DivAssign`, and the zinc traits `MulByScalar`, `FromRef`,
   `InnerTransparentField`, `ProjectableToField`. Conversions are plain
   `From<T>` (primitives, `bool`, `Uint`, signed `Int<N>`), so
   `FromWithConfig<T>` / `FromPrimitiveWithConfig` come for free via the
   crypto-primitives blanket — the same ambient-modulus pattern as
   `ConstMontyField`. Wide `Int<N>` (more limbs than the field) delegates its
   one-time reduction to `MontyField` at the projection boundary.
2. **`type Config = ()`**, and `make_cfg(modulus)` installs the slot as its side
   effect. The prover's existing `field_cfg: F::Config` threading now carries
   `()` harmlessly; the single `secp256k1_field_cfg` call is the install point.
   `Inner = Modulus = crypto-primitives `Uint`` newtype, so all the heavy
   `Inner`/`Modulus` bounds are inherited unchanged.
3. **Wiring**: the protocol library is generic over `F`, so the change is one
   type alias per concrete site — `type F = Fp<ProofSlot, FIELD_LIMBS>` in the
   e2e test module (`protocol/src/lib.rs`) and the e2e benchmark
   (`protocol/benches/e2e.rs`), each with a `define_modulus!(ProofSlot, …)` slot.

**Validation:** all 21 `zinc-protocol` lib tests pass, including the full
SHA+ECDSA e2e (`test_e2e_sha_ecdsa_proof_shape`, `_folded_4x_round_trip`) and
every soundness/tamper test, plus the ~2.0× e2e memory reduction in §5.

The remaining cleanup (optional, follow-up): the `_inner` SoA duplication
(`build_eq_x_r_inner_vec`, the `Vec<F::Inner>` plumbing) is now redundant — with
a 32 B element and ambient modulus the single generic `Vec<F>` path is already
optimal — and can be deleted or kept as thin shims.

## 8. Files

* `utils/src/field/runtime_monty.rs` — `Fp`, `Modulus`, `Params`,
  `define_modulus!`, the CIOS multiply port, the full crypto-primitives + zinc
  trait surface, and the proptest suite.
* `utils/examples/runtime_monty_bench.rs` — the `Fp`-vs-`MontyForm` time +
  memory benchmark (counting allocator + best-of-K kernels).
* `protocol/src/lib.rs`, `protocol/benches/e2e.rs` — the integer-path
  `type F = Fp<ProofSlot, FIELD_LIMBS>` swap (one alias + slot each).

### Build note (unrelated to this change)

`Cargo.lock` is gitignored, and the dependency is declared as a *floating*
`crypto-primitives = { git = …, branch = "main" }`. A fresh resolution picks the
newest commit on that branch (currently `e9c386c`), whose `PrimeField` API
(`Self::Integer`, `cfg()` via `HasPrimeFieldConfig`) `main-beta`'s source predates
— so the workspace fails to compile out of the box. To build this branch, pin
locally to the rev the source matches:

```
cargo update -p crypto-primitives --precise 2cf39db886a76dc3e961cbb9c86fb5ab042381ef
```

This is a pre-existing repo fragility (the floating `branch="main"` dep), not a
consequence of the field change. A durable fix is to pin the dependency to a
specific `rev =` in the workspace `Cargo.toml`.
