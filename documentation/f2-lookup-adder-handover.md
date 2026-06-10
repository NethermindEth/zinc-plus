# F_2 lookup adder — handover

Branch: **`f2-clean-lookup`** (off `f2-clean`). Status: **the char-2-sound,
commitment-bound grand-product lookup adder for mod-2³² addition is built,
tested, measured, carries VIRTUALIZED (zero adder-specific committed
columns), η-batched, and PORTED ONTO THE PRODUCTION OBLONG PIPELINE** —
`prove/verify_f2_full_with_oblong_lookup_adder` is the first fully-sound
SHA-256 F_2 prove path: the 2 ANDs ride the GF(2⁸) oblong zerocheck, the 12
adders ride the lookup, nothing is trusted anywhere. It replaces the 12
trusted carry Hadamards (the "Issue-1" soundness gap), with strictly less
committed data than the Binius identity's 1-committed-β-bit floor. This doc
is the single entry point for whoever continues; the full design history
(including rejected approaches, the dropped-carry attack, and why the tag
exists) lives in `documentation/f2x-sha-todo.md` under the lookup-adder
entries (2026-06-09/10).

**Read first:** `CLAUDE.md` — every design/optimization decision on this path
must be ledgered in `f2x-sha-todo.md` before ending your turn. ⚠️ The working
tree has UNCOMMITTED ledger edits (entangled with other work-streams' WIP —
Keccak, recursion, R1CS entries) — do not lose them; keep editing in place.

---

## 1. What exists and works

Two complete bound pipelines in `protocol/src/f2_prove.rs`, sharing one
lookup phase (`prove/verify_lookup_adder_bound_phase`):

- **`prove/verify_f2_full_with_oblong_lookup_adder` — THE PRODUCTION PATH**:
  GF(2⁸) oblong zerocheck for the 2 ANDs + the lookup for the 12 adders;
  the η-blocks join the oblong's combined multipoint as a third segment
  (`[α-cols] ++ [z-cols] ++ [η-blocks]`); the single open binds both
  discharge families via two weight-vector projections of the same `a′`
  (`ψ_z(a′)` and `proj(a′, W_η)`); `adder_parents` must be empty (verifier-
  enforced — no trusted-operand channel remains).
- **`prove/verify_f2_full_with_lookup_adder_bound`** — the monomial-pipeline
  twin (the original integration target; kept as the like-for-like A/B arm).

Both prove SHA-256 F_2 with the adders enforced by a multiplicative
grand-product lookup, **sound end-to-end, with zero adder-specific committed
columns**:

- **Carries virtualized**: every inter-limb carry is the boundary-bit z-read
  `cin_i = cout_{i−1} = z[ℓi]`, `z = x⊕y(⊕y2)⊕t` — single-bit reads of
  already-committed operands; `cin_0 = 0` is a constant; the LAST limb is
  fingerprinted **marginally** (no cout) against the cout-marginalised table
  `T′`, with a **`γ⁴·emb(2)` tag** that keeps `T′` γ-polynomially disjoint
  from 5-term witness fingerprints (without it, a dropped-carry corruption
  masquerades as a `T′` row — see §4.2 and the ledger).
- **Witness side bound to the commitment** (the part that closes Issue-1):
  fingerprints are family read-offs of committed words (`nf = 2nl−1` families:
  `nl` limb packs + `nl−1` boundary bits), reduced by a binding sumcheck and
  bound through the multipoint fold + the open's `a′` consistency — **η-batched**:
  a fresh `η` (drawn post-absorb of the family evals) collapses the `nf`
  families to ONE block per witness column, ONE claim per witness `(col,Δ)`
  pair, and ONE `a′` equation under the fresh `γ_open`.
- **Table side fixed pre-challenge**: per-row use counts for BOTH tables
  (`m` for `T`, `m′` for `T′`) ship in the clear (system is not ZK), absorbed
  **before δ**; the verifier recomputes `∏(δ−fpT)^m · ∏(δ−fpT′)^{m′}` itself
  and checks it against the witness tree's root.
- **Adversarial suite green** (14 `sha256_f2` tests + 6 unit): wrong add sum
  on a NON-LSB bit (the old trusted-Hadamard gap), the **dropped-carry
  attack** (the new surface opened by virtualization — defeated by the tag;
  a negative-control unit test proves the untagged variant accepts it),
  tampered family evals, lying multiplicities on either table.

### Measured (nvars=16, LB=8 ⇒ 2¹⁷ table, Apple M-series, release)

After the 2026-06-10 optimization sweep (parallel/fused loops, table-driven
fingerprints, bit-bucketed family evals + zero-mult `Q_a`, eq-factored GKR
layer sumcheck — all bit-identical-output refactors, ledger has the
breakdown):

| Arm | Prove | Verify |
|---|---|---|
| A monomial pipeline, 14 rel (12 adders TRUSTED) | 459.0 ms | 4.4 ms |
| B monomial pipeline + SOUND lookup adder | 240.9 ms | 29.7 ms |
| C **production** oblong-GF8, 14 rel (trusted) | 92.7 ms | — |
| D **production oblong-GF8 + SOUND lookup adder** | **206.1 ms** (best 195.9) | 28.9 ms |

**The fully-sound production pipeline (D) is ~2.2× the trusted one (C)
same-run** — from 3.5× post-port and 6.4× at the first measurement. The
machine swings hard thermally (cold-machine historical: A ~400 / C ~80;
same-run ratios are the meaningful comparison). Per-phase (D ≈ 204): commit ~19,
discharge (2 ANDs) 6, α-uair 12, **lookup machinery ~75** (`gkr_tree` 58 —
of which the eq-factored sumchecks 42.6 + suffix build 9.3, `binding` 20.4,
`leaves` 4.1, `counts` 2.5, `lblocks` 2.7, self ~16), z_block 2.6,
multipoint ~30, open ~8. History of the machinery: 265 (committed carries)
→ 229 (virtualized) → 202 (η-batched) → **~75** (sweep). Verify ~28 ms —
still table-side dominated (2×2¹⁷ fingerprint rows, the `m′` product,
~2 MB of absorbed counts; §5.2 kills these).

---

## 2. File map

| File | Contents |
|---|---|
| `piop/src/lookup/gkr_product.rs` | GKR product-tree engine: `prove/verify_product_tree` (∏ leaves = root → leaf-MLE claim at a point). The prover's layer sumchecks run `prove_layer_sumcheck_eq_factored` — eq factored (suffix tensors + prefix scalar, never materialised/folded), char-2 affine-flat fourth node, **byte-identical** to the generic `MLSumcheck` path (which remains the verifier). Field-generic (runtime char-2 detection; prime fields accumulate the 4th node in-pass); tested at GF(2¹²⁸) AND a prime field. |
| `piop/src/lookup/gkr_lookup.rs` | Two-tree grand-product lookup (`prove/verify_lookup`). **Only the honest-first path uses it now**; the bound path uses the witness tree + clear multiplicities. Keep for reference or fold away. |
| `piop/src/lookup/add_lookup.rs` | Public add table `T={(x,y,cin,s,cout)}` + the **marginalised `T′={(x,y,cin,s)}`** (`add_table_fingerprints[_marginal]`), `decompose_add` (honest chain — the honest-first path), **`decompose_add_zread`** (virtualized carries — the bound path), `fingerprint[_marginal]`, `marginal_tag` (= `emb(2)`), `multiplicities`. The z-read unit suite incl. the dropped-carry attack + its negative control. Parameterised by `limb_bits` (tests 4 ⇒ 2⁹ table; production 8 ⇒ 2¹⁷). |
| `protocol/src/f2_lookup_binding.rs` | The witness-binding sumcheck over **families** (`num_families = 2nl−1`: limb packs `L_i` + boundary bits `e_{ℓj}`): `lookup_binding_coeffs` (per-relation `((col,Δ),family)→coeff` maps + the public tag `constant`), `lookup_binding_public_part` (PubMLE, add-only mask eq-sums), `prove/verify_lookup_binding` (degree-3 `eq·Σ_a mask_a·Q_a`; `Q_a` rows built from per-(relation,pair) precombined bit-weight tables — pure adds; family evals + mask evals via the **bit-bucket kit**: `eq_table_gf` + `family_proj_evals_bucketed`, one add-only pass per `(col,Δ)` for all `nf` evals), and the **η-batching kit**: `eta_combined_weights` (the fused per-bit table `W`; `family_weights` is the reference form, pinned by the `eta_table_matches_per_family_weights` canary) + `eta_block_rows`. Module doc has the math. |
| `protocol/src/f2_prove.rs` | **`prove/verify_f2_full_with_oblong_lookup_adder` (PRODUCTION)** and `prove/verify_f2_full_with_lookup_adder_bound` (monomial twin), both over the shared `prove/verify_lookup_adder_bound_phase` (the factored lookup phase: γ/m+m′/δ/leaves/tree/binding/η/blocks); `sha_add_limb_tuples_zread`, `sha_lookup_witness_leaves_tensor_zread`, `F2LookupAdderProof` (incl. `mult_counts_marginal`) + `F2OblongLookupAdderProof` (adds the η-blocks' `r_0` evals — the oblong proof segments its `r_0` evals α/z/lookup), the honest-first wrapper `prove_f2_full_with_lookup_adder` (still on committed carries + old helpers — scaffolding), all tests + the A/B/C/D bench. |

Tests (run with `--features parallel,simd,unchecked`): unit z-read suite in
`add_lookup.rs` (`zread_*` — incl. `zread_rejects_dropped_carry` and the
negative control `zread_dropped_carry_passes_without_tag`); in `f2_prove.rs`:
`sha256_f2_lookup_adder_proves_real_adds`, `…_in_prover_roundtrips`,
`…_binding_identity_holds`, `…_binding_sumcheck_roundtrips`,
`…_bound_roundtrips` (+ tamper arms incl. lying `m′`), `…_rejects_wrong_sum`,
`…_rejects_dropped_carry`, `sha256_f2_drops_adder_hadamards_roundtrips`, and
the PRODUCTION-path suite `sha256_f2_oblong_lookup_adder_roundtrips`
(+ tamper arms: lying `m`/`m′`, tampered family eval, tampered η-block `r_0`
eval, smuggled adder parents), `…_rejects_wrong_sum`,
`…_rejects_dropped_carry`. Bench (4 arms):

```bash
AB_NVARS=16 cargo test --release -p zinc-protocol \
  --features parallel,simd,unchecked \
  sha256_f2_lookup_adder_ab_timing -- --ignored --nocapture
# add OBLONG_PROFILE=1 for the per-phase tree
```

---

## 3. Protocol flow (bound paths) — transcript order is load-bearing

**PRODUCTION path** (`prove_f2_full_with_oblong_lookup_adder`): commit the
plain trace → the GF(2⁸) oblong AND-zerocheck, **ANDs only** (the adders
ride the lookup; `adder_parents` empty) → the α-only UAIR → the shared
lookup phase (steps 3–6 below) → the ψ_z z-block of all witness cols + the
η-block `r*`-evals → absorb `(z_up_evals, pair_evals, adder_parents=[],
lblock r*-evals)` → ONE combined multipoint over
`[α-cols] ++ [z-cols] ++ [η-blocks]` (AND pair claims at `γ_word` against
the z-cols; the η-combined lookup claims at `r★` against the η-blocks) →
`r_0` evals absorbed α/z/lookup → the single γ-batched open. The verifier
additionally checks BOTH `a′` projections against the open's returned batch
γ — `ψ_z(a′) = Σ γ_g·z_r0[g]` and `proj(a′, W_η) = Σ γ_g·F^η_g(r_0)` — and
ties the oblong zerocheck from the bound pair evals only (`k = 2`, no
trusted parents).

**Monomial twin** (`prove_f2_full_with_lookup_adder_bound`):
1. Commit the **plain trace** (no augmentation — carries virtualized).
2. Base UAIR (IC + α + the 2 AND Hadamards + sumcheck) — adders dropped.
3. `γ` (fingerprint challenge) → **absorb `mult_counts` then
   `mult_counts_marginal`** → `δ`.
4. Tensor witness leaves (`leaf[p·n+r]`, `p=(adder,limb)` HIGH vars, rows LOW;
   inactive rows = `1` factors; z-read carries; last limb tagged-marginal) →
   **witness product tree** → claim `(r_w, eval_w)`; split
   `r_w = (r_row, r_pair)`.
5. **Binding sumcheck** at `r_row`: proves `eval_w + PubMLE(r_w) =
   Σ_r eq·Σ_a mask_a·Q_a` (`Q_a` includes the public tag constant), ships
   family shifted evals `F̃_f(col↓Δ)(r★)` per distinct `(col,Δ)` — all
   `nf = 2nl−1` families each — and absorbs them.
6. **η + combined blocks**: a FRESH `η` (drawn right after step 5 ⇒
   post-absorb of the family evals — that freshness is what makes the
   batching sound), then ONE η-combined block per witness col `g`
   (`eta_block_rows` over the `eta_combined_weights` table `W`) joins the
   multipoint fold as a source (`r*` up-evals shipped as
   `lblock_evals_at_rstar`, absorbed pre-fold). ONE pointed-shift claim per
   witness `(col,Δ)` pair, down-eval `Σ_f η^f·F̃_f(col↓Δ)(r★)` — computed by
   the verifier from the shipped per-family evals (PUBLIC cols — `pa_k/a/e`
   — are recomputed directly, all families, for the binding recombination).
7. Multipoint → `r_0` → extended `open_evals_at_r_0` (absorbed) → the single
   γ-batched open.

Verifier mirrors 1–7 plus: two-table root recompute
(`∏(δ−fpT)^m·∏(δ−fpT′)^{m′}`, counts capped at `2^k_bits`), witness-tree
depth check (`nv + pair_vars`), and the **single η-combined `a′`
consistency** `proj(a′, W) == Σ_g γ_open,g·F^η_g(r_0)` (the same `W` table
the blocks were built from) — `γ_open` is the open verify's RETURN VALUE
(`a′ = proof.open.lifted_claim`); no transcript clone is needed on either
bound path.

---

## 4. Soundness invariants — do not break these

1. **Multiplicities absorbed BEFORE δ — both arrays.** Fixes the table
   polynomial pre-challenge; without it the SZ grand-product argument fails.
2. **The marginal tag is load-bearing.** `fpT′` (and the last-limb witness
   family) carries `γ⁴·emb(2)`; `emb(2) = X ∉ {0,1}` = the value set of every
   genuine committed-bit read. Without it, a dropped carry
   (`t′ = x+y − 2^{ℓ(j+1)}`) makes tuple `j` (correct `s`, z-read cout 0)
   IDENTICAL to an untagged `T′` row — the cheat passes. The negative-control
   unit test `zread_dropped_carry_passes_without_tag` pins the attack; if it
   ever fails, the attack model changed — re-derive before touching the tag.
3. **Never ship per-bit-slice evals recombined under an already-known
   challenge.** The sound form: FIXED weight families, recombined ONLY under
   challenges drawn AFTER the evals are absorbed. Both fresh-challenge uses
   here — `η` (combines the `nf` families: drawn right after the binding
   absorbs the family evals) and `γ_open` (combines the witness columns in
   the `a′` equation) — are the sound z-block pattern; the rejected 4c-2
   per-bit form failed precisely because α predated the evals. If you move
   the η draw EARLIER than the binding's absorb, the batching becomes that
   broken pattern.
4. **The leaves and the binding coefficients must express the SAME
   functional.** Leaves use `decompose_add_zread` + `fingerprint[_marginal]`;
   coefficients use limb/boundary-bit reads + the tag constant. The 4c-2
   sumcheck roundtrip's claimed-sum equality is the numeric canary; it holds
   on corrupt traces too (the product check is what rejects them).
5. **Tensor leaf layout** (rows = LOW MLE vars). The binding identity's
   `eq`-factoring depends on it; `sha256_f2_lookup_binding_identity_holds` is
   the canary.
6. **Absorption order**: family evals, family-block `r*` evals, and
   `open_evals_at_r_0` are each absorbed before the challenges that bind them.
7. **`k_bits` cap** on counts — both `m` and `m′` (bounds the table-polynomial
   degree → the SZ error). Production sizing: a hot table row can exceed 2¹⁶
   uses at large nvars — size `k_bits` ~24 and check honestly.
8. The grand product is **multiplicative** — additive LogUp is UNSOUND in
   char 2 (even multiplicities cancel). Never "simplify" to the fraction form.

---

## 5. Roadmap (prioritized; designs already in the ledger)

1. **Remaining prover spots** (the 2026-06-10 sweep took the machinery
   203 → ~75 ms; what's left): `gkr_tree` ~58 (the eq-factored layer
   sumchecks are compute-bound at ~10 GF mults/pair; further cuts are
   PROTOCOL changes — e.g. 48 = 3·16 pair blocks as 3 trees of 2²⁰ with
   multiplied roots, −25%, not byte-identical, needs its own adversarial
   pass; or skipping the ~44% trivial first-round pairs, ~4 ms, branchy),
   `multipoint_eval` ~30, `binding` ~20 (the degree-3 sumcheck prove
   itself), lookup self ~16. Realistic floor without protocol changes
   ≈ 150–170 ms.
2. **Verifier + proof size** (verify is ~28 ms, table-side dominated):
   derive `T′` fingerprints from `T`'s in one add/row
   (`fpT′ = fpT + γ⁴·(emb(cout)+X)`, char 2) instead of a second 2¹⁷-row
   build; sparse-encode `m′` (and `m`: 2¹⁷×8B ≈ 1 MB each); closed-form
   mask-MLEs (periodic 68-row interval structure) replacing 12×O(n) passes;
   **structured-table MLE** for a succinct table side (the v2 path —
   subsumes the first two).
3. **κ/LsbX IC simplification** (new, from virtualization): tuple 0's
   `cin = 0` constant pins the low limb incl. the LSB through the chain, so
   the κ compensator is redundant for adds. Unexplored.
4. **Column savings beyond carries** (user interest): k-ary add tables could
   delete the 5 intermediate-sum columns (`w_w_s1/s2`, `w_t1_s1/s2/s3`) —
   viable only at smaller `ℓ` (table `2^{kℓ}`); longfellow-zk does 7-input
   adds this way. Unexplored trade.
5. **Retire the trusted oblong arm**: once the machinery work lands, make
   `prove_f2_full_with_oblong_lookup_adder` the default SHA-256 F_2 entry
   point (benches, e2e) and demote `prove_f2_full_with_oblong_hadamard`'s
   trusted-adder mode to a comparison arm.

## 6. Gotchas

- Always build/test with `--features parallel,simd,unchecked` (the F_2 path
  assumes them; `gkr_product` needs the rayon prelude under `parallel`).
- `D=32` is assumed throughout the lookup code (Keccak `D=64` would need
  genericizing `limb_bits`/families).
- `pair_vars = next_pow2(12·nl).trailing_zeros()` must agree between prover,
  verifier, and the leaf builder — it's computed independently in each.
- **Two decompositions exist**: `decompose_add` (honest carry chain — the
  honest-first path, carry columns, and "find a row with a real carry" test
  setup) vs `decompose_add_zread` (the bound path). Don't mix them: the
  bound path's leaves/coeffs/multiplicities must ALL be z-read.
- The honest-first path (`prove_f2_full_with_lookup_adder` + the two-tree
  `gkr_lookup::prove_lookup`) still exists on the OLD committed-carry
  helpers (`sha_lookup_carry_columns`, `sha_add_limb_tuples`,
  `sha_lookup_witness_leaves_tensor`, single untagged table); it's
  superseded by the bound path but kept as scaffolding/tests. Its
  `num_primary_columns` is still `NUM_BIN + 12`; the bound path's is plain
  `NUM_BIN`.
- The **Z[X] sibling task** (integer-path range checks) lives on branch
  `lookup` (off `main-beta`): `documentation/range-check-design.md` +
  `documentation/lookup-methods-design.md`. Untouched since design.
- Memory: `~/.claude/.../memory/f2-lookup-adder-status.md` mirrors this
  state; update it when the state changes.
