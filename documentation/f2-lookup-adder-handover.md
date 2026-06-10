# F_2 lookup adder — handover

Branch: **`f2-clean-lookup`** (off `f2-clean`). Status: **the char-2-sound,
commitment-bound grand-product lookup adder for mod-2³² addition is built,
tested, measured — and its carries are now VIRTUALIZED: zero committed
adder-specific columns** (was 12 committed carry columns). It replaces the 12
trusted carry Hadamards (the "Issue-1" soundness gap) on the F_2[X] SHA-256
path, with strictly less committed data than the Binius identity's
1-committed-β-bit floor. This doc is the single entry point for whoever
continues; the full design history (including rejected approaches, the
dropped-carry attack, and why the tag exists) lives in
`documentation/f2x-sha-todo.md` under the lookup-adder entries (2026-06-09/10).

**Read first:** `CLAUDE.md` — every design/optimization decision on this path
must be ledgered in `f2x-sha-todo.md` before ending your turn. ⚠️ The working
tree has UNCOMMITTED ledger edits (entangled with other work-streams' WIP —
Keccak, recursion, R1CS entries) — do not lose them; keep editing in place.

---

## 1. What exists and works

`prove/verify_f2_full_with_lookup_adder_bound` (in `protocol/src/f2_prove.rs`)
prove SHA-256 F_2 with the adders enforced by a multiplicative grand-product
lookup, **sound end-to-end, with zero adder-specific committed columns**:

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
  bound through the multipoint fold + the open's `a′` consistency, one
  equation per family under the fresh `γ_open`.
- **Table side fixed pre-challenge**: per-row use counts for BOTH tables
  (`m` for `T`, `m′` for `T′`) ship in the clear (system is not ZK), absorbed
  **before δ**; the verifier recomputes `∏(δ−fpT)^m · ∏(δ−fpT′)^{m′}` itself
  and checks it against the witness tree's root.
- **Adversarial suite green** (14 `sha256_f2` tests + 6 unit): wrong add sum
  on a NON-LSB bit (the old trusted-Hadamard gap), the **dropped-carry
  attack** (the new surface opened by virtualization — defeated by the tag;
  a negative-control unit test proves the untagged variant accepts it),
  tampered family evals, lying multiplicities on either table.

### Measured (nvars=16, LB=8 ⇒ 2¹⁷ table, Apple M-series, release, same-run)

| Arm | Prove | Verify |
|---|---|---|
| A monomial pipeline, 14 rel (12 adders TRUSTED) | 427.4 ms | 4.2 ms |
| B monomial pipeline + SOUND lookup adder (virtualized carries) | 515.5 ms | 35.4 ms |
| C **production** oblong-GF8, 14 rel (trusted) | **89.0 ms** | — |

(Earlier same-run set with committed carries: A 419.1 / B 535.0 / C 82.5,
verify B 22.6 — the machine swings ~8% thermally; compare within a run.)
Lookup machinery ≈ 229 ms (`gkr_tree` 66, `leaves` 43, `binding` 41,
`lblocks` 39, `tuples` 16) — cheaper than the carry-column version (~265) —
**but `multipoint_eval` exploded to ~156 ms**: ~38 witness pairs × 7 families
≈ 250+ pointed claims, each materialising a `next`+source MLE in the fold.
The fold-claim count is now the top prover lever (§5.2). Verify regressed
22.6 → 35.4 ms: doubled table-side fingerprints + the `m′` product + ~1 MB
more absorbed counts (§5.3 kills these). **Honest position: the sound adder
is ~5.8× the production oblong prove today**; the fight is (i) ride the
oblong path, (ii) batch the fold claims, (iii) verifier table side.

---

## 2. File map

| File | Contents |
|---|---|
| `piop/src/lookup/gkr_product.rs` | GKR product-tree engine: `prove/verify_product_tree` (∏ leaves = root → leaf-MLE claim at a point). Field-generic; tested at GF(2¹²⁸). |
| `piop/src/lookup/gkr_lookup.rs` | Two-tree grand-product lookup (`prove/verify_lookup`). **Only the honest-first path uses it now**; the bound path uses the witness tree + clear multiplicities. Keep for reference or fold away. |
| `piop/src/lookup/add_lookup.rs` | Public add table `T={(x,y,cin,s,cout)}` + the **marginalised `T′={(x,y,cin,s)}`** (`add_table_fingerprints[_marginal]`), `decompose_add` (honest chain — the honest-first path), **`decompose_add_zread`** (virtualized carries — the bound path), `fingerprint[_marginal]`, `marginal_tag` (= `emb(2)`), `multiplicities`. The z-read unit suite incl. the dropped-carry attack + its negative control. Parameterised by `limb_bits` (tests 4 ⇒ 2⁹ table; production 8 ⇒ 2¹⁷). |
| `protocol/src/f2_lookup_binding.rs` | The witness-binding sumcheck over **families** (`num_families = 2nl−1`: limb packs `L_i` + boundary bits `e_{ℓj}`): `lookup_binding_coeffs` (per-relation `((col,Δ),family)→coeff` maps + the public tag `constant`), `lookup_binding_public_part` (PubMLE), `prove/verify_lookup_binding` (degree-3 `eq·Σ_a mask_a·Q_a`), `family_proj_rows_pub`/`family_proj_eval_pub`, `family_weights` (the `a′` weight vectors). Module doc has the math. |
| `protocol/src/f2_prove.rs` | `prove/verify_f2_full_with_lookup_adder_bound` (the real pipeline — UNAUGMENTED trace), `sha_add_limb_tuples_zread`, `sha_lookup_witness_leaves_tensor_zread`, `F2LookupAdderProof` (incl. `mult_counts_marginal`), the honest-first wrapper `prove_f2_full_with_lookup_adder` (still on committed carries + old helpers — scaffolding), all tests + the A/B bench. |

Tests (run with `--features parallel,simd,unchecked`): unit z-read suite in
`add_lookup.rs` (`zread_*` — incl. `zread_rejects_dropped_carry` and the
negative control `zread_dropped_carry_passes_without_tag`); in `f2_prove.rs`:
`sha256_f2_lookup_adder_proves_real_adds`, `…_in_prover_roundtrips`,
`…_binding_identity_holds`, `…_binding_sumcheck_roundtrips`,
`…_bound_roundtrips` (+ tamper arms incl. lying `m′`), `…_rejects_wrong_sum`,
`…_rejects_dropped_carry`, `sha256_f2_drops_adder_hadamards_roundtrips`.
Bench:

```bash
AB_NVARS=16 cargo test --release -p zinc-protocol \
  --features parallel,simd,unchecked \
  sha256_f2_lookup_adder_ab_timing -- --ignored --nocapture
# add OBLONG_PROFILE=1 for the per-phase tree
```

---

## 3. Protocol flow (bound path) — transcript order is load-bearing

Prover (`prove_f2_full_with_lookup_adder_bound`):
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
   `nf = 2nl−1` families each.
6. **Family blocks**: per family `f` × witness col `g`, the projected column
   joins the multipoint fold as a source (`r*` up-evals shipped as
   `lblock_evals_at_rstar`, absorbed pre-fold). Pointed-shift claims at
   `(r★,Δ)` carry the family evals (witness cols only; PUBLIC cols —
   `pa_k/a/e` — are recomputed by the verifier directly, all families).
7. Multipoint → `r_0` → extended `open_evals_at_r_0` (absorbed) → the single
   γ-batched open.

Verifier mirrors 1–7 plus: two-table root recompute
(`∏(δ−fpT)^m·∏(δ−fpT′)^{m′}`, counts capped at `2^k_bits`), witness-tree
depth check (`nv + pair_vars`), and the **per-family `a′` consistency**
`w_f(a′) == Σ_g γ_open,g·F_{f,g}(r_0)` (limb-pack weights `X^b` for `f<nl`,
the single boundary bit for `f≥nl`) — `γ_open` re-derived from a **transcript
clone** taken just before the open verify (`a′ = proof.open.lifted_claim`).
The bound verifier therefore requires `T: Transcript + Clone`.

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
   challenge.** The sound form: FIXED weight families — limb packs AND the
   boundary-bit families each have their own block columns, fold claims, and
   post-absorb-fresh `γ_open` consistency. (The boundary-bit families are NOT
   the rejected pattern: nothing recombines them under a stale challenge.)
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

1. **Port onto the oblong pipeline** (strategic): the lookup currently rides
   the slow monomial path; the real baseline is `prove_f2_full_with_oblong_
   hadamard` (89 ms same-run). The lookup phase + binding + fold-claims should
   slot into the oblong prove the same way (its binding data already flows
   through the same multipoint/open); main work is wiring + the ψ_z-vs-ψ_α
   block bookkeeping on that path.
2. **Batch the fold claims** (sharpened by the virtualization measurement):
   `multipoint_eval` is now ~156 ms because ~38 witness pairs × `nf=7`
   families ≈ 250+ pointed claims each materialise a `next`+source MLE.
   Batch the `nf` claims sharing `(r★, Δ, col)` under a fresh challenge (one
   claim per witness pair), and/or the **sources/evals split** (only
   claim-referenced cols as fold sources). Expected ~100+ ms back — takes B
   below the old 535 baseline. Also: fuse `tuples`+`leaves` into one pass
   with `pack_u64` cell reads + parallelize; sparse bottom tree layer (25%
   pair-padding + inactive `1`-leaves); parallelize the 12 `Q_a`
   materializations.
3. **Verifier + proof size**: derive `T′` fingerprints from `T`'s in one
   add/row (`fpT′ = fpT + γ⁴·(emb(cout)+X)`, char 2) instead of a second
   2¹⁷-row build; sparse-encode `m′` (and `m`: 2¹⁷×8B ≈ 1 MB each);
   closed-form mask-MLEs (periodic 68-row interval structure) replacing
   12×O(n) passes; **structured-table MLE** for a succinct table side (the
   v2 path — subsumes the first two).
4. **κ/LsbX IC simplification** (new, from virtualization): tuple 0's
   `cin = 0` constant pins the low limb incl. the LSB through the chain, so
   the κ compensator is redundant for adds. Unexplored.
5. **Column savings beyond carries** (user interest): k-ary add tables could
   delete the 5 intermediate-sum columns (`w_w_s1/s2`, `w_t1_s1/s2/s3`) —
   viable only at smaller `ℓ` (table `2^{kℓ}`); longfellow-zk does 7-input
   adds this way. Unexplored trade.

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
