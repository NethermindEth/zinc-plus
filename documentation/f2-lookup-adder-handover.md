# F_2 lookup adder — handover

Branch: **`f2-clean-lookup`** (off `f2-clean`, 16 commits). Status: **the
char-2-sound, commitment-bound grand-product lookup adder for mod-2³² addition
is built, tested, and measured** — it replaces the 12 trusted carry Hadamards
(the "Issue-1" soundness gap) on the F_2[X] SHA-256 path. This doc is the
single entry point for whoever continues; the full design history (including
rejected approaches and why) lives in `documentation/f2x-sha-todo.md` under
the lookup-adder entries (2026-06-09/10).

**Read first:** `CLAUDE.md` — every design/optimization decision on this path
must be ledgered in `f2x-sha-todo.md` before ending your turn. ⚠️ The current
working tree has UNCOMMITTED ledger edits (entangled with the user's docs WIP)
— do not lose them; commit them with the user's blessing or keep editing in
place.

---

## 1. What exists and works

`prove/verify_f2_full_with_lookup_adder_bound` (in `protocol/src/f2_prove.rs`)
prove SHA-256 F_2 with the adders enforced by a multiplicative grand-product
lookup, **sound end-to-end**:

- **Witness side bound to the commitment** (the part that closes Issue-1):
  fingerprints are limb-family linear read-offs of committed words + committed
  carry bits, reduced by a binding sumcheck and bound through the multipoint
  fold + the open's `a′` consistency.
- **Table side fixed pre-challenge**: multiplicities ship in the clear
  (system is not ZK), absorbed **before δ**; the verifier recomputes
  `∏_t(δ−fp_t)^{m_t}` itself and checks it against the witness tree's root.
- **Adversarial suite green**: wrong add sum on a NON-LSB bit (exactly the old
  trusted-Hadamard gap), tampered limb evals, lying multiplicities — all
  rejected. 12 `sha256_f2` tests pass.

### Measured (nvars=16, LB=8 ⇒ 2¹⁷ table, Apple M-series, release)

| Arm | Prove | Verify |
|---|---|---|
| A monomial pipeline, 14 rel (12 adders TRUSTED) | 417–419 ms | 3.7 ms |
| B monomial pipeline + SOUND lookup adder | 525–535 ms | 22.6 ms |
| C **production** oblong-GF8, 14 rel (trusted) | **82.5 ms** | — |

Lookup machinery ≈ 265 ms (`gkr_tree` 75–92, `leaves` 57–67, `lblocks` 46–54,
`binding` ~39, `tuples` 18–22). It displaces ~150–175 ms of the *slow*
monomial zerocheck — but only ~20.5 ms of the *production* oblong discharge.
**Honest position: the sound adder is ~6.4× the production prove today**; the
fight is (i) ride the oblong path, (ii) ~10× the lookup machinery.

---

## 2. File map

| File | Contents |
|---|---|
| `piop/src/lookup/gkr_product.rs` | GKR product-tree engine: `prove/verify_product_tree` (∏ leaves = root → leaf-MLE claim at a point). Field-generic; tested at GF(2¹²⁸). |
| `piop/src/lookup/gkr_lookup.rs` | Two-tree grand-product lookup (`prove/verify_lookup`). **Only the honest-first path uses it now**; the bound path uses the witness tree + clear multiplicities. Keep for reference or fold away. |
| `piop/src/lookup/add_lookup.rs` | Public add table `T={(x,y,cin,s,cout)}`, `decompose_add` (limb tuples + carry chain), `fingerprint` (`emb(x)+γ emb(y)+γ²cin+γ³emb(s)+γ⁴cout`), `multiplicities`. Parameterised by `limb_bits` (tests 4 ⇒ 2⁹ table; production 8 ⇒ 2¹⁷). |
| `protocol/src/f2_lookup_binding.rs` | The witness-binding sumcheck: `lookup_binding_coeffs` (per-relation `((col,Δ),limb)→coeff` maps), `lookup_binding_public_part` (PubMLE), `prove/verify_lookup_binding` (degree-3 `eq·Σ_a mask_a·Q_a`), `limb_proj_rows_pub`/`limb_proj_eval_pub`. Module doc has the math. |
| `protocol/src/f2_prove.rs` | `prove/verify_f2_full_with_lookup_adder_bound` (the real pipeline), `sha_add_limb_tuples`, `sha_lookup_witness_leaves_tensor`, `sha_lookup_carry_columns`, `F2LookupAdderProof`, the honest-first wrapper `prove_f2_full_with_lookup_adder`, all tests + the A/B bench. |

Tests (all in `f2_prove.rs`, run with `--features parallel,simd,unchecked`):
`sha256_f2_lookup_adder_proves_real_adds`, `…_in_prover_roundtrips`,
`…_binding_identity_holds`, `…_binding_sumcheck_roundtrips`,
`…_bound_roundtrips` (+ tamper arms), `…_rejects_wrong_sum`,
`sha256_f2_drops_adder_hadamards_roundtrips`. Bench:

```bash
AB_NVARS=16 cargo test --release -p zinc-protocol \
  --features parallel,simd,unchecked \
  sha256_f2_lookup_adder_ab_timing -- --ignored --nocapture
# add OBLONG_PROFILE=1 for the per-phase tree
```

---

## 3. Protocol flow (bound path) — transcript order is load-bearing

Prover (`prove_f2_full_with_lookup_adder_bound`):
1. Augment trace with **12 carry columns** (one per adder relation; carry `j`
   at bit `ℓ·j` so carry reads are limb reads) → commit.
2. Base UAIR (IC + α + the 2 AND Hadamards + sumcheck) — adders dropped.
3. `γ` (fingerprint challenge) → **absorb `mult_counts`** → `δ`.
4. Tensor witness leaves (`leaf[p·n+r]`, `p=(adder,limb)` HIGH vars, rows LOW;
   inactive rows = `1` factors) → **witness product tree** → claim
   `(r_w, eval_w)`; split `r_w = (r_row, r_pair)`.
5. **Binding sumcheck** at `r_row`: proves `eval_w + PubMLE(r_w) =
   Σ_r eq·Σ_a mask_a·Q_a`, ships limb-family shifted evals
   `L̃_i(col↓Δ)(r★)` per distinct `(col,Δ)`.
6. **L-blocks**: per limb family `i` × witness col `g`, the projected column
   joins the multipoint fold as a source (`r*` up-evals shipped as
   `lblock_evals_at_rstar`, absorbed pre-fold). Pointed-shift claims at
   `(r★,Δ)` carry the limb evals (witness cols only; PUBLIC cols — `pa_k/a/e`
   — are recomputed by the verifier directly).
7. Multipoint → `r_0` → extended `open_evals_at_r_0` (absorbed) → the single
   γ-batched open.

Verifier mirrors 1–7 plus: table-root recompute (`∏(δ−fp_t)^{m_t}`, counts
capped at `2^k_bits`), witness-tree depth check (`nv + pair_vars`), and the
**per-family `a′` consistency** `w_i(a′) == Σ_g γ_open,g·L_{i,g}(r_0)` —
`γ_open` re-derived from a **transcript clone** taken just before the open
verify (`a′ = proof.open.lifted_claim`; `gf128poly_project` takes any
weights). The bound verifier therefore requires `T: Transcript + Clone`.

---

## 4. Soundness invariants — do not break these

1. **Multiplicities absorbed BEFORE δ.** Fixes the table polynomial
   pre-challenge; without it the SZ grand-product argument fails (the cheater
   solves a subset-product after seeing δ).
2. **Never ship per-bit-slice evals recombined under an already-known
   challenge.** First 4c-2 cut did; it's unsound (mass-shifting between slices
   preserves the stale-α combination). The sound form: FIXED weight families
   (limb blocks), every shipped eval rides the fold to `r_0`, and the
   consistency uses the **post-absorb-fresh** `γ_open`. Ledger entry
   "REJECTED on soundness" has the attack.
3. **Carry layout = carry `j` at bit `ℓ·j`** (limb-read cleanliness; spare-bit
   garbage only pushes a cheater off-table).
4. **Tensor leaf layout** (rows = LOW MLE vars). The binding identity's
   `eq`-factoring depends on it; `sha256_f2_lookup_binding_identity_holds` is
   the canary.
5. **Absorption order**: limb evals, L-block `r*` evals, and
   `open_evals_at_r_0` are each absorbed before the challenges that bind them.
6. **`k_bits` cap** on counts (bounds the table-polynomial degree → the SZ
   error). Production sizing: a hot table row can exceed 2¹⁶ uses at large
   nvars — size `k_bits` ~24 and check honestly.
7. The grand product is **multiplicative** — additive LogUp is UNSOUND in
   char 2 (even multiplicities cancel). Never "simplify" to the fraction form.

---

## 5. Roadmap (prioritized; designs already in the ledger)

1. **Carry virtualization — 12 committed columns → ZERO** (designed, ledgered
   2026-06-10, "IDENTIFIED, NOT IMPLEMENTED"): the inter-limb carries are
   F_2-linear reads `cout_i = cin_{i+1} = z[8(i+1)]`, `z = x⊕y⊕t` (⊕`y2`) —
   single-bit reads `e_{8(i+1)}` of already-committed words, structurally
   shared between consecutive tuples; `cin_0 = 0` constant; the LAST limb
   looks up the **cout-marginalised table** `T′={(x,y,cin,s)}` so β needs no
   commitment either. Cost: 3 new bit-extraction families (`e_8,e_16,e_24`) →
   3 more L-block sets + consistency equations. Do the falsifiable unit test
   first (z-read tuples, corrupt a middle limb, product must mismatch). Note:
   packing carries instead is a NET LOSS (ledger explains why — per-bit
   families cost more than the saved columns).
2. **Port onto the oblong pipeline** (strategic): the lookup currently rides
   the slow monomial path; the real baseline is `prove_f2_full_with_oblong_
   hadamard` (82.5 ms). The lookup phase + binding + fold-claims should slot
   into the oblong prove the same way (its binding data already flows through
   the same multipoint/open); main work is wiring + the ψ_z-vs-ψ_α block
   bookkeeping on that path.
3. **Hot-spot optimizations** (~265 → ~100–130 ms est., all naive-v1 today):
   fuse `tuples`+`leaves` into one pass with `pack_u64` cell reads +
   parallelize (~80→~15); sparse bottom tree layer (25% pair-padding + ~6%
   inactive `1`-leaves skippable); `lblocks` parallel build + the
   **sources/evals split** (only claim-referenced cols as fold sources; the
   rest only need `r*`/`r_0` evals shipped — same SZ binding); parallelize the
   12 `Q_a` materializations.
4. **Verifier + proof size**: sparse-encode `mult_counts` (2¹⁷×8B ≈ 1 MB →
   support only); closed-form mask-MLEs (periodic 68-row interval structure)
   replacing 12×O(n) passes; **structured-table MLE** for a succinct table
   side (the v2 path — the add table is the graph of add-with-carry).
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
- The honest-first path (`prove_f2_full_with_lookup_adder` + the two-tree
  `gkr_lookup::prove_lookup`) still exists; it's superseded by the bound path
  but kept as scaffolding/tests.
- The **Z[X] sibling task** (integer-path range checks) lives on branch
  `lookup` (off `main-beta`): `documentation/range-check-design.md` +
  `documentation/lookup-methods-design.md`. Untouched since design.
- Memory: `~/.claude/.../memory/f2-lookup-adder-status.md` mirrors this
  state; update it when the state changes.
