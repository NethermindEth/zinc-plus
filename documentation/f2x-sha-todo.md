# SHA-256 F_2 prover — TODO and exploration log

A running ledger of optimization work on the SHA-256 F_2 prover path
(`protocol/src/f2_prove.rs`, `test-uair/src/sha256_f2.rs`, and the
Metal-Blake3 commit pipeline). Each entry captures what was tried,
the measurement that motivated or refuted it, and whether it shipped.

**Keep this current** — any time you investigate an optimization or
design idea and decide *not* to implement it, add an entry here (or
amend an existing one) so the next person doesn't redo the
investigation from scratch. See the "How to use this doc" section
at the bottom.

> **▶ Hadamard-discharge performance — START at
> [`documentation/f2-hadamard-handoff.md`](f2-hadamard-handoff.md).** It
> orients the current state in one page: the discharge is ~92% of the
> nvars=16 prove and **memory-bandwidth-bound** (1536 GF(2¹²⁸) slices, 2.34×
> scaling); the shipped −24.5% fused evaluator; what's been ruled out; and the
> next lever (**word-level AND reduction**, ~32× less data). Detail lives in
> this ledger (the "ROOT CAUSE" + small-value/skip entries) and
> `f2-hadamard-univariate-skip-design.md`.

---

## Fast-described sketch todo's

- [ ] Review how we do PCS open, with lifts and so on. Suspect unnecessary lifting etc
- [ ] GF(128)
- [x] Are we doing multipoint eval? Now yes — see "Multipoint-eval
      phase wired into the F_2 prove path" under Shipped work.
- [ ] SHIFT vector proving — **row-level** shift virtualization (needed
      for the K-discharge work, Issue 1). The per-cell SHIFTR case
      already ships via `F2BitOpVirtualSpec`; see the "Per-cell SHIFTR
      virtualisation in-tree canary" entry under Shipped work below.
- [ ] Review approach to add mod 2^32 in documentation
- [ ] Hadamard product implementation


---

## Branch: Group 8 fully removed (this branch)

This branch (`f2-clean`, cherry-picked from `main-beta`) **does not
include** the Group 8 commits originally on `claude/gkr-virtual-cols`:
the (X^32) ideal swap for additions, the combined CSA-majority +
Binius-carry K column, and the `F2KVirtualSpec`-driven K-col
exclusion from commit + γ-batched open. Test-uair's witness generator
and `Sha256F2Ideal` are at the pre-Group-8 shape (27 binary cols,
`(X^32 - 1)` for additions, no `W_K_*` cols).

The protocol code (`protocol/src/f2_prove.rs`) and the bench
(`protocol/benches/f2_sha256.rs`) have been scrubbed of all K-virt
machinery: no `F2KVirtualSpec` / `F2KSource` struct definitions, no
`k_specs` parameter on any function, no K-tail layout assertions, no
K-aware witness slicing, no `sha_f2_k_virtuals()` helper or call
sites. The intermediate "stub the helper to neutralize the plumbing"
state (commit 7a01531) is now superseded by the deep removal in
commit 57af4e0.

Consequences:
- Issue 1 (trusted K-virtual MLE eval discharge) is structurally
  absent on this branch — there is no K machinery, sound or
  otherwise.
- The "exclude 7 combined-K cols from commit + γ-batched open"
  optimisation is gone (the witness has no K cols to exclude).
- Re-introducing K-virtuals on this branch would require restoring
  the struct definitions and the plumbing through the prove/commit/
  open/verify entry points — not a one-line flag flip.

Entries below referring to "K cols" / "combined-K" / "F2KVirtualSpec"
describe machinery that ran on `claude/gkr-virtual-cols` but is
**not present** on this branch.

---

## Shipped work (chronological, most recent first)

### Fix merged exact-tiling verify bug (ν ≡ 2 mod 8) — `num_compressions` 2-row guard (2026-06-14, working tree)
- **What**: `cols::num_compressions` (`test-uair/src/sha256_f2.rs`) now reserves
  `(2^nv − 6)/68` (was `−4`): the 4-row output anchor **plus 2 trailing
  zero-guard rows**. Added regression test
  `sha256_f2_merged_exact_tiling_verifies` (`protocol/src/f2_prove.rs`, nv9/10/11,
  asserts honest merged prove+verify and the ≥2-trailing-zero invariant).
- **Why**: the unmasked Ch/Maj AND relations (C12/C13) read a `↓2` *forward*
  shift, so the relation at the last rows reads `t+1, t+2`. C13 (Maj) has
  `U = W_A↓2 ⊕ W_A`, whose unshifted term leaves `U = a[n−2]` *nonzero* at row
  `n−2` while the `↓2`/`maj` reads clamp to 0 → the Maj identity fails there
  whenever `a[n−2] ≠ 0`. On an **exact-tiling** trace (`(2^nv−4) % 68 == 0` ⟺
  ν ≡ 2 mod 8: nv = 10, 18, 26, …) the nonzero output anchor lands in the last
  rows, so base-domain `R₀` doesn't vanish, and the merged discharge's msg-only
  (extension-half) shipping drops it → `MergedClaimedSumMismatch`. (C12/Ch is
  safe: its `U = W_E↓2` is a single shifted term → 0 at OOB.) Root cause pinned
  via an `F2_MERGED_DEBUG` per-row `U⊙V==W` check: nv9 0 violations, nv10 exactly
  1 at (rel=C13, row=n−2). The 2-row guard keeps the anchor out of the last two
  rows so those rows are zero and the relation holds trivially.
- **Result**: nv10 + nv18 now verify (were the only failing measured shapes);
  full protocol f2 suite **46/0**, test-uair **27/0**. Only ν ≡ 2 mod 8 counts
  change (drop 1 compression); deployed nv9 (7 comp) and all paper shapes
  nv16/17/19/20/21/22/23 are byte-identical. Paper formula updated `−4`→`−6`;
  GPU-table nv18 3855→3854 comp (within noise). Adder relations were already
  immune (row-masked via `active_rows`); only the unmasked ANDs needed the guard.

### Scaling extended to nv21/22/23 + the in-RAM→over-RAM knee (2026-06-14, working tree)
- **What**: extended the Zinc+ CPU scaling sweep past nv20 to nv21/22/23 (the
  user asked for 2^21/2^22/2^23 rows). Added `implementation.tex §7`
  `tab:scaling-bigmem` + a "Beyond RAM" paragraph. PDF builds clean.
- **Measured** (same-session, CPU, merged, `AB_MERGED_ONLY=1 AB_OPENINGS=987`,
  warm, t=987) — prove / verify / per-doubling:
  - nv16(963) 80.1 / 18.7 ms · nv18(3855) 299.2 / **FAILS** · nv19(7710)
    589.5 / 45.0 · nv20(15,420) 1178.1 / 96.3 — **in-RAM, cleanly LINEAR**
    (nv18→19 1.97×, nv19→20 2.00×).
  - nv21(30,840) **3509.5** / 125.6 (×2.98) · nv22(61,680) **8020.9** / 313.3
    (×2.29) · nv23(123,361) **25,219.5** / 663.0 (×3.14) — **over-RAM,
    SUPER-LINEAR**.
- **Finding — the knee is the 16 GB RAM wall, sharp at nv20→21**: trace ~13 GB
  at nv20 (fits) → ~26 GB at nv21 (1.6× RAM). Below the wall the prover is
  exactly linear (2.0×/doubling); above it, memory compression + SSD swap make
  it super-linear (2.3–3.1×/doubling). This is the HOST's wall, not the O(N)
  compute — on adequate RAM the linear regime extends. **Key comparison point**:
  Zinc+ still *completes* nv23 = 123,361 comp (~104 GB working set, all via swap,
  25 s prove, 133 s total incl. warmup) where Binius64's O(N) circuit *build*
  walls at ~16,000 comp (2^14). So Zinc+ stays tractable ~8× further in N even
  memory-starved. (M4 16 GB, 205 GB free SSD for swap; `ps` is sandbox-blocked
  so peak RSS estimated from the doubling rule + compressor occupancy ~20 GB
  observed at nv21.) Verify keeps its √N growth (45→663 ms over nv19→23).
- **NOTE**: this surfaced the merged-verify exact-tiling bug (nv ≡ 2 mod 8 ⇒
  nv10, nv18 fail `MergedClaimedSumMismatch`) — see the ⚠ entry under "Open
  questions". Prove timings are unaffected (prove always succeeds); only the
  nv18 verify is missing above.

### Per-phase scaling (both systems) + Binius64 preprocessing size recorded (2026-06-14, working tree)
- **What**: added `implementation.tex §7` `tab:perphase` (Zinc+ merged prover by
  phase across nv) + `tab:perphase-b64` (Binius64 prover by phase) + a
  preprocessing-size paragraph. PDF builds clean.
- **Zinc+ per-phase** (ms, CPU, t=987, `OBLONG_PROFILE`, merged): at nv16
  commit 18.0 / linear-PIOP 17.8 / discharge 19.0 / multipoint 11.2 / open 9.6;
  every phase grows ~4× per 4× comp (total ~N^0.97 — LINEAR). commit steepest
  (×4.3, share 7→30%), open shallowest (×3.3, fixed t openings, share 17→9%).
- **Binius64 per-phase** (ms, rate 1/4, prove tree spans Commit-Witness/FRI,
  IntMul, BitAnd, Shift, PCS-Opening): 64c 27.3 / 1024c 116 / 4096c 532 / 8192c
  1370. **SUPER-LINEAR** (×4.6 per ×4 at ~1k, steepening past ×6 by 8k ≈ N^1.36
  at scale) — the **BitAnd reduction** overtakes Shift to dominate (14→41%, its
  "Build BitAnd witness" alone ~1/4 of the prover); GF(2^128) tensors spill cache.
  (Shift reduction dominates at small N, 62→32%.)
- **Binius64 PREPROCESSING (cs + key-collection), via the `save` subcommand**:
  3.7 / 14.4 / 54.7 / **218** / **873** MB at 16/64/256/1024/4096 comp — exactly
  **O(N)** (×4 per ×4); ≥16,384 build >90 s (killed). Measured with
  `./sha256 save --max-len-bytes B --exact-len --cs-path ... --key-collection-path ...`
  then `stat -f%z`. **Zinc+ preprocessing is O(1)** (uniform UAIR + RAA seeds,
  KB-scale, compression-count-independent) — the structural root of b64's O(N)
  verifier + circuit-build wall. Great comparison point.
- **b64 prover sub-span names** (this build): `Prepare/Commit Witness`,
  `FRI Commit`, `[phase] IntMul/BitAnd/Shift Reduction` (Shift has
  `prover_phase_1`/`prove_phase_2`), `[phase] PCS Opening`; verify =
  `Verify Public Input` ($O(N)$, ~99% of verify at scale) + `Verify PCS Opening`.

### Compression scaling sweep (Binius64 vs Zinc+) recorded in the paper (2026-06-14, working tree)
- **What**: added `implementation.tex §7` scaling subsection + `tab:scaling` —
  prove + verify across compression counts, Binius64 (rate 1/4) vs Zinc+ (CPU,
  no-metal, favorable thermal), with proof sizes in the caption. PDF builds clean.
- **Measured** (Apple M-series, CPU, rate 1/4; b64 `--log-inv-rate 2`, kill at
  >70 s/run; Zinc+ merged, `AB_MERGED_ONLY=1`, 6-min cooldown):
  - **b64** prove/verify/proof at $2^k$ comp: $2^4$(16) 27.6 ms/1.6 ms/123 KiB ·
    $2^6$(64) 33.7/3.1/175 · $2^8$(256) 41.6/4.6/187 · $2^{10}$(1024)
    84.1/14.0/249 · $2^{11}$(2048) 155/27.5/264 · $2^{12}$(4096) 480/229/280 ·
    $2^{13}$(8192) 1.22 s/493 ms/299 KiB · **$2^{14}$+ KILLED** (the $O(N)$ circuit
    *build* exceeds 1 min — NOT the prove). b64 can't go below $2^4$ (its example's
    fixed 1024-byte message).
  - **Zinc+** merged prove/verify (CPU, cool), nv→comp: nv12(60) 10.1/4.8 ·
    nv14(240) 23.3/9.3 · nv16(963) 75.0/17.9 · nv17(1927) 148/19.8 · nv19(7710)
    581/45.2 · nv20(15420) 1186/82.9 ms. Proof ≈0.80/1.12/2.57 MiB at nv9/16/20.
  - **Findings**: at matched rate 1/4, the Zinc+ prover is faster and the gap
    WIDENS (~1.1× @1k comp → ~2× @8k); the verifier split is qualitative — b64
    $O(N)$ (3→493 ms), Zinc+ $\sqrt N$ (stays tens of ms; ~11× faster at 8k comp);
    b64 walls at $2^{14}$ where Zinc+ still proves 15,420 comp in ~1.2 s.
- **Tooling**: added `AB_MERGED_ONLY=1` to `sha256_f2_merged_ab_timing` (skips the
  slow fused/oblong arms for a fast, cool scaling sweep). The b64 sweep used a
  `perl -e 'alarm N; exec @ARGV'` timeout (macOS has no `timeout`).
- **Metal arm DONE (2026-06-14): GPU commit gives NO net prover-latency win**
  (corrects the prior "GPU ≈1.25× the CPU prover" claim — now DELETED from the
  paper everywhere). User ran the metal_gpu sweep (`AB_MERGED_ONLY=1 ...
  --features ...,metal_gpu`); 3-run medians land **at or above** the cooled CPU
  sweep at *every* scale: nv16 GPU 107 ms (best 97) vs CPU 75 cooled / 93 warm;
  nv20 GPU 1708 vs CPU 1190. Commit is a minority phase and GPU dispatch/transfer
  overhead swamps the offload at these sizes — the GPU's value is throughput /
  freeing the CPU, not latency. Full GPU prove sweep (ms): nv9 9.8 · nv10 12.0 ·
  nv11 16.2 · nv12 19.2 · nv13 27.6 · nv14 38.8 · nv15 68.7 · nv16 107.3 · nv17
  203.1 · nv18 402.5 · nv19 784.0 · nv20 1708.3 — clean **LINEAR in N** (~2×/nv,
  ~2.2× at the nv19→20 top). Recorded as `tab:scaling-gpu` in §7; the "1.25×
  with GPU" line removed from `tab:binius-rate` + Part II
  (`cmp-{results,analysis,intro}`). (Harness verify is LUT-rebuild-inflated:
  nv16 23.6 ms ≠ the deployed ~12 ms criterion verify; prover figure unaffected.)
- **Sandbox note**: the agent sandbox blocks GPU (`Device::system_default()` →
  None in-sandbox; works sandbox-off), so the user runs the metal sweep in their
  own terminal.

### Main sumcheck rounds 2..n eq-factored — `uair:sumcheck` ~7.4 → ~6.0 ms at the merged arm (2026-06-14, working tree)
- **What**: group-0 of the main GF(2^128) `MultiDegreeSumcheck` — the pure
  `eq(·;r)·weighted_col` degree-2 zerocheck — had only its **round 1**
  eq-factored (`F2EqColRound1FastPath`); rounds 2..n fell back to the generic
  per-point comb gather (~5 GF(2^128) mults/slot). Added `F2EqColRoundEvaluator`
  (`protocol/src/f2_prove.rs`), a `RoundPolyEvaluator` attached via
  `.with_round_evaluator`, that reuses the round-1 algebra every round: the
  framework-folded eq table still factors as `eq[2b']=(1+r_j)·E[b']`,
  `eq[2b'+1]=r_j·E[b']` with `E[b']=eq[2b']+eq[2b'+1]` (char-2) and
  `r_j = ic_eval_point[round−1]`, so `S_A=Σ E·col[2b']`, `S_B=Σ E·col[2b'+1]`
  give `M(0)=(1+r_j)·S_A`, `M(1)=r_j·S_B`,
  `M(2)=eq(2,r_j)·((1+2)·S_A+2·S_B)` — ~2 mults/slot. **Byte-identical** (same
  `[M(0),M(1),M(2)]` at `{0,1,X}`), so the verifier is untouched.
- **Why**: the ledger's "linear PIOP is the under-optimized third" lever (1) —
  rounds 2..n of the main sumcheck were the last un-eq-factored degree-2 block
  after the round-1 fast path and the merged group-1 evaluator shipped. Port
  source is this tree's own round-1 fast path (and `oblong_and.rs`'s
  `QuadraticMleCheckProver`), NOT the lookup branch's `eq_factored.rs` (wrong
  degree / GKR-shaped) — see the cross-branch UPDATE under that "Identified"
  entry.
- **Result**: clean same-binary A/B (`F2_NO_EQCOL_EV=1` toggles the evaluator
  off) at nv16 / t=987, merged arm, warm: `uair:sumcheck` **~7.4 → ~6.0 ms**
  (OFF [7.89, 7.40, 6.56] vs ON [5.90, 6.11, 5.99] — non-overlapping clusters,
  ≈ −1.4 ms / −19% of the scope). End-to-end merged prove ~76 ms either way
  (the −1.4 ms is within total-time noise; group-0 is the small piece, as
  predicted — the big sumcheck scopes are now `mp:sumcheck` ≈5-6 ms and the
  already-fused merged group-1). Tests: new
  `f2_eq_col_round_evaluator_matches_generic` cross-check (evaluator ==
  definitional round poly at every round j=1..n on framework-folded state) +
  all 45 protocol f2 prove/verify roundtrips (merged / oblong / tamper arms)
  green.
- **Toggle kept**: `F2_NO_EQCOL_EV=1` (runtime, like `OBLONG_NO_EQSPLIT`) for
  future A/B and as a safety fallback.
- **Next on this block**: the linear-PIOP eq-factoring lever is now fully
  spent (round 1 + rounds 2..n on group-0; group-1 already fused). `mp:sumcheck`
  is the largest remaining sumcheck scope but its comb is mixed
  (`eq_r·precombined` + Σ selector·data) ⇒ a binius-style shift-eval rewrite,
  flagged low-ROI. Commit-side (rate 1/2, Σ/σ virtualization) is the next real
  lever.

### Hadamard bench group defaults to measuring only `Prove-Hadamard-Merged` (2026-06-14, working tree)
- **What**: `bench_hadamard_compare` in `protocol/benches/f2_sha256.rs` now
  registers `Prove-Hadamard-Merged` first, then early-returns unless `HAD_ALL` is
  set in the env. So the plain
  `cargo bench -p zinc-protocol --bench f2_sha256 --features parallel,simd,unchecked,metal_gpu -- "Zinc\+ F_2 SHA-256 Hadamard"`
  command measures **only** the deployed merged-discharge prove arm.
- **Why**: the group otherwise registered 11 arms (4 prove: NoHadamard / Hadamard
  / Oblong / Merged; 3 discharge-only: Fused / Oblong / Oblong-GF8; 4 verify) and
  *also* built 3 extra full proofs (no-had / had / oblong) purely to feed the
  verify arms — all of that ran on every default invocation, which is slow when
  you're iterating on the merged prover alone (esp. at nv=20/21).
- **Result**: default run does just `Prove-Hadamard-Merged` and skips the three
  extra proof builds. `HAD_ALL=1` restores the full A/B suite (verify arms,
  discharge-only A/B, proof-size report). Composes with the existing `HAD_NVARS`
  env filter. Pure bench-harness ergonomics — no prover/verifier code touched.
  (The pre-existing unused-import warning for `F2LinearOpener` is unrelated.)

### Companion note rewritten to the merged discharge + clean re-measure (2026-06-14, working tree)
- **What**: rewrote `documentation/f2x-sha256-snark-doc/` so the Hadamard
  discharge reads as the **merged (inline ⟨Z_H⟩)** construction "as if always" —
  no trace of the prior oblong framing. The narrative: reading cell bits as
  Lagrange coefficients makes $\AND$ the ideal-membership claim
  `L(u)L(v) ≡ L(w) mod Z_H`, discharged by the SAME ideal-check + $\psi_\alpha$ +
  Step-4 sumcheck as the linear part, at the shared point $\alpha$ (so $\psi_z$ is
  the Lagrange weighting read at $z=\alpha$), bound by one extra projection of the
  same $a'$. Removed: separate $z$, standalone zerocheck, Phase-1/Phase-2, $r_H$,
  z-block, second transcript point. `hadamard.tex` fully rewritten (title
  "Hadamard as Ideal Membership"; new `eq:had-ideal`; kept externally-referenced
  labels `lem:psi-z-sound`, `sec:had-binding`, `eq:psi-z-binding`, `rem:gf8`,
  `rem:adder-trusted`; renamed `sec:oblong`→`sec:had-pipeline`). Consistency edits
  in `introduction/preliminaries/arithmetization/piop/commitment/soundness/
  implementation` + the Part II `cmp-*.tex`. The "two projections $\psi_\alpha,
  \psi_z$ / dual read-off of one $a'$" through-line survives intact (it is
  strengthened — both now live at $\alpha$). PDF builds clean (latexmk EXIT=0, 0
  undefined refs).
- **Clean re-measure** (criterion, nv16, no-metal, proper 10-sample medians,
  machine back at the paper's thermal scale — fused 494.6 ≈ the note's prior 487):
  linear-only **55.3** / fused **494.6** / oblong **112.2** / merged **100.4** ms
  prove; merged **verify 11.5** ms. **Deployed (merged) discharge ≈ 4.9× faster
  than fused**, adds ≈45 ms over the linear part. `implementation.tex §7` table +
  e2e prose updated to these; the stale cmp footnote `136`→`100` ms fixed.
- **Caveat (recorded, not done)**: the Part II per-phase table (`cmp-results.tex`,
  `tab:phases`) is a **GPU-commit** measurement (98.4 ms total, with the oblong
  per-phase split incl. the 2.8 ms z-block). I swapped its terminology to merged
  (Discharge=Hadamard, z-block→Binding) but did NOT re-measure its GPU per-phase
  numbers — the sandbox has no Metal (`Device::system_default()` → None). A GPU
  re-measure would refine the merged per-phase split (merged discharge ≈20 ms vs
  the table's 32.1; no z-block); flagged for a machine with Metal access.

### Merged-group fused round evaluator — uair:sumcheck ~11 → ~6.6 ms (2026-06-14, working tree)
- **What**: the merged Hadamard group-1 (degree-3, `eq·Σ_k γ_h^k(ZU_k·ZV_k + ZW_k)`,
  `1+3·k_rel` ≈ 43–49 operand MLEs) ran the generic `MultiDegreeSumcheck`
  per-point value-array gather over ALL operand MLEs every round (no
  `round_1_fast`, no `round_evaluator`). Added `MergedHadamardRoundEvaluator`
  (`protocol/src/f2_merged_hadamard.rs`) attached via `.with_round_evaluator`,
  mirroring the standalone discharge's `HadamardRoundEvaluator`: relation-outer /
  point-inner streaming, coefficient-space Karatsuba for `ZU·ZV` (3 muls), `eq`
  applied in a second pass, chunked + rayon. **Byte-identical** — same cubic round
  message evaluated at `{0,1,X,X+1}`.
- **Why**: instrumented uair-self (nv16, t=987) — `uair:sumcheck` = ~11 ms, the
  largest linear-PIOP block; the bulk is group-1 (NOT group-0's 2-MLE `eq·col`),
  which had no fused evaluator. ψ_α projection was only ~2.6 ms (GPU-projection
  lever dropped).
- **Result**: **`uair:sumcheck` ~11 → ~6.6 ms (−4.4 ms, robust same-scope)**;
  merged total ~76 ms this run. Tests: 44 protocol f2 (incl. merged/oblong
  roundtrips + tamper arms) + 60 piop, all green.
- **Session linear-PIOP total**: this + the multipoint next_mle parallelization
  (−3.7 ms) = **~8 ms (~9%) off the ~93 ms prover**, both byte-identical/bit-
  identical, both validated.
- **Next on this block**: group-0 `eq·col` is now the small remaining piece; a
  Round1FastPath already covers its round 1. W-bar-precombine (49→34 MLEs,
  protocol-visible) and a round-1 fast path for the merged group remain
  identified. Σ/σ virtualization (commit ↓) still queued — now cheaper to
  evaluate given the lower per-shift multipoint cost.

### Multipoint-eval: parallelize the next-MLE / down-col build (~3.7 ms off the prover, 2026-06-13, working tree)
- **What**: `MultipointEval::prove_as_subprotocol_with_pointed_shifts`
  (`piop/src/multipoint_eval.rs`) built the per-pointed-shift `next_c_r` selector
  MLEs and cloned the source down-columns in a **serial** `.iter().map()`. Swapped
  to `cfg_iter!` (rayon under `parallel`). Order-preserving `collect` ⇒ bit-identical.
- **Why**: instrumented the multipoint internals (`mp:next_mle / mp:precombine /
  mp:sumcheck` prof scopes) at nv16/t=987 — the ~12 ms multipoint split as
  **sumcheck ~4.5 + next_mle ~4.7 + precombine ~2.0 ms**, and next_mle was serial
  (the precombine, long assumed the bulk, is the smallest).
- **Result**: **mp:next_mle ~4.7 → ~1.0 ms** (clean same-scope A/B; ~3.7 ms is the
  robust figure — arm totals also fell E 92.8→82.1 / C 93.5→85.5 but that's partly
  thermal). Multipoint ~12 → ~7.5 ms. Generic piop win — helps any pointed-shift
  multipoint incl. the integer pipeline. Tests: 17 piop multipoint + 4 protocol
  F_2 roundtrips green; ab_timing C+E prove+verify roundtrips pass.
- **Tooling**: added `mp:*` prof scopes (kept) and an `AB_OPENINGS` env knob to
  `sha256_f2_merged_ab_timing` (default 4; use 987 for deployed-shape timing).
- **Next multipoint levers (Tier-1, not yet done)**: (a) eq-factor `mp:sumcheck`
  — its main term is `eq_r·precombined`, eq-factorable like the round-1 fast path
  (~4.5 ms, ~1–2 ms headroom); (b) avoid the down-col clones (the sumcheck owns
  clones of `trace_mles` cols — needs a borrow-API). The main UAIR sumcheck rounds
  2..n also still lack eq-factoring — see the linear-PIOP entry under "Identified".

### 2-AND Ch arithmetization — K_H 16 → 14, witness 20 → 18 cols (2026-06-13, working tree)
- **What**: replaced the two-AND Ch decomposition `Ch = (e∧f) ⊕ (¬e∧g)`
  (committed columns `W_UEF`, `W_UNEG_E_G`) with the **single-AND Binius Ch
  identity** `Ch(e,f,g) = g ⊕ (e ∧ (f⊕g))`. One committed AND product
  `W_UCH = e ∧ (f⊕g)` (Hadamard `W_E^↓2 ⊙ (W_E^↓1 ⊕ W_E) = W_UCH^↓2`); `Ch`
  enters the `T_1` chain as the free XOR `g ⊕ u_ch` (`g = W_E^↓1`). Files:
  `test-uair/src/sha256_f2.rs` (cols, signature shifts, `constrain_general`
  C6, trace gen, PA_C, MLE vec), `protocol/src/f2_hadamard.rs`
  (`Sha256F2HadamardLayout` → 2 AND + 12 adders; new `F2AdderSpec.extra_y_terms`
  XOR-fold so one modular add absorbs `g ⊕ u_ch`), `f2_prove.rs` + the
  `f2_sha256` bench glue/asserts.
- **Why**: bring the deployed arithmetization in line with the companion-note
  Part I / Binius64 comparison (documentation/f2x-sha256-snark-doc, branch
  `f2-clean-lookup`), which is written and measured at **K_H = 14** (2 AND +
  12 adder carries, "12 of the 14 deployed Hadamards"). Also trims the
  discharge: −1 AND Hadamard and −1 `T_1` adder step (chain 5 → 4 steps;
  `W_T1_S4` removed), so −2 committed columns (20 → 18) and one fewer per-step
  κ compensator (`NUM_KAPPA` 13 → 12, bit-op virtuals 12 → 11).
- **Result**: correct end-to-end. `cargo test -p zinc-protocol
  --features parallel,simd` — all 6 `sha256_f2_*` prove/verify roundtrips pass
  (full 14-relation honest prove+verify, the 2 AND relations, the 12 adders,
  merged + oblong arms) and all 29 `hadamard` unit tests pass (the new
  `extra_y_terms` path is exercised by the SHA roundtrips; synthetic adder
  round-trips unaffected). test-uair 4/4. Bench compiles. **Perf delta not yet
  benchmarked** — expected small prover win (−1 AND zerocheck row-set, −1 adder
  step); run `f2_sha256` "Hadamard" A/B to quantify. The trusted-adder-parent
  soundness gap (ledger Issue 1) is unchanged — now 12 (was 13) carry Hadamards.

### Merged (inline) Hadamard discharge — the Lagrange-reinterpretation construction, end-to-end + optimized (2026-06-12)
- **What**: a new prove/verify arm `prove/verify_f2_full_with_merged_hadamard`
  (`f2_prove.rs`) + the module **`protocol/src/f2_merged_hadamard.rs`** (module
  doc has the math). The Hadamard relations ride the MAIN pipeline instead of
  the standalone oblong zerocheck: reinterpreting cell bits as subspace-Lagrange
  coefficients turns bitwise AND into the ring congruence
  `L(u)*L(v) = L(w) mod Z_H`, so the discharge is (1) a **Phase-1 ideal-check
  message** absorbed before alpha — the D extension-domain evals of
  `R0 = sum_rows eq(row; r_had) * sum_k gamma_h^k (L(U_k)L(V_k) - L(W_k))`, a
  base-domain-vanishing polynomial (the `<Z_H>` membership, so the base half
  never travels); (2) a **degree-3 group in the existing Step-4
  `MultiDegreeSumcheck`** over the `L_b(alpha)`-folded operand columns, claimed
  sum `R0(alpha)` (verifier-reconstructed via `r0_at`); (3) **end claims at
  r***: Lagrange-weight pair evals via the weight-generic
  `pair_alpha_evals`/`derive_operand_parents` machinery, folded through the
  SINGLE multipoint (pointed-shifts at r*, only AND-referenced cols appended),
  bound by `psi_W(a') == sum_g gamma_g * lag_r0[g]` (the psi_z binding at
  `z = alpha`). No `r_H`, no oblong Phase 2, no all-cols z-block/`z_up_evals`.
  Soundness is the oblong's verbatim — the Phase-1 messages are in bijection
  with the oblong's off-subspace values (`psi_alpha . L = psi_z|_{z=alpha}`).
  Adders ride with trusted parents (same posture as the fused/oblong arms).
- **The GF(2^8) fast lane (as shipped)**: Phase-1 runs the byte-lookup NTT with
  the eq-split on the hybrid zerocheck point
  `r_had = [s0,s1,s2] ++ r_IC[3..]` (`merged_eq_point`) — the scheme's
  deterministic small challenges on the LOW row vars + the shared IC point on
  the rest; the kernel's composite weighting equals `build_eq_x_r(r_had)`
  exactly (both little-endian, embed is a field hom — pinned by the
  `hybrid_eq_table_matches_kernel_convention` canary). All relations run as
  ONE stacked kernel pass (`Gf8Scheme::round_message_with_eq_big`, new
  additive method): `gamma_h^k` folds into relation k's per-chunk big-eq
  weights, full `K*2^nu`-range parallelism. End claims come off the sumcheck's
  residual state for free (`residual_evals`: the post-last-round length-2
  tables fold once by the final challenge); AND pair evals project each
  referenced col once (`pair_evals_dedup`). Shared-helper win:
  `f2_hadamard::cell_mask`/`cell_from_mask` now use `F2PackU64` (single field
  read/store under `simd` instead of per-bit Boolean loops) — also speeds the
  fused + oblong operand builders. Ported `zinc_utils::prof` (env-gated
  per-region timers, `OBLONG_PROFILE=1`) to support the optimization loop.
- **Measured** (`sha256_f2_merged_ab_timing`, all 16 SHA relations = 3 ANDs +
  13 adders, trusted adders, Apple M-series, release, same-run): nu=16 prove —
  **E merged 109.8 ms vs C oblong-GF8 120.8 ms (E ~9% faster)**, A fused
  582.7 ms; nu=9 prove — E 4.6 vs C 4.4 ms (tied within run variance).
  Verify — nu=9: **E 0.5 vs C 1.1 ms (2x faster)**; nu=16: E 5.6 vs C 5.0 ms
  (E pays two `Gf8Ntt` lut builds per verify — see the OnceLock item below).
  Tests: 2 module unit tests (incl. the eq-convention canary), 3-col roundtrip
  with 5 tamper arms (specific variants for the claimed-sum + closing-eval
  checks), operand-features roundtrip (row-shift/complement/XOR combo, delta>0
  pointed shifts at r*), real-SHA all-16 e2e with an unreferenced-col binding
  tamper arm. All 65 protocol + 150 poly tests green.
- **Identified, not implemented** (the optimization backlog for this arm):
  (a) the degree-3 group's generic rounds (~9 ms @nu=16) — W-bar-precombine of
  ALL W-operand MLEs (they enter the comb linearly; one gamma-weighted MLE;
  sound — adder parts are trusted anyway, AND `zw(r*)` stay derived from the
  bound pair evals; group 49 -> 34 MLEs here) and/or a Round1FastPath for the
  merged group (round 1 is ~half the group cost; group-0's char-2
  eq-factoring pattern applies). (b) **words-direct operand building**: build
  operand u32 words straight into the stacked layout (skip the
  `build_operand_column` -> `BinaryPoly` -> `cell_mask` -> concat round-trip;
  fold the group MLEs from the same words via `fold_word_at`) — Phase-1 is
  now mostly operand materialization, ~2x headroom; also removes the
  **large-nu memory spike** (at nu=22 the three stacks + operand cols are
  ~GB-scale transient — latent at the deployed nu=9; the full fix at scale is
  computing operand words on the fly per kernel chunk). (c) eq-table hygiene:
  move (don't clone) the group's eq table (64 MB @nu=22); build
  `eq_big(r_IC[3..])` once and outer-product into BOTH group-0's `eq(r_IC)`
  and the merged `eq(r_had)`; **OnceLock the `Gf8Scheme`** (built 2x per
  prove and 2x per verify for the lut — the nu=16 verify gap). (d) `lag_r0`
  for unreferenced cols via the open's per-column `a'_g` if exposed (~2 ms).
  (e) **The structural follow-up**: pair the merged AND discharge with a sound
  adder treatment (the lookup-adder workstream lives on `f2-clean-lookup`;
  with only the AND relations in the merged group its cost nearly vanishes).
- **Provenance**: designed + first implemented on `f2-clean-lookup` (where the
  A/B also ran against the lookup arms), then ported here as its home branch
  per review; the lookup branch keeps no copy (its ledger entry points here).
- **Reproduction / measurement caveat (2026-06-13)**: a fresh same-run A/B on
  Apple M-series (release, `parallel,simd,unchecked`, no `metal_gpu`) did **not**
  reproduce the "E ~9% faster prove" headline — it landed within run-to-run
  noise: nu=9 prove E 4.4 ≈ C 4.4 ms (tied); nu=16 prove **E 102.1 vs C 98.2 ms
  (E marginally *slower* this run; E's 3rd run 136.7 ms is the thermal tail)**.
  The merged arm's *reliable* win is **verify** (nu=9 E 0.6 vs C 0.9 ms; nu=16
  E 4.3 vs C 4.5 ms) and the structural collapse (single point r*, ψ_z=ψ_α at
  z=α, one open, no oblong Phase-2 / no all-cols z-block), **not** prove time.
  Treat merged-vs-oblong *prove* as a wash at the deployed nu=9; sell the merge
  on verify + soundness/structure. Always A/B in one process
  (`sha256_f2_merged_ab_timing`, `AB_NVARS=`), never across `cargo bench` runs.
- **Criterion `Prove-/Verify-Hadamard-Merged` arms ADDED (2026-06-13, working
  tree `protocol/benches/f2_sha256.rs`).** Before this, commit `39c00b6` did not
  touch the bench, so `cargo bench -- "Zinc+ F_2 SHA-256 Hadamard"` measured only
  the old `Prove-/Verify-{NoHadamard,Hadamard(fused),Hadamard-Oblong}` +
  `Discharge-*` arms and **could not** show the merge — the merged path was
  exercised only by the `#[ignore]` test `sha256_f2_merged_ab_timing`. The new
  arms call `prove/verify_f2_full_with_merged_hadamard` at `BENCH_NUM_OPENINGS`
  (987). **Bit-ops caveat**: the merged *verify* does not thread bit-op virtuals
  (signature only, unused in the body), so the merged arms pass `&[]` bit-ops and
  commit the 2 SHR cols the oblong arm virtualises (~few-% heavier commit). Wiring
  bit-op virtuals through the merged verify/open is a clean follow-up that would
  make the arm strictly apples-to-apples with oblong.
  - **Measured (criterion, no `metal_gpu`, same run, save-baseline
    `merged-nometal`, Apple M-series):** nv9 prove — Oblong 4.60 vs Merged 4.72 ms
    (Merged ~2.6% slower, incl. the 2 extra committed cols); nv16 prove — Oblong
    113.6 vs Merged 110.8 ms (Merged ~2.4% faster); verify nv9 1.63 vs 1.66 ms,
    nv16 12.4 vs 12.6 ms — **tied** (full verify is dominated by the 987
    column-opening checks, NOT the discharge; the ab_timing test's "merged verify
    ~2× faster" used only 4 openings, which isolates the discharge but is not the
    deployed shape). **Bottom line: at the deployed 987-opening shape merged ≈
    oblong on both prove and verify** — the merge is a structural / single-point /
    one-open / soundness simplification, not a prover speedup. Sell it as such.
  - The e2e `Prove`/`Steps` groups still run the `&[]`-spec no-Hadamard path, so
    the discharge is not in the headline e2e number at all — wiring the merged
    discharge into e2e is a separate decision.
- **GOTCHA — cross-run nv16 "regressions" in this bench are THERMAL, not code.**
  A 2026-06-13 run recorded criterion `change` ≈ **+25–41% at nv16 on every arm
  — including `Prove-NoHadamard` (+26.7%), which has zero Hadamard/projection
  work** — while nv9 prove arms were flat (±0–3%). A projection merge cannot
  slow the no-Hadamard pipeline; the uniform nv16 shift is the machine heating
  across heavy back-to-back arms (worse with `metal_gpu`) vs a cooler saved
  baseline. nv20/21 additionally swap on a 16 GB box (~13/26 GB). Use same-run
  A/B for signal; ignore this bench's cross-invocation absolutes.
- **GOTCHA — `sha256_f2_merged_ab_timing` panics under `--features metal_gpu`**:
  `Device::system_default()` returns `None` in the test binary
  (`zip-plus/src/metal_gpu/mod.rs:143`, "No Metal device found"). Run the A/B
  test **without** `metal_gpu` (`--features parallel,simd,unchecked`); the
  criterion bench is unaffected.

### Prover-path parallelization sweep — Prove-Hadamard-Oblong 154 → ~100 ms (commits `d8cfe4b`, `1a69adc`, `7ea7dff`)
- **What**: a sweep for *serial code on a parallel machine* across the sound-oblong
  prove path. The discharge/open/uair/commit are all rayon-parallel, but several
  hot loops added by the binding/open/discharge were plain `.iter()`/`for` — on a
  10-core M4 that cost ~10× what it should. Profiled each phase, parallelized the
  serial ones (all bit-identical → no proof/soundness change), and re-measured.
  - **`7ea7dff` — the big one: the oblong discharge's Phase-1→Phase-2 word-fold**
    (`oblong_and.rs`, folding every stacked operand word at `z`) was a sequential
    `.iter().map()` — profiled at **~31 ms / 62% of the discharge** at nvars=16
    (~1M stacked words ×3), while round-message + Phase-2 were already parallel.
    `par_iter` → ~8 ms. **Discharge ~50 → ~28 ms; Prove-Hadamard-Oblong 128 → ~100 ms.**
  - **`d8cfe4b` — binding loops**: `oblong_binding_data`'s 13-adder build+project
    loop + `pair_alpha_evals` (shared with fused) + the prover's `z_block`/`z_up`.
    Binding overhead ~48 → ~18 ms.
  - **`1a69adc` — open `combined_row`** (γ-fusion + cache-blocking — see the
    un-lifted-open regression entry): 13.5 → ~7 ms.
- **Result**: **Prove-Hadamard-Oblong 154 → ~100 ms (nvars=16), ~4.9× faster than
  the fused discharge (487 ms)**; verify unchanged (~11 ms). All 60 protocol + 146
  poly tests green.
- **Component breakdown after the sweep (nvars=16, Apple M4, ~100 ms)**: commit ~25 ms,
  discharge ~28 ms (round_message ~9 + word-fold ~8 + phase2 ~9), rest ~27 ms
  (binding ~17 + open ~9 + mp), uair ~13 ms. **Every major component is now
  rayon-parallel** — the easy "serialcode" wins are exhausted.
- **Lesson (for the next agent)**: any new prover hot loop on this path MUST use
  `cfg_iter!`/rayon; serial code stands out sharply (~5–10×) against the otherwise
  fully-parallel pipeline. Profile with per-phase `Instant`/eprintln (criterion's
  ±5 ms median noise hides changes < ~8 ms).
- **Remaining opportunities (harder / smaller, NOT pursued)**:
  - **Commit (~25 ms) — Metal GPU INVESTIGATED (re-measured on the current oblong
    path)**: the `metal_gpu` commit + α-projection offload is **already implemented,
    wired, and functional** (`phase_commit.rs:176` leaf-hash when `num_leaves ≥ 256`;
    `f2_prove.rs:1519` `project_columns_with_powers_gpu_batched`). Fresh A/B
    (Apple M4, `--features parallel,simd,unchecked,metal_gpu`, nvars=16, **unsandboxed
    — the sandbox blocks Metal: "No Metal device found"**): Prove-NoHadamard **55 →
    ~47 ms** (−8 ms), Prove-Hadamard-Oblong **~100 → ~96 ms** (−~8 ms, partly within
    noise). **Modest at nvars=16** because the CPU commit (~25 ms) is already
    rayon-parallel and GPU dispatch has fixed overhead; the win **grows with nvars**
    (more leaves → more GPU parallelism — the ledger's `9b8f178` entry showed
    nvars=22 e2e −19%). So there is **no CPU code change to make** — it's a
    deployment/feature choice (needs a GPU + unsandboxed). **The real un-GPU'd gap is
    the discharge (~28 ms, the single biggest component): the GF(2⁸) NTT round-message
    + the word-fold are not offloaded.** **GPU discharge INVESTIGATED + rejected** —
    the word-fold offload (reusing the α-projection kernel) measured **~50% SLOWER**
    than the parallel CPU fold at nvars=16 (GPU dispatch overhead > the ~3M-cell work;
    CPU-wins regime). See the "GPU discharge … SLOWER than parallel CPU" entry under
    *Investigated, didn't help*.
  - **Binding multipoint** still folds **all** witness z-cols; folding only the
    ~5 AND-referenced ones (the rest are bound by the ψ_z check alone) saves ~5 ms
    but is a **soundness-sensitive** z-block-indexing restructure.
  - **Discharge round_message/phase2 (~9 ms each)** are parallel + algorithmically
    tuned (GF(2⁸) NTT, Gruen MLE-check) — limited CPU headroom (see GPU note above).
  - **Open per-col over 20 cols not 8** (the bit-op-virtual-derivation idea, see
    its own entry) — but the per-col cost is now small post-`combined_row` fix.

### Sound oblong-Hadamard ψ_z binding — NEXT STEP #1 DONE (commits `91eebec`, `4b5aca8`, `317a83a`, `aa05f21`)
- **What**: the Binius64 oblong AND-zerocheck's `ψ_z` operand evals are now **bound
  to the PCS commitment** end-to-end, not trusted against in-memory columns.
  `prove_f2_full_with_oblong_hadamard` / `verify_f2_full_with_oblong_hadamard`
  (`f2_prove.rs`, D=32 impl) + the `F2OblongHadamardProof` wire type + the
  `f2_oblong_hadamard` helpers (`oblong_binding_data_gf8`,
  `oblong_verifier_binding_gf8`, `oblong_tie_from_bound`, `verify_oblong_zerocheck_gf8`).
- **Mechanism (dual-projection multipoint, riding the un-lifted open)**: the prover
  runs the discharge (capturing `[z,γ]`), appends z-projections of **all witness
  primary cols** to the multipoint trace with `ψ_z(col)(r*)` up-evals, folds the
  `ψ_z(col↓Δ)(γ_word)` AND-pair claims as **pointed-shifts**, and reduces everything
  to the single open point `r_0`. The verifier mirrors it, then: (a) the open exposes
  its batch `γ`; (b) the **ψ_z binding check** `ψ_z(a') == Σ_g γ_g·z_r0_evals[g]`
  ties the z-evals to the committed bit-slice claim `a'` (the same `a'` that yields
  `ψ_α` via the open's Check 2 — this is why the un-lifted open was the prerequisite);
  (c) `oblong_tie_from_bound` recombines the now-bound AND pair-evals + the trusted
  adder parents to the zerocheck operand evals.
- **Scope: AND relations bound, adders trusted (= fused-discharge soundness parity,
  Issue 1)**. Adder operands use bit-level `SHR¹`/`β=Maj`/`X·c` (the D-dimension ψ_z
  projects away), so they don't decompose into row-shift `(col,Δ)` pair-evals; their
  parents ship trusted, exactly as the fused path does.
- **Prerequisites shipped**: `91eebec` exposed the prover eval-point `[z,γ]`;
  `4b5aca8` cleared the Stage-1b bench debt (proof-size migration + removed the stale
  `bench_micro_verifier_open` suite); `317a83a` defined the proof type + finalized the
  design (the critical *all-witness-cols* binding correction).
- **Result**: round-trip test (`prove_then_verify_f2_full_with_oblong_hadamard_roundtrips`)
  green — honest accept, flipped-`W` → oblong-zerocheck reject, tampered-`z_r0` →
  ψ_z-binding/multipoint reject, tampered AND pair-eval → multipoint/tie reject. **All
  60 protocol + 35 poly tests green; benches compile.**
- **e2e prove/verify A/B vs the fused discharge — DONE** (`Verify-Hadamard-Oblong`
  bench arm added; Apple M4, `target-cpu=native`, `parallel simd unchecked`):

  | arm | nvars=9 | nvars=16 | discharge overhead @16 |
  | --- | --- | --- | --- |
  | Prove-NoHadamard | 2.28 ms | 59.9 ms | — |
  | Prove-Hadamard (fused, ψ_α) | 7.72 ms | 482.5 ms | +422.6 ms |
  | **Prove-Hadamard-Oblong (sound)** | **5.05 ms** | **136.2 ms** | **+73.0 ms** |
  | Verify-NoHadamard | 1.24 ms | 10.52 ms | — |
  | Verify-Hadamard (fused) | 1.44 ms | 10.83 ms | +0.31 ms |
  | **Verify-Hadamard-Oblong (sound)** | **1.57 ms** | **11.0 ms** | **+0.5 ms** |

  **The bound discharge is ~3.6× faster to prove than fused at nvars=16** (136 vs
  487 ms), **with verify essentially unchanged** (+0.1 ms vs fused, ~1% — both
  ~11 ms, dominated by the shared open/Merkle verify). The run also **validates the
  sound path on the real SHA arithmetization** (adders, public cols, bit-op virtuals)
  beyond the toy round-trip test — both `proof_oblong` and `Verify-Hadamard-Oblong`
  are `.expect()`-guarded and succeeded.
- **Binding overhead optimized (commit `d8cfe4b`): ~48 → ~18 ms (parallelized the
  serial loops)**. The first sound impl added ~53 ms of binding on top of the unbound
  discharge — but most was *serial code on a parallel machine* (the discharge/open/uair
  already use every core). Parallelizing the three sequential loops (pure rayon, no
  soundness change) recovered the bulk (phase timings, Apple M4 nvars=16):
  `oblong_binding_data` (the 13-adder build+project loop + `pair_alpha_evals`)
  **28.5 → 7.0 ms**; the prover's `z_block` projection **8.9 → 2.3 ms**; `z_up_evals`
  **3.1 → 0.9 ms**. Prove-Hadamard-Oblong **154 → 136 ms**. Lesson: any new prover hot
  loop on the F_2 path must use `cfg_iter!`/rayon — serial code stands out sharply
  against the otherwise fully-parallel pipeline.
- **Still open (follow-ups, not blockers)**: the multipoint now folds **all** witness
  z-cols (~+7 ms of the remaining binding); folding only the AND-referenced ~5 cols
  (the others are bound by the `ψ_z` check alone, no multipoint needed) would shave
  most of that — but it's a soundness-sensitive restructure (z-block indexing), so
  deferred. Also: a sound adder-carry binding (Issue 1); re-authoring the per-step
  verify micro-benches against the un-lifted open if that breakdown is wanted.

### Un-lifted GF128[X]<D> open — bit-slice-preserving open binds *both* ψ_α and ψ_z (commits `7827683`, `192e173`, `b840839`)
- **What**: rewrote the F_2 PCS open (`prove_f2_open` /
  `verify_f2_open_with_virtuals` / `F2OpenProof`, `protocol/src/f2_prove.rs`)
  to keep the eq-tensor in `GF(2^128)` instead of lifting it to `F_2[X]` via
  `AlphaPolyBasis`. The per-column claim is now `a' = Σ_b c_b·X^b`, a clean
  degree-`<D` `GF128Poly<D>` whose `D` `GF(2^128)` coefficients **are** the
  bit-slice MLE evals `c_b = MLE[v_b](ρ)`. Staged: Stage 0 (`7827683`) the
  `column_bitslice_evals` kernel + math test; Stage 1a (`192e173`) the
  `GF128Poly<D>` primitives (`gf128poly_accumulate_cell/_bits/_scaled/_project`,
  `F2AddAssign`/`FromRef` chain) + `encode_gf128_lin_open<D>` reusing the generic
  `encode_f2_lin` kernel (the RaaF2 encoder is F_2-linear ⇒ GF128-linear per
  coefficient — **no new encoder**); Stage 1b (`b840839`) the open rewrite itself.
- **Why (the user's insight)**: the lifted open was *monomial-α-specific* — its
  claim was only meaningful at α, so `ψ_z` (the discharge's subspace-Lagrange
  projection) could not ride it. Keeping the eq-tensor un-lifted makes `a'` carry
  the bit-slice evals as its coefficients, so **one bound object yields both
  projections**: `ψ_α(v)(ρ) = Σ_b c_b·α^b` (`gf128poly_project` with α-powers — the
  main / C16–C18 / IC claim, ψ_α stays the ring hom those degree-2 pins need) **and**
  `ψ_z(v)(ρ) = Σ_b c_b·L_b(z)` (same project with `base_lagrange_at(z)` — the
  discharge claim). No second z-open needed.
- **Soundness**: binding is exact (a degree-`<D` poly is its `D` coefficients — no
  underdetermination, unlike the wide-lift form where the `Σ X^i·c_i'` overlapped);
  proximity stays sound because `RaaF2Code` is F_2-linear ⊆ GF128-linear, so a
  GF128-combination of F_2-codewords is a GF128-codeword (same `(δ/2)^NUM_COLUMN_OPENINGS`).
- **Result**: all **59 protocol + 35 poly tests green** — open round-trips, tamper
  rejected, e2e SHA with/without Hadamard. Soundness-preserving for the main α-path
  (byte-different proof, same accepted claims), and the open now **exposes ψ_z as a
  free second functional** — the foundation the sound oblong binding is built on.
- **Tradeoff (accepted, applied uniformly)**: the bit-slice claim carries `D` GF128
  coeffs (~512 B/entry at D=32) vs the lift's compressed ~24–40 B — the lift was also
  a proof-size compression. Per the approved direction the un-lifted form is applied
  to **all** columns (no discharge-only mixed-flavor batching); proof size grows, the
  M_α⁻¹ lift cost disappears.
- **MEASURED prove-time cost of the un-lifted open (A/B at `7d2c468` lifted vs
  current, Apple M4, nvars=16): `Prove-NoHadamard` 47.8 → 63.2 ms (+15.4 ms, +32%)**.
  The open is on **every** F_2 proof's critical path, so this ~15 ms hits all proofs.
  Counter-intuitively, dropping the lift made the open *slower*: the `M_α⁻¹` lift was
  `O(num_rows + row_len)` (small), but the un-lifted per-cell ops + the `combined_row`
  proximity encode are now `GF128Poly<D>` (D=32 GF128 ≈ 512 B, ~13× wider than the old
  narrow `BinaryF2Poly`) across `O(num_cols·2^num_vars)` cells — the width dominates.
  **Net is still hugely positive for the SHA Hadamard prove** (the un-lifted open is a
  ~15 ms *enabling* cost; the oblong discharge it unlocks saves ~350 ms → 482→136 ms).
- **Claw-back DONE — ~8 ms recovered (commit `1a69adc`)**. Profiled the open
  sub-steps (nvars=16): the regression is **100% the `combined_row`** (proximity row)
  build — **13.5 ms / 77% of the open** — while `b_vector` (same scatter count, into
  one hot accumulator) is only 2.5 ms. So the cost is *cache*, not width per se.
  - **The τ-collapse idea (a) is UNSOUND**: `combined_row` is used by **Check 3
    (coherence)** `<combined_row, q1> == <coeffs, b'>` over the full `GF128Poly<D>`,
    which binds the **wide** `b'` to the committed cells. Collapsing it to one GF128
    would weaken that to a single random combination → a wrong `b'` (hence wrong
    ψ_α/ψ_z) could pass. So `combined_row` must stay D-wide.
  - **What actually worked (both bit-identical → no proof/verifier change)**:
    (i) **γ-fusion** — `combined_row[j] = Σ_{g,i}(γ_g·coeffs[i])·cell`, so scatter the
    fused weight and drop the separate `row_len`-long `gf128poly_accumulate_scaled`
    γ-pass (~1.6 ms); (ii) **cache-blocking** — the scatter wrote across the whole
    `row_len×512 B` `col_partial` per row (thrash); tile `j` into 64-wide blocks so
    `col_partial[j0..j1]` stays L1-hot across the `i` sweep (~5 ms; mirrors why
    `b_vector`'s hot-accumulator scatter was fast). `combined_row` 13.5 → ~7 ms;
    `Prove-NoHadamard` 63.2 → 55.1 ms, `Prove-Hadamard-Oblong` 136.8 → 128.3 ms
    (~3.8× vs fused).
  - **Residual ~7 ms is fundamental**: `b_vector`/`a'` + `combined_row` must stay
    `GF128Poly<D>` (D=32) to expose ψ_z; the un-lifted open is inherently ~D× wider
    per cell than the old narrow `BinaryF2Poly`. Remaining (not pursued): (b)
    discharge-cols-only un-lifted (mixed-flavor batching, rejected for simplicity);
    (c) SIMD the per-cell bit scatter.
- **Bench migration (follow-up, done)**: Stage 1b changed `F2OpenProof`'s claim /
  b-vector / combined-row from wide `BinaryF2Poly` to `GF128Poly<D>` but only ran
  `--lib` tests, so the three F_2 benches (`f2_sha256`, `f2_sha256_rs`, `f2_blake3`)
  didn't compile under `--benches`. Fixed: (1) the **proof-size measurement** now
  serialises each poly as `D · ALPHA_BYTES` (its D GF(2^128) coefficients) instead of
  `.words()`; (2) the **`bench_micro_verifier_open` suite was removed** (arms
  a–e: LiftedEqTensor / EvalConsistency / LiftDischarge / Coherence / PerOpening) — it
  micro-timed the *defunct* lifted-open verify steps (`AlphaPolyBasis`,
  `f2_poly_mul::<2,5,7>`, `build_lifted_eq_tensor`), which no longer exist on the open
  path. The un-lifted open's verify steps are structurally different (no lift;
  `gf128poly_project`/`_accumulate_scaled`/`encode_gf128_lin_open`), so this is
  re-authoring, not migration; the e2e + step benches still measure real open cost.
  **Re-author per-step verify micro-benches against the un-lifted open if that
  breakdown is wanted again.**

### Oblong AND zerocheck — e2e prove integration (measurement-first): ~5.6× faster Hadamard prove (working tree)
- **What**: wired the GF(2⁸) oblong discharge into the **e2e F_2 prove path** —
  `ZincPlusPiopF2::prove_f2_full_with_oblong_hadamard` (`protocol/src/f2_prove.rs`,
  on a `D = 32`-specialised `impl` since the oblong is hardwired to 32-bit words)
  runs the standard no-Hadamard pipeline (commit + IC + α + sumcheck +
  multipoint-eval + single α-open) **and**, on the same transcript,
  `prove_oblong_and_batch_gf8` over the 16 SHA relations. Added a
  `Prove-Hadamard-Oblong` bench arm to `protocol/benches/f2_sha256.rs`.
- **Why measurement-first**: the handoff's NEXT STEP #1 (the production
  integration / Gate) is fundamentally an *e2e prove-time* question — does the
  5–14× standalone discharge speedup translate to a measurable e2e win? This
  increment answers that **before** the delicate sound-binding rework below, so
  we see the number first and de-risk the core path.
- **Measured e2e A/B** (Apple M4, `target-cpu=native`, `parallel simd unchecked`,
  `f2_sha256` bench, hot machine):

  | arm | nvars=16 median | discharge overhead vs NoHadamard |
  |---|---|---|
  | Prove-NoHadamard | 46.5 ms | — |
  | Prove-Hadamard (fused, ψ_α bit-slice) | 567 ms (noisy: 483–717) | ~+520 ms |
  | **Prove-Hadamard-Oblong (this)** | **101 ms** (97–104) | **~+54 ms** |

  The oblong discharge cuts the e2e Hadamard prove overhead from ~390–520 ms
  (fused) to **~54 ms**, making the full Hadamard prove **~5.6× faster**
  (567→101 ms) at nvars=16 — better than the handoff's ~70 ms / ~3.7× projection.
  (The fused arm is memory-bound and very noisy; the oblong arm is tight.)
- **Architectural finding (why the sound binding is a separate, sizeable step)**:
  the oblong discharge produces `ψ_z` operand evals — `ψ_z(col) = Σ_b col_b·L_b(z)`,
  the *subspace-Lagrange* projection at the univariate-skip challenge `z`. The
  existing single PCS open (`prove_f2_open`) is **monomial-α-specific**: its lifted
  eq-tensor `(q0,q1)` and the per-cell lift (`Σ_b cell_b·X^b`, evaluated at α) both
  bake in α, and the wide F_2[X] claim is only meaningful at α. So `ψ_z` (a
  *different* linear functional of the same bits) **cannot ride the α-open** — it
  needs its own z-projection multipoint-eval + open (handoff §4-(i) "extra
  openings on the discharge columns"). The current fused "Approach B" works only
  because its pair-evals are `ψ_α`, the same projection as the trace/open.
- **NOT yet sound / scope of this increment**: the oblong discharge's `ψ_z` evals
  are produced (and the discharge sumcheck runs on the real transcript), but they
  are **not bound to the commitment** — hence `prove_f2_full_with_oblong_hadamard`
  returns the main proof + a *standalone* `OblongAndProof` tuple, and there is no
  matching verify arm yet. The measurement is also not perfectly apples-to-apples:
  the oblong runs the discharge as an extra phase on top of the no-Hadamard main
  pipeline, whereas the fused folds its pair-evals into the multipoint-eval; the
  eventual sound binding will add a (small) z-multipoint + z-open cost on top of
  the ~54 ms.
- **Next (completes NEXT STEP #1) — SUPERSEDED design, now that the un-lifted open
  ships**: there is **no second open**. The un-lifted open's bound `a' = Σ_b c_b·X^b`
  already yields `ψ_z(col)(r_0) = Σ_b c_b·L_b(z)` for free. So the sound binding is a
  **dual-projection multipoint-eval** reducing both the main `ψ_α` claims at r* and
  the oblong's `ψ_z(col^↓Δ)(γ_word)` pointed-shift claims to the **same** open point
  r_0, where the *one* open binds them:
  - `trace_mles = [α-projected trace] ++ [z-projected discharge cols]`
    (`project_column_with_powers(col, base_lagrange_at(z))` — derived, not committed);
    `up_evals = column_evals_at_rstar` (ψ_α, ref α-cols); `pointed_shifts =` the
    `pair_alpha_evals(columns, distinct_pairs(specs), L_b(z), γ_word)` claims (ref the
    z-cols, point γ_word, shift Δ); reduce to r_0.
  - `open_evals_at_r_0 = [ψ_α(col)(r_0)] ++ [ψ_z(col)(r_0)]`. α-evals bound by the
    open's Check 2; **z-evals bound by a new oblong-verifier check**
    `gf128poly_project(open.a'_g, base_lagrange_at(z)) == z_evals[g]` (same `a'`, same
    transcript-derived γ-batch ⇒ binds each z-eval). `verify_subclaim_pointed` then
    checks the ψ_z pointed-shifts against the bound z-evals; the `batched_tie_check`
    relation `a/b/c_eval = Σ_rel eq(rel;γ_rel)·ψ_z(operand)(γ_word)` closes from them.
- **Concrete subtleties surfaced while scoping (read before starting — these set the
  real size, ≈ Stage 1b or larger)**:
  1. **Prover doesn't expose the oblong eval point.** `OblongAndProof` carries only
     `a/b/c_eval` + round polys; `[z, γ]` is transcript-derived. To fold ψ_z
     pair-evals the prover must capture/re-derive `z` + `γ_word` (mirror
     `verify_oblong_and_channel`, or extend `prove_oblong_and_channel` to return the
     `AndCheckOutput.eval_point`).
  2. **Adders don't fold as (col,Δ) pairs.** `batched_tie_check` handles ANDs via
     `distinct_pairs`/`pair_alpha_evals` over **trace columns** (foldable as
     pointed-shifts ✓) but adders via `build_adder_operand_columns` +
     `adder_operand_alpha_evals` over **derived carry-AND columns** (NOT simple shifts
     of trace cols ✗). A first sound integration is therefore **AND-only**, or must
     first establish that the carry columns are themselves committed (IC kappa cols)
     so their operands decompose into trace pair-evals. SHA's discharge uses adders,
     so AND-only is **not** a complete SHA binding — flag this, don't silently scope it out.
  3. **`prove_f2_full_impl` is monolithic** (runs multipoint + open internally). The
     oblong path calls it with empty fused specs, so its multipoint folds only the
     column evals. Folding the oblong ψ_z pointed-shifts needs either generalizing it
     (optional extra trace MLEs + pointed-shifts + down-evals, empty ⇒ byte-identical
     for the main path) or duplicating ~120 lines into an oblong-specific impl.
  4. **No verifier yet.** `verify_f2_full_with_oblong_hadamard` must be written from
     scratch (mirror `verify_f2_full_impl` + the oblong channel verify + the
     dual-projection multipoint verify + the new z-binding check).
  5. **γ re-derivation.** The open's per-column γ-batch is derived inside
     `verify_f2_open`; the oblong z-binding check must re-derive it identically.
  - **Perf note (still applies)**: one open, one extra (z-)multipoint sumcheck, no
    second Merkle path — the sound oblong path should stay near the ~54 ms measured
    rather than ~doubling the open.
- **Implementation findings for the prover/verifier wiring (c)/(d), from tracing
  `prove_as_subprotocol_with_pointed_shifts` + `prove_f2_full_impl`** — these set the
  real shape; record so the next push doesn't rediscover them:
  1. **Multipoint `up_evals` is strictly 1:1 with `trace_mles`** (the `precombined`
     MLE γ-combines every col; no "down-only" columns). So the z-projected discharge
     cols must be **appended to `trace_mles` with `ψ_z(col)(r*)` up-evals**, reduced to
     r_0 with everything and bound there.
     - **CORRECTION (critical, from reading the open's Check 2 at `f2_prove.rs:2675`)**:
       append **ALL witness primary cols** (`trace.binary_poly[num_pub_bin..]`),
       *not* just the ~5 AND-referenced ones. The open's Check 2 binds
       `ψ_α(a') = Σ_g γ_g·witness_primary_evals[g]` over the **whole** witness γ-batch;
       the `ψ_z` binding rides the SAME `a'`, so `ψ_z(a') = Σ_g γ_g·ψ_z(col_g)(r_0)`
       also sums over **every** witness col — the batch γ-mixes them, so the full
       `z_r0_evals` vector must be known to verify it. Binding just the AND subset is
       impossible (can't extract per-col `a'_g` from the batched `a'`). So the
       multipoint ~doubles (α-cols + all-witness z-cols); `z_up_evals`/`z_r0_evals` are
       ~num-witness-cols each (not ~5). This is the subtlety that, rushed, yields an
       open that passes the honest round-trip but binds nothing on the z-side.
     - **Indexing**: z-cols are appended after the C α-cols (= `projected_trace_for_mp.len()`,
       which is indexed by the spec col indices). The z-col for trace col `col`
       (a witness primary col, `col >= num_pub_bin`) sits at `C + (col - num_pub_bin)`;
       pointed-shift `source_col` = that. Off-by-one here = unsound/broken.
     - **γ-threading**: the `ψ_z` check needs the open's per-col γ (drawn inside
       `verify_f2_open_with_virtuals` at `:2656`). Thread it out — return γ from the
       open verify, or do the `ψ_z` check inside a variant — don't re-derive (fragile).
  2. **New shipped proof fields** (beyond today's `F2FullProof`): the **z up-evals**
     `ψ_z(col)(r*)` (verifier can't recompute them without the trace — same reason
     `column_evals_at_rstar` is shipped), the **z r_0-evals** (extend
     `open_evals_at_r_0`), the **trusted adder parents**, and the `OblongAndProof`. So
     the e2e oblong path returns a **wrapper proof**, not a bare `F2FullProof`.
  3. **Transcript order**: discharge (draws z,γ) runs *before* uair (current code);
     but the **z up-evals need both z and r***, so they're computed **after uair, before
     the multipoint challenges**, and absorbed there (mirrors the `column_evals_at_rstar`
     absorb). The z r_0-evals ride the existing `open_evals_at_r_0` absorb.
  4. **Architecture decision: a self-contained `prove_f2_full_oblong_impl`, NOT a
     generalized `prove_f2_full_impl`.** Threading the z up-evals (needs r*, only
     available *inside* the impl) + the extra return data across a generalization
     boundary is awkward and touches all 4 main callers' return handling; duplicating
     the ~120-line multipoint+open section into an isolated oblong impl is safer (zero
     risk to the main path) and keeps the oblong-specific transcript ordering local.
  5. **Scheme**: z-weights are `Gf8Scheme::new().base_lagrange(z)` (embed(H₈)), not
     `base_lagrange_at(z)` (monomial) — match `verify_oblong_and_batch_gf8`.
  - **Status**: (a) shipped (`91eebec`); (b) resolved (AND-only, adders trusted); the
    remaining (c)+(d) is **one indivisible ~250-300-line soundness-critical push**
    (prover impl + wrapper proof struct + verifier + round-trip test) — it only becomes
    meaningfully green once the round-trip validates, so it lands as a unit.

### Oblong AND zerocheck — Phase-2 Gruen eq-trick: degree-2 MLE-check, no eq-table fold (working tree)
- **What**: replaced the Phase-2 sumcheck of the oblong discharge
  (`poly/src/univariate/oblong_and.rs`) with Binius64's eq-factored **MLE-check**
  (Gruen, <https://eprint.iacr.org/2024/108> §3; reference
  `crates/ip-prover/src/sumcheck/quadratic_mle.rs` + `crates/ip/src/mlecheck.rs`).
  The naive form folded `eq(·; r)` as a fourth table and sent the degree-3 round
  poly on the 4-point subspace `{0,1,2,3}`. Gruen factors the per-round `eq` out:
  - the prover ships the **degree-2 prime polynomial** `h(t) = Σ_rest
    (a·b−c)(t,rest)·eq_rest[rest]` truncated to its monomial `[c₁, c₂]` — the
    constant `c₀` is recovered by the verifier from the eq-relation
    `c₀ = claim − r_i·(c₁+c₂)` — so **2 field elements/round, not 4**;
  - only `eq_rest` (eq over the **remaining** row vars `r[1..]`, half the size) is
    maintained, **sum-folded** (XOR-only `out[i]=t[2i]+t[2i+1]`, no challenge
    multiply, via `sum_fold_low`) — so Phase-2 folds **3 tables (a,b,c), not 4**,
    and the full `eq_indicator(r)` (2ⁿ GF128 ≈ 16 MB at nvars=16) is never built;
  - the verifier threads the claim by `h(γ_i)` (Horner); the closing check is just
    `a·b−c == claim` — the per-variable `eq` factors thread out, so the old
    `eq_star = ∏ eq1(γ_j;r_j)` closing factor is gone.
  Per-pair Phase-2 mults drop **24 → 6** (4 points × 6 muls → 3 evals × 2 muls),
  plus one fewer table-fold and a half-size, mul-free eq update. Scheme-
  independent: `MonomialScheme` (GF128) and `Gf8Scheme` share this Phase 2, so
  both `Discharge-Oblong` arms benefit.
- **Soundness / detection note**: the MLE-check has **no per-round consistency
  rejection** (the recovered `c₀` always satisfies the round relation), so a
  corrupt witness — a non-vanishing `R₀`, or tampered closing evals — now surfaces
  at the closing `FinalCheck` rather than at round 0. `OblongError::RoundConsistency`
  is removed; the corrupt-witness tests now assert `FinalCheck`. Same Gruen24
  soundness as binius's production MLE-check (Schwartz–Zippel over the random γ).
- **Measured discharge A/B** (Apple M4, `target-cpu=native`, `parallel simd
  unchecked`, `f2_sha256` bench; back-to-back `git stash` A/B on one hot machine,
  so the unchanged `Discharge-Fused` arm is the thermal-drift anchor):

  | arm | nvars=16 before → after | nvars=20 before → after |
  |---|---|---|
  | Fused (unchanged anchor) | 491.9 → 441.7 ms (−10%, pure drift) | — (≈13 s; skipped) |
  | **Oblong GF(2⁸)** | **74.7 → 57.9 ms** | **1.315 → 1.068 s** |
  | Oblong GF128 | 145.7 → 121.4 ms | 2.65 → 2.68 s (noise) |

  The **GF(2⁸) arm (the production candidate) is ~19–22% faster raw** — ~14% after
  subtracting the −10% Fused-anchor drift at nvars=16, and a clean −19% at
  nvars=20 (no usable Fused anchor there — it's ~13 s/iter). The **GF128 arm's win
  is real but Phase-1-dominated**: its full-field NTT dwarfs Phase 2, so the
  (scheme-independent) ~50% Phase-2 cut is only ~7% of total and sits inside the
  ±8% CI at nvars=20 — visible at nvars=16 (−17% raw) but partly drift. Matches the
  handoff's ~11% estimate; the proof also shrinks (2 vs 4 coeffs/round). All 11
  poly + 9 protocol oblong tests green.
- **Still untapped (the 2-eval prover)**: binius computes only `h(1), h(∞)` and
  recovers `h(0)` from the threaded claim (2 evals not 3 → ~6→4 muls/pair), at the
  cost of a per-round field inverse + prover-side claim tracking. Deferred for
  prover simplicity (no inverse, no claim thread); a small further Phase-2 win.
  SIMD-packed `Gf8` lanes remain the bigger (x86/GFNI-only) lever — see below.

### Oblong AND zerocheck — Phase D: batched all-16 discharge + GF(2⁸) + parallel → 5–11× vs fused (A/B)
- **What**: the oblong discharge now covers **all 16 SHA relations in one
  zerocheck**, is **Fiat–Shamir**, and there's a **discharge A/B** vs the current
  fused bit-slice discharge.
  - **Batching** (`prove/verify_oblong_and_batch`, `protocol/src/f2_oblong_hadamard.rs`):
    stack the 3 ANDs then the 13 adders (Binius carry form, via
    `build_adder_operand_columns`) into one oblong zerocheck — relation index as
    the high row-vars, word index low, padded to a power-of-two relation count
    with zero-operand relations. The batched ψ_z tie is
    `a_eval = Σ_rel eq(rel; γ_rel)·ψ_z(operand_rel)(γ_word)` (`γ` splits into
    `γ_word` low + `γ_rel` high); ANDs derive soundly from the pair evals, adders'
    carry evals are recomputed (honest-prover, Issue 1).
  - **Fiat–Shamir** (commit `7ed6318`): a poly `OblongChannel` trait +
    transcript-agnostic core, with `protocol`'s `Blake3` adapter; explicit path
    kept as a `ReplayChannel`.
- **Measured discharge A/B** (Apple M4, `target-cpu=native`, `parallel simd
  unchecked`) — `prove_f2_hadamard_phase` (fused, ψ_α, 1536 GF128 slices) vs
  `prove_oblong_and_batch` (oblong, ψ_z, word-packed), same SHA columns + same 16
  relations (`f2_sha256` bench, `Discharge-Fused`/`Discharge-Oblong`):

  | nvars | fused (ψ_α) | oblong naive (GF128) | **oblong GF(2⁸)** | GF(2⁸) vs fused |
  |---|---|---|---|---|
  | 9 | 3.72 ms | 2.84 ms | **1.70 ms** | **2.2×** |
  | 16 | 395 ms | 102 ms | **74.7 ms** | **5.3×** |
  | 20 | 13.2 s | 1.63 s | **1.17 s** | **11.2×** |

  **The GF(2⁸)-accelerated oblong discharge is 5–11× faster than the fused
  bit-slice one** — and the **win grows with size** (the fused discharge is
  memory-bound / super-linear; the oblong scales well). Two compounding levers:
  (1) the **GF(2⁸) swap** (`Gf8Scheme`, byte-lookup NTT over `embed(H₈)`) — turned
  the 1.1–1.4× naive floor into ~2×; (2) **parallelism** (commit `39ebc1d`) — the
  oblong prover was single-threaded while the fused baseline used all cores;
  parallelizing the round message + Phase-2 (rayon, `with_min_len(2^14)` so the
  nvars=9 default stays serial) took GF(2⁸) from 198→74.7 ms at nvars=16; (3) the
  **eq-split** (commit `9149b91`) — 3 deterministic GF(2⁸) skip challenges
  `{α,α²,α⁴}` + the GF(2¹²⁸) eq-weight/embed once per 8-word chunk — to **70.7 ms**.
  **Key aarch64 finding**: the eq-split alone does *not* pay because `Gf8::mul`
  (log/antilog + `%255` + per-op `OnceLock`) ≈ GF128 CLMUL without GFNI; it needs
  a **64KB direct mul-table fetched once per kernel** to win (then 92→70.7 ms =
  1.30× same-mul; ~1.06× vs the prior log/antilog base). Modest on aarch64, larger
  with GFNI. **Still untapped**: SIMD-packed `Gf8` lanes (16-wide); the Gruen
  eq-trick in Phase-2 (degree-2 round polys, no eq-table fold) — *the Gruen
  eq-trick has since shipped; see the most-recent entry above*. 35 tests green.
- **Remaining for the full Phase D**: (a) the **eq-split** (task #6) for the next
  speed step; (b) the **multipoint-eval binding** (task #7 part 2 — fold the ψ_z
  pair-evals into `f2_prove`, the production integration; doesn't change discharge
  prove cost); (c) a sound carry binding for the adders (Issue 1). Then the e2e
  `Prove` A/B.

### Oblong AND zerocheck — Phase C: ψ_z integration tie, one AND relation round-trips (working tree)
- **What**: the oblong AND zerocheck wired into the F_2 Hadamard discharge for a
  single AND relation, with the **`ψ_z` recombination tie** that binds the
  zerocheck's operand evals to the committed columns
  (`protocol/src/f2_oblong_hadamard.rs`; plan §4). `prove_oblong_and_relation`
  packs the built operand columns to `u32` words and runs the oblong zerocheck;
  `verify_oblong_and_relation` checks the zerocheck then the tie.
- **Why it's a near-mechanical seam (the key realisation, plan §4)**: the oblong
  challenge `z` plays the same role as `α`. Both `F_2`-linearly collapse a word's
  `D` bits into one scalar — `ψ_α(W)=Σ_b W_b·α^b` (monomial) vs
  `ψ_z(W)=Σ_i W_i·L_i(z)` (additive-NTT Lagrange). The zerocheck outputs
  `a_eval=ψ_z(U)(γ)` **by construction**, so the tie **reuses the existing,
  tested `ψ_α` machinery** (`pair_alpha_evals` + `derive_operand_parents`) fed
  `base_lagrange_at(z)` for the α-powers and the Phase-2 sumcheck point `γ` for
  `r*`. `ψ_z` is `F_2`-linear ⇒ commutes with the XOR/shift/complement operand
  structure exactly as `ψ_α` does; soundness is the same `(D−1)/|F|` bound.
- **Result**: 6 new tests, the 9 existing `f2_hadamard` tests still green
  (`build_operand_column`/`cell_mask` exposed `pub(crate)`, no behaviour change).
  Gate (full prove→verify + tie round-trip, à la `plain_and_round_trips`): the
  `ψ_z` tie derives the correct operand evals for **plain / row-shifted /
  complemented / Maj-combo** operands; `corrupt_w_is_rejected` (zerocheck rejects
  a bad W at round 0); and `tie_catches_wrong_operand_wiring` — the tie itself
  rejects when the verifier binds to the wrong column, i.e. the tie has real
  soundness teeth, not just a tautological pass. **The architectural risk
  (does `ψ_z` compose with our operand/column structure?) is resolved.**
- **Remaining for Phase C / D**:
  - **(a) Fiat–Shamir ✅ DONE** (commit `7ed6318`) — `prove/verify_oblong_and_relation`
    now take `&mut impl Transcript`; a `poly` `OblongChannel` trait (+ a
    transcript-agnostic core) lets `protocol` supply a `Blake3` adapter, with the
    explicit-challenge path kept as a `ReplayChannel` wrapper (poly tests
    unchanged). The 6 Phase-C tests run with real same-seed transcripts.
  - **(b) fold the tie into the multipoint-eval** — the binding is a **row-space
    opening at `γ`, not a joint `(z, γ)` open**: `z` is the bit-recombination (it
    picks the `L_i(z)` weights collapsing the `D` bit-slices, exactly as `α^b`
    does today), so the opened object is the `ψ_z`-projected column opened at `γ`.
    The `ψ_z(col↓Δ)(γ)` pair-evals (= `pair_alpha_evals` with `base_lagrange_at(z)`,
    which the tie *already* computes) fold into the **main multipoint-eval** (Δ=0
    point claim, Δ≠0 shift predicate, single open binds them) — the same
    "Approach B" the `ψ_α` pair-evals use in `f2_prove.rs`. **This is the
    production-integration step** (it edits `f2_prove`'s multipoint-eval assembly),
    overlapping Phase D; the §4-(i)/(ii) projection-point choice (z vs α for
    shared columns) is the cost question.
  - **(c) batch all 16 relations** (3 ANDs + 13 adders) into one oblong zerocheck
    (Phase D) + the GF(2⁸)/eq-split prover swap, then A/B vs the current fused
    discharge.

### Oblong AND zerocheck — GF(2⁸) speed lever: subfield + embedding + byte-lookup NTT (working tree)
- **What**: the prover-side speed prerequisite (plan P1 + P4) for the oblong AND
  zerocheck, so the additive-NTT and the per-extension-point products run in
  **GF(2⁸)** (log/antilog mult, 1-byte XOR) instead of GF(2¹²⁸) (CLMUL, 16-byte
  XOR). Two new `poly` modules:
  - `poly/src/univariate/binary_gf8.rs` — `Gf8` (`GF(2⁸)`, `u8`-backed) **derived
    from within GHASH**: `θ = relative-norm N_{GF(2¹²⁸)/GF(2⁸)}(g)` lands in the
    order-255 subfield, and `GF(2⁸)=F_2[X]/m(X)` with `m=minpoly(θ)` (found via an
    XOR linear basis over `θ^0..θ^8`). Then `α↦θ` is **automatically a field
    homomorphism** — no AES-isomorphism search. `θ`, `m`, log/antilog + `embed`
    tables computed once from our GF(2¹²⁸) arithmetic and memoised.
  - `poly/src/univariate/oblong_and_gf8.rs` — `Gf8Ntt` (the
    `[[[Gf8;WORD_BITS];256];WORD_BYTES]` byte-lookup additive NTT, binius
    `ntt_lookup.rs`) + `extend_word` + `gf8_round_message` + `embed_subspaces`
    (the `embed(H₈)` base/full subspaces the verifier uses with this path).
- **Why sound**: the NTT runs over the GF(2⁸) subspace `H₈={Gf8(0)…Gf8(63)}`;
  its image `embed(H₈)` is the verifier's GF(2¹²⁸) subspace. Since `embed` is a
  field hom, every GF(2⁸) NTT value/product equals (after `embed`) the GF(2¹²⁸)
  computation over `embed(H₈)`. (Different subspace than the naive path's
  monomial `{X⁰…X⁵}` — each internally consistent with its own verifier.)
- **Result**: 9 new tests (6 field + 3 NTT), full `zinc-poly` suite 144 passed.
  Gates: `embed_is_a_field_homomorphism` over **all 65536 pairs**; `mul` matches a
  carryless reference; antilog covers all 255 nonzero bytes; the cross-field
  `gf8_extend_embeds_to_gf128_extend` and `gf8_round_message_matches_gf128_over_
  embed_subspace`. Resolves plan Open-Q #2 (subfield generator + the
  `F_2`-independent skip challenges `{α,α²,α⁴}`).
- **Measured** (Apple M4, `--release`, `target-cpu=native`), single-relation
  Phase-1 round message at **nvars=16 (65536 words), best of 5**:

  | path | time | ns/word |
  |---|---|---|
  | GF(2¹²⁸) naive | 16.71 ms | 255.0 |
  | **GF(2⁸) lookup** | **6.53 ms** | **99.6** |
  | — | — | **2.56× faster** |

  And this is with the **eq-weighting still per-word in GF(2¹²⁸)** — binius's
  small/big `eq` split (weight the ≤3 deterministic small-field vars in GF(2⁸),
  embed + big-`eq`-weight once per `2^k` chunk) is the next lever and cuts the
  remaining GF(2¹²⁸) mults by the chunk size. (Bench: `oblong_and_gf8::tests::
  bench_gf8_vs_gf128_round_message`, `--ignored --nocapture`.)
- **Remaining D-prereq**: the eq-split (above) for more speed; **Fiat-Shamir**
  moves to Phase C (it belongs in the `protocol` crate with the integration —
  `poly` has no transcript dep). The `verify_oblong_and` already accepts evals
  from a faster NTT (same values), so swapping in the GF(2⁸) prover is a
  drop-in once the subspace is parameterised.

### Oblong univariate AND zerocheck — Phase A primitives + standalone Phase-1+2 prove/verify (working tree)
- **What**: first landed slice of the Binius64 **oblong univariate zerocheck**
  port (plan: `documentation/f2-hadamard-oblong-port-plan.md`), the verifier-
  visible rearchitecture that replaces the memory-bound 1536-GF(2¹²⁸)-bit-slice
  AND discharge with a **word-packed** check. Two new self-contained modules in
  the `poly` crate, over our `BinaryFieldGF128` (no new field type yet):
  - `poly/src/univariate/binary_subspace.rs` (plan P2+P3): `BinarySubspace`
    (`F_2`-linear subspace by ordered monomial basis, `get(i)=Σ basis[j]·bit(i,j)`),
    `lagrange_evals` + `extrapolate_over_subspace` (O(n) barycentric over the
    additive subgroup), `evaluate_univariate`. Direct ports of binius64
    `crates/math/src/{binary_subspace,univariate}.rs`.
  - `poly/src/univariate/oblong_and.rs` (plan P4/P5, **naive GF(2¹²⁸) path**):
    the additive-NTT word extension (`AdditiveNtt::extend_word` — interpolate a
    word's 32 bits over the 32-point base domain, extrapolate to the 32-point
    extension domain via precomputed Lagrange rows, select-and-XOR, no muls) and
    the Phase-1 univariate-skip round message
    `R₀(Z)=Σ_X (A·B−C)(Z,X)·eq(X;r)`, **and the full standalone protocol** for
    one AND relation: `prove_oblong_and` / `verify_oblong_and` (round message →
    fold operands at the univariate challenge `z` → Phase-2 **eq-weighted
    degree-≤3 sumcheck** over the row vars → closing `a·b−c==eval` check),
    plus `eq_indicator` / `AndCheckOutput` / `OblongError`. `SKIPPED_VARS=5`,
    `WORD_BITS=32` (vs binius's 6/64 for 64-bit words). Challenges (`z`, the
    per-round `γ`) are passed explicitly — Fiat-Shamir wiring is deferred to the
    integration step.
- **Why**: the discharge is memory-bandwidth-bound — it streams **1536 GF(2¹²⁸)
  slices** (`16 relations × 3 operands × D=32`) per sumcheck round (≈1.5 GB at
  nvars=16, 2.34× scaling — see `f2-hadamard-handoff.md`). The oblong protocol
  keeps each operand as **one packed word-column** and handles the 32-bit
  dimension with a single univariate-skip round, so the post-fold Phase-2 rounds
  stream **48 GF(2¹²⁸) MLEs** instead of 1536 — exactly the **×D=32 data-volume
  reduction** (the input is 128× smaller: u32 words vs GF(2¹²⁸) slices). This is
  the only Binius-scale lever left after the byte-identical paths were closed
  (`f2-hadamard-univariate-skip-design.md` §11).
- **Result**: 14 new tests green (7 subspace + 7 oblong), full `zinc-poly` lib
  suite 135 passed, new files clippy-clean. Gates met:
  - **Phase-1 cross-check** (binius64's own `round_message_matches_folded_sum_claim`,
    n=4/6/8): `R₀(z)` recovered from the round message (32 base zeros ++ 32
    extension evals, extrapolated at random `z`) **equals** the brute-force folded
    sum claim `Σ_X (A(z,X)B(z,X)−C(z,X))eq[X]`. `R₀` provably vanishes on the base
    domain when `c=a&b`; `extend_word` matches per-point extrapolation.
  - **Full standalone round-trip** (`full_round_trip_accepts_honest`, n=4/6/8):
    Phase-1+2 prove→verify accepts honest ANDs, and the closing `a/b/c_eval`
    match an independent MLE evaluation.
  - **Soundness**: corrupting one C bit is rejected at **round 0** (base-domain
    vanishing breaks ⇒ reconstructed `R₀(z)` ≠ true sum); a tampered closing eval
    fails the final `(a·b−c)·eq` check.
  - **One real bug found+fixed in-session**: the `eq_indicator` doubling was
    big-endian (last `r` → bit 0) while `fold_low` binds bit 0 first; Phase-1's
    internally-consistent cross-check masked it, the full round-trip's final
    check caught it. Fixed to little-endian (bit `i` ↔ `r[i]`).
  - **De-risks the core math + the field**: the oblong AND check is correct over
    our GHASH field, confirming plan §0 (our `BinaryFieldGF128`, reduction `0x87`,
    monomial LSB-first, is directly usable — no field-isomorphism rewrite; plan
    Open-Q #1 resolved for the GF(2¹²⁸) path).
- **NOT a perf change yet** — correctness-first infrastructure, no prover path
  touched, and the **naive GF(2¹²⁸) NTT** (no byte-lookup). The *memory* win is
  structural and already real: the oblong representation is **48 packed
  word-columns** (post-fold: 48 GF(2¹²⁸) MLEs) vs **1536 GF(2¹²⁸) bit-slices** =
  exactly the **×D=32** data-volume cut (input 128× smaller, u32 vs GF128). The
  *wall-clock* win needs the remaining work, in order: (a) **GF(2⁸) byte-lookup
  NTT** (binius `ntt_lookup.rs`) — the prover speed lever (runs NTT + products in
  the 8-bit AES field; the naive path does them in GF(2¹²⁸)); (b) **Fiat-Shamir**
  transcript wiring (challenges are explicit args today); (c) the **ψ_α
  integration seam** (plan §4, the architectural risk). Then Phase-D A/B. Tasks
  tracked in the port plan "Progress" + §5.

### Round-1 fast path for the Hadamard degree-3 zerocheck (Phase 1; working tree)
- **What**: `HadamardRound1FastPath` (`piop/src/lookup/hadamard.rs`) + a
  `prepare_hadamard_group_with_fast` that takes **packed operand columns**
  (`BinaryPoly<D>`, 3 per relation) and uses `with_round_1_fast` instead of
  materialising the 1536 full-width `GF128::Inner` bit-slices.
  `prove_f2_hadamard_phase` now builds packed columns (`build_operand_column`
  + the existing `build_adder_operand_columns`) rather than calling the
  (now-removed) `build_all_operand_slices`. Modelled on the booleanity
  degree-3 fast path, but over `GF(2^128)`: boundary points are
  `F::from_with_cfg(2/3)` (= `X`, `X+1`), **not** booleanity's `one+one`
  (which collapses to `0,1` in char 2 — that shortcut is correct only in
  booleanity's large-field integer pipeline).
- **Correctness**: `round_1_message` does NOT assume `M(0)=M(1)=0` — unlike
  booleanity's `v(v−1)` (structurally 0 for any bit), the Hadamard term
  `U·V−W` is non-zero on a corrupt row, so it computes all four `M(0..3)`
  to match the generic path (the negative tests `corrupt_w_is_rejected` /
  `adder_rejects_wrong_sum` fail otherwise). Gated by a new piop test
  `fast_path_matches_generic` asserting the fast- and generic-path
  `MultiDegreeSumcheckProof`s are **byte-identical**; 58 piop + 50 protocol
  lib tests green.
- **Result (Prove-Hadamard, M-series, `parallel simd unchecked`)**:

  | nvars | generic | fast path | Δ     |
  |-------|---------|-----------|-------|
  | 9     | 10.35 ms| 8.03 ms   | −22%  |
  | 16    | 650.7 ms| 622.6 ms  | −4%   |
  | 20    | 28.83 s | 18.83 s   | −35%  |

  Verify unchanged. **Partial win**: round-1 compute is similar (4 boundary
  points either way), so where the working set fits RAM (nvars≤16) the gain
  is small; the nvars=20 −35% is the ~2× peak-memory reduction (round-1's
  24 GB materialise removed, but `fold_with_r1` still emits ~12 GB of
  half-size slices). nvars≥20 stays memory-bound until **Phase 2** (slice
  dedup) lands — see "Identified but not implemented".

### Hadamard discharge cost measured — new `f2_sha256` "Hadamard" A/B group (working tree)
- **What**: added a `hadamard` criterion group to `protocol/benches/f2_sha256.rs`
  that A/Bs the *same* `prove_f2_full_with_*` / `verify_f2_full_with_*`
  family **with** the 16 SHA-256 Hadamard relations (3 ANDs + 13 adders,
  from `Sha256F2HadamardLayout::relations()`) vs **without** (`&[]` specs,
  what the rest of the bench measures). Both sides non-pre-paired so the
  A/B delta is purely the discharge. Also extended `f2_full_proof_parts`
  to count the previously-omitted Hadamard fields (`uair.hadamard_proof`'s
  sumcheck + `bit_slice_evals`, `hadamard_pair_evals`,
  `hadamard_adder_parents`) — before this the with/without proof sizes came
  out byte-identical because those fields weren't in the breakdown.
- **Result (M-series, `parallel simd unchecked`, no `metal_gpu`, sample_size 10)** —
  the discharge is **cheap to verify, catastrophic to prove, and worsens
  super-linearly**:

  | nvars | Prove no-had | Prove +16had | ×    | Verify no-had | Verify +16had | × |
  |-------|--------------|--------------|------|---------------|---------------|---|
  | 9     | 2.33 ms      | 10.35 ms     | 4.4× | 1.06 ms       | 1.41 ms       | 1.33× |
  | 16    | 47.0 ms      | 650.7 ms     | 13.8×| 4.67 ms       | 5.15 ms       | 1.10× |
  | 20    | 739.9 ms     | **28.83 s**  | 39×  | 42.9 ms       | 44.2 ms       | 1.03× |

  - **Verify**: negligible (+3% at nvars=20, shrinking with nvars). Not a concern.
  - **Proof size**: ~**constant +26 KB raw / +25 KB zstd**, independent of
    nvars (so 3.0% at nvars=9 → 0.4% at nvars=20). Dominated by
    `bit_slice_evals` (24.6 KB); the zerocheck `sumcheck` is only ~1 KB,
    `pair_evals` 144 B, `adder_parents` 624 B. Size is a non-issue.
  - **Prove**: the headline problem. The no-had prove scales ~linearly
    (47 → 740 ms = 15.7× for the 16× rows of nvars 16→20); the +had prove
    scales **super-linearly** (651 ms → 28.8 s = **44×** for the same 16×
    rows). At nvars=20, **28.1 s of the 28.8 s is discharge overhead.**
- **Root-cause hypothesis (unconfirmed — needs profiling)**: the super-
  linearity lives entirely in the discharge path, consistent with a
  **memory-bandwidth bound**. The Hadamard zerocheck materialises bit-slice
  MLEs for all 16 relations' operands (D=32 slices/column → 32× the per-
  column data); at nvars=20 the working set (~hundreds of MB per operand
  set) blows past cache, so wall-time grows faster than O(rows). The degree-3
  zerocheck round-1 (which touches the full (μ+5)-var hypercube, ≈50% of the
  sumcheck point-work) is the prime suspect for the constant factor.
- **Optimisation opportunities this surfaced** (see "Identified but not
  implemented"): (a) a `{0,1}`-slice round-1 fast path for the degree-3
  zerocheck — round 1 is the biggest round and the bit-slices are still
  0/1-valued there (AND/select instead of GF(2^128) mults); the main
  γ-sumcheck already has `F2EqColRound1FastPath`, this ports it to the
  Hadamard zerocheck. (b) batch the 16 per-relation zerochecks instead of
  running them as separate groups. (c) avoid full bit-slice materialisation
  / cache-block the expansion to fix the super-linear memory behaviour.

### `LEAF_GROUP_SIZE` 8 → 1 to shrink `open.opened/values` 8× (commit `<TBD>`, measurements pending)
- **What**: `protocol/src/f2_prove.rs::LEAF_GROUP_SIZE` flipped from
  `8` to `1`. With group size 1 each Merkle leaf hashes exactly one
  paired-storage column stack (`paired_batch × num_rows × 8 B`),
  so an opening ships one column of cells instead of eight. The
  lower layers (`zip-plus::pcs::phase_commit::commit_grouped`,
  `MerkleTree::new_from_column_groups`) are fully parameterised
  over `group_size`; the only invariant is power-of-two, which 1
  satisfies. No other code changes needed — every call site
  references the constant, not the literal `8`.
- **Why**: the `open.opened/values` region of the proof was
  ~4.34 MB at the bench's nvars=22 shape (`987 openings × 8
  LEAF_GROUP_SIZE × 9 paired_batch × 8 num_rows × 8 B/cell`).
  The `LEAF_GROUP_SIZE` factor exists purely to amortise Blake3
  setup during commit; the wire-side cost is a `LEAF_GROUP_SIZE×`
  bloat on values, partly offset by `log2(LEAF_GROUP_SIZE)`
  fewer Merkle siblings per opening.
- **Expected proof-size delta** (analytic, nvars=22, 987 openings):
  - `open.opened/values`: 4.34 MB → 555 KB (−3.8 MB).
  - `open.opened/merkle`: +3 siblings × 987 openings × 32 B
    (Blake3 hash) ≈ +95 KB.
  - **Net: ~−3.7 MB on the dominant proof region.**
- **Expected prove-time cost**: Blake3 leaf hashing loses 8× of
  its setup amortisation — each leaf hashes 72 B (one paired
  column stack) instead of 576 B, so per-byte throughput
  collapses on the Metal-Blake3 leaf kernel and the CPU fallback.
  Leaf count grows 8× (`codeword_len / 1` instead of
  `codeword_len / 8`), and the Merkle tree gains 3 internal
  levels.
- **Verification**: 12/12 F_2 lib tests pass
  (`cargo test -p zinc-protocol --lib f2_prove --release`),
  including `prove_then_verify_sha256_f2_roundtrips` and the
  full `prove_then_verify_f2_full_*` suite.
- **Measurements pending**: still need to bench
  `Zinc+ F_2 SHA-256/Commit*` and `Prove e2e` at nvars=22 to
  confirm the prover-time regression is acceptable for the
  ~3.7 MB proof-size win. If the commit-time hit exceeds the
  proof-size benefit, this entry should move to "Investigated,
  didn't help" and the constant should revert.
- **How to bench**: `cargo bench -p zinc-protocol --bench
  f2_sha256 --features parallel,simd,unchecked,metal_gpu --
  "Zinc\+ F_2 SHA-256/"` and compare against the previous
  baseline. Look at `Commit-Fused-GPU-Inline`, `Prove`, and
  the proof-size breakdown for `open.opened/{values,merkle}`.

### F_2 native IC: short-circuit `assert_zero` slots to `ZERO` in both builders (commit TBD)
- **What**: in `protocol/src/f2_native_ic.rs`, `F2NativeMleFirstBuilder::assert_zero`
  (used by `prove_linear`) and `F2NativeRowBuilder::assert_zero` (used
  by `prove_combined`) now push `DynamicPolynomialF::ZERO` /
  `F2RowExpr::zero()` instead of the actual constraint expression.
  The expression itself is dropped on the floor. Mirrors the
  `prove_hybrid` pattern in `piop/src/ideal_check.rs::prove_hybrid`
  but at the builder level — sound for the F_2 protocol's flow
  because the F_2 IC's `combined_mle_values` are used only for
  transcript absorption, not as inputs to the downstream γ-batched
  sumcheck (which operates on the trace columns, not on the IC's
  per-constraint polynomials).
- **Why**: for an honest prover, an `assert_zero` constraint
  evaluates to the zero polynomial cell-wise on every row, and the
  IC verifier's `verify_as_subprotocol` already filters zero-ideal
  slots out of `batched_ideal_check`. Computing the (degree-2,
  selector × residual) F[X] expression and absorbing its coefficients
  was pure busywork in `prove_linear` — one `DynamicPolynomialF`
  polynomial multiply + a 32-coefficient transcript absorb per
  zero-ideal slot, scaled by `num_constraints`. The integer PIOP
  builder explicitly notes the same optimization was reverted there
  for sumcheck-consistency reasons in the integer setting; the F_2
  protocol's structure makes it sound here (see "Why" above).
- **Measured (criterion `--quick`, nvars=22, full perf feature set)**:

  | Bench (Prove e2e) | Before | After  | Δ            |
  |-------------------|-------:|-------:|-------------:|
  | F_2 SHA-256       | 2.92 s | 2.22 s | **−24%** (high variance, [1.99 s, 3.14 s] band) |
  | F_2 Blake3        | 5.90 s | 5.12 s | **−13%**     |

  All 27 `zinc-test-uair` tests still pass; SHA + Blake3 e2e bench
  paths complete with successful `verify_f2_uair` /
  `verify_f2_open_with_virtuals`.
- **Constraint mix at play**: SHA-256 has ~3 zero-ideal pins out of
  ~17 constraints (~18%); Blake3 has 30 zero-ideal out of 96 (~31%).
  Surprisingly SHA shows the larger win despite the lower zero-ideal
  fraction — likely the per-slot polynomial product dominates more
  for SHA's shape (fewer total constraints, so each slot is a
  larger fraction of work).

### SIMD-batched α-projection (commit TBD) — 2.07× on random, tied on real SHA-256
- **What**: added two NEON-batched 4-cell α-projection kernels in
  [`poly/src/univariate/binary_gf128.rs`](../poly/src/univariate/binary_gf128.rs)
  plus a column-projection wrapper, wired into
  [`f2_prove.rs:877`](../protocol/src/f2_prove.rs#L877) via
  `project_column_with_powers`:
  - **`eval_f2_poly_d_at_with_powers_simd_x4`** (dense): four cells in
    lockstep, `D=32` unconditional iterations. Each iter does one
    16-byte `vld1q_u64` of α^i shared across four accumulators, plus
    four `vandq_u64(pw, mask_k)` + `veorq_u64` per accumulator. Scalar
    fallback for non-aarch64 keeps the same shape with regular u64
    XOR/AND.
  - **`eval_f2_poly_d_at_with_powers_simd_x4_sparse`** (union-skip):
    same body but iterates only positions present in
    `cells[0] | cells[1] | cells[2] | cells[3]` via `trailing_zeros`,
    paying loop-control overhead in exchange for skipping all-clear
    bit positions.

  Wired the **dense** variant into the prove path. Sparse stays
  exported for future use on UAIRs with low cell-popcount/high cell
  correlation.

- **Why now**: the GF(2^192) → GF(2^128) field swap (entry below)
  delivered the predicted 1.5–3.6× wins on every field-touching
  micro-bench *except* `UAIR-b-AlphaProject` — the dominant single
  phase at 1067 ms / nvars=22 (≈90% of UAIR). That kernel uses a
  branchy `trailing_zeros` walk where one fewer XOR per set bit
  (2 vs 3 u64 words for GF128 vs GF192) is invisible against
  loop-control overhead. SIMD batching is the natural next lever to
  amortise that overhead.

- **Measured (criterion, `--features parallel,simd,unchecked`,
  Apple M-series 8-core):**

  Random 65 536-cell column (`binary_gf_compare::project_col_65536`):
  | Kernel | Time | vs branchy |
  |---|---|---|
  | GF128 branchy | 883 µs | 1.00× |
  | GF128 branchless | 603 µs | 1.46× |
  | **GF128 SIMD-x4 dense** | **407 µs** | **2.17×** |
  | GF128 SIMD-x4 sparse | 473 µs | 1.87× |

  Real SHA-256 trace (`Micro/UAIR-b-AlphaProject/nvars=22`):
  | Kernel | Time | vs branchy |
  |---|---|---|
  | GF128 branchy (baseline) | 1067 ms | 1.00× |
  | GF128 SIMD-x4 dense | 1059–1085 ms | ~1.00× (tied within ±2% noise) |
  | GF128 SIMD-x4 sparse | 1105 ms | 0.97× (regressed) |

- **Why the random 2× win collapses to "tied" on real**: the prove
  path's per-cell cost in branchy mode is `popcount(cell) × ~5 ops`.
  On random 32-bit cells, avg popcount is 16, so branchy does ~80
  ops/cell — matching the binary_gf_compare ~12.5 ns/cell. On the
  actual SHA-256 trace, the same branchy kernel runs at ~6.2 ns/cell
  (172 M cells in 1067 ms), implying real-cell avg popcount is closer
  to ~8 — half of random. SIMD-x4 pays a fixed `D=32` cost per cell
  regardless, so it can't undercut a branchy kernel already running
  at 8-bit-per-cell speed.

- **Why sparse regressed**: my hypothesis was that with avg
  popcount~8 per cell, the union of 4 cells would stay well below
  `D=32`. Measurement disagreed: union saturates near `D` on the
  real trace (cells in the same column are mutually uncorrelated +
  the column carries ~all 32 bit positions across its rows), so
  sparse iterates ~`D` times anyway AND pays the loop-control
  overhead. Lesson: low **per-cell** popcount ≠ low **union**
  popcount when cells are independent.

- **Tests**: three correctness tests in `binary_gf128.rs::tests`:
  `simd_x4_matches_scalar_branchy_random` (64 random batches),
  `simd_x4_matches_scalar_branchy_edges` (zero / all-ones /
  single-bit-per-position), `simd_x4_sparse_matches_simd_x4`
  (64 random + 4 adversarial), `project_column_matches_scalar_per_cell`
  (column lengths covering every `len % 4` remainder).

- **Decision**: keep dense SIMD-x4 wired in. The on-prove-path delta
  is within noise but never negative across runs, and the random-
  workload `2×` win means new UAIRs with denser bit distributions
  (e.g. Blake3, Poseidon-over-F_2) get the speedup for free without
  re-engineering the prove path.

- **Open follow-ups**:
  - The α-projection lever appears exhausted at this UAIR shape. The
    next prove-e2e lever is not in this phase; candidates:
    Blake3/Metal commit pipeline, IC sumcheck round work, witness-gen
    in `test-uair/src/sha256_f2.rs`.
  - If a future UAIR has high cell correlation (e.g. lookup-heavy
    columns), `project_column_with_powers_sparse` may pay off. Kept
    exported for that case.

### Field swap GF(2^192) → GF(2^128) on the F_2 prover path (commit TBD)
- **What**: swapped the projecting field from `BinaryFieldGF192`
  (`Uint<3>` storage, FIPS 186-2 pentanomial, Toom-Cook 3-way clmul
  over 6 inner PMULL/PCLMUL muls, 192×192 F_2 matrix inversion for
  `AlphaPolyBasis`) to `BinaryFieldGF128` (`Uint<2>` storage, GHASH
  pentanomial, Karatsuba 2-way clmul over 3 inner muls, 128×128
  matrix inversion). Touched `poly/src/univariate/binary_gf128.rs`
  (promoted from benchmark-only to production with full
  `Field`/`PrimeField`/`InnerTransparentField` surface, `AlphaPolyBasis`,
  `lift_f2_*_to_gf128`/`eval_f2_*_at`/`lift_bp_to_f2_poly_1`/
  `lift_gf128_to_f2_poly_2`/`eval_bits_at`), deprecated
  `binary_gf192` (kept alive only for `binary_gf_compare` bench),
  and global type/module renames across `protocol/src/f2_prove.rs`,
  `protocol/benches/f2_sha256.rs`, `protocol/src/f2_native_ic.rs`,
  `piop/src/{ideal_check,sumcheck/multi_degree,projections}.rs`,
  `test-uair/src/sha256_f2.rs`,
  `poly/src/univariate/binary_f2_wide.rs`, and the
  `f2_prove_plan.md` / `f2_open_plan.md` / `f2-prove-optimizations.md`
  docs.
- **Why**: the field-size choice was the dominant lever the F_2
  prove path's hot loops have left. Algebraically the `BinaryF2Poly<W>`
  widths in the open proof scale linearly with the projecting-field
  bit width (each is `⌈(D + k·n − k)/64⌉` for some constant `k`),
  so cutting `n` from 192 to 128 shrinks every wide-poly type and
  every wide-poly multiplication kernel by the same factor.
- **Width rescale in `F2OpenProof<D>`** (the single algebraic change
  beyond mechanical s/192/128/):
    | Slot | GF192 | GF128 | Bound (D=32) |
    |---|---|---|---|
    | `q_i'` (lift target) | `BinaryF2Poly<3>` | `BinaryF2Poly<2>` | 128 b |
    | `b_g[i]` mid-product | `BinaryF2Poly<4>` | `BinaryF2Poly<3>` | D + n − 1 |
    | `b_vector`, `combined_row` entry | `BinaryF2Poly<7>` | `BinaryF2Poly<5>` | D + 2n − 1 |
    | `lifted_claim a'` | `BinaryF2Poly<10>` | `BinaryF2Poly<7>` | D + 3n − 2 |
  Const-generic call sites (`f2_poly_mul::<3,7,10>`,
  `f2_inner_product::<3,4,7>`, `encode_f2_lin_open::<7>`,
  `eval_f2_wide_poly_at::<10>`, `absorb_f2_poly_slice::<7|10, _>`)
  all rescaled by the same map.
- **Measured wins** (criterion, `--features parallel,simd,unchecked`,
  nvars=22, Apple M-series 8-core):

    | Bench | GF192 | GF128 | Speedup |
    |---|---|---|---|
    | `binary_gf/mul` | 3 ns | 2 ns | 1.5× |
    | `binary_gf/square` | 3 ns | 2 ns | 1.5× |
    | `binary_gf/inverse` | 2.53 µs | 1.51 µs | 1.68× |
    | `binary_gf/alpha_precompute` (D=32) | 277 ns | 178 ns | 1.56× |
    | `binary_gf/project_col_65536/branchless` | 685 µs | 484 µs | 1.42× |
    | `Micro/Open-a-AlphaBasis/nv=22` | 158.45 µs | 43.99 µs | **3.6×** |
    | `Micro/Open-b-LiftedEqTensor/nv=22` | 22.64 ms | 13.82 ms | **1.64×** |
    | `Micro/Open-d-CombinedRow/nv=22` | 101.33 ms | 76.20 ms | **1.33×** |
    | `Micro/UAIR-c-Sumcheck/nv=22` | 28.64 ms | 19.10 ms | **1.50×** |
    | `Micro/UAIR-b-AlphaProject/nv=22` | 1067 ms | 1065 ms | 1.00× ❌ |

  Proof bytes (per `proof_size_breakdown` table, nvars=22):
  `uair.alpha` 24 → 16, `uair.gamma` 24 → 16,
  `open.lifted_claim` 80 → 56, `open.b_vector` 448 → 320.
  Aggregate proof size shrinks ~150 KB at nvars=22 from these scalar
  components (open.combined_row dominates total bytes and is
  unaffected by the field swap — it's already F_2-poly-bound).

- **Why prove e2e barely moved** (Δ ≈ 1% on `Prove/nvars=22`,
  inside criterion's CI band): UAIR is dominated by
  `UAIR-b-AlphaProject` (1065 ms ≈ 90% of UAIR), which uses the
  **branchy** kernel `eval_f2_poly_d_at_with_powers` at
  [`f2_prove.rs:885`](../protocol/src/f2_prove.rs#L885). That
  kernel's per-set-bit work is dominated by `trailing_zeros` +
  `bits &= bits - 1` + memory load; saving one XOR per set bit (2
  vs 3 u64 words) is invisible. The branchless variant **does**
  benefit (`project_col_branchless 1.42×`) but a previous experiment
  showed it regresses ~8 ms on the actual SHA-256 trace (predictable
  bits favour branch prediction) — see comment at
  [`f2_prove.rs:868-876`](../protocol/src/f2_prove.rs#L868-L876).
  Net: the field-size lever is pulled, but the dominant cost is
  outside its reach. Next lever for prove latency would be a
  SIMD-batched α-projection that processes 2–4 cells per iteration to
  amortise the loop-control overhead, OR an alternative kernel shape
  that benefits from the 2/3 word reduction.
- **GF192 fate**: kept compiled, marked
  `#[deprecated(note = "F_2 prover uses BinaryFieldGF128; binary_gf192
  is kept only for the binary_gf_compare bench. ...")]` on
  `pub mod binary_gf192;` in `poly/src/univariate.rs`.
  `binary_gf_compare.rs` carries `#![allow(deprecated)]` so the
  comparison bench keeps running.
- **Out of scope** (deferred): soundness/security write-up updates in
  `f2_prove_plan.md` / `f2_open_plan.md` — per-challenge SZ error
  drops from `d/2^192` to `d/2^128`, still well above 100-bit total
  but a formal sum-of-errors derivation across all FS draws is owed.
  TODO already flagged at `binary_gf128.rs:31-56`. Itoh–Tsujii
  inverse remains naive Fermat (no production hot path hits it).

### Parallelise q1 lift in `build_lifted_eq_tensor` (commit `d8c14d0`, −86% LiftedEqTensor, −27% Prove e2e)
- **What**: `build_lifted_eq_tensor` (in `protocol/src/f2_prove.rs`)
  built the q1 lifted-eq tensor via a plain
  `q1_gf.iter().map(|g| basis.lift(g)).collect()`. At SHA-256 F_2
  nvars=22 with num_rows=8, q1 has 2^{22−3} = 524 288 entries, each
  one a 192×192 F_2 matrix-vec multiply (`AlphaPolyBasis::lift`,
  ~600 ops per call). All single-threaded.
- **Fix**: switched the q1 collect to
  `cfg_iter!(q1_gf).map(...).collect()`. The lift is
  data-independent across entries, so the parallelism is
  straight-line. q0 stayed sequential (it's only ≤ num_rows
  entries, ~8 — rayon spawn overhead dominates).
- **Measured (criterion `--save-baseline` / `--baseline`,
  nvars=22, `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                          | Before    | After    | Δ      |
  |--------------------------------|-----------|----------|--------|
  | Open-b-LiftedEqTensor/nv=22    | 163.2 ms  | 22.8 ms  | **−86.1%** (7.1×) |
  | Prove e2e/nvars=22             | 3.11 s    | 2.27 s   | **−27.3%** |

  Both p < 0.01. The 7.1× lift speedup matches the parallelism
  factor on M-series (8 cores). Prove e2e baseline of 3.11s was
  on the high end of run-to-run variance; conservative bound on
  the change is −18.5% (lower CI endpoint).
- **Bonus**: `verify_f2_open_with_virtuals` also calls
  `build_lifted_eq_tensor` (line 2237). Verifier gets the same
  parallel speedup for free.
- **Lesson**: scan for `.iter().map().collect()` patterns on
  large data where the per-element work is non-trivial — they're
  free parallelism wins. The audit missed this; finding it took
  a per-Open-micro bench breakdown to see that LiftedEqTensor
  was eating 161 ms out of ~745 ms Open.

### Commit slab: 64 MB capacity floor after GPU dispatch (commit `7550055`, RSS hygiene)
- **What**: `zip-plus/src/pcs/phase_commit.rs::commit_grouped`'s
  GPU branch caches a process-wide `Vec<u8>` (`COMMIT_SLAB_SCRATCH`)
  that grows on demand but was never shrunk. At SHA-256 F_2
  nvars=22 the slab reaches ~768 MB and stays there for the life
  of the process, even after smaller-nvars commits. Added
  `slab.shrink_to(64 * 1024 * 1024)` after the GPU dispatch +
  Merkle tree build, before releasing the Mutex.
- **Behaviour**:
  - `slab.shrink_to(min_cap)` sets `capacity = max(len, min_cap)`.
    `slab.len() == required` post-resize, so capacity becomes
    `max(required, 64 MB)` after the shrink.
  - `required > 64 MB` (e.g., nvars=22): no-op, no realloc.
    Steady-state same-nvars hot loops pay nothing.
  - `required ≤ 64 MB` (e.g., nvars=16 with our commit shape):
    capacity shrinks to 64 MB, releasing the historical peak.
- **Limitation**: this only releases memory when the *current*
  commit is small. The "ran nvars=22 once, now idle" case still
  holds the ~768 MB. Releasing across idle periods would need an
  explicit release API (a `pub fn release_commit_slab_scratch()`
  on `ZipPlus` that callers invoke between batches).
- **Measured (criterion `--save-baseline` / `--baseline`, nvars=22,
  `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                          | Before   | After    | Δ      | verdict |
  |--------------------------------|----------|----------|--------|---------|
  | Commit-Fused-GPU-Inline/nv=22  | 583.4 ms | 585.1 ms | +0.30% | no change (p=0.57) |
  | Prove/nvars=22                 | 2.54 s   | 2.47 s   | −2.74% | no change (p=0.58) |

  Initial Prove A/B run reported a +13% regression with p=0.02
  but a `[+3%, +24%]` confidence interval; the re-run gave
  −2.74% with p=0.58. Prove at nvars=22 has high run-to-run
  variance (10 samples × ~2.5 s each, with thermal/scheduler
  effects across 25–40 s of bench wall time). The Commit micro
  is the direct test of the change and shows no change in either
  run — confirming the shrink is no-op at nvars=22 as expected.
- **Verification**: 13/13 F_2 lib tests pass.
- **Lesson recorded**: at this nvars Prove takes long enough to
  swing 10%+ between criterion sample sets. For changes whose
  direct test (here: Commit micro) shows no change, trust that
  signal over a noisy higher-level bench. See also the GPU
  threadgroup-cap revert/re-ship saga, where the same
  "statistically significant" verdict turned out to be noise on
  a too-small baseline.

### PMULL-accelerated `f2_poly_mul` (commit `613de5d`, −21.5% Prove e2e, −92.4% coherence check)
- **What**: `poly/src/univariate/binary_f2_wide.rs::f2_poly_mul<W_A, W_B, W_OUT>`
  was a bit-by-bit schoolbook: for each set bit of `a`, XOR a
  word-shifted `b` into the accumulator (`xor_shifted`). Replaced
  with a word-level schoolbook using
  `binary_gf128::clmul_64x64` (PMULL on aarch64, PCLMUL on x86_64,
  scalar fallback otherwise): for each `(ai, bi)` pair, one 64×64
  carryless multiply produces a 128-bit partial product whose two
  halves XOR into `acc[ai+bi]` and `acc[ai+bi+1]`.
- **Why this matters**: the audit had categorised this as a
  verify-only optimisation ("coherence-check `<3,7,10>` mults at
  row_len scale"). In practice `f2_poly_mul` is used heavily by
  the **prover's** `prove_f2_open` too — multiple shapes
  (`<1,3,4>` / `<3,4,7>` / `<3,7,10>`) inside the encoding-
  consistency assembly + the per-column `b_vector` / `combined_row`
  builds. All shapes get the same ~5–10× speedup.
- **Measured (criterion `--save-baseline` / `--baseline`, nvars=22,
  `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                          | Before  | After    | Δ      |
  |--------------------------------|---------|----------|--------|
  | Prove e2e                      | 3.17 s  | 2.49 s   | **−21.5%** |
  | VerifyOpen-d-Coherence/nv=22   | 46.3 ms | 3.53 ms  | **−92.4%** (13×) |

  Both p < 0.01.
- **Implementation detail**: made `binary_gf128::clmul_64x64`
  `pub(crate)` so `binary_f2_wide` could reuse the existing
  hardware-dispatched helper (avoids duplicating the aarch64/x86_64
  cfg dance). Added `a_word == 0` / `b_word == 0` early-skip in
  each inner iteration so sparse high-order words don't pay PMULL
  cost.
- **Lesson**: the audit's "verify-only" claim was wrong because it
  didn't trace the function's call sites. A grep for
  `f2_poly_mul` in `protocol/src/` would have surfaced 11+ call
  sites in `prove_f2_open` / `verify_f2_open_with_virtuals`. When
  an audit recommends a function-level change, verify the
  function's actual call graph before sizing the impact.
- **Verification**: 93/93 zinc-poly lib tests pass under
  `--features simd`; 13/13 F_2 protocol lib tests pass.

### Metal Blake3 dispatch: lift `min(256)` threadgroup cap (commit `e4d6a25`, re-shipped after realistic-workload re-test)
- **What**: `zip-plus/src/metal_gpu/mod.rs:201-214`. Removed the
  hard-coded `min(256)` clamp on `threads_per_threadgroup`; the
  pipeline's own `max_total_threads_per_threadgroup()` now wins
  (already accounts for kernel resource use — per-thread private
  memory is ~864 B worst case from the 24-deep subtree-stack).
  Originally landed as `31c422b`, reverted in `58d2124` after a
  +9% A/B regression at Prove/nvars=16. After commit `6e40edd`
  scaled `NUM_COMPRESSIONS` with `num_vars`, a fresh A/B on the
  realistic shape reverses the verdict.
- **Measured (criterion `--save-baseline` / `--baseline`, realistic
  workload, `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench (nvars)               | Before (256 cap) | After (cap removed) | Δ      | criterion verdict |
  |-----------------------------|------------------|---------------------|--------|-------------------|
  | Prove/16                    | 47.4 ms          | 46.6 ms             | −2.24% | improved (p<0.01) |
  | Prove/20                    | 675 ms           | 677 ms              | +0.28% | no change (p=0.61) |
  | Prove/22                    | 3.45 s           | 3.49 s              | +1.15% | no change (p=0.75) |
  | Commit-Fused-GPU-Inline/22  | 584 ms           | 587 ms              | +0.50% | no change (p=0.38) |

  Net: neutral-to-positive everywhere on the realistic shape; the
  +9% regression at nvars=16 from the original A/B was noise on a
  ~30 ms baseline (a 3 ms swing is "statistically significant" at
  p<0.05 but practically within thermal/scheduler noise at that
  scale). The realistic-shape `Prove/nvars=16` baseline is ~47 ms,
  pushing the noise floor below the effect size.
- **Lesson**: the original revert decision was correct given the
  data available — but criterion's statistical significance test
  doesn't distinguish "real effect" from "thermal noise at a small
  baseline." When the baseline is small enough that a 3 ms swing
  is the entire reported delta, treat any verdict with skepticism
  and re-test under more loaded conditions.

### F_2 native IC: bounds-check elimination in `prove_linear` XOR-fold (commit `54a564b`, −10.1% Prove e2e)
- **What**: in `protocol/src/f2_native_ic.rs::prove_linear`, the two
  hot loops (up_evals at ~line 686, down_evals at ~line 714) read
  `coeffs[d] += &eq_table[row]` per set bit. Each access carried a
  bounds check that LLVM couldn't eliminate (the loop control
  structure doesn't make the bounds provable from rustc's vantage).
  Replaced with `get_unchecked_mut(d)` / `get_unchecked(row)` after
  proving the indices safe:
  - `d < D`: `bp_to_u64::<D>(cell)` masks bits above position `D−1`
    (`BinaryPoly<D>` invariant — all constructors mask; `Mul` is
    `unimplemented!()` so multiplication can't break it).
    `debug_assert!(D == 64 || bits >> D == 0)` guards the invariant.
  - `row < eq_table.len()` and `row < col.evaluations.len()` by the
    outer `for row in 0..usable_rows` bound and the trace shape.
  - In the shifts loop: `i = j − s` with `j ≥ s` and
    `j < s + usable_rows`, so `i < usable_rows`.

  Also hoisted `eq_val = eq_table.get_unchecked(row)` out of the
  inner `while bits != 0` loop and added a `bits == 0 → continue`
  short-circuit so the eq_table load is skipped on zero cells.
- **Why**: the audit's #4 had been dismissed at the artificial
  480-row workload as "1.1% Prove ceiling, not worth it." After the
  realistic-shape bench fix (commit `6e40edd`), the real ceiling
  was 9.6% Prove. The agent's specific proposal (loop inversion +
  bit-d masks) was still wrong (more iterations, big memory cost,
  no efficient NEON masked XOR-reduce — see the entry below in
  "Investigated, didn't help"). But the target was real, and the
  XOR-fold had per-iteration overhead LLVM wasn't eliding.
- **Measured (criterion `--save-baseline` / `--baseline`, nvars=22,
  `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                          | Before   | After    | Δ      |
  |--------------------------------|----------|----------|--------|
  | UAIR-a-F2NativeIC (= `prove_linear`) | 446.0 ms | 359.2 ms | **−19.4%** |
  | UAIR-FULL                      | 1.045 s  | 951.6 ms | **−8.9%**  |
  | Prove e2e                      | 3.90 s   | 3.51 s   | **−10.1%** |

  All three criterion-verdict "Performance has improved" with
  p < 0.01. 13/13 F_2 lib tests still pass.
- **Bounds-eliding scope is local**: only the inner XOR-fold's two
  loads. The cell load, the bit-iter primitives (`trailing_zeros`,
  `bits &= bits − 1`), and the GF(2^192) XOR itself are unchanged.
  This kind of micro-tweak shouldn't normally yield 19% — the
  size of the win is a signal that LLVM was paying a real
  per-iteration cost for the bounds checks at the inner-loop's
  unrolled-shape; possibly the bounds check defeated some
  vectorisation or register allocation.

### Scale chained-compression count with `num_vars` (commit `6e40edd`)
- **What**: `NUM_COMPRESSIONS: usize = 7` was a hardcoded constant
  set for the minimum `num_vars = 9`. At every larger `num_vars`,
  the trace still produced only 7 compressions (480 active rows)
  and zero-padded the rest. Replaced with
  `pub const fn num_compressions(num_vars) -> usize`, computed as
  `((1 << num_vars) − 4) / ROWS_PER_COMP` — the largest `N` such
  that `N · 68 + 4 ≤ 2^num_vars`. The witness generator now uses
  `let big_n = cols::num_compressions(num_vars)`.
- **Why this matters for past results**: every prior benchmark in
  this branch (and presumably in `claude/gkr-virtual-cols`, which
  has the same hardcoded constant) was measuring a 480-active-row
  workload at every nvars — 99.99% zero-padded at nvars=22. All
  optimization rankings and impact estimates from the audit were
  derived on that artificial shape. The realistic numbers shift
  the picture substantially.
- **Realistic-workload measurements** (criterion `--features
  parallel,simd,unchecked,metal_gpu`, post-fix):

  | nvars | Old Prove (480 active) | New Prove (realistic) | Active rows |
  |-------|------------------------|-----------------------|-------------|
  | 16    | 30.62 ms               | 47.18 ms (+54%)       | ~65 K       |
  | 20    | 437.7 ms               | 672.1 ms (+54%)       | ~1.1 M      |
  | 22    | 2.251 s                | 4.372 s (+94%)        | ~4.19 M     |

  At nvars=22 component breakdown:
  | Region                      | Time     | % of Prove |
  |-----------------------------|----------|------------|
  | UAIR-FULL                   | 1.004 s  | 23.0%      |
  | └ UAIR-a-F2NativeIC (= `prove_linear`) | 420.4 ms | 9.6% |
  | Commit (micro)              | 768.5 ms | 17.6%      |
  | └ Commit-Fused-GPU-Inline   | 581.6 ms | 13.3%      |
  | Open + misc                 | ~2.60 s  | ~59%       |

  UAIR is now the largest single phase; Open dominates the bucket
  after. `prove_linear` is **9.6% of Prove** — up from 1.1% on the
  artificial shape.
- **Implications for previously-recorded audit verdicts**:
  - The "Investigated, didn't help" entry for the audit's #4
    (`F2NativeIc::prove_linear` loop inversion) was decided on the
    artificial 1.1% number. The target is now 9.6% e2e; the
    specific approach the audit proposed is still wrong (see that
    entry), but `prove_linear` is now a meaningful optimization
    target and needs a different approach. Re-investigation
    pending.
  - The "skip zero rows" alternative I floated alongside #4 is
    **dead** — at realistic shape every row is active.
  - The GPU-cap A/B (commit `31c422b` reverted in `58d2124`)
    used the artificial shape — the +9% regression at nvars=16
    may or may not hold at realistic shape, but the revert was
    correct given the data we had.
  - The "1.5× commit/open regression vs `claude/gkr-virtual-cols`"
    earlier estimate was also on the artificial shape. The
    realistic comparison requires patching the same scaling fix
    onto `claude/gkr-virtual-cols` and re-benching — not done in
    this session.
- **Verification**: 13/13 F_2 protocol lib tests pass at nvars=9
  (where `num_compressions(9) = 7`, unchanged); 19/19 test-uair
  lib tests pass. Compile clean under
  `--features parallel,simd,unchecked,metal_gpu`.

### Parallelise SHA F_2 witness gen post-loop builders (commit `<TBD>`)
- **What**: in `test-uair/src/sha256_f2.rs::generate_random_trace`,
  switch the per-row builder loops that run *after* the
  sequential compression rounds to `cfg_iter!` / `cfg_into_iter!`
  with a `PAR_MIN_LEN = 4096`-row split threshold:
  - 6 Σ / σ / SHR derived columns
    (`sigma0_vals`, `sigma1_vals`, `sig0_vals`, `sig1_vals`,
    `shr3_w_vals`, `shr10_w_vals`)
  - 3 Ch / Maj operands (`u_ef_vals`, `u_neg_e_g_vals`, `maj_vals`)
  - 1 packed-`PA_C` compensator builder (the largest single
    loop: 13 per-step κ-residue extractions per row across all
    `2^num_vars` rows)
  Plumbing: new `parallel` feature on `zinc-test-uair`
  (`rayon` optional dep + `zinc-utils/parallel` propagation);
  `zinc-protocol`'s `parallel` feature now also enables
  `zinc-test-uair/parallel`. The 4096-row min_len keeps small
  fixtures (`nvars = 9`, `n = 512`) on the single-thread path so
  rayon's task-spawn overhead doesn't dominate.
- **Why**: the step-2+3 chained-Binius rewrite traded the prior
  CSA-flattened impl's bounded-active-row CSA work for an
  unconditional `O(n)` packed-`PA_C` builder. At `nvars = 22` this
  alone caused a ~170 ms (+170%) regression in `WitnessGen`, the
  only stage that got slower in step 2+3. The post-loop builders
  are embarrassingly parallel — each row is independent — so the
  fix is mechanical: `cfg_into_iter!(0..n, MIN_LEN)`.
- **Measured impact** (Apple M-series, `--features
  parallel,simd,unchecked`, vs the `pre-step23` baseline saved
  before any of the alignment work):
  | nvars | Stage      | Pre step-2+3 | Post step-2+3 + parallel | Δ      |
  |-------|------------|--------------|--------------------------|--------|
  | 9     | WitnessGen | 76.2 µs      | 12.1 µs                  | −84%   |
  | 9     | Prove      | 2.83 ms      | 2.71 ms                  | −5%    |
  | 9     | Verify     | 2.58 ms      | 1.89 ms                  | −26%   |
  | 20    | WitnessGen | 25.7 ms      | 12.8 ms                  | −50%   |
  | 20    | Prove      | 888 ms       | 543 ms                   | −39%   |
  | 20    | Verify     | 148 ms       | 112 ms                   | −24%   |
  | 22    | WitnessGen | 102.7 ms     | 53.3 ms                  | −49%   |
  | 22    | Prove      | 6.96 s       | 4.09 s                   | −41%   |
  | 22    | Verify     | 585 ms       | 453 ms                   | −22%   |

  Net wall clock at `nvars = 22`: 7.65 s → 4.60 s, **−40%**.
- **Verification**: all 12 `f2_prove::tests` pass.
- **Out of scope (not pursued)**: the sequential compression
  loop (`for j in 16..rpc { ... }` plus the round-update body)
  is *not* parallelised — each iteration depends on prior rows'
  outputs (`w_vals[t - 16]`, `a_vals[k + 3]`, etc.), so the
  inter-row dependency chain rules out a straight `par_iter`.
  A per-compression `cfg_join!` across the 7 independent
  compressions would help (each compression block of 68 rows is
  independent up to the H_0 input), but at `nvars = 22` the
  sequential compression work is already dwarfed by the
  post-loop O(n) builders, so the payoff is small.

### Chained-Binius rewrite of C5–C11 with packed `PA_C` (steps 2+3 of 5, commit `<TBD>`)
- **What** — full structural alignment of the SHA-256 F_2 UAIR with the
  `sha-f2x-doc` spec at the AIR-shape level, modulo the still-deferred
  Hadamard discharge (step 5):
  1. `protocol/src/f2_native_ic.rs` — replaced the zero-stub at
     `prove_linear` and `prove_combined` (lines that used to push
     `DynamicPolynomialF::ZERO` / `F2RowExpr::zero()` for each bit-op
     spec) with proper per-cell `BitOp::Rot(c)` / `BitOp::ShiftR(c)`
     materialisation. For `prove_linear` (the MLE-first path the SHA
     F_2 UAIR takes), bit-op virtuals' MLE eval coefficients are a
     re-indexing of the source column's `up_evals` coefficients —
     `O(D)` per spec, no extra per-cell loop. For `prove_combined`,
     per-row bits flow through `v.rotate_left(c)` / `v >> c`. Unlocks
     UAIR-internal `BitOpSpec` end-to-end for any F_2 UAIR (the
     verifier-side machinery in `verifier.rs` / `prover.rs` was
     already in place; only the F_2-native IC fast path lagged).
  2. `test-uair/src/sha256_f2.rs` — full rewrite of C5–C11:
     - **Columns**: dropped the 6 CSA-majority columns (`W_M_W1`,
       `W_M_W2`, `W_M_T1_{1..4}`), the 7 per-add full-carry columns
       (`W_C_{W,T1,T2,A,E,FF_A,FF_E}`), and the 7 scalar compensator
       columns (`PA_C_{W,T1,T2,A,E,FF_A,FF_E}`). Added 6
       chained-Binius intermediate-sum columns (`W_W_S{1,2}`,
       `W_T1_S{1..4}`) per doc §3.7 Definition 1 and 1 packed
       compensator column `PA_C` (bits 0..12 hold the 13 per-step κ
       compensators per the doc's `binius-packing` allocation; bits
       13..31 are zero). Total column delta: 14 public + 27
       committed = 41 → 8 public + 20 committed = 28 (−13).
     - **Constraints**: replaced the 7 polynomial CSA-flattened
       `LinSum` checks with 13 per-step `BiniusAdd` LSB scalar
       constraints (doc §3.6 `BiniusAdd`): each is
       `(target + x + y + κ_k) ∈ (X)`, where κ_k is bit j_k of
       `PA_C`. Bit extraction via 12 internal `BitOpSpec`s
       (`SHR^j(PA_C)` for j ∈ {1..12}; κ_0 uses `PA_C` directly).
     - **Ideal**: added `Sha256F2Ideal::LsbX` for the principal
       ideal `(X)` (coefficient at position 0 = 0, positions 1..31
       unconstrained). `RotXw1` keeps C1–C4; `ShiftX32` from step 1
       is dormant but kept as a variant for the eventual Hadamard
       wiring.
     - **Witness gen**: dropped `binius_carry` / `csa_step`
       helpers; replaced CSA layers with chained `wrapping_add`
       calls materialising the partial sums. `PA_C` builder packs
       13 per-step LSB residues (= 0 on rows where the witness is
       consistent with the binary add, residue otherwise).
- **Why** — direct shape parity with the doc's column and constraint
  layout, with the per-step structure that step 5 (external Hadamard
  discharge) can plug into without further rewrites. The CSA-flattened
  LinSum was a closed-form trick that didn't generalise.
- **Status vs. soundness** — step 2+3 swaps one AIR-level-incomplete
  scheme for another; both rely on honest witnesses for bits 1..31
  of each addition. The 13 per-step LSB checks pin only bit 0; bits
  1..31 are pinned by the column-level Hadamard
  `(x + X·c) ⊙ (y + X·c) = c + X·c` in `F_shift`, which is still
  external/deferred per doc §3.4. AND-side Hadamards (C12–C14) are
  also deferred. **No new soundness regression**: the prior CSA-
  flattened LinSum was equally AIR-level-incomplete (the CSA
  majority columns and final carries were under-constrained by the
  algebraic identity).
- **Verification** — `cargo test -p zinc-protocol --features parallel
  --lib f2_prove::tests` (12 tests, all pass, incl.
  `prove_then_verify_sha256_f2_roundtrips`); `cargo test -p
  zinc-test-uair --lib sha256_f2` (4 tests, all pass). Benches build
  clean.
- **Out of scope (steps 4+5 still open)**:
  - Step 4 (the LSB check unfold of `BiniusAdd`) — *folded into this
    change*, since the AIR-level half of `BiniusAdd` is exactly the
    LSB check; the Hadamard half is step 5.
  - Step 5 — wire up the external Hadamard-product discharge
    (`(x + X·c) ⊙ (y + X·c) = c + X·c` in `F_shift`) for the 13
    addition-side and 3 AND-side relations. Will need:
    - the `W_β` packed carry-out column (deferred along with step 5
      — committing it without consumers wastes commit cost),
    - virtual carry word `c` reconstruction (bits 0..30 from
      `SHR^1` of XOR-of-(target,x,y); bit 31 = β_k from `W_β`),
    - a Hadamard-product PIOP. Mechanism is a separate design pass.
- **Caveat carried over** — `cargo build -p zinc-protocol` *without*
  `--features parallel` still fails in `f2_prove.rs:603` from
  `cfg_iter!` shape, pre-existing.

### C5–C11 modular-addition constraints moved into `F_shift = F_2[X]/(X^32)` (commit `<TBD>`)
- **What**: `test-uair/src/sha256_f2.rs`. Step 1 of aligning the F_2
  SHA-256 UAIR with `documentation/sha-f2x-doc/`.
  1. `Sha256F2Ideal` lifted from a unit struct to a payload-free
     enum with two variants: `RotXw1` (`(X^32 − 1)` — `F_rot`) and
     `ShiftX32` (`(X^32)` — `F_shift`). `IdealCheck::contains`
     dispatches on the variant; `ShiftX32` membership is "coefficients
     at positions 0..32 are all zero," matching the principal
     ideal `(X^32)`.
  2. The constraint impl now uses two ideal handles: `ideal_rot` for
     the rotation identities (C1–C4) and `ideal_shift` for the
     modular-addition LinSum constraints (C5–C11). C12–C18 are
     unaffected (boundary pins use `assert_zero`; AND Hadamards are
     external).
  3. The witness generator's `PA_C_*` compensator builders now use
     non-cyclic `(x << 1)` (`shl1`) instead of `rotl1` for the
     `X·m_j` / `X·c_*` contributions. Under the `F_shift` convention,
     `X·c` drops bit 31 of `c` via the quotient — the compensator
     residue must match. The constraint expression `mbs(w_c_*,
     x_scalar)` is unchanged: in `F_shift` the resulting polynomial's
     bit-32 coefficient is unconstrained, so the `IdealCheck` only
     pins bits 0..31.
- **Why**: the spec's Lemma `binius-vc` lives in `F_shift` for good
  reason — the `(X^32)` quotient kills the bit-31 carry-out
  automatically, so the per-step Binius Hadamard `(a + X·c) ⊙ (b +
  X·c) = c + X·c` is a *ring* equation in `F_shift`, not a relation
  that has to know about cyclic wrap. Under the prior `RotXw1`
  framing, bit-31 of each `m_j`/`c_*` wrapped to bit 0 via cyclic
  rotation, which (a) made `PA_C_*` non-zero on honest active rows
  whenever the final carry-out was 1, and (b) coupled bit-32 effects
  back into the LSB check — a structural divergence from the doc that
  this swap eliminates. Under `ShiftX32`, `PA_C_*` is zero on every
  active row as the doc specifies (the modular-add identity is exact
  in `F_shift`).
- **Verification**: all 12 tests in
  `cargo test -p zinc-protocol --features parallel --lib
  f2_prove::tests` pass, including
  `prove_then_verify_sha256_f2_roundtrips`. The four
  `zinc-test-uair` `sha256_f2` shape-level lib tests pass.
- **Status vs the broader spec alignment plan**: this is step 1 of 5
  from the gap analysis in conversation. Done: ideal split + matching
  witness compensator. Remaining:
  - **(2)** Rip out the CSA-flattened linear identity in favour of
    chained-Binius per-step `BiniusAdd` constraints. Replace the 6
    `W_M_*` and 7 `W_C_*` columns with the doc's 6 chained-Binius
    intermediate-sum columns (`W_W^{(1,2)}`, `W_{T_1}^{(1..4)}`) +
    1 packed `W_β` carry-out column.
  - **(3)** Collapse the 7 scalar `PA_C_*` columns into one packed
    bit-poly `PA_C` with the doc's bit allocation.
  - **(4)** Rewrite C5–C11 as 13 per-step `BiniusAdd` constraints
    (each = one LSB equation + one column-level Hadamard relation).
  - **(5)** Register the 13 addition-side Hadamards + 3 AND-side
    Hadamards with whatever Hadamard-product mechanism we adopt.
- **Caveat — pre-existing build state**: `cargo build -p zinc-protocol`
  *without* `--features parallel` does not compile on this branch
  (errors in `f2_prove.rs:603` from `cfg_iter!` expanding to
  `Iterator::reduce` with a rayon-shaped call). This is independent
  of the ideal swap and was on the baseline tree; flagged so the
  next agent doesn't blame this change.

### Multipoint-eval phase wired into the F_2 prove path (commit `<TBD>`)
- **What**: added a `MultipointEval<BinaryFieldGF192>` step between
  the F_2 sumcheck and the PCS open. The phase reduces the per-column
  MLE evaluation claims at the sumcheck point `r*` (carried in
  `F2Proof.column_evals_at_rstar`) to a single set of MLE eval
  claims at a fresh point `r_0`, batched across columns. The
  F_2 PCS open now happens at `r_0` instead of `r*`.
  Proof shape: `F2FullProof` gains `multipoint_eval:
  MultipointEvalProof<BinaryFieldGF192>` + `open_evals_at_r_0:
  Vec<BinaryFieldGF192>` (length = num primary + num virtual cols).
- **Why**: the multipoint-eval primitive is the standard mechanism
  for reducing row-shifted MLE eval claims to non-shifted ones at
  a related point — the prerequisite for sound row-shift discharge
  in the K-virtual cols (Issue 1 below) and for the row-shifted
  bit-op virtuals that `f2_native_ic.rs:735-740` currently punts on.
  The integer protocol already runs this (Step 5 in
  `protocol/src/prover.rs`); the F_2 path was the laggard.
- **Today this is degenerate overhead.** The F_2 sumcheck doesn't
  surface any row-shifted column evals yet (no Hadamard product
  support — see the inline comment in `f2_native_ic.rs:735-740`),
  so multipoint-eval is called with empty `down_evals` + `shifts`.
  It just rerandomises the open point and γ-batches up_evals. The
  payoff lands when Hadamard / K-discharge wires shifted-col evals
  into the sumcheck output — those flow through
  `MultipointEval::prove_as_subprotocol`'s `shifts` + `down_evals`
  params without further protocol changes.
- **Perf cost** (estimated, not measured): one extra α-projection
  per prove (~500 ms at nvars=22) because the recompute pattern
  duplicates work already done inside
  `prove_f2_uair_with_groups` (where the original `projected_trace`
  is consumed by the column-evals-at-r* materialisation). Plus one
  additional MLE-eval pass per col at `r_0` (~few hundred ms at
  nvars=22), one degree-2 sumcheck of `num_vars` rounds, and
  `num_total_cols × 24 B` extra proof bytes for `open_evals_at_r_0`.
  Bench will show the regression.
- **Followups**:
  - ~~Refactor to avoid the α-projection recompute~~ **shipped** —
    `prove_f2_uair_with_groups` now returns `projected_trace` as a
    third element; the multipoint-eval phase consumes it directly.
    The savings at small nvars are within bench noise (the inline
    α-recompute was cheaper than my initial estimate), but the
    refactor's cost (one per-col `Vec` clone in
    `column_evals_at_rstar`) scales with `cells × bytes` while the
    saved recompute scales with `cells × ops`, so the win grows
    with `2^N`. See subsequent commit.
  - **Tamper test for the new path**: existing
    `verify_f2_full_rejects_tampered_*` tests cover the sumcheck +
    open paths; add an analogue that tampers with
    `multipoint_eval` or `open_evals_at_r_0` and confirms the
    correct error variant fires.

### Per-cell SHIFTR virtualisation: in-tree canary (commit `<TBD>`)
- **What**: extended `prove_then_verify_sha256_f2_with_k_virtuals_roundtrips`
  in `protocol/src/f2_prove.rs` to also pass `F2BitOpVirtualSpec`s for
  `W_SHR3_W = SHR^3(W_W)` and `W_SHR10_W = SHR^10(W_W)`. Adjusted the
  expected `batch_size` to account for the two extra excluded cols.
- **Why**: the bench (`protocol/benches/f2_sha256.rs`) already declares
  these specs via `sha_f2_bit_op_virtuals()` and threads them through
  every prove/verify callsite (shipped in commit `fdd2d01`), so the
  commit-side savings are already realised at bench time. But no CI-
  runnable test exercised the full prove-then-verify roundtrip with
  non-empty `bit_op_specs` on the real SHA-256 F_2 UAIR — the two
  in-tree tests (`prove_then_verify_sha256_f2_roundtrips` and
  `..._with_k_virtuals_roundtrips`) both passed empty `bit_op_specs`.
  This entry closes that gap so a regression in the per-cell SHIFTR
  reconstruction (`apply_bit_op_u32` at f2_prove.rs:300 or the
  `verify_f2_open_with_virtuals` derivation loop at f2_prove.rs:2575-2700)
  fails a test, not just a bench.
- **Where**: `protocol/src/f2_prove.rs` (test `prove_then_verify_sha256_f2_with_k_virtuals_roundtrips`,
  lines ~4797-4836 after the edit).
- **Related future work — trace-builder JIT (not in this entry)**:
  `Sha256F2Uair::generate_random_trace` still materialises
  `W_SHR3_W` and `W_SHR10_W` row-by-row (lines 836-837 + the MLE push
  at 1052-1053), even though they're now excluded from the commit.
  Cutting that work needs the prover to JIT-materialise bit-op virtual
  MLEs inside `prove_f2_full_with_bit_ops` before
  `prove_f2_uair_with_groups` consumes the trace — currently the
  caller is expected to pre-materialise them at the declared
  `col_idx`. Estimated savings at nvars=22: ~30 MB allocation +
  ~5 ms CPU per prove (2 cols out of 41). Symmetric with the K-cols
  pattern, which also pre-materialises and accepts the cost.

### Reuse the 768 MB commit slab process-wide (commit `<TBD>`)
- **What**: cached the `commit_grouped` GPU-inline slab in a
  process-wide `static OnceLock<Mutex<Vec<u8>>>` in
  `zip-plus/src/pcs/phase_commit.rs`. The Mutex is held for the
  full GPU-branch duration. Slab grows on demand, never shrinks
  — for SHA-256 F_2 the size is constant per protocol, so the
  resize is a one-time first-call cost. No zero-init needed
  between commits since `scatter_matrix_into_gpu_slab` fully
  overwrites the active region.
- **Why**: each commit was allocating a fresh 768 MB
  `Vec<u8>` then writing through it (cold pages, cold cache).
  Caching reuses the same DRAM pages — they stay resident across
  commits and the second touch hits warm cache.
- **Result at nvars=22**:
  `Commit-AfterPrevProve` 772 ms → 566 ms (−206 ms, −27%).
  `Prove e2e` 2.17 s → **1.53 s** (−640 ms, −29.5%).
  Cumulative since CPU baseline: 2.96 s → 1.53 s (−48%).
  The expected-gain estimate (100–200 ms) substantially under-
  shot the actual gain — the cold-DRAM penalty on a fresh
  768 MB write-through was far larger than the alloc cost alone.

### Stream public-col absorb per-column (commit `c5b1c64`)
- **What**: replaced the 235 MB single-buffer absorb in
  `protocol/src/lib.rs::absorb_public_columns` with a per-col
  reusable 17 MB scratch. Blake3 `update` is associative, so the
  post-absorb transcript state is byte-identical.
- **Why**: the 235 MB alloc + cold-cache Blake3 hash cost ~150 ms
  inside the e2e prove loop (where UAIR + Open had just evicted L3),
  vs ~40 ms in a tight micro-bench loop.
- **Result at nvars=22**:
  `Commit-AfterPrevProve` 900 ms → 772 ms (−128 ms, −14%).
  `Prove e2e` 2.34 s → 2.17 s (−170 ms, −7%).

### Drop UAIR clones in `prove_f2_uair_with_groups` (commit `22f618a`)
- **What**: (a) skip `trace.binary_poly.iter().cloned().collect()`
  when `virtual_specs.is_empty()` via a `Cow<'_, [..]>`; (b) move
  `eq_table` and `weighted_col` into `F2EqColRound1FastPath` instead
  of cloning.
- **Why**: trace clone was ~560 MB / ~110 ms at nvars=22; eq+wcol
  clone was ~192 MB / ~75 ms. Both invisible to the UAIR-a..d
  kernel sub-benches because they happened in the "wrap-up" between
  sub-steps.
- **Result at nvars=22**: `UAIR-FULL` 1.22 s → 0.96 s (−260 ms,
  −21%). Surfaced via new `UAIR-FULL` micro bench.

### Pre-paired commit + prove entry points (commit `744f0bd`)
- **What**: new `commit_pre_paired_witness`,
  `commit_and_absorb_pre_paired_witness`,
  `prove_f2_full_pre_paired_with_bit_ops` that take a pre-paired
  witness slice. `ProverFixture` pre-pairs once in `setup_prover`.
- **Why**: pairing + spec translation + MLE clone shouldn't be on
  the per-iter hot path; conceptually it's witness gen.
- **Result at nvars=22**: `Commit-micro` 790 ms → 561 ms (−229 ms,
  −29%). e2e Prove −130 ms.

### Blake3 transcript: `update_rayon` for big absorbs (commit `744f0bd`)
- **What**: `Blake3Transcript::absorb_inner` routes inputs ≥256 KB
  through `Hasher::update_rayon`. Enabled via
  `features = ["rayon"]` on workspace `blake3` dep.
- **Why**: the 235 MB public-col absorb (now ~17 MB per-col after
  the streaming rewrite, still well above the threshold) parallelises
  4-8× on M-series.
- **Result**: rolled in with the pre-paired commit; pre/post can't
  be cleanly isolated. Estimated savings ~100 ms.

### Large-leaf Metal Blake3 kernel + GPU-inline-fused commit (commit `9b8f178`)
- **What**: extended `hash_kernels.metal` from bounded 64-chunk
  multi-chunk path to a streaming subtree-stack kernel (handles up
  to 16 GB / leaf). Lifted the matching 64 KB asserts in
  `merkle.rs` and `metal_gpu/mod.rs`. Dropped `commit_grouped`'s
  dispatch threshold to `num_leaves ≥ 256` (no per-leaf-bytes cap).
- **Why**: SHA-256 F_2's commit shape has `num_leaves=262144` at
  nvars=22 — plenty of GPU parallelism — but the kernel's hard 64 KB
  / leaf cap blocked GPU dispatch entirely.
- **Result at nvars=22**: e2e Prove 2.96 s → 2.40 s (−19%). At
  nvars=16: 32 → 25 ms (−21%). At nvars=9 the threshold
  (`num_leaves=32 < 256`) keeps CPU; no regression.

### Filter standalone benches via `pair_primary_witness_polys_pub` (commit `9b8f178`)
- **What**: bench helpers (`Commit-Pair / Encode / Fused / GPU /
  GPU-Inline` in Steps + Micro groups) now use
  `pair_primary_witness_polys_pub` instead of pairing the full
  35-col trace. They now measure the same shape the production
  commit produces.
- **Why**: standalone benches were overstating commit cost by
  ~2.6× by pairing public + virtuals.

### Combined-K virtual cols + (X^32) ideal (commits `d3c4636`, `7db1e68`, `2274425`)
- **What**: replaced Binius-style per-add carry-save tree (6 majorities
  + 7 carries) with a single combined-K column per addition, under
  the `(X^32)` ideal. New `F2KVirtualSpec` excludes the 7 K cols
  from commit + γ-batched open.
- **Why**: cuts witness col count 27 → 21 (commit `7db1e68`), then
  the K-virtualisation cuts paired batch_size 11 → 7 (commit
  `2274425`), shrinking PCS opened/values by ~40%.
- **Result at nvars=22**: opened/values 5.05 MB → 3.03 MB raw;
  total proof zstd-3 950 KB → 677 KB (−29%).
- **Soundness caveat**: K cols' MLE evals at r* are currently
  trusted (Issue 1 — see `protocol/src/f2_gkr_plan.md`). Local
  soundness for additions moves to the SHA-256 boundary check.
  Full sound discharge needs the "ShiftR-of-MLE" mechanism.

---

## Investigated, didn't help

### GPU discharge (offload the oblong word-fold to Metal) — measured SLOWER than parallel CPU at nvars=16 (NOT worth it)
- **Hypothesis**: the discharge's Phase-1→Phase-2 word-fold (`fold_word_at`: project a
  bit-word at GF(2^128) weights → one GF128) is the *same op* as the α-projection that
  already has a Metal kernel (`project_columns_with_powers_gpu_batched`). So offload the
  fold to GPU by reusing that kernel — the largest remaining GPU lever (the discharge is
  the biggest CPU component, ~28 ms; the commit + α-projection are already GPU'd).
- **Spike** (temporary, in `prove_oblong_and_batch_gf8`; the operand words + GPU access
  are both there): time the CPU par-fold vs (u32→`BinaryPoly<32>` convert + the GPU
  projection kernel, batched over a/b/c) on the real stacked operands, nvars=16
  (~1M words ×3 = 3M cells), `--features …,metal_gpu`, **unsandboxed** (sandbox blocks
  Metal). Data-independent timing.
- **RESULT (Apple M4, warm)**: **CPU par-fold ~4.3 ms vs GPU ~6 ms + convert ~0.46 ms =
  ~6.5 ms — the GPU is ~50% SLOWER.** (First GPU call 42 ms cold; warm steady ~6 ms.)
  The discharge fold (~3M cells) is in the **CPU-wins** regime: Metal dispatch +
  unified-memory buffer + launch overhead (~fixed few-ms) exceeds the actual fold work,
  which 10 rayon cores + SIMD already do in ~4 ms. This is *unlike* the commit's Merkle
  leaf-hash (hundreds of thousands of leaves at nvars=22) where the GPU wins.
- **Conclusion**: a GPU discharge fold would **regress** the prove at the bench/typical
  scale (nvars=16). It could only win at very large nvars (overhead amortizes) and would
  then need a `GPU_MIN`-style word-count threshold to avoid regressing small cases — plus
  cross-crate callback plumbing (the discharge is in the `poly` crate, no `zip-plus` GPU
  access) and a u32→u64 conversion (the kernel is `BinaryPoly`/u64-typed). The bigger
  discharge phases (the GF(2⁸) NTT round-message, the Gruen Phase-2 loop) have **no
  existing GPU kernel** and would face the same small-scale dispatch-overhead problem.
  **Not implemented.** (Couldn't measure the large-nvars crossover: the nvars=19 spike
  OOM'd on the 16 GB box.) The CPU word-fold parallelization (`7ea7dff`) already captured
  the realistic win here.

### Depth-0 IPRS (Vandermonde column-sum) encode for `{0,1}` inputs vs depth-`d` radix-8 FFT (investigated — wins only for small `row_len`; NOT a general win)
- **Scope note**: this is the **integer IPRS** code
  (`zip-plus/src/code/iprs.rs`), the encoder used on the `Z[X]`/`e2e`
  path — *not* the F_2 SHA fast path, which encodes with `RaaF2Code`
  (XOR, no widening). Logged here anyway because it touches `iprs.rs`
  and `merkle.rs` and the entry-size finding generalises.
- **Hypothesis**: a depth-0 IPRS code *is* the full
  `codeword_len × row_len` Vandermonde matvec (at `depth==0`
  `combine_stages` is a no-op; `base_multiply_into_output` does the whole
  product against the materialised `base_matrix`). For a binary message
  `v ∈ {0,1}^k` the matvec degenerates to **summing the generator columns
  at the set positions** — `output[i] = Σ_{c: v[c]=1} base_matrix[i][c]` —
  i.e. *zero multiplications*, just `weight(v)` column-adds per output
  cell. Because `|base_matrix[i][c]| ≤ (p−1)/2 = 2^15` (centered mod
  `p=65537`), the entries also stay small (`≤ row_len·2^15`), so the
  "naive RS over integers blows up" problem does **not** apply to binary
  inputs. Question: does this beat the FFT, and do the smaller entries
  cut the Merkle commit?
- **Setup**: added `IprsCode::encode_binary_columnsum` (depth-0-only,
  multiplication-free) and an ignored harness
  `code::iprs::columnsum_experiment::experiment_depth0_columnsum_vs_fft`
  (run: `cargo test -p zip-plus --release --features parallel
  experiment_depth0_columnsum_vs_fft -- --ignored --nocapture`). The
  harness asserts column-sum is **bit-identical** to the depth-0
  `LinearCode::encode` output, times encode of a 64-row binary matrix
  (min over 12 reps), and times the grouped Merkle commit
  (`new_from_row_major_grouped`, `group_size=1`) at each codeword's
  minimal cell width and at a shared `i128` baseline. `Eval=i32`,
  `Cw=i128`, `CHECK=false`, Apple M-series.
- **Measurement** (encode = ms/matrix of 64 rows; commit@min = grouped
  Merkle at the smallest int cell that fits):

  | row_len | rate | FFT enc | d0-colsum enc | colsum vs FFT | FFT max-bits (cell) | d0 max-bits (cell) | commit FFT@min | commit d0@min |
  |---------|------|---------|---------------|---------------|---------------------|--------------------|----------------|---------------|
  | 64      | 1/4  | 2.01 ms | 1.53 ms       | **1.32× faster** | 33 (8B) | 19 (4B) | 0.160 ms | 0.177 ms |
  | 64      | 1/8  | 2.95 ms | 1.72 ms       | **1.72× faster** | 33 (8B) | 19 (4B) | 0.214 ms | 0.195 ms |
  | 256     | 1/4  | 2.08 ms | 2.07 ms       | 1.01× (tie)      | 35 (8B) | 21 (4B) | 0.269 ms | 0.209 ms |
  | 256     | 1/8  | 4.21 ms | 2.84 ms       | **1.48× faster** | 49 (8B) | 20 (4B) | 0.423 ms | 0.314 ms |
  | 1024    | 1/4  | 6.24 ms | 17.31 ms      | 0.36× (FFT 2.8× faster) | 36 (8B) | 22 (4B) | 0.586 ms | 0.422 ms |
  | 1024    | 1/8  | 8.51 ms | 40.14 ms      | 0.21× (FFT 4.7× faster) | 50 (8B) | 22 (4B) | 1.044 ms | 0.689 ms |

- **Why / crossover**: column-sum is `O(REP·k²)` adds; the optimal-depth
  FFT is `~O(REP·k·base_len)` with `base_len ≤ 256`
  (`MAX_BASE_COLS_LOG2=8`). The ratio `≈ (k/2)/base_len`, so the
  crossover sits at `row_len ≈ 2·base_len ≈ 256–512` — confirmed: a tie
  at 256, FFT decisively ahead by 1024. Below the crossover, column-sum's
  *no multiplies + no recursion + no twiddle loads* wins; above it the
  quadratic dominates. (Skipping just the 0/1 multiply within depth-0 —
  `d0-mul` vs `d0-colsum` — buys ~25% at row_len=1024 but can't rescue
  the quadratic.)
- **Bonus finding (entry size)**: for binary input the **FFT produces
  LARGER integers than depth-0**, not smaller — its intermediate twiddle
  products inflate (33–50 bits, needing an 8-byte cell; 49–50 bits at
  rate 1/8 with depth 2) while the direct column-sum stays at 19–22 bits
  (a 4-byte cell, rate-independent). That narrower cell makes the depth-0
  Merkle commit ~1.3–1.5× cheaper at the minimal width (e.g. 1024@1/8:
  0.69 ms vs 1.04 ms) and would also shrink column-opening bytes in the
  proof. At a shared `i128` cell the two commits are equal (commit is
  dimension-bound), so the saving is entirely from being able to size the
  cell to depth-0's smaller entries.
- **Verdict / residual opportunity**: NOT a drop-in win — at the row
  lengths Zip+ normally runs the FFT (large `row_len`, few rows) it's
  multiple-× slower to encode, and depth-0 also materialises the full
  `codeword_len × row_len` matrix (only feasible for `codeword_len ≤ 2^16`
  and memory-heavy near that bound). The real, unpursued opportunity is a
  **many-short-rows Zip+ layout** (`row_len ≤ 256`, large `num_rows`)
  feeding binary witness columns: there column-sum wins on encode *and*
  the smaller entries cut commit + proof size. Whether that layout's
  larger row count (more column openings) pays for itself end-to-end is
  the open question — would need a full-prover A/B, not just the encode +
  commit micro measured here. The `encode_binary_columnsum` method +
  harness are left in the tree (ignored) for that follow-up.
- **Lesson**: "lift the algorithm, not the code" is the right call for
  *general* integer messages, but for genuinely binary messages the naive
  Vandermonde is both bounded-growth AND multiplication-free — its only
  problem is the `O(k²)` cost, so it's a small-`k` technique, and it
  happens to yield *smaller* codeword entries than the FFT.

### Precompute the `γ'^k·σ^b` weight table in the Hadamard zerocheck comb (rejected)
- **Hypothesis**: the degree-3 Hadamard comb does
  `gpow * sigma_powers[b] * (u·v − w)` = 3 GF(2^128)-mults/term in rounds
  2..n. Folding the two batch challenges into one precomputed
  `weights[k·D + b] = γ'^k·σ^b` (constant across hypercube points) drops it
  to 2 mults/term — expected ~10–17% on Prove-Hadamard.
- **Implementation**: added a flat `weights` field to
  `HadamardRound1FastPath` (replacing `gamma_powers`/`sigma_powers`) and
  built it once in `prepare_hadamard_group_with_fast`, used in both
  `round_1_message` and the `comb_fn`. Value-preserving (the
  `fast_path_matches_generic` byte-identical-proof test still passed; all
  59 piop + 50 protocol lib tests green).
- **Measurement (rigorous Criterion A/B, nvars=16, same machine state via
  `git stash` + `--save-baseline`/`--baseline`)**: change
  **[+1.5% +6.9% +15.4%] (p=0.03) — regressed.** Reverted.
- **Why it didn't help**: rounds 2..n are **memory-bandwidth bound**, not
  arithmetic-bound. The generic multi-degree sumcheck rebuilds a
  1537-element value array per hypercube point (the eq MLE + 1536 slices);
  that traffic dominates. Trading a cheap CLMUL field-mult for a load from
  a larger 512-entry `weights` table (vs the tiny, cache-resident 16-entry
  `gamma_powers` + 32-entry `sigma_powers`) is net-neutral-to-negative.
  **Lesson / culprit for the discharge cost**: the real bottleneck is the
  generic per-point value-array machinery over 1536 polynomials — only a
  *specialised* Hadamard sumcheck that folds slices in place (avoiding that
  rebuild) or a smaller polynomial count would move Prove-Hadamard; comb
  micro-opts won't. (Phase-1 round-1 fast path already bypasses the rebuild
  for round 1, which is why it — not this — was the win.)

### Fast u64 scatter in `scatter_matrix_into_gpu_slab` for F2PackU64 cells (rejected)
- **Hypothesis (from optimization survey)**: the per-cell write in
  `zip-plus/src/merkle.rs::scatter_matrix_into_gpu_slab` goes through
  `cell.write_transcription_bytes_exact(dst)`, which for
  `BinaryU64Poly<D>` expands to
  `dst.copy_from_slice(&value.to_le_bytes())`. Replacing it with
  `ptr::write_unaligned::<u64>(dst.cast(), cell.pack_u64().to_le())`
  removes the slice creation and the `[u8;8]` intermediate.
  Estimated impact: 2–5% Commit wall time (revised down from the
  audit's 5–15%).
- **Implementation**: added a `FastScatterCell` trait in
  `zip-plus/src/merkle.rs` with a default body using
  `write_transcription_bytes_exact`, plus an override for
  `BinaryU64Poly<D>` using direct unaligned u64 write. Threaded
  through `ZipTypes::Cw` as a bound; default-body impls for the
  other Cw types (`BinaryRefPoly<D>`, `DensePolynomial<R, D>`,
  `i128`, `Int<K>`). Bound propagation also required adding
  `FastScatterCell` to the generic `CwR` bound in
  `protocol/benches/e2e.rs:220-227`.
- **Measurement (criterion `--save-baseline` / `--baseline`,
  nvars=22, `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                              | Before   | After    | Δ      | criterion verdict |
  |------------------------------------|----------|----------|--------|-------------------|
  | Commit-Fused-GPU-Inline/nvars=22   | 584.19 ms | 586.67 ms | +0.42% | no change (p=0.19) |

  Reverted. LLVM was already optimising the
  `to_le_bytes`+`copy_from_slice` pattern down to equivalent
  instructions; the architectural cost (new trait + impls in
  4 files + bound propagation through `ZipTypes::Cw`) wasn't
  justified by zero measured benefit.
- **Lesson**: when the optimization is "remove indirection LLVM
  should already see through," prefer measurement before
  architectural changes. The audit's 5–15% estimate was
  speculative — LLVM's IR-level optimisation of small fixed-size
  byte copies is mature on both x86_64 and aarch64.

### `F2NativeIc::prove_linear` loop inversion + bit-d masks (rejected — wrong approach; superseded by `54a564b`)
- **Status note**: the *target* (`prove_linear`'s XOR-fold) turned
  out to be a real optimization opportunity at the realistic
  workload — see the "F_2 native IC: bounds-check elimination in
  `prove_linear` XOR-fold" entry under Shipped work (`54a564b`,
  −10.1% Prove e2e). The agent's specific *approach* (loop
  inversion + precomputed bit-d masks) is still rejected for the
  reasons below; bounds-check elimination on the existing
  sparse-iter shape gave the win instead.
- **Hypothesis (from optimization survey)**: the per-bit
  `coeffs[d] += &eq_table[row]` accumulation in
  `protocol/src/f2_native_ic.rs:686-705` (and the analogous shifts
  loop `:715-742`) could be rewritten with `d` as the outer loop
  and `i` as the inner, using a precomputed `bit_d_mask: Vec<u64>`
  per column. Claimed impact: 5-10% UAIR/IC.
- **Why the claim doesn't hold up**:
  1. The current code already short-circuits on `bits == 0` via
     `while bits != 0 { ... }` — zero cells contribute nothing.
     For random data (~50% set bits per cell), the inner loop
     runs ~16 times per cell.
  2. The proposed inversion iterates ALL `(d, i)` pairs:
     `32 × usable_rows` per column. Strictly MORE iterations than
     the current sparse pattern (~16 × usable_rows for random
     data; far less when cells are zero).
  3. Per-column bit-d masks are large at scale: at nvars=22
     they'd be `usable_rows / 8 × D = 4M/8 × 32 = 16 MB per col`,
     ~450 MB total across 28 cols. Cache-hostile.
  4. The proposed SIMD vectorisation needs an efficient
     "masked XOR-reduce" primitive; NEON (Apple Silicon) lacks a
     direct one. AVX-512's `vpternlogq` could help on x86 but
     doesn't apply to the target platform.
- **Measurement (criterion at nvars=22, --features parallel,simd,
  unchecked,metal_gpu)**:

  | Micro                          | Time     | % of UAIR-FULL | % of Prove e2e |
  |--------------------------------|----------|----------------|----------------|
  | UAIR-a-F2NativeIC (`prove_linear`) | 24.5 ms | 5.8%           | ~1.1%          |
  | UAIR-FULL                      | 424.5 ms | 100%           | ~19%           |
  | Prove (for context, nvars=22)  | 2.25 s   | —              | 100%           |

  Even a *complete* elimination of `prove_linear` time would only
  yield ~1.1% Prove e2e — far below the 5-10% the survey
  estimated. A realistic rewrite that saves, say, 50% of
  `prove_linear`'s time is ~0.5% e2e: too small to justify the
  added complexity.
- **Adjacent opportunity (not pursued either, same code)**: at
  `nvars=22` only ~476 of 4M rows (= 7 compressions × 68 rows
  per comp) carry non-zero cells; the remaining ~99.99% hit the
  `while bits != 0` short-circuit but still pay loop-control +
  cell load + `bp_to_u64`. Plumbing an "active row hint" through
  the Uair trait so `prove_linear` iterates only `0..active_rows`
  would skip ~117 M empty iterations across all cols. Upper bound
  on savings: still ≤ `prove_linear`'s 24 ms, so ≤ 1% Prove e2e.
  Tracked here but not implemented in this session — the payoff
  is small and the abstraction change touches `Uair`/`UairSignature`.

### Metal Blake3 dispatch: lifting the `min(256)` threadgroup cap (rejected on artificial trace — re-shipped on realistic shape; see `e4d6a25` under Shipped work)
- **Hypothesis**: the hard-coded `min(256)` cap on
  `threads_per_threadgroup` in
  `zip-plus/src/metal_gpu/mod.rs:201-214` was under-occupying Apple
  Silicon GPUs at SHA-256 F_2 nvars≥16 where `num_cols` is in the
  10⁴–10⁵ range. Lifting the cap so the pipeline's own
  `max_total_threads_per_threadgroup()` wins would (per the
  optimization-survey agent's estimate) give 10–30% Blake3 throughput
  improvement.
- **Test**: removed the `.min(256)` clamp; the dispatch now uses
  `(max_threads / exec_width) * exec_width` directly. Both
  `gpu_blake3_matches_cpu_*` equivalence tests and 13/13 F_2 lib
  tests still passed.
- **Measurement (A/B with criterion `--save-baseline` /
  `--baseline`,** target `Zinc+ F_2 SHA-256/Prove/nvars=(16|20|22)`,
  10 samples + warmup, `--features parallel,simd,unchecked,metal_gpu`):

  | nvars | Before (cap=256) | After (cap removed) | Δ      | criterion verdict |
  |-------|------------------|---------------------|--------|-------------------|
  | 16    | 30.62 ms         | 33.12 ms            | +9.34% | regressed         |
  | 20    | 437.7 ms         | 434.3 ms            | −0.76% | within noise      |
  | 22    | 2.251 s          | 2.245 s             | −0.24% | no change         |

  Net: clear regression at the smallest GPU-active nvars, neutral
  at larger nvars.
- **Why it didn't help**: the assumption that the cap was the
  binding constraint was wrong. At nvars=16 the commit produces only
  ~4096 leaves (= 4096 work-items). With threadgroup_size = max
  (1024 on M-series), that's 4 threadgroups total — too few to
  occupy all GPU cores. The original 256-thread cap produced 16
  threadgroups at nvars=16, balancing per-threadgroup occupancy
  against having enough threadgroups for the dispatcher to spread
  across cores. At nvars=20+ both threadgroup sizes produce enough
  threadgroups to saturate the GPU, so the cap is invisible. The
  256 was empirically well-tuned.
- **Lesson**: don't lift kernel-dispatch caps without measuring at
  the smallest expected workload size — Apple's
  `max_total_threads_per_threadgroup` reports a *per-threadgroup
  ceiling*, not the optimal threadgroup size for a given total work
  count. The two interact via cores-per-GPU.
- **If revisited**: a size-dependent rule could potentially do
  better than either fixed choice — e.g. clamp so
  `num_threadgroups ≥ 2 × core_count`, then use the largest
  threadgroup that satisfies that. Not pursued in this session.

### GPU warm-up dispatch (rejected)
- **Hypothesis**: the 320 ms gap between `Commit` (isolated, 577 ms)
  and `Commit-AfterPrevProve` (900 ms) was Metal dispatch
  cold-start after ~1.4 s of CPU-only UAIR + Open between commits.
- **Test**: added a tiny 32-byte no-op Blake3 dispatch at the
  start of `commit_pre_paired_witness` to wake the GPU before
  the real dispatch.
- **Result**: no change. `Commit-AfterPrevProve` 772 → 782 ms,
  `Commit` 557 → 563 ms — both moved by the no-op's ~5–10 ms cost
  with zero compensating speedup. **GPU dispatch warmth is NOT
  the bottleneck.**
- **Real culprit**: most likely the 768 MB slab allocation inside
  `commit_grouped` (`vec![0u8; num_leaves × leaf_bytes]` =
  `262144 × 3072` at nvars=22). Cold allocator + cold-cache
  encode work. See open item below.

---

## Identified but not implemented

### Commit is the ONLY phase Binius64 beats us on — and it's Merkle-bound, not encode-bound (investigated 2026-06-13)
- **Question**: "Zinc+ should have a faster prover than Binius64 — why doesn't it
  look that way at nv16?" Investigated with fresh per-phase measurement.
- **Measured warm per-phase** (`sha256_f2_merged_ab_timing` + `OBLONG_PROFILE=1`,
  AB_NVARS=16, release, **no metal**, Apple M-series, merged arm E ≈ 87.9 ms total):
  commit **19.4 (22%)**; uair *self* = ψ_α-projection + main sumcheck **19.4 (22%)**;
  discharge (merged_phase1 12.5 + group 5.0 + end 1.4 + w_block 1.3) **≈20 (23%)**;
  multipoint_eval **12.1 (14%)**; uair:ic (linear) **6.0 (7%)**; open+open_evals
  **10.2 (11%)**. **This warm total (87.9 ms) already ≈ Binius64's 87.6 ms (CPU).**
  vs b64 per-phase (Commit 7.1 / IntMul 1.0 / BitAnd 24.9 / Shift 43.3 / PCS-open
  10.3): **Zinc+ wins the discharge ~3.4×** (20 vs b64 BitAnd+Shift 68) **and the
  open**; **loses ONLY commit** (19.4 vs 7.1). Commit is the entire gap.
- **Commit decomposition** (`Micro/Commit*`, nv16, no metal): Pair 0.42 ms ·
  **Encode (RAA rate-1/4 expansion) 1.20 ms** · EncodeTranspose 10.2 · **Merkle-only
  17.4** · Commit-Fused 20.5 · Commit 24.7 · Commit-AfterPrevProve (cold) 27.0.
  **⇒ commit is ~90% Blake3 Merkle; the linear-time RAA encode is negligible (1.2
  ms).** The rate-1/4 codeword does NOT cost via encode — it costs via Merkle
  (codeword_len = row_len·REP ⇒ REP=4 hashes 4× the leaves of an uncoded commit,
  2× that of b64's rate-1/2). The **cold-cache penalty is small in the no-metal
  path (~2.3 ms, 24.7→27.0)** — NOT the big lever (corrects the earlier hunch).
- **Why criterion shows ~110 ms (no metal) / ~102 (metal) not 88**: per-iteration
  large-proof cache pressure (×32 combined_row + 987 openings built+dropped each
  iter) inflates ALL phases ~20%, plus thermal — distributed, not localized to
  commit. **Use warm same-run A/B for phase truth, not criterion absolutes.**
- **Levers to make the prover decisively win (ranked by commit impact)**:
  1. ⭐ **RAA rate 1/4 → 1/2** (`RaaF2Code` REP 4→2): codeword halves ⇒ Merkle
     leaves halve ⇒ commit ~19→~11 ms (~8 ms). Biggest single lever. **Blocked on
     proximity recalibration** — `recommended_num_column_openings` only calibrates
     REP=4 (`raa_f2.rs:72`); REP=2 needs the first-moment distance analysis (the
     `rma` crate / `calibrate.py`). Costs proof size + verify (more openings t),
     but b64 is ALSO rate 1/2 — so this is the apples-to-apples commitment-rate
     match, after which Zinc+'s linear encoder + base-field discharge win the prover.
  2. **Virtualize Σ0/Σ1/σ0/σ1** (cols 11–14, committed today; each is an
     XOR-of-3-rotations of committed sources W_A/W_E/W_W per C1–C4): 4 of ~20
     committed cols ⇒ ~20% fewer Merkle leaves+encode ⇒ ~3–4 ms off commit. Needs
     the virtual-spec mechanism extended from single-`BitOp` to XOR-of-BitOps
     (verifier reconstructs the eval as a sum of pointed-rotation evals). The
     `sha_f2_bit_op_virtuals()` comment (`f2_sha256.rs:~149`) already flags this.
  3. **metal_gpu Merkle (deployed)**: the ~17 ms CPU Merkle is exactly what the GPU
     leaf-hash offloads (warm GPU commit ~15.5 ms, scales with N) — so the no-metal
     numbers UNDERSTATE the deployed commit. With GPU + warm, Zinc+ already < b64.
- **Bottom line**: the prior is right. **Warm + GPU, Zinc+ already has the faster
  prover; CPU-to-CPU it's a tie that the commit rate (1/4 vs 1/2) explains entirely.**
  Closing it is a rate recalibration + Σ/σ virtualization, both localized to the
  commit, neither touching the discharge/PIOP (which already win). Crossover with
  b64 widens in Zinc+'s favour with N (base-field discharge + linear encode).
- **RATE-MATCHED head-to-head (measured 2026-06-13): at equal commitment rate
  1/4, Zinc+ WINS the prover.** Rather than recalibrate Zinc+ down to rate 1/2,
  raised Binius64 UP to rate 1/4 (`--log-inv-rate 2`; `calculate_n_test_queries`
  auto-adjusts queries so 96-bit holds) — the apples-to-apples commitment-rate
  comparison. b64 sha256 example, nv16-matched (`--max-len-bytes 61632
  --exact-len`), Apple M-series CPU, median of 3 (this clone has the parallel-
  verifier patch, so its verify is ~14–15 ms not the 25.9 ms serial):
  - **b64 rate 1/2 (its default)**: prove **95.5 ms**, verify 14.0 ms, proof 304 KiB.
  - **b64 rate 1/4**: prove **103.7 ms** (+8.6%), verify 15.5 ms, proof **249 KiB**
    (−18%). Raising the rate adds ~8 ms to b64's prover (2× codeword ⇒ more FRI
    commit+fold) and SHRINKS its proof (fewer queries). The AND/shift sumchecks are
    rate-independent, so the prover rises only via commit/FRI.
  - **Zinc+ rate 1/4 (its native rate), DEPLOYED t=987, warm**: prove **~93 ms**
    CPU (merged 92.8 / oblong 93.5; ~83 with metal_gpu), verify ~12 ms (criterion
    median; consistent with the note's ~11 ms), proof ~1.1 MiB.
  - **CORRECTION (supersedes a first pass that said 87.9 ms / 1.18×)**: that used
    the ab_timing default of **t=4** column openings, not the deployed 987.
    Verified the prover is **nearly t-INSENSITIVE**: t=4→987 adds only ~4 ms to
    prove (89.0→92.8 merged; the open's `combined_row`/`b_vector` dominate, path
    gathering is cheap), whereas verify IS t-bound (~4→~20 ms single-run). So the
    warm number was close but the precise deployed figure is ~93 ms, not 88.
    Added an `AB_OPENINGS` env knob to `sha256_f2_merged_ab_timing` (default 4;
    use `AB_OPENINGS=987`). The skill's "verify 4.46 ms / 5.8×" is the t=4-class
    number; at deployed t=987 verify is ~12 ms.
  - **⇒ matched rate 1/4: Zinc+ prover ~93 vs b64 103.7 ms = Zinc+ ~1.12× faster
    (CPU), ~1.25× with metal_gpu. And Zinc+ at 1/4 (~93) even edges b64 at its
    native rate 1/2 (~95.5).** Verify comparable at this scale (~12 vs ~15 ms);
    proof b64 4–5× smaller. The
    rate-1/2 "prover tie" was b64 running at a cheaper rate; equalize the rate and
    Zinc+'s linear encoder + base-field discharge win the prover outright. Note the
    rate dial is a tradeoff: matching it to 1/4 also GROWS b64's proof-size edge
    (304→249 KiB), so it strengthens BOTH Zinc+'s prover story and b64's size story.
  - Repro: `RUSTFLAGS=-Ctarget-cpu=native cargo run --release --example sha256 --
    prove --max-len-bytes 61632 --exact-len --log-inv-rate {1,2}` (binius64 clone;
    build needs sandbox off — writes its own target/).

### Adopt "Blaze" (from `~/bolt-rs`) as the F_2[X] PCS instead of Zip+ over F_2[X] (investigated 2026-06-14 — REJECTED, no win + breaks the dual read-off)
- **Question**: `~/bolt-rs` (impl of *Bolt: Faster SNARKs from Sketched Codes*,
  ePrint 2026/310) has fast binary-field code-commitment machinery. Could we drop
  in "Blaze" (the interleaved-RAA Brakedown PCS, ePrint 2024/1609) in place of our
  bit-sliced Zip+ over `F_2[X]`?
- **What's actually in the folder**: `src/raa.rs` is a Blaze-style RAA encoder
  (repeat → π₁ → prefix-XOR → π₂ → prefix-XOR) but over **GF(2³²)**
  (`ColMajorMatrix<BinaryElem32>`), `src/sketch_code.rs` is Bolt's expander+RS
  "sketched" code, and the only *complete* PCS open/verify is **Ligerito**
  (recursive Basefold-style, `src/ligerito_recursive.rs`). There is **no complete
  Blaze prover/verifier** — "Blaze" appears only as a proof-size *estimate* comment
  (`src/bin/code_queries.rs:408`). The RAA path is an encode + Merkle-commit
  *benchmark*, tuned for **2³⁰** polys (Metal GPU + SIMD + CLMUL + HW-SHA).
- **Why it's not a replacement (structural)**: Zip+ over `F_2[X]` *already is* a
  Blaze-style RAA-Brakedown commitment — `RaaF2Code` (`zip-plus/src/code/raa_f2.rs`)
  is precisely the Blaze RAA construction specialized to an **`F_2` XOR accumulator
  with no codeword widening** (cells stay `BinaryPoly<D>`). The load-bearing part of
  our PCS is **not the code** — it's the **bit-sliced opening that binds one object
  `a'` and reads off BOTH `ψ_α` (linear) and `ψ_z` (Hadamard discharge) with no 2nd
  opening** (the un-lifted-`GF128[X]<D>` open; see Shipped entry `7827683` etc.).
  That dual read-off *depends* on the `F_2`-linearity / no-widening property. Blaze
  and Ligerito give a single-evaluation opening over a widened field (GF(2³²)/RS) —
  adopting either means re-engineering the exact bit-slice + dual-projection layer on
  top of it, i.e. rebuilding Zip+ over `F_2[X]`. Widening the cells to GF(2³²) to
  match bolt-rs's representation would *destroy* the no-widening property the AND/adder
  discharge rides on.
- **Why it wouldn't even help (performance)**: bolt-rs's whole value is a fast
  *encoder* at 2³⁰. But our commit is **~90% Blake3 Merkle, encode is ~1.2 ms
  (negligible)** — see the "Commit is the ONLY phase Binius64 beats us on" entry
  above. We already have a Metal GPU `F_2`-RAA encoder
  (`zip-plus/src/metal_gpu/shaders/raa_f2_encode.metal`) and Metal Blake3 Merkle. At
  SHA scale (ν=9..16) the bottleneck was the discharge, not commit; bolt-rs's 2³⁰
  regime is irrelevant. At the bit level bolt-rs's GF(2³²) RAA XOR is *identical* to
  our `BinaryPoly<32>` XOR, so the kernel is transplantable in principle — but it
  buys nothing the existing encoder doesn't, and the encode isn't the cost.
- **The one genuinely interesting harvest (different question)**: bolt-rs's
  **Ligerito** (recursive Basefold) is in the WHIR family the note already flags as
  the way to shrink our Brakedown-style proof (t=987 openings, ~1.1 MiB). That's a
  *proof-size* lever, not a "replace the PCS" one, and it still wouldn't subsume the
  `ψ_α`/`ψ_z` dual read-off — it would sit *under* it. Tracked as future work; see
  the implementation.tex outlook ("WHIR-style proximity test").
- **Proof-size estimate at nv16 (the follow-up question 2026-06-14)**: modelled
  Blaze's proof (RAA level-0 proximity + inner Basefold tail) for the SHA-256 nv16
  witness (26 cols × 2¹⁶ × 32b ≈ 2²⁰·⁷ GF(2³²) elems) using bolt-rs's exact
  `proof_size_{brakedown,ligerito}` formulas + our calibrated q=987 (RAA rate 1/4).
  **Model validated**: the Brakedown/our-family path reproduces the deployed ~1.1 MiB
  at n_rows≈2⁸ (1.18 MiB computed). **Blaze ≈ 0.3–0.5 MiB** (best tall-skinny shape
  n_rows=2⁴: ~495 KB = 62 KB L0 openings + **241 KB L0 Merkle paths** + ~193 KB
  Basefold tail). **Only ~2–3× smaller than our 1.1 MiB, NOT the ~5× you'd hope.**
  Root cause: Blaze shares our RAA code ⇒ inherits q≈987, and 987-query Merkle
  authentication (~240 KB at depth ~19) is a floor recursion can't remove. The real
  proof-size lever is the **code's distance/query-count, not the opening structure**:
  an RS/FRI/Basefold inner code uses q≈241 (1−δ/2 at δ=0.5) and lands ~200–250 KB —
  e.g. Ligerito@2²⁰ computes to **~201 KB**, and Binius64 measures **~249 KiB** on the
  same SHA workload. ⇒ for proof size, the WHIR/Basefold-over-RS direction
  (implementation.tex outlook) beats Blaze; Blaze's RAA only matches our prover-speed
  story, not a size win. (Estimate, not Blaze's paper figure — eprint behind Cloudflare;
  the ~240 KB RAA-Merkle floor is the robust part, independent of tail modelling.)
- **Verdict**: no — Zip+ over `F_2[X]` already is the Blaze code plus the
  dual-projection opening the SHA PIOP needs, and bolt-rs optimizes a phase (encode)
  and a scale (2³⁰) that aren't our bottleneck. Even on proof size (Blaze's one
  plausible edge) it's only ~2–3× and floored by the shared RAA query count. Not pursued.
- **UPDATE (2026-06-14): the code-switching *idea* IS being pursued, via our own
  Zip++ basefold — not Blaze-as-drop-in (none exists; see search below).** User
  confirmed proof size is a priority. Searched for a Blaze impl: no public standalone
  one exists (GitHub empty; bcc-research = the bolt-rs authors, no `blaze` repo; bolt-rs
  ships RAA + Ligerito as *separate* benchmarks, never the code-switching composition).
  **Correction to the estimate above:** ~0.3–0.5 MiB modelled a Brakedown-with-RAA
  opening (987 direct column openings); Blaze's actual code-switching has an O(log²n)
  verifier, so it does *not* open 987 columns — the Ligerito **~150–250 KB** is the
  right proxy (4–7× shrink, not 2–3×).

### Code-switching opening (Blaze-style) via the Zip++ binary basefold lane — proof-size plan (scoped 2026-06-14, NOT yet implemented)
- **Goal**: cut the SHA-256 proof from ~1.1 MiB to **~150–250 KB** + give an O(log²n)
  verifier, by replacing the Brakedown-style `prove_f2_open` (combined_row' ~5500×7
  words + t=987 column openings, both O(√N)) with a code-switch onto a binary
  Basefold/WHIR inner IOPP. **Prover stays unchanged** (commit keeps `RaaF2Code`); this
  is a size/verifier win only.
- **Vehicle (user-chosen)**: reuse the Zip++ basefold infra
  (`.claude/worktrees/zip-plus-basefold`, module `zip-plus/src/basefold/`), NOT a from-
  scratch Blaze port. **Key simplification**: Zip++'s limb machinery (balanced base-2^p
  decomposition, carry rebalance, mod-q0 claim projection) exists only because folding
  over ℤ widens entries; **over GF(2^128) folding doesn't widen** ⇒ the binary lane is
  the *classic* arity-2 fold, k=1, claims in K — strictly simpler than the integer lane.
- **Phases**: (1) binary foldable chain over `GF(2^128)[X]` from our additive-NTT /
  subspace-Lagrange primitives (`oblong_and`), replacing `chain.rs`'s prime
  `ChainConfigF167772161`; strip limbs in `iopp.rs`; reuse `compiled.rs`
  (Merkle/FS/Q-spot-checks/`size_bytes`) verbatim; checkpoint = standalone binary IOPP
  proving one MLE eval over K, confirms the KB target. (2) code-switch RAA→IPRS-chain in
  `prove_f2_open`/`verify_f2_open`. (3) **dual ψ_α/ψ_z read-off through the fold — the
  load-bearing design**: `a'=Σ eq_x(r_0)w(x)` is F_2-linear, both functionals are inner
  products against the IOPP's folded `a'`; re-derive the 4 opening checks recursively;
  paper-validate BEFORE Phase 2. Fallback: run the IOPP twice (per functional) — a 2nd
  opening, loses "AND for free", still shrinks the proof. (4) soundness: fold proximity
  + per-round IOPP error into FS/BCS; ψ_z-binding gap stays orthogonal. (5) measure +
  update `implementation.tex` (its "WHIR-style proximity" outlook).
- **Risks**: Phase 3 (does the dual read-off survive code-switching) is the real unknown;
  Phase 1 binary additive-NTT chain is real but buildable from existing primitives.
  Logistics: `zip-plus-basefold` (off `main-beta`) and `f2-clean` must converge in one
  worktree before Phase 2.
- **PHASE 3 DONE — CLEARS (paper-validation, 2026-06-14): `documentation/f2-codeswitch-dual-readoff-design.md`.**
  The read-off survives. Reason: `a'` (the bit-slice fold, `lifted_claim`, a *single
  un-collapsed* `K[X]_{<w}` poly — commitment.tex Rem. lines 73-84, f2_prove.rs:3198)
  and both projections are checks (2)+`ψ_z`-binding = **local linear functionals of
  `a'`, decoupled from proximity** (checks 1,3,4). Basefold replaces only (1),(3),(4);
  `ψ_α`/`ψ_z` are applied to the folded object *after*, unchanged. The fold axis (ν
  rows, eq(r_0)) is orthogonal to the width axis (w=32 cell bits / X); eq-fold, the
  code, and every Basefold round op are K-linear *coordinate-wise on X*, so the final
  claim emerges as the full width-`w` `a'`. **Single invariant**: X is passive shared
  width — never folded/batched/projected inside the recursion; fold challenges,
  positions, twiddles are X-independent K-scalars. Enforced at 4 type-level points (see
  note §5); the natural impl satisfies it automatically. **Phase-1 API obligation**:
  oracle/claim type = `K[X]_{<w}` (width-w), NOT scalar K (the one substantive
  generalization of the Zip++ lane; *limbs drop out* — no widening over K). **Design
  refinement (note §6)**: Phase-3 verdict is independent of code-switch-vs-direct — so
  prefer **(3b) commit directly with the foldable IPRS chain (no code-switch)** over
  (3a) RAA+code-switch, since encode is negligible (~1.2 ms; commit is ~90% Merkle) and
  3b removes a soundness term + a collapse risk. Decide 3a/3b at the Phase-1 checkpoint
  by measuring foldable-chain commit cost. **Fallback** if a future variant breaks the
  invariant: run the IOPP twice (per functional) — still < 1.1 MiB, clean degradation.
  Soundness (Phase 4): `ε_open ≤ ε_prox^{WHIR} + O(1/|K|) (+ε_switch under 3a)`;
  `ψ_z`-operand binding gap stays orthogonal.
- **PHASE 1 — sourcing decided + foldable-chain substrate SHIPPED & TESTED (2026-06-14).**
  *Sourcing question (use binius64/bolt-rs RS?)*: **No — build on our in-house
  `binary_subspace` additive-NTT over our own `BinaryFieldGF128`.** Three reasons:
  (i) **field nativeness** — the IOPP claims/`a'`/`ψ_α`/`ψ_z`/Blake3 transcript all
  live in our K; bolt-rs's `BinaryElem128` and binius64's tower-GF(2^128) are *different
  representations*, forcing an F_2-iso bridge at every claim + the FS boundary; (ii) the
  **Zip++ integer chain does not port** — its butterfly `even ± tw·odd` collapses in
  char 2 (`+=−`); the correct binary butterfly is the *additive* NTT (Lin–Chung–Han),
  which `binary_subspace` (+ `oblong_and`) already realises over K; (iii) the **ledger
  says encode isn't the bottleneck** (~1.2 ms; commit ~90% Merkle), so bolt's SIMD
  encode (its fast path is BinaryElem32, N/A to GF128) / binius's packed NTT buy nothing
  at Phase-1 (correctness-first). Revisit binius64's NTT *only* as an optimization
  reference later if encode ever measures hot at large ν (it won't for SHA shapes).
  *Shipped*: `poly/src/univariate/binary_foldable_chain.rs` — `rs_encode` (subspace
  eval) + `fold_codeword` (the additive-NTT FRI fold `c'[i] = c[i] + (y_i+α)·(c[i]+
  c[i+half])/b`, folded domain = `span{ŝ_b(b_j)}`, `ŝ_b(x)=x(x+b)`). **3 tests pass
  under `--features simd`**: `fold_maps_rs_to_smaller_rs` (non-circular: one fold maps
  RS[k,K]→RS[k-1,K-1], checked across 5 (k,r) settings), `fold_of_constant_is_constant`,
  `full_descent_reaches_constant`. Fold formula derived from first principles
  (`P(x)=E(ŝ_b(x))+x·O(ŝ_b(x))`) and **empirically validated**. NOTE: `binary_gf192`
  (deprecated) fails to compile under default features (pre-existing, `BinaryPoly` alias
  is `BinaryRefPoly` vs `BinaryU64Poly`); the poly crate's tests run under `simd`.
  *Next (Phase 1b/c)*: width-`w` lift (oracle/claim type `[K; w]` per §5 invariant);
  multilinear claim-tracking rounds `(1−z_r)e^e + z_r e^o = e_{r-1}` over `K[X]_{<w}`;
  Merkle/FS/Q-spot-check compile (port `compiled.rs`) + `size_bytes` checkpoint to
  confirm the ~150–250 KB target and decide 3a vs 3b.

### The linear PIOP is the under-optimized third of the prover (identified 2026-06-13)
- **Context**: per-phase at nv16 warm CPU (~93 ms): discharge ~20 ms (heavily
  optimized — fused→oblong→GF8→merged, parallel, eq-trick), commit ~19 ms
  (GPU+fused+slab; Merkle-bound, see the commit entry above), **linear PIOP ~31 ms
  = ψ_α projection + main GF(2^128) `MultiDegreeSumcheck` (~19 incl. ic) +
  multipoint-eval (~12)**, open ~7 ms. The discharge and commit have been squeezed;
  the linear PIOP never received the discharge's two big wins. Impact figures below
  are HYPOTHESES from an Explore pass — MEASURE before building.
- **(1) Main sumcheck rounds 2..n have no eq-factoring.** ✅ **DONE
  (2026-06-14, working tree)** — `F2EqColRoundEvaluator` (`f2_prove.rs`), a
  `RoundPolyEvaluator` mirroring the round-1 fast path's eq-factoring into
  rounds 2..n, byte-identical, `uair:sumcheck` ~7.4 → ~6.0 ms at the merged
  arm (−1.4 ms / −19% of the scope; end-to-end neutral within noise — group-0
  is the small piece). `F2_NO_EQCOL_EV=1` toggles it off. See the Shipped-work
  entry. *Original note:* only round 1 had the `F2EqColRound1FastPath`; rounds
  2..n fell back to the generic fold (`piop/src/sumcheck/prover.rs`); the
  analogous round-2..n specialization had shipped for the *Hadamard* discharge
  (−24.5%, commits `9505876`+`7a27606`) but not the LINEAR sumcheck.
- **(2) multipoint-eval is a whole SECOND GF(2^128) sumcheck (~12 ms)** just to fold
  the ψ_α + pointed-shift (+ψ_z) claims to one point r_0. Possibly fusable into the
  main sumcheck, or collapsible (cf. the open question "Multipoint-eval as
  degenerate γ-rerandomization, down_evals=[]"). Impact: potentially large (up to
  ~12 ms); effort: med-high (protocol-visible). The single highest-value unexplored
  block — profile its internals first.
- **(3) GPU α-projection is used for only the main UAIR projection**, not the
  merged-Hadamard W-block / unreferenced-lag projection sites (CPU-bound). Reuse
  `project_columns_with_powers_gpu_batched`. Impact: small-med; effort: med.
- **(4) Small-field (GF(2^8)) accumulation à la the discharge is likely NOT
  portable to the linear path** — ψ_α is a genuine monomial eval at α∈GF(2^128);
  the discharge's GF(2^8) lane works only because ψ_z uses subspace-Lagrange
  weights. Recorded as "tempting but blocked by the ψ_α monomial structure" so it
  isn't re-attempted the wrong way.
- **Recommendation**: start at the multipoint-eval (is it truly a second full
  sumcheck? can it fuse with the main one / go degenerate?), then A/B eq-factoring
  the main sumcheck rounds 2..n. Commit-side levers (rate 1/2, Σ/σ virtualization
  via the multi-source-XOR `F2BitOpVirtualSpec` generalization) remain the answer to
  the Binius64 commit gap but are already catalogued.
- **UPDATE 2026-06-13 — measured the linear-PIOP sub-split (nv16, t=987, warm),
  which redirects the priorities:**
  - **uair:sumcheck (main `MultiDegreeSumcheck`) = ~11 ms** — the single largest
    linear-PIOP block, 2.4× the multipoint sumcheck (~4.5 ms). Its group-0 is a
    PURE `eq·weighted_col` zerocheck with only ROUND 1 eq-factored
    (`F2EqColRound1FastPath`); rounds 2..n use the generic fold. **This is the real
    eq-factoring prize** — extend a Gruen eq-factored degree-2 prover to rounds
    2..n for group-0 (the discharge already runs exactly this via its
    `QuadraticMleCheckProver`/`oblong_and.rs`; port the technique to the linear
    eq·col group). Medium-high effort, soundness-sensitive (add a binius-style
    first-round-vs-next-round cross-check test). ✅ **DONE 2026-06-14**
    (`F2EqColRoundEvaluator` + `f2_eq_col_round_evaluator_matches_generic`
    cross-check); the ~11 ms quote was a stale pre-merged-evaluator figure — at
    the deployed merged arm the realised scope is `uair:sumcheck` ~7.4 → ~6.0 ms
    (the group-0 slice is small once group-1 is fused). See Shipped-work.
  - **uair:alpha_project = only ~2.6 ms** ⇒ lever (3) "GPU projection for all
    sites" is NOT worth it. DROP it.
  - **multipoint eq-factoring is the WRONG lever-#1 target**: its comb is mixed
    (`eq_r·precombined` + Σ `next_k·down_k`, all selector·data) so eq-factoring it
    is a binius-style shift-eval-sumcheck reimplementation for only ~4.5 ms, vs the
    main sumcheck's clean pure-eq ~11 ms. The cheap multipoint win (parallelize
    next_mle, ~3.7 ms) is SHIPPED (see Shipped-work); further multipoint gains are
    low-ROI relative to the main sumcheck.
  - **Revised order**: ~~(1) eq-factor the main sumcheck rounds 2..n~~ ✅ DONE
    2026-06-14 (−1.4 ms `uair:sumcheck`); (2) Σ/σ virtualization (commit ↓ but
    watch the multipoint pointed-shift growth, now cheaper per-shift after the
    next_mle win); (3) commit rate 1/2. **Now (2)/(3) are the live levers; the
    linear-sumcheck eq-factoring is fully spent.**
- **UPDATE 2026-06-14 — cross-branch check: is there a portable eq-factor win on
  `f2-clean-lookup`?** Audited every perf commit on `f2-clean-lookup` (the sound
  lookup-adder branch) against `f2-clean`. The *only* shared-infra artifact unique
  to the lookup branch is `piop/src/sumcheck/eq_factored.rs` — a generalized Gruen
  eq-factored driver for the degree-3 shape `Σ_t eq(·;q_t)·Σ_i L_{t,i}·R_{t,i}`
  (suffix-tensor no-fold + the char-2 affine-flat free 4th node). It is a clean,
  conflict-free, *additive* module (the other `sumcheck/*.rs` blobs are
  byte-identical across the two branches), BUT (a) it is referenced only by
  lookup-only callers (`gkr_product.rs`, `f2_lookup_binding.rs`) that don't exist
  on `f2-clean`, and (b) its technique is already in this tree three times over —
  `oblong_and.rs::QuadraticMleCheckProver` (degree-2, Gruen, no eq fold, char-2
  cheap fold `out[i]=tbl[2i]+tbl[2i+1]`), the fused discharge round-≥2 evaluator
  (commits `9505876`+`7a27606`, −24.5%), and the working-tree
  `MergedHadamardRoundEvaluator` (Karatsuba 3-mul coefficient form, eq second
  pass). **Conclusion: nothing to cherry-pick for a free speed win.** Lever (1)
  above (main sumcheck rounds 2..n) stands as the real prize, and its port source
  remains f2-clean's OWN degree-2 `oblong_and.rs` — `eq_factored.rs` is the wrong
  degree (3, eq·L·R) and bakes in the GKR/binding group shape, so it is at most a
  secondary reference, not a drop-in. (The lookup branch's headline *feature* — the
  sound adder — is a soundness upgrade, not an optimization, and is *slower*: ~179
  ms sound vs the trusted-adder merged path; see the f2-lookup-adder ledger.)

### Small-value / multi-round-skip prover for the Hadamard zerocheck (BUILT in full, A/B-DISCONFIRMED at 40–74× SLOWER, then REMOVED from the tree — record kept so the approach isn't re-attempted the same way; see the A/B RESULT + REMOVED bullet)
- **Source**: Dao–DeStefano–Bagad–Domb–Thaler, "Speeding Up Sum-Check
  Proving" (cs.nyu.edu/~zd2131/papers/26-587.pdf, Mar 2026), §3 (tower-field
  small-value arithmetic), §5 (small-value prover + cost), §6 (eq-poly
  optimisation). Same regime Binius64's AND reduction uses (univariate skip,
  binius.xyz/blueprint/backend/ands). §2.1 explicitly names GF(2^128)/GF(2)
  (Binius) as a target.
- **Idea**: our Phase-1 round-1 fast path is the `v=1` case of the
  small-value prover — round 1 runs on packed 0/1 bits, then `fold_with_r1`
  materialises half-size F-slices and the generic sumcheck takes rounds
  2..n. Extending to `v>1` keeps the first `v` rounds on packed bits and
  defers materialisation, which (a) shrinks the peak F-slice memory
  geometrically (`1536 · 2^(μ-v) · 16 B`: nvars=20 v=1→12 GB, v=2→6 GB,
  v=5→0.75 GB — i.e. it's what actually clears the nvars≥20 swap), and
  (b) moves `v` rounds off the memory-bound generic value-array rebuild.
- **Mechanism (Dao §5.1)**: don't bind `X_1..X_v` one round at a time
  (each binding makes the slices large via the random challenge). Instead
  compute the `v`-variate **prefix polynomial**
  `q(X_1..X_v) = Σ_{x'} comb(X_1..X_v, x')` over the grid `{0,1,X,X+1}^v`
  (the GF(2^128) boundary points `F::from_with_cfg(0,1,2,3)`) from the 0/1
  base evals, then read round `i`'s message off `q` and interpolate
  `q(r_1..r_{i-1}, X_i, ·)` for the bound prefix. After `v` rounds, fold to
  the `2^(μ-v)`-size F-slices.
- **Cost / payoff**: speedup `Θ((d²κ)^{1/δ})`, `δ=log₂(d+1)=2` for our
  `d=3`, so ≈ `3√κ` where `κ = cost(bb)/cost(ss)`. Their measured 10.9×
  (Spartan) is over **256-bit prime fields** (`bb`≈40-100 cyc). We're over
  **GF(2^128)** where `bb` (CLMUL) is ~5-15 cyc → `κ` is small → realistic
  **~2-5×**, mostly a **memory** win at nvars≥19. The §6 eq-optimisation
  (we have the `eq_r` factor) stacks on top. A *naïve* prefix does ~1.33×
  the arithmetic of generic rounds 1+2, so it **regresses small nvars** —
  must be **size-gated** (`v=1` for production nvars=9; `v>1` only when the
  memory benefit dominates). The efficient multiproduct (Dao Procedure 1,
  `O(d log d)` vs `O(d²)` bb) is what keeps `v>1` from regressing; needed
  for a real win.
- **Framework groundwork — BUILT then REMOVED (reverted to `af0cdf2`)**: the `Round1FastPath` hook
  (`piop/src/sumcheck/multi_degree.rs`) is generalised from one round to
  **`num_skip()` rounds**: `round_message(round, prior)` for rounds
  `1..=num_skip`, then a single `fold(challenges)`; the prover loop drives
  the leading rounds then resumes the standard path at `num_skip+1`. It is
  **byte-identical for `num_skip ∈ {0,1}`** (all existing fast paths +
  generic groups), gated by the full piop+protocol suites (59+50 green). A
  `MultiDegreeSumcheckProof::round_tails(group)` accessor was added so a
  multi-round fast path can be validated round-by-round. Verifier/proof are
  **unchanged** (same protocol) so `fast_path_matches_generic` extends to
  gate the optimised version. (NB this is the verifier-preserving
  small-value variant, NOT Binius64's univariate skip, which changes the
  proof/degree.)
- **Math validated — ✅ (commit `eb3138a`, code since REMOVED)**: `prefix_v2_matches_generic`
  computes the `v=2` prefix `q(X_1,X_2) = Σ_{x''} comb` over the
  `{0,1,X,X+1}^2` grid, derives the round-1 message (`Σ_{X_2∈{0,1}} q`) and
  round-2 message (Lagrange-interp `q`'s X_1-column to `r_1`), and asserts
  both equal a generic 2-round run's `round_tails`. Passes — the small-value
  `v>1` math is correct for our `eq_r·Σσ^b(U·V−W)` comb.
- **(Historical plan — the wiring below WAS built per this, then REMOVED)**: a `num_skip=v`
  Hadamard fast path that (a) precomputes the `v`-dim prefix grid `q`
  (`4^v` field values) in `prepare_hadamard_group_with_fast` from the packed
  operand columns; (b) `round_message(round, prior)`: bind dims `1..round-1`
  to `prior` (Lagrange per dim), keep dim `round` at the 4 grid points, sum
  dims `round+1..v` over `{0,1}` — read off `q`; (c) `fold([r_1..r_v])`:
  multilinear-interp the `2^v` operand corners to `(r_1..r_v)` → the
  `2^(μ-v)`-size slices + `eq_folded`. Subsumes `v=1` (the 1-dim prefix is
  exactly today's `M(0..3)`). Then **size-gate** (`v=1` at production
  nvars=9; raise `v` only where the materialised-slice memory
  `1536·2^(μ-v)·16 B` matters) + A/B at nvars=16/20. Gated by
  `fast_path_matches_generic` (proof stays byte-identical) — so correctness
  is mechanical to verify; the risk is purely in the multidim bind/sum
  bookkeeping. Effort ~150-200 lines.
- **⚠ Perf caveat found while scoping the wiring**: a **naïve** `v`-dim grid
  prefix is ~neutral-to-regression at sizes that fit RAM. Of the `4^v` grid
  points only `2^v` are on the `{0,1}` hypercube (cheap packed-bit / select
  arithmetic); the other `4^v − 2^v` (e.g. 12 of 16 at `v=2`) are
  off-hypercube, where the operand slices interpolate to general
  GF(2^128) values → full `bb` arithmetic. The naïve prefix totals ~1.33×
  the arithmetic of generic rounds 1+2 (at `v=2`), so it only pays off where
  the **memory** reduction dominates (nvars≥~19), and even there `v=2`
  merely halves the materialised slices. **The shippable win needs the
  efficient multiproduct (Dao Procedure 1, `O(d log d)` `bb` via
  extrapolation, + the §6 eq-opt)** so the off-hypercube grid isn't
  recomputed per point — that's the real next algorithmic increment, larger
  than the bind/sum wiring. Recommendation: implement Procedure 1 first (or
  alongside), not the naïve grid, else the wiring lands a correct-but-not-
  faster path.
- **Procedure 1 — BUILT (commit `a388588`), then REMOVED**: `piop::sumcheck::multiproduct`
  (`multi_product_eval` + `multi_extrapolate`) is the efficient evaluation-form
  multiproduct (Dao §4 Procedures 1+2): `d` multilinears over `{0,1}^v` →
  product over `U_d^v` in `O(d log d)` `bb` mults, reusing `NatEvaluatedPoly`
  for the per-axis univariate extrapolation. Validated vs the naïve grid
  product (`v=2 d=3`, `v=3 d=4`, `d=1` identity). This is the engine that
  keeps the off-hypercube prefix grid from being recomputed per point.
- **Prefix wiring — BUILT (commit `a4422fb`), then REMOVED**: `HadamardPrefixFastPath`
  (`num_skip=2`) + `prepare_hadamard_group_with_skip` build the 2-variate
  prefix grid `q` over `U_3^2` and answer `round_message` (round 1 =
  `Σ_{X_2} q`, round 2 = `q(r_1,·)` via `NatEvaluatedPoly`) and `fold`
  (bilinear fold of operand corners + `eq_r` to the `2^(μ-2)`-size MLEs).
  Gated by `fast_path_skip2_matches_generic`: byte-identical proof vs the
  generic path. `prefix q` is built directly (bilinear) for now.
- **Procedure-1 swap — BUILT (commit `f615f7f`), then REMOVED**: `build_prefix_q_v2`
  now computes the per-`(relation,bit,x'')` `U·V−W` terms via
  `multi_product_eval`/`multi_extrapolate` (over `U_2^2`, lifted to `U_3^2`)
  instead of the by-hand 16-point bilinear. Same `q`
  (`fast_path_skip2_matches_generic` byte-identical), so the off-hypercube
  extrapolation isn't redone per grid cell.
- **A/B RESULT — ❌ skip=2 DISCONFIRMED, then REMOVED**: wired a temporary
  size-gate (`effective_hadamard_skip`, default skip=1) + a
  `set_hadamard_skip_override` A/B knob + skip1/skip2 Prove arms in the
  `f2_sha256` Hadamard group (`F2_HAD_ONLY=1 HAD_NVARS=…` to scope the run;
  full-prover byte-identity asserted between skips). **Measured (M-series,
  `parallel simd unchecked`):**

  | nvars | Prove-Hadamard skip=1 | skip=2 | ratio |
  |------:|----------------------:|-------:|------:|
  |  9    | ~9.1 ms               | ~368 ms | **40× slower** |
  | 16    | ~626 ms               | ~46.2 s | **74× slower** |

  The gap *widens* with nvars (no crossover): skip=2's prefix build scales
  **exactly** with `2^(n-2)` (125× from nvars 9→16 ≈ the 128× hypercube
  growth), so it dominates the whole prove. **Root cause:**
  `build_prefix_q_v2` calls `multi_product_eval`/`multi_extrapolate` per
  `(x'', relation, bit)` = `2^(n-2)·16·32` calls/prove, and each call runs
  `prepare_eval_aux` → `batch_invert` (GF128 inversions) + Vec allocations.
  Procedure 1's `O(d log d)` *setup* cost dwarfs the actual `v=2,d≤3` work at
  this per-scalar granularity — i.e. the Procedure-1 swap (`f615f7f`) made the
  prefix build *worse*, not better, vs the by-hand bilinear it replaced. The
  per-bit loop also forgoes the bit-packing skip=1's `HadamardRound1FastPath`
  gets from reading packed `u64` cells. **Decision (user call): REMOVED, not
  kept gated.** Reverted all five skip-2 commits' code to `af0cdf2` (the
  pre-skip-2 state, which already carried the shipped skip=1 round-1 fast path)
  and deleted `multiproduct.rs`; `multi_degree.rs`'s `Round1FastPath` is back to
  its single-round form. Production is **skip=1 only**. Verified after removal:
  59 piop + 9 f2_hadamard tests green, `fast_path_matches_generic` (skip=1)
  still gates byte-identity.
- **If ever revived — DON'T repeat the per-scalar multiproduct**: the only path to a win is
  a **bit-packed, allocation-free, aux-hoisted prefix build** that processes
  all `D` bits per `u64` op (like the skip=1 fast path) instead of looping
  `for b in 0..D` with a Procedure-1 call each. Obstacle: the per-bit `σ^b`
  GF(2^128) weight can't be folded bitwise, so the accumulation must stay
  per-bit in GF128 even when the *operand reads* are bit-packed — exactly the
  trick `HadamardRound1FastPath` already pulls off for round 1, so mirror its
  structure into the 2-variate prefix. Expected ceiling: the prior
  direct-bilinear prefix was only **~neutral** at nvars=9, and the theoretical
  prize is just avoiding round-2 slice-MLE materialisation (`2^(n-2)·1536·16 B`
  ≈ 6.4 GB at nvars=20) — so even a perfect rewrite is speculative. Do NOT
  pursue without a cost model showing the bit-packed prefix beats skip=1's
  already-fast-pathed round 1 + one materialised round 2.
- **Memory measurement (2026-06, dev box) — refines the cost model**: the box
  is **16 GB RAM with ~4.3 GB swap already in use**. skip=1's post-round-1
  materialisation is `1536·2^(n-1)·16 B`: ~0.8 GB at nvars=16 (fits), **~12.9
  GB at nvars=20 (swaps hard on 16 GB)**. So the bit-packed-skip cost model is
  regime-dependent: **(a) nvars ≤ ~18 (fits RAM)** → skip=2 is compute-neutral
  → no win, and at production **nvars=9 it's a compute *regression*** (6 MB of
  slices, the per-prefix overhead dominates — this is why the removed version
  was 40× slower at nvars=9); **(b) nvars ≥ ~19 (skip=1 swaps)** → halving
  (skip=2 → 6.4 GB) / quartering (skip=3 → 3.2 GB) the slices can convert a
  swap-thrashing prove into a RAM-resident one → potentially a **large**
  wall-clock win (swap death, not arithmetic, is the cost there). **Net: a
  bit-packed skip is a large-nvars-only (≥19), size-gated lever — worthless or
  negative for the nvars=9 production default.** Only build it if large-batch
  (nvars≥19) proving on memory-constrained machines is a real target;
  otherwise the production-relevant levers are proof size (`LEAF_GROUP_SIZE`)
  and the nvars=9 compute path.
- **Optional (only if revived)**: generalise `num_skip` past 2 (`v`-dim grid +
  multidim bind/sum) — moot unless the bit-packed prefix above lands first.

### ⭐⭐ ROOT CAUSE of the Binius gap: the discharge is MEMORY-BANDWIDTH-BOUND (1536 GF(2¹²⁸) slices) — the real lever is WORD-LEVEL, not the sumcheck
- **The decisive measurement (2026-06, M4)**: Prove-Hadamard nvars=16 is **1085
  ms single-thread → 464 ms on 10 cores = only 2.34× scaling**. A compute-bound
  job on the M4's ~6 P-equivalents should scale ~6×; 2.34× means the discharge
  is **memory-bandwidth-bound**, leaving most cores idle.
- **Why**: the discharge materialises **1536 GF(2¹²⁸) slices ≈ 1.6 GB**
  (`K·3·D = 16·3·32` bit-slices × 16 B) and streams them every round. Bandwidth,
  not CLMUL, is the cap. This is the `×D=32` bit-split × the 16-byte field.
- **Binius vs us, hardware-normalised**: Binius64 = **111.82 ms proof for 1365
  keccak-f on a 64-core Graviton4** (irreducible.com/posts/announcing-binius64
  benchmarks; their published numbers are caveated/"flawed, removed"). Ours =
  464 ms / 963 SHA-256 on 10-core M4. The wall-clock gap is *partly* hardware
  (64 vs 10 cores) + hash type (keccak≠SHA), but our **2.34× scaling** shows that
  on equal HW we'd still be ~3–4× slower — because Binius's **word-level / GF(2)
  representation is ~32× less data** ("64-fold smaller than bits"), so it is NOT
  memory-bound (scales to 64 cores *and* cheaper per-core).
- **Reconciles the whole investigation**: (a) the shipped fused evaluator
  (−24.5%, `9505876`+`7a27606`) fixed the *access pattern* but NOT the root
  cause — it still streams 1536 GF(2¹²⁸) slices; (b) the byte-identical
  small-value prover (closed, ~624×/term) also wouldn't fix it — same slice
  volume post-skip; (c) **only the WORD-LEVEL reformulation reduces the data
  ~32×**, killing both the memory-bound scaling *and* the arithmetic cost. So the
  real Binius-scale lever is the **word-level AND reduction** (drop the `×D=32`
  bit-slice materialisation), more than the sumcheck small-field tricks below.
  The byte-identical paths can't reach it because they don't change the data
  volume.
- **Implication for priority**: the `×32` GF(2¹²⁸) slice volume is THE bottleneck
  (memory-bound, poor scaling). Target reducing the materialised data — word-level
  operands (Binius AND reduction) — over further sumcheck-arithmetic micro-opts.
- **▶ CONCRETE PORT PLAN: `documentation/f2-hadamard-oblong-port-plan.md`** (after
  reading the real Binius64 source at `~/binius64`). Key corrections to the
  univariate-skip entry below (which conflated *original* Binius with **Binius64**):
  **(1) the field is SHARED** — Binius64 `B128` is the **GHASH GF(2¹²⁸)**
  (`X¹²⁸+X⁷+X²+X+1`, `0x87`), *identical to our `BinaryFieldGF128`*; **no tower
  rewrite** (the original Binius tower is a different system; Binius64 uses one
  `GF(2⁸)` subfield only for ≤3 deterministic skip challenges). **(2) The target
  is the `OblongZerocheckProver`** (`~/binius64/crates/prover/src/and_reduction/`,
  verifier `crates/verifier/src/protocols/bitand.rs`): operands stay packed
  (`Vec<Word>`), the bit dim is a univariate-skip round + additive NTT, the 1536
  slices are never materialised. **(3) It's a port over our field**, not a
  re-architecture; the real risk is the integration seam (tying operand evals to
  our columns via ψ_α / a shift-reduction analogue) — plan §4. Benchmark target:
  Binius's ~123 ns/AND (README: 2²⁰ ANDs in 128.58 ms) vs our ~398 ns/relation.

### ⭐ Univariate skip / small-characteristic packed sumcheck — the Binius scaling lever we're MISSING (IDENTIFIED; big, verifier-visible protocol change)
- **⚠ STATUS UPDATE — the oblong discharge realizes this (for the bit dimension)**:
  the word-packed **oblong AND zerocheck** (Shipped work, top of this doc) **is**
  Binius64's univariate-skip AND reduction — its Phase-1 round fuses the 5
  bit-index variables into one univariate `R₀(Z)` and runs the NTT + products in
  **GF(2⁸)** (`Gf8Scheme`), exactly the "univariate skip + small-field arithmetic"
  this entry says we're missing. The e2e measurement (~5.6× faster Hadamard prove
  at nvars=16) confirms it. So once the oblong is the sound production discharge
  (pending the `ψ_z`→commitment binding) this lever is **shipped, not missing**;
  the "we do neither / never attempted" verdict below describes the **fused**
  bit-slice discharge the oblong replaces. (Phase-2's row sumcheck stays GF(2¹²⁸) —
  post-fold-at-`z` values are general — so the skip is on the bit dimension only,
  which is where the win is.)
- **Why this matters**: the whole discharge runs the sumcheck in **GF(2¹²⁸)**,
  but its bit-slices `U_b,V_b,W_b` are **GF(2)-valued** on the hypercube — i.e.
  base-field witness, extension-field sumcheck. That's precisely the regime the
  **univariate skip** (Gruen; used by Binius64) exploits: "fuse several Boolean
  variables into a single higher-degree variable… when `F` is a degree-`2k`
  extension of `B`, extension-field multiplications drop from ~`n` to ~`n/2k`,
  **practical savings up to 128× for GF(2)→GF(2¹²⁸)**." Binius64 has "an
  advanced univariate skip variant… for proving **bitwise-AND constraints**" —
  exactly our Hadamard. Binius also does per-round arithmetic in the 8-bit
  **Rijndael field** "where possible", not GF(2¹²⁸). **We do neither.** Our
  scaling is `O(2ⁿ·K·D)` *GF(2¹²⁸)* muls; Binius keeps the dominant rounds in
  GF(2)/GF(2⁸).
- **Correction to the "small-value / multi-round-skip" entry above**: that entry
  concluded skip was "compute-neutral / large-nvars-only / shelved". That verdict
  applies to the **verifier-preserving Dao prefix** (byte-identical, and for
  `d=3` most of the `4^v` grid is off-hypercube ⇒ still big-field ⇒ limited).
  It does **NOT** apply to the **univariate skip**, which is a *different,
  verifier-visible* protocol (the fused variable has degree ~`d·2^k`; the
  verifier does one univariate check) and is the one that gets the tower-depth
  reduction. We never attempted the univariate skip. Do not let the shelved
  byte-identical prefix imply the univariate skip is also a dead end — it's the
  opposite.
- **What it would take (why it's big, not a drop-in)**: unlike the fused
  evaluator (byte-identical, opt-in `RoundPolyEvaluator`), this **changes the
  proof and the verifier** — a new sumcheck sub-protocol (prover does a
  small-field univariate evaluation / FFT over the fused rounds; verifier checks
  a high-degree univariate; fresh soundness argument). It also wants the
  discharge arithmetic in a small tower subfield (GF(2)/GF(2⁸)), which touches
  the field plumbing. Realistic gain: **multi-× to ~order-of-magnitude on the
  discharge** (vs the 24.5% the byte-identical fused prover got), since the
  discharge is ~92% of the nvars=16 prove and is GF(2¹²⁸)-mul-bound.
- **Design pass — ✅ DONE: `documentation/f2-hadamard-univariate-skip-design.md`
  (see its §8 RESOLUTION).** Initial draft over-indexed on Binius's
  verifier-VISIBLE univariate skip; extracting the Dao paper (`pdftotext` on
  `cs.nyu.edu/~zd2131/papers/26-587.pdf` — eprint is Cloudflare-walled)
  **corrected it**: the relevant technique is the **byte-identical** small-value
  sum-check prover. Cost: speedup `Θ((d²κ)^{1/δ})`, `δ=log₂(d+1)`; for our **d=3
  ⇒ 3√κ**; `κ≈N^{log₂3}` (Karatsuba tower) ⇒ ss=GF(2) `κ≈2180 ⇒ ~140×` (v\*≈7),
  ss=GF(2⁸) `κ≈81 ⇒ ~27×` (v\*≈5). Off-hypercube extrapolation is `Θ(d²)` **sb**
  (small×big = select), not bb — that's why the witness staying small is the
  whole game. **This reframes the removed skip prover (ce177a3..f615f7f):** it
  was byte-identical and we built Procedure 1 + the prefix correctly, but ran
  **every op in GF(2¹²⁸)**, so we paid the larger op-count with none of the √κ
  discount ⇒ 40–74× slower. The fix is doing the `{0,1}^v`-grid arithmetic in
  **bit-packed GF(2)** (ss=AND/XOR, sb=select) with **delayed reduction** — no
  additive NTT, no GF(2⁸) tower required, **no verifier change** (existing
  byte-identity gate applies), skeleton recoverable from git. Revised Phase-0:
  resurrect the prefix, reimplement its hot `{0,1}^v` accumulation in bit-packed
  GF(2), measure vs the `v` standard GF(2¹²⁸) rounds (gate ≥~4×). Caveat: aarch64
  has PMULL but no GFNI/native tower mul, so the real win is "large but < 140×" —
  Phase-0 measures it. Sources: Dao et al. eprint 2026/587 (also `cs.nyu.edu/~zd2131`);
  Bagad et al. "Sum-Check over Fields of Small Characteristic" eprint 2024/1046;
  "Packed Sumcheck over Fields of Small Characteristic" eprint 2025/719;
  binius.xyz blueprint (Univariate Skip / Rijndael Zerocheck / ANDs);
  irreducible.com "Slicing Up Binary Towers"; LambdaClass "Binius Part 2".
- **Phase 0 — ✅ DONE, gate CLEARED (~19× ceiling).** Probe
  `poly/benches/binary_gf_compare.rs::bench_discharge_comb` measures the
  discharge comb at one point (512 terms), bb vs sv: **`bb_GF128_products` 2594
  ns** (512 post-fold GF128 products) vs **`sv_packed_psi_x4` 137 ns** (bit
  operands: `(U&V)⊕W` packed + ψ_σ via the x4 NEON kernel + 17 bb) ⇒ **~19×** on
  M-series. This is the realized √κ on the inner comb — the part the removed skip
  prover ran in `bb` (2594 ns). BUT this is the **ceiling** — only the 4 boolean
  cells of the `(d+1)²=16`-cell v=2 prefix; the **12 off-hypercube cells** use
  *extrapolated* (non-bit) operands ⇒ genuine field products.
- **Phase 1 NET analysis — ❌ byte-identical path is ~NEUTRAL for d=3 (walked
  back the GO).** Procedure 1 costs `O(d^v)` bb/multiproduct ⇒ ~13 bb/term
  (`eq·U·V` 9 + `eq·W` 4) vs the standard round's ~6 bb/term; only the boolean
  base is `ss`, and at **d=3 the off-hypercube `bb` dominates** ⇒ net
  ~neutral-to-worse, not 19×. (The `3√κ` asymptotic needs boolean cells to be a
  *majority* of the work — true at high `d`, not d=3.) Reconciles: the removed
  GF128 prefix was "~neutral"; Dao's 10.9× is **high-degree** Spartan; Binius's
  AND win is **word-level (no `×D=32`) + verifier-visible univariate skip +
  GF(2⁸)**, not the byte-identical small-value prover. **Corrected conclusion:**
  the byte-identical shortcut does NOT pay for our d=3 discharge; the real
  Binius-scale lever is the **verifier-visible word-level + univariate-skip
  rearchitecture** (design doc §1–6 — multi-week, research-grade). See
  `documentation/f2-hadamard-univariate-skip-design.md` §10–11.
- **✅ EMPIRICAL CONFIRM (probe `bench_prefix_vs_standard_term`)**: per
  `(relation,bit)` term — small-value v=2 prefix (Procedure-1 multiproduct +
  extrapolation) **7.12 µs** vs standard fused coeff term (~6 GF128 muls)
  **11.4 ns** ⇒ **~624× worse**. Dominated by ~4 `prepare_eval_aux`
  `batch_invert` GF128 inversions + Vec allocs/term (the removed prover's 40–74×
  overhead); even hoisted, the `O(d^v)`-bb floor is ~2× worse at d=3.
  **Byte-identical small-value path is now empirically closed for d=3.** Only the
  verifier-visible word-level + univariate-skip rearchitecture remains as the
  Binius-scale lever.

### Round-1 fast path for the Hadamard degree-3 zerocheck (Phase 1 — ✅ SHIPPED; see Shipped-work entry for results)
- **Where**: `piop/src/lookup/hadamard.rs` (`prepare_hadamard_group` builds
  the group via `MultiDegreeSumcheckGroup::new(3, …)` — no fast path) and
  `protocol/src/f2_hadamard.rs::prove_f2_hadamard_phase` (calls
  `build_all_operand_slices`, which materialises every operand bit-slice).
- **Root cause being fixed**: the discharge zerocheck materialises **1536
  bit-slices** (16 relations × 3 operands × 32 bits) as full-width
  `GF128::Inner` (= `Uint<2>`, 16 B) MLEs → 24 GB at nvars=20 (the
  super-linear prove blowup; see the "Hadamard discharge cost" shipped
  entry). Round 1 also touches the full (μ)-var hypercube.
- **Fix (proposed approach)**: port the booleanity degree-3 round-1 fast
  path (`piop/src/lookup/booleanity.rs::BooleanityRound1FastPath`) to the
  Hadamard group. The fast path reads the **packed operand columns**
  (`DenseMultilinearExtension<BinaryPoly<D>>`, 4 B/row — the `op_mask` that
  `build_operand_slices` computes internally, and `build_adder_operand_columns`
  already produces) instead of materialising 16-B F-slices; round 1 does
  boolean arithmetic and `fold_with_r1` emits half-size folded F-slices for
  rounds 2..n. The Hadamard round poly factors as
  `M(t) = eq(t,r₀)·Σ_{b'} E_other(b')·G(t,b')` with
  `G(t,b') = Σ_k γ'^k Σ_b σ^b (U_b(t)·V_b(t) + W_b(t))` (char-2), and
  `G = 0` on the honest hypercube ⇒ `M(0)=M(1)=0`, so only `M(2),M(3)` need
  computing (mirrors booleanity's `p1(1)=0`). Each operand bit folds by the
  `{0,1}² → {0,t,1−t,1}` lookup.
- **CRITICAL field subtlety**: the booleanity fast path computes its
  boundary points as `one+one` / `one+one+one` — correct only for the
  **large prime field** of the integer/Q[X] pipeline (where booleanity
  lives). Over `GF(2^128)` (char 2) those collapse: `2 = 0`, `3 = 1`. The
  multi-degree sumcheck instead uses `F::from_with_cfg(k)`, which under the
  bit-pattern convention is the field element `X` for `k=2`, `X+1` for `k=3`
  (distinct) — see `prover.rs:170-196` and `F2EqColRound1FastPath`
  (`f2_prove.rs:639`, the degree-2 GF128 template). The Hadamard fast path
  MUST use `F::from_with_cfg(2/3)` and `eq(t,r₀)=(1−t)(1−r₀)+t·r₀`, not the
  booleanity integer shortcut.
- **Correctness gate**: a test that runs the same zerocheck through the
  generic `new(3,…)` group and the `with_round_1_fast` group and asserts an
  **identical** `MultiDegreeSumcheckProof` (the `Round1FastPath` contract:
  `round_1_message` must equal what a faithful generic round-1 would emit).
- **Expected gain**: ~2× peak memory (round-1 24 GB materialise removed;
  `fold_with_r1` still emits ~12 GB of half-size slices at nvars=20) + the
  round-1 boolean-arithmetic win. Does NOT fully kill the nvars≥20 blowup —
  that needs **Phase 2** below.

### Phase 2 — cut the Hadamard slice count via `(col,Δ)` dedup (❌ INVESTIGATED — ~2% ceiling for SHA-256, NOT worth it)
- **Where**: the post-round-1 `fold_with_r1` (`HadamardRound1FastPath`)
  materialises one folded F-slice per `(operand_column, bit)` = `48 operands ×
  32 bits = 1536` slices (the `12.9 GB`-at-nvars=20 cost, after the round-1
  fast path already halved the pre-fold `24 GB`).
- **Original plan (now seen to be partly infeasible)**: operate on
  per-`(col,Δ)` slices with the XOR/complement structure encoded in the comb
  *coefficients*, so shared pairs materialise once. **This only works for a
  LINEAR comb.** The Hadamard comb is `U·V − W` — **quadratic** in the
  operands; `U·V` cannot be factored into per-`(col,Δ)` slices because bit-level
  XOR is nonlinear over `F` (`a⊕b = a+b−2ab`), so an operand bit-slice is a
  genuinely new `{0,1}` column, not an `F`-combination of its terms' slices.
  Only the linear `−W` term could share slices.
- **Dedup ceiling measured against the real 16-relation layout**: the only
  structurally-identical operands are **C12.V ≡ C13.V** (both `col(W_E)`) → 47
  of 48 distinct. The 13 adders contribute 39 operands, each a *carry-computed*
  column (`build_adder_operand_columns`) with a distinct `(t,x,y)` triple, so
  no structural sharing. The underlying *committed* columns overlap heavily
  (`W_E`, `W_W`, `W_T1`, …) but the materialised slices are XOR/carry compounds
  of them, which don't. **Net: ~2% slice reduction (12.9 → 12.6 GB at
  nvars=20) — not worth the comb-restructure + byte-identity re-gate.**
- **Verdict**: shelved. The memory lever with real headroom is reducing slice
  *size* (skip more rounds — but bit-packed, see the small-value entry's "If
  ever revived" note) or slice *width* (tower-subfield representation), not
  slice *count*.

### Specialised fused Hadamard sumcheck prover for rounds 2..n (✅ SHIPPED, commits `9505876` + `7a27606` — −24.5% discharge at nvars=16, byte-identical, size-gated)
- **✅ SHIPPED RESULT (commit `9505876`)**: built exactly the fix below. Added
  an opt-in `RoundPolyEvaluator` hook to `ProverState`/`MultiDegreeSumcheckGroup`
  (`prove_round` uses it for the round-poly evals instead of the generic
  per-point gather; fold + message formation unchanged ⇒ byte-identical).
  `HadamardRoundEvaluator` loops `(relation,bit)`-outer / point-inner, streams
  each triple's 3 slices, factors `eq_r` into a 2nd pass, precomputes
  `γ'^k·σ^b`, and parallelises over point-chunks (local accumulator stays in
  L1/L2). **Measured (M-series, `parallel simd unchecked`):** nvars=16 discharge
  **554 → 479 ms (−13%)**, Prove-Hadamard 600 → 525 ms; nvars=9 unchanged.
  **Size-gated at `num_vars >= 14`** (`FUSED_MIN_VARS`): the fused path wins once
  the 1536 slices spill cache (≈200 MB at 14, ≈800 MB at 16) but its per-chunk
  setup *regresses* tiny discharges (+~25% at nvars=9), so production nvars=9
  stays on the generic path. Byte-identity gated by `fused_matches_generic_large`
  (nvars=14, fused active) + `fast_path_matches_generic` (nvars=3, round-1 only);
  60 piop + 9 f2_hadamard tests green.
- **✅ FOLLOW-UP SHIPPED — coefficient-form round poly (commit `7a27606`)**: the
  eval-form above lifted U,V,W to the 4 boundary points before multiplying (14
  muls/(k,b,pt)). Since the comb structure is known, compute the round poly's
  **coefficients** directly instead: per `(k,b)` the degree-2 `U·V−W` coeffs
  come from the two folded halves via **Karatsuba** (`p0=u0·v0`, `p2=Δu·Δv`,
  `p1=u1·v1−p0−p2` — cross term free), then subtract `W`, accumulate weighted
  coeffs `g=(g0,g1,g2)` per point, multiply by the linear `eq_pt`, and evaluate
  the resulting cubic at `{0,1,X,X+1}` once at the end. **6 muls/(k,b,pt) vs
  14.** The generic comb can't (black box); the specialised evaluator can.
  Byte-identical (same cubic). **nvars=16 discharge: 554 → eval-form 479 → coeff
  429 → +Karatsuba 418 ms (−24.5%)**, Prove-Hadamard 600 → 464 ms; nvars=9
  unchanged. 60 piop + 9 f2_hadamard tests green.
- **SIMD finding (why literal lane-batching was NOT the lever on aarch64)**: the
  discharge muls are *general* GF(2^128)×GF(2^128) (`vmull_p64`/`pclmulqdq` via
  Karatsuba `clmul_128x128` + reduce). The existing x4 `psi_α` kernels win only
  because **one operand is a bit** (0/1 → a select, no CLMUL); the discharge has
  no bit operand post-fold, so that trick doesn't apply, and the CLMUL units are
  throughput-bound (NEON has no 4-wide CLMUL). The portable lever is **fewer
  muls** (the coeff form above), not wider SIMD. Tellingly, the 2× mul cut
  (14→6) bought only ~12% wall-clock → the bottleneck has **shifted off
  arithmetic** onto memory traffic / `F` clones / the per-point `g` read-modify-
  write. **Next levers (open):** (a) cut `F::Inner` clones in the hot loop
  (read once, work on stack); (b) lazy/deferred GF128 reduction — accumulate
  unreduced `clmul_128x128` outputs (XOR), reduce once per coeff (reduction is
  GF(2)-linear ⇒ byte-identical) — needs `clmul_128x128`/reduce exposed `pub`
  from `poly`; (c) exploit round-2's `{0,1,r1,1−r1}` slice values (small-value,
  round-2-only — ~50% of the work, but edges toward the removed skip prover).
  Also still open: tune `FUSED_MIN_VARS` (12–15 crossover) and `CHUNK` (512).
- **Original profiling + design (for context):**
- **Profiling (2026-06, nvars=16, `parallel simd unchecked`, this box)**: the
  Hadamard discharge is **~92% of the whole prove** — `Prove-NoHadamard` ≈ 46
  ms vs `Prove-Hadamard` ≈ 600 ms (so the discharge alone is ~554 ms). The
  base prove (Commit ~24 ms / UAIR-FULL ~13 ms / Open-FULL ~2.6 ms, from the
  `micro` group) is a rounding error beside it. nvars=16 fits RAM (slices are
  ~0.8 GB), so this is **not** the swap regime — it's the discharge's
  rounds 2..n compute/access pattern.
- **Confirmed bottleneck** (consistent with the rejected weight-precompute
  entry, which was *also* measured at nvars=16): the generic
  `ProverState::prove_round` rebuilds a **1537-element value array per
  hypercube point** — `vals0[j]=poly[j][index]`, `vals1[j]=poly[j][index+1]`
  over all `1 + 1536` MLEs — then calls the `comb_fn` 4× (degree-3 boundary
  pts). That per-point **scattered gather across 1536 separate slice `Vec`s**
  (latency-bound: 1536 live streams thrash cache + TLB) dominates; the CLMUL
  arithmetic is cheap. Memory *streaming* alone would be ~50–100 ms; the 554 ms
  is the scattered-gather + per-point framework overhead (scratch, dyn comb_fn).
- **The fix — a standalone fused prover** (the discharge already runs as its
  own single-group `prove_as_subprotocol(vec![group])`, so no batching to
  preserve): swap the loop nest to **outer `(relation k, bit b)`, inner
  hypercube point**. For each of the 512 `(k,b)` triples, stream its 3 slices
  `U_{k,b}, V_{k,b}, W_{k,b}` (sequential, cache-friendly) and accumulate their
  degree-3 contribution `w_{kb}·(U_b(X)V_b(X)−W_b(X))` into the 4 round-eval
  accumulators; fold `eq_r(X)` in once at the end (it factors out of the `(k,b)`
  sum). Each slice is still read once per round, but **sequentially per-slice**
  instead of in a 1536-wide scatter — which is exactly the access pattern the
  ledger says is needed. Precompute `w_{kb}=γ'^k·σ^b` (free here — no longer a
  micro-opt on top of the bad gather). Fold slices in place per round.
- **Correctness**: gate byte-identical against the generic path (extend the
  existing `fast_path_matches_generic` style to all rounds). Verifier/proof
  unchanged. **Expected**: removes the per-point 1537-gather → should cut a
  large fraction of the 554 ms; measure via the `F2_ONLY=hadamard HAD_NVARS=16`
  Prove-NoHadamard-vs-Hadamard split. Risk: if some residual is genuine
  arithmetic, the win is smaller than hoped — but the access-pattern change is
  the one thing the prior (rejected) micro-opt did NOT address.

### Fast u64 scatter in `scatter_matrix_into_gpu_slab` (rejected on second-pass measurement)
- **Status**: implemented in a temporary branch state via a new
  `FastScatterCell` trait (default body using
  `write_transcription_bytes_exact`, override for `BinaryU64Poly<D>`
  using `ptr::write_unaligned::<u64>(dst, self.pack_u64().to_le())`),
  added as a bound on `ZipTypes::Cw`. Compiled clean, 13/13 F_2 lib
  tests passed, but criterion A/B showed **no measurable change**
  — reverted. See entry under "Investigated, didn't help" below.

### Hadamard discharge for the 13 + 3 column-level relations (step 5 of 5)
- **What**: an external `(x + X·c) ⊙ (y + X·c) = c + X·c` (in
  `F_shift`) check for each of the 13 chained-`BiniusAdd` binary
  steps (C5–C11), plus the 3 AND-flavoured relations
  (C12 = `W_E ⊙ W_E^{↓1}`, C13 = `(1 − W_E) ⊙ W_E^{↓2}`,
  C14 the `Maj` Hadamard from doc §3.5). Total 16 Hadamard
  relations.
- **Why it matters**: this is the last piece of AIR-level soundness
  for the chained-Binius construction. Without it, bits 1..31 of
  every modular addition (and every AND output) are honest-prover-
  only at the AIR layer. Steps 1–3 shipped the surrounding
  algebraic shape (ideal split, packed `PA_C`, 13 LSB checks via
  bit-op-virtual `SHR^j(PA_C)`); step 5 is the actual soundness
  load-bearer.
- **Inputs / outputs at the AIR layer**:
  - For each addition step, the Hadamard inputs are virtual:
    bits 0..30 of `c` = `SHR^1` of XOR-of-(target, x, y); bit 31
    of `c` = β_k, a committed bit (`W_β`, packed analogously to
    `PA_C`; not yet a primary column on this branch — would be
    added with step 5).
  - For each AND step, inputs are free linear combinations of
    `W_E`, `W_A` (and their shifts) per doc §3.5.
  - Output is the per-row Hadamard product, which the
    constraint side declares equals the right-hand side
    (`c + X·c` for adds; the committed AND-result column for
    AND/Maj).
- **What's needed**: a PIOP-layer Hadamard-product check on
  column-level commitments. Distinct from per-cell SHR/Rot
  (already shipped) and from row-shift virtuals — Hadamard is a
  per-bit product across two columns, not a re-indexing.
  Concrete protocol shape is a separate design pass; placeholder
  references include the doc's §3.4 ("mechanism left as a design
  parameter") and the Binius literature's Hadamard-via-grand-
  product approach.
- **Adjacent column work that lands with step 5**: commit `W_β`
  as a primary column (1 packed bit-poly_32, bits 0..12 used for
  the 13 β_k); add 12 internal `BitOpSpec`s for `SHR^j(W_β)`
  bit extraction (mirrors the `PA_C` setup). Until then, β
  values are unconstrained — the deferred Hadamard would be the
  thing that pins them.

### ψ_α-projected sumcheck term for Hadamard discharge — proves convolution, not coefficient-wise (design pass on step 5)
- **What (proposed)**: leave the 16 Hadamard relations unprocessed
  in the IC (already the state on this branch — C12–C14 are
  witness-only, the adds pin only the LSB via the `(X)` ideal), then
  *after* the ψ_α projection add one extra batched-sumcheck term
  `∑_x (ψ(v)·ψ(u) − ψ(w))·eq(x, r) = 0`, reusing the IC point `r`
  (already in `GF(2^128)^μ`, so the proposer's "ψ(r)" is a no-op).
  Attractive because it reuses the already-projected trace and the
  existing degree-2 sumcheck + lift-and-project open with near-zero
  extra commitment — would be the cheapest possible discharge.
- **Why it does not work as written**: ψ_α is an `F_2`-algebra
  homomorphism `F_2[X] → GF(2^128)`, so `ψ(v)·ψ(u) = ψ(v·u)` with
  `·` the **polynomial / convolution** product in `F_2[X]`. The
  SHA Hadamards (AND, Maj, and the Binius adder identity
  `(â + X·ĉ) ⊙ (b̂ + X·ĉ) = ĉ + X·ĉ`) are **coefficient-wise** (`⊙`),
  *not* convolution. So even for an honest prover,
  `ψ(v)·ψ(u) − ψ(w) ≠ 0` when `w = v ⊙ u`. Counterexample:
  `v = u = 1 + X` gives `ψ(v)·ψ(u) = (1+α)² = 1 + α²` (char 2),
  but `ψ(v ⊙ u) = ψ(1 + X) = 1 + α`, and `1 + α² ≠ 1 + α` for
  `α ∉ F_2`. The term proves the wrong relation (per-row
  convolution). The Binius carry identity in particular *requires*
  per-coefficient idempotency `c_i² = c_i`, which convolution (`ĉ²`)
  destroys — so convolution can never stand in for the adder ⊙.
- **Corrected direction**: the sumcheck *shape* is right; the fix is
  to NOT collapse the bit-axis with ψ_α for the Hadamard columns.
  Expand the `D = 32` coefficient positions into 5 extra sumcheck
  variables, so `V, U, W` become MLEs over `{0,1}^{μ+5}` and
  `W(x,b) = V(x,b)·U(x,b)` is a genuine elementwise product; then
  `∑_{x,b} (V·U − W)·eq((x,b),(r,ρ)) = 0` is sound + complete (the
  standard Binius AND zerocheck). Cost: the open must expose
  **bit-level MLE evaluations** at a `(μ+5)`-point — a multilinear /
  eq functional of each cell's bits, not the α-power functional ψ_α —
  same lift-and-project machinery, different weights; and the
  Hadamard sumcheck has arity `μ+5`, so it likely runs as a
  *separate* sumcheck (groups in `MultiDegreeSumcheck` share
  `num_vars`) rather than folded into the μ-var eq·col group. The
  row-axis of its eq can still reuse the IC's `r`, so the "reuse r"
  instinct survives.
- **Correction-polynomial variant (also rejected)**: compute
  `f = ∑_x (u_x·v_x − w_x)·eq(x,r)` in the X-domain during the IC (a
  degree-≤62 poly in X, cheap to send), then check
  `∑_x (ψ(u)ψ(v) − ψ(w))·eq(x,r) = ψ(f)` in the projected sumcheck.
  Fails: the LHS *is* `f★(α)` for `f★ = ∑_x (u_x·v_x − w_x)·eq(x,r)`
  the committed columns' true convolution discrepancy, so the check
  only forces `f = f★` (SZ over α) — `f` is pinned to whatever the
  committed `w` produces, making the term a tautology that constrains
  `w` not at all. Deeper reason: `ψ(u)ψ(v)` only ever exposes the 63
  convolution coefficients `c_n = ∑_{j+k=n} u_j v_k`; the AND diagonal
  `∑_i u_i v_i` is provably not a function of them (it lives on the
  main diagonal the product polynomial doesn't carry), so no X-domain
  `f` — itself just another convolution object — can recover it. The
  coefficient index must be a live sumcheck variable. (If `f` is read
  as the coefficient-wise `∑_x (u_x⊙v_x − w_x)·eq`, the IC cannot even
  form it — ⊙ is not a ring op in F_2[X] — and it collapses to the
  original broken `=0`.)
- **General impossibility (covers the "check f's F_2-coefficients
  are 0" variant and every X-domain variant)**: everything the
  ψ/IC view exposes about a Hadamard pair is a function of the
  *convolution* `u·v` (and of `w`). Convolution does not determine
  the Hadamard. Witness: `u = 1+X, v = 1+X³` vs `u' = 1+X², v' =
  1+X+X²` have the **same** product `u·v = u'·v' = 1+X+X³+X⁴`, but
  **different** coefficient-wise products `u⊙v = 1` vs `u'⊙v' =
  1+X²`. Put `w := 1`: the honest instance `(u,v,w)` (here `w=u⊙v`)
  and the false instance `(u',v',w)` (here `w≠u'⊙v'`) produce
  *identical* `ψ(u)ψ(v)`, `ψ(w)`, and `f = ∑(u·v−w)·eq` — every
  coefficient, including the F_2 parts. No predicate on those can
  accept one and reject the other, so no such check is sound.
  Separately, "f's F_2-coefficients = 0" also breaks **completeness**:
  the honest `f★` is the (nonzero) cross-term discrepancy and its
  F_2-components are generically nonzero, so it rejects honest
  provers. Bottom line: the coefficient index must be a live
  sumcheck variable; convolution-domain tricks cannot recover the
  diagonal.
- **Status**: semantics confirmed = coefficient-wise / bitwise AND
  (case 3). **Chosen design = per-coefficient-slice zerocheck,
  booleanity-style** (full plan: `protocol/src/f2_hadamard_plan.md`).
  Run a μ-var degree-3 zerocheck
  `∑_x eq(x,r)·∑_k∑_b (γ')^k σ^b (U_{k,b}·V_{k,b} − W_{k,b}) = 0` over
  the bit-slice MLEs, reusing the booleanity infrastructure
  (`piop/src/lookup/booleanity.rs`: `build_shifted_bit_slice_mles`,
  `build_virtual_booleanity_mles`, `finalize_booleanity_*`,
  `verify_bit_decomposition_consistency`) with the self-product
  `v(v−1)` swapped for the cross-product `U·V − W`. The 32 slice evals
  per column ride **one** column opening via the recombination check
  `∑_b a^b v_b(r*) = parent_eval` — no per-slice openings.
  - **F_2-specific subtlety**: the recombination element must be fresh
    *after* the bit-slice evals (SZ over a degree-31 poly in `a` pins
    all slices). The integer path reuses its early Step-3 projection
    (`prover.rs:520`) and is sound only because its bit-slices also feed
    the whole CPR; our Hadamard bit-slices are touched only by this
    check, so they need their own fresh binding. Fix: either run the
    Hadamard sumcheck before α (reuse α, Wiring R) or sample a fresh `a`
    + a second `ψ_a` open (Wiring F). Both reuse the existing open.
  - **Alternative (documented, not chosen)**: bit-axis expansion — a
    (μ+5)-var zerocheck folding the bit axis, discharged by an
    eq-over-bits open. Smaller proof (one bit-MLE eval/col vs D
    slice-evals/col) but needs a new sumcheck arity + a new per-cell
    open contraction. Revisit if proof size dominates.
  - **Prereqs**: `W_β` carry column (absent on this branch; needed for
    the 13 adders, not the 3 ANDs); missing `W_E ↓1,↓2` / `W_A ↓1,↓2`
    shifted-bit-slice specs; row-shift discharge for shifted operands
    (shared with Issue 1 below).
  - Supersedes the bare "add one ψ-projected term" idea for AND/adder.
  - **Progress (Wiring R chosen)**: A0 landed — the piop cross-product
    zerocheck at `piop/src/lookup/hadamard.rs`
    (`prepare/finalize_hadamard_{group,prover,verifier}` +
    `HadamardTriple`), mirroring `booleanity.rs` with the self-product
    `v(v−1)` swapped for `U·V − W`. Degree-3 single group, γ'/σ-batched.
    Unit test passes (honest `W=U⊙V` accepted; flipped `W` ⇒ non-zero
    claimed sum, rejected by the `claimed_sum==0` gate) and is
    clippy-clean. Committed `a110d32`.
  - **Progress: A1 core landed** (`ade7ab9`) —
    `protocol/src/f2_hadamard.rs`: `prove_f2_hadamard_phase` /
    `verify_f2_hadamard_phase` + `F2HadamardSpec`. Builds Δ=0 bit-slices
    for the distinct referenced columns via `compute_bit_slices_flat`
    (NOT `build_shifted_bit_slice_mles` — that asserts shift ≠ 0;
    `uair/src/lib.rs:259`), runs the degree-3 group, and exposes
    per-slice evals tied to the committed columns by
    `verify_bit_decomposition_consistency`. Tests over real
    `BinaryPoly<D>` columns: honest round-trip incl. the
    `Σ_b α^b v_b(r*_H)` recombination at a test α, corrupt-W rejected,
    empty-specs no-op. Requires the `parallel` feature (the F_2 path's
    round-1 fast path uses rayon `reduce`).
  - **A1 wiring DONE (working tree, NOT committed)**: the Hadamard
    zerocheck phase is threaded into `prove_f2_uair_with_groups` /
    `verify_f2_uair_with_groups` *before* α (Wiring R), reusing
    `ic_state.evaluation_point` as `r`. Added `F2Proof.hadamard_proof:
    Option<F2HadamardProof>`, a `hadamard_specs` param on both group
    fns (existing callers pass `&[]` → no-op), and two
    `F2VerifyError` variants (`MissingHadamardProof`, `Hadamard`). New
    e2e test `hadamard_phase_roundtrips_in_flow` (3-col `HadF2Uair`,
    trace with `W = U⊙V`, driven through the real group fns via
    `F2Types<D>`): honest round-trip accepts, a flipped `W` bit is
    rejected with `F2VerifyError::Hadamard`. **All 37 protocol lib
    tests pass** (`--features parallel`).
    - **Not committed**: `f2_prove.rs` carries ~600 lines of
      pre-existing uncommitted GF128-refactor WIP from before this
      session; the ~80 Hadamard lines are interleaved and can't be
      isolated without interactive staging. Commit needs the owner to
      either land the WIP or accept a bundled commit. (A0 `a110d32`
      and A1-core `ade7ab9` were committable because they were new /
      clean files.)
  - **A3/A4 discharge wired in-flow (trusted; working tree)**: the
    prover computes each Hadamard column's α-eval at `r*_H`
    (`alpha_parent_evals`, committed `390c846`), absorbs them, and ships
    them in `F2Proof.hadamard_parent_evals`; the verifier recombines
    `Σ_b α^b·v_b == parent_eval` (`verify_bit_decomposition_consistency`)
    after α (Wiring R, so α is fresh w.r.t. the bit-slice evals). The
    in-flow round-trip test now exercises the recombination; all 37
    protocol lib tests pass. The `f2_prove.rs` threading is still
    uncommitted (entangled with the GF128 WIP — option 1: lands after
    the WIP).
  - **SOUND DISCHARGE SHIPPED (Approach A — SUPERSEDED by Approach B
    below; kept for the soundness rationale)**: the trusted `parent_evals`
    are now PCS-bound. Full-pipeline change in `protocol/src/f2_prove.rs`:
    - **`F2FullProof`** gains four fields (all `None`/empty without
      Hadamard): `hadamard_multipoint_eval: Option<MultipointEvalProof>`,
      `hadamard_evals_at_rstar_h: Vec<Gf>` (every projected column's eval
      at `r*_H` — the second mp's `up_evals`),
      `hadamard_open_evals_at_r0h: Vec<Gf>` (evals at `r_0^H`),
      `hadamard_open: Option<F2OpenProof<D>>`.
    - **New entry points** `prove_f2_full_with_hadamard` /
      `verify_f2_full_with_hadamard`. The existing
      `prove/verify_f2_full_with_bit_ops` (+ pre-paired) signatures are
      **unchanged** — all ~15 bench call sites untouched. The post-commit
      body was factored into private `prove_f2_full_impl` /
      `verify_f2_full_impl(.., hadamard_specs)`; the old entries call them
      with `&[]` (this also de-duplicated the prove/pre-paired bodies that
      were copy-pasted before).
    - **The discharge**: after the main mp+open at `r*`/`r_0`, when
      `hadamard_specs` is non-empty, a **second** `MultipointEval` collapses
      *every* projected column's eval at `r*_H` to a fresh `r_0^H`, then a
      second `prove_f2_open` opens the **same contiguous witness slice** at
      `r_0^H` (collapsing all columns — not just the Hadamard subset —
      reuses the witness-slice open verbatim and dodges subset-column
      mapping; the parent-evals are then a subset of the bound
      `evals_at_rstar_h`). Verifier mirrors and adds the **binding check**
      `hadamard_evals_at_rstar_h[distinct] == uair.hadamard_parent_evals`,
      which upgrades the kept-as-is in-flow recombination to sound.
    - **Soundness-critical detail (caught during impl)**: the second mp's
      `up_evals` (`hadamard_evals_at_rstar_h`) **must be absorbed into the
      transcript before the second mp samples its γ** — exactly as
      `prove_f2_uair_with_groups` absorbs `column_evals_at_rstar` before the
      main mp. Without it a prover seeing γ first could fit fake per-column
      `r*_H` evals to the γ-batched sum while keeping the witness slice
      honest, defeating the binding; with it, SZ over the fresh γ pins
      every up-eval. Added to both prover and verifier.
    - **Soundness chain**: second open binds `open_evals_at_r0h[witness]`
      to the commitment; mp `verify_subclaim` ties `open_evals_at_r0h`
      (all cols, public via local recompute / virtual via derive / witness
      via open) back to `evals_at_rstar_h`; the up-eval absorb + SZ pins
      each `evals_at_rstar_h[j]`; the binding check then ties the trusted
      `parent_evals` (hence the bit-slices, via the recombination + the
      zerocheck) to the committed columns. Works for both witness and
      public distinct columns.
    - **Tests**: new e2e `prove_then_verify_f2_full_with_hadamard_roundtrips`
      (HadF2Uair, 3-col `W=U⊙V`, driven through the `_with_hadamard`
      entries): honest round-trip accepts (and asserts the binding equality
      holds by construction), and the reachable tampers are rejected —
      flipped `W` → zerocheck, swapped parent-eval → recombination, dropped
      second open → `MissingHadamardDischarge`, tampered `r*_H` eval →
      second mp. **38 protocol lib tests pass** (`--features parallel`);
      `f2_prove.rs` clippy-clean. The `HadamardParentEvalBindingMismatch`
      variant is defence-in-depth (only a malicious prover shipping
      fake bit-slice MLEs decoupled from the commit can reach it — the
      recombination + zerocheck over-determine the slices, so no single
      tamper of an honest proof reaches it without tripping an earlier
      gate).
    - **Helper extracted**: `recompute_public_col_evals_at::<D>` — the
      public-col α-project + MLE-eval recompute, now shared by all three
      verifier points (`r*`, `r_0`, `r_0^H`).
  - **SOUND DISCHARGE REWORKED → Approach B (two-point multipoint-eval;
    working tree on `f2-clean`)**: Approach A was the user-identified
    "wrong turn" — it opened the witness slice **twice** (a separate
    `r*_H` mp+open) and leaned on a `parent_eval == evals_at_rstar_h`
    binding check. Approach B folds the Hadamard claims into the **main**
    mp instead, exactly as requested: each AND pair's claim
    `MLE[v](r*_H)` enters the *single* multipoint-eval as a **pointed
    shift** and comes out as a claim on `MLE[v]` at the shared `r_0`,
    unifying every `MLE[v]` evaluation across points under **one** open.
    - **New piop primitive** (`piop/src/multipoint_eval.rs`, additive —
      the integer mp is untouched): `PointedShiftClaim { point, shift,
      source_col }` + `prove/verify_as_subprotocol_with_pointed_shifts`
      + `verify_subclaim_pointed`. A pointed shift carries its **own**
      point (here `r*_H`) and a `shift` Δ; `shift = 0` ⇒ the shift
      predicate is `eq` ⇒ a plain point-claim, so Δ=0 AND pairs fold as
      point claims and Δ≠0 pairs (row-shifted operands) fold via the
      shift predicate at `r*_H` — **closing the AND row-shift soundness**
      (ledger Issue 1) for the AND relations in the same pass. Unit test
      `two_point_pointed_shifts_roundtrip` passes (17 piop mp tests green).
    - **`F2FullProof` slimmed**: the four `hadamard_*` fields
      (`hadamard_multipoint_eval`, `hadamard_evals_at_rstar_h`,
      `hadamard_open_evals_at_r0h`, `hadamard_open`) are **removed**. The
      existing `multipoint_eval` + `open` + `open_evals_at_r_0` now carry
      the folded `r*_H` claims; the prover feeds `uair.hadamard_pair_evals`
      as the pointed shifts' `down_evals`, the verifier rebuilds the same
      `PointedShiftClaim`s from `subclaim.hadamard_pairs`/`hadamard_rstar`
      and checks them with `verify_subclaim_pointed(open_evals_at_r_0,
      pointed_shift_sources)`. Seven now-dead `F2FullVerifyError` variants
      removed (`MissingHadamardDischarge`, `Hadamard{EvalsAtRstarH,OpenEvals}
      LengthMismatch`, `Hadamard{Public,Virtual}ColumnEvalMismatchAtR0H`,
      `HadamardParentEval{CountMismatch,BindingMismatch}`).
    - **Why sound without the binding check**: the AND pair-evals
      (`hadamard_pair_evals`) are still the recombination's inputs (corrupt
      → `HadamardRecombination`) AND now the mp's down-terms folded into the
      single `r_0` open — SZ over the mp's γ pins them to the committed
      columns directly, so the separate A-style binding equality is no
      longer needed. Adders stay trusted (their computed-β operands are
      virtual/non-column, so they can't fold as pointed shifts).
    - **Tests**: `prove_then_verify_f2_full_with_hadamard_roundtrips`
      rewritten for B (tampers: flipped `W` → zerocheck, swapped
      parent-eval → recombination, tampered `open_evals_at_r_0` → the
      two-point mp subclaim check); `_with_operand_hadamards_roundtrips`'s
      honest path now exercises the Δ≠0 pointed shifts end-to-end. All 19
      protocol `f2_prove` tests + the SHA-256 e2e tests pass
      (`--features parallel`); `multipoint_eval.rs`/`f2_prove.rs`
      clippy-clean (the only `arithmetic_side_effects` hits are pre-existing
      in the `poly` GF128/GF192 code, out of scope).
  - **Identified, not done (optimisations on the sound discharge)**:
    (a) ~~the second mp+open opens the witness slice a **second** time~~
    **DONE via Approach B above** — folded into one two-point mp + one
    open. (b) The
    `prove_f2_full_impl` / `verify_f2_full_impl` Hadamard blocks duplicate
    the main mp+open block (~60 lines each) — a `*_mp_and_open_at_point`
    helper would DRY it but threads ~13 params; left inline for review
    clarity. (c) `{0,1}`-slice round-1 fast path for the degree-3 zerocheck.
  - **OPERAND MACHINERY SHIPPED (Phase B — row-shift / XOR / complement,
    working tree on `f2-clean`)**: `F2HadamardSpec` generalised from three
    plain column indices to three **operands**. An operand is the bitwise
    XOR of `(col, ↓Δ row-shift)` terms, optionally bitwise-complemented
    (`1−`). This covers the SHA AND relations C12–C14 (Appendix A):
    `W_E ⊙ W_E^↓1`, `(1−W_E) ⊙ W_E^↓2`, and the Maj combo
    `(W_A + W_A^↓2) ⊙ (W_A^↓1 + W_A^↓2) = (W_MAJ + W_A^↓2)`.
    - **`+`/`1−` are F_2[X] addition (bitwise XOR), NOT F-linear sums** —
      so operand bit-slices are built by XOR-ing source bits at the bit
      level (`build_operand_slices`, a `{0,1}`-valued F MLE), *not* via the
      integer prover's `build_virtual_booleanity_mles` (which does
      F-addition and would give `2` for `1+1`). **Correction to plan §5.1**,
      which suggested reusing `build_virtual_booleanity_mles` — wrong for
      AND operands. The zerocheck itself
      (`piop/src/lookup/hadamard.rs`) is operand-agnostic: it just needs `D`
      slices per operand, laid out `U_0,V_0,W_0,U_1,…`.
    - **Discharge** (`f2_hadamard.rs`): because ψ_α is an F_2-algebra hom,
      `ψ_α(A ⊕ B) = ψ_α(A) + ψ_α(B)`, so each operand's parent eval at
      `r*_H` derives *linearly* from the per-`(col, Δ)` evals plus the
      complement constant `Σ_{b<D} α^b`. The prover ships one trusted eval
      per distinct `(col, Δ)` **pair** (`F2Proof.hadamard_pair_evals`,
      renamed from `hadamard_parent_evals`); the verifier derives operand
      parents (`derive_operand_parents`) and recombines via
      `verify_bit_decomposition_consistency`. `F2VerifierSubclaim`'s
      `hadamard_distinct: Vec<usize>` became `hadamard_pairs: Vec<(usize,
      usize)>`.
    - **Soundness**: the binding check (`verify_f2_full_impl`) ties each
      `Δ = 0` pair eval to the bound `hadamard_evals_at_rstar_h[col]` (the
      second-mp discharge), so `Δ = 0`-only operands — including the
      complemented `(1−col)` since the `Σα^b` const is deterministic — are
      **fully sound**. `Δ ≠ 0` pair evals are prover-supplied / **trusted**
      (MLE eval doesn't commute with row shifts; the row-shift discharge is
      shared with Issue 1, honest-prover-first per plan §6). So among
      C12–C14, the complement-only sub-expressions are sound and the
      shifted ones are honest-prover.
    - **Tests**: `f2_hadamard.rs` unit tests (plain / row-shift / complement
      / Maj-combo round-trips + corrupt-W rejection) and a full-flow e2e
      `prove_then_verify_f2_full_with_operand_hadamards_roundtrips`
      (5-col `HadOpF2Uair`, three relations one per operand kind, through
      `prove/verify_f2_full_with_hadamard`): honest accept + flipped-W
      reject. **43 protocol lib tests pass**; clippy-clean on the changed
      files.
  - **Still open after this**: the **X· bit-shift** in operands (a bit-index
    reindex `b ↦ b+1`, drop bit D-1, zero bit 0 — needed for the adder
    operands `x + X·c`; its parent derivation is `α·(ψ_α(c) − α^{D-1}·c_{D-1})`,
    not yet implemented), the **`W_β` carry column** for the 13 adder
    relations, then **register the 16 SHA relations on `sha256_f2`**
    (C12–C14 first — they need no `W_β` and their columns are already
    committed witness cols, so the operand machinery above can drive them;
    then the 13 adders). And the **sound row-shift discharge** (Issue 1) to
    close the `Δ ≠ 0` trust gap.
  - **C12/C13/C14 WIRED & TESTED on real `sha256_f2`**
    (`sha256_f2_and_hadamards_roundtrips`, working tree): the **three SHA
    AND relations** discharge e2e — 3 of the 16. **Row-shift direction
    (resolved):** the codebase `↓Δ` is `row i → col[i+Δ]` (operand
    `row_shift`, `build_shifted_bit_slice_mles`, `ShiftSpec`
    `uair/src/lib.rs:44`). The SHA fills are written `i−Δ`
    (`u_ef[t]=e[t]&e[t−1]`, `u_neg_e_g[t]=(¬e[t])&e[t−2]`,
    `maj[t]=Maj(a[t],a[t−1],a[t−2])`, `sha256_f2.rs:963`), so **Appendix A
    is `i−Δ` and was re-expressed `i+Δ` by shifting the result column up**:
    C12 `u:W_E^↓1,v:W_E,w:W_UEF^↓1`; C13 `u:¬(W_E^↓2),v:W_E,w:W_UNEG_E_G^↓2`;
    C14 `u:W_A^↓2⊕W_A, v:W_A^↓1⊕W_A, w:W_MAJ^↓2⊕W_A` (Binius
    `(x⊕z)(y⊕z)=Maj⊕z`). **Boundary subtlety RESOLVED**: C13's complement
    (`¬(W_E^↓2)`=all-ones at the zero-padded tail) and C14's combo
    (`W_A^↓2⊕W_A`=`W_A` at the tail) only diverge at rows `t≥n−2`, which lie
    in the **zero slack** (`generate_random_trace` zero-inits, fills only
    active rows), so the un-shifted term `W_E[t]`/`W_A[t]`=0 there and the
    products vanish → honest sum 0. `(W_E,0)`/`(W_A,0)` bound; `Δ≠0` pairs
    trusted (row-shift gap). Lesson logged: **always check the honest
    zerocheck sum is 0 to confirm a registration's boundary handling.**
  - **ADDER OPERAND MACHINERY SHIPPED (`F2AdderSpec`, working tree)**: the
    13 adder relations C5–C11 are `(x + X·c) ⊙ (y + X·c) = c + X·c` (Binius
    carry identity). `F2AdderSpec { t, x, y: F2OperandTerm }` +
    `build_adder_operand_columns` build `U,V,W` per row from the three
    committed columns, with the carry `c = SHR¹(t⊕x⊕y)` on bits `0..D−2` and
    `c[D−1] = β = Maj(x_{D−1}, y_{D−1}, (t⊕x⊕y)_{D−1})` **computed** (the
    overflow carry — so **no committed `W_β` is needed for completeness**;
    `W_β`/X·-bit-shift were the plan's route but computing `β` is simpler).
    Because `c` is *defined* from `t⊕x⊕y`, the zerocheck genuinely verifies
    `t = x+y` (the carry recurrence), not a tautology — confirmed by the
    synthetic tests `adder_round_trips` (honest accept) + `adder_rejects_wrong_sum`.
    Threaded through prove/verify (`adder_specs` param on the phase, the
    group fns, `_full_impl`, `_full_with_hadamard`); `F2Proof` gained
    `hadamard_adder_parents` (the operands' **trusted** parents at `r*_H` —
    the carry `β` + bit-shifts break the pair-eval algebra, so adders are
    honest-prover, no Δ=0 binding).
  - **ROW-SELECTIVITY SOLVED — ALL 13 ADDERS WIRED e2e (3+13 = 16/16 SHA
    relations done)** (`sha256_f2_all_adders_with_selectors_roundtrips`).
    The adders are **row-selective** — `target = x + y` holds only on each
    chain/anchor's active rows; the trace zeroes the S-columns (targets)
    off-chain while the inputs stay non-zero, and the IC's LSB
    `assert_in_ideal(target+x+y+κ)` still holds there only via the `κ` (PA_C)
    compensator. A uniform zerocheck would reject (documented by
    `sha256_f2_adders_need_row_selector`, which still asserts the *un*masked
    rejection). **Fix = `F2AdderSpec::active_rows`** (per-row mask): the
    builder zeroes `U,V,W` off-active so the per-row term is `0⊙0−0 = 0`.
    Since adder parents are trusted (honest-prover), the zeroing rides the
    same trust — **no verifier-side selector / zerocheck change needed**
    (simpler than the indicator-MLE route). The masks are a public
    structural property of the SHA layout — per 68-row block
    (`start = i·ROWS_PER_COMP`): C5a/b/c `[start, start+52)`;
    C6a–e/C8/C9 `[start, start+64)` (anchored where the unshifted S-column
    operand lives — even C6e/C8 whose target is row-shifted); C7
    `[start+3, start+67)` (its target `W_T2` is unshifted); C10/C11
    `[start+64, start+68)` (digest feed-forward). The 13 i+Δ specs + masks
    are in the test. **Lesson**: anchor each adder's mask at the row where
    its *unshifted* operand (the S-column) is materialised, NOT at the
    target's storage row (the C6e off-by-shift that the bisect caught).

### Production integration — the 16 relations derive from a layout builder
- **Shipped**: `protocol::f2_hadamard::Sha256F2HadamardLayout` (struct of the
  ~20 SHA column indices + `ROWS_PER_COMP`/`ROUNDS_PER_COMP`/`num_compressions`
  + `num_vars`) with `and_specs()` / `adder_specs()` / `relations()`. The
  relation **topology** (the i−Δ→i+Δ re-expression, the operand wirings, the
  active-row mask ranges derived from `rows_per_comp`/`rounds_per_comp`)
  lives here once, in the SHA-specific `f2_hadamard` module; the caller fills
  the column indices from the SHA UAIR's `cols` (a thin glue) so there's **no
  reverse `test-uair → protocol` dependency** and no duplicated indices.
- **Why**: the 16 relations + masks were previously hand-written, duplicated
  across the AND-only and adder-only e2e tests. Now there's a single source
  of truth; `sha256_f2_full_hadamard_roundtrips` discharges **all 16** via
  one `prove/verify_f2_full_with_hadamard` call built from
  `layout.relations()`, and the focused AND/adder tests + the
  row-selector-rejection test all consume the builder (masks cleared for the
  latter). 50 protocol lib tests pass; clippy-clean.

### `f2_sha256` bench was stale — fixed (and a nvars=9 baseline)
- **Found while running the bench** (`RUSTFLAGS="-C target-cpu=native"
  cargo bench --features "parallel simd unchecked" --bench f2_sha256 --
  "Steps"`): the bench didn't compile (5 `prove_f2_uair_with_groups` call
  sites still used the pre-`hadamard_specs` signature — never updated since
  `13c169e`, because benches aren't in the default `cargo build`/`test`),
  and the `2-VerifyOpen` micro-bench **panicked** (`EvalConsistency`): it
  verified the open at `r*`, but the full proof opens at the multipoint-eval
  output point `r_0`. Fixed both: added the `&[]` hadamard/adder args, and
  made `2-VerifyOpen` replicate the mp phase (mp `verify_as_subprotocol` →
  absorb `open_evals_at_r_0` → subclaim at `r_0`) before timing the open.
- **Sibling benches `f2_sha256_rs.rs` + `f2_blake3.rs` had the *identical*
  pair of drifts** (surfaced later, building with `--all-targets`): the same
  5 stale `prove_f2_uair_with_groups` call sites (missing `&[]` hadamard +
  `&[]` adder args since `13c169e`) **and** the same `2-VerifyOpen` panic
  (opened at `r*`, proof opens at `r_0`). Fixed identically — the mp-phase
  replication is generic (all three pass `&[]` XOR-virtual-bp specs, so
  `open_evals_at_r_0` is exactly the primary col evals); added the two
  imports (`F2VerifierSubclaim`, `multipoint_eval::MultipointEval`) the
  siblings lacked. All three `2-VerifyOpen` micro-benches now run clean
  (smoke-tested at nvars=9). These three benches aren't in the default build,
  so any signature change to `prove_f2_uair_with_groups` / `F2FullProof`
  must remember to touch all three.
- **Baseline (nvars=9, 7-compression / 512-row fixture, M-series CPU,
  `parallel simd unchecked`, no `metal_gpu`)** — prove: 1-Commit ≈ 272 µs,
  2-UAIR ≈ 953 µs, 3-Open ≈ 660 µs (≈ 1.89 ms total); verify:
  1-VerifyUAIR ≈ 113 µs, 2-VerifyOpen ≈ 702 µs (≈ 0.82 ms total). The
  Hadamard/adder code is a **no-op on this path** (the bench registers no
  relations), so these reflect the unchanged baseline. NB Criterion's
  `change:` deltas are vs a stale saved baseline (and the Commit
  "regression" is CPU-vs-`metal_gpu`-GPU) — read absolute numbers, not the
  deltas. The `e2e` group (full prove/verify, nvars sweep to 22) and `micro`
  group were not run here.

### Sound discharge for K-virtual MLE evaluations at r* (Issue 1)
- **What**: currently the 7 K-virtual cols' MLE evaluations at
  `r*` are extracted from the sumcheck transcript and **trusted**
  — no derivation from the source recipe, no PCS-side
  reconstruction. The constraint system relies on the SHA-256
  boundary check (publicly-fixed final digest) to make this
  honest-prover-only design effectively sound.
- **What's needed**: a **row-shift** discharge mechanism for the
  cross-row shifts inside each K source recipe (e.g. `wit(cols::W_W, 16)`,
  `wit(cols::W_SIG0, 1)`). Note: MLE evaluation at a fixed point does
  *not* commute with row shifts, so the verifier **cannot** derive
  `MLE(SHIFT_row^k(v))(r*)` directly from `MLE(v)(r*)`. The mechanism
  has to relate the two via something other than evaluation-at-the-
  same-point — exact shape is left to the design owner; placeholder
  text deliberately vague here. **The per-cell `ShiftR^1` that wraps
  the K recipe is already handled** by `F2BitOpVirtualSpec` and
  `apply_bit_op_u32` (shipped, with the in-tree canary above); the
  K-discharge work just needs the row-shift piece on top.
- **Where**: `protocol/src/f2_prove.rs::verify_f2_full_with_bit_ops`
  would gain a "derive K col evals from source col evals + spec"
  step, analogous to how `F2VirtualBpSpec` (XOR) virtuals are
  derived today. The recipe is already captured in
  `F2KVirtualSpec` (sources + row shifts + final BitOp), waiting
  for the discharge mechanism to land.
- **Estimated effort**: depends on whether the shift-predicate
  machinery is already in `piop/src/shift_predicate.rs` and
  usable as-is. The math sketch lives in
  `protocol/src/f2_gkr_plan.md` rev 2.

### Open-FULL micro overcounts vs in-situ
- **Observation**: `Open-FULL` (520 ms) is higher than the
  in-situ `prove_f2_open` measurement (430 ms) at nvars=22.
- **Suspected cause**: Open-FULL uses `iter_batched` with a
  setup that re-builds `(hint, subclaim, transcript)` each iter
  by re-running commit + UAIR. Some allocation/cache state in
  setup leaks into the timed window.
- **Not blocking** but means the wrap-up accounting is slightly
  off in the other direction; don't read Open-FULL as the true
  open cost.
- **Fix idea**: maybe run setup once outside the batch and clone
  the inputs per-iter? Or accept the inflation and document it.

### `prove_f2_open` per-column loop is over the full witness slice (20 cols), not the paired-commit set (9)
- **Where**: `protocol/src/f2_prove.rs:1945-1991` (the
  `per_col_results` loop) and `:2020-2069` (the
  `per_col_combined` loop). Both iterate `num_cols =
  trace_binary_cols.len()` which the caller passes as
  `&trace.binary_poly[num_pub_bin..]` = 20 cols at
  `prove_f2_full*` call sites (`f2_prove.rs:2684`,
  `f2_prove.rs:2796`).
- **Observation**: those 20 cols include the 12 bit-op virtuals
  (`SHR^j(PA_C)` for j ∈ 1..12). Their MLE eval claims at r* are
  ALREADY independent witness claims because the open's job is to
  bind them via γ-batching + Schwartz-Zippel discharge — except the
  γ_g for each virtual col can be folded into the γ for its source
  via the per-cell BitOp algebra (XOR/Rot/SHR all commute with the
  linear γ-fold). At least for the SHIFTR case (which is all 12
  virtuals here), the virtual's contribution to `b_g`, `a_g'`, and
  `combined_row_g` can be derived from the source's
  contribution by applying the same BitOp to the *cells before
  inner-producting with q1*. In other words: open does NOT need to
  re-lift+inner-product the virtual cell stream — it can derive
  the virtual's per-row partial from the source's per-row partial
  after the cell-level `apply_bit_op_u32`.
- **Expected impact**: drop the per-col loop's outer iteration count
  from 20 → 8 (1 primary witness per source for each of the 5
  non-virtual witness cols plus 3 source cols that have virtuals).
  Estimated 30-50% of `prove_f2_open` wall time given the per-col
  cost dominates (each col is a full `O(num_rows × row_len)`
  cell-lift + multiply pass). Verify-side mirror needs the same
  shape change. **Effort: M**, with a careful soundness pass on
  what "fold γ into the source via BitOp" means at the wide-poly
  product layer (currently the lift is `bp_to_f2_poly_1` then
  multiply by γ as a `BinaryF2Poly<3>` — the BitOp's
  bit-permutation needs to land *before* the lift, not after).

### Multipoint-eval as degenerate γ-rerandomization (down_evals=[])
- **Where**: `protocol/src/f2_prove.rs:2628-2638` and
  `f2_prove.rs:2758-2768` — both call sites pass `down_evals=&[]`
  and `shifts=&[]`.
- **What it's doing today**: with empty shifts/downs, the
  `MultipointEval::prove_as_subprotocol`
  (`piop/src/multipoint_eval.rs:159-180`) reduces to a degree-2
  sumcheck of `eq(b, r*) · Σ_j γ_j · trace_j(b)`. That's
  essentially a γ-rerandomization of the per-column claims from
  r* to a fresh r_0, with the full sumcheck cost (`num_vars`
  rounds of degree-2 messages over GF(2^192)). The combined
  evaluation at r_0 is then sent through the open path.
- **Why it isn't free overhead**: the rerandomization moves the
  open's `r*` to a new `r_0` so the prover can't tailor the open
  to the IC sumcheck's exact point. Without it, the open's
  Schwartz-Zippel batching is restricted to the IC point — fine
  for the current single-open structure but blocks any future
  multi-point lookup batching.
- **Possible optimization**: when `shifts.is_empty()`, the
  multipoint-eval phase could fall back to a leaner γ-fold-only
  protocol that skips the eq-table build and the sumcheck (just
  draws γ and the verifier checks the same fold via a single
  GF(2^192) batch identity). The current shape pays the full
  sumcheck cost for what is effectively a free rerandomization.
- **Expected impact**: small but nonzero — at nvars=20 the inner
  sumcheck is ~25 rounds × O(2^num_vars / 2^round) over
  GF(2^192). The fold-only fallback would save the per-round
  message bytes and the eq-table eval. Estimated 2-4% Prove e2e.
  **Effort: M** (a new sub-protocol shape + verifier mirror).
  Lower priority than the per-col open loop above.

### Sumcheck data-clone elimination
- **Where**: after the two clones we eliminated in commit
  `22f618a`, there's still room. The `projected_trace` at
  `f2_prove.rs:940` builds a length-`num_total_cols` `Vec` of
  `DenseMultilinearExtension<BinaryFieldGF192>`, each ~96 MB at
  nvars=22 — totalling ~3.4 GB of intermediate allocations per
  prove.
- **Already-applied mitigation**: line 1025-1056 zero-copy
  reinterprets each col's `Vec<BinaryFieldGF192>` into
  `Vec<<BinaryFieldGF192 as Field>::Inner>` via `ManuallyDrop` +
  `from_raw_parts`. So at least the per-col-evaluation pass
  doesn't add another copy.
- **Possible next step**: the projection loop itself could
  process columns in chunks and discard the per-col vec as soon
  as it's been consumed by the sumcheck, avoiding the full
  ~3.4 GB live-at-once footprint.

### Update `Open-h-FinalAssembly` + `UAIR-e-WrapUp` per-step benches
- Currently `UAIR-FULL` and `Open-FULL` exist (commit `22f618a`).
  Their delta vs sub-step sums identifies wrap-up regions but
  doesn't decompose them further.
- A pair of `*-WrapUp` benches that time JUST the assembly /
  transcript-absorb / drop chunks (subtracting the kernels) would
  let us isolate each contributor without re-instrumenting the
  prover.
- Low priority — the residual wrap-up at nvars=22 is now ~40 ms
  for Open, ~270 ms for UAIR (post-clone-removal). Not currently
  the largest pain.

### Generalise `F2BitOpVirtualSpec` to a "multi-source XOR with per-source per-cell BitOp" spec (chained-Binius cols)
- **What**: introduce a new virtual-spec type whose evaluation rule
  is `target[i] = XOR_k op_k(source_k[i])` for a list of
  `(source_col, BitOp)` pairs, generalising the current per-cell
  `F2BitOpVirtualSpec` (which is the single-source case). Then
  declare the 6 chained-Binius intermediate cols (`W_W_S1`,
  `W_W_S2`, `W_T1_S1..S4`, in `test-uair/src/sha256_f2.rs:1068-1073`,
  protocol indices `cols::W_W_S1` etc) as virtuals under this spec.
- **Why it works on this branch (vs the K-virt machinery on
  `claude/gkr-virtual-cols`)**: every chained-Binius intermediate
  `s_k` is bit-by-bit determined by its predecessor + XOR
  contributions + a single LSB carry bit pulled from `PA_C`, **but
  the constraint `(s_k + ...)[0] = 0 (mod X^32 − 1)` already locks
  `s_k` bit 0**. The remaining bits 1..31 of `s_k` are NOT yet
  AIR-pinned (that's the Hadamard step 5 work in the entry above).
  Concretely: the LHS of C5a in `test-uair/src/sha256_f2.rs:500-503`
  forces `W_W_S1 ≡ W_W + W_SIG0^{↓1} + κ_{W,1} (mod X^32)`, but
  `mod X^32 − 1` is also `mod X^32` only at bit 0 — bits 1..31 are
  free in the current AIR. So **declaring `W_W_S1` virtual today is
  unsound at the AIR layer** until Hadamard discharge (step 5) lands.
- **Why this is different from the K-virt Issue 1 hole**: this spec
  is purely per-cell (no row shifts) — it derives `target[i]` from
  the *same row* of source cols. MLE evaluation at a fixed `r*`
  commutes with per-cell BitOp + XOR, so the verifier can recompute
  `MLE(target)(r*) = XOR_k op_k(MLE(source_k)(r*))` with no
  row-shift discharge needed. Soundness for the eval-at-r* step is
  free. The blocker is only the AIR-layer bit-1..31 pinning, which
  is the same Hadamard work the K-virt rollback also waits on.
- **Expected impact if Hadamard lands**: drops `paired_batch` from 9
  → 6 (excluding 6 cols from commit halves the leaf-bytes input to
  Blake3, and shrinks paired storage by ~1/3). At nvars=20 the
  per-leaf-byte work and the GPU slab size shrink ~33%, plausibly
  buying back ~10-15% of the Commit phase. The open's per-column
  loop in `prove_f2_open` still runs over the full witness slice
  (20 cols), so this doesn't directly speed open; that would need a
  separate "open-virtuals" change.
- **Effort**: **L** as long as Hadamard discharge is the gate. The
  spec type itself is a small addition; the BlockerN is the
  unrelated Hadamard step 5 work. Calling it out so future-me
  doesn't try the spec change in isolation and produce an unsound
  prover. **Do not attempt without Hadamard.**

### Investigate Metal `MTLCommandQueue` reuse / `MTLCommandBuffer` pooling
- **Why**: the GPU dispatch warmup test (rejected above) shows the
  per-dispatch cost isn't variable — but each commit builds a fresh
  `CommandBuffer` and encoder. Maybe pooling these would shave a
  few ms per commit. Not validated.

### Parallelise Merkle leaf hashing within a leaf? (rejected — kept as note)
- Per-leaf is ~3 KB at SHA-256 F_2 shape (3 Blake3 chunks). Within-
  leaf `update_rayon` is pure overhead at that size. Cross-leaf
  parallelism via `cfg_iter!` already exists.
- Would only matter for protocols with much larger leaves
  (per-leaf > ~256 KB).

---

## Open questions / hypotheses to test

### ✅ RESOLVED (2026-06-14): merged verify `MergedClaimedSumMismatch` when filler rows == 0 (ν ≡ 2 mod 8)
- **Fix**: `cols::num_compressions` now reserves `4 + 2` rows (`−6`, was `−4`):
  the 4-row output anchor **plus 2 trailing zero-guard rows** so the unmasked
  `↓2` Ch/Maj AND relations read zeros (not the anchor) at the tail. Only
  ν ≡ 2 mod 8 shapes change (drop 1 compression); nv9/16/17/19/20/21/22/23
  counts are identical. Regression test
  `sha256_f2_merged_exact_tiling_verifies` (nv9/10/11). Verified: nv10+nv18
  now verify; full f2 suite 46/0, test-uair 27/0. See the Shipped entry
  "Fix merged exact-tiling verify bug". Root-cause analysis kept below.
- **Symptom**: `verify_f2_full_with_merged_hadamard` returns
  `Uair(MergedClaimedSumMismatch { claimed, reconstructed })` — prove always
  succeeds, but the emitted proof fails its own verify. **Deterministic by
  *shape*** (different random claimed/reconstructed values each run — RNG is
  entropy-seeded — but it fails 100% of the time at the affected ν).
- **Exact trigger**: fails iff `(2^ν − 4) % 68 == 0`, i.e. the SHA compressions
  *exactly tile* the trace with **zero filler/padding rows**.
  `num_compressions(ν) = (2^ν−4)/68`; this is integer-with-zero-remainder iff
  **ν ≡ 2 (mod 8)** → ν = 10, 18, 26, … Confirmed failing: **nv10** (15 comp)
  and **nv18** (3855 comp). Confirmed *passing*: nv9 (deployed), nv16, nv17,
  nv19, nv20, nv21, nv22, nv23 (all have ≥1 filler row). Independently
  corroborated by the user's metal_gpu sweep, which silently produced **no
  verify line at nv10 and nv18** (same failure, swallowed).
- **Likely culprit**: the merged claimed-sum *reconstruction* on the verifier
  (the `r0_at` / `merged_eq_point` hybrid-eq path, and/or the zero-padding
  witness-forest term) degenerates when the padding-row count is exactly zero —
  a sum/term that normally ranges over ≥1 filler row becomes empty, so the
  verifier's reconstruction diverges from the prover's claimed sum. The prover
  doesn't notice (it computes the claim directly); only the verifier's
  independent reconstruction hits the empty-range edge case.
- **Impact**: latent — the **deployed shape (nv9) is unaffected** (it has 32
  filler rows). But it is a real correctness gap on the merged path at the
  exact-tiling shapes, and it means the merged arm is not safe to deploy at an
  arbitrary ν without the fix.
- **Repro**: `AB_NVARS=10 AB_MERGED_ONLY=1 AB_OPENINGS=987 <test-bin>
  sha256_f2_merged_ab_timing --ignored --nocapture` (or nv18). **Fix plan**:
  audit the verifier's claimed-sum reconstruction for an implicit
  "≥1 padding row" assumption (empty-product → should be the multiplicative
  identity, empty-sum → 0); add a regression test that sweeps ν over a full
  residue class mod 8 (must include a ν ≡ 2 mod 8) at the merged arm.

### Why is `Commit-AfterPrevProve` still ~47 ms slower than `Commit` (isolated)?
- After the slab-reuse fix the gap dropped from 215 ms to ~47 ms
  (566 vs 519 ms). The big suspect — the 768 MB slab alloc — was
  confirmed and fixed. The residual ~47 ms is probably the
  per-row encode rayon parallelism re-spawning worker thread
  state after the pool sat idle during the prior UAIR/Open, or
  cold L2 cache on the paired-witness MLE evaluations re-read.
- Low priority — at this point we're chasing noise-level variance.

### Does the bench fixture's `num_rows = 8` choice make sense for measurement?
- `setup_prover` hard-codes `num_rows = 8` independent of `nvars`,
  so the codeword/encoder dimensions grow disproportionately with
  `nvars` (`row_len = poly_size / 8 = 2^(nvars-3)`).
- This means SHA-256 F_2's commit has many thin leaves (~3 KB ×
  262k at nvars=22) — atypical for protocols with bigger
  cells/cols.
- Bench numbers reflect this specific choice. Consider whether
  a 2-row variant (matching the original protocol-doc shape) is
  also worth keeping for cross-protocol comparison.

### Can we get sound K-virtual MLE evaluation via the `f2_native_ic` lane?
- The F_2-native IC stage (`f2_native_ic::F2NativeIc::prove_linear`)
  already evaluates degree-1 constraint expressions in bit-pack
  form. The K column's source recipe (`ShiftR^1 of XOR of
  row-shifted sources`) is degree-1. Maybe there's a clean way
  to fold the K derivation into the IC stage's per-cell eval, so
  it lands on the sumcheck-extracted column claims naturally.

---

## How to use this doc

When you investigate any optimization, design alternative, or hypothesis
on the F_2 SHA-256 prover path — even if you don't change code —
record it here. Concretely:

1. **If you ship a change**: add a "Shipped work" entry with the
   commit SHA, what changed, why, and the measured impact.
2. **If you try something and it doesn't work**: add an
   "Investigated, didn't help" entry with the hypothesis, the
   test, the result, and (if known) the real culprit.
3. **If you identify an optimization but defer it**: add an
   "Identified but not implemented" entry with where the
   opportunity lives, the approach, and the expected gain.
4. **If you have a hypothesis but no time to test**: add it to
   "Open questions" with a concrete test plan.

This avoids the next agent (or you, in 3 weeks) repeating dead
ends or re-discovering the same hot-paths.
