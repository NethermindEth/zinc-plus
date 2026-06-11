# Zip+ basefold variant — design ledger

Branch: `zip-plus-basefold` (off `main-beta`). Status: milestones 1-2 done
(interactive IOPP + Merkle/Fiat-Shamir compilation with spot-check queries).

This branch implements the **limbwise-folding (BaseFold-style) variant of Zip+**:
a hash-based IOPP over the integers with polylogarithmic verification, following
the draft paper *"Zip++: Hash-Based IOPPs over the Integers with Polylogarithmic
Verification via Limbwise Folding"* (working draft in
`~/zinc- whir idea/zip-plus-plus-draft/`, Albert, June 2026). Read that draft
first; this file maps it onto the codebase and records implementation decisions.

## Protocol recap (one paragraph)

FRI-style folding over Z fails because each fold adds ~lambda bits to entries.
Fix: keep the round-r state as k_r *limb vectors* with balanced base-2^p digits
(virtual word = sum_j 2^{p(j-1)} v_j); each round splits every limb into
even/odd halves, folds **all** halves with fresh independent lambda-bit
challenges (the radix weights never enter the fold), and re-decomposes the
folded vector into fresh limbs. For p >= lambda + 2 the limb count stabilizes
at k = 2 (arity 2; p >= lambda + s + 1 at arity 2^s). Evaluation claims (which
Zinc+ needs *projected*: values in F_{q0} for the transcript-sampled prime q0)
are tracked **linearly**: per-round half-claims checked with weights
(1 - z_r, z_r), folded claim = alpha-combination, decomposition reflected by
sum_j 2^{p(j-1)} e_j = e'; final limbs sent in clear and checked directly. No
sumcheck. Consistency across rounds is checked at spot positions via the
**lifted butterfly**: c_j = A + tau_j B, c_{j+n/2} = A - tau_j B with
tau_j = centered_lift(omega^j), so the verifier derives tail-code symbols from
two openings. The codes must form a full-depth radix-2 IPRS *chain* (over Z the
FFT depth is part of the code; fold rounds <= depth; the unfolded tail is sent
in clear).

## Module layout

```
zip-plus/src/basefold/
  mod.rs       module doc + error type
  limbs.rs     balanced base-2^p decomposition / recombination over Int<N>
  chain.rs     radix-2 foldable IPRS chain (encode at every level; twiddles)
  iopp.rs      interactive prover + verifier (full oracles, all positions);
               also hosts the shared round/claim/consistency helpers
  compiled.rs  Merkle-compiled prover + verifier: pair-bundled leaves
               (leaf j of round r = (u_t[j], u_t[j+n_{r+1}]) for all limbs t,
               one tree per round), roots absorbed instead of oracles,
               Q spot checks at chained positions j_{r-1} = s mod n_r with
               one leaf opening per round per query; CompiledProof +
               size_bytes accounting
```

`pub mod basefold;` added to `zip-plus/src/lib.rs` — the only touch to
existing files in milestone 1.

## Type decisions

- Reuse `crypto_primitives` fixed-width ints, mirroring `pcs/test_utils.rs`:
  digits/witness `Int<ND>`, codewords `Int<NK>`, consistency-check accumulators
  `Int<NW>` (wide), challenges `Int<NC>` sampled as CHAL_BITS-bit non-negative
  ints. Test instantiation: ND=3 (192b, holds p=130-bit digits), NK=8 (512b),
  NW=8, CHAL_BITS=128.
- Claims live in a generic `F: PrimeField` with runtime config (`MontyField`),
  sampled via `transcript.get_random_field_cfg::<F, Uint<K>, MillerRabin>()` —
  same mechanism as the existing PCS (q0 = random ~256-bit prime in tests).
- Chain twiddles: i64 centered lifts of F65537 roots of unity (same
  `normalize` convention as `code/iprs/pntt`); base case = lifted Vandermonde.
  n | q - 1 caps codeword length at 2^16 with F65537 — fine for tests; larger
  base fields are a later config knob.
- **No rationals anywhere.** The verifier's butterfly inversion
  (A = (c1+c2)/2, B = (c1-c2)/(2 tau)) is never materialized; every
  consistency check is multiplied through by 2*tau_j and verified as an
  integer identity:
    2 tau_j * RHS_recombined  ==  sum_t [ alpha_e_t * tau_j * (c1_t + c2_t)
                                        + alpha_o_t * (c1_t - c2_t) ].
- MLE convention: `DenseMultilinearExtension` pairs index parity with
  `point[0]` (checked in `poly/src/mle/dense.rs::fix_variables`), so the
  even/odd split consumes z coordinates front-to-back — matches the draft.

## Conventions honored

- Clippy: no raw arithmetic in lib paths — `add!`/`mul!`/`sub!` macros or
  scoped, justified `#[allow(clippy::arithmetic_side_effects)]`; no `unwrap`
  in lib code (tests carry a module-level allow like `pcs/test_utils.rs`).
- F_2-ledger rule: this work is on the integer side and adds new modules only;
  if any shared in-scope file (merkle.rs, pcs/phase_commit.rs, ...) gets
  touched, add the `documentation/f2x-sha-todo.md` entry per CLAUDE.md.

## Milestones

1. **done** `limbs.rs` + `chain.rs` + interactive `iopp.rs` (explicit
   oracles, challenges from a transcript, consistency checked at all
   positions), completeness + tamper tests.
2. **done** Merkle per round (leaf = pair-bundled positions {j, j + n_{r+1}},
   all limbs in one leaf) + Fiat–Shamir query phase (j_{r-1} = s mod n_r
   chaining), CompiledProof + size accounting. Toy-scale datapoint:
   ~84 KB at m = 2^8, R = 6, Q = 30 (query openings dominate, as expected;
   real sizing needs m >= 2^16 and the parameter work of milestone 3).
3. Parameters (in progress).
   - **done** `ChainConfigF167772161` (5*2^25+1, two-adicity 25, generator 3):
     codeword lengths up to 2^25; twiddle lifts <= 28 bits.
   - **done** Norm-growth measurement (`norm_growth_report`): at q = 5*2^25+1,
     measured per-level growth is ~26 bits vs the 27.3-bit worst case --- the
     multiplicative twiddle chain dominates and centered-representative
     cancellation does NOT compound (~1.4 bits/level saved; the dramatic
     savings observed at depth 1-2 with radix 8 come from the additive terms,
     not the chain). Depth 10 at 129-bit digits: 387 measured vs 436
     worst-case bits. Consequences: (a) leaf width should be sized from the
     measured profile + margin (~35-40% leaf-byte saving vs worst-case-wide
     ints); (b) prefer the smallest field whose two-adicity covers n
     (F_65537 up to n = 2^16: ~15 bits/level, roughly half the big field's
     growth); (c) the draft paper's hope of a sub-worst-case exponent at
     depth needs revising --- effective gamma ~ (depth+1)(log2(q/2) - 1.4).
   - **done** Depth-dial experiment (`depth_dial_report`, #[ignore]d): proof
     size breakdown + commit/prove/verify times across R at m = 2^12, Q = 100,
     q = 5*2^25+1, 80 B worst-case-wide leaf ints. Measured (release, M-series):
       R= 2: 181 KB, commit 1.08s, prove 1.07s, verify 541ms (dense tail!)
       R= 6: 391 KB, commit  74ms, prove 138ms, verify 3.0ms
       R=12: 678 KB, commit  10ms, prove  20ms, verify 2.0ms (max leaf 435 bits)
     Readings: (a) polylog verifier confirmed (1.6-3 ms at R >= 6; the 541 ms
     at R = 2 is the dense lifted-Vandermonde tail re-encode --- the predicted
     shallow-chain failure mode); (b) with this encoder, DEEP chains are
     cheaper for the prover too (shallow depth pays m*n/2^D for the dense base
     case; full depth is n*D butterflies); (c) at m = 2^12 the query side
     (~5 KB/round/100 queries) dominates and small R wins on size --- the size
     case for depth only opens at larger m, and competitive sizes need the
     levers: arity-8 (~3x fewer rounds), tight leaf widths (~20-38%), path
     dedup across queries, smallest adequate base field. (A Johnson-regime
     query reduction was considered and dropped --- not pursuing.)
   - **done (3b)** Arity-8 folds: `chain8.rs` (true radix-8 lift: one twiddle
     per DFT-block entry, so per-folded-variable norm growth ~9-10 bits vs
     radix-2's ~26 --- measured 240-bit leaves at m = 2^12 full depth vs 435)
     and `arity8.rs` (8-way class split, 8 challenges/limb, 3 z-coords per
     round with Lagrange class weights, 8-coset leaves, p = lambda + 4).
     Verifier derivation solves the transposed system M_j^T z = alpha
     fraction-free per queried position (Bareiss triangularization + exact
     back-substitution; divisions exact since d*z is integral) and checks
     d * recombined_child == sum_t z_t^T c_t --- still no rationals.
     Dial at m = 2^12, Q = 100, 40 B leaves (vs radix-2 full depth:
     678 KB / prove 20 ms / verify 2.0 ms):
       R = 2: tail 2^6 | 164 KB | commit 28ms prove 9.1ms verify 26ms
       R = 3: tail 2^3 | 243 KB | commit 11ms prove 3.6ms verify 40ms
       R = 4: tail 2^0 | 316 KB | commit 13ms prove 3.7ms verify 54ms
     Proof size halved at full depth; prover 5x faster; the verifier regression
     (2 ms -> 26-54 ms) is exactly the lazy per-position Bareiss cost
     (~0.13 ms/solve x Q*R).
   - TODO: precompute the fraction-free LU (or adjugate) of every DFT block
     at setup and keep it as verifier advice --- drops per-proof solve work to
     back-substitution (~2-4 ms verify expected; ~few MB advice at m = 2^12).
   - TODO: tight leaf widths (serialize leaves at measured-width bytes rather
     than the Int<NK> fixed width; 240 measured vs 320 fixed bits at arity 8).
4. Integration into `protocol` for the SHA-ECDSA bench (in progress).
   - **done (4a)** Batch layer: `commit_batch` / `prove_batch` /
     `verify_batch` in `arity8.rs` --- a batch of B columns committed as the
     round-0 "limbs" of one tree; per-column claims `e_{0,j}` ship in the
     proof, the verifier checks `sum_j weights[j] * e_{0,j} == eval_f`, and
     the first fold combines all B columns with fresh challenges
     (batch-then-fold). Matches the protocol's int-lane PCS contract exactly:
     `ZipPlus::prove_f` for the int lane (DEGREE_BOUND = 0, per-poly
     psi-alphas = [1]) proves `sum_j phi_q0(poly_j)~(point) = eval_f` at
     `point = r_0_ext` (num_vars + 2 coords in F, the quartered columns
     concatenated). Tested incl. correlated round-0 lies.
   - **done (4b)** Serialization: `protocol_glue.rs` ---
     `write_arity8_proof` / `read_arity8_proof` over the PCS byte stream
     (stream-only, no FS absorption: prove/verify do their own symmetric
     absorbs; the verifier deserializes first, then replays). All shapes
     derive from (params, batch_size) via the limb schedule. Roundtrip
     tested through the real Pcs{Prover,Verifier}Transcript.
   - **done (4c)** Protocol wiring behind the `basefold-int` feature
     (protocol/Cargo.toml): in `prove_folded_4x_inner` the int lane is
     excluded from the Zip+ batches (`int_witness_for_zip = &[]`), committed
     via `commit_batch` over the arity-8 chain (quartered columns resized
     Int<2> -> Int<3>; the basefold root rides in the int slot of
     `Proof::commitments`, absorbed unchanged by the transcript init), and
     opened after the Zip+ lanes with `prove_batch` at `r_0_ext` with
     all-one weights + `write_arity8_proof`. `verify_folded_4x_inner`
     mirrors: the int slot is zip-empty (`int_zip_batch_size = 0` under the
     feature; the shared-tree call gets a shared-root/empty-batch stand-in
     since `verify_columns_shared` asserts equal roots and sizes its stream
     reads by `batch_size`), then `read_arity8_proof` + `verify_batch`
     against `int_eval_f(ones)` (the int lane's per-poly psi-alphas are
     trivially ONE, FS-silent). Params derived identically on both sides:
     `int_lane_params(r0_ext.len(), IntLc::REPETITION_FACTOR,
     IntZt::NUM_COLUMN_OPENINGS)`, q = 5*2^25+1, R = floor((nv+2)/3),
     widths BF_ND=3 / BF_NK=6 / BF_NW=12 / lambda=128 / p=132.
   - **done (4d)** e2e validation: `cargo bench -p zinc-protocol --features
     "parallel basefold-int" --bench e2e -- "Folded 4x" --test` runs the
     ShaEcdsa folded-4x prove+verify roundtrip (and the N=100 timing
     harness) green. Measured at nvars = 9 (int lane m = 2^11, R = 3,
     Q = 150, rate 1/4, checked arithmetic, parallel):
       baseline (all-Zip+):  proof.zip raw  399 KB, zstd 184 KB;
                             step-7 open 12.7 ms, step-7 verify  59.4 ms
       basefold int lane:    proof.zip raw 1465 KB, zstd 542 KB;
                             step-7 open 10.6 ms, step-7 verify 151.0 ms
     Reading: correctness and the full FS/serialization pipeline validated;
     at this scale the basefold int lane is NOT competitive on size or
     verify time, exactly as the dial predicted --- m = 2^11 with Q = 150
     is deep in the query-dominated regime, the batch round-0 leaves carry
     all B columns' 8-cosets (B*8 values/leaf at 48 B fixed width), and the
     verifier pays the lazy per-position Bareiss solves (~441 of them).
     The integration is the vehicle; the prize remains the big lanes at
     larger m via the X-dimension extension. Levers specific to this lane:
     tight-width leaf serialization (round-0 entries are ~180 bits in 384-bit
     slots; zstd already recovers much of it --- 2.7:1 on the blob), a
     radix-2 round 0 (pairs instead of 8-cosets for the batch round,
     B*2 values/leaf), and the LU-precompute for verify time.
   - Later (4e): binary/arbitrary lanes need the X-dimension-as-variables
     extension (monomial-basis claim rounds with psi-alpha weights
     (1, alpha^{2^i})) --- the big prize, since those lanes dominate the
     SHA-ECDSA witness.

## Open design questions (tracked, not blocking)

- Negative-entry witnesses: balanced digits handle signs; round-0 witness with
  k_0 > 1 limbs deferred (tests use entries < 2^{p-1}, k_0 = 1).
- Soundness-slack bound B* = (p+2)R + p k_0 must be threaded into the outer
  protocol's bitsize accounting when integrating (draft Thm 6.1).
- Query count / proximity regime: unique decoding, production counts
  (Q = 147 at rate 1/4, Q = 100 at rate 1/8). Johnson-regime reduction
  considered and dropped.
