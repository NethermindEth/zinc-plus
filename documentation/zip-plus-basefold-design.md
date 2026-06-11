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
     levers: arity-8 (~3x fewer rounds), Q ~ 50 (Johnson), tight leaf widths
     (~20-38%), path dedup across queries, smallest adequate base field.
   - TODO: arity-8 folds (radix-8 levels + 8x8 lifted-DFT block inversion via
     Bareiss/adjugate with cleared determinant denominators) --- milestone 3b.
   - TODO: tight leaf widths (serialize leaves at measured-width bytes rather
     than the Int<NK> fixed width; needs a variable-width or per-round-width
     leaf format).
4. Integration: `LinearCode`/`ZipTypes`-shaped adapter, batching
   (batch-then-fold round 0 on the interleaved commitment), wiring into
   `protocol` as an alternative opening for the int lane; benches.

## Open design questions (tracked, not blocking)

- Negative-entry witnesses: balanced digits handle signs; round-0 witness with
  k_0 > 1 limbs deferred (tests use entries < 2^{p-1}, k_0 = 1).
- Soundness-slack bound B* = (p+2)R + p k_0 must be threaded into the outer
  protocol's bitsize accounting when integrating (draft Thm 6.1).
- Query count / proximity regime: unique-decoding first; 1.5-Johnson later.
