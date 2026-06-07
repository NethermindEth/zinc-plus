# UAIR Object Model for Production SHA-256

This note records the naming and lifecycle decisions for the generic UAIR
objects we want to use before specializing them to production SHA-256.

## Core Principle

Keep these roles separate:

- `Shape`: static relation metadata and layout.
- `Witness`: the prover's full pre-projection assignment.
- `Instance`: verifier-visible public data plus commitments.
- `FoldedWitness`: prover-private folded state after folding.
- `FoldedInstance`: verifier-visible folded accumulator.

Do not put projected/evaluated data in the initial instance object. Projection
and evaluation happen later after transcript challenges are known.

## Existing e2e.rs Lifecycle

The generic path in `protocol/benches/e2e.rs` works like this:

```rust
let trace = U::generate_random_trace(num_vars, &mut rng);
let sig = U::signature();
let public_trace = trace.public(&sig);
```

The prover receives the full `UairTrace`, splits it with `UairSignature`, commits
only the witness columns, and absorbs the unprojected public columns into the
transcript:

```rust
let public_trace = trace.public(&uair_signature);
let witness_trace = trace.witness(&uair_signature);

commit(witness_trace);
absorb(public_trace);
```

The verifier receives the proof and the unprojected `public_trace`. It absorbs
that same public trace before later projection/evaluation steps.

So the instance's public data is an unprojected public UAIR trace, not a
`ProjectedPublicTrace`.

## UairSignature

`UairSignature` is the layout contract for a UAIR. It defines:

- total column counts for `binary_poly`, `arbitrary_poly`, and `int`
- public column prefix counts
- witness column suffix counts
- shifted columns
- virtual columns
- lookup and booleanity metadata

`UairTrace::public(sig)` and `UairTrace::witness(sig)` use this signature to
split a full trace into public and witness subtraces.

## Generic Objects

`UairShape` is useful as a value-level handle for the static UAIR relation plus
the trace length.

```rust
pub struct UairShape<U: Uair> {
    pub num_vars: usize,
    pub signature: UairSignature,
    _marker: PhantomData<U>,
}
```

`num_vars` should stay on the shape or protocol input. It is the log trace
length, so the row domain has size `1 << num_vars`. The prover and verifier use
it for MLE sizes, sumcheck rounds, public-structure checks, and PCS parameters.

Do not add `shape_digest` in the first pass. A shape digest may be useful later
for serialization or cached transcript binding, but it should be derived from
the shape rather than treated as fundamental state.

The witness is the full pre-projection prover assignment. It includes public
columns and private columns because `UairTrace` itself stores both; public
columns are the prefix determined by `UairSignature`.

```rust
pub struct UairWitness<'a, PolyCoeff: Clone, Int: Clone, const D: usize> {
    pub trace: UairTrace<'a, PolyCoeff, Int, D>,
}
```

Document this clearly: `UairWitness` means full prover assignment, not
private-only data.

The fresh verifier-visible instance contains the unprojected public trace and
commitments to the witness columns.

```rust
pub struct UairInstance<'a, PolyCoeff: Clone, Int: Clone, Commitments, const D: usize> {
    pub public_trace: UairTrace<'a, PolyCoeff, Int, D>,
    pub commitments: Commitments,
}
```

The folded objects should mirror the fresh split:

```rust
pub struct FoldedUairWitness<F: PrimeField, OpeningWitness> {
    pub trace: FoldedUairTrace<F>,
    pub opening_witness: OpeningWitness,
}

pub struct FoldedUairTrace<F: PrimeField> {
    pub binary_poly: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>>,
    pub arbitrary_poly: Vec<DenseMultilinearExtension<DynamicPolynomialF<F>>>,
    pub int: Vec<DenseMultilinearExtension<F>>,
}

pub struct FoldedUairInstance<F, Commitments, Public> {
    pub commitments: Commitments,
    pub public: Public,
    pub u: F,
}
```

`FoldedUairTrace` intentionally keeps the same top-level column families as
`UairTrace`, but its cell types are proof-field objects after projection and
instance folding. Polynomial-valued sources become MLEs whose row values are
univariate proof-field polynomials. Scalar or integer sources become scalar MLEs
over the proof field.

`opening_witness` is prover-only data needed by the PCS to open the folded
commitments, such as commitment randomness or backend-specific prover state. It
is not part of the proof object. Residual and ideal-polynomial caches should
remain prover working state unless a later phase genuinely needs to carry them.

The exact field types can be specialized by the folding protocol. For example,
the SHA-256 production path may use a SHA-specific projected trace with
`bit_slices`, `scalarized_words`, `int_columns`, and `public_columns`, while the
generic UAIR object should preserve the `binary_poly`, `arbitrary_poly`, and
`int` families. The boundary remains the same: witness is prover-private,
instance is verifier-visible.

## LinearIdealFold Proof Objects

`LinearIdealFold` is the generic folding layer for UAIRs whose projected
residue constraints live in linear ideals. The proof object should contain only
verifier messages and claimed evaluations. It should not contain prover-side
caches such as PCS prover data, and it should not contain NeutronNova-specific
objects such as `comm_E` for a committed power-vector witness.

`ProjectionFold Concise` is the source of truth for the production protocol:
the verifier algorithm, Fiat-Shamir ordering, concrete SHA-256 ideal families,
degree bounds, and equations. This file records the UAIR object boundaries and
Rust-facing shape of the proof. If a protocol equation is duplicated here, it is
included only to make the object model unambiguous.

The implementation should reuse the generic proof objects exercised by
`protocol/benches/e2e.rs`. The baseline proof shape in `protocol/src/lib.rs` is:

```rust
pub struct Proof<F: PrimeField, Commitments = pcs::ZipPCSCommitments> {
    pub commitments: Commitments,
    pub zip: Vec<u8>,
    pub ideal_check: IdealCheckProof<F>,
    pub resolver: CombinedPolyResolverProof<F>,
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    pub multipoint_eval: MultipointEvalProof<F>,
    pub witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    pub lookup_proof: Option<BatchedLookupProof<F>>,
}
```

Here `IdealCheckProof`, `CombinedPolyResolverProof`,
`MultiDegreeSumcheckProof`, `MultipointEvalProof`, `DynamicPolynomialF`, and
`BatchedLookupProof` are existing protocol/PIOP types exposed by the baseline
e2e proof shape. The production object model should reuse the active proof
components directly. In particular, do not add a separate family-tag proof layer
such as `IdealFamilyId`, `IdealPolySlot`, `IdealFamilyPolys`, or
`IdealFamilyPoly` just to carry the batched ideal polynomials.

`lookup_proof` is currently a forward-compatible stub in the e2e proof shape.
The prover sets it to `None`, serialization skips it, and the verifier only
carries it through. Production SHA currently has empty `lookup_specs`, so the
production SHA wrapper omits this field.

The current e2e Rust type calls the serialized PCS opening transcript `zip`
because the first backend was Zip+. Semantically this field is the PCS opening
proof. The production object model should not expose this as Zip-specific state.

Production folding may need a thin wrapper around this shape because it also has
an instance-axis SumFold/NIFS proof and multiple fresh commitments. Those extra
fields should reuse existing types:

```rust
pub struct ProductionLinearIdealFoldProof<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub instance_commitments: Vec<PCSCommitments<P, Zt, F, D>>,
    pub ideal_check: IdealCheckProof<F>,
    pub sumfold_proof: MultiDegreeSumcheckProof<F>,
    pub resolver: CombinedPolyResolverProof<F>,
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    pub multipoint_eval: MultipointEvalProof<F>,
    pub witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    pub opening_proof: PCSOpeningProof<P, Zt, F, D>,
}
```

This is intentionally a field-level reuse of the e2e proof object, not a new
PIOP object model. The production wrapper belongs in `protocol`, if needed; it
should not introduce new generic proof structs in `piop/src/neutron_nova`.

The PCS backend remains generic through `ZincPCSTypes` and the component `PCS`
implementations in `zip-plus/src/pcs/generic.rs`. Do not introduce a separate
SHA-specific PCS trait for the proof object. The proof object only needs the
associated commitment and opening-proof types. Production code that actually
folds commitments should put any homomorphic-folding requirements directly on
the component PCS types at the prover/verifier function boundary.

```rust
pub struct PCSOpeningProof<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub binary: <<P as ZincPCSTypes<Zt, F, D>>::BinaryPCS
        as PCS<F, BinaryPoly<D>, D>>::OpeningProof,
    pub arbitrary: <<P as ZincPCSTypes<Zt, F, D>>::ArbitraryPCS
        as PCS<F, DensePolynomial<Zt::Int, D>, D>>::OpeningProof,
    pub int: <<P as ZincPCSTypes<Zt, F, D>>::IntPCS
        as PCS<F, Zt::Int, D>>::OpeningProof,
}
```

That requires the PCS trait to expose the opening proof as an associated type:

```rust
pub trait PCS<F, Eval, const D: usize>: Clone + Debug + Send + Sync
where
    F: PrimeField,
    Eval: Clone + Debug + Send + Sync,
{
    type CommitmentKey: Clone + Debug + Send + Sync;
    type VerifierKey: Clone + Debug + Send + Sync;
    type Commitment: Clone + Debug + Send + Sync;
    type ProverData: Clone + Debug + Send + Sync;
    type OpeningProof: Clone + Debug + Send + Sync + Default;

    fn prove_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        ck: &Self::CommitmentKey,
        polys: &[DenseMultilinearExtension<Eval>],
        point: &[F],
        prover_data: &Self::ProverData,
        field_cfg: &F::Config,
    ) -> Result<Self::OpeningProof, ZipError>;

    fn verify_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        lifted_evals: &[DynamicPolynomialF<F>],
        opening_proof: &Self::OpeningProof,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>;
}
```

The current trait writes and reads opening data through PCS transcripts and
returns `Result<(), ZipError>`. That should be treated as the current adapter
shape, not the production proof-object shape. Zip+ can set
`type OpeningProof = Vec<u8>` while Hyrax or any future PCS can use its native
typed proof.

The aggregate ideal component from `ProjectionFold Concise` should be carried by
the existing ideal-check proof:

```rust
pub struct IdealCheckProof<F: PrimeField> {
    pub combined_mle_values: Vec<DynamicPolynomialF<F>>,
}
```

The family/order information is setup data, not a new proof object. For
production SHA-256, `verify_setup` fixes the canonical mapping from entries of
`combined_mle_values` to the nonzero ideal families:

    ℱ_≠0 = {R₀, R₁, R₄, R₅, R₆, R₉, R₁₀}

The compact production interpretation is:

    ideal_check.combined_mle_values[f] = Ē_f^β(X)
    for f ∈ ℱ_≠0 in setup-defined order

If the generic e2e verifier path is used unchanged, the vector length/order must
match `U::verify_as_subprotocol` and `count_constraints::<U>()`. If the
production verifier uses the seven-family compact form, that compact mapping is
part of `verify_setup`; the carrier is still `IdealCheckProof<F>`.

The honest aggregate polynomial is:

    Ē_f^β(X)
      = ∑_{b ∈ {0,1}^ℓ} eq(β,b)
          ∑_{z ∈ H_row} eq(r_ic,z) · C_f(z,X;w_b,y_b)

In the production transcript, r_ic and β are sampled after binding VS and the
fresh instances, and before E_agg is read. `ProjectionFold Concise` owns the
full Fiat-Shamir sequence.

Thus the submitted ideal component is already batched over both verifier-visible
axes:

    instance axis b via eq(β,b)
    row axis z      via eq(r_ic,z)

The verifier still does not trust these aggregate polynomials blindly. It checks
the shape-level degree bound and ideal membership for each family:

    deg_X Ē_f^β(X) < δ_f
    Ē_f^β(X) ∈ I_f

After accepting and absorbing the aggregate polynomials, the verifier samples
the scalarization and family-batching challenges and computes the initial
NIFS/SumFold claim:

    C₀ = ∑_{f ∈ ℱ_≠0} λ^f · Ē_f^β(a)

The later NIFS, row sumcheck, terminal reconstruction, multipoint reduction, and
PCS opening bind this same scalar to the folded commitments and public trace.

The folded verifier-visible instance is the result of folding fresh instances.
It contains the folded target claim, folded commitments, and any folded public
values needed by the later checks:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedLinearIdealInstance<F: PrimeField, Commitments, Public> {
    pub target: F,
    pub commitments: Commitments,
    pub public: Public,
}
```

The folded witness is prover-private. Its concrete representation can be an
owned folded trace, folded source MLEs, or another protocol-specific witness
bundle. In the generic UAIR model, this is normally a
`FoldedUairWitness<F, OpeningWitness>`:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedLinearIdealWitness<Witness> {
    pub witness: Witness,
}
```

Here `OpeningWitness` is the PCS/backend-specific prover-only state needed to
open the folded commitments.

The ideal-check proof is the production `E_agg` component from
`ProjectionFold Concise`, represented with `IdealCheckProof<F>`. It contains the
seven SHA-256 aggregate ideal polynomials for the nonzero families, or the
shape-defined analogue for another UAIR. These polynomials are verifier-visible
and must be absorbed before sampling a, λ, ρ, ξ.

`sumfold_proof` is the instance-axis `MultiDegreeSumcheckProof`. It proves the
SumFold transition from the verifier-computed C₀ to the folded target. The
verifier derives r_b, folding weights θ_b, and T′:

    θ_b = eq(r_b,b)

    T′ = c_SF / eq(β,r_b)

`resolver` and `combined_sumcheck` are the same terminal-reconstruction objects
used by e2e step 4. `combined_sumcheck` reduces the folded row claim to r⋆, and
`resolver` carries the terminal evaluations needed to close the combined
polynomial resolver:

```rust
pub struct CombinedPolyResolverProof<F: PrimeField> {
    pub up_evals: Vec<F>,
    pub down_evals: Vec<F>,
    pub bit_slice_evals: Vec<F>,
    pub bit_op_down_evals: Vec<F>,
    pub shifted_bit_slice_evals: Vec<F>,
}
```

Together they prove that the folded target is the row-domain sum of the folded
residue expression:

    T′ = ∑_{x ∈ {0,1}^d} eq(r_ic,x) · Φ_folded(x)

This sumcheck reduces the folded claim to a terminal point r⋆.

The verifier uses `resolver` to check:

    terminal = eq(r_ic,r⋆) · Φ_folded(r⋆)

`multipoint_eval` is the existing e2e multipoint proof. It reduces all terminal
evaluation claims at r⋆ and shifted points into one batched opening claim at a
verifier-derived point r₀:

    { p_i(s_i(r⋆)) = v_i }_i  ⇒  P(r₀) = v₀

`witness_lifted_evals` are the existing e2e opening-evaluation carrier. They
are witness-only lifted MLE evaluations at r₀ in F_q[X], ordered as
`[wit_bin..., wit_arb..., wit_int...]`. The verifier recomputes public lifted
evals from the public trace, interleaves public and witness lifted evals,
derives scalar `open_evals` by ψ_a, derives bit-op virtual opens locally, and
checks the `multipoint_eval` subclaim. The serialized PCS opening proof is
`opening_proof`.

The proof chain is:

```text
ideal_check
    → check ideal membership and derive C₀
sumfold_proof
    → fold C₀ into T′
resolver + combined_sumcheck
    → reduce T′ to terminal point r⋆
    → reconstruct the terminal folded expression
multipoint_eval
    → reduce many endpoint claims to one opening point r₀
witness_lifted_evals + opening_proof
    → prove consistency with folded commitments
```

## SHA-256 Domain Objects

For one SHA-256 compression, the semantic relation is:

```rust
H_{i+1} = compress(H_i, M_i)
```

where:

```rust
H_i:     [u32; 8]
M_i:     [u32; 16]
H_{i+1}: [u32; 8]
```

For a chain of `N` compressions:

```rust
pub struct Sha256ChainPublicInput<const N: usize> {
    pub initial_state: [u32; 8],
    pub message_blocks: [[u32; 16]; N],
    pub final_state: [u32; 8],
}
```

For standard SHA-256 hashing from the fixed IV, use a wrapper:

```rust
pub struct Sha256HashPublicInput<const N: usize> {
    pub message_blocks: [[u32; 16]; N],
    pub digest: [u32; 8],
}
```

This wrapper expands into `Sha256ChainPublicInput` by setting
`initial_state = SHA256_IV`.

The current e2e SHA UAIR packs multiple compressions into a single trace. It
does not pass intermediate states as separate public inputs. Witness generation
computes:

```rust
H_1 = compress(H_0, M_0)
H_2 = compress(H_1, M_1)
...
H_N = compress(H_{N-1}, M_{N-1})
```

and writes the relevant public values into public UAIR columns.

## SHA-256 to UAIR Mapping

The SHA domain input is not the same thing as the UAIR public trace.

The SHA public input:

```rust
Sha256ChainPublicInput {
    initial_state,
    message_blocks,
    final_state,
}
```

is used to build the UAIR public trace. For the current SHA UAIR this includes
columns such as:

- `PA_M`: message block words
- `PA_A` / `PA_E`: chaining states and final output prefix
- `PA_K`: SHA-256 round constants
- selector columns
- implementation-specific public helper columns

The prover also builds the full `UairWitness` trace containing the public
columns plus private/witness columns such as:

- message schedule `W`
- round state columns
- sigma/Sigma columns
- Ch/Maj auxiliary columns
- carry columns
- compensator columns

The verifier should build or receive only the public trace, then verify the
proof against commitments to the private columns.

## Recommended Production Flow

Witness generation is outside the prover. It consumes semantic public input and
produces a full pre-projection UAIR witness:

```rust
pub fn build_uair_witness<U, Input, PolyCoeff, Int, const D: usize>(
    shape: &UairShape<U>,
    public: &Input,
) -> Result<UairWitness<'static, PolyCoeff, Int, D>, UairWitnessError>
where
    U: Uair,
    PolyCoeff: Clone,
    Int: Clone;
```

The SHA-256 instantiation can be written more concretely as:

```rust
pub fn build_sha256_witness<Zt, const N: usize, const D: usize>(
    shape: &UairShape<ProductionShaUair<Zt::Int, N>>,
    public: &Sha256ChainPublicInput<N>,
) -> Result<UairWitness<'static, Zt::Int, Zt::Int, D>, ShaWitnessError>
where
    Zt: ZincTypes<D>;
```

The prover receives witnesses and commits to the witness columns internally. It
returns fresh verifier-visible instances, the folded accumulator pair, and the
proof:

```rust
pub struct LinearIdealFoldProveOutput<Instance, FoldedInstance, FoldedWitness, Proof> {
    pub fresh_instances: Vec<Instance>,
    pub folded_instance: FoldedInstance,
    pub folded_witness: FoldedWitness,
    pub proof: Proof,
}

pub fn prove_linear_ideal_fold<P, U, Zt, F, const D: usize>(
    pp: &LinearIdealFoldProverParams<P, U, Zt, F, D>,
    shape: &UairShape<U>,
    witnesses: &[UairWitness<'_, Zt::Int, Zt::Int, D>],
    transcript: &mut impl Transcript,
) -> Result<
    LinearIdealFoldProveOutput<
        UairInstance<'static, Zt::Int, Zt::Int, PCSCommitments<P, Zt, F, D>, D>,
        FoldedLinearIdealInstance<F, PCSCommitments<P, Zt, F, D>, ProjectedShaPublic<F>>,
        FoldedLinearIdealWitness<ProductionShaFoldedWitness<P, Zt, F, D>>,
        ProductionLinearIdealFoldProof<P, Zt, F, D>,
    >,
    LinearIdealFoldError<F>,
>
where
    U: Uair,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>;
```

The SHA production folded witness keeps the structured folded SHA trace and the
folded PCS opening witness:

```rust
pub struct ProductionShaFoldedWitness<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub trace: ProjectedShaTrace<F>,
    pub opening_witness: PCSProverData<P, Zt, F, D>,
}
```

The folded verifier-visible public value is `ProjectedShaPublic<F>`, not a flat
field vector, because the verifier needs structured SHA public columns for
terminal reconstruction and multipoint checks.

The production verifier interface in `ProjectionFold Concise` is the acceptance
predicate:

    verify(VS, {Inst_b}_{b ∈ {0,1}^ℓ}, π) → {true, false}

When VS is fixed by context, the shorthand is:

    verify({Inst_b}_{b ∈ {0,1}^ℓ}, π) → {true, false}

The Rust-facing API uses the same two-step shape. Setup verification checks and
stores static material:

```rust
pub fn setup_verify_linear_ideal_fold<P, U, Zt, F, const D: usize>(
    params: LinearIdealFoldVerifierParams<P, U, Zt, F, D>,
    shape: UairShape<U>,
) -> Result<VerifiedLinearIdealFoldSetup<P, U, Zt, F, D>, LinearIdealFoldError<F>>
where
    U: Uair + ProductionShaProjectionAdapter<Zt, F, D>,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>;
```

The verifier then receives `VS`, fresh instances, and the proof. It derives the
same folded instance, but it never receives the folded witness:

```rust
pub fn verify_linear_ideal_fold<P, U, Zt, F, const D: usize>(
    vs: &VerifiedLinearIdealFoldSetup<P, U, Zt, F, D>,
    instances: &[UairInstance<'_, Zt::Int, Zt::Int, PCSCommitments<P, Zt, F, D>, D>],
    proof: &ProductionLinearIdealFoldProof<P, Zt, F, D>,
    transcript: &mut impl Transcript,
) -> Result<
    FoldedLinearIdealInstance<F, PCSCommitments<P, Zt, F, D>, ProjectedShaPublic<F>>,
    LinearIdealFoldError<F>,
>
where
    U: Uair + ProductionShaProjectionAdapter<Zt, F, D>,
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>;
```

Returning `Ok(folded_instance)` means the production verifier accepts. Returning
`Err(_)` means rejection.

```rust
Sha256ChainPublicInput
    -> build public UairTrace

Sha256ChainPublicInput + witness generation
    -> full UairWitness

UairWitness + UairShape
    -> commit witness columns
    -> UairInstance { public_trace, commitments }
    -> prove

UairInstance + proof
    -> verify
```

Projection to the proof field and evaluation at verifier challenges are internal
protocol phases. They should not be part of the initial public instance type.
