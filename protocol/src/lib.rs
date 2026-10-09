//! Zinc+ PIOP for UCS - end-to-end protocol.
//!
//! Implements the Zinc+ compiler pipeline (cf. paper, Section "Zinc+
//! Compiler"):
//!
//! ```text
//! Z[X]  --\phi_q-->  F_q[X]  --MLE eval-->  F_q[X]  --\psi_a-->  F_q
//!         Step 1               Step 2                  Step 3
//! ```
//!
//! After the three compiler steps, the protocol continues with:
//!
//! - Step 4: combined CPR + Lookup multi-degree sumcheck (CPR group at degree
//!   `max_deg+2`, one lookup group per table type; shared eval point `r*`)
//! - Step 5: multi-point evaluation sumcheck (combines up/down evals at r* into
//!   a single evaluation point r_0)
//! - Step 6: lift-and-project (unprojected MLE evaluations at r_0)
//! - Step 7: Zip+ PCS open/verify at r_0

pub mod fixed_prime;
pub mod prover;
pub mod verifier;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{
    ConstIntRing, ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig, PrimeField, Semiring,
};
use std::{fmt::Debug, marker::PhantomData};
use thiserror::Error;
use zinc_piop::{
    bin_multipoint_reducer::Proof as BinReducerProof,
    combined_poly_resolver::{CombinedPolyResolverError, Proof as CombinedPolyResolverProof},
    ideal_check::{IdealCheckError, Proof as IdealCheckProof},
    lookup::{LookupError, gkr_logup::GkrLogupLookupProof},
    multipoint_eval::{MultipointEvalError, Proof as MultipointEvalProof},
    projections::ProjectedTrace,
    sumcheck::multi_degree::MultiDegreeSumcheckProof,
};
use zinc_poly::{
    ConstCoeffBitWidth, EvaluationError as PolyEvaluationError,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly,
        dense::DensePolynomial,
        dynamic::over_field::{DynamicPolyVecF, DynamicPolynomialF},
    },
};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_uair::{Uair, ideal::Ideal};
use zinc_utils::{
    cfg_extend, cfg_into_iter, cfg_iter, inner_transparent_field::InnerTransparentField, mul,
    named::Named,
};
use zip_plus::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlusCommitment, ZipTypes},
};

//
// Data structures
//

/// Full proof produced by the Zinc+ PIOP for UCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    /// Zip+ commitments to the witness columns.
    pub commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip: Vec<u8>,
    /// Randomized ideal check proof.
    pub ideal_check: IdealCheckProof<F>,
    /// Combined polynomial resolver proof (up_evals + down_evals +
    /// bit_op_down_evals).
    pub resolver: CombinedPolyResolverProof<F>,
    /// Multi-degree sumcheck proof (CPR group + future lookup groups).
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Multi-point evaluation sumcheck proof. Reduces all CPR claims at
    /// `r*` (up evals + row-shift down evals + bit-op virtual-column
    /// down evals) to a single evaluation point `r_0`. Bit-op sources
    /// are folded in as additional `up` slots; their consistency is
    /// discharged at `r_0` in Step 6 by applying the bit-op locally to
    /// the source's lifted eval.
    pub multipoint_eval: MultipointEvalProof<F>,
    /// Witness-only polynomial MLE evaluations at r_0 in F_q[X]
    /// (after \phi_q, before \psi_a), ordered as
    /// `[wit_bin..., wit_arb..., wit_int...]`.
    /// The verifier recomputes public lifted_evals from public data,
    /// interleaves them with these, and derives scalar open_evals via
    /// \psi_a for the sumcheck consistency check and Zip+ PCS verify.
    pub witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    /// GKR-LogUp lookup proof. `groups: vec![]` (default) when the UAIR
    /// has no lookup specs.
    pub lookup_proof: GkrLogupLookupProof<F>,
    /// Multi-point reducer proof for the binary_poly batch — `Some` iff
    /// the UAIR has at least one lookup spec. Reduces all (per-group
    /// `r_inner^(g)` + step-7 `r_0`) bin claims into ONE Zip+ opening
    /// at the reduced point `r*`. `None` when there are no lookup
    /// groups; in that case step 7 opens the bin commitment at `r_0`
    /// directly.
    pub bin_reducer_proof: Option<BinReducerProof<F>>,
    /// Polynomial-valued MLE evals at `r*` for each witness binary_poly
    /// column, in column order. Empty when `bin_reducer_proof` is
    /// `None`. Used for cross-checking the reducer's `P(r*)` and
    /// computing the alpha-projected eval for the single bin Zip+
    /// open at `r*`.
    pub bin_lifts_at_r_star: Vec<DynamicPolynomialF<F>>,
    /// Int twin of `bin_reducer_proof` — `Some` iff the UAIR has at least
    /// one `Word`-table (int range-check) lookup group: the int multipoint
    /// reducer folds every int-group `r_inner` claim plus the step-7 `r_0`
    /// claim into ONE int Zip+ open at its reduced point. `None` otherwise
    /// (int opened at `r_0` directly).
    pub int_reducer_proof: Option<BinReducerProof<F>>,
    /// Scalar MLE evals at the int reducer's `r*` for each witness int
    /// column, in column order. Empty when `int_reducer_proof` is `None`.
    pub int_evals_at_r_star: Vec<F>,
}

impl<F> GenTranscribable for Proof<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (commit0, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);
        let (commit1, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);
        let (commit2, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);

        let (zip_len, bytes) = u32::read_transcription_bytes_subset(bytes);
        let zip_len = usize::try_from(zip_len).expect("zip length must fit into usize");
        let (zip_bytes, bytes) = bytes.split_at(zip_len);
        let zip = zip_bytes.to_vec();

        let (ideal_check, bytes) = IdealCheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (resolver, bytes) =
            CombinedPolyResolverProof::<F>::read_transcription_bytes_subset(bytes);
        let (combined_sumcheck, bytes) =
            MultiDegreeSumcheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (multipoint_eval, bytes) =
            MultipointEvalProof::<F>::read_transcription_bytes_subset(bytes);

        let (witness_vec, bytes) = DynamicPolyVecF::<F>::read_transcription_bytes_subset(bytes);
        let witness_lifted_evals = witness_vec.0;

        let (lookup_proof, bytes) =
            GkrLogupLookupProof::<F>::read_transcription_bytes_subset(bytes);

        // [reducer_present: u8] [reducer_proof? : subset] [bin_lifts? : subset]
        let reducer_present = bytes[0];
        let bytes = &bytes[1..];
        let (bin_reducer_proof, bin_lifts_at_r_star, bytes) = match reducer_present {
            0 => (None, Vec::new(), bytes),
            1 => {
                let (rp, rest) = BinReducerProof::<F>::read_transcription_bytes_subset(bytes);
                let (bv, rest) = DynamicPolyVecF::<F>::read_transcription_bytes_subset(rest);
                (Some(rp), bv.0, rest)
            }
            v => panic!("invalid bin_reducer_proof presence flag: {v}"),
        };

        // [int_reducer_present: u8] [reducer_proof? : subset]
        // [modulus] [u32 n] [n × F::Inner]   (the last three only if present)
        let int_present = bytes[0];
        let bytes = &bytes[1..];
        let (int_reducer_proof, int_evals_at_r_star, bytes) = match int_present {
            0 => (None, Vec::new(), bytes),
            1 => {
                let (rp, rest) = BinReducerProof::<F>::read_transcription_bytes_subset(bytes);
                let mod_size = F::Modulus::NUM_BYTES;
                let cfg = zinc_transcript::read_field_cfg::<F>(&rest[..mod_size]);
                let rest = &rest[mod_size..];
                let (n, rest) = u32::read_transcription_bytes_subset(rest);
                let n = usize::try_from(n).expect("int eval count must fit in usize");
                let inner_size = F::Inner::NUM_BYTES;
                let (evals_bytes, rest) = rest.split_at(mul!(n, inner_size));
                let evals = zinc_transcript::read_field_vec_with_cfg::<F>(evals_bytes, &cfg);
                (Some(rp), evals, rest)
            }
            v => panic!("invalid int_reducer_proof presence flag: {v}"),
        };
        assert!(bytes.is_empty(), "All bytes should be consumed");

        Self {
            commitments: (commit0, commit1, commit2),
            zip,
            ideal_check,
            resolver,
            combined_sumcheck,
            multipoint_eval,
            witness_lifted_evals,
            lookup_proof,
            bin_reducer_proof,
            bin_lifts_at_r_star,
            int_reducer_proof,
            int_evals_at_r_star,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        // 3 commitments (ConstTranscribable - no length prefix)
        buf = self.commitments.0.write_transcription_bytes_subset(buf);
        buf = self.commitments.1.write_transcription_bytes_subset(buf);
        buf = self.commitments.2.write_transcription_bytes_subset(buf);

        // zip: u32 length + raw bytes
        let zip_len = u32::try_from(self.zip.len()).expect("zip length must fit into u32");
        zip_len.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        buf = &mut buf[u32::NUM_BYTES..];
        buf[..self.zip.len()].copy_from_slice(&self.zip);
        buf = &mut buf[self.zip.len()..];

        // ideal_check: u32 length prefix + data
        buf = self.ideal_check.write_transcription_bytes_subset(buf);

        // resolver: u32 length prefix + data
        buf = self.resolver.write_transcription_bytes_subset(buf);

        // combined_sumcheck: u32 length prefix + data
        buf = self.combined_sumcheck.write_transcription_bytes_subset(buf);

        // multipoint_eval: u32 length prefix + data
        buf = self.multipoint_eval.write_transcription_bytes_subset(buf);

        // witness_lifted_evals: u32 length prefix + DynamicPolyVecF encoding
        let buf = DynamicPolyVecF::reinterpret(&self.witness_lifted_evals)
            .write_transcription_bytes_subset(buf);

        // lookup_proof: u32 length prefix + GkrLogupLookupProof encoding
        let buf = self.lookup_proof.write_transcription_bytes_subset(buf);

        // [reducer_present: u8] then optional reducer_proof + bin_lifts.
        let buf = match &self.bin_reducer_proof {
            None => {
                assert!(
                    self.bin_lifts_at_r_star.is_empty(),
                    "bin_lifts_at_r_star must be empty when reducer is absent"
                );
                buf[0] = 0;
                &mut buf[1..]
            }
            Some(rp) => {
                buf[0] = 1;
                let buf = &mut buf[1..];
                let buf = rp.write_transcription_bytes_subset(buf);
                DynamicPolyVecF::reinterpret(&self.bin_lifts_at_r_star)
                    .write_transcription_bytes_subset(buf)
            }
        };

        // [int_reducer_present: u8] then optional reducer_proof + modulus +
        // [u32 n] + n × F::Inner (the evals at the int reducer's r*).
        match &self.int_reducer_proof {
            None => {
                assert!(
                    self.int_evals_at_r_star.is_empty(),
                    "int_evals_at_r_star must be empty when the int reducer is absent"
                );
                buf[0] = 0;
            }
            Some(rp) => {
                buf[0] = 1;
                let buf = &mut buf[1..];
                let buf = rp.write_transcription_bytes_subset(buf);
                let modulus = rp.sumcheck_proof.claimed_sum.modulus();
                let buf = zinc_transcript::append_field_cfg::<F>(buf, &modulus);
                let n = u32::try_from(self.int_evals_at_r_star.len())
                    .expect("int eval count must fit in u32");
                n.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
                let buf = &mut buf[u32::NUM_BYTES..];
                zinc_transcript::append_field_vec_inner(buf, &self.int_evals_at_r_star);
            }
        }
    }
}

impl<F> Transcribable for Proof<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let witness_vec = DynamicPolyVecF::reinterpret(&self.witness_lifted_evals);
        3 * ZipPlusCommitment::NUM_BYTES
            + u32::NUM_BYTES
            + self.zip.len()
            + IdealCheckProof::<F>::LENGTH_NUM_BYTES
            + self.ideal_check.get_num_bytes()
            + CombinedPolyResolverProof::<F>::LENGTH_NUM_BYTES
            + self.resolver.get_num_bytes()
            + MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES
            + self.combined_sumcheck.get_num_bytes()
            + MultipointEvalProof::<F>::LENGTH_NUM_BYTES
            + self.multipoint_eval.get_num_bytes()
            + DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
            + witness_vec.get_num_bytes()
            + GkrLogupLookupProof::<F>::LENGTH_NUM_BYTES
            + self.lookup_proof.get_num_bytes()
            + 1 // reducer_present flag
            + match &self.bin_reducer_proof {
                None => 0,
                Some(rp) => {
                    let bv = DynamicPolyVecF::reinterpret(&self.bin_lifts_at_r_star);
                    BinReducerProof::<F>::LENGTH_NUM_BYTES
                        + rp.get_num_bytes()
                        + DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
                        + bv.get_num_bytes()
                }
            }
            + 1 // int_reducer_present flag
            + match &self.int_reducer_proof {
                None => 0,
                Some(rp) => {
                    BinReducerProof::<F>::LENGTH_NUM_BYTES
                        + rp.get_num_bytes()
                        + F::Modulus::NUM_BYTES
                        + u32::NUM_BYTES
                        + self.int_evals_at_r_star.len() * F::Inner::NUM_BYTES
                }
            }
    }
}

/// Target security level (bits) of this crate's tests and benches: 100 by
/// default, raised by the `sec-114` / `sec-128` cargo features. It sets the
/// Zip+ column-opening count for the chosen code rate
/// ([`zip_plus::pcs::structs::num_column_openings`]) and, where the
/// projecting prime is transcript-drawn, guides the prime width the benches
/// pick (128-bit up to 100 bits, 192-bit above — the fingerprinting error
/// is `≈ (bit-size of the largest constraint residue) / 2^(prime bits)`, and
/// the LogUp lookups spend `≈ (#lookups + table size) / 2^(prime bits)`).
pub const SECURITY_BITS: usize = if cfg!(feature = "sec-128") {
    128
} else if cfg!(feature = "sec-114") {
    114
} else {
    100
};

/// Trait bundling the various type parameters for the public inputs (NYI),
/// witness and Zinc+ PIOP.
pub trait ZincTypes<const DEGREE_PLUS_ONE: usize>: Clone + Debug {
    /// Main integer type for the protocol, used as a coefficient type for the
    /// arbitrary polynomial trace columns and for the integer trace columns.
    type Int: Semiring
        + ConstTranscribable
        + ConstCoeffBitWidth
        + Named
        + Default
        + Clone
        + Send
        + Sync
        + 'static;

    /// Projecting element to project Zip+ evaluations and UAIR scalars to the
    /// field.
    type Chal: ConstIntRing + ConstTranscribable + Named;

    /// Evaluation point type, used for all column types in Zip+ to evaluate
    /// multilinear polynomials.
    type Pt: ConstIntRing;

    /// Randomly sampled field modulus type, used throughout the protocol for
    /// finite field operations.
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;

    /// Primality test for the field modulus.
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// The projecting prime `q` of Step 1 (`\phi_q : Z[X] -> F_q[X]`).
    ///
    /// `None` (the default): `q` is drawn from the Fiat–Shamir transcript
    /// after the witness commitments — the sound, general behaviour. The
    /// prime has exactly `8 · Fmod::NUM_BYTES` bits, so `Fmod` sizes it
    /// (`Uint<2>` → 128-bit, `Uint<3>` → 192-bit, …).
    ///
    /// `Some(bytes)`: pin `q` to the prime whose little-endian
    /// transcription bytes these are (must be exactly `Fmod::NUM_BYTES`
    /// long). Only for arithmetizations whose constraints are identities
    /// modulo that specific prime (the SHA+ECDSA demo pins
    /// [`fixed_prime::SECP256K1_P_LE_BYTES`]); see `fixed_prime` for the
    /// soundness caveat.
    const FIXED_PROJECTING_PRIME: Option<&'static [u8]> = None;

    /// Zip+ types for the binary polynomial trace columns.
    /// `CombR` is independent per witness type — sized to fit the
    /// inner products that lane actually performs (binary is much
    /// narrower than arb/int).
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns.
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the integer trace columns.
    type IntZt: ZipTypes<
            Eval = Self::Int,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Linear code used in Zip+ for the binary polynomial trace columns.
    type BinaryLc: LinearCode<Self::BinaryZt>;

    /// Linear code used in Zip+ for the arbitrary polynomial trace columns.
    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;

    /// Linear code used in Zip+ for the integer trace columns.
    type IntLc: LinearCode<Self::IntZt>;
}

/// Type bundle for the **folded** Zinc+ PIOP (1× fold, 2× column splitting).
///
/// The PIOP runs at trace degree `D` (so the trace and UAIR are unchanged
/// from the unfolded path), but the binary commitment is over `BinaryPoly<HALF_D>`
/// — each `BinaryPoly<D>` witness column is split into two `BinaryPoly<HALF_D>`
/// halves before commit. This decouples the trace's `BinaryPoly<D>` from the
/// PCS's `BinaryPoly<HALF_D>`, which `ZincTypes<D>` would otherwise force to
/// be the same (`BinaryZt::Eval = BinaryPoly<DEGREE_PLUS_ONE>`).
///
/// Arbitrary and integer commitments are unchanged.
pub trait FoldedZincTypes<const D: usize, const HALF_D: usize>: Clone + Debug {
    type Int: Semiring
        + ConstTranscribable
        + ConstCoeffBitWidth
        + Named
        + Default
        + Clone
        + Send
        + Sync
        + 'static;

    type Chal: ConstIntRing + ConstTranscribable + Named;

    type Pt: ConstIntRing;

    type Fmod: ConstIntSemiring + ConstTranscribable + Named;

    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// The projecting prime `q` of Step 1 (`\phi_q : Z[X] -> F_q[X]`).
    ///
    /// `None` (the default): `q` is drawn from the Fiat–Shamir transcript
    /// after the witness commitments — the sound, general behaviour. The
    /// prime has exactly `8 · Fmod::NUM_BYTES` bits, so `Fmod` sizes it
    /// (`Uint<2>` → 128-bit, `Uint<3>` → 192-bit, …).
    ///
    /// `Some(bytes)`: pin `q` to the prime whose little-endian
    /// transcription bytes these are (must be exactly `Fmod::NUM_BYTES`
    /// long). Only for arithmetizations whose constraints are identities
    /// modulo that specific prime (the SHA+ECDSA demo pins
    /// [`fixed_prime::SECP256K1_P_LE_BYTES`]); see `fixed_prime` for the
    /// soundness caveat.
    const FIXED_PROJECTING_PRIME: Option<&'static [u8]> = None;

    /// Zip+ types for the **split** binary trace columns.
    /// `Eval = BinaryPoly<HALF_D>` — one round of 2× folding.
    /// `CombR` is independent per witness type.
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<HALF_D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns (unchanged from
    /// the unfolded path: degree-`D` polynomials).
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the integer trace columns (unchanged).
    type IntZt: ZipTypes<
            Eval = Self::Int,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type BinaryLc: LinearCode<Self::BinaryZt>;

    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;

    type IntLc: LinearCode<Self::IntZt>;
}

/// Like [`FoldedZincTypes`], but additionally folds int witness columns by
/// 2× via the `v = lo + 2^128 · hi` decomposition, with halves stored as
/// `Int<INT_HALF_LIMBS>`. The int Zip+ commits the split witness at length
/// `2n` and the protocol opens at the same extended point `(r_0 ‖ γ)` as
/// the binary fold; the verifier mirrors with `(1−γ) c1 + γ c2` and
/// `alpha_stride = 1` (since `IntZt::Cw` is scalar).
pub trait IntFoldedZincTypes<
    const D: usize,
    const HALF_D: usize,
    const INT_LIMBS: usize,
    const INT_HALF_LIMBS: usize,
>: Clone + Debug
{
    type Chal: ConstIntRing + ConstTranscribable + Named;
    type Pt: ConstIntRing;
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// The projecting prime `q` of Step 1 (`\phi_q : Z[X] -> F_q[X]`).
    ///
    /// `None` (the default): `q` is drawn from the Fiat–Shamir transcript
    /// after the witness commitments — the sound, general behaviour. The
    /// prime has exactly `8 · Fmod::NUM_BYTES` bits, so `Fmod` sizes it
    /// (`Uint<2>` → 128-bit, `Uint<3>` → 192-bit, …).
    ///
    /// `Some(bytes)`: pin `q` to the prime whose little-endian
    /// transcription bytes these are (must be exactly `Fmod::NUM_BYTES`
    /// long). Only for arithmetizations whose constraints are identities
    /// modulo that specific prime (the SHA+ECDSA demo pins
    /// [`fixed_prime::SECP256K1_P_LE_BYTES`]); see `fixed_prime` for the
    /// soundness caveat.
    const FIXED_PROJECTING_PRIME: Option<&'static [u8]> = None;

    /// Zip+ types for the split binary trace columns
    /// (`Eval = BinaryPoly<HALF_D>`).
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<HALF_D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns
    /// (unchanged from the unfolded path).
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<crypto_primitives::crypto_bigint_int::Int<INT_LIMBS>, D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the split integer trace columns
    /// (`Eval = Int<INT_HALF_LIMBS>`).
    type IntZt: ZipTypes<
            Eval = crypto_primitives::crypto_bigint_int::Int<INT_HALF_LIMBS>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type BinaryLc: LinearCode<Self::BinaryZt>;
    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;
    type IntLc: LinearCode<Self::IntZt>;
}

/// 4× counterpart of [`IntFoldedZincTypes`]: also folds binary by 4× via
/// `BinaryPoly<D> → BinaryPoly<QUARTER_D>` AND folds int by 4× via
/// quarters (`v = q_0 + 2^64·q_1 + 2^128·q_2 + 2^192·q_3`), each stored
/// as `Int<INT_QUARTER_LIMBS>`. Both binary and int commit at length
/// `4n` and open at `(r_0 ‖ γ_1 ‖ γ_2)`. Verifier mirrors with the
/// 4-block algebra `(1−γ_1)(1−γ_2) c[0] + γ_1(1−γ_2) c[2] +
/// (1−γ_1)γ_2 c[1] + γ_1·γ_2 c[3]` and `alpha_stride = 1`.
pub trait IntFoldedZincTypes4x<
    const D: usize,
    const QUARTER_D: usize,
    const INT_LIMBS: usize,
    const INT_QUARTER_LIMBS: usize,
>: Clone + Debug
{
    type Chal: ConstIntRing + ConstTranscribable + Named;
    type Pt: ConstIntRing;
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// The projecting prime `q` of Step 1 (`\phi_q : Z[X] -> F_q[X]`).
    ///
    /// `None` (the default): `q` is drawn from the Fiat–Shamir transcript
    /// after the witness commitments — the sound, general behaviour. The
    /// prime has exactly `8 · Fmod::NUM_BYTES` bits, so `Fmod` sizes it
    /// (`Uint<2>` → 128-bit, `Uint<3>` → 192-bit, …).
    ///
    /// `Some(bytes)`: pin `q` to the prime whose little-endian
    /// transcription bytes these are (must be exactly `Fmod::NUM_BYTES`
    /// long). Only for arithmetizations whose constraints are identities
    /// modulo that specific prime (the SHA+ECDSA demo pins
    /// [`fixed_prime::SECP256K1_P_LE_BYTES`]); see `fixed_prime` for the
    /// soundness caveat.
    const FIXED_PROJECTING_PRIME: Option<&'static [u8]> = None;

    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<QUARTER_D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<crypto_primitives::crypto_bigint_int::Int<INT_LIMBS>, D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type IntZt: ZipTypes<
            Eval = crypto_primitives::crypto_bigint_int::Int<INT_QUARTER_LIMBS>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type BinaryLc: LinearCode<Self::BinaryZt>;
    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;
    type IntLc: LinearCode<Self::IntZt>;
}

/// Main struct for the Zinc+ PIOP. The protocol is implemented as associated
/// functions on it.
///
/// (Note that type parameters are further constrained in the impl blocks for
/// the prover and verifier)
#[derive(Copy, Clone, Default, Debug)]
pub struct ZincPlusPiop<Zt, U, F, const DEGREE_PLUS_ONE: usize>(PhantomData<(Zt, U, F)>)
where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    U: Uair,
    F: PrimeField;

/// Error type for error happening during the protocol execution (prover and
/// verifier).
#[derive(Debug, Error)]
pub enum ProtocolError<F: PrimeField, I: Ideal> {
    #[error("ideal check failed: {0}")]
    IdealCheck(#[from] IdealCheckError<F, I>),
    #[error("combined poly resolver failed: {0}")]
    Resolver(#[from] CombinedPolyResolverError<F>),
    #[error("scalar projection failed: {0}")]
    ScalarProjection(PolyEvaluationError),
    #[error("multi-point evaluation failed: {0}")]
    MultipointEval(#[from] MultipointEvalError<F>),
    #[error("lifted eval psi_a projection failed: {0}")]
    LiftedEvalProjection(PolyEvaluationError),
    #[error("lifted_evals bit-op consistency mismatch at bit_op spec {spec}")]
    LiftedEvalsBitOpMismatch { spec: usize },
    #[error("lookup argument failed: {0}")]
    Lookup(#[from] LookupError),
    #[error("booleanity check failed: {0}")]
    Booleanity(zinc_piop::lookup::booleanity::BooleanityError<F>),
    #[error("public-trace consistency check failed: {0}")]
    PublicConsistency(String),
    #[error("public-column structural check failed: {0}")]
    PublicStructure(zinc_uair::PublicStructureError),
    #[error("shifted bit-slice evaluation failed: {0}")]
    ShiftedBitSliceEval(zinc_poly::EvaluationError),
    #[error("PCS error: {0}")]
    Pcs(#[from] ZipError),
    #[error("PCS verification failed at column {0}: {1}")]
    PcsVerification(usize, ZipError),
    #[error("assert_zero constraint {0} does not vanish at the projecting element")]
    AssertZero(usize),
}

//
// Helper functions
//

/// Absorb public column entries into the Fiat-Shamir transcript.
///
/// Each entry is serialized via `ConstTranscribable::write_transcription_bytes`
/// and absorbed. This must be called in the same order by both prover and
/// verifier, after commitments and before the random prime draw.
fn absorb_public_columns<T: ConstTranscribable>(
    transcript: &mut impl Transcript,
    cols: &[DenseMultilinearExtension<T>],
) {
    let mut buf = vec![0u8; T::NUM_BYTES];
    for col in cols {
        for entry in col.iter() {
            entry.write_transcription_bytes_exact(&mut buf);
            transcript.absorb_slice(&buf);
        }
    }
}

/// Compute per-column lifted MLE evaluations at `point`.
///
/// For each column j, returns `\sum_b eq(b, point) * v_j(b)` as a polynomial
/// in `F_q[X]` (coefficient-wise MLE evaluation). Dispatches on the trace
/// layout internally.
///
/// Binary columns exploit the 0/1 structure for conditional additions only.
/// The `eq(point, *)` table is built once and reused across all columns.
#[allow(clippy::arithmetic_side_effects)]
fn compute_lifted_evals<F: PrimeField, const D: usize>(
    point: &[F],
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_trace: &ProjectedTrace<F>,
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>> {
    compute_lifted_evals_capped::<F, D>(point, trace_bin_poly, projected_trace, field_cfg, None)
}

/// Like [`compute_lifted_evals`] but the non-binary section is capped at
/// `non_binary_cap` entries (counted from the start of the non-binary
/// region). Use `None` for full compute (matches `compute_lifted_evals`).
///
/// Use case: int-fold provers compute the int section separately via
/// [`compute_int_fold_lifted_evals`] / [`compute_int_fold_4x_lifted_evals`]
/// (returning 2/4-coeff bar_us), so computing the standard 1-coeff int
/// section here is wasted work. Pass `Some(num_total_arb_cols)` to stop
/// the non-binary iter right after arbitrary cols.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_lifted_evals_capped<F: PrimeField, const D: usize>(
    point: &[F],
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_trace: &ProjectedTrace<F>,
    field_cfg: &F::Config,
    non_binary_cap: Option<usize>,
) -> Vec<DynamicPolynomialF<F>> {
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_lifted_evals: eq table build failed");

    let n_bin = trace_bin_poly.len();
    let zero = F::zero_with_cfg(field_cfg);

    // Binary columns: exploit 0/1 structure for conditional additions.
    // Pack each entry's up-to-64 boolean coefficients into a u64 so we
    // can (a) skip entries that are identically zero, and (b) walk only
    // the SET bits via `trailing_zeros` + Brian Kernighan's clear-lowest
    // instead of branching on every slot.
    debug_assert!(D <= 64, "compute_lifted_evals: bitmask packing assumes D <= 64");
    let mut result: Vec<DynamicPolynomialF<F>> = cfg_iter!(trace_bin_poly)
        .map(|col| {
            let mut coeffs = vec![zero.clone(); D];
            for (b, entry) in col.iter().enumerate() {
                let mut bits: u64 = 0;
                for (l, coeff) in entry.iter().enumerate().take(D) {
                    if coeff.into_inner() {
                        bits |= 1u64 << l;
                    }
                }
                if bits == 0 {
                    continue;
                }
                let eq_b = &eq_table[b];
                let mut remaining = bits;
                while remaining != 0 {
                    let l = remaining.trailing_zeros() as usize;
                    coeffs[l] += eq_b;
                    remaining &= remaining - 1;
                }
            }
            DynamicPolynomialF::new_trimmed(coeffs)
        })
        .collect();

    // Non-binary columns: coefficient-wise eq-weighted sum.
    fn weighted_eq_sum<'a, F2: PrimeField + 'a>(
        col: impl Iterator<Item = &'a DynamicPolynomialF<F2>> + Clone,
        eq_table: &[F2],
        zero: &F2,
    ) -> DynamicPolynomialF<F2> {
        let num_coeffs = col.clone().map(|e| e.coeffs.len()).max().unwrap_or(0);
        let mut coeffs = vec![zero.clone(); num_coeffs];
        for (b, entry) in col.enumerate() {
            for (l, coeff) in entry.coeffs.iter().enumerate() {
                let mut term = eq_table[b].clone();
                term *= coeff;
                coeffs[l] += &term;
            }
        }
        DynamicPolynomialF::new_trimmed(coeffs)
    }

    match projected_trace {
        ProjectedTrace::RowMajor(t) => {
            let num_cols = t.first().map(|r| r.len()).unwrap_or(0);
            let non_binary_end = match non_binary_cap {
                Some(cap) => (n_bin + cap).min(num_cols),
                None => num_cols,
            };
            cfg_extend!(
                result,
                cfg_into_iter!(n_bin..non_binary_end).map(|col_idx| weighted_eq_sum(
                    t.iter().map(|row| &row[col_idx]),
                    &eq_table,
                    &zero,
                ))
            );
        }
        ProjectedTrace::ColumnMajor(t) => {
            let non_binary_end = match non_binary_cap {
                Some(cap) => (n_bin + cap).min(t.len()),
                None => t.len(),
            };
            cfg_extend!(
                result,
                cfg_iter!(t[n_bin..non_binary_end]).map(|col_mle| weighted_eq_sum(
                    col_mle.iter(),
                    &eq_table,
                    &zero,
                ))
            );
        }
    }

    result
}

/// 1× int-fold lifted-eval helper. Produces 2-coeff bar_us per int
/// column: `[lo_eval, hi_eval]` where each coeff is the MLE eval at
/// `point` of the corresponding 128-bit half.
///
/// `lo` is zero-extended into `Int<HALF_H>` (always non-negative);
/// `hi` is the signed arithmetic shift (sign-preserving). The original
/// column's lifted eval at `point` is recoverable as
/// `coeffs[0] + 2^128 · coeffs[1]` in F.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_int_fold_lifted_evals<F, const H: usize, const HALF_H: usize>(
    point: &[F],
    int_trace: &[DenseMultilinearExtension<crypto_primitives::crypto_bigint_int::Int<H>>],
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: PrimeField
        + for<'a> FromWithConfig<&'a crypto_primitives::crypto_bigint_int::Int<HALF_H>>,
{
    use crypto_primitives::crypto_bigint_int::Int;
    assert!(HALF_H >= 2);
    assert!(H >= HALF_H);
    const LO_LIMBS: usize = 2;
    let shift: u32 = (LO_LIMBS * 64) as u32;
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_int_fold_lifted_evals: eq table build failed");
    let zero = F::zero_with_cfg(field_cfg);

    cfg_iter!(int_trace)
        .map(|col| {
            let mut lo_eval = zero.clone();
            let mut hi_eval = zero.clone();
            for (b, entry) in col.iter().enumerate() {
                let v_words = entry.as_uint().to_words();
                let mut lo_words = [0u64; HALF_H];
                lo_words[0] = v_words[0];
                lo_words[1] = v_words[1];
                let lo: Int<HALF_H> = Int::from_words(lo_words);
                let hi: Int<HALF_H> = (*entry >> shift).resize();

                let mut term_lo = F::from_with_cfg(&lo, field_cfg);
                term_lo *= &eq_table[b];
                lo_eval += &term_lo;
                let mut term_hi = F::from_with_cfg(&hi, field_cfg);
                term_hi *= &eq_table[b];
                hi_eval += &term_hi;
            }
            DynamicPolynomialF::new_trimmed(vec![lo_eval, hi_eval])
        })
        .collect()
}

/// 4× int-fold lifted-eval helper. Produces 4-coeff bar_us per int
/// column: `[q0_eval, q1_eval, q2_eval, q3_eval]` where each coeff is
/// the MLE eval at `point` of the corresponding radix-`R` quarter,
/// `R = 2^(64·(Q−1))`.
///
/// `q_0, q_1, q_2` are zero-extended `(Q−1)`-word source chunks (always
/// non-negative); `q_3` is `(v >> 3·log R).resize()` (signed). The
/// original column's lifted eval at `point` is recoverable as
/// `c[0] + R·c[1] + R²·c[2] + R³·c[3]` in F.
///
/// Two fast-paths shave most of the per-cell cost on traces with
/// many small/zero int values (SHA carries):
/// 1. **Zero-quarter skip**: if `words[i] == 0` (and `q_3` is zero),
///    skip the `F::from_with_cfg + mul + add` for that quarter entirely.
///    For typical SHA carry columns where `|v| < 2^64`, this elides
///    3 of 4 monty-muls per row.
/// 2. **u64 fast-lift**: lift `q_0..q_2` via `F::from_with_cfg(u64)`
///    rather than building an `Int<Q>` and going through the signed
///    `Int → F` path (which calls `is_negative`, `abs`, and `resize`).
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_int_fold_4x_lifted_evals<F, const H: usize, const Q: usize>(
    point: &[F],
    int_trace: &[DenseMultilinearExtension<crypto_primitives::crypto_bigint_int::Int<H>>],
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: PrimeField
        + FromWithConfig<u64>
        + for<'a> FromWithConfig<&'a crypto_primitives::crypto_bigint_int::Int<Q>>,
{
    use crypto_primitives::crypto_bigint_int::Int;
    assert!(Q >= 2);
    let w = Q - 1; // words per quarter; radix = 2^(64·w). w = 1 is the
    // original 64-bit quartering with its u64 fast paths.
    assert!(H > 3 * w);
    let shift3: u32 = (192 * w) as u32;
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_int_fold_4x_lifted_evals: eq table build failed");
    let zero = F::zero_with_cfg(field_cfg);

    cfg_iter!(int_trace)
        .map(|col| {
            let mut q0_eval = zero.clone();
            let mut q1_eval = zero.clone();
            let mut q2_eval = zero.clone();
            let mut q3_eval = zero.clone();
            for (b, entry) in col.iter().enumerate() {
                let words = entry.as_uint().to_words();
                let eq_b = &eq_table[b];

                if w == 1 {
                    // q_0..q_2: unsigned single-limb lift via u64 fast-path,
                    // skipping the multiply when the limb is zero.
                    if words[0] != 0 {
                        let mut t = F::from_with_cfg(words[0], field_cfg);
                        t *= eq_b;
                        q0_eval += &t;
                    }
                    if words[1] != 0 {
                        let mut t = F::from_with_cfg(words[1], field_cfg);
                        t *= eq_b;
                        q1_eval += &t;
                    }
                    if words[2] != 0 {
                        let mut t = F::from_with_cfg(words[2], field_cfg);
                        t *= eq_b;
                        q2_eval += &t;
                    }
                } else {
                    // q_0..q_2: unsigned (Q−1)-word chunks, zero-skipped.
                    for (k, acc) in
                        [&mut q0_eval, &mut q1_eval, &mut q2_eval].into_iter().enumerate()
                    {
                        let chunk = &words[k * w..(k + 1) * w];
                        if chunk.iter().any(|&x| x != 0) {
                            let mut cw = [0u64; Q];
                            cw[..w].copy_from_slice(chunk);
                            let v = Int::<Q>::from_words(cw);
                            let mut t = F::from_with_cfg(&v, field_cfg);
                            t *= eq_b;
                            *acc += &t;
                        }
                    }
                }

                // q_3: signed arithmetic shift; skip the lift when the
                // shifted value is zero (non-negative v < radix³).
                let q3_v: Int<Q> = (*entry >> shift3).resize();
                if q3_v.as_uint().to_words().iter().any(|&x| x != 0) {
                    let mut t = F::from_with_cfg(&q3_v, field_cfg);
                    t *= eq_b;
                    q3_eval += &t;
                }
            }
            DynamicPolynomialF::new_trimmed(vec![q0_eval, q1_eval, q2_eval, q3_eval])
        })
        .collect()
}

/// Project a DensePolynomial scalar to DynamicPolynomialF by projecting each
/// coefficient via \phi_q.
pub fn project_scalar_fn<R, F, const D: usize>(
    scalar: &DensePolynomial<R, D>,
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: PrimeField + for<'a> FromWithConfig<&'a R>,
{
    scalar
        .iter()
        .map(|coeff| F::from_with_cfg(coeff, field_cfg))
        .collect()
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::U64;
    use crypto_primitives::FromPrimitiveWithConfig;
    use num_traits::Zero;
    use zinc_utils::{inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar};
    use crypto_primitives::{
        Field, crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use rand::rng;
    use zinc_piop::{
        combined_poly_resolver::CombinedPolyResolverError, multipoint_eval::MultipointEvalError,
    };
    use zinc_poly::univariate::{binary::BinaryPolyInnerProduct, dense::DensePolyInnerProduct};
    use zinc_primality::MillerRabin;
    use zinc_test_uair::{
        BigLinearUair, BigLinearUairWithPublicInput, BinLookup16MultiGroupUair,
        BinLookup16NoLookupUair, BinLookup16Uair, BinaryDecompositionUair,
        IntLookup16OutOfRangeUair, IntLookup16Uair, MixedBinIntLookupUair,
        BitOpRotUair, EC_FP_INT_LIMBS, GenerateRandomTrace, Sha256CompressionSliceUair,
        Sha256Ideal, ShaEcdsaUair, TestUairMixedDegrees, TestUairMixedShifts,
        TestUairNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_uair::{
        ideal::{DegreeOneIdeal, rotation::RotationIdeal},
        ideal_collector::IdealOrZero,
    };
    use zinc_utils::{
        CHECKED,
        field::runtime_monty::Fp,
        from_ref::FromRef,
        inner_product::{MBSInnerProduct, ScalarProduct},
        projectable_to_field::ProjectableToField,
    };
    use zip_plus::{
        code::{
            iprs::{IprsCode, PnttConfigF65537},
            raa::{RaaCode, RaaConfig},
        },
        pcs::structs::{ZipPlus, ZipPlusParams},
        pcs_transcript::PcsProverTranscript,
    };

    const INT_LIMBS: usize = U64::LIMBS;
    // `fixed-prime` branch: 256-bit field modulus (4 × u64 limbs) so the
    // hardcoded secp256k1 base prime fits in `Fmod = Uint<FIELD_LIMBS>`.
    const FIELD_LIMBS: usize = U64::LIMBS * 4;
    const DEGREE_PLUS_ONE: usize = 32;

    // Zip+ type parameters.

    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    /// Repetition factor for linear code, an inverse rate. Defaults to 4
    /// (rate 1/4); enabling the `iprs-rate-1-8` cargo feature switches
    /// every `IprsCode<..., REP, ...>` instance in this test module to
    /// inverse-rate 8 (rate 1/8), and `iprs-rate-1-16` switches to
    /// inverse-rate 16 (rate 1/16). `iprs-rate-1-16` takes precedence if
    /// both are enabled.
    const REP: usize = if cfg!(feature = "iprs-rate-1-16") {
        16
    } else if cfg!(feature = "iprs-rate-1-8") {
        8
    } else {
        4
    };

    /// Number of column openings the PCS performs for `crate::SECURITY_BITS`
    /// bits at rate `1/REP` (150 / 100 / 75 at 100 bits for rates 1/4, 1/8,
    /// 1/16; see `zip_plus::pcs::structs::num_column_openings`).
    const NUM_COL_OPENINGS_FOR_REP: usize =
        zip_plus::pcs::structs::num_column_openings(REP, crate::SECURITY_BITS);

    // Value-sized field with the modulus installed once into `ProofSlot`
    // (drop-in for `MontyField<FIELD_LIMBS>`; see utils/src/field/runtime_monty.rs).
    // `F::make_cfg` (called via `secp256k1_field_cfg`) installs the slot before
    // any field arithmetic runs.
    zinc_utils::define_modulus!(ProofSlot, FIELD_LIMBS);
    type F = Fp<ProofSlot, FIELD_LIMBS>;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypes {}
    impl ZipTypes for BinPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    pub struct ArbitraryPolyZipTypesIprs {}
    impl ZipTypes for ArbitraryPolyZipTypesIprs {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal =
            DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Arbitrary poly ZipTypes with wider codewords for RAA encoding.
    /// RAA accumulation grows the bit-width, so Cw needs more bits than Eval.
    #[derive(Debug, Clone)]
    pub struct ArbitraryPolyZipTypesRaa {}
    impl ZipTypes for ArbitraryPolyZipTypesRaa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<K>, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal =
            DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    type ZtInt = i64;

    #[derive(Debug, Clone)]
    pub struct IntZipTypes {}
    impl ZipTypes for IntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = ZtInt;
        type Cw = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    // ── Transcript-drawn 128-bit projecting prime ─────────────────────────
    //
    // The sound, general Step-1 behaviour (`FIXED_PROJECTING_PRIME = None`):
    // the prime is a 128-bit Fiat–Shamir draw, so it differs per proof and
    // the slot is re-installed each time. That is why this bundle gets its
    // own slot: every other test pins secp256k1 into `ProofSlot`, and tests
    // run in parallel threads.
    const FIELD_LIMBS_128: usize = U64::LIMBS * 2;
    zinc_utils::define_modulus!(RandomPrimeSlot, FIELD_LIMBS_128);
    type F128 = Fp<RandomPrimeSlot, FIELD_LIMBS_128>;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypes128 {}
    impl ZipTypes for BinPolyZipTypes128 {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS_128>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    pub struct ArbitraryPolyZipTypesIprs128 {}
    impl ZipTypes for ArbitraryPolyZipTypesIprs128 {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS_128>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal =
            DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    pub struct IntZipTypes128 {}
    impl ZipTypes for IntZipTypes128 {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = ZtInt;
        type Cw = i128;
        type Fmod = Uint<FIELD_LIMBS_128>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// IPRS bundle with a transcript-drawn 128-bit projecting prime
    /// (`FIXED_PROJECTING_PRIME` left at its `None` default).
    #[derive(Clone, Debug)]
    struct TestZincTypesIprs128;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesIprs128 {
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS_128>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes128;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs128;
        type IntZt = IntZipTypes128;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    #[derive(Clone, Debug)]
    struct TestZincTypesIprs;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesIprs {
        const FIXED_PROJECTING_PRIME: Option<&'static [u8]> =
            Some(&crate::fixed_prime::SECP256K1_P_LE_BYTES);
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    #[derive(Copy, Clone)]
    struct TestRaaConfig;
    impl RaaConfig for TestRaaConfig {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = true;
    }

    #[derive(Clone, Debug)]
    struct TestZincTypesRaa;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesRaa {
        const FIXED_PROJECTING_PRIME: Option<&'static [u8]> =
            Some(&crate::fixed_prime::SECP256K1_P_LE_BYTES);
        type Int = i64;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesRaa;
        type IntZt = IntZipTypes;

        type BinaryLc = RaaCode<Self::BinaryZt, TestRaaConfig, REP>;
        type ArbitraryLc = RaaCode<Self::ArbitraryZt, TestRaaConfig, REP>;
        type IntLc = RaaCode<Self::IntZt, TestRaaConfig, REP>;
    }

    /// Use row size equal to poly size, resulting in flat single-row matrices
    fn make_iprs<Zt: ZipTypes>(num_vars: usize) -> IprsCode<Zt, PnttConfigF65537, REP, CHECKED> {
        let poly_size = 1 << num_vars;
        IprsCode::new_with_optimal_depth(poly_size).unwrap()
    }

    /// Set up Zip+ PCS parameters for a given number of MLE variables.
    #[allow(clippy::type_complexity)]
    fn setup_pp<Zt>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
    ) -> (
        ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
        ZipPlusParams<Zt::IntZt, Zt::IntLc>,
    )
    where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
    {
        let poly_size = 1 << num_vars;
        (
            ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::setup(poly_size, linear_codes.0),
            ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::setup(poly_size, linear_codes.1),
            ZipPlus::<Zt::IntZt, Zt::IntLc>::setup(poly_size, linear_codes.2),
        )
    }

    macro_rules! default_project_ideal {
        () => {
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
        };
    }

    #[allow(clippy::result_large_err)]
    fn do_test<Zt, U>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
        project_ideal: impl Fn(
            &IdealOrZero<U::Ideal>,
            &<F as PrimeField>::Config,
        ) -> IdealOrZero<DegreeOneIdeal<F>>
        + Copy,
        tamper: impl Fn(&mut Proof<F>),
        check_verification: impl Fn(Result<(), ProtocolError<F, IdealOrZero<DegreeOneIdeal<F>>>>),
    ) where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
        Zt::Int: num_traits::Zero,
        <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
        <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
        U: Uair<Scalar = DensePolynomial<Zt::Int, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Zt::Int, Int = Zt::Int>
            + 'static,
        F: for<'a> FromWithConfig<&'a Zt::Int>
            + for<'a> FromWithConfig<&'a <Zt::BinaryZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a <Zt::ArbitraryZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a <Zt::IntZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>,
        <F as Field>::Inner: FromRef<Zt::Fmod>,
        <F as Field>::Modulus: FromRef<Zt::Fmod>,
    {
        do_test_with_field::<Zt, U, F>(num_vars, linear_codes, project_ideal, tamper, check_verification)
    }

    /// `do_test` generic over the projecting field `Fq` (so a bundle whose
    /// prime is transcript-drawn into its own slot can reuse the harness).
    #[allow(clippy::result_large_err)]
    fn do_test_with_field<Zt, U, Fq>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
        project_ideal: impl Fn(
            &IdealOrZero<U::Ideal>,
            &<Fq as PrimeField>::Config,
        ) -> IdealOrZero<DegreeOneIdeal<Fq>>
        + Copy,
        tamper: impl Fn(&mut Proof<Fq>),
        check_verification: impl Fn(Result<(), ProtocolError<Fq, IdealOrZero<DegreeOneIdeal<Fq>>>>),
    ) where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
        Zt::Int: num_traits::Zero,
        <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<Fq>,
        <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<Fq>,
        <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<Fq>,
        <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<Fq>,
        U: Uair<Scalar = DensePolynomial<Zt::Int, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Zt::Int, Int = Zt::Int>
            + 'static,
        Fq: for<'a> FromWithConfig<&'a Zt::Int>
            + for<'a> FromWithConfig<&'a <Zt::BinaryZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a <Zt::ArbitraryZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a <Zt::IntZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>,
        <Fq as Field>::Inner: FromRef<Zt::Fmod>,
        <Fq as Field>::Modulus: FromRef<Zt::Fmod>,
        Fq: InnerTransparentField
            + FromPrimitiveWithConfig
            + for<'a> MulByScalar<&'a Fq>
            + FromRef<Fq>
            + Send
            + Sync
            + 'static,
        <Fq as Field>::Inner: ConstIntSemiring + ConstTranscribable + Send + Sync + Zero + Default,
        <Fq as Field>::Modulus: ConstTranscribable,
        Zt::Int: ProjectableToField<Fq>,
    {
        let mut rng = rng();
        let pp = setup_pp::<Zt>(num_vars, linear_codes);

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = U::signature();
        let public_trace = trace.public(&sig);

        macro_rules! run_protocol {
            ($mle_first:ident) => {
                let mut proof = ZincPlusPiop::<Zt, U, Fq, DEGREE_PLUS_ONE>::prove::<
                    { $mle_first },
                    CHECKED,
                >(&pp, &trace, num_vars, project_scalar_fn)
                .expect("Prover failed");

                // Checking that the proof can be properly serialized and deserialized
                let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
                transcript.write(&proof).expect("Failed to serialize proof");
                let mut transcript = transcript.into_verification_transcript();
                let proof_2 = transcript
                    .read()
                    .expect("Failed to deserialize proof after serialization");
                assert_eq!(proof, proof_2);

                tamper(&mut proof);

                let verification_result =
                    ZincPlusPiop::<Zt, U, Fq, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                        &pp,
                        proof,
                        &public_trace,
                        num_vars,
                        project_scalar_fn,
                        project_ideal,
                    );
                check_verification(verification_result);
            };
        }

        run_protocol!(false);

        // `MLE_FIRST = true` is now safe for any UAIR: it dispatches at
        // runtime to MLE-first (all-linear), Combined (all-non-linear), or
        // Hybrid (mixed). Always exercise it.
        run_protocol!(true);
    }

    /// End-to-end test: TestUairNoMultiplication.
    ///
    /// UAIR constraint: a + b - c \in (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = <X - 2>).
    #[test]
    fn test_e2e_no_multiplication() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairNoMultiplication<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test of the wired GKR-LogUp lookup path: 16 binary_poly
    /// columns, all declared as a single BitPoly{32,8} lookup group
    /// (n_groups = 1 → step-7 two-open fast path). Exercises step4b_lookup
    /// (prove + verify), the chunk-lift parent binding, the proof
    /// serialization round-trip, and the G=1 bin opens at r_inner + r_0.
    #[test]
    fn test_e2e_bin_lookup16() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BinLookup16Uair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// Exercises the >=2-group step-7 bin multipoint reducer: 16 columns
    /// split into a BitPoly{32,8} group (12 cols) and a BitPoly{32,16}
    /// group (4 cols) → n_groups = 2. Validates the wired
    /// BinMultipointReducer prove/verify plus the verifier-side P(r*)
    /// cross-check (the reducer path that G=1 skips).
    #[test]
    fn test_e2e_bin_lookup16_multigroup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BinLookup16MultiGroupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// Negative test: corrupting a lookup chunk-lift coefficient must make
    /// verification fail (the step-4b GKR leaf / parent-binding check
    /// rejects it).
    #[test]
    fn test_e2e_bin_lookup16_tampered_chunk_lift_rejected() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BinLookup16Uair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |proof| {
                let c = &mut proof.lookup_proof.groups[0].chunk_lifts[0][0].coeffs[0];
                *c = c.clone() + c.clone();
            },
            |res| {
                assert!(res.is_err(), "verifier must reject a tampered lookup chunk lift");
            },
        );
    }

    /// End-to-end test of the `Word`-table (integer range-check) lookup:
    /// 4 witness int columns holding 16-bit values, all declared
    /// `Word { width: 16 }`. Exercises `prove_group_int` / `verify_group_int`,
    /// the int multipoint reducer (two claims: r_inner + r_0) and the single
    /// int Zip+ open at its reduced point, plus proof serialization.
    #[test]
    fn test_e2e_int_lookup16() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, IntLookup16Uair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// A `Word`-table lookup on a subset of the int columns (one column is
    /// NOT range-checked, so its `r_inner` eval goes through the non-parent
    /// path) together with a `BitPoly` lookup on binary_poly columns — both
    /// reducers run in one proof.
    #[test]
    fn test_e2e_mixed_bin_int_lookup() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, MixedBinIntLookupUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// Negative tests for the int range check: a tampered per-column eval
    /// at `r_inner` (the lookup's own lift) and a tampered eval at the int
    /// reducer's `r*` must both be rejected.
    #[test]
    fn test_e2e_int_lookup16_tampered_rejected() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, IntLookup16Uair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |proof| {
                let g = &mut proof.lookup_proof.groups[0];
                let e = g.chunk_lifts[0][1].coeffs.first().cloned().unwrap_or_else(F::zero);
                g.chunk_lifts[0][1] = DynamicPolynomialF::new_trimmed(vec![e + F::one()]);
                g.int_evals_at_r_inner[1] = g.int_evals_at_r_inner[1] + F::one();
            },
            |res| assert!(res.is_err(), "tampered r_inner eval must be rejected"),
        );
        do_test::<TestZincTypesIprs, IntLookup16Uair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |proof| {
                proof.int_evals_at_r_star[2] = proof.int_evals_at_r_star[2] + F::one();
            },
            |res| assert!(res.is_err(), "tampered r* eval must be rejected"),
        );
    }

    /// The prover refuses a witness with a cell outside the declared range
    /// (`Word { width: 16 }` with a value of `2^16`): the range check is
    /// not something an honest-looking proof can be produced for.
    #[test]
    fn test_e2e_int_lookup16_out_of_range_witness_rejected() {
        let num_vars = 6;
        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );
        let mut rng = rng();
        let trace = IntLookup16OutOfRangeUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let res = ZincPlusPiop::<TestZincTypesIprs, IntLookup16OutOfRangeUair<ZtInt>, F, DEGREE_PLUS_ONE>::prove::<
            false,
            CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn);
        assert!(
            matches!(res, Err(ProtocolError::Lookup(LookupError::WitnessNotInTable))),
            "an out-of-range int cell must make the prover fail with WitnessNotInTable, got {res:?}"
        );
    }

    /// Opt-in A/B benchmark isolating the GKR-LogUp lookup cost: the
    /// lookup-bearing BinLookup16 UAIRs (G=1 fast path, G=2 reducer path)
    /// vs the no-lookup control with the identical 16-column layout.
    /// Reports prove/verify wall-time (min of reps) and serialized proof
    /// size. Run with:
    ///   cargo test -p zinc-protocol --release -- --ignored --nocapture bench_bin_lookup16_ab
    #[test]
    #[ignore]
    fn bench_bin_lookup16_ab() {
        macro_rules! time_uair {
            ($U:ty, $nv:expr, $reps:expr) => {{
                let num_vars: usize = $nv;
                let mut rng = rng();
                let pp = setup_pp::<TestZincTypesIprs>(
                    num_vars,
                    (make_iprs(num_vars), make_iprs(num_vars), make_iprs(num_vars)),
                );
                let trace =
                    <$U as GenerateRandomTrace<32>>::generate_random_trace(num_vars, &mut rng);
                let sig = <$U as Uair>::signature();
                let public_trace = trace.public(&sig);

                let mut best_prove = f64::MAX;
                let mut proof_bytes = 0usize;
                let mut proof_keep: Option<Proof<F>> = None;
                for _ in 0..$reps {
                    let t = std::time::Instant::now();
                    let proof = ZincPlusPiop::<TestZincTypesIprs, $U, F, DEGREE_PLUS_ONE>::prove::<
                        false,
                        CHECKED,
                    >(&pp, &trace, num_vars, project_scalar_fn)
                    .expect("prove");
                    best_prove = best_prove.min(t.elapsed().as_secs_f64() * 1e3);
                    proof_bytes = proof.get_num_bytes();
                    proof_keep = Some(proof);
                }

                let mut best_verify = f64::MAX;
                for _ in 0..$reps {
                    let proof = proof_keep.clone().expect("proof");
                    let t = std::time::Instant::now();
                    ZincPlusPiop::<TestZincTypesIprs, $U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                        &pp,
                        proof,
                        &public_trace,
                        num_vars,
                        project_scalar_fn,
                        |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
                    )
                    .expect("verify");
                    best_verify = best_verify.min(t.elapsed().as_secs_f64() * 1e3);
                }
                (best_prove, best_verify, proof_bytes)
            }};
        }

        println!("\n== BinLookup16 lookup A/B (16 bin cols, IPRS, CHECKED, min of reps) ==");
        for &nv in &[8usize, 10, 12] {
            let reps = if nv >= 12 { 3 } else { 6 };
            let (np, nq, nb) = time_uair!(BinLookup16NoLookupUair<ZtInt>, nv, reps);
            let (lp, lv, lb) = time_uair!(BinLookup16Uair<ZtInt>, nv, reps);
            let (gp, gv, gb) = time_uair!(BinLookup16MultiGroupUair<ZtInt>, nv, reps);
            println!("\nnv={nv}  ({} rows)", 1usize << nv);
            println!("  no-lookup ctl : prove {np:8.2} ms | verify {nq:7.2} ms | proof {nb:7} B");
            println!(
                "  lookup  (G=1) : prove {lp:8.2} ms | verify {lv:7.2} ms | proof {lb:7} B   (Δ +{:.2} / +{:.2} ms, +{} B)",
                lp - np,
                lv - nq,
                lb as i64 - nb as i64
            );
            println!(
                "  lookup  (G=2) : prove {gp:8.2} ms | verify {gv:7.2} ms | proof {gb:7} B   (Δ +{:.2} / +{:.2} ms, +{} B, reducer path)",
                gp - np,
                gv - nq,
                gb as i64 - nb as i64
            );
        }
        println!();
    }

    /// End-to-end test: TestUairSimpleMultiplication.
    ///
    /// UAIR constraints (3 total, no ideals):
    ///   up[0] * up[1] = down[0]
    ///   up[1] * up[2] = down[1]
    ///   up[0] * up[2] = down[2]
    ///
    /// Uses RAA code with small num_vars (2) because chained polynomial
    /// multiplication causes exponential growth in both degree and coefficient
    /// magnitude. With num_vars=2 (4 rows), max degree=6 and max coefficient
    /// ~= 127^8 ~= 2^56, which fits in i64.
    #[test]
    fn test_e2e_simple_multiplication() {
        let num_vars = 2;
        do_test::<TestZincTypesRaa, TestUairSimpleMultiplication<ZtInt>>(
            num_vars,
            (
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    #[test]
    fn test_e2e_simple_multiplication_false_witness() {
        type Zt = TestZincTypesRaa;
        type U = TestUairSimpleMultiplication<ZtInt>;
        let num_vars = 2;
        let pp = setup_pp::<Zt>(
            num_vars,
            (
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
            ),
        );
        let mut trace = U::generate_random_trace(num_vars, &mut rng());
        trace.arbitrary_poly.to_mut()[0].evaluations[1].coeffs[0] += 1;
        let public_trace = trace.public(&U::signature());

        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");
        let res = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
        );
        assert!(matches!(res, Err(ProtocolError::AssertZero(_))));
    }

    #[test]
    fn test_e2e_sha_ecdsa_false_witness() {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;
        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let mut trace = U::generate_random_trace(SHA_ECDSA_NUM_VARS, &mut rng());
        trace.int.to_mut()[zinc_test_uair::sha_ecdsa::cols::ECDSA_W_Z_PA].evaluations[5] +=
            ShaEcdsaInt::from(1);
        let public_trace = trace.public(&U::signature());

        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("Prover failed");
        let res = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
            sha256_test_project_ideal,
        );
        assert!(matches!(res, Err(ProtocolError::AssertZero(_))));
    }

    /// End-to-end test: TestUairMixedDegrees.
    ///
    /// Two non-zero-ideal `assert_in_ideal` constraints — one linear
    /// (degree 1), one quadratic (degree 2). Exercises the hybrid
    /// ideal-check dispatch (`prove_hybrid`), which routes the linear
    /// constraint through the MLE-first lane and the quadratic constraint
    /// through the combined-poly lane, merging the per-constraint values
    /// into a single proof. Honest witness is the all-zero trace, which
    /// trivially satisfies both constraints.
    #[test]
    fn test_e2e_mixed_degrees() {
        // Use TestZincTypesRaa because the quadratic constraint with
        // arbitrary_poly column multiplication needs an RAA-style code.
        let num_vars = 2;
        do_test::<TestZincTypesRaa, TestUairMixedDegrees<ZtInt>>(
            num_vars,
            (
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
            ),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: TestUairMixedShifts.
    ///
    /// Uses mixed shift amounts (col a: shift 1, col b: shift 2).
    /// Constraints: a[i+1] = a[i] + b[i], c[i] = b[i+2].
    #[test]
    fn test_e2e_mixed_shifts() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairMixedShifts<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: BinaryDecompositionUair.
    ///
    /// Uses binary_poly (1 col) and int (1 col) trace types.
    /// UAIR constraint: binary_poly[0] - int[0] \in <X - 2>
    #[test]
    fn test_e2e_binary_decomposition() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BinaryDecompositionUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: BigLinearUair.
    ///
    /// Uses 16 binary_poly cols and 1 int col.
    /// UAIR constraints:
    ///   sum(up.binary_poly[0..16]) - up.int[0] \in <X - 1>
    ///   down.binary_poly[0] - up.int[0] \in <X - 2>
    ///   up.binary_poly[i] - down.binary_poly[i] = 0, for i=1..15
    #[test]
    fn test_e2e_big_linear() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test of the transcript-drawn projecting prime
    /// (`FIXED_PROJECTING_PRIME = None`, 128-bit `Fmod = Uint<2>`): the
    /// prover draws `q` from the transcript after step 0, the verifier
    /// re-derives it, and the value-sized field slot is re-installed per
    /// proof. Runs two ideal-bearing UAIRs (binary+int lanes) honestly,
    /// then checks that a tampered lifted eval is rejected.
    /// All three runs live in ONE test because they share
    /// `RandomPrimeSlot` and a slot must not be re-installed concurrently.
    #[test]
    fn test_e2e_random_projecting_prime_128() {
        let num_vars = 8;
        let codes = || {
            (
                make_iprs::<BinPolyZipTypes128>(num_vars),
                make_iprs::<ArbitraryPolyZipTypesIprs128>(num_vars),
                make_iprs::<IntZipTypes128>(num_vars),
            )
        };
        do_test_with_field::<TestZincTypesIprs128, BigLinearUair<ZtInt>, F128>(
            num_vars,
            codes(),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
        do_test_with_field::<TestZincTypesIprs128, BinaryDecompositionUair<ZtInt>, F128>(
            num_vars,
            codes(),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
        do_test_with_field::<TestZincTypesIprs128, BigLinearUair<ZtInt>, F128>(
            num_vars,
            codes(),
            default_project_ideal!(),
            |proof| {
                let c = &mut proof.witness_lifted_evals[0].coeffs[0];
                *c = c.clone() + c.clone() + F128::one_with_cfg(&());
            },
            |res| assert!(res.is_err(), "tampered lifted eval must be rejected"),
        );
    }

    /// End-to-end test: BigLinearUairWithPublicInput.
    ///
    /// Same as [`BigLinearUair`], but with the first few binary_poly columns as
    /// public inputs.
    #[test]
    fn test_e2e_big_linear_with_public_input() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    //
    // Negative tests for BigLinearUairWithPublicInput: verify that proof
    // tampering is detected.
    //

    #[test]
    fn test_big_linear_tamper_lifted_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.witness_lifted_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                ));
            },
        );
    }

    #[test]
    fn test_big_linear_tamper_up_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.resolver.up_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(
                        CombinedPolyResolverError::ClaimValueDoesNotMatch { .. }
                    )
                ));
            },
        );
    }

    #[test]
    fn test_big_linear_tamper_down_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.resolver.down_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(
                        CombinedPolyResolverError::ClaimValueDoesNotMatch { .. }
                    )
                ));
            },
        );
    }

    // Tampering the commitment root causes the verifier to derive different
    // challenges from the Fiat–Shamir transcript. On the `fixed-prime`
    // branch the projecting prime `q` is hardcoded (not transcript-derived),
    // so prover and verifier still agree on `q` after the tamper; the first
    // observable divergence is at the combined-poly resolver, which catches
    // it as a sumcheck-sum mismatch.
    #[test]
    fn test_big_linear_tamper_commitment() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.commitments.0.root = Default::default(),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(
                        CombinedPolyResolverError::WrongSumcheckSum { .. }
                    )
                ));
            },
        );
    }

    /// End-to-end test: BitOpRotUair (synthetic UAIR with one
    /// `BitOp::Rot(7)` virtual column).
    ///
    /// Two binary witness columns W (col 0) and V (col 1); witness sets
    /// V[i] = Rot(7)(W[i]). Constraint: V[i] − Rot(7)(W[i]) ∈ <X − 2>.
    /// Exercises the bit-op virtual column path end-to-end: CPR
    /// materialises an extra down MLE for the bit-op, the prover
    /// publishes a `bit_op_down_evals` entry, and the verifier checks
    /// it in Step 4.5 against ψ(rot_c(lifted_eval[col 0])).
    #[test]
    fn test_e2e_bit_op_rot() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// Tampering the CPR-emitted `bit_op_down_evals` triggers the new
    /// `LiftedAtRStarBitOpMismatch` error at Step 4.5.
    ///
    /// We tamper the F_q[X]-lifted source eval at r* — easier to
    /// engineer than tampering `bit_op_down_evals` directly, because
    /// the latter participates in the CPR claim-value reconstruction
    /// (so the verifier rejects earlier with `ClaimValueDoesNotMatch`).
    /// Tampering the source's `lifted_evals_at_rstar` makes the source's
    /// up-eval ψ-projection still match `cpr_subclaim.up_evals[0]`
    /// only with extreme luck — but with a single coefficient swap the
    /// up-eval check fires first. To target the bit-op check, we
    /// instead permute coefficients of the source's lifted eval such
    /// that ψ_α projects to the same value (preserves dot product
    /// against `α^j`); a swap that holds the dot product fixed but
    /// changes `rot_c(·)` is not directly engineerable in a black-box
    /// test. As a robust proxy: tamper `bit_op_down_evals[0]` and
    /// observe the verifier rejects (whatever the precise error
    /// variant). For this test we verify it does reject — we check for
    /// the bit-op mismatch when the ResolveCheckValue happens to pass,
    /// otherwise any rejection is acceptable.
    #[test]
    fn test_bit_op_rot_tamper_bit_op_down_eval() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| {
                // Mutate the only bit_op_down_eval — verifier must
                // reject (CPR claim-value reconstruction will catch it
                // first as a `ClaimValueDoesNotMatch`).
                if let Some(ev) = proof.resolver.bit_op_down_evals.first_mut() {
                    *ev = ev.clone() + ev.clone();
                }
            },
            |res| {
                assert!(res.is_err());
            },
        );
    }

    /// Tamper a coefficient of the source's witness lifted eval at
    /// r_0 (W_W's slot in `proof.witness_lifted_evals`). The swap
    /// changes `rot_c(·)` (so the bit-op slot's derived `open_eval`
    /// no longer matches mp_eval's expectation) and at the same time
    /// changes the source's own `open_eval`. The verifier rejects via
    /// mp_eval's `ClaimMismatch` (whichever slot trips the equation
    /// first — the per-slot identity is not separately surfaced).
    #[test]
    fn test_bit_op_rot_tamper_witness_lifted_source() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| {
                if let Some(p) = proof.witness_lifted_evals.first_mut() {
                    if p.coeffs.len() >= 2 {
                        p.coeffs.swap(0, 1);
                    }
                }
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                        | ProtocolError::LiftedEvalsBitOpMismatch { .. }
                ));
            },
        );
    }

    //
    // SHA-ECDSA E2E + tampering tests for the new Step 4.5 layer.
    //
    // These pin the post-Commit-D behaviour:
    //   * lifted_evals_at_rstar up-half tamper rejected
    //     with `LiftedAtRStarUpMismatch`.
    //   * lifted_evals_at_rstar down-half tamper rejected
    //     with `LiftedAtRStarDownMismatch`.
    //   * SHA `W_W` source tamper rejected upstream of mp_eval (any
    //     `LiftedAtRStar*` variant).
    //   * No-tamper proof-shape pin: `lifted_evals_at_rstar.len()`
    //     and `bit_op_down_evals.len()` match the SHA-ECDSA
    //     signature.
    //

    type ShaEcdsaInt = Int<EC_FP_INT_LIMBS>;

    /// Binary-poly Zip+ types tuned for the wider SHA-ECDSA cell type
    /// (Int<5>, 320-bit). Mirrors `BinPolyZipTypes` but with a wider
    /// `CombR` to soak up SHA-ECDSA's per-row inner products.
    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesShaEcdsa {}
    impl ZipTypes for BinPolyZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Arbitrary-poly Zip+ types over `Int<EC_FP_INT_LIMBS>` cells.
    /// SHA-ECDSA itself has no arbitrary-poly columns; this is only
    /// here to satisfy the `ZincTypes` bundle.
    #[derive(Debug, Clone)]
    pub struct ArbPolyZipTypesShaEcdsa {}
    impl ZipTypes for ArbPolyZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = DensePolynomial<ShaEcdsaInt, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<6>, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = DensePolyInnerProduct<
            ShaEcdsaInt,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Int Zip+ types over `Int<EC_FP_INT_LIMBS>` cells (the ECDSA
    /// Jacobian columns and the SHA mu_{W,a,e} carries).
    #[derive(Debug, Clone)]
    pub struct IntZipTypesShaEcdsa {}
    impl ZipTypes for IntZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = ShaEcdsaInt;
        type Cw = Int<6>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    // ── 4× int-fold variant of the ShaEcdsa types ──────────────────────
    //
    // For the `prove_folded_4x` round-trip test below.
    // Binary: `BinaryPoly<8>` quartered (matches existing
    // `BenchFoldedRealEcdsaZincTypes4x` from `protocol/benches/e2e.rs`).
    // Int: `Int<INT_QUARTER_LIMBS_TEST>` quartered.
    //
    // With ECDSA's centered representation `|v| < 2^255` (commit
    // `dde6f2e`), the 64-bit signed quarters `q_3 = (v >> 192).resize()`
    // satisfy `|q_3| < 2^63` so the upper quarter never overflows
    // `Int<2>`'s positive range; the lower three quarters are
    // bit-extracted single source limbs into `Int<2>` (top bit zeroed,
    // always non-negative). `Cw = Int<3>` gives one limb of headroom
    // for the encoder accumulator.

    const QUARTER_DEGREE_PLUS_ONE_TEST: usize = 8;
    const HALF_DEGREE_PLUS_ONE_TEST: usize = 16;
    const INT_QUARTER_LIMBS_TEST: usize = 2;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesShaEcdsaQuarter {}
    impl ZipTypes for BinPolyZipTypesShaEcdsaQuarter {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<QUARTER_DEGREE_PLUS_ONE_TEST>;
        type Cw = DensePolynomial<i64, QUARTER_DEGREE_PLUS_ONE_TEST>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, QUARTER_DEGREE_PLUS_ONE_TEST>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, QUARTER_DEGREE_PLUS_ONE_TEST>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            QUARTER_DEGREE_PLUS_ONE_TEST,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    pub struct IntZipTypesShaEcdsaQuarter {}
    impl ZipTypes for IntZipTypesShaEcdsaQuarter {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = Int<INT_QUARTER_LIMBS_TEST>;
        type Cw = Int<{ INT_QUARTER_LIMBS_TEST + 1 }>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<6>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct TestShaEcdsaFolded4xZincTypes;

    impl
        IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE_TEST,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_TEST,
        > for TestShaEcdsaFolded4xZincTypes
    {
        const FIXED_PROJECTING_PRIME: Option<&'static [u8]> =
            Some(&crate::fixed_prime::SECP256K1_P_LE_BYTES);
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesShaEcdsaQuarter;
        type ArbitraryZt = ArbPolyZipTypesShaEcdsa;
        type IntZt = IntZipTypesShaEcdsaQuarter;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    #[allow(clippy::type_complexity)]
    fn setup_folded_4x_pp_sha_ecdsa(
        num_vars: usize,
    ) -> (
        ZipPlusParams<
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::BinaryZt,
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::BinaryLc,
        >,
        ZipPlusParams<
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::ArbitraryZt,
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::ArbitraryLc,
        >,
        ZipPlusParams<
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::IntZt,
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::IntLc,
        >,
    ) {
        let split4_size = 1 << (num_vars + 2);
        let normal_size = 1 << num_vars;
        (
            ZipPlus::setup(
                split4_size,
                IprsCode::new_with_optimal_depth(split4_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
            ZipPlus::setup(
                split4_size,
                IprsCode::new_with_optimal_depth(split4_size).unwrap(),
            ),
        )
    }

    /// `ZincTypes` bundle wiring SHA-ECDSA's `Int<5>` cells through
    /// IPRS-coded Zip+ commitments. Mirrors `RealEcdsaBenchZincTypes`
    /// from `protocol/benches/e2e.rs` (which already exercises this
    /// configuration in production benchmarks).
    #[derive(Clone, Debug)]
    struct TestShaEcdsaZincTypes;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestShaEcdsaZincTypes {
        const FIXED_PROJECTING_PRIME: Option<&'static [u8]> =
            Some(&crate::fixed_prime::SECP256K1_P_LE_BYTES);
        type Int = ShaEcdsaInt;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesShaEcdsa;
        type ArbitraryZt = ArbPolyZipTypesShaEcdsa;
        type IntZt = IntZipTypesShaEcdsa;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    /// Project an `IdealOrZero<Sha256Ideal<ShaEcdsaInt>>` to
    /// `Sha256Ideal<F>` for the SHA-ECDSA verifier. Mirrors
    /// `sha256_real_project_ideal` in `protocol/benches/e2e.rs`.
    fn sha256_test_project_ideal(
        ideal: &IdealOrZero<Sha256Ideal<ShaEcdsaInt>>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Sha256Ideal<F> {
        match ideal {
            IdealOrZero::NonZero(Sha256Ideal::RotX2(r)) => {
                Sha256Ideal::RotX2(RotationIdeal::from_with_cfg(r, field_cfg))
            }
            IdealOrZero::NonZero(Sha256Ideal::RotXw1) => Sha256Ideal::RotXw1,
            IdealOrZero::Zero => {
                unreachable!("zero ideals are filtered before this closure runs")
            }
        }
    }

    /// Run a SHA-ECDSA round-trip end-to-end. Calls `tamper` on the
    /// generated proof before verification and feeds the resulting
    /// `Result` into `check_verification`. Patterned after `do_test`
    /// but specialised to the SHA-ECDSA UAIR / `Sha256Ideal` ideal
    /// type (which doesn't fit the `IdealOrZero<DegreeOneIdeal<F>>`
    /// signature `do_test` hard-codes).
    ///
    /// `MLE_FIRST` is forced to `false` since SHA-ECDSA has constraints
    /// up to degree 6 (the ECDSA Y output-selection D4 term), which
    /// `count_effective_max_degree` reports above 1.
    #[allow(clippy::result_large_err)]
    fn do_test_sha_ecdsa(
        num_vars: usize,
        tamper: impl Fn(&mut Proof<F>),
        check_verification: impl Fn(Result<(), ProtocolError<F, Sha256Ideal<F>>>),
    ) {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        let mut rng = rng();
        let pp = setup_pp::<Zt>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = <U as Uair>::signature();
        let public_trace = trace.public(&sig);

        let mut proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // Round-trip the proof through (de)serialisation as a sanity
        // check; mirrors `do_test`.
        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript.write(&proof).expect("Failed to serialize proof");
        let mut transcript = transcript.into_verification_transcript();
        let proof_2 = transcript
            .read()
            .expect("Failed to deserialize proof after serialization");
        assert_eq!(proof, proof_2);

        tamper(&mut proof);

        let verification_result = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<
            _,
            CHECKED,
        >(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            sha256_test_project_ideal,
        );
        check_verification(verification_result);
    }

    /// `num_vars` for SHA-ECDSA tests. ECDSA's Shamir scalar
    /// multiplication needs `n_rows > 256`, so `num_vars >= 9`.
    const SHA_ECDSA_NUM_VARS: usize = 9;

    /// Tamper a coefficient of the SHA `W_W` slot in
    /// `proof.witness_lifted_evals` (the source of all six bit-op
    /// virtual columns). The verifier rejects via mp_eval's
    /// consistency check or `LiftedEvalsBitOpMismatch` (whichever
    /// trips first). Replaces the four pre-existing Step-4.5-specific
    /// tamper tests since Step 4.5 is gone.
    #[test]
    fn test_e2e_sha_ecdsa_tamper_witness_lifted_w() {
        do_test_sha_ecdsa(
            SHA_ECDSA_NUM_VARS,
            |proof| {
                // SHA-ECDSA's witness layout puts W_W at the same flat
                // index as in `cols::W_W` minus `NUM_BIN_PUB`. Easier
                // and equally diagnostic to tamper the first witness
                // slot instead — any of them feeds mp_eval.
                let p = &mut proof.witness_lifted_evals[0];
                assert!(
                    p.coeffs.len() >= 2,
                    "witness lifted eval polynomial has < 2 coefficients; cannot tamper",
                );
                p.coeffs.swap(0, 1);
            },
            |res| {
                let err = res.unwrap_err();
                assert!(
                    matches!(
                        err,
                        ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                            | ProtocolError::LiftedEvalsBitOpMismatch { .. },
                    ),
                    "expected mp_eval ClaimMismatch or LiftedEvalsBitOpMismatch, got {err:?}",
                );
            },
        );
    }

    /// 4×-folded ShaEcdsa round-trip — binary AND int both quartered
    /// (BinaryPoly<8> / Int<2>) and committed under one Merkle tree
    /// via `MultiZip3`. Prints the serialized proof size.
    #[test]
    fn test_e2e_sha_ecdsa_folded_4x_round_trip() {
        use crate::prover::prove_folded_4x;
        use crate::verifier::verify_folded_4x;

        type ZtF = TestShaEcdsaFolded4xZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        let mut rng = rng();
        let pp = setup_folded_4x_pp_sha_ecdsa(SHA_ECDSA_NUM_VARS);
        let trace = U::generate_random_trace(SHA_ECDSA_NUM_VARS, &mut rng);
        let sig = <U as Uair>::signature();
        let public_trace = trace.public(&sig);

        let proof = prove_folded_4x::<
            ZtF,
            U,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE_TEST,
            QUARTER_DEGREE_PLUS_ONE_TEST,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_TEST,
            false,
            CHECKED,
        >(&pp, &trace, SHA_ECDSA_NUM_VARS, project_scalar_fn)
        .expect("4× int-fold prover failed");

        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript.write(&proof).expect("Failed to serialize proof");
        let serialized_len = transcript.stream.get_ref().len();
        println!(
            "4× folded ShaEcdsa proof size: {} bytes ({} KiB)",
            serialized_len,
            serialized_len.div_ceil(1024),
        );
        let mut transcript = transcript.into_verification_transcript();
        let proof_2 = transcript
            .read()
            .expect("Failed to deserialize proof");
        assert_eq!(proof, proof_2);

        verify_folded_4x::<
            ZtF,
            U,
            F,
            Sha256Ideal<F>,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE_TEST,
            QUARTER_DEGREE_PLUS_ONE_TEST,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_TEST,
            CHECKED,
        >(
            &pp,
            proof,
            &public_trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
            sha256_test_project_ideal,
        )
        .expect("Verifier rejected an honest 4× folded ShaEcdsa proof");
    }

    /// No-tamper SHA-ECDSA round-trip + structural-shape pins. Prints
    /// the proof size so refactors that grow the proof are easy to
    /// catch.
    #[test]
    fn test_e2e_sha_ecdsa_proof_shape() {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        // SHA standalone signature pins: NUM_BIN = 20 (9 public + 11
        // witness — 9 = PA_A, PA_E, 4× PA_OV_*, 2× PA_R_*_CORR, PA_M).
        // bit_op_specs = 6, virtual_binary_poly_cols = 3.
        let sha_sig = <Sha256CompressionSliceUair<ShaEcdsaInt> as Uair>::signature();
        assert_eq!(
            sha_sig.total_cols().num_binary_poly_cols(),
            20,
            "SHA-256 NUM_BIN drifted: expected 20 binary_poly columns",
        );
        assert_eq!(
            sha_sig.bit_op_specs().len(),
            11,
            "SHA-256 bit_op_specs.len() drifted: expected 11 (6 σ_0/σ_1 + 5 W_MU_PACKED ShiftRs)",
        );
        assert_eq!(
            sha_sig.virtual_binary_poly_cols().len(),
            3,
            "SHA-256 virtual_binary_poly_cols.len() drifted: expected 3 (B_1/B_2/B_3)",
        );

        // SHA-ECDSA composed signature (the actual UAIR exercised here).
        let sig = <U as Uair>::signature();
        let num_bit_op = sig.bit_op_specs().len();
        assert_eq!(
            sig.total_cols().num_binary_poly_cols(),
            20,
            "SHA-ECDSA NUM_BIN drifted: expected 20 binary_poly columns",
        );
        assert_eq!(
            num_bit_op, 11,
            "SHA-ECDSA bit_op_specs.len() drifted: expected 11",
        );
        assert_eq!(
            sig.virtual_binary_poly_cols().len(),
            3,
            "SHA-ECDSA virtual_binary_poly_cols.len() drifted: expected 3",
        );

        // Round-trip a real proof and pin the post-rewrite proof size.
        let mut rng = rng();
        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let trace = U::generate_random_trace(SHA_ECDSA_NUM_VARS, &mut rng);
        let public_trace = trace.public(&sig);

        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("Prover failed");

        assert_eq!(
            proof.resolver.bit_op_down_evals.len(),
            num_bit_op,
            "Proof.resolver.bit_op_down_evals.len() must equal bit_op_specs.len()",
        );

        let total_proof_bytes = proof.get_num_bytes();
        println!("total proof bytes: {total_proof_bytes}");

        // Verifier still accepts the un-tampered proof.
        ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
            sha256_test_project_ideal,
        )
        .expect("Verifier rejected an honest SHA-ECDSA proof");
    }

    #[test]
    fn test_big_linear_tamper_ideal_check() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.ideal_check.combined_mle_values.swap(0, 1),
            |res| {
                assert!(matches!(res.unwrap_err(), ProtocolError::IdealCheck(..)));
            },
        );
    }

    //
    // Folded Zip+ (1× fold) — round-trip test
    //

    /// Half-degree binary Zip+ types for the split commitment side of the
    /// folded path. Mirrors [`BinPolyZipTypes`] but with `Eval = BinaryPoly<16>`
    /// and `Cw` over `DensePolynomial<i64, 16>`, so the PCS commits the
    /// post-split BinaryPoly<16> witnesses.
    const HALF_DEGREE_PLUS_ONE: usize = 16;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesHalf {}
    impl ZipTypes for BinPolyZipTypesHalf {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<HALF_DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, HALF_DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, HALF_DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, HALF_DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            HALF_DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct TestFoldedZincTypesIprs;

    impl FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE> for TestFoldedZincTypesIprs {
        const FIXED_PROJECTING_PRIME: Option<&'static [u8]> =
            Some(&crate::fixed_prime::SECP256K1_P_LE_BYTES);
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesHalf;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    /// Set up Zip+ params for the folded path. The binary commitment is over
    /// the split column (length `2n` with `BinaryPoly<HALF_D>` entries), so
    /// its `num_vars` is `num_vars + 1`. Arbitrary and int are sized normally.
    #[allow(clippy::type_complexity)]
    fn setup_folded_pp(
        num_vars: usize,
    ) -> (
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::BinaryZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::BinaryLc,
        >,
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::ArbitraryZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::ArbitraryLc,
        >,
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::IntZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::IntLc,
        >,
    ) {
        let split_size = 1 << (num_vars + 1);
        let normal_size = 1 << num_vars;
        (
            ZipPlus::setup(
                split_size,
                IprsCode::new_with_optimal_depth(split_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
        )
    }

    /// End-to-end test: BinaryDecompositionUair via the **folded** prover/
    /// verifier. Same UAIR, same trace generator, same field — only the
    /// binary commitment is over `BinaryPoly<16>` split columns, opened at
    /// the extended point `(r_0 ‖ γ)`.
    #[test]
    fn test_e2e_folded_binary_decomposition() {
        use crate::prover::prove_folded;
        use crate::verifier::verify_folded;

        let num_vars = 8;
        let mut rng = rng();
        let pp = setup_folded_pp(num_vars);

        let trace = BinaryDecompositionUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let sig = <BinaryDecompositionUair<ZtInt> as Uair>::signature();
        let public_trace = trace.public(&sig);

        let proof = prove_folded::<
            TestFoldedZincTypesIprs,
            BinaryDecompositionUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            false,
            CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
        .expect("Folded prover failed");

        verify_folded::<
            TestFoldedZincTypesIprs,
            BinaryDecompositionUair<ZtInt>,
            F,
            IdealOrZero<DegreeOneIdeal<F>>,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            CHECKED,
        >(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            default_project_ideal!(),
        )
        .expect("Folded verifier rejected a valid proof");
    }

    //
    // Folded Zip+ (4× fold) — round-trip test
    //

    /// Quarter-degree binary Zip+ types for the doubly-split commitment side
    /// of the 4× folded path. Mirrors [`BinPolyZipTypesHalf`] but with
    /// `Eval = BinaryPoly<8>` and 8-coeff codewords.
    const QUARTER_DEGREE_PLUS_ONE: usize = 8;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesQuarter {}
    impl ZipTypes for BinPolyZipTypesQuarter {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<QUARTER_DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, QUARTER_DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, QUARTER_DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, QUARTER_DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            QUARTER_DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

}
