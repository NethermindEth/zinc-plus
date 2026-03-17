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
//! - Step 4: finite-field sumcheck over F_q
//! - Step 5: multi-point evaluation sumcheck (combines up/down evals at r' into
//!   a single evaluation point r_0)
//! - Step 6: lift-and-project (unprojected MLE evaluations at r_0)
//! - Step 7: Zip+ PCS open/verify at r_0

pub mod prover;
pub mod verifier;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{ConstIntRing, ConstIntSemiring, FromWithConfig, PrimeField};
use std::marker::PhantomData;
use thiserror::Error;
use zinc_piop::{
    combined_poly_resolver::{CombinedPolyResolverError, Proof as CombinedPolyResolverProof},
    ideal_check::{IdealCheckError, Proof as IdealCheckProof},
    multipoint_eval::{MultipointEvalError, Proof as MultipointEvalProof},
    projections::RowMajorTrace,
};
use zinc_poly::{
    ConstCoeffBitWidth, EvaluationError as PolyEvaluationError,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{Uair, ideal::Ideal};
use zinc_utils::{cfg_into_iter, cfg_iter, named::Named};
use zip_plus::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlusCommitment, ZipTypes},
};

//
// Data structures
//

/// Full proof produced by the Zinc+ PIOP for UCS.
#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    /// Zip+ commitments to the witness columns.
    pub commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip: Vec<u8>,
    /// Randomized ideal check proof.
    pub ideal_check: IdealCheckProof<F>,
    /// Combined polynomial resolver proof (F_q sumcheck).
    pub resolver: CombinedPolyResolverProof<F>,
    /// Multi-point evaluation sumcheck proof (combines up_evals and
    /// down_evals at r' into a single evaluation point r_0).
    pub multipoint_eval: MultipointEvalProof<F>,
    /// Witness-only polynomial MLE evaluations at r_0 in F_q[X]
    /// (after \phi_q, before \psi_a), ordered as
    /// `[wit_bin..., wit_arb..., wit_int...]`.
    /// The verifier recomputes public lifted_evals from public data,
    /// interleaves them with these, and derives scalar open_evals via
    /// \psi_a for the sumcheck consistency check and Zip+ PCS verify.
    pub witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
}

/// Trait bundling the various type parameters for the public inputs (NYI),
/// witness and Zinc+ PIOP.
pub trait ZincTypes<const DEGREE_PLUS_ONE: usize> {
    /// Main integer type for the protocol, used as a coefficient type for the
    /// arbitrary polynomial trace columns and for the integer trace columns.
    type Int: ConstTranscribable + ConstCoeffBitWidth + Named + Default + Clone + Send + Sync;

    /// Projecting element to project Zip+ evaluations and UAIR scalars to the
    /// field.
    type Chal: ConstIntRing + ConstTranscribable + Named;

    /// Evaluation point type, used for all column types in Zip+ to evaluate
    /// multilinear polynomials.
    type Pt: ConstIntRing;

    type CombR;

    /// Randomly sampled field modulus type, used throughout the protocol for
    /// finite field operations.
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;

    /// Primality test for the field modulus.
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// Zip+ types for the binary polynomial trace columns.
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns.
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the integer trace columns.
    type IntZt: ZipTypes<
            Eval = Self::Int,
            Chal = Self::Chal,
            Pt = Self::Pt,
            CombR = Self::CombR,
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
    #[error("PCS error: {0}")]
    Pcs(#[from] ZipError),
    #[error("PCS verification failed at column {0}: {1}")]
    PcsVerification(usize, ZipError),
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
            entry.write_transcription_bytes(&mut buf);
            transcript.absorb_slice(&buf);
        }
    }
}

/// Compute per-column lifted MLE evaluations at `point`.
///
/// For each column j, returns `\sum_b eq(b, point) * v_j(b)` as a polynomial
/// in `F_q[X]` (coefficient-wise MLE evaluation).
///
/// Binary columns exploit the 0/1 structure for conditional additions only.
/// The `eq(point, *)` table is built once and reused across all columns.
#[allow(clippy::arithmetic_side_effects)]
fn compute_lifted_evals<F: PrimeField, const D: usize>(
    point: &[F],
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_trace: &RowMajorTrace<F>,
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>> {
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_lifted_evals: eq table build failed");

    let n_bin = trace_bin_poly.len();
    let num_cols = projected_trace.first().map(|r| r.len()).unwrap_or(0);
    let zero = F::zero_with_cfg(field_cfg);

    cfg_iter!(trace_bin_poly)
        .map(|col| {
            let mut coeffs = vec![zero.clone(); D];
            for (b, entry) in col.iter().enumerate() {
                for (l, coeff) in entry.iter().enumerate() {
                    if coeff.into_inner() {
                        coeffs[l] += &eq_table[b];
                    }
                }
            }
            coeffs
        })
        .chain(cfg_into_iter!(n_bin..num_cols).map(|col_idx| {
            let num_coeffs = projected_trace
                .iter()
                .map(|row| row[col_idx].coeffs.len())
                .max()
                .unwrap_or(0);

            let mut coeffs = vec![zero.clone(); num_coeffs];
            for (b, row) in projected_trace.iter().enumerate() {
                let entry = &row[col_idx];
                for (l, coeff) in entry.coeffs.iter().enumerate() {
                    let mut term = eq_table[b].clone();
                    term *= coeff;
                    coeffs[l] += &term;
                }
            }
            coeffs
        }))
        .map(DynamicPolynomialF::new_trimmed)
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
    use crypto_primitives::{
        crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use rand::rng;
    use zinc_poly::univariate::{binary::BinaryPolyInnerProduct, dense::DensePolyInnerProduct};
    use zinc_primality::MillerRabin;
    use zinc_test_uair::{
        BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair,
        GenerateMultiTypeWitness, GenerateSingleTypeWitness, TestAirNoMultiplication,
        TestUairSimpleMultiplication,
    };
    use zinc_uair::{ideal::degree_one::DegreeOneIdeal, ideal_collector::IdealOrZero};
    use zinc_utils::{
        CHECKED,
        inner_product::{MBSInnerProduct, ScalarProduct},
    };
    use zip_plus::{
        code::{
            iprs::{IprsCode, PnttConfigF2_16_1},
            raa::{RaaCode, RaaConfig},
        },
        pcs::structs::{ZipPlus, ZipPlusParams},
    };

    const INT_LIMBS: usize = U64::LIMBS;
    const FIELD_LIMBS: usize = 4;
    const DEGREE_PLUS_ONE: usize = 32;

    // Zip+ type parameters.

    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const IPRS_DEPTH: usize = 1;

    type F = MontyField<FIELD_LIMBS>;

    pub struct BinPolyZipTypes {}
    impl ZipTypes for BinPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<K>;
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

    pub struct ArbitraryPolyZipTypesIprs {}
    impl ZipTypes for ArbitraryPolyZipTypesIprs {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<K>;
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
    pub struct ArbitraryPolyZipTypesRaa {}
    impl ZipTypes for ArbitraryPolyZipTypesRaa {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<K>, DEGREE_PLUS_ONE>;
        type Fmod = Uint<K>;
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

    pub struct IntZipTypes {}
    impl ZipTypes for IntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = ZtInt;
        type Cw = i128;
        type Fmod = Uint<K>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    struct TestZincTypesIprs;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesIprs {
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Fmod = Uint<K>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF2_16_1<IPRS_DEPTH>, CHECKED>;
    }

    const RAA_REP: usize = 4;

    #[derive(Copy, Clone)]
    struct TestRaaConfig;
    impl RaaConfig for TestRaaConfig {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = true;
    }

    struct TestZincTypesRaa;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesRaa {
        type Int = i64;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Fmod = Uint<K>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesRaa;
        type IntZt = IntZipTypes;

        type BinaryLc = RaaCode<Self::BinaryZt, TestRaaConfig, RAA_REP>;
        type ArbitraryLc = RaaCode<Self::ArbitraryZt, TestRaaConfig, RAA_REP>;
        type IntLc = RaaCode<Self::IntZt, TestRaaConfig, RAA_REP>;
    }

    /// Set up Zip+ PCS parameters for a given number of MLE variables.
    #[allow(clippy::type_complexity)]
    fn setup_pp<Wzt>(
        num_vars: usize,
    ) -> (
        ZipPlusParams<Wzt::BinaryZt, Wzt::BinaryLc>,
        ZipPlusParams<Wzt::ArbitraryZt, Wzt::ArbitraryLc>,
        ZipPlusParams<Wzt::IntZt, Wzt::IntLc>,
    )
    where
        Wzt: ZincTypes<DEGREE_PLUS_ONE>,
    {
        let poly_size = 1 << num_vars;
        (
            ZipPlus::<Wzt::BinaryZt, Wzt::BinaryLc>::setup(
                poly_size,
                Wzt::BinaryLc::new(poly_size),
            ),
            ZipPlus::<Wzt::ArbitraryZt, Wzt::ArbitraryLc>::setup(
                poly_size,
                Wzt::ArbitraryLc::new(poly_size),
            ),
            ZipPlus::<Wzt::IntZt, Wzt::IntLc>::setup(poly_size, Wzt::IntLc::new(poly_size)),
        )
    }

    fn do_test_multi_witness<U>()
    where
        U: Uair<Ideal = DegreeOneIdeal<ZtInt>, Scalar = DensePolynomial<ZtInt, DEGREE_PLUS_ONE>>
            + GenerateMultiTypeWitness<PolyCoeff = ZtInt, Int = ZtInt>
            + 'static,
        F: for<'a> FromWithConfig<&'a ZtInt>,
    {
        let mut rng = rng();
        let num_vars = 8;
        let pp = setup_pp::<TestZincTypesIprs>(num_vars);

        let (bin_trace, arb_trace, int_trace) = U::generate_witness(num_vars, &mut rng);

        let sig = U::signature();
        let public_bin = &bin_trace[..sig.public_binary_poly_cols];
        let public_arb = &arb_trace[..sig.public_arbitrary_poly_cols];
        let public_int = &int_trace[..sig.public_int_cols];

        // Prover: pass full trace (public first, then witness)
        let proof = ZincPlusPiop::<TestZincTypesIprs, U, F, DEGREE_PLUS_ONE>::prove::<CHECKED>(
            &pp,
            &bin_trace,
            &arb_trace,
            &int_trace,
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // Verifier: pass only public columns separately
        ZincPlusPiop::<TestZincTypesIprs, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            proof,
            public_bin,
            public_arb,
            public_int,
            num_vars,
            project_scalar_fn,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect("Verifier failed");
    }

    /// End-to-end test: TestAirNoMultiplication.
    ///
    /// UAIR constraint: a + b - c \in (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = <X - 2>).
    #[test]
    fn test_e2e_no_multiplication() {
        let mut rng = rng();
        let num_vars = 8;
        let pp = setup_pp::<TestZincTypesIprs>(num_vars);

        type TestUair = TestAirNoMultiplication<i64>;

        type Piop = ZincPlusPiop<TestZincTypesIprs, TestUair, F, DEGREE_PLUS_ONE>;

        // Generate a valid witness satisfying the UAIR constraints.
        let trace = TestUair::generate_witness(num_vars, &mut rng);

        // Prover
        let proof = Piop::prove::<CHECKED>(&pp, &[], &trace, &[], num_vars, project_scalar_fn)
            .expect("Prover failed");

        // Verifier
        Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            &[],
            &[],
            &[],
            num_vars,
            project_scalar_fn,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect("Verifier failed");
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
        let mut rng = rng();
        let num_vars = 2;
        let pp = setup_pp::<TestZincTypesRaa>(num_vars);

        type TestUair = TestUairSimpleMultiplication<i64>;

        type Piop = ZincPlusPiop<TestZincTypesRaa, TestUair, F, DEGREE_PLUS_ONE>;

        // Generate a valid witness satisfying the UAIR constraints.
        let trace = TestUair::generate_witness(num_vars, &mut rng);

        // Prover
        let proof = Piop::prove::<CHECKED>(&pp, &[], &trace, &[], num_vars, project_scalar_fn)
            .expect("Prover failed");

        // Verifier
        Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            &[],
            &[],
            &[],
            num_vars,
            project_scalar_fn,
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
        )
        .expect("Verifier failed");
    }

    /// End-to-end test: BinaryDecompositionUair.
    ///
    /// Uses binary_poly (1 col) and int (1 col) trace types.
    /// UAIR constraint: binary_poly[0] - int[0] \in <X - 2>
    #[test]
    fn test_e2e_binary_decomposition() {
        let mut rng = rng();
        let num_vars = 8;
        let pp = setup_pp::<TestZincTypesIprs>(num_vars);

        type TestUair = BinaryDecompositionUair<i64>;

        type Piop = ZincPlusPiop<TestZincTypesIprs, TestUair, F, DEGREE_PLUS_ONE>;

        // Generate a valid witness satisfying the UAIR constraints.
        let (binary_trace, arb_trace, int_trace) = TestUair::generate_witness(num_vars, &mut rng);

        // Prover
        let proof = Piop::prove::<CHECKED>(
            &pp,
            &binary_trace,
            &arb_trace,
            &int_trace,
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // Verifier
        Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            &[],
            &[],
            &[],
            num_vars,
            project_scalar_fn,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect("Verifier failed");
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
        do_test_multi_witness::<BigLinearUair<ZtInt>>();
    }

    /// End-to-end test: BigLinearUairWithPublicInput.
    ///
    /// Same as [`BigLinearUair`], but with the first few binary_poly columns as
    /// public inputs.
    #[test]
    fn test_e2e_big_linear_with_public_input() {
        do_test_multi_witness::<BigLinearUairWithPublicInput<ZtInt>>();
    }

    //
    // Negative tests for BigLinearUair: verify that proof tampering is detected.
    //

    fn big_linear_verify_tampered(tamper: impl FnOnce(&mut Proof<F>)) {
        let mut rng = rng();
        let num_vars = 8;
        let pp = setup_pp::<TestZincTypesIprs>(num_vars);

        type TestUair = BigLinearUair<i64>;

        type Piop = ZincPlusPiop<TestZincTypesIprs, TestUair, F, DEGREE_PLUS_ONE>;

        let (bin, arb, int) = BigLinearUair::<i64>::generate_witness(num_vars, &mut rng);

        let mut proof = Piop::prove::<CHECKED>(&pp, &bin, &arb, &int, num_vars, project_scalar_fn)
            .expect("Prover failed");

        tamper(&mut proof);

        Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            &[],
            &[],
            &[],
            num_vars,
            project_scalar_fn,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect_err("Verifier should have rejected tampered proof");
    }

    #[test]
    fn test_big_linear_tamper_lifted_evals() {
        big_linear_verify_tampered(|proof| {
            proof.witness_lifted_evals.swap(0, 1);
        });
    }

    #[test]
    fn test_big_linear_tamper_up_evals() {
        big_linear_verify_tampered(|proof| {
            proof.resolver.up_evals.swap(0, 1);
        });
    }

    #[test]
    fn test_big_linear_tamper_down_evals() {
        big_linear_verify_tampered(|proof| {
            proof.resolver.down_evals.swap(0, 1);
        });
    }

    #[test]
    fn test_big_linear_tamper_commitment() {
        big_linear_verify_tampered(|proof| {
            proof.commitments.0.root = Default::default();
        });
    }

    #[test]
    fn test_big_linear_tamper_ideal_check() {
        big_linear_verify_tampered(|proof| {
            proof.ideal_check.combined_mle_values.swap(0, 1);
        });
    }
}
