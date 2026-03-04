//! Zinc+ PIOP for UCS — end-to-end protocol (without PCS).
//!
//! Implements the four steps of the Zinc+ compiler from
//! Section 2.2 "Combining the three steps" of the paper:
//!
//! ```text
//! Q[X]  --φ_q-->  F_q[X]  --MLE eval-->  F_q[X]  --ψ_a-->  F_q
//!       Step 1             Step 2                  Step 3
//! ```
//!
//! Step 4 runs a finite-field PIOP (sumcheck) over F_q.
//!
//! The verifier's output is a [`Subclaim`] containing evaluation
//! claims about the trace column MLEs. In the full protocol,
//! these would be resolved by the Zip+ PCS.

pub mod prover;
pub mod verifier;

use crypto_primitives::{ConstIntRing, ConstIntSemiring, PrimeField};
use std::marker::PhantomData;
use thiserror::Error;
use zinc_piop::{
    combined_poly_resolver, combined_poly_resolver::CombinedPolyResolverError, ideal_check,
    ideal_check::IdealCheckError,
};
use zinc_poly::{
    ConstCoeffBitWidth,
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dense::DensePolynomial},
};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::ConstTranscribable;
use zinc_uair::{Uair, ideal::Ideal};
use zinc_utils::named::Named;
use zip_plus::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlusCommitment, ZipTypes},
};

//
// Data structures
//

/// Proof produced by the Zinc+ PIOP for UCS (without PCS).
///
/// Contains the two subproofs from Steps 2 and 4:
/// - `ideal_check`: MLE evaluations in F_q\[X\] (Step 2).
/// - `resolver`: sumcheck proof + trace evaluation claims (Step 4).
#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub num_witness_cols: (usize, usize, usize),
    pub zip_commitments: Vec<ZipPlusCommitment>,
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip_proof: Vec<u8>,
    pub ideal_check: ideal_check::Proof<F>,
    pub resolver: combined_poly_resolver::Proof<F>,
}

/// Subclaim returned by the verifier, to be resolved by PCS.
///
/// Contains evaluation claims: "the trace column MLEs, evaluated at
/// `evaluation_point`, should yield `up_evals` (current row) and
/// `down_evals` (next row)."
#[derive(Clone, Debug)]
pub struct Subclaim<F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub up_evals: Vec<F>,
    pub down_evals: Vec<F>,
}

/// Prover auxiliary data for subclaim resolution without PCS.
pub struct ProverAux<F: PrimeField> {
    /// The random field configuration (derived from transcript in Step 1).
    pub field_cfg: F::Config,
    /// The trace projected to F_q (after Steps 1 and 3).
    pub projected_trace_f: Vec<DenseMultilinearExtension<F::Inner>>,
}

/// Trait bundling the various type parameters for the witness and Zinc+ PIOP.
pub trait WitnessZincTypes<const DEGREE_PLUS_ONE: usize> {
    type Int: Named + ConstCoeffBitWidth + Default + Clone + Send + Sync;
    type Chal: ConstIntRing + ConstTranscribable + Named;
    type Pt: ConstIntRing;
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;
    type PrimeTest: PrimalityTest<Self::Fmod>;

    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;
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

/// Main struct for the Zinc+ PIOP.
#[derive(Copy, Clone, Default, Debug)]
pub struct ZincPlusPiop<Wzt, U, F, const DEGREE_PLUS_ONE: usize>(PhantomData<(Wzt, U, F)>)
where
    Wzt: WitnessZincTypes<DEGREE_PLUS_ONE>,
    U: Uair,
    F: PrimeField;

//
// Error type and conversion
//

#[derive(Debug, Error)]
pub enum ProtocolError<F: PrimeField, I: Ideal> {
    #[error("ideal check failed: {0}")]
    IdealCheck(#[from] IdealCheckError<F, I>),
    #[error("combined poly resolver failed: {0}")]
    Resolver(#[from] CombinedPolyResolverError<F>),
    #[error("scalar projection failed: {0}")]
    ScalarProjection(zinc_poly::EvaluationError),
    #[error("subclaim resolution: MLE evaluation failed: {0}")]
    MleEvaluation(zinc_poly::EvaluationError),
    #[error("subclaim mismatch at column {column}: expected {expected:?}, got {actual:?}")]
    SubclaimMismatch {
        column: usize,
        expected: F,
        actual: F,
    },
    #[error("PCS error: {0}")]
    Pcs(#[from] ZipError),
    #[error("PCS verification failed at column {0}: {1}")]
    PcsVerification(usize, ZipError),
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::U64;
    use crypto_primitives::{
        FromWithConfig, crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
        crypto_bigint_uint::Uint,
    };
    use rand::rng;
    use zinc_poly::univariate::{
        binary::BinaryPolyInnerProduct, dense::DensePolyInnerProduct,
        dynamic::over_field::DynamicPolynomialF, ideal::DegreeOneIdeal,
    };
    use zinc_primality::MillerRabin;
    use zinc_test_uair::{
        BigLinearUair, BinaryDecompositionUair, GenerateMultiTypeWitness,
        GenerateSingleTypeWitness, TestAirNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_uair::ideal_collector::IdealOrZero;
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

    pub struct IntZipTypes {}
    impl ZipTypes for IntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = i64;
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

    struct TestWitnessZincTypesIprs;

    impl WitnessZincTypes<DEGREE_PLUS_ONE> for TestWitnessZincTypesIprs {
        type Int = i64;
        type Chal = i128;
        type Pt = i128;
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

    struct TestWitnessZincTypesRaa;

    impl WitnessZincTypes<DEGREE_PLUS_ONE> for TestWitnessZincTypesRaa {
        type Int = i64;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<K>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesRaa;
        type IntZt = IntZipTypes;

        type BinaryLc = RaaCode<Self::BinaryZt, TestRaaConfig, RAA_REP>;
        type ArbitraryLc = RaaCode<Self::ArbitraryZt, TestRaaConfig, RAA_REP>;
        type IntLc = RaaCode<Self::IntZt, TestRaaConfig, RAA_REP>;
    }

    /// Helper: project a DensePolynomial scalar to DynamicPolynomialF
    /// by projecting each coefficient via φ_q.
    fn project_scalar_fn<R>(
        scalar: &DensePolynomial<R, 32>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> DynamicPolynomialF<F>
    where
        F: for<'a> FromWithConfig<&'a R>,
    {
        scalar
            .iter()
            .map(|coeff| F::from_with_cfg(coeff, field_cfg))
            .collect()
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
        Wzt: WitnessZincTypes<DEGREE_PLUS_ONE>,
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

    /// End-to-end test: TestAirNoMultiplication.
    ///
    /// UAIR constraint: a + b - c ∈ (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = ⟨X - 2⟩).
    #[test]
    fn test_e2e_no_multiplication() {
        let mut rng = rng();
        let num_vars = 8;
        let pp = setup_pp::<TestWitnessZincTypesIprs>(num_vars);

        type TestUair = TestAirNoMultiplication<i64>;

        type Piop = ZincPlusPiop<TestWitnessZincTypesIprs, TestUair, F, DEGREE_PLUS_ONE>;

        // Generate a valid witness satisfying the UAIR constraints.
        let trace = TestUair::generate_witness(num_vars, &mut rng);

        // Prover
        let (proof, prover_aux) =
            Piop::prove::<CHECKED>(&pp, &[], &trace, &[], num_vars, project_scalar_fn)
                .expect("Prover failed");

        // Verifier
        let subclaim = Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            num_vars,
            project_scalar_fn,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect("Verifier failed");

        // Subclaim resolution (in lieu of PCS)
        Piop::resolve_subclaim(
            &subclaim,
            &prover_aux.projected_trace_f,
            &prover_aux.field_cfg,
        )
        .expect("Subclaim resolution failed");
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
    /// ≈ 127^8 ≈ 2^56, which fits in i64.
    #[test]
    fn test_e2e_simple_multiplication() {
        let mut rng = rng();
        let num_vars = 2;
        let pp = setup_pp::<TestWitnessZincTypesRaa>(num_vars);

        type TestUair = TestUairSimpleMultiplication<i64>;

        type Piop = ZincPlusPiop<TestWitnessZincTypesRaa, TestUair, F, DEGREE_PLUS_ONE>;

        let trace = TestUair::generate_witness(num_vars, &mut rng);

        // Prover
        let (proof, prover_aux) =
            Piop::prove::<CHECKED>(&pp, &[], &trace, &[], num_vars, project_scalar_fn)
                .expect("Prover failed");

        // Verifier
        let subclaim = Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            num_vars,
            project_scalar_fn,
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
        )
        .expect("Verifier failed");

        // Subclaim resolution (in lieu of PCS)
        Piop::resolve_subclaim(
            &subclaim,
            &prover_aux.projected_trace_f,
            &prover_aux.field_cfg,
        )
        .expect("Subclaim resolution failed");
    }

    /// End-to-end test: BinaryDecompositionUair.
    ///
    /// Uses binary_poly (1 col) and int (1 col) trace types.
    /// UAIR constraint: binary_poly[0] - int[0] ∈ ⟨X - 2⟩
    #[test]
    fn test_e2e_binary_decomposition() {
        let mut rng = rng();
        let num_vars = 8;
        let pp = setup_pp::<TestWitnessZincTypesIprs>(num_vars);

        type TestUair = BinaryDecompositionUair<i64>;

        type Piop = ZincPlusPiop<TestWitnessZincTypesIprs, TestUair, F, DEGREE_PLUS_ONE>;

        let (binary_trace, arb_trace, int_trace) = TestUair::generate_witness(num_vars, &mut rng);

        // Prover
        let (proof, prover_aux) = Piop::prove::<CHECKED>(
            &pp,
            &binary_trace,
            &arb_trace,
            &int_trace,
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // Verifier
        let subclaim = Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            num_vars,
            project_scalar_fn,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect("Verifier failed");

        // Subclaim resolution (in lieu of PCS)
        Piop::resolve_subclaim(
            &subclaim,
            &prover_aux.projected_trace_f,
            &prover_aux.field_cfg,
        )
        .expect("Subclaim resolution failed");
    }

    /// End-to-end test: BigLinearUair.
    ///
    /// Uses 16 binary_poly cols and 1 int col.
    /// UAIR constraints:
    ///   sum(up.binary_poly[0..16]) - up.int[0] ∈ ⟨X - 1⟩
    ///   down.binary_poly[0] - up.int[0] ∈ ⟨X - 2⟩
    ///   up.binary_poly[i] - down.binary_poly[i] = 0, for i=1..15
    #[test]
    fn test_e2e_big_linear() {
        let mut rng = rng();
        let num_vars = 8;
        let pp = setup_pp::<TestWitnessZincTypesIprs>(num_vars);

        type TestUair = BigLinearUair<i64>;

        type Piop = ZincPlusPiop<TestWitnessZincTypesIprs, TestUair, F, DEGREE_PLUS_ONE>;

        let (binary_trace, arb_trace, int_trace) = TestUair::generate_witness(num_vars, &mut rng);

        // Prover
        let (proof, prover_aux) = Piop::prove::<CHECKED>(
            &pp,
            &binary_trace,
            &arb_trace,
            &int_trace,
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // Verifier
        let subclaim = Piop::verify::<_, CHECKED>(
            &pp,
            proof,
            num_vars,
            project_scalar_fn,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect("Verifier failed");

        // Subclaim resolution (in lieu of PCS)
        Piop::resolve_subclaim(
            &subclaim,
            &prover_aux.projected_trace_f,
            &prover_aux.field_cfg,
        )
        .expect("Subclaim resolution failed");
    }
}
