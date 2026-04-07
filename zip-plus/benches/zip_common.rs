//! This module contains common benchmarking code for the Zip+ PCS,
//! both for Zip (integer coefficients) and Zip+ (polynomial coefficients).

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

use criterion::{BenchmarkGroup, measurement::WallTime};
use crypto_bigint::U64;
use crypto_primitives::{
    DenseRowMatrix, Field, FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig,
    crypto_bigint_monty::MontyField,
};
use itertools::Itertools;
use num_traits::One;
use rand::{distr::StandardUniform, prelude::*};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_utils::{
    from_ref::FromRef, mul_by_scalar::MulByScalar, named::Named,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    code::LinearCode,
    merkle::MerkleTree,
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipTypes},
    pcs_transcript::PcsProverTranscript,
    utils::eprint_proof_size,
};

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;

pub fn do_bench<Zt: ZipTypes, Lc: LinearCode<Zt>, const CHECK_FOR_OVERFLOWS: bool>(
    group: &mut BenchmarkGroup<WallTime>,
    make_linear_code: impl Fn(/* poly_size: */ usize) -> Lc + Copy,
) where
    StandardUniform: Distribution<Zt::Eval> + Distribution<Zt::Cw>,
    F: FromPrimitiveWithConfig
        + FromRef<F>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>,
    <F as Field>::Inner: FromRef<Zt::Fmod> + Transcribable,
    Zt::Eval: ProjectableToField<F>,
{
    encode_rows::<Zt, Lc, 12>(group, make_linear_code);
    encode_rows::<Zt, Lc, 13>(group, make_linear_code);
    encode_rows::<Zt, Lc, 14>(group, make_linear_code);
    encode_rows::<Zt, Lc, 15>(group, make_linear_code);
    encode_rows::<Zt, Lc, 16>(group, make_linear_code);

    for lc in [128, 256, 512, 1024]
        .map(make_linear_code)
        .into_iter()
        // These might be duplicate depending on linear code construction logic
        .dedup()
    {
        encode_single_row::<Zt, Lc>(group, lc)
    }

    merkle_root::<Zt, 12>(group);
    merkle_root::<Zt, 13>(group);
    merkle_root::<Zt, 14>(group);
    merkle_root::<Zt, 15>(group);
    merkle_root::<Zt, 16>(group);

    commit::<Zt, Lc, 12, 1>(group, make_linear_code);
    commit::<Zt, Lc, 13, 1>(group, make_linear_code);
    commit::<Zt, Lc, 14, 1>(group, make_linear_code);
    commit::<Zt, Lc, 15, 1>(group, make_linear_code);
    commit::<Zt, Lc, 16, 1>(group, make_linear_code);

    commit::<Zt, Lc, 14, 2>(group, make_linear_code);
    commit::<Zt, Lc, 14, 5>(group, make_linear_code);
    commit::<Zt, Lc, 16, 2>(group, make_linear_code);
    commit::<Zt, Lc, 16, 5>(group, make_linear_code);

    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 12, 1>(group, make_linear_code);
    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 13, 1>(group, make_linear_code);
    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 14, 1>(group, make_linear_code);
    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 15, 1>(group, make_linear_code);
    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 16, 1>(group, make_linear_code);

    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 14, 2>(group, make_linear_code);
    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 14, 5>(group, make_linear_code);
    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 16, 2>(group, make_linear_code);
    prove::<Zt, Lc, CHECK_FOR_OVERFLOWS, 16, 5>(group, make_linear_code);

    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 12, 1>(group, make_linear_code);
    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 13, 1>(group, make_linear_code);
    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 14, 1>(group, make_linear_code);
    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 15, 1>(group, make_linear_code);
    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 16, 1>(group, make_linear_code);

    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 14, 2>(group, make_linear_code);
    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 14, 5>(group, make_linear_code);
    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 16, 2>(group, make_linear_code);
    verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, 16, 5>(group, make_linear_code);
}

pub fn encode_rows<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    make_linear_code: impl Fn(usize) -> Lc,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    group.bench_function(
        format!(
            "EncodeRows: {} -> {}, poly_size = 2^{P}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name()
        ),
        |b| {
            let mut rng = ThreadRng::default();
            let poly_size: usize = 1 << P;
            let linear_code = make_linear_code(poly_size);
            let params = ZipPlus::setup(poly_size, linear_code);
            let poly = DenseMultilinearExtension::<<Zt as ZipTypes>::Eval>::rand(P, &mut rng);
            b.iter(|| {
                let cw = ZipPlus::encode_rows(&params, &poly);
                black_box(cw)
            })
        },
    );
}

pub fn encode_single_row<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
    linear_code: Lc,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let row_len = linear_code.row_len();
    group.bench_function(
        format!(
            "EncodeMessage: {} -> {}, row_len = {row_len}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name()
        ),
        |b| {
            let message: Vec<<Zt as ZipTypes>::Eval> =
                (0..row_len).map(|_i| rng.random()).collect();
            b.iter(|| {
                let encoded_row: Vec<<Zt as ZipTypes>::Cw> = linear_code.encode(&message);
                black_box(encoded_row);
            })
        },
    );
}

pub fn merkle_root<Zt: ZipTypes, const P: usize>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Cw>,
{
    let mut rng = ThreadRng::default();

    let num_leaves = 1 << P;
    let leaves = (0..num_leaves)
        .map(|_| rng.random::<<Zt as ZipTypes>::Cw>())
        .collect_vec();
    let matrix: DenseRowMatrix<_> = vec![leaves.clone()].into();
    let rows = matrix.to_rows_slices();

    group.bench_function(
        format!("MerkleRoot: {}, leaves=2^{P}", Zt::Cw::type_name()),
        |b| {
            b.iter(|| {
                let tree = MerkleTree::new(&rows);
                black_box(tree.root());
            })
        },
    );
}

pub fn commit<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize, const BATCH: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    make_linear_code: impl Fn(usize) -> Lc,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size: usize = 1 << P;
    let linear_code = make_linear_code(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "Commit(batch={BATCH}): Eval={}, Cw={}, Comb={}, poly_size=2^{P}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name()
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let polys: Vec<_> = (0..BATCH)
                        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
                        .collect();
                    let timer = Instant::now();
                    let res = ZipPlus::commit(&params, &polys).expect("Failed to commit");
                    black_box(res);
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}

pub fn prove<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
    const BATCH: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
    make_linear_code: impl Fn(usize) -> Lc,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>,
    <F as Field>::Inner: FromRef<Zt::Fmod> + Transcribable,
    Zt::Eval: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();

    let poly_size: usize = 1 << P;
    let linear_code = make_linear_code(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    let polys: Vec<_> = (0..BATCH)
        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
        .collect();
    let (hint, commitment) = ZipPlus::commit(&params, &polys).unwrap();

    let mut transcript = PcsProverTranscript::new_from_commitment(&commitment);
    let field_cfg = transcript
        .fs_transcript
        .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

    let point = vec![Zt::Pt::one(); P];

    macro_rules! do_prove {
        ($t:expr) => {
            ZipPlus::prove::<F, CHECK_FOR_OVERFLOWS>($t, &params, &polys, &point, &hint, &field_cfg)
                .expect("Prove failed")
        };
    }

    let combined_proof = {
        let mut t = transcript.clone();
        do_prove!(&mut t);
        CombinedProof {
            comm: commitment.clone(),
            proof_transcript: t.stream.into_inner(),
        }
    };

    group.bench_function(
        format!(
            "Prove(batch={BATCH}): Eval={}, Cw={}, Comb={}, poly_size=2^{P}, modulus=({} bits)",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            Zt::Fmod::NUM_BYTES * 8
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = transcript.clone();
                    let timer = Instant::now();
                    let result = do_prove!(&mut transcript);
                    total_duration += timer.elapsed();
                    black_box(result);
                }
                total_duration
            })
        },
    );

    eprint_proof_size(
        format_args!("batch={BATCH}, poly_size=2^{P}"),
        &combined_proof,
    );
}

pub fn verify<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
    const BATCH: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
    make_linear_code: impl Fn(usize) -> Lc,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: FromPrimitiveWithConfig
        + FromRef<F>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>,
    <F as Field>::Inner: FromRef<Zt::Fmod> + Transcribable,
    Zt::Eval: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();
    let poly_size: usize = 1 << P;
    let linear_code = make_linear_code(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    let polys: Vec<_> = (0..BATCH)
        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
        .collect();
    let (hint, commitment) = ZipPlus::commit(&params, &polys).unwrap();

    let mut transcript = PcsProverTranscript::new_from_commitment(&commitment);
    let field_cfg = transcript
        .fs_transcript
        .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
    let point = vec![Zt::Pt::one(); P];

    let eval_f = ZipPlus::prove::<F, CHECK_FOR_OVERFLOWS>(
        &mut transcript,
        &params,
        &polys,
        &point,
        &hint,
        &field_cfg,
    )
    .expect("Prove failed");

    let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

    let transcript = transcript.into_verification_transcript();

    let combined_proof = CombinedProof {
        comm: commitment.clone(),
        proof_transcript: transcript.stream.get_ref().clone(),
    };

    group.bench_function(
        format!(
            "Verify(batch={BATCH}): Eval={}, Cw={}, Comb={}, poly_size=2^{P}, modulus=({} bits)",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            Zt::Fmod::NUM_BYTES * 8
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = transcript.clone();

                    let timer = Instant::now();

                    // Initializing the verification transcript with the commitment and getting
                    // field config and projecting element are (arguably) verifier's responsibility
                    // and should be included in the verification time.
                    transcript.fs_transcript.absorb_slice(&commitment.root);
                    let field_cfg = transcript
                        .fs_transcript
                        .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

                    ZipPlus::verify::<F, CHECK_FOR_OVERFLOWS>(
                        &mut transcript,
                        &params,
                        &commitment,
                        &field_cfg,
                        &point_f,
                        &eval_f,
                    )
                    .expect("Verification failed");
                    total_duration += timer.elapsed();
                    black_box(transcript);
                }
                total_duration
            })
        },
    );

    eprint_proof_size(
        format_args!("batch={BATCH}, poly_size=2^{P}"),
        &combined_proof,
    );
}

//
// Helpers
//

/// Used to calculate total proof size
struct CombinedProof {
    comm: ZipPlusCommitment,
    proof_transcript: Vec<u8>,
}

impl GenTranscribable for CombinedProof {
    fn read_transcription_bytes_exact(_bytes: &[u8]) -> Self {
        unimplemented!("We don't need to read this proof")
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let buf = self.comm.write_transcription_bytes_subset(buf);
        buf.copy_from_slice(&self.proof_transcript);
    }
}

impl Transcribable for CombinedProof {
    fn get_num_bytes(&self) -> usize {
        self.comm.get_num_bytes() + self.proof_transcript.len()
    }
}
