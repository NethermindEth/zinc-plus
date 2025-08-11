#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;
use ark_std::{
    str::FromStr,
    test_rng,
    time::{Duration, Instant},
};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use crypto_bigint::Random;
use itertools::Itertools;
use zip_plus::{
    define_random_field_zip_types,
    field::{BigInt, ConfigRef, FieldConfig, RandomField},
    implement_random_field_zip_types,
    poly_z::mle::{DenseMultilinearExtension, MultilinearExtension},
    traits::{Config, ConfigReference, FieldMap, ZipTypes},
    transcript::KeccakTranscript,
    code::{DefaultLinearCodeSpec, LinearCode},
    code_raa::RaaCode,
    pcs::{structs::MultilinearZip, MerkleTree},
    pcs_transcript::PcsTranscript,
};

const INT_LIMBS: usize = 1;
const FIELD_LIMBS: usize = 4;

define_random_field_zip_types!();
implement_random_field_zip_types!(INT_LIMBS);

type ZT = RandomFieldZipTypes<INT_LIMBS>;
type LC = RaaCode<ZT>;
type BenchZip = MultilinearZip<ZT, LC>;

fn encode_rows<const P: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    group.bench_function(
        format!("EncodeRows: Int<{FIELD_LIMBS}>, poly_size = 2^{P}(Int limbs = {INT_LIMBS}), ZipSpec{spec}"),
        |b| {
            let mut rng = test_rng();
            type T = KeccakTranscript;
            let mut keccak_transcript = T::new();
            let poly_size = 1 << P;
            let linear_code =
                LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
            let params = BenchZip::setup(poly_size, linear_code);
            let row_len = params.linear_code.row_len();
            let codeword_len = params.linear_code.codeword_len();
            let poly = DenseMultilinearExtension::rand(P, &mut rng);
            b.iter(|| {
                let _ = BenchZip::encode_rows(&params, codeword_len, row_len, &poly.evaluations);
            })
        },
    );
}

fn encode_single_row<const ROW_LEN: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    group.bench_function(
        format!("EncodeMessage: Int<{FIELD_LIMBS}>, row_len = {ROW_LEN}(Int limbs = {INT_LIMBS}), ZipSpec{spec}"),
        |b| {
            let mut rng = test_rng();
            let mut keccak_transcript = KeccakTranscript::new();
            let poly_size = ROW_LEN * ROW_LEN;
            let linear_code =
                LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
            assert_eq!(linear_code.row_len(), ROW_LEN, "Unexpected row_len");
            let message: Vec<_> = (0..ROW_LEN).map(|_i| <ZT as ZipTypes>::N::random(&mut rng)).collect();
            b.iter(|| {
                let encoded_row: Vec<<ZT as ZipTypes>::K> = linear_code.encode_wide(&message);
                black_box(encoded_row);
            })
        },
    );
}

fn merkle_root<const P: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    use ark_std::test_rng;
    let mut rng = test_rng();

    let num_leaves = 1 << P;
    let leaves = (0..num_leaves)
        .map(|_| <ZT as ZipTypes>::K::random(&mut rng))
        .collect_vec();

    group.bench_function(
        format!("MerkleRoot: Int<{INT_LIMBS}>, leaves=2^{P}, spec={spec}"),
        |b| {
            b.iter(|| {
                let tree = MerkleTree::new(P, &leaves);
                black_box(tree.root);
            })
        },
    );
}

fn commit<const P: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    let mut rng = test_rng();
    type T = KeccakTranscript;
    let mut keccak_transcript = T::new();
    let poly_size = 1 << P;
    let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let params = BenchZip::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "Commit: Int<{FIELD_LIMBS}>, poly_size = 2^{P}(Int limbs = {INT_LIMBS}), ZipSpec{spec}"
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let poly = DenseMultilinearExtension::rand(P, &mut rng);
                    let timer = Instant::now();
                    let _ = BenchZip::commit::<RandomField<FIELD_LIMBS>>(&params, &poly)
                        .expect("Failed to commit");
                    total_duration += timer.elapsed()
                }

                total_duration / iters as u32
            })
        },
    );
}

fn open<const P: usize>(group: &mut BenchmarkGroup<WallTime>, modulus: &str, spec: usize) {
    let mut rng = test_rng();
    let config = FieldConfig::new(BigInt::<FIELD_LIMBS>::from_str(modulus).unwrap());
    let field_config = ConfigRef::from(&config);

    type T = KeccakTranscript;
    let mut keccak_transcript = T::new();
    let poly_size = 1 << P;
    let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let params = BenchZip::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = BenchZip::commit::<RandomField<FIELD_LIMBS>>(&params, &poly).unwrap();
    let point = vec![1i64; P];

    group.bench_function(
        format!("Open: RandomField<{FIELD_LIMBS}>, poly_size = 2^{P}(Int limbs = {INT_LIMBS}), ZipSpec{spec}, modulus={modulus}"),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = PcsTranscript::<RandomField<FIELD_LIMBS>>::new();
                    let timer = Instant::now();
                    BenchZip::open(
                        &params,
                        &poly,
                        &data,
                        &point.map_to_field(field_config),
                        field_config,
                        &mut transcript,
                    )
                    .expect("Failed to make opening");
                    total_duration += timer.elapsed();
                }
                total_duration / iters as u32
            })
        },
    );
}
fn verify<const P: usize>(group: &mut BenchmarkGroup<WallTime>, modulus: &str, spec: usize) {
    let mut rng = test_rng();
    let config = FieldConfig::new(BigInt::<FIELD_LIMBS>::from_str(modulus).unwrap());
    let field_config = ConfigRef::from(&config);

    type T = KeccakTranscript;
    let mut keccak_transcript = T::new();
    let poly_size = 1 << P;
    let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let params = BenchZip::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = BenchZip::commit::<RandomField<FIELD_LIMBS>>(&params, &poly).unwrap();
    let point = vec![1i64; P];
    let eval = poly.evaluations.last().unwrap();
    let mut transcript = PcsTranscript::<RandomField<FIELD_LIMBS>>::new();

    BenchZip::open(
        &params,
        &poly,
        &data,
        &point.map_to_field(field_config),
        field_config,
        &mut transcript,
    )
    .unwrap();

    let proof = transcript.into_proof();
    field_config
        .reference()
        .expect("Field config cannot be none");
    group.bench_function(
        format!("Verify: RandomField<{FIELD_LIMBS}>, poly_size = 2^{P}(Int limbs = {INT_LIMBS}), ZipSpec{spec}, modulus={modulus}"),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = PcsTranscript::<RandomField<FIELD_LIMBS>>::from_proof(&proof);
                    let timer = Instant::now();
                    BenchZip::verify(
                        &params,
                        &commitment,
                        &point.map_to_field(field_config),
                        eval.map_to_field(field_config),
                        &mut transcript,
                        field_config,
                    )
                    .expect("Failed to verify");
                    total_duration += timer.elapsed();
                }
                total_duration / iters as u32
            })
        },
    );
}

fn zip_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip");

    encode_rows::<12>(&mut group, 1);
    encode_rows::<16>(&mut group, 1);

    encode_single_row::<128>(&mut group, 1);
    encode_single_row::<256>(&mut group, 1);
    encode_single_row::<512>(&mut group, 1);
    encode_single_row::<1024>(&mut group, 1);
    encode_single_row::<2048>(&mut group, 1);
    encode_single_row::<4096>(&mut group, 1);

    merkle_root::<12>(&mut group, 1);
    merkle_root::<16>(&mut group, 1);
    merkle_root::<22>(&mut group, 1);

    commit::<12>(&mut group, 1);
    commit::<16>(&mut group, 1);

    open::<12>(
        &mut group,
        "106319353542452952636349991594949358997917625194731877894581586278529202198383",
        1,
    );
    open::<16>(
        &mut group,
        "106319353542452952636349991594949358997917625194731877894581586278529202198383",
        1,
    );

    verify::<12>(
        &mut group,
        "106319353542452952636349991594949358997917625194731877894581586278529202198383",
        1,
    );
    verify::<16>(
        &mut group,
        "106319353542452952636349991594949358997917625194731877894581586278529202198383",
        1,
    );

    group.finish();
}

criterion_group!(benches, zip_benchmarks);
criterion_main!(benches);
