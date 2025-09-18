#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use ark_std::{
    hint::black_box,
    time::{Duration, Instant},
};
use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use crypto_bigint::{Random, U256, const_monty_params, modular::ConstMontyParams};
use crypto_primitives::crypto_bigint_int::Int;
use itertools::Itertools;
use rand::rng;
use zip_plus::{
    code::{DefaultLinearCodeSpec, LinearCode, raa::RaaCode},
    field::F256,
    pcs::{MerkleTree, structs::MultilinearZip},
    pcs_transcript::PcsTranscript,
    poly::{dense::DenseMultilinearExtension, mle::MultilinearExtensionRand},
    transcript::KeccakTranscript,
    utils::WORD_FACTOR,
};

const INT_LIMBS: usize = WORD_FACTOR;

const FIELD_LIMBS: usize = 4 * WORD_FACTOR;

const N: usize = INT_LIMBS;
const L: usize = INT_LIMBS * 2;
const K: usize = INT_LIMBS * 4;
const M: usize = INT_LIMBS * 8;

type LC = RaaCode<N, L, K, M>;
type BenchZip = MultilinearZip<N, L, K, M, LC>;

const_monty_params!(
    ModP,
    U256,
    "EB0E9F20F7BFC231327A11792F585AC6C20C74ACCCAB538BE6B0C3AB2E3D176F"
);
type F = F256<ModP>;

fn encode_rows<const P: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    group.bench_function(
        format!("EncodeRows: Int<{FIELD_LIMBS}>, poly_size = 2^{P}(Int limbs = {INT_LIMBS}), ZipSpec{spec}"),
        |b| {
            let mut rng = rng();
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
            let mut rng = rng();
            let mut keccak_transcript = KeccakTranscript::new();
            let poly_size = ROW_LEN * ROW_LEN;
            let linear_code =
                LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
            assert_eq!(linear_code.row_len(), ROW_LEN, "Unexpected row_len");
            let message: Vec<_> = (0..ROW_LEN).map(|_i| Int::<N>::random(&mut rng)).collect();
            b.iter(|| {
                let encoded_row: Vec<Int<K>> = linear_code.encode_wide(&message);
                black_box(encoded_row);
            })
        },
    );
}

fn merkle_root<const P: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    let mut rng = rng();

    let num_leaves = 1 << P;
    let leaves = (0..num_leaves)
        .map(|_| Int::<K>::random(&mut rng))
        .collect_vec();

    group.bench_function(
        format!("MerkleRoot: Int<{INT_LIMBS}>, leaves=2^{P}, spec={spec}"),
        |b| {
            b.iter(|| {
                let tree = MerkleTree::new(&leaves, num_leaves);
                black_box(tree.root());
            })
        },
    );
}

fn commit<const P: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    let mut rng = rng();
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
                    let res = BenchZip::commit(&params, &poly).expect("Failed to commit");
                    black_box(res);
                    total_duration += timer.elapsed();
                }

                total_duration
            })
        },
    );
}

fn open<const P: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    let mut rng = rng();

    type T = KeccakTranscript;
    let mut keccak_transcript = T::new();
    let poly_size = 1 << P;
    let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let params = BenchZip::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = BenchZip::commit(&params, &poly).unwrap();
    let point = vec![1i64; P];
    let field_point: Vec<F> = point.iter().map(F::from).collect();

    group.bench_function(
        format!("Open: RandomField<{FIELD_LIMBS}>, poly_size = 2^{P}(Int limbs = {INT_LIMBS}), ZipSpec{spec}, modulus={}", ModP::PARAMS.modulus()),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = PcsTranscript::new();
                    let timer = Instant::now();
                    BenchZip::open(
                        &params,
                        &poly,
                        &data,
                        &field_point,
                        &mut transcript,
                    )
                    .expect("Failed to make opening");
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}
fn verify<const P: usize>(group: &mut BenchmarkGroup<WallTime>, spec: usize) {
    let mut rng = rng();
    type T = KeccakTranscript;
    let mut keccak_transcript = T::new();
    let poly_size = 1 << P;
    let linear_code = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let params = BenchZip::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = BenchZip::commit(&params, &poly).unwrap();
    let point = vec![1i64; P];
    let field_point: Vec<F> = point.iter().map(F::from).collect();
    let eval = poly.evaluations.last().unwrap();
    let eval_field = eval.into();
    let mut transcript = PcsTranscript::new();

    BenchZip::open(&params, &poly, &data, &field_point, &mut transcript).unwrap();

    let proof = transcript.into_proof();

    group.bench_function(
        format!("Verify: RandomField<{FIELD_LIMBS}>, poly_size = 2^{P}(Int limbs = {INT_LIMBS}), ZipSpec{spec}, modulus={}", ModP::PARAMS.modulus()),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = PcsTranscript::from_proof(&proof);
                    let timer = Instant::now();
                    BenchZip::verify(
                        &params,
                        &commitment,
                        &field_point,
                        &eval_field,
                        &mut transcript,
                    )
                    .expect("Failed to verify");
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}

fn zip_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip");

    encode_rows::<12>(&mut group, 1);
    encode_rows::<13>(&mut group, 1);
    encode_rows::<14>(&mut group, 1);
    encode_rows::<15>(&mut group, 1);
    encode_rows::<16>(&mut group, 1);

    encode_single_row::<128>(&mut group, 1);
    encode_single_row::<256>(&mut group, 1);
    encode_single_row::<512>(&mut group, 1);
    encode_single_row::<1024>(&mut group, 1);
    encode_single_row::<2048>(&mut group, 1);
    encode_single_row::<4096>(&mut group, 1);

    merkle_root::<12>(&mut group, 1);
    merkle_root::<13>(&mut group, 1);
    merkle_root::<14>(&mut group, 1);
    merkle_root::<15>(&mut group, 1);
    merkle_root::<16>(&mut group, 1);

    commit::<12>(&mut group, 1);
    commit::<13>(&mut group, 1);
    commit::<14>(&mut group, 1);
    commit::<15>(&mut group, 1);
    commit::<16>(&mut group, 1);

    open::<12>(&mut group, 1);
    open::<13>(&mut group, 1);
    open::<14>(&mut group, 1);
    open::<15>(&mut group, 1);
    open::<16>(&mut group, 1);

    verify::<12>(&mut group, 1);
    verify::<13>(&mut group, 1);
    verify::<14>(&mut group, 1);
    verify::<15>(&mut group, 1);
    verify::<16>(&mut group, 1);

    group.finish();
}

criterion_group!(benches, zip_benchmarks);
criterion_main!(benches);
