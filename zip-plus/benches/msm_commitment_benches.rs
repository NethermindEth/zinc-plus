use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, PrimeGroup};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use zip_plus::pcs::msm_commitment::{
    BoolSubsetMsm, MsmCommitmentEngine, ScalarPippengerMsm, U8BucketMsm,
};

fn fr(value: usize) -> Fr {
    Fr::from(u64::try_from(value).expect("benchmark value must fit into u64"))
}

fn setup(width: usize, n: usize) -> zip_plus::pcs::msm_commitment::MsmCommitmentKey<G1Affine> {
    let generator = G1Projective::generator();
    let bases = (1..=width)
        .map(|idx| (generator * fr(idx)).into_affine())
        .collect();
    let h = generator * fr(width + 1);
    let (ck, _) = MsmCommitmentEngine::<G1Affine>::setup_from_bases(width, bases, h)
        .expect("benchmark setup must be valid");
    let _blind = MsmCommitmentEngine::<G1Affine>::blind(&ck, n);
    ck
}

fn bool_values(n: usize) -> Vec<bool> {
    (0..n).map(|idx| idx % 3 == 0 || idx % 11 == 1).collect()
}

fn u8_values(n: usize, modulus: u8) -> Vec<u8> {
    (0..n)
        .map(|idx| {
            let value = (idx * 17 + 5) % usize::from(modulus);
            u8::try_from(value).expect("benchmark u8 value must fit")
        })
        .collect()
}

fn scalar_values(values: &[u8]) -> Vec<Fr> {
    values
        .iter()
        .map(|value| Fr::from(u64::from(*value)))
        .collect()
}

fn msm_commitment_benches(c: &mut Criterion) {
    let width = 64;
    let n = width * 1024;
    let ck = setup(width, n);
    let blind = MsmCommitmentEngine::<G1Affine>::blind(&ck, n);
    let bools = bool_values(n);
    let u8_small = u8_values(n, 32);
    let u8_full = u8_values(n, 255);
    let scalars = scalar_values(&u8_full);

    let mut group = c.benchmark_group("msm_commitment");
    group.bench_with_input(BenchmarkId::new("bool_subset", width), &width, |b, _| {
        b.iter(|| {
            MsmCommitmentEngine::<G1Affine>::commit_with::<bool, BoolSubsetMsm<6>>(
                black_box(&ck),
                black_box(&bools),
                black_box(&blind),
            )
            .expect("bool benchmark commit must succeed")
        });
    });
    group.bench_with_input(BenchmarkId::new("u8_0_32", width), &width, |b, _| {
        b.iter(|| {
            MsmCommitmentEngine::<G1Affine>::commit_with::<u8, U8BucketMsm>(
                black_box(&ck),
                black_box(&u8_small),
                black_box(&blind),
            )
            .expect("u8 small benchmark commit must succeed")
        });
    });
    group.bench_with_input(BenchmarkId::new("u8_0_255", width), &width, |b, _| {
        b.iter(|| {
            MsmCommitmentEngine::<G1Affine>::commit_with::<u8, U8BucketMsm>(
                black_box(&ck),
                black_box(&u8_full),
                black_box(&blind),
            )
            .expect("u8 full benchmark commit must succeed")
        });
    });
    group.bench_with_input(
        BenchmarkId::new("scalar_pippenger", width),
        &width,
        |b, _| {
            b.iter(|| {
                MsmCommitmentEngine::<G1Affine>::commit_with::<Fr, ScalarPippengerMsm>(
                    black_box(&ck),
                    black_box(&scalars),
                    black_box(&blind),
                )
                .expect("scalar benchmark commit must succeed")
            });
        },
    );
    group.finish();
}

criterion_group!(benches, msm_commitment_benches);
criterion_main!(benches);
