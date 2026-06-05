use ark_bn254::G1Affine;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use zip_plus::pcs::msm_commitment::{
    BoolSubsetMsm, MsmCommitmentEngine, MsmCommitmentKey, ScalarPippengerMsm, U8BucketMsm,
};

fn scalar<C: AffineRepr>(value: usize) -> C::ScalarField {
    C::ScalarField::from(u64::try_from(value).expect("benchmark value must fit into u64"))
}

fn setup<C: AffineRepr>(width: usize, n: usize) -> MsmCommitmentKey<C> {
    let generator = C::Group::generator();
    let bases = (1..=width)
        .map(|idx| (generator * scalar::<C>(idx)).into_affine())
        .collect();
    let h = generator * scalar::<C>(width + 1);
    let (ck, _) = MsmCommitmentEngine::<C>::setup_from_bases(width, bases, h)
        .expect("benchmark setup must be valid");
    let _blind = MsmCommitmentEngine::<C>::blind(&ck, n);
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

fn scalar_values<C: AffineRepr>(values: &[u8]) -> Vec<C::ScalarField> {
    values
        .iter()
        .map(|value| C::ScalarField::from(u64::from(*value)))
        .collect()
}

fn bench_curve<C: AffineRepr>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    curve_name: &str,
    width: usize,
    n: usize,
) {
    let ck = setup::<C>(width, n);
    let blind = MsmCommitmentEngine::<C>::blind(&ck, n);
    let bools = bool_values(n);
    let u8_small = u8_values(n, 32);
    let u8_full = u8_values(n, 255);
    let scalars = scalar_values::<C>(&u8_full);

    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/bool_subset"), width),
        &width,
        |b, _| {
            b.iter(|| {
                MsmCommitmentEngine::<C>::commit_with::<bool, BoolSubsetMsm<6>>(
                    black_box(&ck),
                    black_box(&bools),
                    black_box(&blind),
                )
                .expect("bool benchmark commit must succeed")
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/u8_0_32"), width),
        &width,
        |b, _| {
            b.iter(|| {
                MsmCommitmentEngine::<C>::commit_with::<u8, U8BucketMsm>(
                    black_box(&ck),
                    black_box(&u8_small),
                    black_box(&blind),
                )
                .expect("u8 small benchmark commit must succeed")
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/u8_0_255"), width),
        &width,
        |b, _| {
            b.iter(|| {
                MsmCommitmentEngine::<C>::commit_with::<u8, U8BucketMsm>(
                    black_box(&ck),
                    black_box(&u8_full),
                    black_box(&blind),
                )
                .expect("u8 full benchmark commit must succeed")
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/scalar_pippenger"), width),
        &width,
        |b, _| {
            b.iter(|| {
                MsmCommitmentEngine::<C>::commit_with::<C::ScalarField, ScalarPippengerMsm>(
                    black_box(&ck),
                    black_box(&scalars),
                    black_box(&blind),
                )
                .expect("scalar benchmark commit must succeed")
            });
        },
    );
}

fn msm_commitment_benches(c: &mut Criterion) {
    let width = 64;
    let n = width * 1024;

    let mut group = c.benchmark_group("msm_commitment");
    bench_curve::<G1Affine>(&mut group, "bn254", width, n);
    bench_curve::<ark_secp256k1::Affine>(&mut group, "secp256k1", width, n);
    group.finish();
}

criterion_group!(benches, msm_commitment_benches);
criterion_main!(benches);
