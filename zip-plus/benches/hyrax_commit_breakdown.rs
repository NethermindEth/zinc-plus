use ark_bn254::G1Affine;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::Zero as ArkZero;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use crypto_primitives::crypto_bigint_monty::MontyField;
use std::hint::black_box;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::binary::BinaryPoly};
use zip_plus::pcs::{
    generic::PCS,
    hyrax::{BinaryLanes, HyraxCommitmentKey, HyraxPCS},
    msm_commitment::{BoolSubsetMsm, MsmCommitmentEngine, MsmCommitmentKey, RowMsmStrategy},
};

type F = MontyField<4>;

fn scalar<C: AffineRepr>(value: usize) -> C::ScalarField {
    C::ScalarField::from(u64::try_from(value).expect("benchmark value must fit into u64"))
}

fn bases_and_h<C: AffineRepr>(width: usize) -> (Vec<C>, C::Group) {
    let generator = C::Group::generator();
    let bases = (1..=width)
        .map(|idx| (generator * scalar::<C>(idx)).into_affine())
        .collect();
    let h = generator * scalar::<C>(width + 1);
    (bases, h)
}

fn msm_ck<C: AffineRepr>(width: usize) -> MsmCommitmentKey<C> {
    let (bases, h) = bases_and_h::<C>(width);
    MsmCommitmentEngine::<C>::setup_from_bases(width, bases, h)
        .expect("benchmark setup must be valid")
        .0
}

fn hyrax_ck<C: AffineRepr>(width: usize) -> HyraxCommitmentKey<C> {
    let (bases, h) = bases_and_h::<C>(width);
    HyraxPCS::<C, BinaryLanes>::setup_from_bases(width, bases, h)
        .expect("benchmark setup must be valid")
        .0
}

fn bool_row(width: usize) -> Vec<bool> {
    (0..width)
        .map(|idx| idx % 3 == 0 || idx % 11 == 1)
        .collect()
}

fn bool_values(num_lanes: usize, width: usize) -> Vec<Vec<bool>> {
    (0..num_lanes)
        .map(|lane| {
            (0..width)
                .map(|idx| (idx + lane) % 3 == 0 || (idx * 7 + lane) % 19 == 2)
                .collect()
        })
        .collect()
}

fn bit_mask(bits: &[bool]) -> usize {
    bits.iter().enumerate().fold(
        0usize,
        |mask, (idx, bit)| {
            if *bit { mask | (1usize << idx) } else { mask }
        },
    )
}

fn subset_tables<C: AffineRepr>(bases: &[C], window_bits: usize) -> Vec<Vec<C::Group>> {
    bases
        .chunks(window_bits)
        .map(|window| {
            let table_len = 1usize << window.len();
            let mut table = vec![C::Group::zero(); table_len];
            for mask in 1..table_len {
                let bit = mask.trailing_zeros() as usize;
                let previous = mask & !(1usize << bit);
                table[mask] = table[previous] + window[bit];
            }
            table
        })
        .collect()
}

fn precomputed_bool_row<C: AffineRepr>(
    tables: &[Vec<C::Group>],
    values: &[bool],
    window_bits: usize,
) -> C::Group {
    let mut acc = C::Group::zero();
    for (window_idx, bits) in values.chunks(window_bits).enumerate() {
        acc += tables[window_idx][bit_mask(bits)];
    }
    acc
}

fn binary_polys(
    batch_size: usize,
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let n = 1usize << num_vars;
    (0..batch_size)
        .map(|poly_idx| {
            let evals = (0..n)
                .map(|row_idx| {
                    let mut value = (row_idx as u32).wrapping_mul(0x9e37_79b9);
                    value ^= (poly_idx as u32).wrapping_mul(0x85eb_ca6b);
                    value = value.rotate_left(
                        u32::try_from((row_idx + poly_idx) % 32).expect("rotation must fit"),
                    );
                    value ^= value >> 16;
                    BinaryPoly::<32>::from(value)
                })
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, BinaryPoly::zero())
        })
        .collect()
}

fn bench_curve<C: AffineRepr>(
    c: &mut Criterion,
    curve_name: &str,
    batch_size: usize,
    width: usize,
    num_vars: usize,
) {
    let mut group = c.benchmark_group("hyrax_commit_breakdown");
    let lanes = batch_size * 32;
    let row = bool_row(width);
    let lane_rows = bool_values(lanes, width);
    let (bases, h) = bases_and_h::<C>(width);
    let tables = subset_tables::<C>(&bases, 6);
    let precomputed_blinds = (0..lanes)
        .map(|idx| scalar::<C>(idx + 17))
        .collect::<Vec<_>>();
    let msm_ck = msm_ck::<C>(width);
    let blind_one = MsmCommitmentEngine::<C>::blind(&msm_ck, width);
    let hyrax_ck = hyrax_ck::<C>(width);
    let polys = binary_polys(batch_size, num_vars);

    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/bool_row_msm"), width),
        &width,
        |b, _| {
            b.iter(|| {
                <BoolSubsetMsm<6> as RowMsmStrategy<C, bool>>::msm_row(
                    black_box(&msm_ck),
                    black_box(&row),
                )
                .expect("row MSM must succeed")
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/precomputed_bool_row_msm"), width),
        &width,
        |b, _| {
            b.iter(|| precomputed_bool_row::<C>(black_box(&tables), black_box(&row), 6));
        },
    );

    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/commit_one_bool_lane"), width),
        &width,
        |b, _| {
            b.iter(|| {
                MsmCommitmentEngine::<C>::commit_with::<bool, BoolSubsetMsm<6>>(
                    black_box(&msm_ck),
                    black_box(&row),
                    black_box(&blind_one),
                )
                .expect("one-lane commitment must succeed")
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            format!("{curve_name}/precomputed_commit_352_bool_lanes"),
            width,
        ),
        &width,
        |b, _| {
            b.iter(|| {
                let mut acc = Vec::with_capacity(lane_rows.len());
                for (values, blind) in lane_rows.iter().zip(precomputed_blinds.iter()) {
                    let mut commitment =
                        precomputed_bool_row::<C>(black_box(&tables), black_box(values), 6);
                    commitment += h * blind;
                    acc.push(commitment);
                }
                acc
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/commit_352_bool_lanes"), width),
        &width,
        |b, _| {
            b.iter(|| {
                let mut acc = Vec::with_capacity(lane_rows.len());
                for values in &lane_rows {
                    let blind = MsmCommitmentEngine::<C>::blind(&msm_ck, values.len());
                    let commitment =
                        MsmCommitmentEngine::<C>::commit_with::<bool, BoolSubsetMsm<6>>(
                            black_box(&msm_ck),
                            black_box(values),
                            black_box(&blind),
                        )
                        .expect("lane commitment must succeed");
                    acc.push(commitment);
                }
                acc
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new(format!("{curve_name}/hyrax_binary_commit_batch11"), width),
        &width,
        |b, _| {
            b.iter(|| {
                <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<32>, 32>>::commit(
                    black_box(&hyrax_ck),
                    black_box(&polys),
                )
                .expect("Hyrax binary commit must succeed")
            });
        },
    );

    group.finish();
}

fn hyrax_commit_breakdown(c: &mut Criterion) {
    let width = 512;
    let num_vars = 9;
    let batch_size = 11;

    bench_curve::<G1Affine>(c, "bn254", batch_size, width, num_vars);
    bench_curve::<ark_secp256k1::Affine>(c, "secp256k1", batch_size, width, num_vars);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = hyrax_commit_breakdown
}
criterion_main!(benches);
