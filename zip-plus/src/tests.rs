#![allow(clippy::unwrap_used)]

use ark_std::{UniformRand, vec, vec::Vec};

use crate::{
    code::DefaultLinearCodeSpec,
    code_raa::RaaCode,
    define_field_config, define_random_field_zip_types,
    field::{Int, RandomField},
    implement_random_field_zip_types,
    pcs::structs::MultilinearZip,
    pcs_transcript::PcsTranscript,
    poly_z::mle::DenseMultilinearExtension,
    traits::FieldMap,
    transcript::KeccakTranscript,
};

const I: usize = 1;
const N: usize = 2;

define_random_field_zip_types!();
implement_random_field_zip_types!(I);

type ZT = RandomFieldZipTypes<I>;
type LC = RaaCode<ZT>;
type TestZip<LC> = MultilinearZip<ZT, LC>;

define_field_config!(FC, "57316695564490278656402085503");

#[test]
fn test_zip_commitment() {
    let mut transcript = KeccakTranscript::new();
    let poly_size = 8;
    let linear_code: LC = LC::new(&DefaultLinearCodeSpec, poly_size, &mut transcript);
    let param = TestZip::setup(poly_size, linear_code);

    let evaluations: Vec<_> = (0..8).map(Int::<I>::from).collect();

    let n = 3;
    let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

    let res = TestZip::commit::<RandomField<N, FC<N>>>(&param, &mle);

    assert!(res.is_ok())
}

#[test]
fn test_failing_zip_commitment() {
    let mut transcript = KeccakTranscript::new();
    let poly_size = 8;
    let linear_code: LC = LC::new(&DefaultLinearCodeSpec, poly_size, &mut transcript);
    let param = TestZip::setup(poly_size, linear_code);

    let evaluations: Vec<_> = (0..16).map(Int::<I>::from).collect();
    let n = 4;
    let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

    let res = TestZip::commit::<RandomField<N, FC<N>>>(&param, &mle);

    assert!(res.is_err())
}

#[test]
fn test_zip_opening() {
    let poly_size = 8;
    let mut keccak_transcript = KeccakTranscript::new();
    let linear_code: LC = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let param = TestZip::setup(poly_size, linear_code);

    let mut transcript = PcsTranscript::<RandomField<N, FC<N>>>::new();

    let evaluations: Vec<_> = (0..8).map(Int::<I>::from).collect();
    let n = 3;
    let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

    let (data, _) = TestZip::commit::<RandomField<N, FC<N>>>(&param, &mle).unwrap();

    let point = vec![0i64, 0i64, 0i64].map_to_field();

    let res = TestZip::open(&param, &mle, &data, &point, &mut transcript);

    assert!(res.is_ok())
}

#[test]
fn test_failing_zip_evaluation() {
    type F = RandomField<N, FC<N>>;

    let poly_size = 8;
    let mut keccak_transcript = KeccakTranscript::new();
    let linear_code: LC = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let param = TestZip::setup(poly_size, linear_code);

    let evaluations: Vec<_> = (0..8).map(Int::<I>::from).collect();
    let n = 3;
    let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

    let (data, comm) = TestZip::commit::<RandomField<N, FC<N>>>(&param, &mle).unwrap();

    let point = vec![0i64, 0i64, 0i64].map_to_field();
    let eval: F = 7i64.map_to_field();

    let mut transcript = PcsTranscript::new();
    let _ = TestZip::open(&param, &mle, &data, &point, &mut transcript);

    let proof = transcript.into_proof();
    let mut transcript = PcsTranscript::from_proof(&proof);
    let res = TestZip::verify(&param, &comm, &point, eval, &mut transcript);

    assert!(res.is_err())
}

#[test]
fn test_zip_evaluation() {
    type F<'cfg> = RandomField<N, FC<N>>;
    let mut rng = ark_std::test_rng();

    let n = 8;
    let poly_size = 1 << n;
    let mut keccak_transcript = KeccakTranscript::new();
    let linear_code: LC = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let param = TestZip::setup(poly_size, linear_code);
    let evaluations: Vec<_> = (0..(1 << n))
        .map(|_| Int::<I>::from(i8::rand(&mut rng)))
        .collect();
    let mle = DenseMultilinearExtension::from_evaluations_slice(n, &evaluations);

    let (data, comm) = TestZip::commit::<RandomField<N, FC<N>>>(&param, &mle).unwrap();

    let point: Vec<_> = (0..n).map(|_| Int::<I>::from(i8::rand(&mut rng))).collect();
    let eval: F = mle.evaluate(&point).unwrap().map_to_field();

    let point = point.map_to_field();
    let mut transcript = PcsTranscript::new();
    let _ = TestZip::open(&param, &mle, &data, &point, &mut transcript);

    let proof = transcript.into_proof();
    let mut transcript = PcsTranscript::from_proof(&proof);
    TestZip::verify(&param, &comm, &point, eval, &mut transcript).expect("Failed to verify");
}
#[test]
fn test_zip_batch_evaluation() {
    type F<'cfg> = RandomField<N, FC<N>>;
    let mut rng = ark_std::test_rng();

    let n = 8;
    // the number of polynomials we will batch verify;
    let m = 10;
    let poly_size = 1 << n;
    let mut keccak_transcript = KeccakTranscript::new();
    let linear_code: LC = LC::new(&DefaultLinearCodeSpec, poly_size, &mut keccak_transcript);
    let param = TestZip::setup(poly_size, linear_code);
    let evaluations: Vec<Vec<Int<I>>> = (0..m)
        .map(|_| {
            (0..(1 << n))
                .map(|_| Int::<I>::from(i8::rand(&mut rng)))
                .collect::<Vec<Int<I>>>()
        })
        .collect();

    let mles: Vec<_> = evaluations
        .iter()
        .map(|evaluations| DenseMultilinearExtension::from_evaluations_slice(n, evaluations))
        .collect();

    let commitments: Vec<_> =
        TestZip::batch_commit::<RandomField<N, FC<N>>>(&param, &mles).unwrap();
    let (data, commitments): (Vec<_>, Vec<_>) = commitments.into_iter().unzip();
    let point: Vec<_> = (0..n).map(|_| Int::<I>::from(i8::rand(&mut rng))).collect();
    let eval: Vec<_> = mles
        .iter()
        .map(|mle| mle.evaluate(&point).unwrap().map_to_field())
        .collect();

    let point: Vec<F> = point.map_to_field();
    let points: Vec<_> = (0..m).map(|_| point.clone()).collect();
    let mut transcript = PcsTranscript::new();
    let _ = TestZip::batch_open(&param, &mles, &data, &points, &mut transcript);

    let proof = transcript.into_proof();
    let mut transcript = PcsTranscript::from_proof(&proof);
    TestZip::batch_verify_z(&param, &commitments, &points, &eval, &mut transcript)
        .expect("Failed to verify");
}
