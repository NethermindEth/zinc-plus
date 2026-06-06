use std::{fmt::Debug, marker::PhantomData};

use ark_ec::AffineRepr;
use crypto_primitives::PrimeField;
use zinc_poly::univariate::{binary::BinaryPoly, dense::DensePolynomial};
use zip_plus::pcs::{
    generic::{FoldablePCS, PCS, ZipPlusPCS},
    hyrax::{BinaryLanes, DensePolyScalarLanes, HyraxPCS, IntScalarLane},
    structs::ZipPlusCommitment,
};

use crate::ZincTypes;

pub type ZipPCSCommitments = (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment);

pub trait ZincPCSTypes<Zt, F, const D: usize>: Clone + Debug + Send + Sync
where
    Zt: ZincTypes<D>,
    F: PrimeField,
{
    type BinaryPCS: PCS<F, BinaryPoly<D>, D>;
    type ArbitraryPCS: PCS<F, DensePolynomial<Zt::Int, D>, D>;
    type IntPCS: PCS<F, Zt::Int, D>;
}

/// PCS bundle allowed by production SHA ProjectionFold.
///
/// Production ProjectionFold folds instance commitments after SumFold, so all
/// committed witness domains must be homomorphic. `AllHyraxPCSTypes` satisfies
/// this today; Zip+/Merkle-based bundles intentionally do not.
pub trait ProductionShaPCS<Zt, F, const D: usize>: ZincPCSTypes<Zt, F, D>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    Self::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    Self::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    Self::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
}

#[derive(Clone, Debug)]
pub struct AllZipPCSTypes;

impl<Zt, F, const D: usize> ZincPCSTypes<Zt, F, D> for AllZipPCSTypes
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    ZipPlusPCS<Zt::BinaryZt, Zt::BinaryLc>: PCS<F, BinaryPoly<D>, D>,
    ZipPlusPCS<Zt::ArbitraryZt, Zt::ArbitraryLc>: PCS<F, DensePolynomial<Zt::Int, D>, D>,
    ZipPlusPCS<Zt::IntZt, Zt::IntLc>: PCS<F, Zt::Int, D>,
{
    type BinaryPCS = ZipPlusPCS<Zt::BinaryZt, Zt::BinaryLc>;
    type ArbitraryPCS = ZipPlusPCS<Zt::ArbitraryZt, Zt::ArbitraryLc>;
    type IntPCS = ZipPlusPCS<Zt::IntZt, Zt::IntLc>;
}

#[derive(Clone, Debug)]
pub struct BinaryHyraxZipRest<C: AffineRepr>(PhantomData<C>);

impl<Zt, F, C, const D: usize> ZincPCSTypes<Zt, F, D> for BinaryHyraxZipRest<C>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    C: AffineRepr,
    HyraxPCS<C, BinaryLanes>: PCS<F, BinaryPoly<D>, D>,
    ZipPlusPCS<Zt::ArbitraryZt, Zt::ArbitraryLc>: PCS<F, DensePolynomial<Zt::Int, D>, D>,
    ZipPlusPCS<Zt::IntZt, Zt::IntLc>: PCS<F, Zt::Int, D>,
{
    type BinaryPCS = HyraxPCS<C, BinaryLanes>;
    type ArbitraryPCS = ZipPlusPCS<Zt::ArbitraryZt, Zt::ArbitraryLc>;
    type IntPCS = ZipPlusPCS<Zt::IntZt, Zt::IntLc>;
}

/// Homomorphic PCS bundle for production ProjectionFold paths.
///
/// Every witness domain uses Hyrax/MSM commitments so verifier-derived
/// instance-axis folds can be opened against folded prover data.
#[derive(Clone, Debug)]
pub struct AllHyraxPCSTypes<C: AffineRepr>(PhantomData<C>);

impl<Zt, F, C, const D: usize> ZincPCSTypes<Zt, F, D> for AllHyraxPCSTypes<C>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    C: AffineRepr,
    HyraxPCS<C, BinaryLanes>: PCS<F, BinaryPoly<D>, D>,
    HyraxPCS<C, DensePolyScalarLanes>: PCS<F, DensePolynomial<Zt::Int, D>, D>,
    HyraxPCS<C, IntScalarLane>: PCS<F, Zt::Int, D>,
{
    type BinaryPCS = HyraxPCS<C, BinaryLanes>;
    type ArbitraryPCS = HyraxPCS<C, DensePolyScalarLanes>;
    type IntPCS = HyraxPCS<C, IntScalarLane>;
}

impl<Zt, F, C, const D: usize> ProductionShaPCS<Zt, F, D> for AllHyraxPCSTypes<C>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    C: AffineRepr,
    HyraxPCS<C, BinaryLanes>: PCS<F, BinaryPoly<D>, D> + FoldablePCS<F, BinaryPoly<D>, D>,
    HyraxPCS<C, DensePolyScalarLanes>:
        PCS<F, DensePolynomial<Zt::Int, D>, D> + FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    HyraxPCS<C, IntScalarLane>: PCS<F, Zt::Int, D> + FoldablePCS<F, Zt::Int, D>,
{
}

#[derive(Clone, Debug)]
pub struct PCSParams<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub binary:
        <<P as ZincPCSTypes<Zt, F, D>>::BinaryPCS as PCS<F, BinaryPoly<D>, D>>::CommitmentKey,
    pub arbitrary: <<P as ZincPCSTypes<Zt, F, D>>::ArbitraryPCS as PCS<
        F,
        DensePolynomial<Zt::Int, D>,
        D,
    >>::CommitmentKey,
    pub int: <<P as ZincPCSTypes<Zt, F, D>>::IntPCS as PCS<F, Zt::Int, D>>::CommitmentKey,
}

#[derive(Clone, Debug)]
pub struct PCSVerifierParams<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub binary: <<P as ZincPCSTypes<Zt, F, D>>::BinaryPCS as PCS<F, BinaryPoly<D>, D>>::VerifierKey,
    pub arbitrary: <<P as ZincPCSTypes<Zt, F, D>>::ArbitraryPCS as PCS<
        F,
        DensePolynomial<Zt::Int, D>,
        D,
    >>::VerifierKey,
    pub int: <<P as ZincPCSTypes<Zt, F, D>>::IntPCS as PCS<F, Zt::Int, D>>::VerifierKey,
}

#[derive(Clone, Debug)]
pub struct PCSCommitments<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub binary: <<P as ZincPCSTypes<Zt, F, D>>::BinaryPCS as PCS<F, BinaryPoly<D>, D>>::Commitment,
    pub arbitrary: <<P as ZincPCSTypes<Zt, F, D>>::ArbitraryPCS as PCS<
        F,
        DensePolynomial<Zt::Int, D>,
        D,
    >>::Commitment,
    pub int: <<P as ZincPCSTypes<Zt, F, D>>::IntPCS as PCS<F, Zt::Int, D>>::Commitment,
}

#[derive(Clone, Debug)]
pub struct PCSProverData<P, Zt, F, const D: usize>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
{
    pub binary: <<P as ZincPCSTypes<Zt, F, D>>::BinaryPCS as PCS<F, BinaryPoly<D>, D>>::ProverData,
    pub arbitrary: <<P as ZincPCSTypes<Zt, F, D>>::ArbitraryPCS as PCS<
        F,
        DensePolynomial<Zt::Int, D>,
        D,
    >>::ProverData,
    pub int: <<P as ZincPCSTypes<Zt, F, D>>::IntPCS as PCS<F, Zt::Int, D>>::ProverData,
}
