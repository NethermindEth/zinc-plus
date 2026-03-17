//! UAIR description tools.

pub mod collect_scalars;
pub mod constraint_counter;
pub mod degree_counter;
pub mod do_nothing_builder;
pub mod dummy_semiring;
pub mod ideal;
pub mod ideal_collector;

use crypto_primitives::Semiring;
use zinc_utils::{UNCHECKED, from_ref::FromRef, mul_by_scalar::MulByScalar};

use crate::ideal::{Ideal, IdealCheck};

/// The abstract interface to constraint building logic.
/// In essence it allows to create constraints modulo ideals.
pub trait ConstraintBuilder {
    /// The expressions the constraint builder operates on.
    /// It is opaque from the PoV of an AIR apart from
    /// the fact that arithmetic operations are available on it
    /// and one can check if an expression is in an ideal.
    type Expr: Semiring;
    /// The type of ideals used by the constraint builder.
    type Ideal: Ideal + IdealCheck<Self::Expr>;

    /// Add a constraint saying that `expr` belongs to the ideal `ideal`.
    fn assert_in_ideal(&mut self, expr: Self::Expr, ideal: &Self::Ideal);

    /// Add a constraint saying that `expr` is equal to zero which is
    /// the same as saying that `expr` belongs to the zero ideal.
    fn assert_zero(&mut self, expr: Self::Expr);
}

/// The signature of a UAIR.
/// Contains the number of columns of
/// each of the types: binary polynomials,
/// polynomials with arbitrary coefficients,
/// and integers.
///
/// Public columns precede witness columns within each type group.
/// The flattened trace ordering is:
/// `[pub_bin, wit_bin, pub_arb, wit_arb, pub_int, wit_int]`.
#[derive(Default)]
pub struct UairSignature {
    /// Number of public columns with binary polynomial elements.
    pub public_binary_poly_cols: usize,
    /// Number of witness columns with binary polynomial elements.
    pub witness_binary_poly_cols: usize,
    /// Number of public columns with arbitrary polynomial elements.
    pub public_arbitrary_poly_cols: usize,
    /// Number of witness columns with arbitrary polynomial elements.
    pub witness_arbitrary_poly_cols: usize,
    /// Number of public columns with integers.
    pub public_int_cols: usize,
    /// Number of witness columns with integers.
    pub witness_int_cols: usize,
}

#[allow(clippy::arithmetic_side_effects)]
impl UairSignature {
    pub fn total_binary_poly_cols(&self) -> usize {
        self.public_binary_poly_cols + self.witness_binary_poly_cols
    }

    pub fn total_arbitrary_poly_cols(&self) -> usize {
        self.public_arbitrary_poly_cols + self.witness_arbitrary_poly_cols
    }

    pub fn total_int_cols(&self) -> usize {
        self.public_int_cols + self.witness_int_cols
    }

    pub fn total_public_cols(&self) -> usize {
        self.public_binary_poly_cols + self.public_arbitrary_poly_cols + self.public_int_cols
    }

    /// Maximum number of columns across the three types (public + witness per
    /// type).
    pub fn max_cols(&self) -> usize {
        [
            self.total_binary_poly_cols(),
            self.total_arbitrary_poly_cols(),
            self.total_int_cols(),
        ]
        .into_iter()
        .max()
        .expect("the iterator is not empty")
    }

    /// The sum of the numbers of columns across all types (public + witness).
    pub fn total_cols(&self) -> usize {
        self.total_binary_poly_cols() + self.total_arbitrary_poly_cols() + self.total_int_cols()
    }
}

/// A view on a row of the trace.
/// Contains references to cells of the trace
/// of all types lying in the same trace row.
#[derive(Clone, Copy)]
pub struct TraceRow<'a, Expr> {
    pub binary_poly: &'a [Expr],
    pub arbitrary_poly: &'a [Expr],
    pub int: &'a [Expr],
}

impl<'a, Expr> TraceRow<'a, Expr> {
    /// Given a slice that represents a raw row of the trace,
    /// creates a `TraceRow` from it.
    /// Subdivides the slice according to the given signature `signature`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_slice_with_signature(row: &'a [Expr], signature: &UairSignature) -> Self {
        let nb = signature.total_binary_poly_cols();
        let na = signature.total_arbitrary_poly_cols();
        Self {
            binary_poly: &row[0..nb],
            arbitrary_poly: &row[nb..nb + na],
            int: &row[nb + na..],
        }
    }
}

/// The trait that a universal AIR description has to implement.
/// This must include all the constraint description logic of an UAIR.
///
/// One type might implement different UAIR logics for different underlying
/// semirings hence the generic type parameter.
pub trait Uair {
    /// The ideal type the AIR operates with.
    /// Since a `ConstraintBuilder` is "opaque" for a `Uair`
    /// a `Uair` has to have a means to create ideals
    /// so ideals are fixed by this associated types.
    /// At the `constrain*` methods a `Uair` is given
    /// a way to convert its own ideals into builder's ideals
    /// via the `FromRef` trait.
    type Ideal: Ideal;

    /// The type of scalars of the UAIR.
    /// For now, we assume they are of
    /// the type "arbitrary polynomials".
    // Note: This is usually Z_32[X] (i.e. DensePolynomial<Ring, 32>), but according
    // to @agareta, this in not always the case.
    type Scalar: Semiring;

    /// Signature of the UAIR.
    fn signature() -> UairSignature;

    /// A general method for describing constraints.
    ///
    /// # Arguments
    /// - `b`: a builder encapsulating the constraint storing logic. Its type
    ///   `B` has to have compatible `B::Ideal` with the `Self::Ideal`, i.e. it
    ///   must implement `FromRef<Self::Ideal>` trait.
    /// - `up`: a `TraceRow` of expressions representing the current row of
    ///   UAIR.
    /// - `down`: a `TraceRow` of expressions representing the next row of UAIR.
    ///   It is safe to assume all the members have the same lengths as
    ///   corresponding members of `up`.
    /// - `from_ref`: a closure that turns the underlying ring `R` into
    ///   `B::Expr`. Sometimes (e.g. when dealing with random fields) it is
    ///   convenient to provide a closure instead of a `FromRef` implementation.
    /// - `mbs`: a closure that allows to multiply expressions by `R`. Same
    ///   rationale as for `from_ref`.
    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal;

    // Same as `constrain_general` but `from_ref` and `mbs`
    // come from the trait implementations.
    fn constrain<B>(b: &mut B, up: TraceRow<B::Expr>, down: TraceRow<B::Expr>)
    where
        B: ConstraintBuilder,
        B::Expr: FromRef<Self::Scalar> + for<'b> MulByScalar<&'b Self::Scalar>,
        B::Ideal: FromRef<Self::Ideal>,
    {
        Self::constrain_general(
            b,
            up,
            down,
            B::Expr::from_ref,
            |x, y| B::Expr::mul_by_scalar::<UNCHECKED>(x, y),
            B::Ideal::from_ref,
        )
    }
}
