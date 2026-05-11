use zinc_poly::mle::DenseMultilinearExtension;

/// Fold a trace of one type into a trace of another (similar but smaller) type.
///
/// Note that folding will increase number of variables for MLEs by
/// `ilog2(Self::FOLDING_FACTOR)`.
pub trait FoldTrace<From, To> {
    /// Folding factor, a positive power of 2.
    const FOLDING_FACTOR: usize;

    fn fold_trace_column(column: &DenseMultilinearExtension<From>)
    -> DenseMultilinearExtension<To>;
}

//
// NOOP fold
//

pub struct NoopFoldTrace;

impl<T: Clone> FoldTrace<T, T> for NoopFoldTrace {
    const FOLDING_FACTOR: usize = 1;

    fn fold_trace_column(trace: &DenseMultilinearExtension<T>) -> DenseMultilinearExtension<T> {
        trace.clone()
    }
}
