use crypto_primitives::PrimeField;

/// A trait for fields that allow to perform operations
/// on inner Montgomery representations of field elements.
pub trait InnerTransparentField: PrimeField {
    /// Add inner Montgomery representations using a config.
    fn add_inner(lhs: &Self::Inner, rhs: &Self::Inner, config: &Self::Config) -> Self::Inner;

    /// Subtract inner Montgomery representations using a config.
    fn sub_inner(lhs: &Self::Inner, rhs: &Self::Inner, config: &Self::Config) -> Self::Inner;

    /// Multiply a field element by an inner Montgomery representation.
    fn mul_assign_by_inner(&mut self, rhs: &Self::Inner);
}
