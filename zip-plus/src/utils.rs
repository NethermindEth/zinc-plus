use crate::{
    pcs::structs::MulByScalar,
    traits::{ConstTranscribable, Named, SimpleSemiring, Transcribable},
};
use crypto_bigint::{BitOps, BoxedUint, Integer, Uint, Word};
use crypto_primitives::crypto_bigint_int::Int;
use itertools::Itertools;
use num_traits::{CheckedAdd, ConstOne, One, Zero};
use rand::{rngs::StdRng, seq::SliceRandom};
use rand_core::SeedableRng;
use std::ops::{Mul, SubAssign};

#[cfg(target_pointer_width = "64")]
const WORD_FACTOR: usize = 1;
#[cfg(target_pointer_width = "32")]
const WORD_FACTOR: usize = 2;

#[macro_export]
macro_rules! neg {
    ($a:expr) => {
        neg!($a, "Negation overflow")
    };
    ($a:expr, $msg:expr) => {
        $a.checked_neg().expect($msg)
    };
}

#[macro_export]
macro_rules! add {
    ($a:expr, $b:expr) => {
        add!($a, $b, "Addition overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_add($b).expect($msg)
    };
}

#[macro_export]
macro_rules! sub {
    ($a:expr, $b:expr) => {
        sub!($a, $b, "Subtraction overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_sub($b).expect($msg)
    };
}

#[macro_export]
macro_rules! mul {
    ($a:expr, $b:expr) => {
        mul!($a, $b, "Multiplication overflow")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_mul($b).expect($msg)
    };
}

#[macro_export]
macro_rules! div {
    ($a:expr, $b:expr) => {
        div!($a, $b, "Division by zero")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_div($b).expect($msg)
    };
}

#[macro_export]
macro_rules! rem {
    ($a:expr, $b:expr) => {
        rem!($a, $b, "Division by zero")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        $a.checked_rem($b).expect($msg)
    };
}

#[macro_export]
macro_rules! ilog_round_up {
    ($a:expr, $tp: ty) => {{
        let res = if $a.is_power_of_two() {
            $a.ilog2()
        } else {
            add!($a.ilog2(), 1)
        };
        <$tp>::try_from(res).expect(concat!("ilog doesn't fit ", stringify!($tp)))
    }};
}

pub(crate) fn inner_product<'a, 'b, Coeff, El, L, R>(lhs: L, rhs: R, zero: El) -> El
where
    Coeff: Clone + 'a + 'b,
    El: Clone + CheckedAdd + for<'z> MulByScalar<&'z Coeff> + 'a + 'b,
    L: IntoIterator<Item = &'a Coeff>,
    R: IntoIterator<Item = &'b El>,
{
    lhs.into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| {
            rhs.mul_by_scalar(lhs)
                .expect("Cannot multiply a codeword element by a coefficient")
        })
        .reduce(|acc, product| add!(acc, &product))
        .unwrap_or(zero)
}

pub(crate) fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    return rayon::current_num_threads();

    #[cfg(not(feature = "parallel"))]
    return 1;
}

pub(crate) fn parallelize_iter<I, T, F>(iter: I, f: F)
where
    I: Send + Iterator<Item = T>,
    T: Send,
    F: Fn(T) + Send + Sync + Clone,
{
    #[cfg(feature = "parallel")]
    rayon::scope(|scope| {
        iter.for_each(|item| {
            let f = &f;
            scope.spawn(move |_| f(item))
        })
    });

    #[cfg(not(feature = "parallel"))]
    iter.for_each(f);
}

pub(crate) fn parallelize<T, F>(v: &mut [T], f: F)
where
    T: Send,
    F: Fn((&mut [T], usize)) + Send + Sync + Clone,
{
    #[cfg(feature = "parallel")]
    {
        let num_threads = num_threads();
        let chunk_size = v.len().div_ceil(num_threads);
        if chunk_size < num_threads {
            f((v, 0));
        } else {
            parallelize_iter(v.chunks_mut(chunk_size).zip((0..).step_by(chunk_size)), f);
        }
    }

    #[cfg(not(feature = "parallel"))]
    f((v, 0));
}

/// Computes a linear combination of multiple evaluation rows into a single
/// combined row.
///
/// Given a flat `evaluations` vector, interpreted as a matrix with `row_len`
/// columns, this function treats each consecutive `row_len` values as one row.
/// The output is a single row, computed by multiplying each input row by the
/// corresponding coefficient from `coeffs`, and summing these scaled rows
/// column-wise.
///
/// This is equivalent to performing a matrix-vector multiplication where the
/// matrix is formed by the evaluations and the vector is formed by the
/// coefficients.
///
/// # Arguments
///
/// - `coeffs`: Coefficients applied to each row.
/// - `evaluations`: Flattened evaluations arranged row-wise.
/// - `row_len`: Number of columns per evaluation row.
///
/// # Returns
///
/// A vector of length `row_len` representing the combined row.
pub(super) fn combine_rows<Coeff, El>(
    coeffs: &[Coeff],
    evaluations: &[El],
    row_len: usize,
    zero: El,
) -> Vec<El>
where
    Coeff: Send + Sync,
    El: Clone + CheckedAdd + for<'z> MulByScalar<&'z Coeff> + Send + Sync,
{
    let mut combined_row = vec![zero; row_len];
    parallelize(&mut combined_row, |(combined_row, offset)| {
        combined_row
            .iter_mut()
            .zip(offset..)
            .for_each(|(combined, column)| {
                coeffs
                    .iter()
                    .zip(evaluations.iter().skip(column).step_by(row_len))
                    .for_each(|(coeff, eval)| {
                        *combined = add!(
                            combined,
                            &eval
                                .mul_by_scalar(coeff)
                                .expect("Cannot multiply evaluation by coefficient")
                        );
                    });
            })
    });

    combined_row
}

/// Reorder the elements in slice using the given randomness seed
pub(super) fn shuffle_seeded<T>(slice: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    slice.shuffle(&mut rng);
}

/// Reinterpret a `Vec<Self>` as a `Vec<Target>` **without copying**.
///
/// This trait provides a zero-copy conversion between vectors whose element
/// types are *layout-compatible*. It is meant for cases like transparent
/// newtypes (`#[repr(transparent)] struct Newtype(Inner);`) where the two
/// types have identical ABI (size and alignment) and the *value validity*
/// requirements of `Target` are **no stricter** than those of `Self`.
///
/// # What this does
///
/// Given ownership of a `Vec<Self>`, the implementation:
/// - Prevents the original vector from being dropped (`ManuallyDrop`),
/// - Reuses its allocation, pointer, length, and capacity unchanged,
/// - Reinterprets the element pointer as `*mut Target`,
/// - Constructs a new `Vec<Target>` with the same buffer.
///
/// No bytes are moved or rewritten. This is O(1).
///
/// # Why this trait is `unsafe`
///
/// This is an **`unsafe trait`** because **implementing** it asserts a
/// *per-type* contract that the compiler cannot verify. A bad impl can cause
/// undefined behavior in safe code that *uses* the impl. See
/// the *Implementor requirements* section below.
///
/// Additionally, the **method** is `unsafe` because each **call site** must
/// ensure the conversion is valid for the particular `Vec<Self>` value being
/// reinterpreted. (You may choose to make the method safe and keep only the
/// trait unsafe; keeping both is conservative but redundant.)
///
/// # Safety
///
/// ## Caller requirements (for each call to [`reinterpret_vector`])
///
/// You must uphold all of the following at the call site:
///
/// 1. **Layout equivalence** `size_of::<Self>() == size_of::<Target>()` and
///    `align_of::<Self>() == align_of::<Target>()`. This implementation
///    *asserts* these at compile time, failing the build if they do not hold.
///    (Do not remove those checks.)
///
/// 2. **Validity subset** Every bit-pattern that is a valid `Self` value must
///    also be a valid `Target` value. In other words, `Target` must **not**
///    impose stricter validity or niche invariants than `Self`.
///
///    **Counterexamples (do NOT reinterpret into these):**
///    - `*mut T` → `NonNull<T>` (null is valid for `*mut T` but **invalid** for
///      `NonNull<T>`).
///    - `u64` → `core::num::NonZeroU64`.
///    - `Option<NonNull<T>>` has niche optimization differences compared to raw
///      pointers.
///
/// 3. **Drop semantics remain sound** Dropping `Vec<Target>` must be sound for
///    all elements coming from `Vec<Self>`. If `Target` has a custom `Drop`
///    impl that relies on invariants not guaranteed by `Self`, the conversion
///    is unsound.
///
/// 4. **Ownership/uniqueness is preserved** After calling this function, the
///    original `Vec<Self>` **must not** be used again. (The function takes it
///    by value and places it in `ManuallyDrop` to prevent double-free.)
///
/// ## Implementor requirements (for each `unsafe impl`)
///
/// When you write `unsafe impl ReinterpretVector<Target> for Self`, you are
/// promising that for **any** `Vec<Self>`:
///
/// - `Self` and `Target` are layout-identical (same size and alignment) — e.g.
///   `Target` is `#[repr(transparent)]` over `Self`, or is otherwise guaranteed
///   to share ABI with `Self`.
/// - The set of valid `Self` values is a subset of the valid `Target` values.
///   No stricter validity or niche invariants are introduced by `Target`.
/// - Dropping a `Target` produced from arbitrary valid `Self` bytes is sound
///   (no additional assumptions in `Drop`).
///
/// If you cannot *globally* guarantee these at the type level, do **not**
/// implement this trait. Prefer a dedicated constructor that validates data or
/// a fallible conversion (`TryFrom<Vec<Self>> for Vec<Target>` that maps
/// elements).
///
/// ### Recommended: seal the trait
///
/// To prevent downstream crates from writing unsound impls, consider “sealing”
/// the trait (making it unimplementable outside your crate) and providing only
/// vetted `unsafe impl`s for your own types.
///
/// # Edge cases and notes
///
/// - **Zero-Sized Types (ZSTs):** Reinterpreting between ZSTs is fine if
///   alignment matches. `Vec<T>` for ZST `T` uses a dangling non-null pointer;
///   `Vec::from_raw_parts` supports this. The compile-time size/alignment
///   asserts still gate the conversion.
/// - **Uninhabited types (`!`):** `Vec<!>` cannot exist; do not attempt.
/// - **Provenance / aliasing:** We preserve the original allocation, pointer,
///   length, and capacity. Only the *type* of the element pointer changes (with
///   equal alignment), which is allowed under Rust’s aliasing rules for
///   layout-identical types when validity is preserved.
/// - **Send/Sync auto-traits:** `Vec<Target>` may have different `Send`/`Sync`
///   auto-trait behavior than `Vec<Self>`. That is a *type-system* effect at
///   compile time; the runtime conversion does not change the buffer contents.
/// - **Alternative (borrowed view):** If you only need a temporary view, prefer
///   reinterpreting slices to avoid ownership transfer: ```ignore // SAFETY:
///   same layout + lifetime/aliasing upheld let view: &[Target] = unsafe {
///   core::slice::from_raw_parts(vec_self.as_ptr().cast::<Target>(),
///   vec_self.len()) }; ```
///
/// # Examples
///
/// Transparent newtype over a primitive:
///
/// # Implementation notes
///
/// - The internal `const` block uses compile-time `assert!`s to fail builds
///   when layout differs. Keep these assertions to defend against accidental
///   misuse.
/// - The method uses `ManuallyDrop` to avoid dropping `source` before we
///   reconstruct a `Vec<Target>` from its raw parts. This prevents double-free.
/// - `Vec::from_raw_parts` is used in an `unsafe` block because we must
///   guarantee that `(ptr, len, cap)` originated from a
///   `Vec<Target>`-compatible allocation. The trait’s contract + layout asserts
///   provide that guarantee.
/// - Consider enabling `#![deny(unsafe_op_in_unsafe_fn)]` at the crate root so
///   that all unsafe operations within this `unsafe fn` remain explicitly
///   marked and audited.
///
/// # When to prefer a safe, copying conversion
///
/// If `Target` applies *any* additional invariants (non-zero, non-null, range
/// restrictions, encoding constraints, etc.), use a per-element conversion
/// (`map(Into::into)` or `try_map`) that can validate or fail gracefully.
/// Zero-copy reinterprets are only appropriate when the types are truly the
/// same at the byte level.
///
/// ---
pub unsafe trait ReinterpretVector<Target: Sized>: Sized {
    /// Reinterpret a `Vec<Self>` as a `Vec<Target>` **without copying**.
    ///
    /// See the trait-level documentation for full safety requirements and
    /// examples.
    ///
    /// # Safety
    ///
    /// Caller must ensure layout equivalence, validity preservation, and sound
    /// drop semantics for `Target` given elements produced from `Self`. After
    /// calling, the `source` vector must not be used again.
    unsafe fn reinterpret_vector(source: Vec<Self>) -> Vec<Target> {
        use std::mem::ManuallyDrop;
        const {
            // Keep these to enforce layout at compile time.
            assert!(std::mem::size_of::<Self>() == std::mem::size_of::<Target>());
            assert!(std::mem::align_of::<Self>() == std::mem::align_of::<Target>());
        }
        // Prevent `source` from being dropped while we steal its parts.
        let mut v = ManuallyDrop::new(source);
        let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
        // SAFETY:
        // - ptr/len/cap come from a `Vec<Self>` allocation of `cap` elements,
        // - size/align equality ensures `Vec<Target>` layout compatibility,
        // - ownership of the buffer is transferred, `source` will not be used again.
        unsafe { Vec::from_raw_parts(ptr.cast::<Target>(), len, cap) }
    }

    /// Reinterpret a borrowed slice `&[Self]` as `&[Target]` **without
    /// copying**.
    ///
    /// This is an O(1) borrowed view that preserves the original lifetime and
    /// does not take ownership of the buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// 1. **Layout equivalence**: `size_of::<Self>() == size_of::<Target>()`
    ///    and `align_of::<Self>() == align_of::<Target>()`. (Enforced via
    ///    compile-time asserts.)
    /// 2. **Validity subset**: Every valid `Self` bit-pattern is also a valid
    ///    `Target`. (No stricter validity or niche invariants are introduced by
    ///    `Target`.)
    /// 3. **Aliasing rules**: The returned `&[Target]` must not be used to
    ///    violate Rust’s aliasing model (e.g., you must not concurrently mutate
    ///    the same memory via `Self` references while holding this
    ///    `&[Target]`).
    ///
    /// Unlike the vector method, there are no drop-semantics concerns here
    /// because no ownership is transferred; this is a view only.
    unsafe fn reinterpret_slice(source: &[Self]) -> &[Target] {
        const {
            assert!(std::mem::size_of::<Self>() == std::mem::size_of::<Target>());
            assert!(std::mem::align_of::<Self>() == std::mem::align_of::<Target>());
        }
        // SAFETY:
        // - `source.as_ptr()` points to `len` contiguous `Self` elements.
        // - Equal size/align means it is valid to view the same bytes as `Target`.
        // - Lifetime `'a` is preserved; no ownership is taken.
        unsafe { core::slice::from_raw_parts(source.as_ptr().cast::<Target>(), source.len()) }
    }
}

unsafe impl<T> ReinterpretVector<T> for T {}

//
// Semiring wrapper for Uint
//

pub struct UintSemiring<const LIMBS: usize>(pub Uint<LIMBS>);

impl<const LIMBS: usize> ConstOne for UintSemiring<LIMBS> {
    const ONE: Self = Self(Uint::<LIMBS>::ONE);
}

impl<const LIMBS: usize> One for UintSemiring<LIMBS> {
    fn one() -> Self {
        Self::ONE
    }
}

impl<const LIMBS: usize> SubAssign for UintSemiring<LIMBS> {
    #[allow(clippy::arithmetic_side_effects)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<const LIMBS: usize> Mul for UintSemiring<LIMBS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl<const LIMBS: usize> SimpleSemiring for UintSemiring<LIMBS> {
    const BYTES: usize = Uint::<LIMBS>::NUM_BYTES;

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn is_even(&self) -> bool {
        self.0.is_even().into()
    }
}

//
// Named implementations
//

macro_rules! impl_named_for_primitives {
    ($($type:ty),+) => {
        $(
            impl Named for $type {
                fn type_name() -> String {
                    stringify!($type).to_string()
                }
            }
        )+
    };
}

impl_named_for_primitives!(i8, i16, i32, i64, i128);
impl_named_for_primitives!(u8, u16, u32, u64, u128);

impl<const LIMBS: usize> Named for Int<LIMBS> {
    fn type_name() -> String {
        format!("Int<{}>", LIMBS)
    }
}

impl<const LIMBS: usize> Named for UintSemiring<LIMBS> {
    fn type_name() -> String {
        format!("Uint<{}>", LIMBS)
    }
}

//
// Transcribable implementations
//

macro_rules! impl_transcribable_for_primitives {
    ($($type:ty),+) => {
        $(
            impl ConstTranscribable for $type {
                const NUM_BYTES: usize = std::mem::size_of::<$type>();

                fn read_transcription_bytes(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(bytes.try_into().expect("Invalid byte slice length"))
                }

                fn write_transcription_bytes(&self, buf: &mut [u8]) {
                    assert_eq!(buf.len(), std::mem::size_of::<$type>());
                    buf.copy_from_slice(&self.to_le_bytes());
                }
            }
        )+
    };
}

impl_transcribable_for_primitives!(u8, u16, u32, u64, u128);
impl_transcribable_for_primitives!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize> ConstTranscribable for Uint<LIMBS> {
    const NUM_BYTES: usize = 8 * LIMBS / WORD_FACTOR;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        let (chunked, rem) = bytes.as_chunks::<{ 8 / WORD_FACTOR }>();
        assert!(rem.is_empty(), "Invalid byte slice length for Uint");
        let words = chunked
            .iter()
            .map(|chunk| Word::from_le_bytes(*chunk))
            .collect_array::<LIMBS>()
            .expect("Invalid length for Uint");
        Uint::<LIMBS>::from_words(words)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        assert_eq!(buf.len(), Self::NUM_BYTES, "Buffer size mismatch for Uint");
        const W_SIZE: usize = size_of::<Word>();
        for (i, w) in self.as_words().iter().enumerate() {
            // Performance: reuse buffer and help compiler optimize away materializing
            // vector
            buf[(i * W_SIZE)..(i * W_SIZE + W_SIZE)].copy_from_slice(w.to_le_bytes().as_ref());
        }
    }
}

impl<const LIMBS: usize> ConstTranscribable for Int<LIMBS> {
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        (*<Uint<LIMBS> as ConstTranscribable>::read_transcription_bytes(bytes).as_int()).into()
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        ConstTranscribable::write_transcription_bytes(self.inner().as_uint(), buf)
    }
}

impl Transcribable for BoxedUint {
    /// Up to 255 bytes - so up to 2040 bits - should be plenty.
    const LENGTH_NUM_BYTES: usize = 1;

    fn read_num_bytes(bytes: &[u8]) -> usize {
        assert_eq!(bytes.len(), Self::LENGTH_NUM_BYTES);
        usize::from(bytes[0])
    }

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        // crypto_bigint::BoxedUint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        let (chunked, rem) = bytes.as_chunks::<{ 8 / WORD_FACTOR }>();
        assert!(rem.is_empty(), "Invalid byte slice length for BoxedUint");
        let words = chunked
            .iter()
            .map(|chunk| Word::from_le_bytes(*chunk))
            .collect_vec();
        BoxedUint::from_words(words)
    }

    fn get_num_bytes(&self) -> usize {
        usize::from(u8::try_from(self.bytes_precision()).expect("BoxedUint size must fit into u8"))
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        // crypto_bigint::BoxedUint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        assert_eq!(
            buf.len(),
            self.bytes_precision(),
            "Buffer size mismatch for BoxedUint"
        );
        const W_SIZE: usize = size_of::<Word>();
        for (i, w) in self.as_words().iter().enumerate() {
            // Performance: reuse buffer and help compiler optimize away materializing
            // vector
            buf[(i * W_SIZE)..(i * W_SIZE + W_SIZE)].copy_from_slice(w.to_le_bytes().as_ref());
        }
    }
}

impl<const LIMBS: usize> ConstTranscribable for UintSemiring<LIMBS> {
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        Self(<Uint<LIMBS> as ConstTranscribable>::read_transcription_bytes(bytes))
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        ConstTranscribable::write_transcription_bytes(&self.0, buf)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_inner_product_basic() {
        let lhs = [1, 2, 3];
        let rhs = [4, 5, 6];
        assert_eq!(inner_product(lhs.iter(), rhs.iter(), 0), 4 + 2 * 5 + 3 * 6);
    }

    #[test]
    fn test_basic_combination() {
        let coeffs = vec![1, 2];
        let evaluations = vec![3, 4, 5, 6];
        let row_len = 2;

        let result = combine_rows(&coeffs, &evaluations, row_len, 0);

        assert_eq!(result, vec![(3 + 2 * 5), (4 + 2 * 6)]);
    }

    #[test]
    fn test_second_combination() {
        let coeffs = vec![3, 4];
        let evaluations = vec![2, 4, 6, 8];
        let row_len = 2;

        let result = combine_rows(&coeffs, &evaluations, row_len, 0);

        assert_eq!(result, vec![(3 * 2 + 4 * 6), (3 * 4 + 4 * 8)]);
    }
    #[test]
    fn test_large_values() {
        let coeffs = vec![1000, -500];
        let evaluations = vec![2000, -3000, 4000, -5000];
        let row_len = 2;

        let result = combine_rows(&coeffs, &evaluations, row_len, 0);

        assert_eq!(
            result,
            vec![
                (1000 * 2000 + (-500) * 4000),
                (1000 * -3000 + (-500) * -5000)
            ]
        );
    }
}
