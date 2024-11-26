//! A collection of traits and macros which convert primitive integer types into field elements.

/// A simple macro which lets us cleanly define the function `from_Int`
/// with `Int` can be replaced by any integer type.
///
/// Running, `from_integer_types(Int)` adds the following code to a trait:
///
/// ```rust,ignore
/// /// Given an integer `r`, return the sum of `r` copies of `ONE`:
/// ///
/// /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
/// ///
/// /// Note that the output only depends on `r mod p`.
/// ///
/// /// This should be avoided in performance critical locations.
/// fn from_Int(int: Int) -> Self {
///     Self::from_char(Self::Char::from_int(int))
/// }
/// ```
///
/// This macro can be run for any `Int` where `Self::Char` implements `QuotientMap<Int>`.
/// It considerably cuts down on the amount of copy/pasted code.
macro_rules! from_integer_types {
    ($($type:ty),* $(,)? ) => {
        $( paste!{
        /// Given an integer `r`, return the sum of `r` copies of `ONE`:
        ///
        /// `r.Self::ONE =  Self::ONE + ... + Self::ONE (r times)`.
        ///
        /// Note that the output only depends on `r mod p`.
        ///
        /// This should be avoided in performance critical locations.
        fn [<from_ $type>](int: $type) -> Self {
            Self::from_char(Self::Char::from_int(int))
        }
    }
        )*
    };
}

pub(crate) use from_integer_types;

/// Implementation of the quotient map `ℤ -> ℤ/p` which sends an integer `r` to its conjugacy class `[r]`.
///
/// This is the key trait allowing us to convert integers into field elements. Each prime field
/// should implement this for all primitive integer types.
pub trait QuotientMap<Int>: Sized {
    /// Convert a given integer into an element of the field `ℤ/p`.
    ///   
    /// This is the most generic method which makes no assumptions on the size of the input.
    /// Where possible, this method should be used with the smallest possible integer type.
    /// For example, if a 32-bit integer `x` is known to be less than `2^16`, then
    /// `from_int(x as u16)` will often be faster than `from_int(x)`.
    ///
    /// This method is also strongly preferred over `from_canonical_checked/from_canonical_unchecked`.
    /// It will usually be identical when `Int` is a small type, e.g. `u8/u16` and is safer for
    /// larger types.
    fn from_int(int: Int) -> Self;

    /// Convert a given integer into an element of the field `ℤ/p`. The input is checked to
    /// ensure it lies within a given range.
    /// - If `Int` is an unsigned integer type the input must lie in `[0, p - 1]`.
    /// - If `Int` is a signed integer type the input must lie in `[-(p - 1)/2, (p - 1)/2]`.
    ///
    /// Return `None` if the input lies outside this range and `Some(val)` otherwise.
    fn from_canonical_checked(int: Int) -> Option<Self>;

    /// Convert a given integer into an element of the field `ℤ/p`. The input is guaranteed
    /// to lie within a specific range depending on `p`. If the input lies outside of this
    /// range, the output is undefined.
    ///
    /// In general `from_canonical_unchecked` will be faster for either `signed` or `unsigned`
    /// types but the specifics will depend on the field.
    ///
    /// # Safety
    /// - If `Int` is an unsigned integer type then the allowed range will include `[0, p - 1]`.
    /// - If `Int` is a signed integer type then the allowed range will include `[-(p - 1)/2, (p - 1)/2]`.
    unsafe fn from_canonical_unchecked(int: Int) -> Self;
}

/// If the integer type is smaller than the field order we all possible inputs are canonical.
/// In such a case we can easily implement `QuotientMap<Int>` as all three methods will coincide.
/// The range of acceptable integer types depends on the size of the field:
/// - For 31 bit fields, `Int = u8, u16`.
/// - For 64 bit fields, `Int = u8, u16, u32`.
/// - For large fields (E.g. `Bn254`), `Int` can be anything except for the largest primitive integer type `u128`
///
/// TODO: Give a run down for how this works:
#[macro_export]
macro_rules! quotient_small_u_int {
    ($field:ty, $max_size:ty, $($prim_int:ty),* ) => {
        $(
        impl QuotientMap<$prim_int> for $field {
            /// Due to the size of the `BN254` prime, the input value is always canonical.
            #[inline]
            fn from_int(int: $prim_int) -> $field {
                assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
                unsafe {
                    Self::from_canonical_unchecked(int as $max_size)
                }
            }

            /// Due to the size of the `BN254` prime, the input value is always canonical.
            #[inline]
            fn from_canonical_checked(int: $prim_int) -> Option<$field> {
                assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
                Some(unsafe {
                    Self::from_canonical_unchecked(int as $max_size)
                })
            }

            /// Due to the size of the `BN254` prime, the input value is always canonical.
            #[inline]
            unsafe fn from_canonical_unchecked(int: $prim_int) -> $field {
                Self::from_canonical_unchecked(int as $max_size)
            }
        }
        )*
    };
}

pub use quotient_small_u_int;
