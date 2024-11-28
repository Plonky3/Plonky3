//! A collection of traits and macros which convert primitive integer types into field elements.

/// A simple macro which lets us cleanly define the function `from_Int`
/// with `Int` can be replaced by any integer type.
///
/// Running, `from_integer_types!(Int)` adds the following code to a trait:
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
/// - For 31 bit fields, `Int = u8, u16, i8, i16`.
/// - For 64 bit fields, `Int = u8, u16, u32, i8, i16, i32`.
/// - For large fields (E.g. `Bn254`), `Int` can be anything except for the largest primitive integer type `u128/i128`
///
/// This macro accepts 3 inputs.
/// - The name of the prime field `P`
/// - The natural integer type `Int` in which the field characteristic lives.
/// - A list of smaller integer types to auto implement `QuotientMap<SmallInt>`.
///
/// Then `from_int`, `from_canonical_checked`, `from_canonical_unchecked` are all
/// implemented by casting the input to an `Int` and using the `from_canonical_unchecked`
/// method from `QuotientMap<Int>`.
///
/// For a concrete example, `quotient_small_u_int!(Mersenne31, u32, [u8])` would produce the following code:
///
/// ```rust,ignore
/// impl QuotientMap<u8> for Mersenne31 {
///     /// Due to the integer type, the input value is always canonical.
///     #[inline]
///     fn from_int(int: u8) -> Mersenne31 {
///         // Should be removed by the compiler.
///         assert!(size_of::<u8>() <= size_of::<u32>());
///         unsafe {
///             Self::from_canonical_unchecked(int as u32)
///         }
///     }
///
///     /// Due to the integer type, the input value is always canonical.
///     #[inline]
///     fn from_canonical_checked(int: u8) -> Option<Mersenne31> {
///         // Should be removed by the compiler.
///         assert!(size_of::<u8>() <= size_of::<u32>());
///         Some(unsafe {
///             Self::from_canonical_unchecked(int as u32)
///         })
///     }
///
///     /// Due to the integer type, the input value is always canonical.
///     #[inline]
///     unsafe fn from_canonical_unchecked(int: u8) -> Mersenne31 {
///         // We use debug_assert to ensure this is removed by the compiler in release mode.
///         debug_assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
///         Self::from_canonical_unchecked(int as u32)
///     }
/// }
///```
///
/// Similarly, `quotient_small_u_int!(Mersenne31, u32, [u8, u16])` produces the above code along with
/// an identical version with `u8` replaced by `u16`.
///
/// All fields will need to use this method twice. Once for unsigned ints and once for signed ints.
#[macro_export]
macro_rules! quotient_map_small_int {
    ($field:ty, $max_size:ty, [$($prim_int:ty),*] ) => {
        $(
        paste!{
            impl QuotientMap<$prim_int> for $field {
                #[doc = "Convert a given "]
                #[doc = stringify!(type_name::<$prim_int>())]
                #[doc = " integer into an element of the "]
                #[doc = stringify!(type_name::<$field>())]
                #[doc = " field.
                \n Due to the integer type, the input value is always canonical."]
                #[inline]
                fn from_int(int: $prim_int) -> $field {
                    // Should be removed by the compiler.
                    assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
                    unsafe {
                        Self::from_canonical_unchecked(int as $max_size)
                    }
                }

                /// Due to the integer type, the input value is always canonical.
                #[inline]
                fn from_canonical_checked(int: $prim_int) -> Option<$field> {
                    // Should be removed by the compiler.
                    assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
                    Some(unsafe {
                        Self::from_canonical_unchecked(int as $max_size)
                    })
                }

                /// Due to the integer type, the input value is always canonical.
                #[inline]
                unsafe fn from_canonical_unchecked(int: $prim_int) -> $field {
                    // We use debug_assert to ensure this is removed by the compiler in release mode.
                    debug_assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
                    Self::from_canonical_unchecked(int as $max_size)
                }
            }
        }
        )*
    };
}

/// If the integer type is large enough, there is often no method better for `from_int` than
/// just doing a modular reduction to reduce to a smaller type.
///
/// This provides a simple macro for this this implementation.
///
/// This macro accepts 4 inputs.
/// - The name of the prime field `P`
/// - The natural integer type `Int` in which the field characteristic lives.
/// - The characteristic of the field.
/// - A list of smaller integer types to auto implement `QuotientMap<SmallInt>`.
///
/// Then `from_int`, `from_canonical_checked`, `from_canonical_unchecked` are all
/// implemented by casting the input to an `Int` and using the `from_canonical_unchecked`
/// method from `QuotientMap<Int>`.
///
/// For a concrete example, `quotient_small_u_int!(Mersenne31, u32, [u8])` would produce the following code:
///
/// ```rust,ignore
/// impl QuotientMap<u8> for Mersenne31 {
///     /// Due to the integer type, the input value is always canonical.
///     #[inline]
///     fn from_int(int: u8) -> Mersenne31 {
///         // Should be removed by the compiler.
///         assert!(size_of::<u8>() <= size_of::<u32>());
///         unsafe {
///             Self::from_canonical_unchecked(int as u32)
///         }
///     }
///
///     /// Due to the integer type, the input value is always canonical.
///     #[inline]
///     fn from_canonical_checked(int: u8) -> Option<Mersenne31> {
///         // Should be removed by the compiler.
///         assert!(size_of::<u8>() <= size_of::<u32>());
///         Some(unsafe {
///             Self::from_canonical_unchecked(int as u32)
///         })
///     }
///
///     /// Due to the integer type, the input value is always canonical.
///     #[inline]
///     unsafe fn from_canonical_unchecked(int: u8) -> Mersenne31 {
///         // We use debug_assert to ensure this is removed by the compiler in release mode.
///         debug_assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
///         Self::from_canonical_unchecked(int as u32)
///     }
/// }
///```
///
/// Similarly, `quotient_small_u_int!(Mersenne31, u32, [u8, u16])` produces the above code along with
/// an identical version with `u8` replaced by `u16`.
///
/// All fields will need to use this method twice. Once for unsigned ints and once for signed ints.
#[macro_export]
macro_rules! quotient_map_large_int {
    ($field:ty, $max_size:ty, $order:expr, [$($prim_int:ty),*] ) => {
        $(
        impl QuotientMap<$prim_int> for $field {
            /// Convert a given `$prim_int` integer into an element of the `$field` field.
            ///
            /// Due to the integer type, the input value is always canonical.
            #[inline]
            fn from_int(int: $prim_int) -> $field {
                // Should be removed by the compiler.
                assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
                unsafe {
                    Self::from_canonical_unchecked(int as $max_size)
                }
            }

            /// Due to the integer type, the input value is always canonical.
            #[inline]
            fn from_canonical_checked(int: $prim_int) -> Option<$field> {
                // Should be removed by the compiler.
                assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
                Some(unsafe {
                    Self::from_canonical_unchecked(int as $max_size)
                })
            }

            /// Due to the integer type, the input value is always canonical.
            #[inline]
            unsafe fn from_canonical_unchecked(int: $prim_int) -> $field {
                // We use debug_assert to ensure this is removed by the compiler in release mode.
                debug_assert!(size_of::<$prim_int>() <= size_of::<$max_size>());
                Self::from_canonical_unchecked(int as $max_size)
            }
        }
        )*
    };
}

pub use quotient_map_small_int;
