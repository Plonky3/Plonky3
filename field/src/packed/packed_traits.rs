use alloc::vec::Vec;
use core::iter::{Product, Sum};
use core::mem::MaybeUninit;
use core::ops::{Div, DivAssign};
use core::{array, slice};

use crate::field::Field;
use crate::{Algebra, BasedVectorSpace, ExtensionField, Powers, PrimeCharacteristicRing};

/// A trait to constrain types that can be packed into a packed value.
///
/// The `Packable` trait allows us to specify implementations for potentially conflicting types.
pub trait Packable: 'static + Default + Copy + Send + Sync + PartialEq + Eq {}

/// A trait for array-like structs made up of multiple scalar elements.
///
/// # Safety
/// - If `P` implements `PackedField` then `P` must be castable to/from `[P::Value; P::WIDTH]`
///   without UB.
pub unsafe trait PackedValue: 'static + Copy + Send + Sync {
    /// The scalar type that is packed into this value.
    type Value: Packable;

    /// Number of scalar values packed together.
    const WIDTH: usize;

    /// Interprets a slice of scalar values as a packed value reference.
    ///
    /// # Panics:
    /// This function will panic if `slice.len() != Self::WIDTH`
    #[must_use]
    fn from_slice(slice: &[Self::Value]) -> &Self;

    /// Interprets a mutable slice of scalar values as a mutable packed value.
    ///
    /// # Panics:
    /// This function will panic if `slice.len() != Self::WIDTH`
    #[must_use]
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self;

    /// Constructs a packed value using a function to generate each element.
    ///
    /// Similar to `core:array::from_fn`.
    #[must_use]
    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value;

    /// Returns the underlying scalar values as an immutable slice.
    #[must_use]
    fn as_slice(&self) -> &[Self::Value];

    /// Returns the underlying scalar values as a mutable slice.
    #[must_use]
    fn as_slice_mut(&mut self) -> &mut [Self::Value];

    /// Packs a slice of scalar values into a slice of packed values.
    ///
    /// # Panics
    /// Panics if the slice length is not divisible by `WIDTH`.
    #[inline]
    #[must_use]
    fn pack_slice(buf: &[Self::Value]) -> &[Self] {
        // Sources vary, but this should be true on all platforms we care about.
        // This should be a const assert, but trait methods can't access `Self` in a const context,
        // even with inner struct instantiation. So we will trust LLVM to optimize this out.
        assert!(align_of::<Self>() <= align_of::<Self::Value>());
        assert!(
            buf.len().is_multiple_of(Self::WIDTH),
            "Slice length (got {}) must be a multiple of packed field width ({}).",
            buf.len(),
            Self::WIDTH
        );
        let buf_ptr = buf.as_ptr().cast::<Self>();
        let n = buf.len() / Self::WIDTH;
        unsafe { slice::from_raw_parts(buf_ptr, n) }
    }

    /// Packs a slice into packed values and returns the packed portion and any remaining suffix.
    #[inline]
    #[must_use]
    fn pack_slice_with_suffix(buf: &[Self::Value]) -> (&[Self], &[Self::Value]) {
        let (packed, suffix) = buf.split_at(buf.len() - buf.len() % Self::WIDTH);
        (Self::pack_slice(packed), suffix)
    }

    /// Converts a mutable slice of scalar values into a mutable slice of packed values.
    ///
    /// # Panics
    /// Panics if the slice length is not divisible by `WIDTH`.
    #[inline]
    #[must_use]
    fn pack_slice_mut(buf: &mut [Self::Value]) -> &mut [Self] {
        assert!(align_of::<Self>() <= align_of::<Self::Value>());
        assert!(
            buf.len().is_multiple_of(Self::WIDTH),
            "Slice length (got {}) must be a multiple of packed field width ({}).",
            buf.len(),
            Self::WIDTH
        );
        let buf_ptr = buf.as_mut_ptr().cast::<Self>();
        let n = buf.len() / Self::WIDTH;
        unsafe { slice::from_raw_parts_mut(buf_ptr, n) }
    }

    /// Converts a mutable slice of possibly uninitialized scalar values into
    /// a mutable slice of possibly uninitialized packed values.
    ///
    /// # Panics
    /// Panics if the slice length is not divisible by `WIDTH`.
    #[inline]
    #[must_use]
    fn pack_maybe_uninit_slice_mut(
        buf: &mut [MaybeUninit<Self::Value>],
    ) -> &mut [MaybeUninit<Self>] {
        assert!(align_of::<Self>() <= align_of::<Self::Value>());
        assert!(
            buf.len().is_multiple_of(Self::WIDTH),
            "Slice length (got {}) must be a multiple of packed field width ({}).",
            buf.len(),
            Self::WIDTH
        );
        let buf_ptr = buf.as_mut_ptr().cast::<MaybeUninit<Self>>();
        let n = buf.len() / Self::WIDTH;
        unsafe { slice::from_raw_parts_mut(buf_ptr, n) }
    }

    /// Converts a mutable slice of scalar values into a pair:
    /// - a slice of packed values covering the largest aligned portion,
    /// - and a remainder slice of scalar values that couldn't be packed.
    #[inline]
    #[must_use]
    fn pack_slice_with_suffix_mut(buf: &mut [Self::Value]) -> (&mut [Self], &mut [Self::Value]) {
        let (packed, suffix) = buf.split_at_mut(buf.len() - buf.len() % Self::WIDTH);
        (Self::pack_slice_mut(packed), suffix)
    }

    /// Converts a mutable slice of possibly uninitialized scalar values into a pair:
    /// - a slice of possibly uninitialized packed values covering the largest aligned portion,
    /// - and a remainder slice of possibly uninitialized scalar values that couldn't be packed.
    #[inline]
    #[must_use]
    fn pack_maybe_uninit_slice_with_suffix_mut(
        buf: &mut [MaybeUninit<Self::Value>],
    ) -> (&mut [MaybeUninit<Self>], &mut [MaybeUninit<Self::Value>]) {
        let (packed, suffix) = buf.split_at_mut(buf.len() - buf.len() % Self::WIDTH);
        (Self::pack_maybe_uninit_slice_mut(packed), suffix)
    }

    /// Reinterprets a slice of packed values as a flat slice of scalar values.
    ///
    /// Each packed value contains `Self::WIDTH` scalar values, which are laid out
    /// contiguously in memory. This function allows direct access to those scalars.
    #[inline]
    #[must_use]
    fn unpack_slice(buf: &[Self]) -> &[Self::Value] {
        assert!(align_of::<Self>() >= align_of::<Self::Value>());
        let buf_ptr = buf.as_ptr().cast::<Self::Value>();
        let n = buf.len() * Self::WIDTH;
        unsafe { slice::from_raw_parts(buf_ptr, n) }
    }
}

unsafe impl<T: Packable, const WIDTH: usize> PackedValue for [T; WIDTH] {
    type Value = T;
    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &*slice.as_ptr().cast() }
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }

    #[inline]
    fn from_fn<Fn>(f: Fn) -> Self
    where
        Fn: FnMut(usize) -> Self::Value,
    {
        core::array::from_fn(f)
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Value] {
        self
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        self
    }
}

/// An array of field elements which can be packed into a vector for SIMD operations.
///
/// # Safety
/// - See `PackedValue` above.
pub unsafe trait PackedField: Algebra<Self::Scalar>
    + PackedValue<Value = Self::Scalar>
    // TODO: Implement packed / packed division
    + Div<Self::Scalar, Output = Self>
    + DivAssign<Self::Scalar>
    + Sum<Self::Scalar>
    + Product<Self::Scalar>
{
    type Scalar: Field;

    /// Construct an iterator which returns powers of `base` packed into packed field elements.
    ///
    /// E.g. if `Self::WIDTH = 4`, returns: `[base^0, base^1, base^2, base^3], [base^4, base^5, base^6, base^7], ...`.
    #[must_use]
    fn packed_powers(base: Self::Scalar) -> Powers<Self> {
        Self::packed_shifted_powers(base, Self::Scalar::ONE)
    }

    /// Construct an iterator which returns powers of `base` multiplied by `start` and packed into packed field elements.
    ///
    /// E.g. if `Self::WIDTH = 4`, returns: `[start, start*base, start*base^2, start*base^3], [start*base^4, start*base^5, start*base^6, start*base^7], ...`.
    #[must_use]
    fn packed_shifted_powers(base: Self::Scalar, start: Self::Scalar) -> Powers<Self> {
        let mut current: Self = start.into();
        let slice = current.as_slice_mut();
        for i in 1..Self::WIDTH {
            slice[i] = slice[i - 1] * base;
        }

        Powers {
            base: base.exp_u64(Self::WIDTH as u64).into(),
            current,
        }
    }

    /// Compute a linear combination of a slice of base field elements and
    /// a slice of packed field elements. The slices must have equal length
    /// and it must be a compile time constant.
    ///
    /// # Panics
    ///
    /// May panic if the length of either slice is not equal to `N`.
    #[must_use]
    fn packed_linear_combination<const N: usize>(coeffs: &[Self::Scalar], vecs: &[Self]) -> Self {
        assert_eq!(coeffs.len(), N);
        assert_eq!(vecs.len(), N);
        let combined: [Self; N] = array::from_fn(|i| vecs[i] * coeffs[i]);
        Self::sum_array::<N>(&combined)
    }
}

/// # Safety
/// - `WIDTH` is assumed to be a power of 2.
pub unsafe trait PackedFieldPow2: PackedField {
    /// Take interpret two vectors as chunks of `block_len` elements. Unpack and interleave those
    /// chunks. This is best seen with an example. If we have:
    /// ```text
    /// A = [x0, y0, x1, y1]
    /// B = [x2, y2, x3, y3]
    /// ```
    ///
    /// then
    ///
    /// ```text
    /// interleave(A, B, 1) = ([x0, x2, x1, x3], [y0, y2, y1, y3])
    /// ```
    ///
    /// Pairs that were adjacent in the input are at corresponding positions in the output.
    ///
    /// `r` lets us set the size of chunks we're interleaving. If we set `block_len = 2`, then for
    ///
    /// ```text
    /// A = [x0, x1, y0, y1]
    /// B = [x2, x3, y2, y3]
    /// ```
    ///
    /// we obtain
    ///
    /// ```text
    /// interleave(A, B, block_len) = ([x0, x1, x2, x3], [y0, y1, y2, y3])
    /// ```
    ///
    /// We can also think about this as stacking the vectors, dividing them into 2x2 matrices, and
    /// transposing those matrices.
    ///
    /// When `block_len = WIDTH`, this operation is a no-op.
    ///
    /// # Panics
    /// This may panic if `block_len` does not divide `WIDTH`. Since `WIDTH` is specified to be a power of 2,
    /// `block_len` must also be a power of 2. It cannot be 0 and it cannot exceed `WIDTH`.
    #[must_use]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self);
}

/// Fix a field `F` a packing width `W` and an extension field `EF` of `F`.
///
/// By choosing a basis `B`, `EF` can be transformed into an array `[F; D]`.
///
/// A type should implement PackedFieldExtension if it can be transformed into `[F::Packing; D] ~ [[F; W]; D]`
///
/// This is interpreted by taking a transpose to get `[[F; D]; W]` which can then be reinterpreted
/// as `[EF; W]` by making use of the chosen basis `B` again.
pub trait PackedFieldExtension<
    BaseField: Field,
    ExtField: ExtensionField<BaseField, ExtensionPacking = Self>,
>: Algebra<ExtField> + Algebra<BaseField::Packing> + BasedVectorSpace<BaseField::Packing>
{
    /// Given a slice of extension field `EF` elements of length `W`,
    /// convert into the array `[[F; D]; W]` transpose to
    /// `[[F; W]; D]` and then pack to get `[PF; D]`.
    #[must_use]
    fn from_ext_slice(ext_slice: &[ExtField]) -> Self;

    /// Given a iterator of packed extension field elements, convert to an iterator of
    /// extension field elements.
    ///
    /// This performs the inverse transformation to `from_ext_slice`.
    #[inline]
    #[must_use]
    fn to_ext_iter(iter: impl IntoIterator<Item = Self>) -> impl Iterator<Item = ExtField> {
        iter.into_iter().flat_map(|x| {
            let packed_coeffs = x.as_basis_coefficients_slice();
            (0..BaseField::Packing::WIDTH)
                .map(|i| ExtField::from_basis_coefficients_fn(|j| packed_coeffs[j].as_slice()[i]))
                .collect::<Vec<_>>() // PackedFieldExtension's should reimplement this to avoid this allocation.
        })
    }

    /// Similar to `packed_powers`, construct an iterator which returns
    /// powers of `base` packed into `PackedFieldExtension` elements.
    #[must_use]
    fn packed_ext_powers(base: ExtField) -> Powers<Self>;

    /// Similar to `packed_ext_powers` but only returns `unpacked_len` powers of `base`.
    ///
    /// Note that the length of the returned iterator will be `unpacked_len / WIDTH` and
    /// not `len` as the iterator is over packed extension field elements. If `unpacked_len`
    /// is not divisible by `WIDTH`, `unpacked_len` will be rounded up to the next multiple of `WIDTH`.
    #[must_use]
    fn packed_ext_powers_capped(base: ExtField, unpacked_len: usize) -> impl Iterator<Item = Self> {
        Self::packed_ext_powers(base).take(unpacked_len.div_ceil(BaseField::Packing::WIDTH))
    }
}

unsafe impl<T: Packable> PackedValue for T {
    type Value = Self;

    const WIDTH: usize = 1;

    #[inline]
    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        &slice[0]
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        &mut slice[0]
    }

    #[inline]
    fn from_fn<Fn>(mut f: Fn) -> Self
    where
        Fn: FnMut(usize) -> Self::Value,
    {
        f(0)
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Value] {
        slice::from_ref(self)
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        slice::from_mut(self)
    }
}

unsafe impl<F: Field> PackedField for F {
    type Scalar = Self;
}

unsafe impl<F: Field> PackedFieldPow2 for F {
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        match block_len {
            1 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }
}

impl<F: Field> PackedFieldExtension<F, F> for F::Packing {
    #[inline]
    fn from_ext_slice(ext_slice: &[F]) -> Self {
        *F::Packing::from_slice(ext_slice)
    }

    #[inline]
    fn packed_ext_powers(base: F) -> Powers<Self> {
        F::Packing::packed_powers(base)
    }
}

impl Packable for u8 {}

impl Packable for u16 {}

impl Packable for u32 {}

impl Packable for u64 {}

impl Packable for u128 {}
