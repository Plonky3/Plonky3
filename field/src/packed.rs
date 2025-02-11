use core::mem::MaybeUninit;
use core::ops::Div;
use core::slice;

use crate::field::Field;
use crate::{Algebra, BasedVectorSpace, ExtensionField, Powers, PrimeCharacteristicRing};

/// A trait to constrain types that can be packed into a packed value.
///
/// The `Packable` trait allows us to specify implementations for potentially conflicting types.
pub trait Packable: 'static + Default + Copy + Send + Sync + PartialEq + Eq {}

/// # Safety
/// - If `P` implements `PackedField` then `P` must be castable to/from `[P::Value; P::WIDTH]`
///   without UB.
pub unsafe trait PackedValue: 'static + Copy + Send + Sync {
    type Value: Packable;

    const WIDTH: usize;

    fn from_slice(slice: &[Self::Value]) -> &Self;
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self;

    /// Similar to `core:array::from_fn`.
    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value;

    fn as_slice(&self) -> &[Self::Value];
    fn as_slice_mut(&mut self) -> &mut [Self::Value];

    fn pack_slice(buf: &[Self::Value]) -> &[Self] {
        // Sources vary, but this should be true on all platforms we care about.
        // This should be a const assert, but trait methods can't access `Self` in a const context,
        // even with inner struct instantiation. So we will trust LLVM to optimize this out.
        assert!(align_of::<Self>() <= align_of::<Self::Value>());
        assert!(
            buf.len() % Self::WIDTH == 0,
            "Slice length (got {}) must be a multiple of packed field width ({}).",
            buf.len(),
            Self::WIDTH
        );
        let buf_ptr = buf.as_ptr().cast::<Self>();
        let n = buf.len() / Self::WIDTH;
        unsafe { slice::from_raw_parts(buf_ptr, n) }
    }

    fn pack_slice_with_suffix(buf: &[Self::Value]) -> (&[Self], &[Self::Value]) {
        let (packed, suffix) = buf.split_at(buf.len() - buf.len() % Self::WIDTH);
        (Self::pack_slice(packed), suffix)
    }

    fn pack_slice_mut(buf: &mut [Self::Value]) -> &mut [Self] {
        assert!(align_of::<Self>() <= align_of::<Self::Value>());
        assert!(
            buf.len() % Self::WIDTH == 0,
            "Slice length (got {}) must be a multiple of packed field width ({}).",
            buf.len(),
            Self::WIDTH
        );
        let buf_ptr = buf.as_mut_ptr().cast::<Self>();
        let n = buf.len() / Self::WIDTH;
        unsafe { slice::from_raw_parts_mut(buf_ptr, n) }
    }

    fn pack_maybe_uninit_slice_mut(
        buf: &mut [MaybeUninit<Self::Value>],
    ) -> &mut [MaybeUninit<Self>] {
        assert!(align_of::<Self>() <= align_of::<Self::Value>());
        assert!(
            buf.len() % Self::WIDTH == 0,
            "Slice length (got {}) must be a multiple of packed field width ({}).",
            buf.len(),
            Self::WIDTH
        );
        let buf_ptr = buf.as_mut_ptr().cast::<MaybeUninit<Self>>();
        let n = buf.len() / Self::WIDTH;
        unsafe { slice::from_raw_parts_mut(buf_ptr, n) }
    }

    fn pack_slice_with_suffix_mut(buf: &mut [Self::Value]) -> (&mut [Self], &mut [Self::Value]) {
        let (packed, suffix) = buf.split_at_mut(buf.len() - buf.len() % Self::WIDTH);
        (Self::pack_slice_mut(packed), suffix)
    }

    fn pack_maybe_uninit_slice_with_suffix_mut(
        buf: &mut [MaybeUninit<Self::Value>],
    ) -> (&mut [MaybeUninit<Self>], &mut [MaybeUninit<Self::Value>]) {
        let (packed, suffix) = buf.split_at_mut(buf.len() - buf.len() % Self::WIDTH);
        (Self::pack_maybe_uninit_slice_mut(packed), suffix)
    }

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

    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        slice.try_into().unwrap()
    }

    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        slice.try_into().unwrap()
    }

    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        core::array::from_fn(f)
    }

    fn as_slice(&self) -> &[Self::Value] {
        self
    }

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

    /// Compute the dot product of two iterators of packed elements.
    fn iter_dot_product<'a, Inner1: Into<Self> + 'a + Copy, Inner2: Into<Self> + 'a + Copy, I: Iterator<Item = &'a Inner1>, J: Iterator<Item = &'a Inner2>>(
        iter_1: I,
        iter_2: J,
    ) -> Self {
        let mut acc = Self::ZERO;
        iter_1
            .zip(iter_2)
            .for_each(|(&elem_1, &elem_2)| acc += elem_1.into() * elem_2.into());
        acc
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
    /// When `block_len = WIDTH`, this operation is a no-op. `block_len` must divide `WIDTH`. Since
    /// `WIDTH` is specified to be a power of 2, `block_len` must also be a power of 2. It cannot be
    /// 0 and it cannot exceed `WIDTH`.
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
    fn from_ext_slice(ext_slice: &[ExtField]) -> Self;

    /// Similar to packed_powers, construct an iterator which returns
    /// powers of `base` packed into `PackedFieldExtension` elements.
    fn packed_ext_powers(base: ExtField) -> Powers<Self>;

    /// Compute the dot product of an iterator of packed extension field elements
    /// and an iterator of packed base field elements.
    fn base_dot_product<I: Iterator<Item = Self>, J: Iterator<Item = BaseField::Packing>>(
        ext_iter: I,
        base_iter: J,
    ) -> Self {
        let mut acc = Self::ZERO;
        ext_iter
            .zip(base_iter)
            .for_each(|(ext, base)| acc += ext * base);
        acc
    }
}

unsafe impl<T: Packable> PackedValue for T {
    type Value = Self;

    const WIDTH: usize = 1;

    fn from_slice(slice: &[Self::Value]) -> &Self {
        &slice[0]
    }

    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        &mut slice[0]
    }

    fn from_fn<Fn>(mut f: Fn) -> Self
    where
        Fn: FnMut(usize) -> Self::Value,
    {
        f(0)
    }

    fn as_slice(&self) -> &[Self::Value] {
        slice::from_ref(self)
    }

    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        slice::from_mut(self)
    }
}

unsafe impl<F: Field> PackedField for F {
    type Scalar = Self;
}

unsafe impl<F: Field> PackedFieldPow2 for F {
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        match block_len {
            1 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }
}

impl<F: Field> PackedFieldExtension<F, F> for F::Packing {
    fn from_ext_slice(ext_slice: &[F]) -> Self {
        ext_slice[0].into()
    }

    fn packed_ext_powers(base: F) -> Powers<Self> {
        F::Packing::packed_powers(base)
    }
}

impl Packable for u8 {}

impl Packable for u16 {}

impl Packable for u32 {}

impl Packable for u64 {}

impl Packable for u128 {}
