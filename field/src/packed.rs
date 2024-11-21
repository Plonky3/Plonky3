use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};
use core::slice;

use alloc::vec::Vec;

use crate::{CommutativeRing, ExtensionField, Field, FieldAlgebra, Powers};

/// A trait to constrain types that can be packed into a packed value.
///
/// The `Packable` trait allows us to specify implementations for potentially conflicting types.
pub trait Packable: 'static + Default + Copy + Send + Sync + PartialEq + Eq {}

/// # Safety
/// - If `P` implements `PackedValue` then `P` must be castable to/from `[P::Value; P::WIDTH]`
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

/// # Safety
/// - See `PackedValue` above.
pub unsafe trait PackedField: FieldAlgebra<Self::Scalar>
    + PackedValue<Value = Self::Scalar>
    + From<Self::Scalar>
    + Add<Self::Scalar, Output = Self>
    + AddAssign<Self::Scalar>
    + Sub<Self::Scalar, Output = Self>
    + SubAssign<Self::Scalar>
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar>
    // TODO: Implement packed / packed division
    + Div<Self::Scalar, Output = Self>
{
    type Scalar: Field;

    /// Construct an iterator which returns powers of `base` packed into packed field elements.
    /// 
    /// E.g. if `Self::WIDTH = 4` returns: `[base^0, base^1, base^2, base^3], [base^4, base^5, base^6, base^7], ...`
    #[must_use]
    fn packed_powers(base: Self::Scalar) -> Powers<Self> {
        Self::packed_shifted_powers(base, Self::Scalar::ONE)
    }

    /// Construct an iterator which returns powers of `base` multiplied by `start` and packed into packed field elements.
    /// 
    /// E.g. if `Self::WIDTH = 4` returns: `[start, start*base, start*base^2, start*base^3], [start*base^4, start*base^5, start*base^6, start*base^7], ...`
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
pub trait PackedFieldExtension:
    FieldAlgebra<Self::BaseField>
    + Add<<Self::BaseField as Field>::Packing, Output = Self>
    + AddAssign<<Self::BaseField as Field>::Packing>
    + Sub<<Self::BaseField as Field>::Packing, Output = Self>
    + SubAssign<<Self::BaseField as Field>::Packing>
    + Mul<<Self::BaseField as Field>::Packing, Output = Self>
    + MulAssign<<Self::BaseField as Field>::Packing>
{
    type BaseField: Field;
    type ExtField: ExtensionField<Self::BaseField, ExtensionPacking = Self>;

    /// Given an extension field `EF` element, decompose it into
    /// a collection of field `F` elements, cast each field element
    /// to the corresponding packed element and then pack the result into an array.
    fn from_ext_element(ext_elem: Self::ExtField) -> Self;

    /// Given a slice of extension field `EF` elements of length `W`,
    /// convert into the array `[[F; D]; W]` transpose to
    /// decompose it into `[[F; W]; D]` and then pack to get `[PF; W]`
    fn from_ext_slice(ext_slice: &[Self::ExtField]) -> Self;

    // TODO: Do we need from iterator/from_fns as well?

    /// Convert self to an array of form `[[F; W]; D]`. Then
    /// transpose this array to get something of form `[[F; D]; W]`
    /// Interpret each row as an extension field element so this is
    /// `[EF; W]`. We return a Vec and not an array due to Const Generics
    /// restrictions.
    fn to_ext_vec(packed_ext_elem: &Self) -> Vec<Self::ExtField>;

    /// Similar to packed_powers, construct an iterator which returns
    /// powers of `base` packed into `PackedFieldExtension` elements.
    fn ext_powers_packed(base: Self::ExtField) -> Powers<Self>;
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

impl Packable for u8 {}

impl Packable for u16 {}

impl Packable for u32 {}

impl Packable for u64 {}

impl Packable for u128 {}
