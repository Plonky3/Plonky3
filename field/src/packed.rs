use core::mem::MaybeUninit;
use core::ops::Div;
use core::{array, slice};

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

    /// Compute a linear combination of a slice of base field elements and
    /// a slice of packed field elements. The slices must have equal length
    /// and it must be a compile time constant.
    /// 
    /// # Panics
    ///
    /// May panic if the length of either slice is not equal to `N`.
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
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self);
}

/// Fix a field `F` a packing width `W` and an extension field `EF` of `F`.
///
/// By choosing a basis `B`, `EF` can be transformed into an array `[F; D]`.
///
/// A type should implement PackedFieldExtension if it can be transformed into `[F; D]` ~ `[[F; W]; D]`
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

#[cfg(test)]
mod tests {
    use super::*;
    use core::fmt::Debug;
    use core::mem::align_of;

    #[test]
    fn test_pack_slice() {
        // Test with a simple array that implements PackedValue
        struct TestPackable(u32);

        impl Packable for TestPackable {}

        impl Default for TestPackable {
            fn default() -> Self {
                Self(0)
            }
        }

        impl Copy for TestPackable {}

        impl Clone for TestPackable {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl PartialEq for TestPackable {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl Eq for TestPackable {}

        // Create an array type as a PackedValue
        unsafe impl PackedValue for [TestPackable; 4] {
            type Value = TestPackable;
            const WIDTH: usize = 4;

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

        // Create a test buffer with 8 elements (2 packs of 4)
        let mut buffer = [
            TestPackable(1),
            TestPackable(2),
            TestPackable(3),
            TestPackable(4),
            TestPackable(5),
            TestPackable(6),
            TestPackable(7),
            TestPackable(8),
        ];

        // Test pack_slice
        let packed = <[TestPackable; 4]>::pack_slice(&buffer);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0][0].0, 1);
        assert_eq!(packed[0][3].0, 4);
        assert_eq!(packed[1][0].0, 5);
        assert_eq!(packed[1][3].0, 8);

        // Test pack_slice_with_suffix with even length
        let (packed, suffix) = <[TestPackable; 4]>::pack_slice_with_suffix(&buffer);
        assert_eq!(packed.len(), 2);
        assert_eq!(suffix.len(), 0);

        // Test with a non-multiple of WIDTH
        let buffer_odd = [
            TestPackable(1),
            TestPackable(2),
            TestPackable(3),
            TestPackable(4),
            TestPackable(5),
            TestPackable(6),
        ];

        // Test pack_slice_with_suffix with odd length
        let (packed, suffix) = <[TestPackable; 4]>::pack_slice_with_suffix(&buffer_odd);
        assert_eq!(packed.len(), 1);
        assert_eq!(suffix.len(), 2);
        assert_eq!(suffix[0].0, 5);
        assert_eq!(suffix[1].0, 6);

        // Test unpack_slice
        let unpacked = <[TestPackable; 4]>::unpack_slice(packed);
        assert_eq!(unpacked.len(), 4);
        assert_eq!(unpacked[0].0, 1);
        assert_eq!(unpacked[3].0, 4);

        // Test mutable functions
        let packed_mut = <[TestPackable; 4]>::pack_slice_mut(&mut buffer);
        packed_mut[0][2] = TestPackable(30);
        assert_eq!(buffer[2].0, 30);

        // Test pack_slice_with_suffix_mut
        let (packed_mut, suffix_mut) = <[TestPackable; 4]>::pack_slice_with_suffix_mut(&mut buffer_odd);
        packed_mut[0][1] = TestPackable(20);
        suffix_mut[0] = TestPackable(50);
        assert_eq!(buffer_odd[1].0, 20);
        assert_eq!(buffer_odd[4].0, 50);
    }

    #[test]
    fn test_packed_linear_combination() {
        // Use a simple mock implementation of PackedField for testing
        struct MockPacked([u8; 4]);

        impl Packable for u8 {}

        unsafe impl PackedValue for MockPacked {
            type Value = u8;
            const WIDTH: usize = 4;

            fn from_slice(slice: &[Self::Value]) -> &Self {
                unsafe { &*(slice.as_ptr() as *const Self) }
            }

            fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
                unsafe { &mut *(slice.as_mut_ptr() as *mut Self) }
            }

            fn from_fn<F>(mut f: F) -> Self
            where
                F: FnMut(usize) -> Self::Value,
            {
                MockPacked([f(0), f(1), f(2), f(3)])
            }

            fn as_slice(&self) -> &[Self::Value] {
                &self.0
            }

            fn as_slice_mut(&mut self) -> &mut [Self::Value] {
                &mut self.0
            }
        }

        impl core::ops::Mul<u8> for MockPacked {
            type Output = Self;

            fn mul(self, rhs: u8) -> Self::Output {
                MockPacked([
                    self.0[0] * rhs,
                    self.0[1] * rhs,
                    self.0[2] * rhs,
                    self.0[3] * rhs,
                ])
            }
        }

        // Minimal Field implementation for testing
        impl PrimeCharacteristicRing for u8 {
            const CHARACTERISTIC: u64 = 0;
        }

        impl Field for u8 {
            fn inverse(&self) -> Self {
                unimplemented!()
            }

            fn inverse_or_zero(&self) -> Self {
                unimplemented!()
            }

            fn exp_u64(&self, _exp: u64) -> Self {
                unimplemented!()
            }

            fn constants() -> &'static Self::Constants {
                unimplemented!()
            }

            type Constants = ();
        }

        // Minimal implementation of required traits for MockPacked
        impl Algebra<u8> for MockPacked {
            fn zero() -> Self {
                MockPacked([0, 0, 0, 0])
            }

            fn one() -> Self {
                MockPacked([1, 1, 1, 1])
            }

            fn add(&mut self, rhs: &Self) {
                for i in 0..Self::WIDTH {
                    self.0[i] += rhs.0[i];
                }
            }

            fn sub(&mut self, rhs: &Self) {
                for i in 0..Self::WIDTH {
                    self.0[i] -= rhs.0[i];
                }
            }

            fn mul(&mut self, rhs: &Self) {
                for i in 0..Self::WIDTH {
                    self.0[i] *= rhs.0[i];
                }
            }
        }

        // Make sure it implements Div<u8>
        impl core::ops::Div<u8> for MockPacked {
            type Output = Self;

            fn div(self, rhs: u8) -> Self::Output {
                MockPacked([
                    self.0[0] / rhs,
                    self.0[1] / rhs,
                    self.0[2] / rhs,
                    self.0[3] / rhs,
                ])
            }
        }

        // Implement required traits to make MockPacked a PackedField
        unsafe impl PackedField for MockPacked {
            type Scalar = u8;
        }

        // Helper to simplify array work
        impl MockPacked {
            fn sum_array<const N: usize>(elements: &[Self; N]) -> Self {
                let mut result = Self::zero();
                for element in elements {
                    result.add(element);
                }
                result
            }
        }

        // Create test vectors
        let vecs = [
            MockPacked([1, 2, 3, 4]),
            MockPacked([5, 6, 7, 8]),
            MockPacked([9, 10, 11, 12]),
        ];

        // Create coefficients
        let coeffs = [2u8, 3u8, 4u8];

        // Calculate the linear combination
        let result = MockPacked::packed_linear_combination::<3>(&coeffs, &vecs);

        // Calculate expected result manually
        let expected = MockPacked([
            1 * 2 + 5 * 3 + 9 * 4,
            2 * 2 + 6 * 3 + 10 * 4,
            3 * 2 + 7 * 3 + 11 * 4,
            4 * 2 + 8 * 3 + 12 * 4,
        ]);

        // Compare results
        assert_eq!(result.0, expected.0);

        // Test with different length slices
        let shorter_coeffs = [2u8, 3u8];
        let shorter_vecs = [MockPacked([1, 2, 3, 4]), MockPacked([5, 6, 7, 8])];
        
        let result2 = MockPacked::packed_linear_combination::<2>(&shorter_coeffs, &shorter_vecs);
        
        // Calculate expected result manually
        let expected2 = MockPacked([
            1 * 2 + 5 * 3,
            2 * 2 + 6 * 3,
            3 * 2 + 7 * 3,
            4 * 2 + 8 * 3,
        ]);
        
        assert_eq!(result2.0, expected2.0);
    }

    #[test]
    fn test_interleave() {
        // Define a simple implementation for testing interleave
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct MockPacked([u8; 4]);

        impl Packable for u8 {}

        unsafe impl PackedValue for MockPacked {
            type Value = u8;
            const WIDTH: usize = 4;

            fn from_slice(slice: &[Self::Value]) -> &Self {
                unsafe { &*(slice.as_ptr() as *const Self) }
            }

            fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
                unsafe { &mut *(slice.as_mut_ptr() as *mut Self) }
            }

            fn from_fn<F>(mut f: F) -> Self
            where
                F: FnMut(usize) -> Self::Value,
            {
                MockPacked([f(0), f(1), f(2), f(3)])
            }

            fn as_slice(&self) -> &[Self::Value] {
                &self.0
            }

            fn as_slice_mut(&mut self) -> &mut [Self::Value] {
                &mut self.0
            }
        }

        // Minimal implementations of required traits
        impl PrimeCharacteristicRing for u8 {
            const CHARACTERISTIC: u64 = 0;
        }

        impl Field for u8 {
            fn inverse(&self) -> Self { unimplemented!() }
            fn inverse_or_zero(&self) -> Self { unimplemented!() }
            fn exp_u64(&self, _exp: u64) -> Self { unimplemented!() }
            fn constants() -> &'static Self::Constants { unimplemented!() }
            type Constants = ();
        }

        impl Algebra<u8> for MockPacked {
            fn zero() -> Self { MockPacked([0, 0, 0, 0]) }
            fn one() -> Self { MockPacked([1, 1, 1, 1]) }
            fn add(&mut self, _rhs: &Self) { unimplemented!() }
            fn sub(&mut self, _rhs: &Self) { unimplemented!() }
            fn mul(&mut self, _rhs: &Self) { unimplemented!() }
        }

        impl core::ops::Div<u8> for MockPacked {
            type Output = Self;
            fn div(self, _rhs: u8) -> Self::Output { unimplemented!() }
        }

        unsafe impl PackedField for MockPacked {
            type Scalar = u8;
        }

        unsafe impl PackedFieldPow2 for MockPacked {
            fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
                assert!(block_len > 0 && block_len <= Self::WIDTH);
                assert!(Self::WIDTH % block_len == 0);
                
                // For WIDTH = 4, we have a few possible block_len values
                match block_len {
                    // block_len = 1: Interleave individual elements
                    1 => {
                        let result1 = MockPacked([self.0[0], other.0[0], self.0[2], other.0[2]]);
                        let result2 = MockPacked([self.0[1], other.0[1], self.0[3], other.0[3]]);
                        (result1, result2)
                    },
                    // block_len = 2: Interleave pairs of elements
                    2 => {
                        let result1 = MockPacked([self.0[0], self.0[1], other.0[0], other.0[1]]);
                        let result2 = MockPacked([self.0[2], self.0[3], other.0[2], other.0[3]]);
                        (result1, result2)
                    },
                    // block_len = 4: No interleaving (identity operation)
                    4 => (*self, other),
                    _ => panic!("Unsupported block_len for WIDTH = 4"),
                }
            }
        }

        // Test case 1: block_len = 1 (interleave individual elements)
        let a = MockPacked([1, 2, 3, 4]);
        let b = MockPacked([5, 6, 7, 8]);
        
        let (c, d) = a.interleave(b, 1);
        
        assert_eq!(c, MockPacked([1, 5, 3, 7]));
        assert_eq!(d, MockPacked([2, 6, 4, 8]));
        
        // Test case 2: block_len = 2 (interleave pairs)
        let (e, f) = a.interleave(b, 2);
        
        assert_eq!(e, MockPacked([1, 2, 5, 6]));
        assert_eq!(f, MockPacked([3, 4, 7, 8]));
        
        // Test case 3: block_len = 4 (no interleaving)
        let (g, h) = a.interleave(b, 4);
        
        assert_eq!(g, a);
        assert_eq!(h, b);
        
        // Test that it panics with invalid block_len
        let result = std::panic::catch_unwind(|| {
            a.interleave(b, 3); // 3 doesn't divide 4
        });
        assert!(result.is_err());
        
        let result = std::panic::catch_unwind(|| {
            a.interleave(b, 0); // 0 is invalid
        });
        assert!(result.is_err());
        
        let result = std::panic::catch_unwind(|| {
            a.interleave(b, 8); // 8 exceeds WIDTH
        });
        assert!(result.is_err());
    }
}
