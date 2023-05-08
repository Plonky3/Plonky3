use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};
use core::slice;

use crate::field::{AbstractField, Field};

/// # Safety
/// - `WIDTH` is assumed to be a power of 2.
/// - If `P` implements `PackedField` then `P` must be castable to/from `[P::Scalar; P::WIDTH]`
///   without UB.
pub unsafe trait PackedField:
 AbstractField
    + 'static
    + Copy
    + Default
    + Add<Self::Scalar, Output = Self>
    + AddAssign<Self::Scalar>
    + Sub<Self::Scalar, Output = Self>
    + SubAssign<Self::Scalar>
    + From<Self::Scalar>
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar>
    // TODO: Implement packed / packed division
    + Div<Self::Scalar, Output = Self>
    + Send
    + Sync
    where
        Self::Scalar: Add<Self, Output = Self>,
        Self::Scalar: Mul<Self, Output = Self>,
        Self::Scalar: Sub<Self, Output = Self>,
{
    type Scalar: Field;

    const WIDTH: usize;
    const ZEROS: Self;
    const ONES: Self;

    fn from_slice(slice: &[Self::Scalar]) -> &Self;
    fn from_slice_mut(slice: &mut [Self::Scalar]) -> &mut Self;
    fn as_slice(&self) -> &[Self::Scalar];
    fn as_slice_mut(&mut self) -> &mut [Self::Scalar];

    /// Take interpret two vectors as chunks of block_len elements. Unpack and interleave those
    /// chunks. This is best seen with an example. If we have:
    ///     A = [x0, y0, x1, y1],
    ///     B = [x2, y2, x3, y3],
    /// then
    ///     interleave(A, B, 1) = ([x0, x2, x1, x3], [y0, y2, y1, y3]).
    /// Pairs that were adjacent in the input are at corresponding positions in the output.
    ///   r lets us set the size of chunks we're interleaving. If we set block_len = 2, then for
    ///     A = [x0, x1, y0, y1],
    ///     B = [x2, x3, y2, y3],
    /// we obtain
    ///     interleave(A, B, block_len) = ([x0, x1, x2, x3], [y0, y1, y2, y3]).
    ///   We can also think about this as stacking the vectors, dividing them into 2x2 matrices, and
    /// transposing those matrices.
    ///   When block_len = WIDTH, this operation is a no-op. block_len must divide WIDTH. Since
    /// WIDTH is specified to be a power of 2, block_len must also be a power of 2. It cannot be 0
    /// and it cannot be > WIDTH.
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self);

    fn pack_slice(buf: &[Self::Scalar]) -> &[Self] {
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
    fn pack_slice_mut(buf: &mut [Self::Scalar]) -> &mut [Self] {
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

    fn doubles(&self) -> Self {
        *self * Self::Scalar::TWO
    }
}

unsafe impl<F: Field> PackedField for F {
    type Scalar = Self;

    const WIDTH: usize = 1;
    const ZEROS: Self = F::ZERO;
    const ONES: Self = F::ONE;

    fn from_slice(slice: &[Self::Scalar]) -> &Self {
        &slice[0]
    }
    fn from_slice_mut(slice: &mut [Self::Scalar]) -> &mut Self {
        &mut slice[0]
    }
    fn as_slice(&self) -> &[Self::Scalar] {
        slice::from_ref(self)
    }
    fn as_slice_mut(&mut self) -> &mut [Self::Scalar] {
        slice::from_mut(self)
    }

    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        match block_len {
            1 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }
}
