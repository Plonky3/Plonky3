//! Byte-wise `F_2`-linear maps and AES-field products, one register at a time.
//!
//! One instruction multiplies bytes in the AES field.
//!
//! Another applies an arbitrary `F_2`-linear map to each byte, and a third inverts first.
//!
//! Every such map is a matrix over `F_2`.
//!
//! Basis change, Frobenius and any tabulated map therefore share one kernel.

use super::{invert_byte, mul_bytes};

/// The quadword whose byte `k` is `1 << (7 - k)`.
///
/// Read as a matrix it is the identity.
///
/// Used as a multiplier it collects the low bit of every byte into one byte.
const BIT_LADDER: u64 = 0x0102_0408_1020_4080;

/// The same byte in all eight positions of a quadword.
const SPREAD: u64 = 0x0101_0101_0101_0101;

/// An `F_2`-linear map on a byte, held the way the affine-byte instruction reads it.
///
/// Output bit `i` is the parity of the input byte against byte `7 - i` of the quadword.
///
/// ```text
///     out[i] = XOR over b of row[7 - i][b] * in[b]
/// ```
///
/// Squaring, scaling by a constant, a basis change and a linearized polynomial all fit here.
///
/// Tabulated once, each becomes a single instruction per register.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
#[must_use]
pub struct ByteMatrix(u64);

impl ByteMatrix {
    /// The map sending every byte to zero.
    pub const ZERO: Self = Self(0);

    /// The map leaving every byte alone.
    pub const IDENTITY: Self = Self(BIT_LADDER);

    /// The map sending basis vector `1 << b` to entry `b` of the argument.
    ///
    /// A linear map is pinned down by where it sends each of the eight basis vectors.
    pub const fn from_images(images: [u8; 8]) -> Self {
        let mut quadword = 0u64;
        let mut i = 0;
        while i < 8 {
            // Row `7 - i` pairs bit `b` of the input with bit `i` of that input's image.
            let mut row = 0u64;
            let mut b = 0;
            while b < 8 {
                if (images[b] >> i) & 1 == 1 {
                    row |= 1 << b;
                }
                b += 1;
            }
            quadword |= row << (8 * (7 - i));
            i += 1;
        }
        Self(quadword)
    }

    /// Where the map sends each of the eight basis vectors.
    #[must_use]
    pub const fn images(self) -> [u8; 8] {
        let mut images = [0u8; 8];
        let mut b = 0;
        while b < 8 {
            let mut i = 0;
            while i < 8 {
                if (self.0 >> (8 * (7 - i) + b)) & 1 == 1 {
                    images[b] |= 1 << i;
                }
                i += 1;
            }
            b += 1;
        }
        images
    }

    /// The image of one byte.
    #[must_use]
    pub const fn apply(self, byte: u8) -> u8 {
        // The input byte against all eight rows at once.
        let hit = self.0 & (byte as u64).wrapping_mul(SPREAD);

        // Fold each byte onto its own low bit, which then holds that row's parity.
        let mut parity = hit;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        parity &= SPREAD;

        // The ladder gathers those eight bits into the top byte, with no two sharing a bit.
        //
        // Row `7 - i` drives output bit `i`, so the gathered byte comes out reversed.
        ((parity.wrapping_mul(BIT_LADDER) >> 56) as u8).reverse_bits()
    }

    /// The map applying the argument first and this one second.
    pub const fn compose(self, first: Self) -> Self {
        let images = first.images();
        let mut composed = [0u8; 8];
        let mut b = 0;
        while b < 8 {
            composed[b] = self.apply(images[b]);
            b += 1;
        }
        Self::from_images(composed)
    }

    /// The row layout the affine-byte instruction takes as its matrix operand.
    #[must_use]
    pub const fn to_quadword(self) -> u64 {
        self.0
    }

    /// The map already held in that row layout.
    ///
    /// Every quadword is a matrix, so this cannot fail.
    pub const fn from_quadword(rows: u64) -> Self {
        Self(rows)
    }

    /// Replaces every byte of the slice with its image.
    ///
    /// Inlined so a caller with a fixed-length block sweeps it without a call.
    #[inline]
    pub fn apply_slice(self, bytes: &mut [u8]) {
        map_slice(&Affine(self), bytes);
    }
}

/// A lane-wise map, written once and instantiated at every register width the target has.
///
/// The matrix operand is built once per sweep rather than once per register.
trait ByteMap {
    /// The matrix the map carries, as one quadword.
    fn quadword(&self) -> u64;

    /// The map on one register, against the matrix already in register shape.
    fn wide<L: ByteLanes>(&self, x: L, matrix: L::Matrix) -> L;

    /// The map on one byte.
    fn scalar(&self, x: u8) -> u8;
}

/// One fixed `F_2`-linear map applied to every byte.
struct Affine(ByteMatrix);

impl ByteMap for Affine {
    #[inline(always)]
    fn quadword(&self) -> u64 {
        self.0.to_quadword()
    }

    #[inline(always)]
    fn wide<L: ByteLanes>(&self, x: L, matrix: L::Matrix) -> L {
        x.affine(matrix)
    }

    #[inline(always)]
    fn scalar(&self, x: u8) -> u8 {
        self.0.apply(x)
    }
}

/// Inversion of every byte in the AES field, with zero left alone.
struct Invert;

impl ByteMap for Invert {
    /// The hardware inverse composes with a map, so the identity leaves a bare inverse.
    #[inline(always)]
    fn quadword(&self) -> u64 {
        ByteMatrix::IDENTITY.to_quadword()
    }

    #[inline(always)]
    fn wide<L: ByteLanes>(&self, x: L, matrix: L::Matrix) -> L {
        x.inverse_then_affine(matrix)
    }

    #[inline(always)]
    fn scalar(&self, x: u8) -> u8 {
        invert_byte(x)
    }
}

/// The byte-lane operations the kernels below are written against.
///
/// Every method acts on each byte of the register independently.
///
/// # Safety
///
/// Reading or writing one value must touch exactly its own width in contiguous bytes.
///
/// That is what lets the kernels hand a slice chunk of that length to the two accessors.
unsafe trait ByteLanes: Copy {
    /// Bytes per register.
    const WIDTH: usize;

    /// A byte matrix in the shape one register of this width consumes.
    type Matrix: Copy;

    /// The matrix this width consumes, from its quadword form.
    fn matrix(quadword: u64) -> Self::Matrix;

    /// Reads one register of consecutive bytes.
    ///
    /// # Safety
    ///
    /// The address must be readable for one register of bytes, at any alignment.
    unsafe fn load(from: *const u8) -> Self;

    /// Writes one register of consecutive bytes.
    ///
    /// # Safety
    ///
    /// The address must be writable for one register of bytes, at any alignment.
    unsafe fn store(to: *mut u8, value: Self);

    /// The AES-field product of the bytes at each position.
    fn mul(self, other: Self) -> Self;

    /// One `F_2`-linear map applied to every byte.
    fn affine(self, matrix: Self::Matrix) -> Self;

    /// The AES-field inverse of every byte, sending zero to zero, then one map.
    fn inverse_then_affine(self, matrix: Self::Matrix) -> Self;
}

// One byte at a time, where the target has no byte-wise instruction to sweep with.
unsafe impl ByteLanes for u8 {
    const WIDTH: usize = 1;

    type Matrix = u64;

    #[inline(always)]
    fn matrix(quadword: u64) -> Self::Matrix {
        quadword
    }

    #[inline(always)]
    unsafe fn load(from: *const u8) -> Self {
        // SAFETY: the readability of the address is the caller's obligation.
        unsafe { *from }
    }

    #[inline(always)]
    unsafe fn store(to: *mut u8, value: Self) {
        // SAFETY: the writability of the address is the caller's obligation.
        unsafe { *to = value }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        mul_bytes(self, other)
    }

    #[inline(always)]
    fn affine(self, matrix: Self::Matrix) -> Self {
        ByteMatrix(matrix).apply(self)
    }

    #[inline(always)]
    fn inverse_then_affine(self, matrix: Self::Matrix) -> Self {
        let inverse = invert_byte(self);

        // Only the hardware form composes the two, and it does so for free.
        //
        // Here the map is a parity fold per byte, so an identity is worth skipping.
        if matrix == ByteMatrix::IDENTITY.to_quadword() {
            inverse
        } else {
            ByteMatrix(matrix).apply(inverse)
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
mod x86_64;

// Which registers the kernels sweep with, widest first.
#[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
use x86_64::{Narrow, Wide};

/// The widest register this target runs these operations in.
#[cfg(not(all(target_arch = "x86_64", target_feature = "gfni")))]
type Wide = u8;

/// The narrower register this target runs these operations in.
///
/// Both widths collapse onto the byte where neither instruction set is available.
#[cfg(not(all(target_arch = "x86_64", target_feature = "gfni")))]
type Narrow = u8;

/// Maps the longest prefix of whole registers, returning what is left over.
//
// The chunk size comes from a trait constant, which a fixed-size chunking cannot take.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline(always)]
fn map_registers<'a, L: ByteLanes, M: ByteMap>(map: &M, bytes: &'a mut [u8]) -> &'a mut [u8] {
    const {
        assert!(
            size_of::<L>() == L::WIDTH,
            "a register must be its own width in bytes"
        );
    };

    let covered = (bytes.len() / L::WIDTH) * L::WIDTH;
    let (head, tail) = bytes.split_at_mut(covered);

    // Built once here, so the sweep below is one instruction per register.
    let matrix = L::matrix(map.quadword());

    for chunk in head.chunks_exact_mut(L::WIDTH) {
        // SAFETY: the chunk is one register of bytes, by the length just computed.
        // Both accesses are unaligned forms.
        unsafe {
            let mapped = map.wide(L::load(chunk.as_ptr()), matrix);
            L::store(chunk.as_mut_ptr(), mapped);
        }
    }
    tail
}

/// Maps every byte of the slice, widest register first.
#[inline]
fn map_slice<M: ByteMap>(map: &M, bytes: &mut [u8]) {
    let rest = map_registers::<Wide, M>(map, bytes);
    let rest = map_registers::<Narrow, M>(map, rest);
    for byte in rest {
        *byte = map.scalar(*byte);
    }
}

/// Multiplies the longest prefix of whole registers, returning what is left over.
//
// The chunk size comes from a trait constant, which a fixed-size chunking cannot take.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline(always)]
fn mul_registers<'a, 'b, L: ByteLanes>(
    dst: &'a mut [u8],
    src: &'b [u8],
) -> (&'a mut [u8], &'b [u8]) {
    const {
        assert!(
            size_of::<L>() == L::WIDTH,
            "a register must be its own width in bytes"
        );
    };

    let covered = (dst.len() / L::WIDTH) * L::WIDTH;
    let (head, tail) = dst.split_at_mut(covered);
    let (factors, rest) = src.split_at(covered);
    for (chunk, factor) in head
        .chunks_exact_mut(L::WIDTH)
        .zip(factors.chunks_exact(L::WIDTH))
    {
        // SAFETY: both chunks are one register of bytes, by the length just computed.
        // Every access is an unaligned form.
        unsafe {
            let product = L::load(chunk.as_ptr()).mul(L::load(factor.as_ptr()));
            L::store(chunk.as_mut_ptr(), product);
        }
    }
    (tail, rest)
}

/// Replaces every byte of the first slice with its AES-field product against the second.
///
/// # Panics
///
/// Panics if the slice lengths differ.
#[inline]
pub(crate) fn mul_slice(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "elementwise product lengths differ");
    let (dst, src) = mul_registers::<Wide>(dst, src);
    let (dst, src) = mul_registers::<Narrow>(dst, src);
    for (value, factor) in dst.iter_mut().zip(src) {
        *value = mul_bytes(*value, *factor);
    }
}

/// Replaces every byte of the slice with its AES-field inverse, leaving zero alone.
#[inline]
pub(crate) fn invert_slice(bytes: &mut [u8]) {
    map_slice(&Invert, bytes);
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{ByteMatrix, invert_slice, mul_slice};
    use crate::aes::{invert_byte, mul_bytes};

    /// The image of a byte under the map whose columns are given, summed column by column.
    ///
    /// This is the definition of a linear map, written without the row packing under test.
    fn apply_by_columns(images: [u8; 8], byte: u8) -> u8 {
        (0..8).fold(0, |acc, b| {
            acc ^ if (byte >> b) & 1 == 1 { images[b] } else { 0 }
        })
    }

    #[test]
    fn the_row_layout_is_the_one_the_instruction_documents() {
        // Invariant: the identity matrix is the byte ladder the hardware manual spells out.
        //
        // Row `7 - i` must be `1 << i`, which stacks to `0x0102040810204080`.
        assert_eq!(ByteMatrix::IDENTITY.to_quadword(), 0x0102_0408_1020_4080);
        assert_eq!(ByteMatrix::IDENTITY.images(), [1, 2, 4, 8, 16, 32, 64, 128]);

        // A single entry: sending bit 0 to bit 7 and everything else to zero.
        //
        //     column 0 = 0x80  ->  row 0 (byte 7) has bit 0 set
        let one_entry = ByteMatrix::from_images([0x80, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(one_entry.to_quadword(), 1);
        assert_eq!(one_entry.apply(0xff), 0x80);
        assert_eq!(one_entry.apply(0xfe), 0);
    }

    #[test]
    fn the_constant_maps_behave() {
        // Fixture state: the two maps every linear algebra starts from.
        for byte in 0..=u8::MAX {
            assert_eq!(ByteMatrix::ZERO.apply(byte), 0);
            assert_eq!(ByteMatrix::IDENTITY.apply(byte), byte);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// The packed rows must agree with summing the columns of the set input bits.
        #[test]
        fn the_row_packing_agrees_with_the_column_definition(images: [u8; 8], byte: u8) {
            let matrix = ByteMatrix::from_images(images);
            prop_assert_eq!(matrix.apply(byte), apply_by_columns(images, byte));
        }

        /// Reading the columns back out must return the ones that went in.
        #[test]
        fn the_column_round_trip_is_the_identity(images: [u8; 8]) {
            prop_assert_eq!(ByteMatrix::from_images(images).images(), images);
        }

        /// Composition must be the same map as applying one after the other.
        #[test]
        fn composition_applies_the_second_map_to_the_first_image(
            outer: [u8; 8],
            inner: [u8; 8],
            byte: u8,
        ) {
            let (outer, inner) = (ByteMatrix::from_images(outer), ByteMatrix::from_images(inner));
            prop_assert_eq!(outer.compose(inner).apply(byte), outer.apply(inner.apply(byte)));
        }

        /// The kernels must agree with the scalar routines they vectorize, at every length.
        ///
        /// Lengths past one register exercise the wide pass, the narrow pass and the tail.
        #[test]
        fn the_slice_kernels_agree_with_the_scalar_routines(
            values in prop::collection::vec(any::<u8>(), 0..200),
            factors in prop::collection::vec(any::<u8>(), 0..200),
            images: [u8; 8],
        ) {
            let len = values.len().min(factors.len());
            let (values, factors) = (&values[..len], &factors[..len]);

            let mut products = values.to_vec();
            mul_slice(&mut products, factors);
            let expected: alloc::vec::Vec<u8> =
                values.iter().zip(factors).map(|(&a, &b)| mul_bytes(a, b)).collect();
            prop_assert_eq!(&products, &expected);

            let mut inverses = values.to_vec();
            invert_slice(&mut inverses);
            let expected: alloc::vec::Vec<u8> = values.iter().map(|&a| invert_byte(a)).collect();
            prop_assert_eq!(&inverses, &expected);

            let matrix = ByteMatrix::from_images(images);
            let mut mapped = values.to_vec();
            matrix.apply_slice(&mut mapped);
            let expected: alloc::vec::Vec<u8> =
                values.iter().map(|&a| apply_by_columns(images, a)).collect();
            prop_assert_eq!(&mapped, &expected);
        }
    }

    #[test]
    #[should_panic = "elementwise product lengths differ"]
    fn the_product_kernel_rejects_mismatched_lengths() {
        mul_slice(&mut [0, 0], &[0]);
    }
}
