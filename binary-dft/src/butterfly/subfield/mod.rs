//! Butterflies whose twiddle lies in a small byte-aligned tower subfield.
//!
//! Over the subspace itself, the twiddle of stage `j` lies in the span of the first `l - j` basis vectors.
//!
//! So most twiddles of a transform sit in `GF(2^8)`, `GF(2^16)` or `GF(2^32)`.
//!
//! Scaling by such a twiddle acts on each subfield coordinate alone, which is far cheaper than a full product.

use core::ops::Mul;

use p3_binary_field::poly_basis::HAS_HARDWARE_CLMUL;
use p3_binary_field::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, TowerLevel,
};
use p3_field::PrimeCharacteristicRing;

use super::packed_butterfly;

// The byte-map kernels, and the register abstraction they are written against.
//
// A build without the instruction never reaches them, so it compiles none of them.
//
// Tests compile them on every target, against a scalar model of the register.
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )
))]
mod gfni;
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ),
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )
))]
mod lanes;
#[cfg(test)]
mod model;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )
))]
mod neon;

/// A tower level whose bytes are its `GF(2^8)` coordinates, lowest first.
///
/// # Safety
///
/// - An implementor is a transparent wrapper over an unsigned integer of its own width.
/// - Every bit pattern of that integer is a field element.
/// - On a little-endian target its bytes are its tower coordinates, in order.
pub(super) unsafe trait ByteCoordinates: TowerLevel {}

// The tower levels from GF(2^8) up, each a transparent wrapper over a matching integer.
//
// Below a byte, a level does not fill its backing integer.
//
// The GHASH and Poly64 representations use other bases, so their bytes are not tower coordinates.
unsafe impl ByteCoordinates for BinaryField8 {}
unsafe impl ByteCoordinates for BinaryField16 {}
unsafe impl ByteCoordinates for BinaryField32 {}
unsafe impl ByteCoordinates for BinaryField64 {}
unsafe impl ByteCoordinates for BinaryField128 {}

/// The smallest byte-aligned tower subfield a twiddle lies in.
///
/// The subfield of `b` bytes occupies the low `b` bytes of the tower basis.
///
/// So the magnitude of the bit pattern alone names the subfield.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum TwiddleWidth {
    /// In `GF(2^8)`, carrying its single coordinate.
    Byte(u8),
    /// In `GF(2^16)` but not in `GF(2^8)`.
    Word(u16),
    /// In `GF(2^32)` but not in `GF(2^16)`.
    DoubleWord(u32),
    /// Wider than four bytes, which no typed product here covers.
    Wide,
}

impl TwiddleWidth {
    /// Classify a twiddle by the bit pattern of its tower coordinates.
    pub(super) const fn of(bits: u128) -> Self {
        if bits <= u8::MAX as u128 {
            Self::Byte(bits as u8)
        } else if bits <= u16::MAX as u128 {
            Self::Word(bits as u16)
        } else if bits <= u32::MAX as u128 {
            Self::DoubleWord(bits as u32)
        } else {
            Self::Wide
        }
    }
}

/// Whether the two- and four-byte typed products beat the full product at the 64- and 128-bit levels.
///
/// - A typed product costs one subfield product per coordinate.
/// - A carryless-multiply instruction makes the full product cheaper than the two- and four-byte walk.
/// - The one-byte product stays ahead either way, being a single table lookup per coordinate.
const WIDE_TYPED_PRODUCTS_PAY: bool = !HAS_HARDWARE_CLMUL;

/// The butterfly with a twiddle typed as an element of the subfield `S`.
///
/// ```text
///     forward:  u += t * v,  v += u
///     inverse:  v += u,      u += t * v
/// ```
#[inline]
fn typed_butterfly<F, S, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: S)
where
    F: Copy + PrimeCharacteristicRing + Mul<S, Output = F>,
    S: Copy,
{
    if INVERSE {
        for (u, v) in lo.iter_mut().zip(hi) {
            // Recover the upper value first, then remove its scaled copy from the lower one.
            *v += *u;
            *u += *v * t;
        }
    } else {
        for (u, v) in lo.iter_mut().zip(hi) {
            // Scale the upper value into the lower one, then add the result to the upper one.
            *u += *v * t;
            *v += *u;
        }
    }
}

/// Scale by a one-byte twiddle, or fall through to the full-width kernel.
///
/// No byte-aligned subfield is narrower, so this ends the chain.
#[inline]
fn by_byte<F, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F, width: TwiddleWidth)
where
    F: TowerLevel + Mul<BinaryField8, Output = F>,
{
    match width {
        TwiddleWidth::Byte(s) => {
            typed_butterfly::<F, _, INVERSE>(lo, hi, BinaryField8::from_repr(s));
        }
        _ => packed_butterfly::<F, INVERSE>(lo, hi, t),
    }
}

/// Scale by a two-byte twiddle, or hand a narrower one down the chain.
#[inline]
fn by_word<F, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F, width: TwiddleWidth)
where
    F: TowerLevel + Mul<BinaryField8, Output = F> + Mul<BinaryField16, Output = F>,
{
    match width {
        TwiddleWidth::Word(s) => {
            typed_butterfly::<F, _, INVERSE>(lo, hi, BinaryField16::from_repr(s));
        }
        _ => by_byte::<F, INVERSE>(lo, hi, t, width),
    }
}

/// Scale by a four-byte twiddle, or hand a narrower one down the chain.
#[inline]
fn by_double_word<F, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F, width: TwiddleWidth)
where
    F: TowerLevel
        + Mul<BinaryField8, Output = F>
        + Mul<BinaryField16, Output = F>
        + Mul<BinaryField32, Output = F>,
{
    match width {
        TwiddleWidth::DoubleWord(s) => {
            typed_butterfly::<F, _, INVERSE>(lo, hi, BinaryField32::from_repr(s));
        }
        _ => by_word::<F, INVERSE>(lo, hi, t, width),
    }
}

/// A byte-coordinate level, with the narrowest typed product that covers each twiddle.
///
/// A typed product needs the subfield to be strictly narrower than the level.
pub(super) trait SubfieldScaled: ByteCoordinates {
    /// Apply the butterfly through the narrowest typed product that covers the twiddle.
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    );
}

impl SubfieldScaled for BinaryField8 {
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        _width: TwiddleWidth,
    ) {
        // Every twiddle is the whole element, so the full kernel already is the typed one.
        packed_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl SubfieldScaled for BinaryField16 {
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        // Only the one-byte subfield is narrower than this level.
        by_byte::<Self, INVERSE>(lo, hi, t, width);
    }
}

impl SubfieldScaled for BinaryField32 {
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        // The one- and two-byte subfields are narrower than this level.
        by_word::<Self, INVERSE>(lo, hi, t, width);
    }
}

impl SubfieldScaled for BinaryField64 {
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        // The wider typed products lose to a carryless multiply, the one-byte product never does.
        if WIDE_TYPED_PRODUCTS_PAY {
            by_double_word::<Self, INVERSE>(lo, hi, t, width);
        } else {
            by_byte::<Self, INVERSE>(lo, hi, t, width);
        }
    }
}

impl SubfieldScaled for BinaryField128 {
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        // The wider typed products lose to a carryless multiply, the one-byte product never does.
        if WIDE_TYPED_PRODUCTS_PAY {
            by_double_word::<Self, INVERSE>(lo, hi, t, width);
        } else {
            by_byte::<Self, INVERSE>(lo, hi, t, width);
        }
    }
}

/// The bit width of the widest twiddle the register kernel scales byte by byte.
///
/// Its byte maps cover the one-, two- and four-byte subfields.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(super) const BYTE_MAP_TWIDDLE_BITS: usize = 32;

/// The bit width of the widest twiddle the butterfly scales byte by byte where the target multiplies carrylessly.
///
/// Only the one-byte product beats a carryless multiply, through the NEON byte map or the typed product.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
pub(super) const BYTE_MAP_TWIDDLE_BITS: usize = 8;

/// Run the leading whole registers through the byte map the twiddle's subfield allows.
///
/// # Returns
///
/// The number of elements covered, which the caller finishes from.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[inline]
fn register_prefix<F: ByteCoordinates, const INVERSE: bool>(
    lo: &mut [F],
    hi: &mut [F],
    width: TwiddleWidth,
) -> usize {
    use core::arch::x86_64::__m512i;

    use lanes::ByteRegister;

    const {
        // The byte layout holds only where the bytes run from the low coordinate up.
        assert!(cfg!(target_endian = "little"));

        // One element is its whole backing integer, so its size is its coordinate count.
        assert!(size_of::<F>() == 1 << (F::LOG_BITS - 3));

        // A register is 64 bytes, so a whole number of registers is a whole number of elements.
        assert!(64 % size_of::<F>() == 0);
    }

    // A run below one register covers nothing.
    //
    // Building the blocks first would then be pure waste, on the short runs that dominate narrow stages.
    let bytes = size_of_val(lo);
    if bytes < <__m512i as ByteRegister>::BYTES {
        return 0;
    }

    let (lo, hi) = as_bytes(lo, hi);

    // A group is one coordinate of the twiddle's subfield.
    //
    // An element must hold whole groups for the map to act coordinate by coordinate.
    let covered = match width {
        TwiddleWidth::Byte(t) => {
            gfni::subfield_butterfly::<__m512i, 1, INVERSE>(lo, hi, &gfni::byte_blocks(t))
        }
        TwiddleWidth::Word(t) if size_of::<F>() >= 2 => {
            gfni::subfield_butterfly::<__m512i, 2, INVERSE>(lo, hi, &gfni::word_blocks(t))
        }
        TwiddleWidth::DoubleWord(t) if size_of::<F>() >= 4 => {
            gfni::subfield_butterfly::<__m512i, 4, INVERSE>(lo, hi, &gfni::dword_blocks(t))
        }
        _ => 0,
    };
    covered / size_of::<F>()
}

/// Run the leading whole registers through the one-byte nibble-table kernel.
///
/// # Returns
///
/// The number of elements covered, which the caller finishes from.
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
#[inline]
fn register_prefix<F: ByteCoordinates, const INVERSE: bool>(
    lo: &mut [F],
    hi: &mut [F],
    width: TwiddleWidth,
) -> usize {
    use core::arch::aarch64::uint8x16_t;

    use lanes::ByteRegister;

    const {
        // One element is its whole backing integer, so its size is its coordinate count.
        assert!(size_of::<F>() == 1 << (F::LOG_BITS - 3));

        // A register is 16 bytes, so a whole number of registers is a whole number of elements.
        assert!(<uint8x16_t as ByteRegister>::BYTES.is_multiple_of(size_of::<F>()));
    }

    // Only a one-byte twiddle has a kernel here.
    let TwiddleWidth::Byte(t) = width else {
        return 0;
    };

    // Short runs do not repay building the two lookup tables.
    let bytes = size_of_val(lo);
    if bytes < neon::MIN_BYTES {
        return 0;
    }

    let (lo, hi) = as_bytes(lo, hi);

    // The map is linear, so its images of the eight basis bytes determine it.
    let t = BinaryField8::from_repr(t);
    let columns: [u8; 8] =
        core::array::from_fn(|b| (t * BinaryField8::from_repr(1 << b)).to_repr());

    neon::byte_butterfly::<INVERSE>(lo, hi, &columns) / size_of::<F>()
}

/// Without a byte-map instruction there is no register prefix, so the whole run falls to the caller.
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ),
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )
)))]
#[inline]
const fn register_prefix<F: ByteCoordinates, const INVERSE: bool>(
    _lo: &mut [F],
    _hi: &mut [F],
    _width: TwiddleWidth,
) -> usize {
    0
}

/// View two equal-length element runs as their bytes.
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ),
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )
))]
#[inline]
fn as_bytes<'a, F: ByteCoordinates>(
    lo: &'a mut [F],
    hi: &'a mut [F],
) -> (&'a mut [u8], &'a mut [u8]) {
    let bytes = size_of_val(lo);
    debug_assert_eq!(bytes, size_of_val(hi));

    // SAFETY: the marker trait makes each run exactly that many initialised bytes.
    //
    // - It rules out padding, and any byte pattern a write could turn into an invalid element.
    // - The two runs are distinct borrows, so no byte belongs to both.
    // - The runs have equal length, so one byte count is in bounds for both.
    unsafe {
        (
            core::slice::from_raw_parts_mut(lo.as_mut_ptr().cast::<u8>(), bytes),
            core::slice::from_raw_parts_mut(hi.as_mut_ptr().cast::<u8>(), bytes),
        )
    }
}

/// The butterfly of a level whose bytes are its tower coordinates.
///
/// - A twiddle in a byte-aligned subfield drives a byte map over whole registers.
/// - The tail the registers leave falls to a typed product of the same subfield.
/// - A wider twiddle takes the packed full-width kernel.
///
/// # Panics
///
/// Panics if the two runs have different lengths.
#[inline]
pub(super) fn coordinate_butterfly<F, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F)
where
    F: SubfieldScaled,
    F::Repr: Into<u128>,
{
    // Every kernel below stops at the shorter run, so a mismatch would silently drop work.
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");

    // A zero twiddle leaves an addition, which the packed kernel already shortcuts.
    if t.is_zero() {
        return packed_butterfly::<F, INVERSE>(lo, hi, t);
    }

    let width = TwiddleWidth::of(t.to_repr().into());
    let covered = register_prefix::<F, INVERSE>(lo, hi, width);

    // A subfield twiddle scales each coordinate alone, which beats a full product even without a byte map.
    F::scale_butterfly::<INVERSE>(&mut lo[covered..], &mut hi[covered..], t, width);
}

#[cfg(test)]
mod tests {
    use super::TwiddleWidth;

    #[test]
    fn the_classification_follows_the_twiddle_magnitude() {
        // Fixture: both ends of each subfield band, plus the first value past the widest.
        //
        //     0             ..= 0xff          one byte
        //     0x100         ..= 0xffff        two bytes
        //     0x1_0000      ..= 0xffff_ffff   four bytes
        //     0x1_0000_0000 and up            no typed product
        assert_eq!(TwiddleWidth::of(0), TwiddleWidth::Byte(0));
        assert_eq!(TwiddleWidth::of(0xff), TwiddleWidth::Byte(0xff));
        assert_eq!(TwiddleWidth::of(0x100), TwiddleWidth::Word(0x100));
        assert_eq!(TwiddleWidth::of(0xffff), TwiddleWidth::Word(0xffff));
        assert_eq!(
            TwiddleWidth::of(0x1_0000),
            TwiddleWidth::DoubleWord(0x1_0000)
        );
        assert_eq!(
            TwiddleWidth::of(0xffff_ffff),
            TwiddleWidth::DoubleWord(0xffff_ffff)
        );
        assert_eq!(TwiddleWidth::of(0x1_0000_0000), TwiddleWidth::Wide);
        assert_eq!(TwiddleWidth::of(u128::MAX), TwiddleWidth::Wide);
    }
}
