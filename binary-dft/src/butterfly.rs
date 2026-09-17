//! The butterfly kernel each field runs, and the subfield structure it exploits.

use core::ops::Mul;

use p3_binary_field::poly_basis::HAS_HARDWARE_CLMUL;
use p3_binary_field::{
    BinaryField2, BinaryField4, BinaryField8, BinaryField16, BinaryField32, BinaryField64,
    BinaryField128, Gf2, Ghash128, TowerLevel,
};
use p3_field::{PackedValue, PrimeCharacteristicRing};

/// A field the additive transform has a butterfly kernel for.
///
/// Every tower level implements this, so the bound admits exactly what the transform did.
/// The supertrait is sealed, so no type outside the field crate can gain an implementation.
///
/// What differs between levels is how much of the twiddle's structure the kernel can use.
pub trait ButterflyField: TowerLevel {
    /// Send `(lo, hi)` to `(lo + t*hi, lo + (t + 1)*hi)`, element by element.
    ///
    /// Setting the flag applies the inverse of that map instead.
    ///
    /// Callers outside a transform may use this as a standalone kernel over two runs.
    ///
    /// # Panics
    /// Panics if the two runs have different lengths.
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self);
}

/// A tower level whose bytes are its `GF(2^8)` coordinates, lowest first.
///
/// # Safety
///
/// An implementor is a transparent wrapper over an unsigned integer of its own width.
///
/// Every bit pattern of that integer is a field element.
///
/// On a little-endian target its bytes are its tower coordinates, in order.
unsafe trait ByteCoordinates: TowerLevel {}

// The tower levels from `GF(2^8)` up, each a transparent wrapper over a matching integer.
//
// Below a byte a level does not fill its backing integer, so it has no byte coordinates.
//
// The GHASH representation is a different basis, so its bytes are not tower coordinates.
unsafe impl ByteCoordinates for BinaryField8 {}
unsafe impl ByteCoordinates for BinaryField16 {}
unsafe impl ByteCoordinates for BinaryField32 {}
unsafe impl ByteCoordinates for BinaryField64 {}
unsafe impl ByteCoordinates for BinaryField128 {}

/// The smallest byte-aligned tower subfield a twiddle lies in.
///
/// The coordinates of the subfield of `b` bytes occupy the low `b` bytes of the tower basis.
///
/// A twiddle's magnitude alone therefore says which subfield holds it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TwiddleWidth {
    /// The twiddle lies in `GF(2^8)`, and this is its single coordinate.
    Byte(u8),
    /// The twiddle lies in `GF(2^16)` but not in `GF(2^8)`.
    Word(u16),
    /// The twiddle lies in `GF(2^32)` but not in `GF(2^16)`.
    DoubleWord(u32),
    /// The twiddle needs more than four bytes, which no byte map here covers.
    Wide,
}

/// Which subfield a twiddle's bit pattern places it in.
const fn classify(bits: u128) -> TwiddleWidth {
    if bits <= u8::MAX as u128 {
        TwiddleWidth::Byte(bits as u8)
    } else if bits <= u16::MAX as u128 {
        TwiddleWidth::Word(bits as u16)
    } else if bits <= u32::MAX as u128 {
        TwiddleWidth::DoubleWord(bits as u32)
    } else {
        TwiddleWidth::Wide
    }
}

/// Apply a butterfly to full SIMD vectors and any remaining scalar elements.
///
/// Every level falls back to this, and every specialised path is pinned against it.
#[inline]
pub(crate) fn packed_butterfly<F: TowerLevel, const INVERSE: bool>(
    lo: &mut [F],
    hi: &mut [F],
    t: F,
) {
    // Both sides have equal length, so their packed prefixes and tails pair exactly.
    let (lo, lo_tail) = F::Packing::pack_slice_with_suffix_mut(lo);
    let (hi, hi_tail) = F::Packing::pack_slice_with_suffix_mut(hi);
    let zero = t.is_zero();
    butterfly_values::<_, INVERSE>(lo, hi, t.into(), zero);
    butterfly_values::<_, INVERSE>(lo_tail, hi_tail, t, zero);
}

/// Apply the same field identities to scalar or packed values.
#[inline]
fn butterfly_values<R: PrimeCharacteristicRing + Copy, const INVERSE: bool>(
    lo: &mut [R],
    hi: &mut [R],
    t: R,
    zero: bool,
) {
    if zero {
        // A zero twiddle reduces both transform directions to (u, u + v).
        for (u, v) in lo.iter_mut().zip(hi) {
            *v += *u;
        }
    } else if INVERSE {
        // Recover v first, then remove its twiddle contribution from u.
        for (u, v) in lo.iter_mut().zip(hi) {
            *v += *u;
            *u += t * *v;
        }
    } else {
        // Evaluate the pair as (u + t*v, u + t*v + v).
        for (u, v) in lo.iter_mut().zip(hi) {
            *u += t * *v;
            *v += *u;
        }
    }
}

/// Apply a butterfly whose twiddle is typed as an element of a subfield.
///
/// Such a product is the subfield product applied to each coordinate.
///
/// That is cheaper than a full-width product at every level above the subfield.
#[inline]
fn typed_butterfly<F, S, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: S)
where
    F: Copy + PrimeCharacteristicRing + Mul<S, Output = F>,
    S: Copy,
{
    if INVERSE {
        // Recover v first, then remove its twiddle contribution from u.
        for (u, v) in lo.iter_mut().zip(hi) {
            *v += *u;
            *u += *v * t;
        }
    } else {
        // Evaluate the pair as (u + t*v, u + t*v + v).
        for (u, v) in lo.iter_mut().zip(hi) {
            *u += *v * t;
            *v += *u;
        }
    }
}

/// Scale by a one-byte twiddle, or fall through to the full-width kernel.
///
/// This is the end of the chain: no byte-aligned subfield is narrower than one byte.
#[inline]
fn scale_by_byte<F, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F, width: TwiddleWidth)
where
    F: TowerLevel + Mul<BinaryField8, Output = F>,
{
    if let TwiddleWidth::Byte(s) = width {
        typed_butterfly::<F, BinaryField8, INVERSE>(lo, hi, BinaryField8::from_repr(s));
    } else {
        packed_butterfly::<F, INVERSE>(lo, hi, t);
    }
}

/// Scale by a two-byte twiddle, or hand a narrower one down the chain.
#[inline]
fn scale_by_word<F, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F, width: TwiddleWidth)
where
    F: TowerLevel + Mul<BinaryField8, Output = F> + Mul<BinaryField16, Output = F>,
{
    if let TwiddleWidth::Word(s) = width {
        typed_butterfly::<F, BinaryField16, INVERSE>(lo, hi, BinaryField16::from_repr(s));
    } else {
        scale_by_byte::<F, INVERSE>(lo, hi, t, width);
    }
}

/// Scale by a four-byte twiddle, or hand a narrower one down the chain.
#[inline]
fn scale_by_dword<F, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F, width: TwiddleWidth)
where
    F: TowerLevel
        + Mul<BinaryField8, Output = F>
        + Mul<BinaryField16, Output = F>
        + Mul<BinaryField32, Output = F>,
{
    if let TwiddleWidth::DoubleWord(s) = width {
        typed_butterfly::<F, BinaryField32, INVERSE>(lo, hi, BinaryField32::from_repr(s));
    } else {
        scale_by_word::<F, INVERSE>(lo, hi, t, width);
    }
}

/// Whether a typed product beats the full-width product at the two widest levels.
///
/// - Scaling by a `D`-byte subfield costs one subfield product per coordinate.
/// - Recursive multiplication triples in cost per doubling of the width.
/// - The coordinate count only halves, so the narrower subfield always wins.
///
/// A carryless-multiply instruction breaks that at the widest two levels.
/// Only the one-byte coordinates stay ahead of it, being single table lookups.
const WIDE_TYPED_PRODUCTS_PAY: bool = !HAS_HARDWARE_CLMUL;

/// A tower level together with the typed subfield products its own width admits.
///
/// A typed product needs the subfield to be strictly narrower than the level.
/// So the narrow levels route more of the twiddle widths to the full-width kernel.
///
/// The chain above holds the shared logic, and an implementation only names its entry point.
trait SubfieldScaled: ByteCoordinates {
    /// Apply the butterfly through the narrowest typed product that covers the twiddle.
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    );
}

impl SubfieldScaled for BinaryField8 {
    /// One byte wide, so every twiddle is the whole element and no subfield is left.
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        scale_by_byte::<Self, INVERSE>(lo, hi, t, width);
    }
}

impl SubfieldScaled for BinaryField16 {
    /// A two-byte twiddle is the whole element here, so the one-byte product is the only gain.
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        scale_by_byte::<Self, INVERSE>(lo, hi, t, width);
    }
}

impl SubfieldScaled for BinaryField32 {
    /// A four-byte twiddle is the whole element here, so the chain stops at two bytes.
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        scale_by_word::<Self, INVERSE>(lo, hi, t, width);
    }
}

impl SubfieldScaled for BinaryField64 {
    /// Wide enough for every typed product, where the level's own product is recursive.
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        if WIDE_TYPED_PRODUCTS_PAY {
            scale_by_dword::<Self, INVERSE>(lo, hi, t, width);
        } else {
            scale_by_byte::<Self, INVERSE>(lo, hi, t, width);
        }
    }
}

impl SubfieldScaled for BinaryField128 {
    /// Wide enough for every typed product, where the level's own product is recursive.
    #[inline]
    fn scale_butterfly<const INVERSE: bool>(
        lo: &mut [Self],
        hi: &mut [Self],
        t: Self,
        width: TwiddleWidth,
    ) {
        if WIDE_TYPED_PRODUCTS_PAY {
            scale_by_dword::<Self, INVERSE>(lo, hi, t, width);
        } else {
            scale_by_byte::<Self, INVERSE>(lo, hi, t, width);
        }
    }
}

/// Run the leading whole registers through the byte map the twiddle's width allows.
///
/// Returns the number of elements covered, which the caller finishes from.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[inline]
fn subfield_prefix<F: ByteCoordinates, const INVERSE: bool>(
    lo: &mut [F],
    hi: &mut [F],
    width: TwiddleWidth,
) -> usize {
    use core::arch::x86_64::__m512i;

    use crate::affine;
    use crate::lanes::ByteLanes;

    const {
        // The layout contract holds only where the bytes run from the low coordinate up.
        assert!(cfg!(target_endian = "little"));

        // One element is its whole backing integer, so its size is its coordinate count.
        assert!(size_of::<F>() == 1 << (F::LOG_BITS - 3));

        // A register is 64 bytes, so a whole number of them is a whole number of elements.
        assert!(64 % size_of::<F>() == 0);
    }

    let bytes = size_of_val(lo);

    // A run below one register covers nothing, so its blocks would be built and thrown away.
    // The narrow stages of a narrow matrix are almost all of the butterflies, and all short.
    if bytes < <__m512i as ByteLanes>::BYTES {
        return 0;
    }

    // SAFETY: the marker trait's contract makes each run exactly that many initialised bytes.
    //
    // It rules out padding, and any byte pattern a write could turn into an invalid element.
    let (lo, hi) = unsafe {
        (
            core::slice::from_raw_parts_mut(lo.as_mut_ptr().cast::<u8>(), bytes),
            core::slice::from_raw_parts_mut(hi.as_mut_ptr().cast::<u8>(), bytes),
        )
    };

    // A group is one coordinate of the twiddle's subfield.
    //
    // So an element has to hold whole groups for the map to act coordinate by coordinate.
    let covered = match width {
        TwiddleWidth::Byte(t) => {
            affine::subfield_butterfly::<__m512i, 1, INVERSE>(lo, hi, &affine::byte_blocks(t))
        }
        TwiddleWidth::Word(t) if size_of::<F>() >= 2 => {
            affine::subfield_butterfly::<__m512i, 2, INVERSE>(lo, hi, &affine::word_blocks(t))
        }
        TwiddleWidth::DoubleWord(t) if size_of::<F>() >= 4 => {
            affine::subfield_butterfly::<__m512i, 4, INVERSE>(lo, hi, &affine::dword_blocks(t))
        }
        _ => 0,
    };
    covered / size_of::<F>()
}

/// Without the byte map there is no prefix to take, so the whole run falls to the caller.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
#[inline]
const fn subfield_prefix<F: ByteCoordinates, const INVERSE: bool>(
    _lo: &mut [F],
    _hi: &mut [F],
    _width: TwiddleWidth,
) -> usize {
    0
}

/// The butterfly of a level whose bytes are tower coordinates.
///
/// A twiddle narrow enough to sit in a byte-aligned subfield drives a byte map.
///
/// Whatever the byte map leaves over falls to a typed product of the same subfield.
///
/// # Panics
/// Panics if the two runs have different lengths.
#[inline]
fn coordinate_butterfly<F, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F)
where
    F: SubfieldScaled,
    F::Repr: Into<u128>,
{
    // Invariant: the two sides are paired element for element.
    //
    // Every kernel below stops at the shorter one, so a mismatch would silently drop work.
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");

    // A zero twiddle leaves an addition, which the packed kernel already shortcuts.
    if t.is_zero() {
        return packed_butterfly::<F, INVERSE>(lo, hi, t);
    }

    let width = classify(t.to_repr().into());
    let covered = subfield_prefix::<F, INVERSE>(lo, hi, width);

    // Whatever the register loop left over, down to one element.
    //
    // A subfield twiddle scales each coordinate on its own.
    // That beats a full-width product even with no byte map to run it through.
    F::scale_butterfly::<INVERSE>(&mut lo[covered..], &mut hi[covered..], t, width);
}

/// The butterfly of a level with no subfield structure to exploit.
///
/// # Panics
/// Panics if the two runs have different lengths.
#[inline]
fn plain_butterfly<F: TowerLevel, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F) {
    // Invariant: the two sides are paired element for element.
    //
    // The packed kernel stops at the shorter one, so a mismatch would silently drop work.
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");
    packed_butterfly::<F, INVERSE>(lo, hi, t);
}

impl ButterflyField for BinaryField8 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        coordinate_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl ButterflyField for BinaryField16 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        coordinate_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl ButterflyField for BinaryField32 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        coordinate_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl ButterflyField for BinaryField64 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        coordinate_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl ButterflyField for BinaryField128 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        coordinate_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl ButterflyField for Gf2 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        plain_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl ButterflyField for BinaryField2 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        plain_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl ButterflyField for BinaryField4 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        plain_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

impl ButterflyField for Ghash128 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        plain_butterfly::<Self, INVERSE>(lo, hi, t);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::TowerLevel;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::{
        BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, ButterflyField,
        Ghash128, TwiddleWidth, classify,
    };

    /// The twiddles a random search is unlikely to reach.
    ///
    /// Each one sits at a boundary of the subfield classification or of the butterfly itself:
    ///
    /// ```text
    ///     0               the butterfly collapses to a single addition
    ///     1               the identity multiplier
    ///     0x80            the highest-degree basis element of the one-byte subfield
    ///     0xff            the widest twiddle the one-byte map still covers
    ///     0x100           the narrowest that needs the two-byte map
    ///     0x8000          the highest-degree basis element of the two-byte subfield
    ///     0xffff          the widest the two-byte map covers
    ///     0x1_0000        the narrowest that needs the four-byte map
    ///     0x8000_0000     the highest-degree basis element of the four-byte subfield
    ///     0xffff_ffff     the widest the four-byte map covers
    ///     0x1_0000_0000   the narrowest that falls through to the packed kernel
    ///     1 << 127        the highest-degree basis element of the widest level
    ///     0x87            the tail of the GHASH modulus, for the level that uses it
    /// ```
    const CORNERS: [u128; 13] = [
        0,
        1,
        0x80,
        0xff,
        0x100,
        0x8000,
        0xffff,
        0x1_0000,
        0x8000_0000,
        0xffff_ffff,
        0x1_0000_0000,
        1 << 127,
        0x87,
    ];

    /// The lengths a sweep covers, in elements.
    ///
    /// A 512-bit register holds 64 bytes.
    ///
    /// That is 64 elements of the narrowest level and 4 of the widest.
    ///
    /// So this range spans several whole registers plus a tail of every size.
    const LENGTHS: core::ops::RangeInclusive<usize> = 0..=70;

    /// A run whose elements share no structure with one another.
    fn sample<F: TowerLevel>(len: usize, seed: u64) -> Vec<F> {
        (0..len)
            .map(|i| {
                // Two odd multipliers apart, so neighbouring positions share no low bits.
                let bits = seed
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(i as u64 + 1)
                    .wrapping_mul(0xbf58_476d_1ce4_e5b9);

                // Repeating the pattern fills a level wider than the eight bytes it holds.
                F::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
            })
            .collect()
    }

    /// The butterfly written out one element at a time, through the field's own product.
    fn reference<F: TowerLevel>(lo: &mut [F], hi: &mut [F], t: F, inverse: bool) {
        for (u, v) in lo.iter_mut().zip(hi) {
            if inverse {
                // Recover the upper half, then take the scaled result out of the lower one.
                *v += *u;
                *u += t * *v;
            } else {
                // Scale the upper half into the lower one, then sum both into the upper one.
                *u += t * *v;
                *v += *u;
            }
        }
    }

    /// Both directions of the kernel against the element-at-a-time loop, at one length.
    fn agrees<F: ButterflyField>(len: usize, t: F) -> Result<(), TestCaseError> {
        // Two runs with nothing in common, so a kernel that mixed them would show it.
        let (lo, hi) = (sample::<F>(len, 1), sample::<F>(len, 2));

        // Forward: the kernel and the reference must produce the same pair of runs.
        let (mut want_lo, mut want_hi) = (lo.clone(), hi.clone());
        reference(&mut want_lo, &mut want_hi, t, false);

        let (mut got_lo, mut got_hi) = (lo.clone(), hi.clone());
        F::butterfly::<false>(&mut got_lo, &mut got_hi, t);
        prop_assert_eq!(&got_lo, &want_lo);
        prop_assert_eq!(&got_hi, &want_hi);

        // The inverse applied to that output must return the input, for every twiddle.
        F::butterfly::<true>(&mut got_lo, &mut got_hi, t);
        prop_assert_eq!(&got_lo, &lo);
        prop_assert_eq!(&got_hi, &hi);

        // Inverse on its own, against its own reference loop.
        let (mut want_lo, mut want_hi) = (lo.clone(), hi.clone());
        reference(&mut want_lo, &mut want_hi, t, true);

        let (mut got_lo, mut got_hi) = (lo, hi);
        F::butterfly::<true>(&mut got_lo, &mut got_hi, t);
        prop_assert_eq!(&got_lo, &want_lo);
        prop_assert_eq!(&got_hi, &want_hi);
        Ok(())
    }

    /// Every corner twiddle at every length in the sweep, for one level.
    fn sweep<F: ButterflyField>(name: &str) {
        for &bits in &CORNERS {
            // A corner wider than the level keeps only the coordinates the level has.
            let t =
                F::from_le_byte_iter(bits.to_le_bytes().into_iter().chain(core::iter::repeat(0)));
            for len in LENGTHS {
                agrees::<F>(len, t)
                    .unwrap_or_else(|e| panic!("{name}, twiddle {bits:#x}, len {len}: {e}"));
            }
        }
    }

    #[test]
    fn every_level_agrees_with_the_element_loop_at_the_corners() {
        // The five byte-aligned levels take the subfield paths, the GHASH basis does not.
        sweep::<BinaryField8>("BinaryField8");
        sweep::<BinaryField16>("BinaryField16");
        sweep::<BinaryField32>("BinaryField32");
        sweep::<BinaryField64>("BinaryField64");
        sweep::<BinaryField128>("BinaryField128");
        sweep::<Ghash128>("Ghash128");
    }

    #[test]
    fn the_classification_follows_the_twiddle_magnitude() {
        // Fixture: both ends of each subfield band, plus the first value past the widest.
        //
        // ```text
        //     0             ..= 0xff          one byte
        //     0x100         ..= 0xffff        two bytes
        //     0x1_0000      ..= 0xffff_ffff   four bytes
        //     0x1_0000_0000 and up            no byte map
        // ```
        assert_eq!(classify(0), TwiddleWidth::Byte(0));
        assert_eq!(classify(0xff), TwiddleWidth::Byte(0xff));
        assert_eq!(classify(0x100), TwiddleWidth::Word(0x100));
        assert_eq!(classify(0xffff), TwiddleWidth::Word(0xffff));
        assert_eq!(classify(0x1_0000), TwiddleWidth::DoubleWord(0x1_0000));
        assert_eq!(classify(0xffff_ffff), TwiddleWidth::DoubleWord(0xffff_ffff));
        assert_eq!(classify(0x1_0000_0000), TwiddleWidth::Wide);
        assert_eq!(classify(u128::MAX), TwiddleWidth::Wide);
    }

    #[test]
    #[should_panic = "butterfly lengths differ"]
    fn a_length_mismatch_is_refused() {
        // A short upper half would leave the rest of the lower one untransformed.
        let mut lo = [BinaryField32::ONE; 4];
        let mut hi = [BinaryField32::ONE; 3];
        BinaryField32::butterfly::<false>(&mut lo, &mut hi, BinaryField32::ONE);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Random twiddles and lengths, over the levels the three kernels split between.
        #[test]
        fn random_twiddles_agree_with_the_element_loop(
            bits in any::<u128>(),
            len in 0usize..70,
        ) {
            let bytes = bits.to_le_bytes();
            agrees::<BinaryField32>(len, BinaryField32::from_le_byte_iter(bytes.into_iter()))?;
            agrees::<BinaryField64>(len, BinaryField64::from_le_byte_iter(bytes.into_iter()))?;
            agrees::<BinaryField128>(len, BinaryField128::from_le_byte_iter(bytes.into_iter()))?;
        }

        /// The same, with the twiddle drawn from the one-byte subfield the byte map covers.
        #[test]
        fn byte_twiddles_agree_with_the_element_loop(t in any::<u8>(), len in 0usize..70) {
            agrees::<BinaryField8>(len, BinaryField8::from_repr(t))?;
            agrees::<BinaryField16>(len, BinaryField16::from_repr(t as u16))?;
            agrees::<BinaryField32>(len, BinaryField32::from_repr(t as u32))?;
            agrees::<BinaryField64>(len, BinaryField64::from_repr(t as u64))?;
            agrees::<BinaryField128>(len, BinaryField128::from_repr(t as u128))?;
        }

        /// The same, with the twiddle drawn from the two-byte subfield.
        #[test]
        fn word_twiddles_agree_with_the_element_loop(t in 0x100u16.., len in 0usize..70) {
            agrees::<BinaryField16>(len, BinaryField16::from_repr(t))?;
            agrees::<BinaryField32>(len, BinaryField32::from_repr(t as u32))?;
            agrees::<BinaryField64>(len, BinaryField64::from_repr(t as u64))?;
            agrees::<BinaryField128>(len, BinaryField128::from_repr(t as u128))?;
        }

        /// The same, with the twiddle drawn from the four-byte subfield.
        #[test]
        fn double_word_twiddles_agree_with_the_element_loop(
            t in 0x1_0000u32..,
            len in 0usize..70,
        ) {
            agrees::<BinaryField32>(len, BinaryField32::from_repr(t))?;
            agrees::<BinaryField64>(len, BinaryField64::from_repr(t as u64))?;
            agrees::<BinaryField128>(len, BinaryField128::from_repr(t as u128))?;
        }
    }
}
