//! The butterfly kernel each field runs.
//!
//! Every level shares one scalar-or-packed kernel, and two families of levels specialise it:
//!
//! - Tower levels from `GF(2^8)` up scale by a small-subfield twiddle coordinate by coordinate.
//! - The 64-bit polynomial basis multiplies a whole register of elements per carryless multiply.

mod poly64;
mod subfield;

use p3_binary_field::poly_basis::HAS_HARDWARE_CLMUL;
use p3_binary_field::{
    BinaryField2, BinaryField4, BinaryField8, BinaryField16, BinaryField32, BinaryField64,
    BinaryField128, Gf2, Ghash128, Poly64, TowerLevel,
};
use p3_field::{PackedValue, PrimeCharacteristicRing};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use subfield::{BYTE_MAP_TWIDDLE_BITS, coordinate_butterfly};

use crate::lch;
use crate::poly::stages::{INTO_POLY, INTO_TOWER, convert};

/// A field the additive transform has a butterfly kernel for.
///
/// Every tower level implements it.
///
/// The levels differ in how much of the twiddle's structure their kernel exploits.
///
/// A level runs the transform network in its own representation, unless an isomorphic one has
/// cheaper products. The result is the same either way.
pub trait ButterflyField: TowerLevel {
    /// Send each pair `(u, v)` to `(u + t*v, u + (t + 1)*v)`, in place.
    ///
    /// The inverse flag applies the inverse map instead.
    ///
    /// # Panics
    ///
    /// Panics if the two runs have different lengths.
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self);

    /// Run the Lin-Chung-Han transform in place, over a row-major buffer of `width` columns and
    /// the coset `shift + S_l`.
    ///
    /// The inverse flag recovers the coefficients from the evaluations instead.
    ///
    /// # Panics
    ///
    /// - Panics if the row count is not a power of two.
    /// - Panics if the domain dimension exceeds the bit width of the level.
    #[inline]
    fn lch_transform<const INVERSE: bool>(values: &mut [Self], width: usize, shift: Self) {
        lch::transform::<Self, INVERSE>(values, width, shift);
    }

    /// Run the forward Lin-Chung-Han transform in place, over a row-major buffer of `width`
    /// columns whose leading `2^log_message` rows hold the message and the rest zeros.
    ///
    /// # Panics
    ///
    /// - Panics if the buffer is not a whole number of message-sized cosets.
    /// - Panics if the dimension of the domain the cosets cover exceeds the bit width of the level.
    #[inline]
    fn lch_transform_cosets(values: &mut [Self], width: usize, log_message: usize) {
        lch::transform_cosets::<Self>(values, width, log_message);
    }
}

/// The butterfly over whole SIMD packings, then the scalar tail.
///
/// Every specialised kernel falls back to this one, and is tested against it.
#[inline]
pub(crate) fn packed_butterfly<F: TowerLevel, const INVERSE: bool>(
    lo: &mut [F],
    hi: &mut [F],
    t: F,
) {
    // Both runs have equal length, so their packed prefixes and scalar tails pair exactly.
    let (lo, lo_tail) = F::Packing::pack_slice_with_suffix_mut(lo);
    let (hi, hi_tail) = F::Packing::pack_slice_with_suffix_mut(hi);
    let zero = t.is_zero();
    butterfly_values::<_, INVERSE>(lo, hi, t.into(), zero);
    butterfly_values::<_, INVERSE>(lo_tail, hi_tail, t, zero);
}

/// The butterfly over scalar or packed values, with the zero twiddle taken apart.
#[inline]
fn butterfly_values<R: PrimeCharacteristicRing + Copy, const INVERSE: bool>(
    lo: &mut [R],
    hi: &mut [R],
    t: R,
    zero: bool,
) {
    if zero {
        // A zero twiddle reduces both directions to (u, u + v).
        for (u, v) in lo.iter_mut().zip(hi) {
            *v += *u;
        }
    } else if INVERSE {
        // Recover the upper value first, then remove its scaled copy from the lower one.
        for (u, v) in lo.iter_mut().zip(hi) {
            *v += *u;
            *u += t * *v;
        }
    } else {
        // Scale the upper value into the lower one, then add the result to the upper one.
        for (u, v) in lo.iter_mut().zip(hi) {
            *u += t * *v;
            *v += *u;
        }
    }
}

/// The butterfly of a level with no structure to exploit beyond its packing.
///
/// # Panics
///
/// Panics if the two runs have different lengths.
#[inline]
fn plain_butterfly<F: TowerLevel, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F) {
    // The packed kernel stops at the shorter run, so a mismatch would silently drop work.
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");
    packed_butterfly::<F, INVERSE>(lo, hi, t);
}

/// Implement the butterfly trait for each listed level by forwarding to one kernel.
macro_rules! impl_butterfly_field {
    ($kernel:ident: $($field:ty),* $(,)?) => {$(
        impl ButterflyField for $field {
            #[inline]
            fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
                $kernel::<Self, INVERSE>(lo, hi, t);
            }
        }
    )*};
}

// The byte-aligned tower levels, whose bytes are their subfield coordinates.
impl_butterfly_field!(coordinate_butterfly: BinaryField8, BinaryField16, BinaryField32, BinaryField64);

// The sub-byte levels and the GHASH basis, which only have their packing.
impl_butterfly_field!(plain_butterfly: Gf2, BinaryField2, BinaryField4, Ghash128);

impl ButterflyField for Poly64 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        poly64::butterfly::<INVERSE>(lo, hi, t);
    }
}

// The widest byte-aligned level, whose network runs in the polynomial basis when a twiddle is wider than the byte map covers.
//
// - A tower product by such a twiddle changes basis three times around one carryless product.
// - Changing the whole buffer once each way costs two changes per element instead.
// - The byte map scales by a twiddle of its own subfield with no change of basis, so a transform with only those stays put.
// - The change of basis is a field isomorphism that sends the tower Cantor basis to the `Ghash128` one.
// - So every twiddle and domain point maps to its image, and the network computes the image of the tower result.
// - Changing that back gives the tower result exactly.
//
// Without a carryless multiply a `Ghash128` product is bit-serial, so the network stays in the tower basis.
impl ButterflyField for BinaryField128 {
    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        coordinate_butterfly::<Self, INVERSE>(lo, hi, t);
    }

    fn lch_transform<const INVERSE: bool>(values: &mut [Self], width: usize, shift: Self) {
        let log_n = log2_strict_usize(values.len() / width);
        if HAS_HARDWARE_CLMUL && has_wide_twiddles(log_n, shift) {
            ghash_transform::<INVERSE>(values, width, shift);
        } else {
            lch::transform::<Self, INVERSE>(values, width, shift);
        }
    }

    fn lch_transform_cosets(values: &mut [Self], width: usize, log_message: usize) {
        // Each coset's shift is a domain point below the height, so the subspace the height spans holds every coset.
        let log_n = log2_ceil_usize(values.len() / width);
        if HAS_HARDWARE_CLMUL && has_wide_twiddles(log_n, Self::ZERO) {
            ghash_transform_cosets(values, width, log_message);
        } else {
            lch::transform_cosets::<Self>(values, width, log_message);
        }
    }
}

/// Whether a twiddle of the transform over `shift + S_l` can lie past the subfield the byte map covers.
///
/// - Each twiddle is `W_j(shift)` plus a point of `S_l`, with `l = log_n`.
/// - The first `2^t` Cantor basis vectors span the tower subfield of `2^t` bits, which each `W_j` maps into itself.
/// - So every twiddle lies in the smallest tower subfield that holds both the shift and `S_l`.
///
/// That subfield fits within the byte map's exactly when both the shift and `S_l` do.
#[inline]
fn has_wide_twiddles(log_n: usize, shift: BinaryField128) -> bool {
    log_n > BYTE_MAP_TWIDDLE_BITS || shift.to_repr() >> BYTE_MAP_TWIDDLE_BITS != 0
}

/// The widest level's transform, with its network run over `Ghash128` between two changes of basis.
pub(crate) fn ghash_transform<const INVERSE: bool>(
    values: &mut [BinaryField128],
    width: usize,
    shift: BinaryField128,
) {
    let words = BinaryField128::as_repr_slice_mut(values);
    convert(words, INTO_POLY);
    lch::transform::<Ghash128, INVERSE>(as_ghash(words), width, Ghash128::from(shift));
    convert(words, INTO_TOWER);
}

/// The widest level's padded transform, with its network run over `Ghash128` between two changes of basis.
pub(crate) fn ghash_transform_cosets(
    values: &mut [BinaryField128],
    width: usize,
    log_message: usize,
) {
    let words = BinaryField128::as_repr_slice_mut(values);

    // Every coset is written from the message, so the zero tail needs no change of basis.
    convert(&mut words[..width << log_message], INTO_POLY);

    // Each coset's shift is a domain point, and those of `Ghash128` are the images of the tower's.
    lch::transform_cosets::<Ghash128>(as_ghash(words), width, log_message);
    convert(words, INTO_TOWER);
}

/// Polynomial coordinates read in place as the `Ghash128` elements they are.
#[inline]
const fn as_ghash(words: &mut [u128]) -> &mut [Ghash128] {
    // SAFETY: `Ghash128` is `#[repr(transparent)]` over `u128`.
    //
    // - A run of one is therefore a run of the other, of the same length and alignment.
    // - Every `u128` is a canonical `Ghash128`, so every word read through the view is a valid element.
    // - Every element written through the view is a `u128`, so the words stay initialised.
    // - The view borrows the same words exclusively for the same lifetime.
    unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<Ghash128>(), words.len()) }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::TowerLevel;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::{
        BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, ButterflyField,
        Ghash128, Poly64,
    };

    /// The twiddles a random search is unlikely to reach.
    ///
    /// Each one sits at a boundary of the subfield classification or of the butterfly itself:
    ///
    /// - `0` collapses the butterfly to a single addition.
    /// - `1` is the identity multiplier.
    /// - `0x80`, `0x8000`, `0x8000_0000` are the top basis elements of the one-, two- and four-byte subfields.
    /// - `0xff`, `0xffff`, `0xffff_ffff` are the widest twiddles each of those maps covers.
    /// - `0x100`, `0x1_0000`, `0x1_0000_0000` are the narrowest that need the next map up.
    /// - `1 << 63` and `u64::MAX` are the top basis element and the largest reduction spill at 64 bits.
    /// - `1 << 127` is the top basis element of the widest level.
    /// - `0x87` is the tail of the GHASH modulus.
    const CORNERS: [u128; 15] = [
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
        1 << 63,
        u64::MAX as u128,
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
    fn every_tower_level_agrees_with_the_element_loop_at_the_corners() {
        // The five byte-aligned levels take the subfield paths, and the GHASH basis its packing.
        sweep::<BinaryField8>("BinaryField8");
        sweep::<BinaryField16>("BinaryField16");
        sweep::<BinaryField32>("BinaryField32");
        sweep::<BinaryField64>("BinaryField64");
        sweep::<BinaryField128>("BinaryField128");
        sweep::<Ghash128>("Ghash128");
    }

    #[test]
    fn the_polynomial_basis_level_agrees_with_the_element_loop_at_the_corners() {
        // The 64-bit polynomial basis takes its carryless-multiply register kernel where the build enables one.
        //
        // CI interprets this test alone under Miri with that kernel enabled, since no runner is guaranteed to execute it.
        sweep::<Poly64>("Poly64");
    }

    #[test]
    fn one_byte_twiddles_agree_around_every_register_boundary() {
        // Lengths around the register and threshold boundaries.
        const LENS: [usize; 14] = [0, 1, 15, 16, 17, 31, 32, 33, 255, 256, 257, 511, 512, 513];
        for t in [0x01u8, 0x02, 0x03, 0x80, 0xa5, 0xff] {
            for len in LENS {
                agrees::<BinaryField8>(len, BinaryField8::from_repr(t)).unwrap();
                agrees::<BinaryField16>(len, BinaryField16::from_repr(t as u16)).unwrap();
                agrees::<BinaryField32>(len, BinaryField32::from_repr(t as u32)).unwrap();
                agrees::<BinaryField64>(len, BinaryField64::from_repr(t as u64)).unwrap();
                agrees::<BinaryField128>(len, BinaryField128::from_repr(t as u128)).unwrap();
            }
        }
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
            agrees::<Poly64>(len, Poly64::from_le_byte_iter(bytes.into_iter()))?;
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

        /// The same at lengths that reach the register kernels.
        #[test]
        fn byte_twiddles_agree_on_runs_past_the_register_threshold(
            t in any::<u8>(),
            len in 0usize..600,
        ) {
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
