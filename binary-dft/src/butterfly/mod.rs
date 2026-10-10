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
    BinaryField128, Gf2, Ghash128, Poly64, Rijndael8b, TowerLevel,
};
use p3_field::{PackedValue, PrimeCharacteristicRing};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use subfield::{byte_map_twiddle_bits, coordinate_butterfly};

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
    /// Whether the transform driver should group three stages for this backend.
    ///
    /// The default keeps the ordinary per-stage sweep. A specialized register
    /// kernel can opt in without imposing its traversal on other field types.
    const FUSE_RADIX8: bool = false;

    /// Send each pair `(u, v)` to `(u + t*v, u + (t + 1)*v)`, in place.
    ///
    /// The inverse flag applies the inverse map instead.
    ///
    /// # Panics
    ///
    /// Panics if the two runs have different lengths.
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self);

    /// Apply three stages to eight equal-length, disjoint rows.
    ///
    /// Twiddles are in breadth-first order: one for the outer stage, two
    /// for the middle stage, and four for the inner stage. The inverse
    /// executes these stages in reverse order, using inverse butterflies.
    ///
    /// # Panics
    /// Panics if the row lengths differ.
    #[inline]
    fn butterfly_radix8<const INVERSE: bool>(rows: &mut [&mut [Self]; 8], t: &[Self; 7]) {
        radix8::<Self, INVERSE>(rows, t);
    }

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

/// Apply the twelve butterflies of three stages through each field's kernel.
#[inline]
fn radix8<F: ButterflyField, const INVERSE: bool>(rows: &mut [&mut [F]; 8], t: &[F; 7]) {
    assert!(
        rows.iter().all(|row| row.len() == rows[0].len()),
        "radix-8 row lengths differ"
    );
    let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
    if INVERSE {
        F::butterfly::<true>(r0, r1, t[3]);
        F::butterfly::<true>(r2, r3, t[4]);
        F::butterfly::<true>(r4, r5, t[5]);
        F::butterfly::<true>(r6, r7, t[6]);
        F::butterfly::<true>(r0, r2, t[1]);
        F::butterfly::<true>(r1, r3, t[1]);
        F::butterfly::<true>(r4, r6, t[2]);
        F::butterfly::<true>(r5, r7, t[2]);
        F::butterfly::<true>(r0, r4, t[0]);
        F::butterfly::<true>(r1, r5, t[0]);
        F::butterfly::<true>(r2, r6, t[0]);
        F::butterfly::<true>(r3, r7, t[0]);
    } else {
        F::butterfly::<false>(r0, r4, t[0]);
        F::butterfly::<false>(r1, r5, t[0]);
        F::butterfly::<false>(r2, r6, t[0]);
        F::butterfly::<false>(r3, r7, t[0]);
        F::butterfly::<false>(r0, r2, t[1]);
        F::butterfly::<false>(r1, r3, t[1]);
        F::butterfly::<false>(r4, r6, t[2]);
        F::butterfly::<false>(r5, r7, t[2]);
        F::butterfly::<false>(r0, r1, t[3]);
        F::butterfly::<false>(r2, r3, t[4]);
        F::butterfly::<false>(r4, r5, t[5]);
        F::butterfly::<false>(r6, r7, t[6]);
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
impl_butterfly_field!(plain_butterfly: Gf2, BinaryField2, BinaryField4, Ghash128, Rijndael8b);

impl ButterflyField for Poly64 {
    const FUSE_RADIX8: bool = cfg!(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    ));

    #[inline]
    fn butterfly<const INVERSE: bool>(lo: &mut [Self], hi: &mut [Self], t: Self) {
        poly64::butterfly::<INVERSE>(lo, hi, t);
    }

    #[inline]
    fn butterfly_radix8<const INVERSE: bool>(rows: &mut [&mut [Self]; 8], t: &[Self; 7]) {
        poly64::radix8::<INVERSE>(rows, t);
    }
}

// The widest byte-aligned level, whose network runs in the polynomial basis when enough stages hold a twiddle wider than the byte map covers.
//
// - A tower product by such a twiddle changes basis three times around one carryless product.
// - Changing the whole buffer once each way costs two changes per element instead.
// - The byte map scales by a twiddle it covers on every run with no change of basis, so a transform with only those, or too few stages of the others, stays put.
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
        if HAS_HARDWARE_CLMUL && wide_stages_repay_conversion(width, log_n, shift) {
            ghash_transform::<INVERSE>(values, width, shift);
        } else {
            lch::transform::<Self, INVERSE>(values, width, shift);
        }
    }

    fn lch_transform_cosets(values: &mut [Self], width: usize, log_message: usize) {
        // Each coset's shift is a domain point below the height, so the subspace the height spans holds every coset.
        let log_n = log2_ceil_usize(values.len() / width);
        if HAS_HARDWARE_CLMUL && wide_stages_repay_conversion(width, log_n, Self::ZERO) {
            ghash_transform_cosets(values, width, log_message);
        } else {
            lch::transform_cosets::<Self>(values, width, log_message);
        }
    }
}

/// Rows shorter than this many bytes repay the two changes of basis with fewer stages of wide twiddles.
///
/// - Measured on Sapphire Rapids, Zen 5 and Graviton4, running the network over `Ghash128` against the tower one.
/// - On shorter rows the `Ghash128` route wins once one stage holds a wide twiddle.
/// - On rows of this size and up it needs two.
const SHORT_ROW_BYTES: usize = 64;

/// Whether enough stages of the transform over `shift + S_l`, on rows of `width`, hold a twiddle past the byte map to repay the two changes of basis.
///
/// - Each twiddle of stage `j` is `W_j(shift)` plus a point of `S_(l - j)`, with `l = log_n`.
/// - The first `2^t` Cantor basis vectors span the tower subfield of `2^t` bits, which each `W_j` maps into itself.
/// - So a shift within the byte map's subfield leaves wide twiddles in the first `l - bits` stages only.
/// - A shift past it keeps stage `j` wide while `W_j(shift)` is.
/// - That is every stage for a shift of full width and fewer for one just past the byte map, so counting all of them overcounts only the latter.
#[inline]
fn wide_stages_repay_conversion(width: usize, log_n: usize, shift: BinaryField128) -> bool {
    let row_bytes = width * size_of::<BinaryField128>();
    let bits = byte_map_twiddle_bits(row_bytes);
    let wide_stages = if shift.to_repr() >> bits != 0 {
        log_n
    } else {
        log_n.saturating_sub(bits)
    };
    let needed = if row_bytes < SHORT_ROW_BYTES { 1 } else { 2 };
    wide_stages >= needed
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
        Ghash128, Poly64, byte_map_twiddle_bits, wide_stages_repay_conversion,
    };

    #[test]
    fn radix8_zero_corners_and_tails_match_separate_stages() {
        for width in [0, 1, 2, 3, 7, 8, 15, 16, 17] {
            for corner in [0, 1, 1 << 63, u64::MAX] {
                let t = core::array::from_fn(|i| Poly64::new(if i % 2 == 0 { corner } else { 0 }));
                let input: [Vec<Poly64>; 8] = core::array::from_fn(|r| {
                    (0..width)
                        .map(|lane| Poly64::new((r * width + lane) as u64 ^ corner))
                        .collect()
                });
                for inverse in [false, true] {
                    let mut got = input.clone();
                    let mut want = input.clone();
                    if inverse {
                        Poly64::butterfly_radix8::<true>(
                            &mut got.each_mut().map(Vec::as_mut_slice),
                            &t,
                        );
                        super::radix8::<Poly64, true>(
                            &mut want.each_mut().map(Vec::as_mut_slice),
                            &t,
                        );
                    } else {
                        Poly64::butterfly_radix8::<false>(
                            &mut got.each_mut().map(Vec::as_mut_slice),
                            &t,
                        );
                        super::radix8::<Poly64, false>(
                            &mut want.each_mut().map(Vec::as_mut_slice),
                            &t,
                        );
                    }
                    assert_eq!(
                        got, want,
                        "width={width}, corner={corner}, inverse={inverse}"
                    );
                }
                let mut roundtrip = input.clone();
                Poly64::butterfly_radix8::<false>(
                    &mut roundtrip.each_mut().map(Vec::as_mut_slice),
                    &t,
                );
                Poly64::butterfly_radix8::<true>(
                    &mut roundtrip.each_mut().map(Vec::as_mut_slice),
                    &t,
                );
                assert_eq!(roundtrip, input);
            }
        }
    }

    #[test]
    #[should_panic(expected = "radix-8 row lengths differ")]
    fn radix8_rejects_mismatched_rows() {
        let mut input: [Vec<Poly64>; 8] = core::array::from_fn(|i| alloc::vec![Poly64::ZERO; i]);
        Poly64::butterfly_radix8::<false>(
            &mut input.each_mut().map(Vec::as_mut_slice),
            &[Poly64::ZERO; 7],
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        #[test]
        fn radix8_random_lanes_match_separate_stages(
            raw in any::<[[u64; 17];8]>(), t in any::<[u64;7]>(), width in 0usize..=17
        ) {
            let t=t.map(Poly64::new);
            let input: [Vec<Poly64>;8]=raw.map(|row|row[..width].iter().copied().map(Poly64::new).collect());
            let mut got=input.clone();
            let mut want=input.clone();
            Poly64::butterfly_radix8::<false>(&mut got.each_mut().map(Vec::as_mut_slice), &t);
            super::radix8::<Poly64, false>(&mut want.each_mut().map(Vec::as_mut_slice), &t);
            prop_assert_eq!(&got,&want);
            Poly64::butterfly_radix8::<true>(&mut got.each_mut().map(Vec::as_mut_slice), &t);
            prop_assert_eq!(&got,&input);
        }
    }

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

    /// The `Ghash128` route waits for two stages of wide twiddles on rows of 64 bytes and up, and for one below that.
    ///
    /// A single wide stage on such rows leaves the route slower than the tower network, so taking it there would regress those shapes.
    #[test]
    fn rows_of_64_bytes_and_up_need_two_wide_stages_for_the_ghash_route() {
        // A shift of full width widens every stage, and the zero shift widens only the stages past the byte map's span.
        let full = BinaryField128::from_repr(1 << 127);

        // Fixture: rows of one and three elements fall short of 64 bytes, rows of four and sixteen reach it.
        for (width, needed) in [(1, 1), (3, 1), (4, 2), (16, 2)] {
            let bits = byte_map_twiddle_bits(width * size_of::<BinaryField128>());
            for stages in 0..=3 {
                let want = stages >= needed;
                assert_eq!(
                    wide_stages_repay_conversion(width, bits + stages, BinaryField128::ZERO),
                    want,
                    "width={width}, unshifted, wide stages={stages}"
                );
                assert_eq!(
                    wide_stages_repay_conversion(width, stages, full),
                    want,
                    "width={width}, shifted, wide stages={stages}"
                );
            }
        }
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
