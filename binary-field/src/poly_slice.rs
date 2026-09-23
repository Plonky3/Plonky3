//! Whole-slice kernels for the polynomial basis: one scaling and two butterflies.
//!
//! A prefix runs packed wherever the carryless multiply reaches several 128-bit lanes.
//!
//! Everything else runs one element at a time.
//!
//! The lowest stages of a single-column transform have their own kernel, which runs them
//! together over short runs of the column.

use crate::clmul;
use crate::poly_basis::LowStageTwiddles;

/// Stages the low-stage kernel runs together, over runs of `2^LOW_STAGES` elements.
///
/// Four stages span sixteen elements, which is four registers of the widest packing.
pub(crate) const LOW_STAGES: usize = 4;

/// Elements in one run of the low-stage kernel.
const LOW_RUN: usize = 1 << LOW_STAGES;

/// Run the lowest [`LOW_STAGES`] stages of a forward transform over whole runs of the slice.
///
/// The slice starts at run `first_run` of the transform, and its length is a whole number of
/// runs, which the caller asserts.
#[inline]
pub(crate) fn forward_low_stages(
    values: &mut [u128],
    first_run: usize,
    twiddles: &LowStageTwiddles,
) {
    low::forward(values, first_run, twiddles);
}

/// Multiply every element of a slice by the same scalar.
#[inline]
pub(crate) fn scale(values: &mut [u128], scalar: u128) {
    // Below one register there is nothing to pack, so the call boundary buys nothing.
    //
    // Otherwise the packed kernel reports how many leading elements it covered.
    let packed = if values.len() < wide::WIDTH {
        0
    } else {
        wide::scale(values, scalar)
    };

    // Whatever is left over costs one carryless multiply per element.
    for value in &mut values[packed..] {
        *value = clmul::poly_mul_128_batch(*value, scalar);
    }
}

/// Send `(lo, hi)` to `(lo + scalar*hi, lo + (scalar + 1)*hi)`, element by element.
#[inline]
pub(crate) fn butterfly_forward(lo: &mut [u128], hi: &mut [u128], scalar: u128) {
    // Invariant: the two halves are paired element for element.
    //
    // Both kernels below stop at the shorter side, so a mismatch would silently drop work.
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");

    let packed = if lo.len() < wide::WIDTH {
        0
    } else {
        wide::butterfly_forward(lo, hi, scalar)
    };

    for (lo, hi) in lo[packed..].iter_mut().zip(&mut hi[packed..]) {
        // The scaled upper half lands in the lower one first.
        *lo ^= clmul::poly_mul_128_batch(scalar, *hi);
        // The upper half then carries the sum of both, so the two differ by its old value.
        *hi ^= *lo;
    }
}

/// Undo [`butterfly_forward`] with the same scalar.
#[inline]
pub(crate) fn butterfly_inverse(lo: &mut [u128], hi: &mut [u128], scalar: u128) {
    // Invariant: the two halves are paired element for element.
    //
    // Both kernels below stop at the shorter side, so a mismatch would silently drop work.
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");

    let packed = if lo.len() < wide::WIDTH {
        0
    } else {
        wide::butterfly_inverse(lo, hi, scalar)
    };

    for (lo, hi) in lo[packed..].iter_mut().zip(&mut hi[packed..]) {
        // Undo the second step of the forward pass, recovering the old upper half.
        *hi ^= *lo;
        // Then take the scaled upper half back out of the lower one.
        *lo ^= clmul::poly_mul_128_batch(scalar, *hi);
    }
}

/// The packed prefix, over the widest carryless multiply the target has.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
mod wide {
    use crate::packed::lanes;
    pub(super) use crate::packed::lanes::WIDTH;
    use crate::packed::split::SplitScalar;

    /// Scale a prefix of the slice, returning the number of elements it covered.
    ///
    /// Out of line so that the dispatch above stays small enough to inline into its callers.
    ///
    /// That is what keeps the shortcut for a lone element as cheap as a bare product.
    #[inline(never)]
    pub(super) fn scale(values: &mut [u128], scalar: u128) -> usize {
        // Splitting into fixed-size blocks hands the tail back and needs no bounds check.
        let (blocks, _) = values.as_chunks_mut::<WIDTH>();

        let split = SplitScalar::new(scalar);

        for block in blocks.iter_mut() {
            // SAFETY: a block holds one 128-bit integer per lane, so it is one whole register.
            //
            // Both accesses are the unaligned forms, so the alignment of the slice is free.
            //
            // The target features the instructions need gate this module.
            unsafe {
                let v = lanes::load(block.as_ptr());
                lanes::store(block.as_mut_ptr(), split.apply(v));
            }
        }
        blocks.len() * WIDTH
    }

    /// Apply the forward butterfly to a prefix, returning the number of elements it covered.
    ///
    /// Out of line for the same reason as the scaling kernel above.
    ///
    /// The two halves must be the same length, which every caller asserts before arriving.
    #[inline(never)]
    pub(super) fn butterfly_forward(lo: &mut [u128], hi: &mut [u128], scalar: u128) -> usize {
        let (lo, hi) = (lo.as_chunks_mut::<WIDTH>().0, hi.as_chunks_mut::<WIDTH>().0);

        // The dispatcher asserts the two halves match, so either chunk count is the answer.
        let blocks = lo.len();
        let split = SplitScalar::new(scalar);

        for (lo, hi) in lo.iter_mut().zip(hi.iter_mut()) {
            // SAFETY: as in the scaling kernel above, for two blocks from two distinct slices.
            unsafe {
                let a = lanes::load(lo.as_ptr());
                let b = lanes::load(hi.as_ptr());

                // The lower half absorbs the scaled upper one.
                let a = lanes::xor(a, split.apply(b));
                lanes::store(lo.as_mut_ptr(), a);

                // The upper half then carries the sum of both.
                lanes::store(hi.as_mut_ptr(), lanes::xor(b, a));
            }
        }
        blocks * WIDTH
    }

    /// Apply the inverse butterfly to a prefix, returning the number of elements it covered.
    ///
    /// Out of line for the same reason as the scaling kernel above.
    ///
    /// The two halves must be the same length, which every caller asserts before arriving.
    #[inline(never)]
    pub(super) fn butterfly_inverse(lo: &mut [u128], hi: &mut [u128], scalar: u128) -> usize {
        let (lo, hi) = (lo.as_chunks_mut::<WIDTH>().0, hi.as_chunks_mut::<WIDTH>().0);

        // The dispatcher asserts the two halves match, so either chunk count is the answer.
        let blocks = lo.len();
        let split = SplitScalar::new(scalar);

        for (lo, hi) in lo.iter_mut().zip(hi.iter_mut()) {
            // SAFETY: as in the forward kernel above.
            unsafe {
                let a = lanes::load(lo.as_ptr());
                let b = lanes::load(hi.as_ptr());

                // Recover the old upper half, which the forward pass added the lower one to.
                let b = lanes::xor(b, a);
                lanes::store(hi.as_mut_ptr(), b);

                // Then take the scaled upper half back out of the lower one.
                lanes::store(lo.as_mut_ptr(), lanes::xor(a, split.apply(b)));
            }
        }
        blocks * WIDTH
    }
}

/// The low stages of a run, with its four registers held through all of them.
///
/// # Algorithm
///
/// Register `r` holds elements `4r .. 4r + 4` of the run. Stages 3 and 2 pair whole registers,
/// eight and four elements apart, and every pair shares one twiddle:
///
/// ```text
///     stage 3   (R0, R2)  (R1, R3)          block 0 of the run
///     stage 2   (R0, R1)                    block 0
///               (R2, R3)                    block 1
/// ```
///
/// Stages 1 and 0 pair lanes of one register, so each pair of registers is interleaved until
/// the partners sit in the same lane of two registers, one block per lane:
///
/// ```text
///     a = R0, b = R1        lower half               upper half
///     stage 1               a0 a1 b0 b1   blocks 0 0 1 1    a2 a3 b2 b3
///     stage 0               a0 a2 b0 b2   blocks 0 1 2 3    a1 a3 b1 b3
/// ```
///
/// Undoing both interleaves in reverse order puts the elements back in their own lanes.
/// The second pair of registers is the same, with blocks two and four further on.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    target_feature = "avx512f"
))]
mod low {
    use super::LOW_RUN;
    use crate::packed::lanes::{self, Reg, WIDTH};
    use crate::packed::split::SplitScalar;
    use crate::poly_basis::LowStageTwiddles;

    // A run is exactly four registers, which the register assignment above relies on.
    const _: () = assert!(LOW_RUN == 4 * WIDTH);

    /// A value and its companion as two registers, each the same in every lane.
    #[inline(always)]
    fn broadcast(pair: [u128; 2]) -> [Reg; 2] {
        [lanes::broadcast(pair[0]), lanes::broadcast(pair[1])]
    }

    /// Four values and their companions, one per lane, as two registers.
    #[inline(always)]
    fn per_lane(pairs: [[u128; 2]; WIDTH]) -> [Reg; 2] {
        let values = pairs.map(|pair| pair[0]);
        let companions = pairs.map(|pair| pair[1]);
        // SAFETY: each array is one register of `u128` lanes, read unaligned.
        unsafe {
            [
                lanes::load(values.as_ptr()),
                lanes::load(companions.as_ptr()),
            ]
        }
    }

    /// The lane-by-lane sum of two values held beside their companions.
    #[inline(always)]
    fn add(a: [Reg; 2], b: [Reg; 2]) -> [Reg; 2] {
        [lanes::xor(a[0], b[0]), lanes::xor(a[1], b[1])]
    }

    /// The multiplier a value and its companion describe.
    #[inline(always)]
    const fn twiddle(pair: [Reg; 2]) -> SplitScalar<Reg> {
        SplitScalar::from_parts(pair[0], pair[1])
    }

    /// `(lo, hi) -> (lo + t·hi, lo + (t + 1)·hi)`, lane by lane.
    #[inline(always)]
    fn butterfly(lo: &mut Reg, hi: &mut Reg, twiddle: SplitScalar<Reg>) {
        *lo = lanes::xor(*lo, twiddle.apply(*hi));
        *hi = lanes::xor(*hi, *lo);
    }

    /// Stages 1 and 0 of one pair of registers, each twiddle laid out lane by lane as above.
    #[inline(always)]
    fn lane_stages(
        a: &mut Reg,
        b: &mut Reg,
        twiddle_1: SplitScalar<Reg>,
        twiddle_0: SplitScalar<Reg>,
    ) {
        let (mut lo, mut hi) = lanes::interleave(*a, *b, 2);
        butterfly(&mut lo, &mut hi, twiddle_1);
        let (mut lo, mut hi) = lanes::interleave(lo, hi, 1);
        butterfly(&mut lo, &mut hi, twiddle_0);
        let (lo, hi) = lanes::interleave(lo, hi, 1);
        (*a, *b) = lanes::interleave(lo, hi, 2);
    }

    /// Runs the low stages over every whole run of the slice.
    #[inline(never)]
    pub(super) fn forward(values: &mut [u128], first_run: usize, twiddles: &LowStageTwiddles) {
        let (runs, _) = values.as_chunks_mut::<LOW_RUN>();

        // Within a run, stage `j` offsets its twiddle by the span of its block index there.
        let span = |block: usize| twiddles.span(block, 0);
        // Stage 2, second pair of registers: block 1.
        let second_block = broadcast(span(1));
        // Stage 1, lanes as interleaved above: blocks 0 0 1 1, then 2 2 3 3.
        let offsets_1 = [
            per_lane([span(0), span(0), span(1), span(1)]),
            per_lane([span(2), span(2), span(3), span(3)]),
        ];
        // Stage 0: blocks 0 1 2 3, then 4 5 6 7.
        let offsets_0 = [
            per_lane([span(0), span(1), span(2), span(3)]),
            per_lane([span(4), span(5), span(6), span(7)]),
        ];

        // The twiddles walk from run to run in registers, each stage's the same in every lane.
        let mut run_twiddles = twiddles.run_twiddles(first_run).map(broadcast);
        for (index, run) in runs.iter_mut().enumerate() {
            if index != 0 {
                let step = twiddles.step(first_run + index - 1);
                for (twiddle, step) in run_twiddles.iter_mut().zip(step) {
                    *twiddle = add(*twiddle, broadcast(*step));
                }
            }
            let [t0, t1, t2, t3] = run_twiddles;

            // SAFETY: a run is four whole registers, read and written unaligned. The target
            // features the instructions need gate this module.
            unsafe {
                let base = run.as_mut_ptr();
                let [mut r0, mut r1, mut r2, mut r3] =
                    core::array::from_fn(|r| lanes::load(base.add(r * WIDTH)));

                let twiddle_3 = twiddle(t3);
                butterfly(&mut r0, &mut r2, twiddle_3);
                butterfly(&mut r1, &mut r3, twiddle_3);

                butterfly(&mut r0, &mut r1, twiddle(t2));
                butterfly(&mut r2, &mut r3, twiddle(add(t2, second_block)));

                lane_stages(
                    &mut r0,
                    &mut r1,
                    twiddle(add(t1, offsets_1[0])),
                    twiddle(add(t0, offsets_0[0])),
                );
                lane_stages(
                    &mut r2,
                    &mut r3,
                    twiddle(add(t1, offsets_1[1])),
                    twiddle(add(t0, offsets_0[1])),
                );

                for (r, register) in [r0, r1, r2, r3].into_iter().enumerate() {
                    lanes::store(base.add(r * WIDTH), register);
                }
            }
        }
    }
}

/// The low stages one at a time within each run, through the slice butterflies.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    target_feature = "avx512f"
)))]
mod low {
    use super::{LOW_RUN, LOW_STAGES};
    use crate::poly_basis::LowStageTwiddles;

    pub(super) fn forward(values: &mut [u128], first_run: usize, twiddles: &LowStageTwiddles) {
        // The scalar multiply takes the twiddle alone, so the companions stay behind.
        let mut run_twiddles = twiddles.run_twiddles(first_run).map(|pair| pair[0]);
        // Blocks within a run always use the same basis offsets.
        let offsets: [u128; 8] = core::array::from_fn(|block| twiddles.span(block, 0)[0]);
        for (index, run) in values.as_chunks_mut::<LOW_RUN>().0.iter_mut().enumerate() {
            if index != 0 {
                let step = twiddles.step(first_run + index - 1);
                for (twiddle, step) in run_twiddles.iter_mut().zip(step) {
                    *twiddle ^= step[0];
                }
            }
            for j in (0..LOW_STAGES).rev() {
                let half = 1 << j;
                for (block, pair) in run.chunks_exact_mut(2 * half).enumerate() {
                    let (lo, hi) = pair.split_at_mut(half);
                    let twiddle = run_twiddles[j] ^ offsets[block];
                    super::butterfly_forward(lo, hi, twiddle);
                }
            }
        }
    }
}

/// No carryless multiply reaches past one element here, so no prefix is packed.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
)))]
mod wide {
    /// A width no slice can reach, so every dispatch above takes the scalar path outright.
    pub(super) const WIDTH: usize = usize::MAX;

    /// Reports that no element was scaled.
    #[inline]
    pub(super) const fn scale(_values: &mut [u128], _scalar: u128) -> usize {
        0
    }

    /// Reports that no pair took the forward butterfly.
    #[inline]
    pub(super) const fn butterfly_forward(
        _lo: &mut [u128],
        _hi: &mut [u128],
        _scalar: u128,
    ) -> usize {
        0
    }

    /// Reports that no pair took the inverse butterfly.
    #[inline]
    pub(super) const fn butterfly_inverse(
        _lo: &mut [u128],
        _hi: &mut [u128],
        _scalar: u128,
    ) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use crate::clmul;

    /// The multipliers a random search is unlikely to reach.
    ///
    /// Each one drives the split of the multiplier, or the fold of the modulus, to an extreme:
    ///
    /// ```text
    ///     0                 the whole product vanishes
    ///     1                 the identity, so the slice must come back unchanged
    ///     all ones          every coefficient of every half product is live
    ///     x^127             the highest degree, so the companion spills furthest
    ///     x^64              the companion is the multiplier's own shift, already reduced
    ///     lower half only   the companion is the only source of upper limbs
    ///     0x87              the modulus tail itself
    /// ```
    const CORNERS: [u128; 7] = [0, 1, u128::MAX, 1 << 127, 1 << 64, (1u128 << 64) - 1, 0x87];

    /// Whether this build is one the packed prefix is compiled for.
    ///
    /// Stated a second time here, independently of the gate on the kernels themselves.
    ///
    /// A gate that drifts then shows up as a failure rather than as a silent scalar fallback.
    const PACKED_BUILD: bool = cfg!(all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    ));

    /// Elements per register on the widest packing this crate has.
    ///
    /// The build assertion below forces a wider one to bump this rather than pass silently.
    const WIDEST_LANES: usize = 4;

    /// Guard elements on each side of the payload, two of the widest registers.
    ///
    /// One register wide would catch only a store that overran by exactly one register.
    ///
    /// Two make the next one out detectable as well, still inside the buffer's own allocation.
    const SENTINELS: usize = 2 * WIDEST_LANES;

    // The gate on the packed prefix has to agree with the target predicate restated above.
    //
    // Drift in either direction leaves the fallback's sentinel width behind, so checking the
    // width here turns it into a build failure instead of waiting for someone to run this leg.
    const _: () = assert!(if PACKED_BUILD {
        super::wide::WIDTH == 2 || super::wide::WIDTH == WIDEST_LANES
    } else {
        super::wide::WIDTH == usize::MAX
    });

    /// What those elements hold, which no product of the inputs below can reproduce.
    const SENTINEL: u128 = 0x5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a;

    /// A slice whose elements share no structure with one another.
    fn sample(len: usize) -> Vec<u128> {
        (0..len)
            .map(|i| (i as u128 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835))
            .collect()
    }

    /// The same values, walled off from the rest of the allocation on both sides.
    ///
    /// Both walls are sentinels, so an overrun in either direction shows as a changed guard.
    ///
    /// ```text
    ///     [ guard | offset | values | guard ]
    ///                ^ slides the payload across every alignment one register can see
    /// ```
    fn padded(values: &[u128], offset: usize) -> Vec<u128> {
        let lead = SENTINELS + offset;

        let mut buffer = Vec::with_capacity(lead + values.len() + SENTINELS);
        buffer.extend(core::iter::repeat_n(SENTINEL, lead));
        buffer.extend_from_slice(values);
        buffer.extend(core::iter::repeat_n(SENTINEL, SENTINELS));
        buffer
    }

    /// The payload of such a buffer, as the kernels see it.
    fn payload_mut(buffer: &mut [u128], offset: usize, len: usize) -> &mut [u128] {
        let start = SENTINELS + offset;
        &mut buffer[start..start + len]
    }

    /// Both walls of such a buffer, still holding the guard value.
    fn walls_intact(buffer: &[u128], offset: usize, len: usize) -> bool {
        let start = SENTINELS + offset;

        // Nothing may have run off the front, and nothing off the back either.
        buffer[..start].iter().all(|&v| v == SENTINEL)
            && buffer[start + len..].iter().all(|&v| v == SENTINEL)
    }

    /// The payload of such a buffer, once both walls are confirmed intact.
    fn payload(buffer: &[u128], offset: usize, len: usize) -> Result<&[u128], TestCaseError> {
        prop_assert!(walls_intact(buffer, offset, len));

        let start = SENTINELS + offset;
        Ok(&buffer[start..start + len])
    }

    /// Scaling, one element at a time through the scalar backend.
    fn reference_scale(values: &[u128], scalar: u128) -> Vec<u128> {
        values
            .iter()
            .map(|&v| clmul::poly_mul_128(v, scalar))
            .collect()
    }

    /// The forward butterfly, one element at a time through the scalar backend.
    fn reference_forward(lo: &[u128], hi: &[u128], scalar: u128) -> (Vec<u128>, Vec<u128>) {
        let mut lo = lo.to_vec();
        let mut hi = hi.to_vec();
        for (lo, hi) in lo.iter_mut().zip(&mut hi) {
            *lo ^= clmul::poly_mul_128(scalar, *hi);
            *hi ^= *lo;
        }
        (lo, hi)
    }

    /// The inverse butterfly, one element at a time through the scalar backend.
    fn reference_inverse(lo: &[u128], hi: &[u128], scalar: u128) -> (Vec<u128>, Vec<u128>) {
        let mut lo = lo.to_vec();
        let mut hi = hi.to_vec();
        for (lo, hi) in lo.iter_mut().zip(&mut hi) {
            *hi ^= *lo;
            *lo ^= clmul::poly_mul_128(scalar, *hi);
        }
        (lo, hi)
    }

    /// Drive all three kernels at one length, offset and multiplier against the scalar loops.
    ///
    /// Reports failure rather than panicking, so the sweep and the random cases can share it.
    fn kernels_agree(values: &[u128], scalar: u128, offset: usize) -> Result<(), TestCaseError> {
        let len = values.len();

        // The second operand of the butterflies, reversed so the two slices differ.
        let other: Vec<u128> = values.iter().rev().copied().collect();

        // Scaling: the packed prefix and the scalar tail must together match the scalar loop.
        let mut scaled = padded(values, offset);
        super::scale(payload_mut(&mut scaled, offset, len), scalar);
        prop_assert_eq!(
            payload(&scaled, offset, len)?,
            &reference_scale(values, scalar)[..]
        );

        // Forward butterfly over two distinct slices, both at the same offset.
        let mut lo = padded(values, offset);
        let mut hi = padded(&other, offset);
        super::butterfly_forward(
            payload_mut(&mut lo, offset, len),
            payload_mut(&mut hi, offset, len),
            scalar,
        );
        let (want_lo, want_hi) = reference_forward(values, &other, scalar);
        prop_assert_eq!(payload(&lo, offset, len)?, &want_lo[..]);
        prop_assert_eq!(payload(&hi, offset, len)?, &want_hi[..]);

        // The inverse kernel applied to the forward output must return the input.
        super::butterfly_inverse(
            payload_mut(&mut lo, offset, len),
            payload_mut(&mut hi, offset, len),
            scalar,
        );
        prop_assert_eq!(payload(&lo, offset, len)?, values);
        prop_assert_eq!(payload(&hi, offset, len)?, &other[..]);

        // The inverse kernel on its own, against its own scalar loop.
        let mut lo = padded(values, offset);
        let mut hi = padded(&other, offset);
        super::butterfly_inverse(
            payload_mut(&mut lo, offset, len),
            payload_mut(&mut hi, offset, len),
            scalar,
        );
        let (want_lo, want_hi) = reference_inverse(values, &other, scalar);
        prop_assert_eq!(payload(&lo, offset, len)?, &want_lo[..]);
        prop_assert_eq!(payload(&hi, offset, len)?, &want_hi[..]);

        Ok(())
    }

    #[test]
    fn the_packed_prefix_and_the_scalar_tail_agree_at_every_short_length() {
        // Invariant: a register holds two or four elements depending on the build.
        //
        // An unbroken run of short lengths covers every combination either way.
        //
        //     len 0..1    below one register on both builds, so nothing is packed
        //     len 2..3    one register on the narrow build, nothing on the wide one
        //     len 4       exactly one register on the wide build, two on the narrow one
        //     len 5..7    at least one register plus a tail of every possible size
        //     len 16,17   several whole registers, with and without a tail
        for len in 0..=17 {
            let values = sample(len);
            for scalar in CORNERS {
                // Offsets 0 through 3 place the payload at every alignment one register can see.
                for offset in 0..4 {
                    kernels_agree(&values, scalar, offset)
                        .unwrap_or_else(|e| panic!("len {len}, scalar {scalar:#x}: {e}"));
                }
            }
        }
    }

    #[test]
    fn the_packed_prefix_covers_every_whole_register() {
        // Invariant: the kernels report how many leading elements they covered.
        //
        // The dispatchers use that count only as a tail offset.
        //
        // So a build that lost the packed path still returns right answers, silently.
        //
        // Checking the reported count is what turns that into a failure.
        //
        // The width itself is pinned at build time by the assertion near the guard constants.
        if !PACKED_BUILD {
            return;
        }

        let width = super::wide::WIDTH;

        // Every length from an empty prefix to several whole registers plus a tail.
        for len in 0..4 * width {
            // Whole registers only: the remainder is the scalar tail's business.
            let want = len - len % width;

            let values = sample(len);

            // The second operand of the butterflies, reversed so the two slices differ.
            let other: Vec<u128> = values.iter().rev().copied().collect();

            // Walled on both sides, so a store past either end fails here rather than
            // corrupting the allocator the way a bare sample would.
            let mut scaled = padded(&values, 0);
            let covered = super::wide::scale(payload_mut(&mut scaled, 0, len), 3);
            assert_eq!(covered, want, "scale at len {len}");
            assert!(walls_intact(&scaled, 0, len), "scale overran at len {len}");

            let mut lo = padded(&values, 0);
            let mut hi = padded(&other, 0);

            let covered = super::wide::butterfly_forward(
                payload_mut(&mut lo, 0, len),
                payload_mut(&mut hi, 0, len),
                3,
            );
            assert_eq!(covered, want, "forward butterfly at len {len}");
            assert!(walls_intact(&lo, 0, len), "forward overran lo at len {len}");
            assert!(walls_intact(&hi, 0, len), "forward overran hi at len {len}");

            let covered = super::wide::butterfly_inverse(
                payload_mut(&mut lo, 0, len),
                payload_mut(&mut hi, 0, len),
                3,
            );
            assert_eq!(covered, want, "inverse butterfly at len {len}");
            assert!(walls_intact(&lo, 0, len), "inverse overran lo at len {len}");
            assert!(walls_intact(&hi, 0, len), "inverse overran hi at len {len}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn the_kernels_agree_with_the_scalar_backend(
            values in prop::collection::vec(any::<u128>(), 0..40),
            scalar in any::<u128>(),
            offset in 0usize..4,
        ) {
            // Random slices, spanning several registers plus a tail of every residue.
            kernels_agree(&values, scalar, offset)?;
        }

        #[test]
        fn a_zero_multiplier_degenerates(values in prop::collection::vec(any::<u128>(), 0..40)) {
            // A zero multiplier makes every product vanish, so scaling clears the slice.
            let mut scaled = values.clone();
            super::scale(&mut scaled, 0);
            prop_assert!(scaled.iter().all(|&v| v == 0));

            // The butterfly then reduces to adding the lower half into the upper one.
            let mut lo = values.clone();
            let mut hi: Vec<u128> = values.iter().rev().copied().collect();
            let expected: Vec<u128> = hi.iter().zip(&lo).map(|(h, l)| h ^ l).collect();
            super::butterfly_forward(&mut lo, &mut hi, 0);
            prop_assert_eq!(&lo, &values);
            prop_assert_eq!(&hi, &expected);
        }
    }
}
