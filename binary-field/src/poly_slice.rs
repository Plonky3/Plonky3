//! Whole-slice kernels for the polynomial basis: one scaling and two butterflies.
//!
//! A prefix runs packed wherever the carryless multiply reaches several 128-bit lanes.
//!
//! Everything else runs one element at a time.

use crate::clmul;

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
