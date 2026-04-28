//! Shared utilities for Goldilocks NEON assembly.

use core::arch::asm;

use super::packing::PackedGoldilocksNeon;
use crate::{Goldilocks, P};

const EPSILON: u64 = P.wrapping_neg(); // 2^32 - 1

// ---------------------------------------------------------------------------
// Scalar field arithmetic (inline assembly)
// ---------------------------------------------------------------------------

/// Multiply two Goldilocks elements using inline assembly.
///
/// Computes `a * b mod P` where P = 2^64 - 2^32 + 1. The reduction
/// uses the identity `2^64 = 2^32 - 1 (mod P)` (i.e. EPSILON) to fold
/// the 128-bit product back into a single limb.
#[inline(always)]
pub(super) unsafe fn mul_asm(a: u64, b: u64) -> u64 {
    let _lo: u64;
    let _hi: u64;
    let _t0: u64;
    let _t1: u64;
    let _t2: u64;
    let result: u64;

    unsafe {
        asm!(
            // Compute 128-bit product: hi:lo = a * b
            "mul   {lo}, {a}, {b}",
            "umulh {hi}, {a}, {b}",

            // Reduce: result = lo - hi_hi + hi_lo * EPSILON
            // where hi = hi_hi * 2^32 + hi_lo

            // t0 = lo - (hi >> 32), with borrow detection
            "lsr   {t0}, {hi}, #32",          // t0 = hi >> 32
            "subs  {t1}, {lo}, {t0}",         // t1 = lo - t0, set flags
            "csetm {t2:w}, cc",               // t2 = -1 if borrow, 0 otherwise
            "sub   {t1}, {t1}, {t2}",         // Adjust for borrow (subtract EPSILON)

            // t0 = (hi & EPSILON) * EPSILON
            "and   {t0}, {hi}, {epsilon}",    // t0 = hi & EPSILON
            "mul   {t0}, {t0}, {epsilon}",    // t0 = t0 * EPSILON

            // result = t1 + t0, with overflow detection
            "adds  {result}, {t1}, {t0}",     // result = t1 + t0, set flags
            "csetm {t2:w}, cs",               // t2 = -1 if carry, 0 otherwise
            "add   {result}, {result}, {t2}", // Add EPSILON on overflow

            a = in(reg) a,
            b = in(reg) b,
            epsilon = in(reg) EPSILON,
            lo = out(reg) _lo,
            hi = out(reg) _hi,
            t0 = out(reg) _t0,
            t1 = out(reg) _t1,
            t2 = out(reg) _t2,
            result = out(reg) result,
            options(pure, nomem, nostack),
        );
    }

    result
}

/// Compute `a * b + c` in the Goldilocks field using inline assembly.
///
/// Fused multiply-add: forms the 128-bit product `a * b`, adds `c` into
/// the low limb (with carry propagation), then reduces modulo P.
#[inline(always)]
pub(super) unsafe fn mul_add_asm(a: u64, b: u64, c: u64) -> u64 {
    let _lo: u64;
    let _hi: u64;
    let _t0: u64;
    let _t1: u64;
    let _t2: u64;
    let result: u64;

    unsafe {
        asm!(
            // Compute 128-bit product: hi:lo = a * b
            "mul   {lo}, {a}, {b}",
            "umulh {hi}, {a}, {b}",

            // Accumulate c into the 128-bit product: hi:lo = hi:lo + c
            "adds  {lo}, {lo}, {c}",
            "adc   {hi}, {hi}, xzr",

            // Reduce: result = lo - hi_hi + hi_lo * EPSILON
            // where hi = hi_hi * 2^32 + hi_lo

            // t0 = lo - (hi >> 32), with borrow detection
            "lsr   {t0}, {hi}, #32",          // t0 = hi >> 32
            "subs  {t1}, {lo}, {t0}",         // t1 = lo - t0, set flags
            "csetm {t2:w}, cc",               // t2 = -1 if borrow, 0 otherwise
            "sub   {t1}, {t1}, {t2}",         // Adjust for borrow (subtract EPSILON)

            // t0 = (hi & EPSILON) * EPSILON
            "and   {t0}, {hi}, {epsilon}",    // t0 = hi & EPSILON
            "mul   {t0}, {t0}, {epsilon}",    // t0 = t0 * EPSILON

            // result = t1 + t0, with overflow detection
            "adds  {result}, {t1}, {t0}",     // result = t1 + t0, set flags
            "csetm {t2:w}, cs",               // t2 = -1 if carry, 0 otherwise
            "add   {result}, {result}, {t2}", // Add EPSILON on overflow

            a = in(reg) a,
            b = in(reg) b,
            c = in(reg) c,
            epsilon = in(reg) EPSILON,
            lo = out(reg) _lo,
            hi = out(reg) _hi,
            t0 = out(reg) _t0,
            t1 = out(reg) _t1,
            t2 = out(reg) _t2,
            result = out(reg) result,
            options(pure, nomem, nostack),
        );
    }

    result
}

/// Add two Goldilocks elements with overflow handling using inline assembly.
///
/// Computes `a + b mod P`, accepting non-canonical inputs.
#[inline(always)]
pub(super) unsafe fn add_asm(a: u64, b: u64) -> u64 {
    let result: u64;
    let _adj: u64;

    unsafe {
        asm!(
            "adds  {result}, {a}, {b}",
            "csetm {adj:w}, cs",
            "add   {result}, {result}, {adj}",
            a = in(reg) a,
            b = in(reg) b,
            result = out(reg) result,
            adj = out(reg) _adj,
            options(pure, nomem, nostack),
        );
    }

    result
}

// ---------------------------------------------------------------------------
// Lane conversion (packed NEON <-> raw u64 arrays)
// ---------------------------------------------------------------------------

/// Unpack a packed NEON state into two raw `u64` lane arrays.
///
/// Each packed slot contains two Goldilocks elements (lane 0, lane 1).
/// This function extracts the internal `u64` representation of each
/// element into two separate arrays, one per lane.
///
/// # Layout
///
/// ```text
///     packed[i] = (field_elem_a, field_elem_b)
///
///     lane0[i] = field_elem_a.value    (raw u64)
///     lane1[i] = field_elem_b.value    (raw u64)
/// ```
#[inline]
pub(super) fn unpack_lanes<const WIDTH: usize>(
    state: &[PackedGoldilocksNeon; WIDTH],
) -> ([u64; WIDTH], [u64; WIDTH]) {
    // Extract the raw u64 representation from each packed slot.
    let lane0: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[0].value);
    let lane1: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[1].value);
    (lane0, lane1)
}

/// Pack two raw `u64` lane arrays back into a packed NEON state.
///
/// Each raw value is wrapped into a Goldilocks field element (with
/// reduction modulo P) and paired into a packed slot.
///
/// # Layout
///
/// ```text
///     lane0[i], lane1[i]  ->  packed[i] = (Goldilocks(lane0[i]), Goldilocks(lane1[i]))
/// ```
#[inline]
pub(super) fn pack_lanes<const WIDTH: usize>(
    state: &mut [PackedGoldilocksNeon; WIDTH],
    lane0: &[u64; WIDTH],
    lane1: &[u64; WIDTH],
) {
    for i in 0..WIDTH {
        // Wrap each raw u64 into a field element and pair them.
        state[i] = PackedGoldilocksNeon([Goldilocks::new(lane0[i]), Goldilocks::new(lane1[i])]);
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use proptest::prelude::*;

    use super::*;

    type F = Goldilocks;

    /// Reduce a raw `u64` to its canonical Goldilocks representative.
    fn canon(x: u64) -> u64 {
        F::new(x).as_canonical_u64()
    }

    proptest! {
        // ----------------------------------------------------------------
        // Scalar field arithmetic
        // ----------------------------------------------------------------

        /// Verify ASM addition against field addition.
        #[test]
        fn test_add_asm(a: u64, b: u64) {
            let expected = (F::new(a) + F::new(b)).as_canonical_u64();
            let got = canon(unsafe { add_asm(a, b) });
            prop_assert_eq!(got, expected);
        }

        /// Verify ASM multiplication against field multiplication.
        #[test]
        fn test_mul_asm(a: u64, b: u64) {
            let expected = (F::new(a) * F::new(b)).as_canonical_u64();
            let got = canon(unsafe { mul_asm(a, b) });
            prop_assert_eq!(got, expected);
        }

        /// Verify ASM fused multiply-add against field multiply-add.
        #[test]
        fn test_mul_add_asm(a: u64, b: u64, c: u64) {
            let expected = (F::new(a) * F::new(b) + F::new(c)).as_canonical_u64();
            let got = canon(unsafe { mul_add_asm(a, b, c) });
            prop_assert_eq!(got, expected);
        }

        // ----------------------------------------------------------------
        // Unpack: packed state -> two raw u64 lane arrays
        // ----------------------------------------------------------------

        #[test]
        fn test_unpack_lanes_w8(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
        ) {
            // Build a packed state from two independent lane arrays.
            let packed: [PackedGoldilocksNeon; 8] =
                core::array::from_fn(|i| PackedGoldilocksNeon([F::new(lane_a[i]), F::new(lane_b[i])]));

            // Unpack into raw u64 lane arrays.
            let (got0, got1) = unpack_lanes(&packed);

            // Each raw value must be the internal representation of the field element.
            for i in 0..8 {
                prop_assert_eq!(got0[i], F::new(lane_a[i]).value);
                prop_assert_eq!(got1[i], F::new(lane_b[i]).value);
            }
        }

        #[test]
        fn test_unpack_lanes_w12(
            lane_a in prop::array::uniform12(any::<u64>()),
            lane_b in prop::array::uniform12(any::<u64>()),
        ) {
            // Same verification, width 12.
            let packed: [PackedGoldilocksNeon; 12] =
                core::array::from_fn(|i| PackedGoldilocksNeon([F::new(lane_a[i]), F::new(lane_b[i])]));

            let (got0, got1) = unpack_lanes(&packed);

            for i in 0..12 {
                prop_assert_eq!(got0[i], F::new(lane_a[i]).value);
                prop_assert_eq!(got1[i], F::new(lane_b[i]).value);
            }
        }

        // ----------------------------------------------------------------
        // Pack: two raw u64 lane arrays -> packed state
        // ----------------------------------------------------------------

        #[test]
        fn test_pack_lanes_w8(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
        ) {
            // Pack two raw lane arrays into packed state.
            let mut packed = [PackedGoldilocksNeon([F::ZERO; 2]); 8];
            pack_lanes(&mut packed, &lane_a, &lane_b);

            // Each packed element must hold the two corresponding field elements.
            for i in 0..8 {
                prop_assert_eq!(packed[i].0[0], F::new(lane_a[i]));
                prop_assert_eq!(packed[i].0[1], F::new(lane_b[i]));
            }
        }

        #[test]
        fn test_pack_lanes_w12(
            lane_a in prop::array::uniform12(any::<u64>()),
            lane_b in prop::array::uniform12(any::<u64>()),
        ) {
            // Same verification, width 12.
            let mut packed = [PackedGoldilocksNeon([F::ZERO; 2]); 12];
            pack_lanes(&mut packed, &lane_a, &lane_b);

            for i in 0..12 {
                prop_assert_eq!(packed[i].0[0], F::new(lane_a[i]));
                prop_assert_eq!(packed[i].0[1], F::new(lane_b[i]));
            }
        }

        // ----------------------------------------------------------------
        // Roundtrip: pack then unpack recovers canonical values
        // ----------------------------------------------------------------

        #[test]
        fn test_roundtrip_pack_unpack_w8(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
        ) {
            // Pack two lane arrays, then unpack them.
            let mut packed = [PackedGoldilocksNeon([F::ZERO; 2]); 8];
            pack_lanes(&mut packed, &lane_a, &lane_b);
            let (out0, out1) = unpack_lanes(&packed);

            // The canonical form of the recovered values must match.
            for i in 0..8 {
                prop_assert_eq!(F::new(out0[i]).as_canonical_u64(), F::new(lane_a[i]).as_canonical_u64());
                prop_assert_eq!(F::new(out1[i]).as_canonical_u64(), F::new(lane_b[i]).as_canonical_u64());
            }
        }

        #[test]
        fn test_roundtrip_pack_unpack_w12(
            lane_a in prop::array::uniform12(any::<u64>()),
            lane_b in prop::array::uniform12(any::<u64>()),
        ) {
            // Same roundtrip, width 12.
            let mut packed = [PackedGoldilocksNeon([F::ZERO; 2]); 12];
            pack_lanes(&mut packed, &lane_a, &lane_b);
            let (out0, out1) = unpack_lanes(&packed);

            for i in 0..12 {
                prop_assert_eq!(F::new(out0[i]).as_canonical_u64(), F::new(lane_a[i]).as_canonical_u64());
                prop_assert_eq!(F::new(out1[i]).as_canonical_u64(), F::new(lane_b[i]).as_canonical_u64());
            }
        }

        // ----------------------------------------------------------------
        // Roundtrip: unpack then pack preserves packed state
        // ----------------------------------------------------------------

        #[test]
        fn test_roundtrip_unpack_pack_w8(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
        ) {
            // Start from a packed state.
            let original: [PackedGoldilocksNeon; 8] =
                core::array::from_fn(|i| PackedGoldilocksNeon([F::new(lane_a[i]), F::new(lane_b[i])]));

            // Unpack into raw lanes, then pack back.
            let (raw0, raw1) = unpack_lanes(&original);
            let mut restored = [PackedGoldilocksNeon([F::ZERO; 2]); 8];
            pack_lanes(&mut restored, &raw0, &raw1);

            // The restored packed state must equal the original.
            for i in 0..8 {
                prop_assert_eq!(restored[i].0[0], original[i].0[0]);
                prop_assert_eq!(restored[i].0[1], original[i].0[1]);
            }
        }

        #[test]
        fn test_roundtrip_unpack_pack_w12(
            lane_a in prop::array::uniform12(any::<u64>()),
            lane_b in prop::array::uniform12(any::<u64>()),
        ) {
            // Same reverse roundtrip, width 12.
            let original: [PackedGoldilocksNeon; 12] =
                core::array::from_fn(|i| PackedGoldilocksNeon([F::new(lane_a[i]), F::new(lane_b[i])]));

            let (raw0, raw1) = unpack_lanes(&original);
            let mut restored = [PackedGoldilocksNeon([F::ZERO; 2]); 12];
            pack_lanes(&mut restored, &raw0, &raw1);

            for i in 0..12 {
                prop_assert_eq!(restored[i].0[0], original[i].0[0]);
                prop_assert_eq!(restored[i].0[1], original[i].0[1]);
            }
        }
    }
}
