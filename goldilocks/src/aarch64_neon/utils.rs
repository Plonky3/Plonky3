//! Lane conversion utilities for packed Goldilocks NEON state.
//!
//! # Overview
//!
//! The fused Poseidon permutations operate on **raw `u64` arrays** for maximum
//! performance (direct ASM, no field-element overhead). But the public API
//! exposes **packed NEON state** (two Goldilocks elements per slot).
//!
//! These two helpers bridge the gap:
//!
//! ```text
//!     Packed state                  Raw lanes
//!
//!     ┌──────────────┐        ┌──────────┐  ┌──────────┐
//!     │ (a0, b0)     │   ──►  │ a0       │  │ b0       │
//!     │ (a1, b1)     │   ──►  │ a1       │  │ b1       │
//!     │  ...         │        │  ...     │  │  ...     │
//!     │ (aN, bN)     │   ──►  │ aN       │  │ bN       │
//!     └──────────────┘        └──────────┘  └──────────┘
//!        WIDTH slots            lane 0        lane 1
//! ```
//!
//! # Why Two Separate Lanes?
//!
//! Goldilocks has a 64-bit prime, so NEON's 128-bit registers hold exactly
//! **2 independent field elements** (2 lanes). The fused ASM kernels process
//! two permutations simultaneously by working on lane 0 and lane 1 in
//! parallel, hiding multiplication latency through interleaving.
//!
//! The conversion is done **once at entry and once at exit** of the
//! permutation, keeping the hot inner loop entirely in raw `u64` land.

use super::packing::PackedGoldilocksNeon;
use crate::Goldilocks;

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

    proptest! {
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
