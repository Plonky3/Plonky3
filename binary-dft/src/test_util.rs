//! A scalar stand-in for a byte register, shared by the tests of the subfield kernels.
//!
//! Without it only a target carrying the byte-map instruction would exercise any of them.

use crate::lanes::ByteLanes;

/// Bytes one modelled register holds.
///
/// This matches the widest register the crate compiles a backend for.
///
/// Kernel tests size their payloads by it, so a run spans whole registers plus a tail.
pub(crate) const LANE_BYTES: usize = 64;

/// One register of bytes, held in a plain array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Model(pub(crate) [u8; LANE_BYTES]);

/// The image of one byte under one block, in the instruction's own layout.
fn apply(matrix: u64, input: u8) -> u8 {
    let rows = matrix.to_le_bytes();
    let mut out = 0u8;

    // Output bit `i` reads row `7 - i`.
    //
    // Within that row, bit `b` pairs with bit `b` of the input byte.
    for i in 0..8 {
        if (rows[7 - i] & input).count_ones() % 2 == 1 {
            out |= 1 << i;
        }
    }
    out
}

impl ByteLanes for Model {
    const BYTES: usize = LANE_BYTES;

    unsafe fn load(from: *const u8) -> Self {
        // SAFETY: readability for one whole register is the caller's obligation.
        Self(core::array::from_fn(|i| unsafe { *from.add(i) }))
    }

    unsafe fn store(to: *mut u8, value: Self) {
        for (i, byte) in value.0.into_iter().enumerate() {
            // SAFETY: writability for one whole register is the caller's obligation.
            unsafe { *to.add(i) = byte };
        }
    }

    fn xor(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] ^ other.0[i]))
    }

    fn rotate_group<const GROUP: usize>(self, shift: usize) -> Self {
        assert!(shift < GROUP, "rotation leaves the group");

        // A position takes the byte `shift` further along its own group.
        //
        // The walk wraps inside the group, so nothing crosses into a neighbour.
        Self(core::array::from_fn(|p| {
            let base = p - p % GROUP;
            self.0[base + (p % GROUP + shift) % GROUP]
        }))
    }

    fn affine(self, matrix: u64) -> Self {
        Self(core::array::from_fn(|i| apply(matrix, self.0[i])))
    }

    fn affine_merge(self, source: Self, matrix: u64, mask: u64) -> Self {
        Self(core::array::from_fn(|i| {
            if (mask >> i) & 1 == 1 {
                apply(matrix, source.0[i])
            } else {
                self.0[i]
            }
        }))
    }
}
