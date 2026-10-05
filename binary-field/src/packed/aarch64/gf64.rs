//! The 128-bit register the `GF(2^64)` packings use: two elements per register.
//!
//! The lane operations are the ones the scalar kernels already run on.

use core::arch::aarch64::{uint64x2_t, uint64x2x3_t, vld3q_u64, vst3q_u64, vzip1q_u64, vzip2q_u64};

/// The register one packed value occupies.
pub(crate) type Reg = uint64x2_t;

/// 64-bit elements per register.
pub(crate) const WIDTH_64: usize = 2;

/// Interleaves blocks of 64-bit elements between two registers.
///
/// # Panics
///
/// Panics if the block length does not divide the width.
#[inline(always)]
pub(crate) fn interleave_64(a: Reg, b: Reg, block_len: usize) -> (Reg, Reg) {
    match block_len {
        // `[a0 a1], [b0 b1]` to `[a0 b0], [a1 b1]`.
        //
        // SAFETY: the packing compiles only with `aes`, which implies `neon`.
        1 => unsafe { (vzip1q_u64(a, b), vzip2q_u64(a, b)) },
        WIDTH_64 => (a, b),
        _ => panic!("unsupported block_len"),
    }
}

/// Two consecutive three-quadword elements, as three coordinate registers.
///
/// `LD3` de-interleaves by quadword as it loads:
///
/// ```text
///     memory      [ a_0 a_1 a_2 b_0 b_1 b_2 ]
///     registers   [ a_0 b_0 ], [ a_1 b_1 ], [ a_2 b_2 ]
/// ```
///
/// # Safety
///
/// The address must be readable for six quadwords.
///
/// No alignment is required.
#[inline(always)]
pub(crate) unsafe fn gather_3(from: *const u64) -> [Reg; 3] {
    // SAFETY: the readability of the six quadwords is the caller's obligation.
    let uint64x2x3_t(c0, c1, c2) = unsafe { vld3q_u64(from) };
    [c0, c1, c2]
}

/// Three coordinate registers, written back as two consecutive elements by one `ST3`.
///
/// # Safety
///
/// The address must be writable for six quadwords.
///
/// No alignment is required.
#[inline(always)]
pub(crate) unsafe fn scatter_3(to: *mut u64, [c0, c1, c2]: [Reg; 3]) {
    // SAFETY: the writability of the six quadwords is the caller's obligation.
    unsafe { vst3q_u64(to, uint64x2x3_t(c0, c1, c2)) }
}
