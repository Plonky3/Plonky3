//! The 512-bit register: four `GF(2^128)` elements.

use core::arch::x86_64::{
    __m512i, _mm_set_epi64x, _mm512_broadcast_i32x4, _mm512_clmulepi64_epi128, _mm512_loadu_si512,
    _mm512_setzero_si512, _mm512_shuffle_epi32, _mm512_storeu_si512, _mm512_unpacklo_epi64,
    _mm512_xor_si512,
};

use p3_field::interleave::{interleave_u128, interleave_u256};

/// The register one packed value occupies.
pub(crate) type Reg = __m512i;

/// 128-bit lanes per register.
pub(crate) const WIDTH: usize = 4;

// SAFETY for every wrapper below: this module compiles only with the features they require.
//
// That makes each intrinsic call sound.

/// All lanes zero.
#[inline(always)]
pub(crate) fn zero() -> Reg {
    unsafe { _mm512_setzero_si512() }
}

/// Bitwise exclusive or.
#[inline(always)]
pub(crate) fn xor(a: Reg, b: Reg) -> Reg {
    unsafe { _mm512_xor_si512(a, b) }
}

/// The low quadword of each operand, paired within each lane.
#[inline(always)]
pub(crate) fn unpack_low_64(a: Reg, b: Reg) -> Reg {
    unsafe { _mm512_unpacklo_epi64(a, b) }
}

/// The carryless product of one quadword of each operand, in every lane.
#[inline(always)]
pub(crate) fn clmul<const IMM: i32>(a: Reg, b: Reg) -> Reg {
    unsafe { _mm512_clmulepi64_epi128::<IMM>(a, b) }
}

/// Exchanges the two halves of every lane.
#[inline(always)]
pub(crate) fn swap_halves(a: Reg) -> Reg {
    unsafe { _mm512_shuffle_epi32::<{ super::SWAP_QUADWORDS }>(a) }
}

/// The same 128-bit value in every lane.
#[inline(always)]
pub(crate) fn broadcast(value: u128) -> Reg {
    unsafe {
        // Arguments run from the highest quadword down.
        let lane = _mm_set_epi64x((value >> 64) as i64, value as i64);
        _mm512_broadcast_i32x4(lane)
    }
}

/// Reads one register from memory.
///
/// # Safety
///
/// The address must be readable for 64 bytes.
/// No alignment is required.
#[inline(always)]
pub(crate) unsafe fn load(from: *const u128) -> Reg {
    // SAFETY: the readability of the address is the caller's obligation.
    unsafe { _mm512_loadu_si512(from.cast()) }
}

/// Writes one register to memory.
///
/// # Safety
///
/// The address must be writable for 64 bytes.
/// No alignment is required.
#[inline(always)]
pub(crate) unsafe fn store(to: *mut u128, value: Reg) {
    // SAFETY: the writability of the address is the caller's obligation.
    unsafe { _mm512_storeu_si512(to.cast(), value) }
}

/// Interleaves whole 128-bit elements between two registers.
///
/// # Panics
/// Panics if the block length does not divide the width.
#[inline(always)]
pub(crate) fn interleave(a: Reg, b: Reg, block_len: usize) -> (Reg, Reg) {
    match block_len {
        1 => interleave_u128(a, b),
        2 => interleave_u256(a, b),
        WIDTH => (a, b),
        _ => panic!("unsupported block_len"),
    }
}
