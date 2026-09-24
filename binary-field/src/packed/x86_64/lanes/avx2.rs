//! The 256-bit register: two `GF(2^128)` elements, or four `GF(2^64)` ones.

#[cfg(not(target_feature = "avx512f"))]
use core::arch::x86_64::_mm256_shuffle_epi32;
#[cfg(target_feature = "avx512vl")]
use core::arch::x86_64::_mm256_ternarylogic_epi64;
use core::arch::x86_64::{
    __m256i, _mm_set_epi64x, _mm256_blend_epi32, _mm256_broadcastsi128_si256,
    _mm256_clmulepi64_epi128, _mm256_loadu_si256, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi64,
    _mm256_srli_epi64, _mm256_storeu_si256, _mm256_unpackhi_epi64, _mm256_unpacklo_epi64,
    _mm256_xor_si256,
};

use super::triples::Plan;

/// The register one packed value occupies.
pub(crate) type Reg = __m256i;

/// 128-bit lanes per register.
pub(crate) const WIDTH: usize = 2;

// SAFETY for every wrapper below: this module compiles only with the features they require.
//
// That makes each intrinsic call sound.

/// All lanes zero.
#[inline(always)]
pub(crate) fn zero() -> Reg {
    unsafe { _mm256_setzero_si256() }
}

/// Bitwise exclusive or.
#[inline(always)]
pub(crate) fn xor(a: Reg, b: Reg) -> Reg {
    unsafe { _mm256_xor_si256(a, b) }
}

/// Three-way exclusive or, in one instruction where `avx512vl` supplies a ternary logic op.
#[inline(always)]
pub(crate) fn xor3(a: Reg, b: Reg, c: Reg) -> Reg {
    #[cfg(target_feature = "avx512vl")]
    {
        unsafe { _mm256_ternarylogic_epi64::<{ super::XOR3 }>(a, b, c) }
    }
    #[cfg(not(target_feature = "avx512vl"))]
    {
        xor(xor(a, b), c)
    }
}

/// The low quadword of each operand, paired within each lane.
#[inline(always)]
pub(crate) fn unpack_low_64(a: Reg, b: Reg) -> Reg {
    unsafe { _mm256_unpacklo_epi64(a, b) }
}

/// The high quadword of each operand, paired within each lane.
#[inline(always)]
pub(crate) fn unpack_high_64(a: Reg, b: Reg) -> Reg {
    unsafe { _mm256_unpackhi_epi64(a, b) }
}

/// Each quadword shifted left.
#[inline(always)]
pub(crate) fn shl_64<const N: i32>(a: Reg) -> Reg {
    unsafe { _mm256_slli_epi64::<N>(a) }
}

/// Each quadword shifted right.
#[inline(always)]
pub(crate) fn shr_64<const N: i32>(a: Reg) -> Reg {
    unsafe { _mm256_srli_epi64::<N>(a) }
}

/// Each byte of `indices` looked up in the same lane of `table`.
#[inline(always)]
pub(crate) fn shuffle_bytes(table: Reg, indices: Reg) -> Reg {
    unsafe { _mm256_shuffle_epi8(table, indices) }
}

/// The carryless product of one quadword of each operand, in every lane.
#[inline(always)]
pub(crate) fn clmul<const IMM: i32>(a: Reg, b: Reg) -> Reg {
    unsafe { _mm256_clmulepi64_epi128::<IMM>(a, b) }
}

/// Exchanges the two halves of every lane.
// Only the GHASH packing uses it, and that packing takes 512 bits where they exist.
#[cfg(not(target_feature = "avx512f"))]
#[inline(always)]
pub(crate) fn swap_halves(a: Reg) -> Reg {
    unsafe { _mm256_shuffle_epi32::<{ super::SWAP_QUADWORDS }>(a) }
}

/// The same 128-bit value in every lane.
#[inline(always)]
pub(crate) fn broadcast(value: u128) -> Reg {
    unsafe {
        // Arguments run from the highest quadword down.
        let lane = _mm_set_epi64x((value >> 64) as i64, value as i64);
        _mm256_broadcastsi128_si256(lane)
    }
}

/// Reads one register from memory.
///
/// # Safety
///
/// The address must be readable for 32 bytes.
/// No alignment is required.
#[inline(always)]
pub(crate) unsafe fn load(from: *const u128) -> Reg {
    // SAFETY: the readability of the address is the caller's obligation.
    unsafe { _mm256_loadu_si256(from.cast()) }
}

/// Writes one register to memory.
///
/// # Safety
///
/// The address must be writable for 32 bytes.
/// No alignment is required.
#[inline(always)]
pub(crate) unsafe fn store(to: *mut u128, value: Reg) {
    // SAFETY: the writability of the address is the caller's obligation.
    unsafe { _mm256_storeu_si256(to.cast(), value) }
}

/// Interleaves whole 128-bit elements between two registers.
///
/// # Panics
/// Panics if the block length does not divide the width.
// Only the GHASH packing uses it, and that packing takes 512 bits where they exist.
#[cfg(not(target_feature = "avx512f"))]
#[inline(always)]
pub(crate) fn interleave(a: Reg, b: Reg, block_len: usize) -> (Reg, Reg) {
    match block_len {
        1 => interleave_u128(a, b),
        WIDTH => (a, b),
        _ => panic!("unsupported block_len"),
    }
}

/// Interleaves blocks of 64-bit elements between two registers.
///
/// # Panics
/// Panics if the block length does not divide the width in quadwords.
#[inline(always)]
pub(crate) fn interleave_64(a: Reg, b: Reg, block_len: usize) -> (Reg, Reg) {
    match block_len {
        1 => interleave_u64(a, b),
        2 => interleave_u128(a, b),
        4 => (a, b),
        _ => panic!("unsupported block_len"),
    }
}

/// The transpose plan at this width.
const PLAN: Plan<4> = Plan::NEW;

/// A quadword mask as the immediate of a doubleword blend, two bits per quadword.
const fn blend_imm(mask: u32) -> i32 {
    let mut imm = 0;
    let mut p = 0;
    while p < 4 {
        // A quadword is two doublewords, so each mask bit sets two immediate bits.
        if mask >> p & 1 == 1 {
            imm |= 0b11 << (2 * p);
        }
        p += 1;
    }
    imm
}

/// A lane index as the immediate of a quadword permute, two bits per lane.
const fn permute_imm(index: [usize; 4]) -> i32 {
    let mut imm = 0;
    let mut k = 0;
    while k < 4 {
        // Lane k of the result reads the lane named by bits 2k and 2k + 1.
        imm |= (index[k] as i32) << (2 * k);
        k += 1;
    }
    imm
}

/// Two blends then one permute, all by immediate.
#[inline(always)]
fn blend_permute<const SECOND: i32, const THIRD: i32, const PERMUTE: i32>(
    a: Reg,
    b: Reg,
    c: Reg,
) -> Reg {
    // SAFETY: `avx2` is enabled whenever this module is compiled.
    unsafe {
        // Each position takes the quadword of the one register that holds this coordinate there.
        let blended = _mm256_blend_epi32::<THIRD>(_mm256_blend_epi32::<SECOND>(a, b), c);

        // Then the quadwords move into element order.
        _mm256_permute4x64_epi64::<PERMUTE>(blended)
    }
}

/// One permute then two blends, all by immediate.
#[inline(always)]
fn permute_blend<
    const P0: i32,
    const P1: i32,
    const P2: i32,
    const SECOND: i32,
    const THIRD: i32,
>(
    [a, b, c]: [Reg; 3],
) -> Reg {
    // SAFETY: `avx2` is enabled whenever this module is compiled.
    unsafe {
        // Each coordinate moves its lanes to the positions they take in memory.
        let (a, b, c) = (
            _mm256_permute4x64_epi64::<P0>(a),
            _mm256_permute4x64_epi64::<P1>(b),
            _mm256_permute4x64_epi64::<P2>(c),
        );

        // Each position of this register then takes the coordinate stored there.
        _mm256_blend_epi32::<THIRD>(_mm256_blend_epi32::<SECOND>(a, b), c)
    }
}

/// The gather of one coordinate, its immediates evaluated at compile time.
///
/// Immediates must be constants, so the coordinate index is a literal rather than a variable.
macro_rules! gather {
    ($d:literal, $regs:expr) => {{
        const SECOND: i32 = blend_imm(PLAN.gather_mask[$d][0]);
        const THIRD: i32 = blend_imm(PLAN.gather_mask[$d][1]);
        const PERMUTE: i32 = permute_imm(PLAN.gather_index[$d]);
        let [a, b, c] = $regs;
        blend_permute::<SECOND, THIRD, PERMUTE>(a, b, c)
    }};
}

/// The scatter into one register, its immediates evaluated at compile time.
macro_rules! scatter {
    ($r:literal, $coordinates:expr) => {{
        const P0: i32 = permute_imm(PLAN.scatter_index[0]);
        const P1: i32 = permute_imm(PLAN.scatter_index[1]);
        const P2: i32 = permute_imm(PLAN.scatter_index[2]);
        const SECOND: i32 = blend_imm(PLAN.scatter_mask[$r][0]);
        const THIRD: i32 = blend_imm(PLAN.scatter_mask[$r][1]);
        permute_blend::<P0, P1, P2, SECOND, THIRD>($coordinates)
    }};
}

/// Three registers of consecutive three-quadword elements, as three coordinate registers.
#[inline(always)]
pub(crate) fn deinterleave_3(registers: [Reg; 3]) -> [Reg; 3] {
    // Two blends and one permute per coordinate: six blends and three permutes in all.
    [
        gather!(0, registers),
        gather!(1, registers),
        gather!(2, registers),
    ]
}

/// Three coordinate registers, as three registers of consecutive three-quadword elements.
#[inline(always)]
pub(crate) fn interleave_3(coordinates: [Reg; 3]) -> [Reg; 3] {
    // The same instructions as the gather, in the opposite order.
    [
        scatter!(0, coordinates),
        scatter!(1, coordinates),
        scatter!(2, coordinates),
    ]
}

/// Interleaves quadwords: `[a0 .. a3], [b0 .. b3]` to `[a0 b0 a2 b2], [a1 b1 a3 b3]`.
#[inline(always)]
fn interleave_u64(a: Reg, b: Reg) -> (Reg, Reg) {
    (unpack_low_64(a, b), unpack_high_64(a, b))
}

/// Interleaves 128-bit halves: `[a0 a1], [b0 b1]` to `[a0 b0], [a1 b1]`.
#[inline(always)]
fn interleave_u128(a: Reg, b: Reg) -> (Reg, Reg) {
    // The high half of `a` beside the low half of `b`.
    let crossed = unsafe { _mm256_permute2x128_si256::<0x21>(a, b) };

    // Each output keeps one half in place and takes the other from the crossed register.
    unsafe {
        (
            _mm256_blend_epi32::<0b1111_0000>(a, crossed),
            _mm256_blend_epi32::<0b1111_0000>(crossed, b),
        )
    }
}
