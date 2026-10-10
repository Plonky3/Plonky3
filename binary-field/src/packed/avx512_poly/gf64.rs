//! The register moves and reductions of the explicit eight-lane polynomial packing.

use core::arch::x86_64::*;

use p3_field::interleave::{interleave_u64, interleave_u128, interleave_u256};

use crate::clmul::wide::{Lanes64, TOP_NIBBLE_FOLD};

/// The eight-word coordinate register.
pub(crate) type Reg = __m512i;
/// The number of coefficient-field elements in one register.
pub(crate) const WIDTH_64: usize = 8;

/// Interleave blocks of whole field elements.
#[inline(always)]
pub(crate) fn interleave_64(a: Reg, b: Reg, block_len: usize) -> (Reg, Reg) {
    match block_len {
        1 => interleave_u64(a, b),
        2 => interleave_u128(a, b),
        4 => interleave_u256(a, b),
        8 => (a, b),
        _ => panic!("unsupported block_len"),
    }
}

/// Load eight consecutive three-word elements into three coordinate registers.
///
/// # Safety
/// The address must provide twenty-four readable words at any alignment.
#[inline(always)]
pub(crate) unsafe fn gather_3(from: *const u64) -> [Reg; 3] {
    // SAFETY: each register load covers eight of the twenty-four readable words.
    unsafe {
        let rows = core::array::from_fn::<_, 3, _>(|r| _mm512_loadu_si512(from.add(8 * r).cast()));
        core::array::from_fn(|d| {
            // A coordinate's flat position is 3*k+d, split into register and word indices.
            let mask = |r| {
                (0..8).fold(0u8, |mask, k| {
                    if (3 * k + d) / 8 == r {
                        mask | 1 << ((3 * k + d) % 8)
                    } else {
                        mask
                    }
                })
            };
            let indices: [i64; 8] = core::array::from_fn(|k| ((3 * k + d) % 8) as i64);
            let blended = _mm512_mask_blend_epi64(
                mask(2),
                _mm512_mask_blend_epi64(mask(1), rows[0], rows[1]),
                rows[2],
            );
            _mm512_permutexvar_epi64(_mm512_loadu_si512(indices.as_ptr().cast()), blended)
        })
    }
}

/// Store three coordinate registers as eight consecutive three-word elements.
///
/// # Safety
/// The address must provide twenty-four writable words at any alignment.
#[inline(always)]
pub(crate) unsafe fn scatter_3(to: *mut u64, coordinates: [Reg; 3]) {
    // SAFETY: each store covers eight of the twenty-four writable words.
    unsafe {
        for r in 0..3 {
            let indices: [i64; 8] = core::array::from_fn(|p| ((8 * r + p) / 3) as i64);
            let index = _mm512_loadu_si512(indices.as_ptr().cast());
            let values = coordinates.map(|c| _mm512_permutexvar_epi64(index, c));
            let mask = |d| {
                (0..8).fold(0u8, |mask, p| {
                    if (8 * r + p) % 3 == d {
                        mask | 1 << p
                    } else {
                        mask
                    }
                })
            };
            let row = _mm512_mask_blend_epi64(
                mask(2),
                _mm512_mask_blend_epi64(mask(1), values[0], values[1]),
                values[2],
            );
            _mm512_storeu_si512(to.add(8 * r).cast(), row);
        }
    }
}

// SAFETY: the module is gated on every target feature these register operations require.
impl Lanes64 for Reg {
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    fn xor(self, b: Self) -> Self {
        unsafe { _mm512_xor_si512(self, b) }
    }
    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        unsafe { _mm512_ternarylogic_epi64::<0x96>(self, b, c) }
    }
    #[inline(always)]
    fn clmul<const IMM: i32>(self, b: Self) -> Self {
        unsafe { _mm512_clmulepi64_epi128::<IMM>(self, b) }
    }
    #[inline(always)]
    fn unpack_low(self, b: Self) -> Self {
        unsafe { _mm512_unpacklo_epi64(self, b) }
    }
    #[inline(always)]
    fn unpack_high(self, b: Self) -> Self {
        unsafe { _mm512_unpackhi_epi64(self, b) }
    }
    #[inline(always)]
    fn shl<const N: i32>(self) -> Self {
        unsafe { _mm512_sll_epi64(self, _mm_cvtsi32_si128(N)) }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        unsafe { _mm512_srl_epi64(self, _mm_cvtsi32_si128(N)) }
    }
    #[inline(always)]
    fn fold_top_nibble(self) -> Self {
        unsafe {
            let table = _mm512_broadcast_i32x4(_mm_loadu_si128(TOP_NIBBLE_FOLD.as_ptr().cast()));
            _mm512_shuffle_epi8(table, _mm512_srli_epi64::<60>(self))
        }
    }
}
