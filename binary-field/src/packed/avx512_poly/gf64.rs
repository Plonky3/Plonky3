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

#[cfg(miri)]
#[inline(always)]
fn map_words(r: Reg, f: impl Fn(u64) -> u64) -> Reg {
    // SAFETY: a register and an eight-word array share size and every bit pattern is valid.
    unsafe {
        core::mem::transmute::<[u64; 8], Reg>(core::mem::transmute::<Reg, [u64; 8]>(r).map(f))
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
    #[cfg(not(miri))]
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
        // Miri has no shim for the 512-bit shifts.
        // Preserve their word-wise semantics, including zero for a count outside 0..64, while interpreting the real algebra.
        #[cfg(miri)]
        {
            map_words(self, |word| word.checked_shl(N as u32).unwrap_or(0))
        }
        #[cfg(not(miri))]
        unsafe {
            _mm512_sll_epi64(self, _mm_cvtsi32_si128(N))
        }
    }
    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        // The right-shift counterpart of the Miri shim above.
        #[cfg(miri)]
        {
            map_words(self, |word| word.checked_shr(N as u32).unwrap_or(0))
        }
        #[cfg(not(miri))]
        unsafe {
            _mm512_srl_epi64(self, _mm_cvtsi32_si128(N))
        }
    }
    #[inline(always)]
    fn fold_top_nibble(self) -> Self {
        unsafe {
            let table = _mm512_broadcast_i32x4(_mm_loadu_si128(TOP_NIBBLE_FOLD.as_ptr().cast()));
            _mm512_shuffle_epi8(table, _mm512_srli_epi64::<60>(self))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{gather_3, scatter_3};

    #[test]
    fn unaligned_coordinate_moves_preserve_every_word_and_the_boundaries() {
        // Every word is distinct, so any misplaced word fails the comparison.
        #[repr(align(64))]
        struct Buffer([u64; 26]);
        let input = Buffer(core::array::from_fn(|i| (i as u64 + 1) << (i % 32)));
        // SAFETY: the offset leaves exactly twenty-four readable words and a sentinel.
        let coordinates = unsafe { gather_3(input.0.as_ptr().add(1)) };
        for (d, coordinate) in coordinates.iter().enumerate() {
            // SAFETY: both representations contain exactly eight arbitrary u64 words.
            let words: [u64; 8] = unsafe { core::mem::transmute(*coordinate) };
            assert_eq!(words, core::array::from_fn(|k| input.0[1 + 3 * k + d]));
        }
        let mut output = Buffer([u64::MAX; 26]);
        // SAFETY: the offset leaves twenty-four writable words before the sentinel.
        unsafe { scatter_3(output.0.as_mut_ptr().add(1), coordinates) };
        assert_eq!(output.0[0], u64::MAX);
        assert_eq!(&output.0[1..25], &input.0[1..25]);
        assert_eq!(output.0[25], u64::MAX);
    }
}
