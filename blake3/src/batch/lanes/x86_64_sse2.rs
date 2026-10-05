//! Four lanes per 128-bit SSE2 register, the baseline of every x86-64 target.

use core::arch::x86_64::*;

use blake3::{BLOCK_LEN, OUT_LEN};

use super::{Backend, Kernel, Word};
use crate::batch::compress::{BLOCK_WORDS, STATE_WORDS};

/// Lanes in one register.
const WIDTH: usize = 4;

/// Independent register groups hashed together.
///
/// A second group spills more than its extra chains recover.
const GROUPS: usize = 1;

// SAFETY (every block below): SSE2 is part of the x86-64 baseline, and SSSE3 is checked.
impl Word for __m128i {
    #[inline(always)]
    fn splat(value: u32) -> Self {
        unsafe { _mm_set1_epi32(value as i32) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { _mm_add_epi32(self, rhs) }
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        unsafe { _mm_and_si128(self, rhs) }
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        unsafe { _mm_xor_si128(self, rhs) }
    }

    #[inline(always)]
    fn rotr_16(self) -> Self {
        // Swapping the two 16-bit halves of every word is the rotation.
        unsafe { _mm_shufflehi_epi16::<0xB1>(_mm_shufflelo_epi16::<0xB1>(self)) }
    }

    #[inline(always)]
    fn rotr_12(self) -> Self {
        unsafe { _mm_or_si128(_mm_srli_epi32::<12>(self), _mm_slli_epi32::<20>(self)) }
    }

    #[inline(always)]
    fn rotr_8(self) -> Self {
        // A byte shuffle moves each word's bytes [0, 1, 2, 3] to [1, 2, 3, 0].
        #[cfg(target_feature = "ssse3")]
        unsafe {
            let bytes = _mm_set_epi8(12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1);
            _mm_shuffle_epi8(self, bytes)
        }
        #[cfg(not(target_feature = "ssse3"))]
        unsafe {
            _mm_or_si128(_mm_srli_epi32::<8>(self), _mm_slli_epi32::<24>(self))
        }
    }

    #[inline(always)]
    fn rotr_7(self) -> Self {
        unsafe { _mm_or_si128(_mm_srli_epi32::<7>(self), _mm_slli_epi32::<25>(self)) }
    }
}

/// Transpose a 4 x 4 matrix of words held one row per register.
#[inline(always)]
fn transpose([a, b, c, d]: [__m128i; 4]) -> [__m128i; 4] {
    // SAFETY: SSE2 is part of the x86-64 baseline.
    unsafe {
        // Interleave 32-bit words of row pairs, then 64-bit pairs of those.
        let ab_lo = _mm_unpacklo_epi32(a, b);
        let ab_hi = _mm_unpackhi_epi32(a, b);
        let cd_lo = _mm_unpacklo_epi32(c, d);
        let cd_hi = _mm_unpackhi_epi32(c, d);
        [
            _mm_unpacklo_epi64(ab_lo, cd_lo),
            _mm_unpackhi_epi64(ab_lo, cd_lo),
            _mm_unpacklo_epi64(ab_hi, cd_hi),
            _mm_unpackhi_epi64(ab_hi, cd_hi),
        ]
    }
}

/// The batched driver on this backend.
pub(super) const KERNEL: Kernel = Kernel::new::<__m128i, WIDTH, GROUPS>("SSE2");

impl Backend<WIDTH> for __m128i {
    /// A single group: every register runs alone.
    const LONE_REGISTER_COST: usize = 16;

    #[inline]
    fn supported() -> bool {
        // SSE2 is part of the x86-64 baseline.
        true
    }

    /// Load one block from each of four lanes as sixteen message words.
    #[inline(always)]
    fn load_block(rows: &[&[u8; BLOCK_LEN]; WIDTH]) -> [Self; BLOCK_WORDS] {
        // Each block is four registers, and quarter k holds words 4k to 4k + 3.
        //
        // SAFETY: each quarter is 16 readable bytes, and the load has no alignment requirement.
        let quarters: [[Self; 4]; 4] = core::array::from_fn(|k| {
            transpose(core::array::from_fn(|l| unsafe {
                _mm_loadu_si128(rows[l][16 * k..].as_ptr().cast())
            }))
        });
        core::array::from_fn(|w| quarters[w / 4][w % 4])
    }

    /// Write the digests of four lanes.
    #[inline(always)]
    fn store_digests(state: &[Self; STATE_WORDS], out: &mut [[u8; OUT_LEN]; WIDTH]) {
        // Words 0 to 3, then words 4 to 7, of every lane.
        let low = transpose([state[0], state[1], state[2], state[3]]);
        let high = transpose([state[4], state[5], state[6], state[7]]);
        for ((digest, low), high) in out.iter_mut().zip(low).zip(high) {
            // SAFETY: each half of a digest is 16 writable bytes, with no alignment requirement.
            unsafe {
                _mm_storeu_si128(digest.as_mut_ptr().cast(), low);
                _mm_storeu_si128(digest[16..].as_mut_ptr().cast(), high);
            }
        }
    }

    out_of_line_steps!(WIDTH);
}
