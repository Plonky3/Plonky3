//! Eight lanes per 256-bit AVX2 register.

use core::arch::x86_64::*;

use super::Word;
use crate::DIGEST_BYTES;
use crate::batch::compress::{BLOCK_BYTES, BLOCK_WORDS};

/// One state or message word for eight lanes.
pub(super) type Vector = __m256i;

/// Lanes in one register.
pub(super) const WIDTH: usize = 8;

/// Independent register groups hashed together.
///
/// Two beat one and four here: one leaves the pipeline waiting on the dependency chains of
/// G, and four spill the working vectors out of the 16 registers.
pub(super) const GROUPS: usize = 2;

// SAFETY (every block below): this module only compiles when the target enables AVX2.
impl Word for __m256i {
    #[inline(always)]
    fn splat(value: u32) -> Self {
        unsafe { _mm256_set1_epi32(value as i32) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { _mm256_add_epi32(self, rhs) }
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        unsafe { _mm256_and_si256(self, rhs) }
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        unsafe { _mm256_xor_si256(self, rhs) }
    }

    #[inline(always)]
    fn rotr_16(self) -> Self {
        // A byte shuffle moves each word's bytes [0, 1, 2, 3] to [2, 3, 0, 1].
        unsafe {
            let bytes = _mm256_set_epi8(
                13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10,
                5, 4, 7, 6, 1, 0, 3, 2,
            );
            _mm256_shuffle_epi8(self, bytes)
        }
    }

    #[inline(always)]
    fn rotr_12(self) -> Self {
        unsafe { _mm256_or_si256(_mm256_srli_epi32::<12>(self), _mm256_slli_epi32::<20>(self)) }
    }

    #[inline(always)]
    fn rotr_8(self) -> Self {
        // A byte shuffle moves each word's bytes [0, 1, 2, 3] to [1, 2, 3, 0].
        unsafe {
            let bytes = _mm256_set_epi8(
                12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1, 12, 15, 14, 13, 8, 11, 10, 9,
                4, 7, 6, 5, 0, 3, 2, 1,
            );
            _mm256_shuffle_epi8(self, bytes)
        }
    }

    #[inline(always)]
    fn rotr_7(self) -> Self {
        unsafe { _mm256_or_si256(_mm256_srli_epi32::<7>(self), _mm256_slli_epi32::<25>(self)) }
    }
}

/// Transpose an 8 x 8 matrix of words held one row per register.
///
/// ```text
///     phase 1: unpack 32-bit pairs, then 64-bit pairs, inside each 128-bit half
///     phase 2: swap 128-bit halves between the two quads of rows
/// ```
#[inline(always)]
fn transpose(r: [__m256i; 8]) -> [__m256i; 8] {
    // SAFETY: this module only compiles when the target enables AVX2.
    unsafe {
        // Phase 1: a 4 x 4 transpose inside each 128-bit half of each quad of rows.
        //
        //     u[q][j], half k = word 4k + j of rows 4q .. 4q + 3
        let u: [[__m256i; 4]; 2] = core::array::from_fn(|q| {
            let [a, b, c, d] = [r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]];
            let ab_lo = _mm256_unpacklo_epi32(a, b);
            let ab_hi = _mm256_unpackhi_epi32(a, b);
            let cd_lo = _mm256_unpacklo_epi32(c, d);
            let cd_hi = _mm256_unpackhi_epi32(c, d);
            [
                _mm256_unpacklo_epi64(ab_lo, cd_lo),
                _mm256_unpackhi_epi64(ab_lo, cd_lo),
                _mm256_unpacklo_epi64(ab_hi, cd_hi),
                _mm256_unpackhi_epi64(ab_hi, cd_hi),
            ]
        });

        // Phase 2: word 4k + j joins half k of both quads.
        let mut out = [_mm256_setzero_si256(); 8];
        for j in 0..4 {
            out[j] = _mm256_permute2x128_si256::<0x20>(u[0][j], u[1][j]);
            out[4 + j] = _mm256_permute2x128_si256::<0x31>(u[0][j], u[1][j]);
        }
        out
    }
}

/// Load one block from each of eight lanes as sixteen message words.
#[inline(always)]
pub(super) fn load_block(rows: &[&[u8; BLOCK_BYTES]; WIDTH]) -> [__m256i; BLOCK_WORDS] {
    // Each block is two registers: words 0 to 7, then words 8 to 15.
    //
    // SAFETY: each half is 32 readable bytes, and the load has no alignment requirement.
    let half = |h: usize| {
        transpose(rows.map(|row| unsafe { _mm256_loadu_si256(row[32 * h..].as_ptr().cast()) }))
    };
    let (low, high) = (half(0), half(1));
    core::array::from_fn(|w| if w < 8 { low[w] } else { high[w - 8] })
}

/// Write the digests of eight lanes.
#[inline(always)]
pub(super) fn store_digests(state: &[__m256i; 8], out: &mut [[u8; DIGEST_BYTES]; WIDTH]) {
    // Eight words of eight lanes is a square, so row l comes back as lane l.
    for (digest, row) in out.iter_mut().zip(transpose(*state)) {
        // SAFETY: a digest is 32 writable bytes, and the store has no alignment requirement.
        unsafe { _mm256_storeu_si256(digest.as_mut_ptr().cast(), row) };
    }
}
