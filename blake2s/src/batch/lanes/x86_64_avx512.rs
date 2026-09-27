//! Sixteen lanes per 512-bit AVX-512 register.

use core::arch::x86_64::*;

use super::Word;
use crate::DIGEST_BYTES;
use crate::batch::compress::{BLOCK_BYTES, BLOCK_WORDS};

/// One state or message word for sixteen lanes.
pub(super) type Vector = __m512i;

/// Lanes in one register.
pub(super) const WIDTH: usize = 16;

/// Independent register groups hashed together.
///
/// Two beat one and four here: one leaves the pipeline waiting on the dependency chains of
/// G, and four spill the working vectors out of the 32 registers.
pub(super) const GROUPS: usize = 2;

// SAFETY (every block below): this module only compiles when the target enables AVX-512F.
impl Word for __m512i {
    #[inline(always)]
    fn splat(value: u32) -> Self {
        unsafe { _mm512_set1_epi32(value as i32) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { _mm512_add_epi32(self, rhs) }
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        unsafe { _mm512_and_si512(self, rhs) }
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        unsafe { _mm512_xor_si512(self, rhs) }
    }

    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        // 0x96 is the truth table of a ^ b ^ c.
        unsafe { _mm512_ternarylogic_epi32::<0x96>(self, b, c) }
    }

    #[inline(always)]
    fn rotr_16(self) -> Self {
        unsafe { _mm512_ror_epi32::<16>(self) }
    }

    #[inline(always)]
    fn rotr_12(self) -> Self {
        unsafe { _mm512_ror_epi32::<12>(self) }
    }

    #[inline(always)]
    fn rotr_8(self) -> Self {
        unsafe { _mm512_ror_epi32::<8>(self) }
    }

    #[inline(always)]
    fn rotr_7(self) -> Self {
        unsafe { _mm512_ror_epi32::<7>(self) }
    }
}

/// Transpose a 16 x 16 matrix of words held one row per register.
///
/// ```text
///     phase 1: unpack 32-bit pairs, then 64-bit pairs, inside each 128-bit block
///     phase 2: shuffle whole 128-bit blocks across the four rows of each quad
/// ```
#[inline(always)]
fn transpose(r: [__m512i; 16]) -> [__m512i; 16] {
    // SAFETY: this module only compiles when the target enables AVX-512F.
    unsafe {
        // Phase 1: a 4 x 4 transpose inside every 128-bit block of every quad of rows.
        //
        //     u[q][j], block k = word 4k + j of rows 4q .. 4q + 3
        let u: [[__m512i; 4]; 4] = core::array::from_fn(|q| {
            let [a, b, c, d] = [r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]];
            let ab_lo = _mm512_unpacklo_epi32(a, b);
            let ab_hi = _mm512_unpackhi_epi32(a, b);
            let cd_lo = _mm512_unpacklo_epi32(c, d);
            let cd_hi = _mm512_unpackhi_epi32(c, d);
            [
                _mm512_unpacklo_epi64(ab_lo, cd_lo),
                _mm512_unpackhi_epi64(ab_lo, cd_lo),
                _mm512_unpacklo_epi64(ab_hi, cd_hi),
                _mm512_unpackhi_epi64(ab_hi, cd_hi),
            ]
        });

        // Phase 2: word 4k + j gathers block k of u[0][j] .. u[3][j], in quad order.
        let mut out = [_mm512_setzero_si512(); 16];
        for j in 0..4 {
            // Blocks 0 and 1, then blocks 2 and 3, of each pair of quads.
            let q01_lo = _mm512_shuffle_i32x4::<0x44>(u[0][j], u[1][j]);
            let q01_hi = _mm512_shuffle_i32x4::<0xEE>(u[0][j], u[1][j]);
            let q23_lo = _mm512_shuffle_i32x4::<0x44>(u[2][j], u[3][j]);
            let q23_hi = _mm512_shuffle_i32x4::<0xEE>(u[2][j], u[3][j]);

            // Even blocks, then odd blocks, of each half.
            out[j] = _mm512_shuffle_i32x4::<0x88>(q01_lo, q23_lo);
            out[4 + j] = _mm512_shuffle_i32x4::<0xDD>(q01_lo, q23_lo);
            out[8 + j] = _mm512_shuffle_i32x4::<0x88>(q01_hi, q23_hi);
            out[12 + j] = _mm512_shuffle_i32x4::<0xDD>(q01_hi, q23_hi);
        }
        out
    }
}

/// Load one block from each of sixteen lanes as sixteen message words.
#[inline(always)]
pub(super) fn load_block(rows: &[&[u8; BLOCK_BYTES]; WIDTH]) -> [__m512i; BLOCK_WORDS] {
    // SAFETY: each row is 64 readable bytes, and the load has no alignment requirement.
    transpose(rows.map(|row| unsafe { _mm512_loadu_si512(row.as_ptr().cast()) }))
}

/// Write the digests of sixteen lanes.
#[inline(always)]
pub(super) fn store_digests(state: &[__m512i; 8], out: &mut [[u8; DIGEST_BYTES]; WIDTH]) {
    // SAFETY: this module only compiles when the target enables AVX-512F.
    let zero = unsafe { _mm512_setzero_si512() };

    // The eight state words fill the top half of a square, so row l comes back as lane l.
    let rows = transpose(core::array::from_fn(
        |w| if w < 8 { state[w] } else { zero },
    ));
    for (digest, row) in out.iter_mut().zip(rows) {
        // SAFETY: a digest is 32 writable bytes, and the store has no alignment requirement.
        unsafe { _mm256_storeu_si256(digest.as_mut_ptr().cast(), _mm512_castsi512_si256(row)) };
    }
}
