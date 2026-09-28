//! Four lanes per 128-bit SIMD128 register.

use core::arch::wasm32::*;

use blake3::{BLOCK_LEN, OUT_LEN};

use super::{Backend, Kernel, Word};
use crate::batch::compress::{BLOCK_WORDS, STATE_WORDS};

/// Lanes in one register.
const WIDTH: usize = 4;

/// Independent register groups hashed together.
///
/// The engine maps each register onto a host one, and x86-64 hosts have only 16.
///
/// A second group spills more than its extra chains recover.
const GROUPS: usize = 1;

impl Word for v128 {
    #[inline(always)]
    fn splat(value: u32) -> Self {
        u32x4_splat(value)
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        u32x4_add(self, rhs)
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        v128_and(self, rhs)
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        v128_xor(self, rhs)
    }

    #[inline(always)]
    fn rotr_16(self) -> Self {
        // A byte shuffle moves each word's bytes [0, 1, 2, 3] to [2, 3, 0, 1].
        u8x16_shuffle::<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(self, self)
    }

    #[inline(always)]
    fn rotr_12(self) -> Self {
        v128_or(u32x4_shr(self, 12), u32x4_shl(self, 20))
    }

    #[inline(always)]
    fn rotr_8(self) -> Self {
        // A byte shuffle moves each word's bytes [0, 1, 2, 3] to [1, 2, 3, 0].
        u8x16_shuffle::<1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12>(self, self)
    }

    #[inline(always)]
    fn rotr_7(self) -> Self {
        v128_or(u32x4_shr(self, 7), u32x4_shl(self, 25))
    }
}

/// Transpose a 4 x 4 matrix of words held one row per register.
#[inline]
#[target_feature(enable = "simd128")]
fn transpose([a, b, c, d]: [v128; 4]) -> [v128; 4] {
    // Interleave 32-bit words of row pairs, then 64-bit pairs of those.
    let ab_lo = u32x4_shuffle::<0, 4, 1, 5>(a, b);
    let ab_hi = u32x4_shuffle::<2, 6, 3, 7>(a, b);
    let cd_lo = u32x4_shuffle::<0, 4, 1, 5>(c, d);
    let cd_hi = u32x4_shuffle::<2, 6, 3, 7>(c, d);
    [
        u64x2_shuffle::<0, 2>(ab_lo, cd_lo),
        u64x2_shuffle::<1, 3>(ab_lo, cd_lo),
        u64x2_shuffle::<0, 2>(ab_hi, cd_hi),
        u64x2_shuffle::<1, 3>(ab_hi, cd_hi),
    ]
}

/// Load one block from each of four lanes as sixteen message words.
#[inline]
#[target_feature(enable = "simd128")]
fn load(rows: &[&[u8; BLOCK_LEN]; WIDTH]) -> [v128; BLOCK_WORDS] {
    // Each block is four registers, and quarter k holds words 4k to 4k + 3.
    //
    // SAFETY: each quarter is 16 readable bytes, and the load has no alignment requirement.
    //
    // wasm32 is little-endian, so the bytes of each word land in the order BLAKE3 reads.
    let quarters: [[v128; 4]; 4] = core::array::from_fn(|k| {
        transpose(core::array::from_fn(|l| unsafe {
            v128_load(rows[l][16 * k..].as_ptr().cast())
        }))
    });
    core::array::from_fn(|w| quarters[w / 4][w % 4])
}

/// Write the digests of four lanes.
#[inline]
#[target_feature(enable = "simd128")]
fn store(state: &[v128; STATE_WORDS], out: &mut [[u8; OUT_LEN]; WIDTH]) {
    // Words 0 to 3, then words 4 to 7, of every lane.
    let low = transpose([state[0], state[1], state[2], state[3]]);
    let high = transpose([state[4], state[5], state[6], state[7]]);
    for ((digest, low), high) in out.iter_mut().zip(low).zip(high) {
        // SAFETY: each half of a digest is 16 writable bytes, with no alignment requirement.
        unsafe {
            v128_store(digest.as_mut_ptr().cast(), low);
            v128_store(digest[16..].as_mut_ptr().cast(), high);
        }
    }
}

/// The batched driver on this backend.
pub(super) const KERNEL: Kernel = Kernel::new::<v128, WIDTH, GROUPS>("SIMD128");

impl Backend<WIDTH> for v128 {
    #[inline]
    fn supported() -> bool {
        // An engine without SIMD128 rejects the whole module, so a running one has it.
        true
    }

    /// Load one block from each of four lanes as sixteen message words.
    #[inline(always)]
    fn load_block(rows: &[&[u8; BLOCK_LEN]; WIDTH]) -> [Self; BLOCK_WORDS] {
        load(rows)
    }

    /// Write the digests of four lanes.
    #[inline(always)]
    fn store_digests(state: &[Self; STATE_WORDS], out: &mut [[u8; OUT_LEN]; WIDTH]) {
        store(state, out);
    }

    out_of_line_steps!(WIDTH, "simd128");
}
