//! Vector words for the batched compression, one lane per message.
//!
//! Each target compiles exactly one backend, picked from its enabled features at build time.

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[path = "x86_64_avx512.rs"]
mod backend;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
#[path = "x86_64_avx2.rs"]
mod backend;

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[path = "x86_64_sse2.rs"]
mod backend;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
#[path = "aarch64_neon.rs"]
mod backend;

#[cfg(not(any(
    target_arch = "x86_64",
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )
)))]
#[path = "portable.rs"]
mod backend;

use blake3::{BLOCK_LEN, OUT_LEN};

use super::compress::{BLOCK_WORDS, STATE_WORDS};

/// One state or message word for every lane of one register.
pub(super) type Vector = backend::Vector;

/// Lanes in one register.
pub(crate) const WIDTH: usize = backend::WIDTH;

/// Independent register groups hashed together.
///
/// One group leaves the core waiting on the dependency chains of G.
///
/// Too many spill the working vectors out of the register file.
pub(super) const GROUPS: usize = backend::GROUPS;

/// Messages one batched compression advances at once.
pub(crate) const LANES: usize = WIDTH * GROUPS;

/// The arithmetic the compression function needs, on every lane at once.
pub(super) trait Word: Copy {
    /// The same word in every lane.
    fn splat(value: u32) -> Self;

    /// Lane-wise addition modulo 2^32.
    fn add(self, rhs: Self) -> Self;

    /// Lane-wise bitwise and.
    fn and(self, rhs: Self) -> Self;

    /// Lane-wise exclusive or.
    fn xor(self, rhs: Self) -> Self;

    /// Lane-wise rotation right by 16 bits.
    fn rotr_16(self) -> Self;

    /// Lane-wise rotation right by 12 bits.
    fn rotr_12(self) -> Self;

    /// Lane-wise rotation right by 8 bits.
    fn rotr_8(self) -> Self;

    /// Lane-wise rotation right by 7 bits.
    fn rotr_7(self) -> Self;

    /// Advance `G` groups by one block with a hand-scheduled kernel, if this backend has one.
    ///
    /// `params` is the second half of the working vector: IV[0..4], counter, block length, flags.
    ///
    /// Returns false when there is none, and the generic rounds run instead.
    #[inline(always)]
    fn compress_scheduled<const G: usize>(
        _h: &mut [[Self; STATE_WORDS]; G],
        _m: &[[Self; BLOCK_WORDS]; G],
        _params: &[u32; STATE_WORDS],
    ) -> bool {
        false
    }
}

/// Every lane of one vector, in lane order.
#[cfg(test)]
pub(super) const fn to_lanes(vector: Vector) -> [u32; WIDTH] {
    // SAFETY: a vector is exactly its lanes, packed from lane 0 at the lowest address.
    unsafe { core::mem::transmute_copy(&vector) }
}

/// One vector holding the given lanes, in lane order.
#[cfg(test)]
pub(super) const fn from_lanes(lanes: [u32; WIDTH]) -> Vector {
    // SAFETY: a vector is exactly its lanes, packed from lane 0 at the lowest address.
    unsafe { core::mem::transmute_copy(&lanes) }
}

/// Transpose one block from each lane into the sixteen message words of one register group.
///
/// - `rows[l]` is one block of lane `l`.
/// - Word `w` of the result holds word `w` of every lane.
#[inline(always)]
pub(super) fn load_block(rows: &[&[u8; BLOCK_LEN]; WIDTH]) -> [Vector; BLOCK_WORDS] {
    backend::load_block(rows)
}

/// Write the chaining value of every lane of one register group as a digest.
#[inline(always)]
pub(super) fn store_digests(state: &[Vector; STATE_WORDS], out: &mut [[u8; OUT_LEN]; WIDTH]) {
    backend::store_digests(state, out);
}
