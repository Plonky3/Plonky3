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

use crate::DIGEST_BYTES;
use crate::batch::compress::{BLOCK_BYTES, BLOCK_WORDS, STATE_WORDS};

/// One state or message word for every lane of one register.
pub(super) type Vector = backend::Vector;

/// Lanes in one register.
const WIDTH: usize = backend::WIDTH;

/// Independent register groups hashed together, as many as this backend wants in flight.
///
/// One group leaves the pipeline waiting on the dependency chains of G, and too many spill
/// the working vectors out of the register file. Where the balance falls is a property of
/// the target, so each backend carries its own count and the reason for it.
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

    /// Lane-wise exclusive or of three words.
    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        self.xor(b).xor(c)
    }

    /// Lane-wise rotation right by 16 bits.
    fn rotr_16(self) -> Self;

    /// Lane-wise rotation right by 12 bits.
    fn rotr_12(self) -> Self;

    /// Lane-wise rotation right by 8 bits.
    fn rotr_8(self) -> Self;

    /// Lane-wise rotation right by 7 bits.
    fn rotr_7(self) -> Self;
}

/// Every lane of one vector, in lane order.
#[cfg(test)]
pub(super) const fn to_lanes(vector: Vector) -> [u32; WIDTH] {
    // SAFETY: a vector is exactly its lanes, packed from lane 0 at the lowest address.
    unsafe { core::mem::transmute_copy(&vector) }
}

/// Transpose one block from each lane into the sixteen message words of each group.
///
/// ```text
///     rows[l]     = [ m_l[0], m_l[1], ..., m_l[15] ]    one block of message l
///     block[g][w] = word w of messages g * WIDTH .. (g + 1) * WIDTH
/// ```
#[inline(always)]
pub(super) fn load_block(rows: &[&[u8; BLOCK_BYTES]; LANES]) -> [[Vector; BLOCK_WORDS]; GROUPS] {
    let (groups, _) = rows.as_chunks::<WIDTH>();
    core::array::from_fn(|g| backend::load_block(&groups[g]))
}

/// Write the digest of every lane out of the chaining values of every group.
#[inline(always)]
pub(super) fn store_digests(
    state: &[[Vector; STATE_WORDS]; GROUPS],
    out: &mut [[u8; DIGEST_BYTES]; LANES],
) {
    let (groups, _) = out.as_chunks_mut::<WIDTH>();
    for (state, out) in state.iter().zip(groups) {
        backend::store_digests(state, out);
    }
}
