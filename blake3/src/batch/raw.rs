//! Raw compression with caller-supplied chaining values and parameter words.

use super::compress::{self, BLOCK_WORDS, STATE_WORDS};
use super::lanes::{Backend, from_lanes, to_lanes};

/// Compress one block and return the eight-word chaining value.
///
/// The working vector starts with `cv` in slots `0..8`, the first four BLAKE3 IV words in slots `8..12`, and `parameter_words` verbatim in slots `12..16`.
///
/// Conventionally these four words are counter low, counter high, block length in bytes, and flags.
///
/// All 32 bits of every parameter word are accepted.
///
/// This performs one compression without padding, tree framing, or parameter validation.
///
/// To compress bytes, decode each consecutive four-byte group as one little-endian block word.
#[must_use]
pub fn compress_raw(
    cv: &[u32; STATE_WORDS],
    block: &[u32; BLOCK_WORDS],
    parameter_words: &[u32; 4],
) -> [u32; STATE_WORDS] {
    let mut state = [*cv];
    compress::compress_raw(&mut state, &[*block], parameter_words);
    state[0]
}

/// Compress one block and return all sixteen output words.
///
/// Inputs and parameter placement are the same as [`compress_raw`].
///
/// After the seven rounds, output word `i` is `v[i] ^ v[i + 8]` for `i < 8`, and `v[i] ^ cv[i - 8]` for `i >= 8`, using the original chaining value.
#[must_use]
pub fn compress_raw_xof(
    cv: &[u32; STATE_WORDS],
    block: &[u32; BLOCK_WORDS],
    parameter_words: &[u32; 4],
) -> [u32; BLOCK_WORDS] {
    compress::compress_xof(cv, block, parameter_words)
}

/// Compress `L` independent blocks with shared parameter words.
///
/// Each array is word-major: `cv[word][lane]` and `block[word][lane]` belong to the same independent compression.
///
/// Each output lane equals [`compress_raw`] with that lane's inputs and the shared `parameter_words`, including arbitrary 32-bit parameters.
///
/// Any `L` is supported, including zero.
///
/// Uses the same backend selection as [`crate::Blake3`]'s batched hashing, with full register groups first, single registers next, and scalar tails.
#[must_use]
pub fn compress_raw_packed<const L: usize>(
    cv: &[[u32; L]; STATE_WORDS],
    block: &[[u32; L]; BLOCK_WORDS],
    parameter_words: &[u32; 4],
) -> [[u32; L]; STATE_WORDS] {
    super::detect().compress_raw_packed(cv, block, parameter_words)
}

/// The backend entrypoint erases the public lane count but preserves its stride.
/// Called only through the backend's target-feature wrapper.
#[inline(always)]
pub(super) fn packed_with<V: Backend<W>, const W: usize, const G: usize>(
    cv: &[u32],
    block: &[u32],
    parameter_words: &[u32; 4],
    out: &mut [u32],
    lanes: usize,
) {
    let mut first = 0;
    while lanes - first >= W * G {
        packed_group::<V, W, G>(cv, block, parameter_words, out, lanes, first);
        first += W * G;
    }
    while lanes - first >= W {
        packed_group::<V, W, 1>(cv, block, parameter_words, out, lanes, first);
        first += W;
    }
    for lane in first..lanes {
        let h = core::array::from_fn(|w| cv[w * lanes + lane]);
        let m = core::array::from_fn(|w| block[w * lanes + lane]);
        for (w, value) in compress_raw(&h, &m, parameter_words)
            .into_iter()
            .enumerate()
        {
            out[w * lanes + lane] = value;
        }
    }
}

/// Load full registers directly from word-major inputs, then write their CVs.
#[inline(always)]
fn packed_group<V: Backend<W>, const W: usize, const G: usize>(
    cv: &[u32],
    block: &[u32],
    parameter_words: &[u32; 4],
    out: &mut [u32],
    lanes: usize,
    first: usize,
) {
    let mut state: [[V; STATE_WORDS]; G] = core::array::from_fn(|g| {
        core::array::from_fn(|w| {
            from_lanes::<V, W>(cv[w * lanes + first + g * W..][..W].try_into().unwrap())
        })
    });
    let message: [[V; BLOCK_WORDS]; G] = core::array::from_fn(|g| {
        core::array::from_fn(|w| {
            from_lanes::<V, W>(block[w * lanes + first + g * W..][..W].try_into().unwrap())
        })
    });
    compress::compress_raw(&mut state, &message, parameter_words);
    for (g, group) in state.into_iter().enumerate() {
        for (w, word) in group.into_iter().enumerate() {
            out[w * lanes + first + g * W..][..W].copy_from_slice(&to_lanes::<V, W>(word));
        }
    }
}
