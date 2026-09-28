//! Four-way SHA-256 for x86-64 SHA-NI.
//!
//! A vector holds four consecutive words of *one* message, not one word of four messages.
//! Parallelism therefore comes from running four independent streams side by side.
//!
//! ```text
//!     stream k:   abef[k]  cdgh[k]  w[k][0..4]
//! ```
//!
//! Padding and batching live in [`crate::four_lane`]; this module supplies the vector core.
//!
//! # Performance
//!
//! `sha256rnds2` advances a single state.
//! It has a four-cycle latency for a one-cycle issue rate.
//!
//! One stream is a pure dependency chain, so it leaves three of every four cycles unused.
//! Four interleaved streams fill them, which is why [`LANES`] is four.

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128,
    _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_shuffle_epi8,
    _mm_shuffle_epi32, _mm_storeu_si128,
};
use core::mem::transmute;

use crate::H256_256;
use crate::four_lane::{BLOCK_BYTES, FourLane, LANES, ROUND_CONSTANTS};

/// The four-lane SHA-NI backend.
pub(crate) struct ShaNi;

/// Words of message schedule held in one vector.
///
/// The SHA-NI message instructions consume and produce four words at a time.
const WORDS_PER_VECTOR: usize = 4;

/// Number of four-round groups in one compression.
///
/// SHA-256 has 64 rounds and the kernel issues them four at a time.
const ROUND_GROUPS: usize = 64 / WORDS_PER_VECTOR;

/// Number of message-schedule vectors held live per stream.
///
/// The schedule recurrence reaches back sixteen words, which is four vectors.
const SCHEDULE_VECTORS: usize = 16 / WORDS_PER_VECTOR;

/// Byte permutation reversing each 32-bit lane.
///
/// SHA-256 reads message words big-endian while a vector load is little-endian.
/// The permutation is its own inverse, so the same mask serves loads and stores.
const REVERSE_DWORD_BYTES: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];

/// Immediate for `pshufd` reversing the four lanes.
///
/// Lane `i` of the result takes lane `3 - i` of the source.
const REVERSE_LANES: i32 = 0x1b;

/// Immediate for `pshufd` swapping the lanes within each 64-bit half.
const SWAP_LANE_PAIRS: i32 = 0xb1;

/// Immediate for `pshufd` moving the upper 64 bits down.
///
/// The second `sha256rnds2` of a group needs the third and fourth `W + K` words in the low half.
const HIGH_HALF_TO_LOW: i32 = 0x0e;

/// Immediate for `pblendw` taking the upper two lanes from the second operand.
///
/// The mask is per 16-bit word, so a whole 32-bit lane is two adjacent bits.
const TAKE_UPPER_LANES: i32 = 0xf0;

/// The `abef` half of the SHA-256 initial state, in SHA-NI lane order.
///
/// `sha256rnds2` expects `a` in the top lane and `f` in the bottom one.
const INITIAL_ABEF: [u32; 4] = [H256_256[5], H256_256[4], H256_256[1], H256_256[0]];

/// The `cdgh` half of the SHA-256 initial state, in SHA-NI lane order.
const INITIAL_CDGH: [u32; 4] = [H256_256[7], H256_256[6], H256_256[3], H256_256[2]];

/// The two register halves of one SHA-256 state, for each of the four streams.
#[derive(Clone, Copy)]
pub(crate) struct State {
    /// Working words `a`, `b`, `e`, `f`, packed from the top lane down.
    abef: [__m128i; LANES],
    /// Working words `c`, `d`, `g`, `h`, packed from the top lane down.
    cdgh: [__m128i; LANES],
}

/// The first sixteen message words of one block, four to a vector.
#[inline]
fn load_schedule(block: &[u8; BLOCK_BYTES]) -> [__m128i; SCHEDULE_VECTORS] {
    // SAFETY: `[u8; 16]` and `__m128i` are both 16 bytes and every bit pattern is valid.
    let mask = unsafe { transmute::<[u8; 16], __m128i>(REVERSE_DWORD_BYTES) };

    core::array::from_fn(|vector| {
        // SAFETY: `block` is 64 bytes and `vector < 4`, so the 16-byte unaligned read is in bounds.
        // The module compiles only with `ssse3` (implied by `sse4.1`), which `pshufb` requires.
        unsafe {
            let raw = _mm_loadu_si128(block.as_ptr().add(vector * 16).cast());
            _mm_shuffle_epi8(raw, mask)
        }
    })
}

/// Extend the message schedule by four words.
///
/// # Arguments
///
/// The four arguments are the last sixteen words, oldest vector first.
///
/// ```text
///     oldest = W[i-16..i-12]   older = W[i-12..i-8]
///     recent = W[i-8..i-4]     newest = W[i-4..i]
/// ```
///
/// # Returns
///
/// The vector `W[i..i+4]`.
#[inline]
fn extend_schedule(oldest: __m128i, older: __m128i, recent: __m128i, newest: __m128i) -> __m128i {
    // SAFETY: the module compiles only when `sha` and `sse4.1` are enabled for the crate.
    // `sha256msg1` and `sha256msg2` need `sha`, and `palignr` needs `ssse3`, implied by `sse4.1`.
    unsafe {
        // `sha256msg1` folds sigma0 over the four words fifteen positions back.
        let sigma0 = _mm_sha256msg1_epu32(oldest, older);

        // The recurrence also adds W[i-7..i-3], which straddles `recent` and `newest`.
        let carried = _mm_alignr_epi8::<4>(newest, recent);

        // `sha256msg2` supplies sigma1 of the two previous words, including its own feedback.
        _mm_sha256msg2_epu32(_mm_add_epi32(sigma0, carried), newest)
    }
}

/// Advance one stream by four rounds.
///
/// # Arguments
///
/// - `schedule` holds `W[4g..4g+4]` for round group `g`.
/// - `group` selects the matching four round constants.
///
/// # Panics
///
/// Panics if `group` is not a valid round group. The two call sites below range over
/// `0..ROUND_GROUPS`, so the bound check folds away once this is inlined into them.
#[inline]
fn round_group(abef: &mut __m128i, cdgh: &mut __m128i, schedule: __m128i, group: usize) {
    let constants = ROUND_CONSTANTS.as_chunks::<WORDS_PER_VECTOR>().0[group];

    // SAFETY: `__m128i` and `[u32; 4]` are both 16 bytes and every bit pattern is valid.
    let constants = unsafe { transmute::<[u32; WORDS_PER_VECTOR], __m128i>(constants) };

    // SAFETY: the module compiles only when `sha` is enabled, which the round instruction needs.
    // The addition and the shuffle are `sse2`, always present on x86-64.
    unsafe {
        let round_input = _mm_add_epi32(schedule, constants);

        // The instruction consumes two rounds from the low half and returns the new `abef`.
        *cdgh = _mm_sha256rnds2_epu32(*cdgh, *abef, round_input);

        // Rotating the halves feeds it the remaining two rounds.
        let upper = _mm_shuffle_epi32::<HIGH_HALF_TO_LOW>(round_input);
        *abef = _mm_sha256rnds2_epu32(*abef, *cdgh, upper);
    }
}

impl FourLane for ShaNi {
    type State = State;

    #[inline]
    fn initial_state() -> Self::State {
        // SAFETY: `__m128i` and `[u32; 4]` are both 16 bytes and every bit pattern is valid.
        let abef = unsafe { transmute::<[u32; 4], __m128i>(INITIAL_ABEF) };
        // SAFETY: same as above.
        let cdgh = unsafe { transmute::<[u32; 4], __m128i>(INITIAL_CDGH) };

        State {
            abef: [abef; LANES],
            cdgh: [cdgh; LANES],
        }
    }

    /// Compress one block into each of the four states.
    ///
    /// The first four round groups read the block directly. The remaining twelve overwrite the
    /// oldest schedule vector in place.
    ///
    /// ```text
    ///     slot g % 4 holds W[4g-16 .. 4g-12] on entry to group g
    /// ```
    #[inline]
    fn compress(state: &mut Self::State, blocks: [&[u8; BLOCK_BYTES]; LANES]) {
        let entry = *state;
        let mut schedule: [[__m128i; SCHEDULE_VECTORS]; LANES] =
            core::array::from_fn(|lane| load_schedule(blocks[lane]));

        for group in 0..SCHEDULE_VECTORS {
            let streams = state
                .abef
                .iter_mut()
                .zip(state.cdgh.iter_mut())
                .zip(schedule.iter());
            for ((abef, cdgh), window) in streams {
                round_group(abef, cdgh, window[group], group);
            }
        }

        for group in SCHEDULE_VECTORS..ROUND_GROUPS {
            let streams = state
                .abef
                .iter_mut()
                .zip(state.cdgh.iter_mut())
                .zip(schedule.iter_mut());
            for ((abef, cdgh), window) in streams {
                let next = extend_schedule(
                    window[group % SCHEDULE_VECTORS],
                    window[(group + 1) % SCHEDULE_VECTORS],
                    window[(group + 2) % SCHEDULE_VECTORS],
                    window[(group + 3) % SCHEDULE_VECTORS],
                );
                window[group % SCHEDULE_VECTORS] = next;
                round_group(abef, cdgh, next, group);
            }
        }

        // SAFETY: `paddd` is `sse2`, always available on x86-64.
        unsafe {
            for (abef, saved) in state.abef.iter_mut().zip(entry.abef) {
                *abef = _mm_add_epi32(*abef, saved);
            }
            for (cdgh, saved) in state.cdgh.iter_mut().zip(entry.cdgh) {
                *cdgh = _mm_add_epi32(*cdgh, saved);
            }
        }
    }

    #[inline]
    fn write_digests(state: &Self::State, out: &mut [[u8; 32]; LANES]) {
        // SAFETY: `[u8; 16]` and `__m128i` are both 16 bytes and every bit pattern is valid.
        let mask = unsafe { transmute::<[u8; 16], __m128i>(REVERSE_DWORD_BYTES) };

        let streams = out.iter_mut().zip(&state.abef).zip(&state.cdgh);
        for ((digest, &packed_abef), &packed_cdgh) in streams {
            // SAFETY: the blend needs `sse4.1` and the align needs `ssse3`.
            //
            // Both are guaranteed by the module's `sse4.1` gate.
            //
            // The two stores write sixteen bytes each into the halves of a 32-byte array.
            unsafe {
                // Undo the SHA-NI packing. Every name here reads from the top lane down, as
                // `State` does, so `feba` is the reversal of `abef`.
                let feba = _mm_shuffle_epi32::<REVERSE_LANES>(packed_abef);
                let dchg = _mm_shuffle_epi32::<SWAP_LANE_PAIRS>(packed_cdgh);

                // Recombining them gives the state words in their natural order.
                let dcba = _mm_blend_epi16::<TAKE_UPPER_LANES>(feba, dchg);
                let hgfe = _mm_alignr_epi8::<8>(dchg, feba);

                let bytes = digest.as_mut_ptr();
                _mm_storeu_si128(bytes.cast(), _mm_shuffle_epi8(dcba, mask));
                _mm_storeu_si128(bytes.add(16).cast(), _mm_shuffle_epi8(hgfe, mask));
            }
        }
    }
}
