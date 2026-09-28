//! Four-way SHA-256 for the ARMv8 SHA-2 extension.
//!
//! As on x86-64, a vector holds four consecutive words of *one* message, not one word of four
//! messages, and parallelism comes from running four independent streams side by side.
//!
//! ```text
//!     stream k:   abcd[k]  efgh[k]  w[k][0..4]
//! ```
//!
//! Unlike SHA-NI, the ARM round instructions keep the state in its natural word order, so no
//! lane shuffles are needed on entry or exit.
//!
//! Padding and batching live in [`crate::four_lane`]; this module supplies the vector core.
//!
//! # Performance
//!
//! `sha2` already reaches these instructions on its own, through runtime detection, one block at
//! a time, and the out-of-order core overlaps independent blocks across loop iterations. So unlike
//! SHA-NI, where one stream leaves three of every four cycles idle, a single stream here already
//! runs near the issue rate of the SHA unit, and `compress_many` measures at parity with `sha2`.
//!
//! The gain is on `hash_many`, from the shared padding path together with the interleaving. Of
//! one, two and four streams in flight, four measured fastest at every benchmarked length on an
//! Apple M2, which is why [`LANES`] stays four here too.

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u8_u32, vreinterpretq_u32_u8,
    vrev32q_u8, vsha256h2q_u32, vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u8,
};

use crate::H256_256;
use crate::four_lane::{BLOCK_BYTES, FourLane, LANES, ROUND_CONSTANTS};

/// The four-lane ARMv8 SHA-2 backend.
pub(crate) struct ArmSha2;

/// Words of message schedule held in one vector.
///
/// The ARM message instructions consume and produce four words at a time.
const WORDS_PER_VECTOR: usize = 4;

/// Number of four-round groups in one compression.
///
/// SHA-256 has 64 rounds and the kernel issues them four at a time.
const ROUND_GROUPS: usize = 64 / WORDS_PER_VECTOR;

/// Number of message-schedule vectors held live per stream.
///
/// The schedule recurrence reaches back sixteen words, which is four vectors.
const SCHEDULE_VECTORS: usize = 16 / WORDS_PER_VECTOR;

/// The two register halves of one SHA-256 state, for each of the four streams.
#[derive(Clone, Copy)]
pub(crate) struct State {
    /// Working words `a`, `b`, `c`, `d`, in lane order.
    abcd: [uint32x4_t; LANES],
    /// Working words `e`, `f`, `g`, `h`, in lane order.
    efgh: [uint32x4_t; LANES],
}

/// The first sixteen message words of one block, four to a vector.
#[inline]
fn load_schedule(block: &[u8; BLOCK_BYTES]) -> [uint32x4_t; SCHEDULE_VECTORS] {
    core::array::from_fn(|vector| {
        // SAFETY: `block` is 64 bytes and `vector < 4`, so the 16-byte read is in bounds.
        // The module compiles only with `neon`, which the load and the byte reversal need.
        //
        // SHA-256 reads message words big-endian while a vector load is little-endian, so each
        // 32-bit lane has its bytes reversed.
        unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(vector * 16)))) }
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
fn extend_schedule(
    oldest: uint32x4_t,
    older: uint32x4_t,
    recent: uint32x4_t,
    newest: uint32x4_t,
) -> uint32x4_t {
    // SAFETY: the module compiles only when `sha2` is enabled, which both instructions need.
    unsafe {
        // `sha256su0` folds sigma0 over the four words fifteen positions back.
        let sigma0 = vsha256su0q_u32(oldest, older);

        // `sha256su1` adds the word seven back and sigma1 of the two previous words, feeding its
        // own output back for the upper pair.
        vsha256su1q_u32(sigma0, recent, newest)
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
fn round_group(abcd: &mut uint32x4_t, efgh: &mut uint32x4_t, schedule: uint32x4_t, group: usize) {
    let constants = &ROUND_CONSTANTS.as_chunks::<WORDS_PER_VECTOR>().0[group];

    // SAFETY: the round instructions need `sha2` and the load and addition need `neon`; the
    // module compiles only with both. `constants` is four `u32`s, exactly one vector.
    unsafe {
        let round_input = vaddq_u32(schedule, vld1q_u32(constants.as_ptr()));

        // `sha256h` produces the new `abcd` and `sha256h2` the new `efgh`; the second needs the
        // `abcd` from *before* the first, so it is kept aside.
        let entry_abcd = *abcd;
        *abcd = vsha256hq_u32(*abcd, *efgh, round_input);
        *efgh = vsha256h2q_u32(*efgh, entry_abcd, round_input);
    }
}

impl FourLane for ArmSha2 {
    type State = State;

    #[inline]
    fn initial_state() -> Self::State {
        // SAFETY: `H256_256` is eight `u32`s, so both four-word reads are in bounds, and the
        // module compiles only with `neon`.
        let (abcd, efgh) = unsafe {
            (
                vld1q_u32(H256_256.as_ptr()),
                vld1q_u32(H256_256.as_ptr().add(WORDS_PER_VECTOR)),
            )
        };

        State {
            abcd: [abcd; LANES],
            efgh: [efgh; LANES],
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
        let mut schedule: [[uint32x4_t; SCHEDULE_VECTORS]; LANES] =
            core::array::from_fn(|lane| load_schedule(blocks[lane]));

        for group in 0..SCHEDULE_VECTORS {
            let streams = state
                .abcd
                .iter_mut()
                .zip(state.efgh.iter_mut())
                .zip(schedule.iter());
            for ((abcd, efgh), window) in streams {
                round_group(abcd, efgh, window[group], group);
            }
        }

        for group in SCHEDULE_VECTORS..ROUND_GROUPS {
            let streams = state
                .abcd
                .iter_mut()
                .zip(state.efgh.iter_mut())
                .zip(schedule.iter_mut());
            for ((abcd, efgh), window) in streams {
                let next = extend_schedule(
                    window[group % SCHEDULE_VECTORS],
                    window[(group + 1) % SCHEDULE_VECTORS],
                    window[(group + 2) % SCHEDULE_VECTORS],
                    window[(group + 3) % SCHEDULE_VECTORS],
                );
                window[group % SCHEDULE_VECTORS] = next;
                round_group(abcd, efgh, next, group);
            }
        }

        // SAFETY: `vaddq_u32` is `neon`, which the module requires.
        unsafe {
            for (abcd, saved) in state.abcd.iter_mut().zip(entry.abcd) {
                *abcd = vaddq_u32(*abcd, saved);
            }
            for (efgh, saved) in state.efgh.iter_mut().zip(entry.efgh) {
                *efgh = vaddq_u32(*efgh, saved);
            }
        }
    }

    #[inline]
    fn write_digests(state: &Self::State, out: &mut [[u8; 32]; LANES]) {
        let streams = out.iter_mut().zip(&state.abcd).zip(&state.efgh);
        for ((digest, &abcd), &efgh) in streams {
            // SAFETY: the byte reversal and the stores are `neon`, which the module requires.
            // The two stores write sixteen bytes each into the halves of a 32-byte array.
            //
            // The state is already in natural word order, so each half only needs its words
            // turned big-endian.
            unsafe {
                let bytes = digest.as_mut_ptr();
                vst1q_u8(bytes, vrev32q_u8(vreinterpretq_u8_u32(abcd)));
                vst1q_u8(bytes.add(16), vrev32q_u8(vreinterpretq_u8_u32(efgh)));
            }
        }
    }
}
