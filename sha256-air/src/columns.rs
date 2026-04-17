//! Column layout for the SHA-256 compression AIR.
//!
//! # Overview
//!
//! One trace row encodes one full compression.
//!
//! No transition constraints link adjacent rows.
//!
//! # Representation choices
//!
//! A 32-bit word appears in one of two forms:
//!
//! - Unpacked: 32 boolean columns, one per bit. Used for bitwise operations.
//! - Packed: 2 columns holding `[lo, hi]` 16-bit limbs. Used for additions.
//!
//! Packing identity used everywhere a bridge is needed:
//!
//! ```text
//!     packed[0] = bit[0]  + 2 * bit[1]  + ... + 2^15 * bit[15]
//!     packed[1] = bit[16] + 2 * bit[17] + ... + 2^15 * bit[31]
//! ```
//!
//! # Working-variable shift chain
//!
//! The compression round updates `(a, b, c, d, e, f, g, h)` by shifting:
//!
//! ```text
//!     round t:   (a_t,      b_t,   c_t,   d_t,   e_t,      f_t,   g_t,   h_t)
//!     round t+1: (new_a_t,  a_t,   b_t,   c_t,   new_e_t,  e_t,   f_t,   g_t)
//! ```
//!
//! Two chains capture every value that flows through the "a" and "e" slots.
//!
//! # Chain layout
//!
//! Length: `4 + 64 = 68`.
//!
//! ```text
//!     a_chain[0..4]  = [ H_3, H_2, H_1, H_0 ]
//!     a_chain[4..68] = [ new_a_0, new_a_1, ..., new_a_63 ]
//!
//!     e_chain[0..4]  = [ H_7, H_6, H_5, H_4 ]
//!     e_chain[4..68] = [ new_e_0, new_e_1, ..., new_e_63 ]
//! ```
//!
//! The first four positions are stored in reverse to line up with the round
//! shift: at round 0 the `d` slot holds `H_3`, not `H_0`.
//!
//! # Per-round lookups
//!
//! Round `t` reads its eight working variables with simple offsets:
//!
//! ```text
//!     a = a_chain[t + 3]    e = e_chain[t + 3]
//!     b = a_chain[t + 2]    f = e_chain[t + 2]
//!     c = a_chain[t + 1]    g = e_chain[t + 1]
//!     d = a_chain[t + 0]    h = e_chain[t + 0]
//! ```

use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use crate::constants::{
    NUM_COMPRESSION_ROUNDS, SCHEDULE_EXTENSIONS, STATE_WORDS, U32_LIMBS, WORD_BITS,
};

/// Length of each working-variable chain: four initial-state entries plus one
/// entry per compression round.
pub const CHAIN_LEN: usize = 4 + NUM_COMPRESSION_ROUNDS;

/// Packed intermediates produced by a single compression round.
///
/// Every field is a 2-limb little-endian representation of a 32-bit value:
/// `limbs[0]` holds bits 0..16 and `limbs[1]` holds bits 16..32.
#[repr(C)]
pub struct Sha256RoundCols<T> {
    /// Big-sigma-1 applied to the round's `e` input.
    ///
    /// Formula: `ROTR_6(e) XOR ROTR_11(e) XOR ROTR_25(e)`.
    pub sigma1_e: [T; U32_LIMBS],

    /// The SHA-256 `Ch` combinator.
    ///
    /// Formula: `(e AND f) XOR (NOT e AND g)`.
    pub ch: [T; U32_LIMBS],

    /// First partial sum toward `T_1`.
    ///
    /// Value: `h + sigma1_e + ch  (mod 2^32)`.
    ///
    /// Exists so the five-term `T_1` sum can be split into two three-term
    /// additions, each handled by the standard `add3` helper.
    pub tmp1: [T; U32_LIMBS],

    /// Full value of `T_1`.
    ///
    /// Value: `tmp1 + K[t] + W[t]  (mod 2^32)`.
    pub t1: [T; U32_LIMBS],

    /// Big-sigma-0 applied to the round's `a` input.
    ///
    /// Formula: `ROTR_2(a) XOR ROTR_13(a) XOR ROTR_22(a)`.
    pub sigma0_a: [T; U32_LIMBS],

    /// The SHA-256 `Maj` combinator.
    ///
    /// Formula: `(a AND b) XOR (a AND c) XOR (b AND c)`.
    pub maj: [T; U32_LIMBS],

    /// Full value of `T_2`.
    ///
    /// Value: `sigma0_a + maj  (mod 2^32)`.
    pub t2: [T; U32_LIMBS],

    /// Packed value of the new `a` slot.
    ///
    /// Value: `t1 + t2  (mod 2^32)`.
    ///
    /// The bit decomposition lives in the `a` chain at offset `t + 4`.
    pub new_a_packed: [T; U32_LIMBS],

    /// Packed value of the new `e` slot.
    ///
    /// Value: `d + t1  (mod 2^32)`.
    ///
    /// The bit decomposition lives in the `e` chain at offset `t + 4`.
    pub new_e_packed: [T; U32_LIMBS],
}

/// One trace row.
///
/// Stores the input chaining state, the full message schedule, the 64 rounds
/// of compression auxiliary values, and the output chaining state.
#[repr(C)]
pub struct Sha256Cols<T> {
    /// Input chaining state, one word per index in packed form.
    ///
    /// Committed in packed form so the output addition can feed it directly
    /// into the `add2` helper without an extra repack step.
    pub h_in: [[T; U32_LIMBS]; STATE_WORDS],

    /// Full bit decomposition of every value that flows through the `a` slot.
    ///
    /// See the module-level docs for the layout.
    pub a_chain: [[T; WORD_BITS]; CHAIN_LEN],

    /// Full bit decomposition of every value that flows through the `e` slot.
    ///
    /// See the module-level docs for the layout.
    pub e_chain: [[T; WORD_BITS]; CHAIN_LEN],

    /// Bit decomposition of the full 64-word message schedule.
    ///
    /// Indices 0..16 are witnesses supplied by the prover (the block).
    ///
    /// Indices 16..64 are constrained by the message-schedule recurrence.
    pub w: [[T; WORD_BITS]; NUM_COMPRESSION_ROUNDS],

    /// Packed form of `W[16 + i]`, one entry per expanded word.
    ///
    /// Committed because the add helpers require their output as a committed
    /// `[Var; 2]`. A bridging constraint ties this back to the matching bit
    /// decomposition.
    pub w_packed: [[T; U32_LIMBS]; SCHEDULE_EXTENSIONS],

    /// Packed small-sigma-0 values used by the message schedule.
    ///
    /// Indexed by `i = t - 16`, so entry `i` holds `small_sigma0(W[t - 15])`.
    pub sched_sigma0: [[T; U32_LIMBS]; SCHEDULE_EXTENSIONS],

    /// Packed small-sigma-1 values used by the message schedule.
    ///
    /// Indexed by `i = t - 16`, so entry `i` holds `small_sigma1(W[t - 2])`.
    pub sched_sigma1: [[T; U32_LIMBS]; SCHEDULE_EXTENSIONS],

    /// Message-schedule partial sum, entry `i` holds
    /// `small_sigma1(W[t - 2]) + W[t - 7]` with `i = t - 16`.
    ///
    /// Splits the four-term recurrence into two three-term additions.
    pub sched_tmp: [[T; U32_LIMBS]; SCHEDULE_EXTENSIONS],

    /// Per-round intermediates, one entry per compression round.
    pub rounds: [Sha256RoundCols<T>; NUM_COMPRESSION_ROUNDS],

    /// Output chaining state in packed form.
    ///
    /// Each entry equals the input `H[i]` plus the matching final working
    /// variable, reduced modulo `2^32`.
    pub h_out: [[T; U32_LIMBS]; STATE_WORDS],
}

/// Total number of scalar columns required per row.
///
/// Computed as the byte size of the row struct parameterised over `u8`:
/// every field is a flat array of `u8`, so the byte count equals the column
/// count.
pub const NUM_SHA256_COLS: usize = size_of::<Sha256Cols<u8>>();

impl<T> Borrow<Sha256Cols<T>> for [T] {
    fn borrow(&self) -> &Sha256Cols<T> {
        // Defense in depth: callers must pass a slice sized to exactly one row.
        debug_assert_eq!(self.len(), NUM_SHA256_COLS);
        // Re-interpret the slice as a typed row. Safe because the struct is
        // `repr(C)`, contains only `T` arrays, and the slice is exactly the
        // right size.
        let (prefix, shorts, suffix) = unsafe { self.align_to::<Sha256Cols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<Sha256Cols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut Sha256Cols<T> {
        // Defense in depth: callers must pass a slice sized to exactly one row.
        debug_assert_eq!(self.len(), NUM_SHA256_COLS);
        // Same re-interpretation as the shared borrow path, but mutable.
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<Sha256Cols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
