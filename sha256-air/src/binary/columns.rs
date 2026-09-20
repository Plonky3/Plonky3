use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

use super::{SCHEDULE_CARRIES, T1_CARRIES};
use crate::{CHAIN_LEN, NUM_COMPRESSION_ROUNDS, SCHEDULE_EXTENSIONS, STATE_WORDS};

/// Columns for a binary SHA-256 AIR which computes one compression per row.
///
/// Words are stored as 32 bits, least significant bit first.
///
/// # Working state
///
/// The chains hold every value that passes through the `a` and `e` slots:
///
/// ```text
///     a_chain[0..4]  = [ H_3, H_2, H_1, H_0 ]      e_chain[0..4]  = [ H_7, H_6, H_5, H_4 ]
///     a_chain[4..68] = new a of rounds 0..64        e_chain[4..68] = new e of rounds 0..64
/// ```
///
/// Round `t` reads `a, b, c, d = a_chain[t + 3], [t + 2], [t + 1], [t]`, and `e, f, g, h`
/// from `e_chain` at the same offsets.
///
/// # Inputs
///
/// The input chaining value is `a_chain[0..4]` and `e_chain[0..4]`, and the message block is
/// `w[0..16]`. These are the only columns constrained to be boolean. Every other column is
/// then forced to a bit by the constraints that define it.
#[repr(C)]
pub struct Sha256BinaryCols<T> {
    /// Every value that passes through the `a` slot.
    pub a_chain: [[T; 32]; CHAIN_LEN],

    /// Every value that passes through the `e` slot.
    pub e_chain: [[T; 32]; CHAIN_LEN],

    /// The message schedule. Words `0..16` are the block.
    pub w: [[T; 32]; NUM_COMPRESSION_ROUNDS],

    /// Witness columns for the schedule words `w[16..64]`, indexed by `t - 16`.
    pub schedule: [Sha256BinaryScheduleCols<T>; SCHEDULE_EXTENSIONS],

    /// Witness columns for every compression round.
    pub rounds: [Sha256BinaryRoundCols<T>; NUM_COMPRESSION_ROUNDS],

    /// The output chaining value, `H[i] + (a, b, c, d, e, f, g, h)[i] mod 2^32`.
    pub h_out: [[T; 32]; STATE_WORDS],
}

/// Witness columns for one schedule word, `W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16]`.
#[repr(C)]
pub struct Sha256BinaryScheduleCols<T> {
    /// Carries into bits `1..32` of the partial sums, in the order the words are added.
    ///
    /// The last addition reads its carries off `W[t]`, so it needs none.
    pub carries: [[T; 31]; SCHEDULE_CARRIES],
}

/// Witness columns for one compression round.
///
/// ```text
///     T1 = h + Σ1(e) + Ch(e, f, g) + K[t] + W[t]      new e = d + T1
///     T2 = Σ0(a) + Maj(a, b, c)                       new a = T1 + T2
/// ```
///
/// `Ch` and `Maj` have degree 2, and every addition needs linear operands, so both are
/// stored. `new e` and `new a` are the next entries of the chains.
#[repr(C)]
pub struct Sha256BinaryRoundCols<T> {
    /// `Ch(e, f, g) = (e AND f) XOR (NOT e AND g)`.
    pub ch: [T; 32],

    /// `Maj(a, b, c) = (a AND b) XOR (a AND c) XOR (b AND c)`.
    pub maj: [T; 32],

    /// Carries into bits `1..32` of the partial sums of `T1`, in the order the words are added.
    pub t1_carries: [[T; 31]; T1_CARRIES],

    /// The word `T1`.
    pub t1: [T; 32],

    /// Carries into bits `1..32` of `T1 + Σ0(a)`.
    ///
    /// Adding `Maj` to that reads its carries off `new a`.
    pub new_a_carries: [T; 31],
}

/// Number of main trace columns of the binary SHA-256 AIR.
pub const NUM_SHA256_BINARY_COLS: usize = size_of::<Sha256BinaryCols<u8>>();

impl<T> Borrow<Sha256BinaryCols<T>> for [T] {
    fn borrow(&self) -> &Sha256BinaryCols<T> {
        debug_assert_eq!(self.len(), NUM_SHA256_BINARY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<Sha256BinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<Sha256BinaryCols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut Sha256BinaryCols<T> {
        debug_assert_eq!(self.len(), NUM_SHA256_BINARY_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<Sha256BinaryCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
