//! Stateless adapter between values and a challenger.

use alloc::vec::Vec;

use crate::fs::error::TranscriptError;

/// Statistical-security budget a codec must meet before it may sample a challenge.
///
/// 128 bits is the budget the DSFS analysis and IETF §6 work with.
///
/// Enforced by the sampling methods of both drivers, so a codec whose decoding
/// is measurably biased cannot silently be used for challenge material.
pub const MIN_CHALLENGE_SECURITY_BITS: u32 = 128;

/// Stateless absorb / sample / serialize adapter for values of type `T` against challenger `C`.
///
/// Codecs are zero-sized types picked at the call site.
///
/// One transcript may invoke several codecs in different roles.
///
/// A codec owns both halves of a step:
///
/// - the sponge half, [`Codec::observe`] and [`Codec::sample`];
/// - the wire half, [`Codec::encode`] and [`Codec::decode`].
///
/// Keeping them together is what stops the two from drifting apart.
pub trait Codec<C, T> {
    /// Bits of statistical security: `-log2` distance from uniform on `T`.
    const SECURITY_BITS: u32;

    /// Bytes one value of `T` occupies on the wire.
    fn wire_len() -> usize;

    /// Absorb `value` into the challenger.
    fn observe(challenger: &mut C, value: &T);

    /// Sample a fresh value from the challenger.
    fn sample(challenger: &mut C) -> T;

    /// Append the canonical wire encoding of `value` to `out`.
    ///
    /// Writes exactly [`Codec::wire_len`] bytes.
    fn encode(value: &T, out: &mut Vec<u8>);

    /// Decode one value from the first [`Codec::wire_len`] bytes of `bytes`.
    ///
    /// # Errors
    ///
    /// When `bytes` is short, or does not hold a canonical encoding of a `T`.
    fn decode(bytes: &[u8]) -> Result<T, TranscriptError>;
}
