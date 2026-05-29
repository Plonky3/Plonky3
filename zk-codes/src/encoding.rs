use alloc::vec::Vec;

use p3_field::Field;
use rand::Rng;

/// A randomized encoding such that any `t` codeword positions reveal nothing about the message.
pub trait ZkEncoding<F: Field> {
    /// The codeword type produced by the encoding.
    type Codeword;

    /// The length of the message being encoded.
    fn message_len(&self) -> usize;

    /// The length of the randomness used for the encoding.
    fn randomness_len(&self) -> usize;

    /// The maximum number of queries that can be perfectly simulated.
    fn query_bound(&self) -> usize;

    /// Statistical simulation error:
    fn error(&self) -> f64;

    /// Samples a uniformly random message from this encoding's message space.
    ///
    /// # Returns
    ///
    /// A vector of length `message_len`.
    ///
    /// # Why on the encoding
    ///
    /// - The message space is an encoding invariant.
    /// - Constrained message spaces (e.g. punctured codes) need a non-trivial draw.
    /// - Callers stay agnostic to the field's sampling bound.
    fn sample_message<R: Rng>(&self, rng: &mut R) -> Vec<F>;

    /// Encodes a message with random masking.
    fn encode<R: Rng>(&self, msg: &[F], rng: &mut R) -> Self::Codeword;

    /// Produces identically distributed evaluations without access to the message.
    ///
    /// - The number of distinct positions must not exceed the query bound.
    /// - Duplicate positions yield the same value (matching the real codeword).
    /// - The output distribution is identical (or close, up to the simulation error)
    ///   to the real codeword at the queried positions.
    fn simulate<R: Rng>(&self, query_set: &[usize], rng: &mut R) -> Vec<F>;
}

/// A zero-knowledge encoding that supports deterministic encoding when randomness is provided.
pub trait ZkEncodingWithRandomness<F: Field>: ZkEncoding<F> {
    /// Encodes a message using an explicitly provided randomness array.
    fn encode_with_randomness(&self, msg: &[F], randomness: &[F]) -> Self::Codeword;
}

/// A zero-knowledge encoding whose generator matrix rows can be accessed.
pub trait LinearZkEncoding<F: Field>: ZkEncoding<F> {
    /// Returns the generator-matrix row for the message part.
    ///
    /// This is `G^#` of Definition 3.17 of eprint 2026/391.
    fn message_row(&self, position: usize) -> Vec<F>;

    /// Returns the generator-matrix row for the randomness part.
    ///
    /// This is `G^$` of Definition 3.17 of eprint 2026/391.
    fn randomness_row(&self, position: usize) -> Vec<F>;
}
