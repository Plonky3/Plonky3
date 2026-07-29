use alloc::vec::Vec;

use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
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

    /// The length of the codeword (the domain size that query positions range over).
    fn codeword_len(&self) -> usize;

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

    /// Samples encoding randomness from this encoding's randomness space.
    ///
    /// # Returns
    ///
    /// A vector of length `randomness_len`.
    ///
    /// # Why an explicit draw
    ///
    /// - Some callers later reveal blinded linear combinations of the
    ///   randomness (e.g. an HVZK base case).
    /// - Those callers draw it here and encode with the explicit-randomness
    ///   entry point, retaining the vector.
    fn sample_randomness<R: Rng>(&self, rng: &mut R) -> Vec<F>;

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

    /// Encodes same-shape messages as a single width-`k` matrix.
    ///
    /// The default preserves the scalar semantics by encoding each column
    /// independently, then interleaving the resulting codewords by row.
    ///
    /// # Panics
    ///
    /// Panics if the batch is empty, if the message and randomness batch
    /// lengths differ, or if any member has an invalid shape for this encoding.
    fn encode_batch_with_randomness<Msg, Rand>(
        &self,
        messages: &[Msg],
        randomness: &[Rand],
    ) -> RowMajorMatrix<F>
    where
        Msg: AsRef<[F]>,
        Rand: AsRef<[F]>,
        Self::Codeword: Matrix<F>,
    {
        assert!(
            !messages.is_empty(),
            "batch must contain at least one message"
        );
        assert_eq!(
            messages.len(),
            randomness.len(),
            "batch message/randomness counts must match"
        );
        let codewords: Vec<Self::Codeword> = messages
            .iter()
            .zip(randomness)
            .map(|(message, randomness)| {
                self.encode_with_randomness(message.as_ref(), randomness.as_ref())
            })
            .collect();
        stack_codewords(&codewords)
    }
}

/// Interleaves same-domain codewords into one width-`k` matrix.
///
/// ```text
///     row z = ( cw_1(z), cw_2(z), ..., cw_k(z) )
/// ```
///
/// # Panics
///
/// Panics if the batch is empty, if any codeword is not a single column, or if
/// the codewords do not all have the same height.
pub fn stack_codewords<F: Field, Cw: Matrix<F>>(codewords: &[Cw]) -> RowMajorMatrix<F> {
    assert!(
        !codewords.is_empty(),
        "batch must contain at least one codeword"
    );
    let height = codewords[0].height();
    let width = codewords.len();
    let mut values = F::zero_vec(height * width);
    for (column, codeword) in codewords.iter().enumerate() {
        assert_eq!(codeword.width(), 1, "each codeword must be a column");
        assert_eq!(
            codeword.height(),
            height,
            "all codewords must have the same height"
        );
        for (row, value) in codeword.rows().enumerate() {
            values[row * width + column] = value.into_iter().next().unwrap();
        }
    }
    RowMajorMatrix::new(values, width)
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
