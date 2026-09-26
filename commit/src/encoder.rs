//! Linear codes applied column-wise to a matrix.

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};

/// Elements one row of the default borrowed-message copy holds, so each row is a task of its
/// own that outweighs the fork-join overhead.
const COPY_CHUNK: usize = 1 << 16;

/// A linear code applied to every column of a matrix.
///
/// The blanket impl below covers every [`TwoAdicSubgroupDft`], which restricts what an
/// implementor may write: `impl<F: Field> Encoder<F> for MyEncoder` overlaps it and is rejected
/// (E0119), since a downstream crate could implement [`TwoAdicSubgroupDft`] for `MyEncoder`. An
/// impl must therefore name the concrete field(s) it encodes over, as in
/// `impl Encoder<MyField> for MyEncoder`.
///
/// The randomized counterpart is `p3_zk_codes::ZkEncoding`, whose codewords additionally hide the
/// message from a bounded number of queries.
pub trait Encoder<F: Field> {
    /// Encodes each column of `message` into a codeword.
    ///
    /// `message` has height `2^k`; the result has the same width and height
    /// `2^(k + log_inv_rate)`. Output row `i` is codeword symbol `i`.
    ///
    /// # Panics
    /// Panics if the height of `message` is not a power of two, or if the codeword height
    /// `2^(k + log_inv_rate)` overflows `usize`.
    fn encode_batch(&self, message: RowMajorMatrix<F>, log_inv_rate: usize) -> RowMajorMatrix<F>;

    /// Encodes a coefficient matrix already padded to its final height.
    ///
    /// The original message occupies the first `height / 2^log_inv_rate` rows; the
    /// caller must set the remaining entries to zero. Implementations may ignore
    /// that tail and use the rate to avoid work on zero coefficients.
    /// The default preserves the ordinary transform of the full padded matrix.
    ///
    /// # Panics
    /// Panics if the height is not a power of two or the padding exceeds its height.
    fn encode_batch_padded(
        &self,
        message: RowMajorMatrix<F>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<F> {
        let log_height = p3_util::log2_strict_usize(message.height());
        assert!(log_inv_rate <= log_height, "padding exceeds matrix height");
        self.encode_batch(message, 0)
    }

    /// Encodes each column of a borrowed `message` into a codeword, leaving the message as it is.
    ///
    /// The codeword is the one [`Self::encode_batch_padded`] makes of the message zero-padded to
    /// `2^(k + log_inv_rate)` rows. The default builds that padded matrix, copying the message
    /// into it; an encoder that reads the message where it lies skips the copy.
    ///
    /// # Panics
    /// Panics if the height of `message` is not a power of two, or if the codeword height
    /// `2^(k + log_inv_rate)` overflows `usize`.
    fn encode_batch_borrowed(
        &self,
        message: RowMajorMatrixView<'_, F>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<F> {
        let len = message.values.len();
        let padded_len = u32::try_from(log_inv_rate)
            .ok()
            .and_then(|rate| len.checked_shl(rate))
            // `checked_shl` only rejects a shift amount that is too wide, so recovering `len`
            // from the shifted result is what proves no bits were lost.
            .filter(|&padded| padded >> log_inv_rate == len)
            .expect("codeword length overflows usize");
        let mut values = F::zero_vec(padded_len);
        // Rows of whole chunks copy in parallel, and a message no chunk divides is copied at once.
        if len.is_multiple_of(COPY_CHUNK) {
            RowMajorMatrixViewMut::new(&mut values[..len], COPY_CHUNK)
                .copy_from(&RowMajorMatrixView::new(message.values, COPY_CHUNK));
        } else {
            values[..len].copy_from_slice(message.values);
        }
        self.encode_batch_padded(RowMajorMatrix::new(values, message.width), log_inv_rate)
    }
}

/// Reed-Solomon over the two-adic subgroup of order `2^(k + log_inv_rate)`: each column of
/// `message` is the low-degree coefficient vector of a polynomial, and the codeword is its
/// evaluation vector on that subgroup.
impl<F: TwoAdicField, D: TwoAdicSubgroupDft<F>> Encoder<F> for D {
    fn encode_batch(
        &self,
        mut message: RowMajorMatrix<F>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<F> {
        if log_inv_rate > 0 {
            // Appending zero rows extends every column's coefficient vector.
            let len = message.values.len();
            let padded_len = u32::try_from(log_inv_rate)
                .ok()
                .and_then(|rate| len.checked_shl(rate))
                // `checked_shl` only rejects a shift amount that is too wide; it does not
                // detect the value itself overflowing, so recovering `len` from the shifted
                // result is what actually proves no bits were lost.
                .filter(|&padded| padded >> log_inv_rate == len)
                .expect("codeword length overflows usize");
            let mut values = F::zero_vec(padded_len);
            values[..len].copy_from_slice(&message.values);
            message.values = values;
        }
        self.dft_batch(message).to_row_major_matrix()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::Encoder;

    /// `encode_batch` must agree with zero-padding the message and calling `dft_batch`.
    fn check_matches_padded_dft<D: TwoAdicSubgroupDft<BabyBear>>(dft: &D) {
        let mut rng = SmallRng::seed_from_u64(1);
        let message = RowMajorMatrix::<BabyBear>::rand(&mut rng, 8, 3);

        let mut padded = message.clone();
        padded
            .values
            .resize(message.values.len() * 4, BabyBear::ZERO);
        let expected = dft.dft_batch(padded.clone()).to_row_major_matrix();

        assert_eq!(dft.encode_batch_padded(padded, 2), expected);
        assert_eq!(dft.encode_batch_borrowed(message.as_view(), 2), expected);
        assert_eq!(dft.encode_batch(message, 2), expected);
    }

    #[test]
    fn a_borrowed_message_long_enough_to_copy_in_chunks_encodes_as_its_padding() {
        // The message fills whole copy chunks, so the default copies it row by row in parallel.
        let mut rng = SmallRng::seed_from_u64(2);
        let message = RowMajorMatrix::<BabyBear>::rand(&mut rng, 1 << 14, 8);
        assert!(message.values.len().is_multiple_of(super::COPY_CHUNK));

        let dft = Radix2DitParallel::<BabyBear>::default();
        let expected = dft.encode_batch(message.clone(), 1);
        assert_eq!(dft.encode_batch_borrowed(message.as_view(), 1), expected);
    }

    #[test]
    fn small_batch_encoder_matches_padded_dft() {
        check_matches_padded_dft(&Radix2DFTSmallBatch::<BabyBear>::default());
    }

    #[test]
    fn dit_parallel_encoder_matches_padded_dft() {
        check_matches_padded_dft(&Radix2DitParallel::<BabyBear>::default());
    }

    #[test]
    #[should_panic = "codeword length overflows usize"]
    fn encode_batch_panics_when_the_codeword_length_overflows() {
        let message = RowMajorMatrix::<BabyBear>::new(vec![BabyBear::ZERO; 2], 1);
        let _ = Radix2DitParallel::<BabyBear>::default()
            .encode_batch(message, usize::BITS as usize - 1);
    }
}
