//! Linear codes applied column-wise to a matrix.

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

/// A linear code applied to every column of a matrix.
pub trait Encoder<F: Field> {
    /// Encodes each column of `message` into a codeword.
    ///
    /// `message` has height `2^k`; the result has the same width and height
    /// `2^(k + log_inv_rate)`. Output row `i` is codeword symbol `i`.
    fn encode_batch(&self, message: RowMajorMatrix<F>, log_inv_rate: usize) -> RowMajorMatrix<F>;
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
        // Appending zero rows extends every column's coefficient vector in place.
        message
            .values
            .resize(message.values.len() << log_inv_rate, F::ZERO);
        self.dft_batch(message).to_row_major_matrix()
    }
}

#[cfg(test)]
mod tests {
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
        let expected = dft.dft_batch(padded).to_row_major_matrix();

        assert_eq!(dft.encode_batch(message, 2), expected);
    }

    #[test]
    fn small_batch_encoder_matches_padded_dft() {
        check_matches_padded_dft(&Radix2DFTSmallBatch::<BabyBear>::default());
    }

    #[test]
    fn dit_parallel_encoder_matches_padded_dft() {
        check_matches_padded_dft(&Radix2DitParallel::<BabyBear>::default());
    }
}
