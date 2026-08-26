//! Linear codes applied column-wise to a matrix.

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

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
    /// Panics if the height of `message` is not a power of two, or if `log_inv_rate` is at least
    /// the width of `usize`.
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
        if log_inv_rate > 0 {
            // Appending zero rows extends every column's coefficient vector.
            let len = message.values.len();
            let padded_len = u32::try_from(log_inv_rate)
                .ok()
                .and_then(|rate| len.checked_shl(rate))
                .expect("log_inv_rate must be smaller than the width of usize");
            let mut values = F::zero_vec(padded_len);
            values[..len].copy_from_slice(&message.values);
            message.values = values;
        }
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
