//! Reed–Solomon encoder over the additive NTT domain.

use core::marker::PhantomData;

use p3_binary_field::BinaryField128;
use p3_commit::Encoder;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::lch::LchNtt;
use crate::traits::AdditiveNtt;

/// Reed–Solomon over the additive NTT domain.
///
/// The message holds the low-index novel-basis coefficients of each column, so the codeword is
/// the evaluation of `f̂(Ŵ_0(x), …, Ŵ_{k−1}(x))` on `S_{k + log_inv_rate}`.
#[derive(Clone, Debug, Default)]
pub struct AdditiveRsEncoder<F, Ntt = LchNtt<F>> {
    ntt: Ntt,
    _marker: PhantomData<F>,
}

/// The alphabet is fixed at `BinaryField128` (D9), as [`Encoder`] requires of every impl outside
/// `p3-commit`'s blanket one.
impl<Ntt: AdditiveNtt<BinaryField128>> Encoder<BinaryField128>
    for AdditiveRsEncoder<BinaryField128, Ntt>
{
    fn encode_batch(
        &self,
        mut message: RowMajorMatrix<BinaryField128>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField128> {
        if log_inv_rate > 0 {
            // Appending zero rows extends every column's coefficient vector.
            let len = message.values.len();
            let padded_len = u32::try_from(log_inv_rate)
                .ok()
                .and_then(|rate| len.checked_shl(rate))
                .expect("log_inv_rate must be smaller than the width of usize");
            let mut values = BinaryField128::zero_vec(padded_len);
            values[..len].copy_from_slice(&message.values);
            message.values = values;
        }
        self.ntt.ntt_batch(message)
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_commit::Encoder;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::AdditiveRsEncoder;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    type F = BinaryField128;

    /// Encoding is zero-padding the novel-basis coefficients and evaluating on the whole domain.
    #[test]
    fn encodes_by_padding_and_transforming() {
        let mut rng = SmallRng::seed_from_u64(1);
        let message = RowMajorMatrix::<F>::rand(&mut rng, 1 << 5, 3);

        let mut padded = message.clone();
        padded.values.resize(message.values.len() * 4, F::ZERO);
        let expected = NaiveAdditiveNtt::<F>::default().ntt_batch(padded);

        let encoded = AdditiveRsEncoder::<F>::default().encode_batch(message, 2);
        assert_eq!(encoded.height(), 1 << 7);
        assert_eq!(encoded, expected);
    }

    /// The codeword restricted to the message-sized prefix is the message's own transform: the
    /// correspondence Phase 3 folds along.
    #[test]
    fn codeword_prefix_is_the_message_transform() {
        let mut rng = SmallRng::seed_from_u64(2);
        let message = RowMajorMatrix::<F>::rand(&mut rng, 1 << 5, 2);

        let encoded = AdditiveRsEncoder::<F>::default().encode_batch(message.clone(), 1);
        let direct = AdditiveRsEncoder::<F>::default().encode_batch(message, 0);
        assert_eq!(&encoded.values[..direct.values.len()], &direct.values[..]);
    }
}
