//! Reed–Solomon encoder over the additive NTT domain.

use core::marker::PhantomData;

use p3_binary_field::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Poly64, TowerLevel,
};
use p3_commit::Encoder;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use crate::lch::LchNtt;
use crate::poly::PolyBasisNtt;
use crate::traits::AdditiveNtt;

/// The length a message grows to once its coefficients are zero-padded to the target rate.
///
/// # Panics
///
/// Panics if that length does not fit the address space.
pub(crate) fn padded_message_len(len: usize, log_inv_rate: usize) -> usize {
    // A shift amount below the word size still leaves the value itself free to overflow.
    // Recovering the original length from the shifted one is what proves no bits were lost.
    u32::try_from(log_inv_rate)
        .ok()
        .and_then(|rate| len.checked_shl(rate))
        .filter(|&padded| padded >> log_inv_rate == len)
        .expect("codeword length overflows usize")
}

/// Reed–Solomon over the additive NTT domain.
///
/// The message holds the low-index novel-basis coefficients of each column, so the codeword is
/// the evaluation of `f̂(Ŵ_0(x), …, Ŵ_{k−1}(x))` on `S_{k + log_inv_rate}`.
///
/// The transform is a type parameter, because the fastest one differs by alphabet.
/// The default is the one the widest tower level uses.
#[derive(Clone, Debug, Default)]
pub struct AdditiveRsEncoder<F, Ntt = PolyBasisNtt> {
    ntt: Ntt,
    _marker: PhantomData<F>,
}

impl<F, Ntt> AdditiveRsEncoder<F, Ntt> {
    /// Builds an encoder around the given additive NTT.
    pub const fn new(ntt: Ntt) -> Self {
        Self {
            ntt,
            _marker: PhantomData,
        }
    }
}

/// A tower level this crate encodes over, together with the encoder it picks for that level.
///
/// A caller naming a level rather than a transform reaches the right pair through this.
/// Nothing but the implementations below decides which transform an alphabet gets.
pub trait EncodableLevel: TowerLevel {
    /// The Reed-Solomon encoder this crate uses for this level.
    type Encoder: Encoder<Self> + Default + Sync;
}

/// The widest level has a polynomial-basis transform.
/// It falls back to the portable tower transform where the target has no carryless multiply.
impl EncodableLevel for BinaryField128 {
    type Encoder = AdditiveRsEncoder<Self, PolyBasisNtt>;
}

impl EncodableLevel for BinaryField64 {
    type Encoder = AdditiveRsEncoder<Self, LchNtt<Self>>;
}

impl EncodableLevel for Poly64 {
    type Encoder = AdditiveRsEncoder<Self, LchNtt<Self>>;
}

impl EncodableLevel for BinaryField32 {
    type Encoder = AdditiveRsEncoder<Self, LchNtt<Self>>;
}

impl EncodableLevel for BinaryField16 {
    type Encoder = AdditiveRsEncoder<Self, LchNtt<Self>>;
}

impl EncodableLevel for BinaryField8 {
    type Encoder = AdditiveRsEncoder<Self, LchNtt<Self>>;
}

/// Zero-extend each column's coefficient vector to the codeword length, then transform.
///
/// The rate is passed through, so a transform may skip the work the zero tail would do.
///
/// # Panics
///
/// Panics if the message height is not a power of two, or if the codeword length overflows.
fn encode_by_padding<F, Ntt>(
    ntt: &Ntt,
    mut message: RowMajorMatrix<F>,
    log_inv_rate: usize,
) -> RowMajorMatrix<F>
where
    F: TowerLevel,
    Ntt: AdditiveNtt<F> + Sync,
{
    if log_inv_rate == 0 {
        return ntt.ntt_batch(message);
    }

    // Zero-padding the novel-basis coefficients extends the evaluation domain.
    let padded_len = padded_message_len(message.values.len(), log_inv_rate);
    let _ = log2_strict_usize(message.height());
    message.values.resize(padded_len, F::ZERO);
    ntt.ntt_batch_padded(message, log_inv_rate)
}

// One implementation per alphabet, each forwarding to the shared body above.
//
// A blanket implementation over every level would overlap the two-adic one in `p3-commit`.
// No downstream crate may resolve that overlap, so the alphabets are named instead.
macro_rules! impl_additive_rs_encoder {
    ($($field:ty),* $(,)?) => {$(
        impl<Ntt: AdditiveNtt<$field> + Sync> Encoder<$field> for AdditiveRsEncoder<$field, Ntt> {
            fn encode_batch(
                &self,
                message: RowMajorMatrix<$field>,
                log_inv_rate: usize,
            ) -> RowMajorMatrix<$field> {
                encode_by_padding(&self.ntt, message, log_inv_rate)
            }

            fn encode_batch_padded(
                &self,
                message: RowMajorMatrix<$field>,
                log_inv_rate: usize,
            ) -> RowMajorMatrix<$field> {
                self.ntt.ntt_batch_padded(message, log_inv_rate)
            }
        }
    )*};
}

impl_additive_rs_encoder!(
    BinaryField128,
    BinaryField64,
    Poly64,
    BinaryField32,
    BinaryField16,
    BinaryField8,
);

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_commit::Encoder;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{AdditiveRsEncoder, padded_message_len};
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    type F = BinaryField128;

    /// Builds a matrix whose entries are distinct functions of the seed and the position.
    fn matrix(log_n: usize, width: usize, seed: u64) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            (0..(width << log_n))
                .map(|i| {
                    let bits = seed
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(i as u64);
                    F::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
                })
                .collect(),
            width,
        )
    }

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

    #[test]
    #[should_panic = "codeword length overflows usize"]
    fn encode_batch_panics_when_the_codeword_length_overflows() {
        let message = RowMajorMatrix::new(vec![F::ZERO; 2], 1);
        let _ = AdditiveRsEncoder::<F>::default().encode_batch(message, usize::BITS as usize - 1);
    }

    #[test]
    fn padded_encoding_matches_naive() {
        for width in [1, 4, 16, 64] {
            for rate in [0, 1, 2, 3] {
                let mut mat = matrix(4, width, 13);
                mat.values.resize(mat.values.len() << rate, F::ZERO);
                let expected = NaiveAdditiveNtt::default().ntt_batch(mat.clone());
                assert_eq!(
                    AdditiveRsEncoder::<F>::default().encode_batch_padded(mat, rate),
                    expected
                );
            }
        }
    }

    #[test]
    #[should_panic = "padding exceeds matrix height"]
    fn padded_encoding_rejects_excessive_padding() {
        let _ = AdditiveRsEncoder::<F>::default().encode_batch_padded(matrix(2, 4, 0), 3);
    }

    #[test]
    #[should_panic]
    fn padded_encoding_rejects_non_power_of_two_height() {
        let mat = RowMajorMatrix::new(vec![F::ZERO; 12], 4);
        let _ = AdditiveRsEncoder::<F>::default().encode_batch_padded(mat, 1);
    }

    #[test]
    fn the_padded_length_is_the_message_length_shifted() {
        // Fixture state: a rate of 1/8 multiplies the coefficient count by eight.
        assert_eq!(padded_message_len(48, 3), 384);

        // No added dimension leaves the message length alone.
        assert_eq!(padded_message_len(48, 0), 48);

        // An empty message stays empty at every rate.
        assert_eq!(padded_message_len(0, 60), 0);
    }

    #[test]
    #[should_panic = "codeword length overflows usize"]
    fn the_padded_length_refuses_a_shift_past_the_word_size() {
        // A shift of the whole word width has no result `usize` can hold.
        let _ = padded_message_len(2, usize::BITS as usize);
    }

    #[test]
    #[should_panic = "codeword length overflows usize"]
    fn the_padded_length_refuses_a_value_that_overflows() {
        // The shift amount fits the word, and the shifted value does not.
        //
        //     1 << (BITS - 1)  shifted once more drops its only set bit
        let _ = padded_message_len(1 << (usize::BITS - 1), 1);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// The coset-wise encoder agrees with zero-padding and transforming, across the sizes
        /// D8's coset decomposition actually branches on: `log_inv_rate = 0` (no cosets),
        /// `log_n = 0` (single-row cosets), and ordinary cases in between.
        #[test]
        fn encode_batch_matches_padding_and_transforming(
            log_n in 0usize..=6,
            log_inv_rate in 0usize..=3,
            width in 1usize..=4,
            seed in any::<u64>(),
        ) {
            let message = matrix(log_n, width, seed);

            let mut padded = message.clone();
            padded
                .values
                .resize(message.values.len() << log_inv_rate, F::ZERO);
            let expected = NaiveAdditiveNtt::<F>::default().ntt_batch(padded);

            let encoded = AdditiveRsEncoder::<F>::default().encode_batch(message, log_inv_rate);
            prop_assert_eq!(encoded, expected);
        }
    }
}
