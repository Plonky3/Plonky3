//! Reed-Solomon encoding over the additive NTT domain.

use core::marker::PhantomData;

use p3_binary_field::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Poly64, TowerLevel,
};
use p3_commit::Encoder;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};

use crate::lch::LchNtt;
use crate::poly::PolyBasisNtt;
use crate::traits::{AdditiveNtt, padded_len};

/// Reed-Solomon encoding over the additive NTT domain.
///
/// - Each column holds the low-index novel-basis coefficients of one polynomial.
/// - Its codeword is that polynomial evaluated on the subspace `log_inv_rate` dimensions larger.
/// - The transform is a type parameter, since the fastest one differs by alphabet.
#[derive(Clone, Debug, Default)]
pub struct AdditiveRsEncoder<F, Ntt = PolyBasisNtt> {
    /// The additive NTT that evaluates the padded coefficients.
    ntt: Ntt,
    _marker: PhantomData<F>,
}

impl<F, Ntt> AdditiveRsEncoder<F, Ntt> {
    /// An encoder around the given additive NTT.
    pub const fn new(ntt: Ntt) -> Self {
        Self {
            ntt,
            _marker: PhantomData,
        }
    }
}

/// A tower level this crate encodes over, with the encoder it picks for it.
///
/// A caller naming only a level reaches the right transform through this.
pub trait EncodableLevel: TowerLevel {
    /// The Reed-Solomon encoder this crate uses for the level.
    type Encoder: Encoder<Self> + Default + Sync;
}

// The widest level runs in the polynomial basis, which falls back to the tower basis without a carryless multiply.
impl EncodableLevel for BinaryField128 {
    type Encoder = AdditiveRsEncoder<Self, PolyBasisNtt>;
}

// Every narrower level runs the tower transform on its own elements.
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

// One implementation per alphabet.
//
// A blanket implementation over every level would overlap the two-adic one in the commit crate.
macro_rules! impl_additive_rs_encoder {
    ($($field:ty),* $(,)?) => {$(
        impl<Ntt: AdditiveNtt<$field> + Sync> Encoder<$field> for AdditiveRsEncoder<$field, Ntt> {
            fn encode_batch(
                &self,
                mut message: RowMajorMatrix<$field>,
                log_inv_rate: usize,
            ) -> RowMajorMatrix<$field> {
                // Zero-padding the coefficients extends the domain without changing the polynomials.
                //
                // The transform is told the rate, so it can skip the work the zero tail would do.
                let len = padded_len(message.values.len(), log_inv_rate);
                message.values.resize(len, <$field>::ZERO);
                self.ntt.ntt_batch_padded(message, log_inv_rate)
            }

            fn encode_batch_padded(
                &self,
                message: RowMajorMatrix<$field>,
                log_inv_rate: usize,
            ) -> RowMajorMatrix<$field> {
                self.ntt.ntt_batch_padded(message, log_inv_rate)
            }

            fn encode_batch_borrowed(
                &self,
                message: RowMajorMatrixView<'_, $field>,
                log_inv_rate: usize,
            ) -> RowMajorMatrix<$field> {
                self.ntt.ntt_batch_borrowed(message, log_inv_rate)
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

    use super::AdditiveRsEncoder;
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
