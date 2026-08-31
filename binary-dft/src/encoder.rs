//! Reed–Solomon encoder over the additive NTT domain.

use core::marker::PhantomData;

use p3_binary_field::BinaryField128;
use p3_commit::Encoder;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::domain::domain_point;
use crate::poly::PolyBasisNtt;
use crate::traits::AdditiveNtt;

/// Reed–Solomon over the additive NTT domain.
///
/// The message holds the low-index novel-basis coefficients of each column, so the codeword is
/// the evaluation of `f̂(Ŵ_0(x), …, Ŵ_{k−1}(x))` on `S_{k + log_inv_rate}`.
///
/// The alphabet is `BinaryField128`, where [`PolyBasisNtt`] is the faster transform and falls
/// back to [`LchNtt`](crate::LchNtt) on a target without a carryless multiply, so it is the default.
///
/// `F` is phantom: [`Encoder`] is only implemented below for `F = BinaryField128`, and stays
/// that way as long as the alphabet is fixed (D9), so the parameter carries no other instance.
#[derive(Clone, Debug, Default)]
pub struct AdditiveRsEncoder<F, Ntt = PolyBasisNtt> {
    ntt: Ntt,
    _marker: PhantomData<F>,
}

/// The alphabet is fixed at `BinaryField128` (D9), as [`Encoder`] requires of every impl outside
/// `p3-commit`'s blanket one.
impl<Ntt: AdditiveNtt<BinaryField128> + Sync> Encoder<BinaryField128>
    for AdditiveRsEncoder<BinaryField128, Ntt>
{
    fn encode_batch(
        &self,
        message: RowMajorMatrix<BinaryField128>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField128> {
        if log_inv_rate == 0 {
            return self.ntt.ntt_batch(message);
        }

        let width = message.width();
        let len = message.values.len();
        let padded_len = u32::try_from(log_inv_rate)
            .ok()
            .and_then(|rate| len.checked_shl(rate))
            // `checked_shl` only rejects a shift amount that is too wide; it does not detect
            // the value itself overflowing, so recovering `len` from the shifted result is
            // what actually proves no bits were lost.
            .filter(|&padded| padded >> log_inv_rate == len)
            .expect("codeword length overflows usize");
        let log_message_height = log2_strict_usize(message.height());

        // Zero-padding to `S_{k+r}` and transforming the whole codeword leaves the appended
        // `hi` coefficient half zero at every stage `j >= k`, so those butterflies only ever
        // replicate `lo` — yet the carryless multiply against that zero still runs, `r` stages
        // deep. The equivalent computation is `2^r` independent height-`2^k` transforms of the
        // unpadded message, one per coset `c` of `S_k` in `S_{k+r}`, evaluated at
        // `domain_point(c << k)`: every codeword row decomposes as `c*2^k + m` with
        // `domain_point(c*2^k + m) = domain_point(c << k) + domain_point(m)`, and the padding
        // rows never enter the computation at all. Each coset is independent, so they run in
        // parallel; `shifted_ntt_batch` takes its message by value, so each needs its own copy.
        let mut values = BinaryField128::zero_vec(padded_len);
        values
            .par_chunks_mut(len)
            .enumerate()
            .for_each(|(c, chunk)| {
                let shift = domain_point::<BinaryField128>(c << log_message_height);
                let coset = self
                    .ntt
                    .shifted_ntt_batch(RowMajorMatrix::new(message.values.clone(), width), shift);
                chunk.copy_from_slice(&coset.values);
            });

        RowMajorMatrix::new(values, width)
    }
}

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
