//! Zero-knowledge commitments for the HVZK-WHIR oracles.
//!
//! Each oracle is a prefix-interleaved ZK Reed-Solomon encoding.
//! This is Definition 3.22 of eprint 2026/391:
//!
//! ```text
//!     column b = DFT( message chunk b || randomness chunk b || zeros )
//! ```
//!
//! - Chunks are contiguous high-bit slices, matching the prefix layout used
//!   by the non-ZK committer.
//! - Each limb carries its own randomness coefficients.
//! - Folding a leaf row by `eq(., gamma)` therefore yields the folded ZK
//!   code of the folded message and randomness (Lemma 3.26).

use alloc::vec::Vec;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
use p3_util::log2_strict_usize;

use crate::utils::padded_ood_t1;

/// Builds the pre-DFT matrix for an interleaved ZK encoding.
///
/// Layout per column `b` of width `2^folding`:
///
/// ```text
///     rows [0, msg_rows)               message chunk b (transposed in)
///     rows [msg_rows, msg_rows + t)    randomness chunk b
///     rows [msg_rows + t, height)      zeros
/// ```
pub(crate) fn zk_padded_matrix<A: Field>(
    message: &[A],
    randomness: &[A],
    folding: usize,
    height: usize,
) -> RowMajorMatrix<A> {
    let width = 1 << folding;
    let msg_rows = message.len() >> folding;
    let randomness_rows = randomness.len() >> folding;
    assert_eq!(randomness.len(), randomness_rows << folding);
    assert!(msg_rows + randomness_rows <= height);

    let mut values = A::zero_vec(height * width);

    // Message: cache-blocked transpose of high-bit chunks into the top rows.
    let folded = RowMajorMatrixView::new(message, msg_rows);
    folded.transpose_into(&mut RowMajorMatrixViewMut::new(
        &mut values[..message.len()],
        width,
    ));

    // Randomness: limb-major chunks land column-wise after the message rows.
    for b in 0..width {
        for j in 0..randomness_rows {
            values[(msg_rows + j) * width + b] = randomness[b * randomness_rows + j];
        }
    }

    RowMajorMatrix::new(values, width)
}

/// The folded ZK Reed-Solomon code seen through one committed oracle's leaves.
///
/// Folding a leaf row by `eq(., gamma)` produces one position of
///
/// ```text
///     Enc(Fold(message, gamma), Fold(randomness, gamma))
/// ```
///
/// over a base-field two-adic domain.
///
/// The descriptor exposes:
/// - generator rows,
/// - evaluation points,
/// - a direct (width-one) encoder for fresh base-case masks living in the
///   same code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldedRsCode<F> {
    /// Message length (power of two).
    pub message_len: usize,
    /// ZK randomness coefficients.
    pub randomness_len: usize,
    /// Codeword length, equal to the committed oracle's leaf count.
    pub domain_size: usize,
    /// Generator of the evaluation domain.
    pub domain_gen: F,
}

impl<F: TwoAdicField> FoldedRsCode<F> {
    /// Builds the folded code descriptor for a domain of the given size.
    #[must_use]
    pub fn new(message_len: usize, randomness_len: usize, domain_size: usize) -> Self {
        assert!(message_len.is_power_of_two());
        assert!(message_len + randomness_len <= domain_size);
        Self {
            message_len,
            randomness_len,
            domain_size,
            domain_gen: F::two_adic_generator(log2_strict_usize(domain_size)),
        }
    }

    /// Codeword value at position `z` from an explicit message and randomness:
    ///
    /// ```text
    ///     Enc(msg, rand)(z) = sum_j msg_j x^j  +  sum_s rand_s x^{l + s}
    /// ```
    #[must_use]
    pub fn evaluate_at<EF: ExtensionField<F>>(
        &self,
        position: usize,
        message: &[EF],
        randomness: &[EF],
    ) -> EF {
        assert_eq!(message.len(), self.message_len);
        assert_eq!(randomness.len(), self.randomness_len);
        padded_ood_t1(
            self.domain_gen.exp_u64(position as u64),
            message,
            randomness,
        )
    }

    /// Encodes a single (non-interleaved) message into this code.
    ///
    /// The base case uses it for fresh masks in the virtual oracle's code.
    #[must_use]
    pub fn encode_column<EF, Dft>(
        &self,
        dft: &Dft,
        message: &[EF],
        randomness: &[EF],
    ) -> RowMajorMatrix<EF>
    where
        EF: ExtensionField<F> + TwoAdicField,
        Dft: TwoAdicSubgroupDft<F>,
    {
        assert_eq!(message.len(), self.message_len);
        assert_eq!(randomness.len(), self.randomness_len);
        // Single allocation: the full domain is reserved up front.
        //
        // The two copies and the zero padding all fill in place.
        //
        //     [ message | randomness | 0 ... 0 ]   (domain_size slots)
        let mut coeffs = Vec::with_capacity(self.domain_size);
        coeffs.extend_from_slice(message);
        coeffs.extend_from_slice(randomness);
        coeffs.resize(self.domain_size, EF::ZERO);
        dft.dft_algebra_batch(RowMajorMatrix::new_col(coeffs))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type MyDft = Radix2DFTSmallBatch<F>;

    /// Folds a limb-major chunked vector by the eq table at `gamma`.
    fn fold_chunks(values: &[EF], chunk: usize, gamma: &Point<EF>) -> Vec<EF> {
        let weights = Poly::new_from_point(gamma.as_slice(), EF::ONE);
        let mut out = EF::zero_vec(chunk);
        for (b, &w) in weights.as_slice().iter().enumerate() {
            for (dst, &src) in out.iter_mut().zip(&values[b * chunk..(b + 1) * chunk]) {
                *dst += w * src;
            }
        }
        out
    }

    #[test]
    fn folded_leaf_is_folded_code_position() {
        // Invariant (Lemma 3.26):
        //
        //     Fold(Enc(msg, rand), gamma)(z) == Enc(Fold(msg), Fold(rand))(z)
        //
        // for the interleaved ZK extension commitment and the FoldedRsCode
        // evaluation, at every leaf position.
        let mut rng = SmallRng::seed_from_u64(7);
        let num_variables = 5;
        let folding = 2;
        let inv_rate = 2;
        let t = 3;

        let message = Poly::<EF>::rand(&mut rng, num_variables);
        let randomness: Vec<EF> = (0..(t << folding)).map(|_| rng.random()).collect();

        let dft = MyDft::default();
        let height = inv_rate << (num_variables - folding);
        let padded = zk_padded_matrix(message.as_slice(), &randomness, folding, height);
        let encoded = dft.dft_algebra_batch(padded);

        let gamma = Point::<EF>::rand(&mut rng, folding);
        let folded_message =
            fold_chunks(message.as_slice(), 1 << (num_variables - folding), &gamma);
        let folded_randomness = fold_chunks(&randomness, t, &gamma);

        let code = FoldedRsCode::<F>::new(1 << (num_variables - folding), t, height);
        for z in 0..height {
            let leaf: Vec<EF> = (0..1 << folding)
                .map(|b| encoded.values[z * (1 << folding) + b])
                .collect();
            // Fold the leaf row by the eq table at gamma, as the verifier does after a query.
            let folded_leaf = Poly::new(leaf).eval_ext::<F>(&gamma);
            assert_eq!(
                folded_leaf,
                code.evaluate_at(z, &folded_message, &folded_randomness),
                "mismatch at leaf {z}",
            );
        }
    }

    #[test]
    fn encode_column_matches_evaluate_at() {
        // The width-one encoder and the row evaluator describe the same code.
        let mut rng = SmallRng::seed_from_u64(11);
        let code = FoldedRsCode::<F>::new(8, 2, 32);
        let message: Vec<EF> = (0..8).map(|_| rng.random()).collect();
        let randomness: Vec<EF> = (0..2).map(|_| rng.random()).collect();

        let dft = MyDft::default();
        let codeword = code.encode_column(&dft, &message, &randomness);
        for z in 0..code.domain_size {
            assert_eq!(
                codeword.values[z],
                code.evaluate_at(z, &message, &randomness),
                "mismatch at position {z}",
            );
        }
    }

    #[test]
    fn zk_base_matrix_with_no_randomness_matches_plain_layout() {
        // With t = 0 the padded matrix degenerates to the non-ZK prefix
        // transpose-and-pad layout.
        let mut rng = SmallRng::seed_from_u64(3);
        let message = Poly::<F>::rand(&mut rng, 4);
        let folding = 2;
        let height = 2 << (4 - folding);

        let zk = zk_padded_matrix(message.as_slice(), &[], folding, height);

        let width = 1 << folding;
        let mut plain = F::zero_vec(height * width);
        let folded = RowMajorMatrixView::new(message.as_slice(), 1 << (4 - folding));
        folded.transpose_into(&mut RowMajorMatrixViewMut::new(&mut plain[..1 << 4], width));

        assert_eq!(zk.values, plain);
    }
}
