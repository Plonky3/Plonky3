//! Evaluation-domain operations used by the non-hiding WHIR prover.

use p3_commit::{Encoder, zero_padded};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_security::SecurityAssumption;
use p3_util::log2_strict_usize;

/// Selector coordinates for one queried codeword position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhirQueryPoint<F> {
    /// A univariate point expanded through successive squares.
    Univariate(F),
    /// Direct coordinates in the code's multilinear basis.
    Multilinear(Point<F>),
}

/// A nested linear code and its evaluation-point map.
///
/// WHIR only needs nested power-of-two domains. Implementations are statically
/// dispatched, so choosing a domain adds no virtual calls to the prover.
pub trait WhirDomain<F, EF>: Encoder<F> + Sync
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Stable identifier bound into the Fiat--Shamir instance label.
    ///
    /// The identifier must cover the code and query-point map.
    ///
    /// An empty identifier selects the canonical two-adic transcript.
    fn protocol_id(&self) -> &'static [u8];

    /// Whether the domain supports the requested Reed--Solomon soundness regime.
    fn supports_security_assumption(&self, assumption: SecurityAssumption) -> bool {
        // Capacity requires a domain-specific assumption.
        // New domains must opt into it explicitly.
        assumption != SecurityAssumption::CapacityBound
    }

    /// Whether queries use the canonical power-of-two stratified schedule.
    fn stratified_queries(&self) -> bool {
        false
    }

    /// Largest supported base-two logarithm of a domain size.
    fn max_log_domain_size(&self) -> usize;

    /// Encodes an already-padded extension-field coefficient matrix.
    ///
    /// Only the first `message.height() >> log_inv_rate` rows contain coefficients;
    /// every subsequent row must be zero. The codeword is returned in natural row order.
    fn encode_extension_batch_padded(
        &self,
        message: RowMajorMatrix<EF>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<EF>;

    /// Encodes an extension-field coefficient matrix held elsewhere, at `2^log_inv_rate` times its height.
    ///
    /// The default copies it under a zero tail, then encodes that.
    fn encode_extension_batch_borrowed(
        &self,
        message: RowMajorMatrixView<'_, EF>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<EF> {
        self.encode_extension_batch_padded(zero_padded(message, log_inv_rate), log_inv_rate)
    }

    /// Returns selector coordinates for a queried codeword position.
    fn query_point(
        &self,
        log_domain_size: usize,
        num_variables: usize,
        index: usize,
    ) -> WhirQueryPoint<F>;
}

impl<F, EF, Dft> WhirDomain<F, EF> for Dft
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F> + Sync,
{
    fn protocol_id(&self) -> &'static [u8] {
        b""
    }

    fn supports_security_assumption(&self, _assumption: SecurityAssumption) -> bool {
        // Preserve every regime supported by the canonical two-adic protocol.
        true
    }

    fn max_log_domain_size(&self) -> usize {
        F::TWO_ADICITY
    }

    fn encode_extension_batch_padded(
        &self,
        message: RowMajorMatrix<EF>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<EF> {
        // Coset setup pays off once enough zero-padding FFT layers are skipped.
        if log_inv_rate < 4 {
            return self.dft_algebra_batch(message);
        }

        let log_height = log2_strict_usize(message.height());
        assert!(log_inv_rate <= log_height);
        let width = message.width;
        let num_cosets = 1 << log_inv_rate;
        let len = message.height() >> log_inv_rate;
        debug_assert!(
            message.values[len * width..].iter().all(Field::is_zero),
            "padded coefficient rows must be zero"
        );
        let base_width = width * EF::DIMENSION;
        let batch_width = num_cosets * base_width;
        let generator = F::two_adic_generator(log_height);

        // Flatten here rather than inside dft_algebra_batch, so coset scaling can use
        // base-field SIMD even when extension elements themselves have scalar packing.
        let mut values = EF::flatten_to_base(message.values);
        // Reuse the padded allocation. Moving backwards preserves every unread coefficient
        // row while making room for its cosets, without another codeword-sized buffer.
        for c in (0..len).rev() {
            values.copy_within(c * base_width..(c + 1) * base_width, c * batch_width);
        }
        // Seed one short power run per parallel chunk instead of exponentiating for every
        // coefficient row. Bound the chunk to the live height so its size cannot overflow.
        let rows_per_chunk = len.min(256);
        values
            .par_chunks_mut(batch_width * rows_per_chunk)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let mut ratio = generator.exp_u64((chunk_index * rows_per_chunk) as u64);
                for row in chunk.chunks_exact_mut(batch_width) {
                    let (coeffs, cosets) = row.split_at_mut(base_width);
                    let (packed_coeffs, coeff_suffix) = F::Packing::pack_slice_with_suffix(coeffs);
                    let mut scale = ratio;
                    for values in cosets.chunks_exact_mut(base_width) {
                        let (packed_values, value_suffix) =
                            F::Packing::pack_slice_with_suffix_mut(values);
                        let packed_scale: F::Packing = scale.into();
                        for (value, &coeff) in packed_values.iter_mut().zip(packed_coeffs) {
                            *value = coeff * packed_scale;
                        }
                        for (value, &coeff) in value_suffix.iter_mut().zip(coeff_suffix) {
                            *value = coeff * scale;
                        }
                        scale *= ratio;
                    }
                    ratio *= generator;
                }
            });

        // For i = a + num_cosets * b, g^i = g^a * (g^num_cosets)^b.
        // Batch the size-len transforms of coefficients scaled by g^(a*c).
        // Natural-order output [b][a][column] is already the required height-by-width layout.
        let encoded = self
            .dft_batch(RowMajorMatrix::new(values, batch_width))
            .to_row_major_matrix();
        RowMajorMatrix::new(EF::reconstitute_from_base(encoded.values), width)
    }

    fn query_point(
        &self,
        log_domain_size: usize,
        _num_variables: usize,
        index: usize,
    ) -> WhirQueryPoint<F> {
        WhirQueryPoint::Univariate(F::two_adic_generator(log_domain_size).exp_u64(index as u64))
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::Matrix;

    use super::*;

    /// The reference includes every padded row and compares the complete natural-order codeword.
    /// Nonconstant extension coefficients make row/coset/column permutations observable.
    fn check_padded_encoding<F, EF, Dft>(dft: &Dft)
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Dft: TwoAdicSubgroupDft<F> + Sync,
    {
        for log_len in [0, 1, 2, 5, 9] {
            for log_inv_rate in [0, 1, 2, 3, 5, 7] {
                for width in [1, 2, 3, 8] {
                    let len = 1 << log_len;
                    let height = len << log_inv_rate;
                    let mut values = EF::zero_vec(height * width);
                    for (i, value) in values[..len * width].iter_mut().enumerate() {
                        *value = EF::from_basis_coefficients_fn(|j| {
                            F::from_usize((i + 1) * (i + 3) + j * (i + 7) + 11)
                        });
                    }
                    let message = RowMajorMatrix::new(values, width);
                    let expected = dft.dft_algebra_batch(message.clone());
                    let actual = dft.encode_extension_batch_padded(message, log_inv_rate);
                    assert_eq!(actual.dimensions(), expected.dimensions());
                    assert_eq!(
                        actual.values, expected.values,
                        "log_len={log_len}, log_inv_rate={log_inv_rate}, width={width}"
                    );
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "padded coefficient rows must be zero")]
    fn padded_encoding_rejects_nonzero_padding() {
        type EF = BinomialExtensionField<BabyBear, 4>;
        let mut values = EF::zero_vec(32 * 3);
        // The first two rows are live; this is the first padding coefficient.
        values[2 * 3] = EF::ONE;
        let message = RowMajorMatrix::new(values, 3);
        Radix2DFTSmallBatch::<BabyBear>::default().encode_extension_batch_padded(message, 4);
    }

    #[test]
    fn padded_encoding_does_not_transform_zero_padding() {
        #[derive(Clone, Default)]
        struct DegreeSizedDft;

        impl TwoAdicSubgroupDft<BabyBear> for DegreeSizedDft {
            type Evaluations = RowMajorMatrix<BabyBear>;

            fn dft_batch(&self, message: RowMajorMatrix<BabyBear>) -> Self::Evaluations {
                assert_eq!(
                    message.height(),
                    2,
                    "transform only the live coefficient height"
                );
                Radix2DFTSmallBatch::default().dft_batch(message)
            }
        }

        type EF = BinomialExtensionField<BabyBear, 4>;
        let mut values = EF::zero_vec(256 * 3);
        values[..6].fill(EF::ONE);
        let message = RowMajorMatrix::new(values, 3);
        let expected =
            Radix2DFTSmallBatch::<BabyBear>::default().dft_algebra_batch(message.clone());
        let actual = DegreeSizedDft.encode_extension_batch_padded(message, 7);
        assert_eq!(actual.values, expected.values);
    }

    #[test]
    fn padded_encoding_matches_full_dft_baby_bear_small_batch() {
        check_padded_encoding::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(
            &Radix2DFTSmallBatch::default(),
        );
    }

    #[test]
    fn padded_encoding_matches_full_dft_baby_bear_parallel_layout() {
        check_padded_encoding::<BabyBear, BinomialExtensionField<BabyBear, 5>, _>(
            &Radix2DitParallel::default(),
        );
    }

    #[test]
    fn padded_encoding_matches_full_dft_koala_bear_small_batch() {
        check_padded_encoding::<KoalaBear, BinomialExtensionField<KoalaBear, 4>, _>(
            &Radix2DFTSmallBatch::default(),
        );
    }

    #[test]
    fn padded_encoding_matches_full_dft_koala_bear_parallel_layout() {
        check_padded_encoding::<KoalaBear, BinomialExtensionField<KoalaBear, 8>, _>(
            &Radix2DitParallel::default(),
        );
    }
}
