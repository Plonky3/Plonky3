//! The interleaved Reed–Solomon codeword a multi-rate commitment's levels read.

use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};

use crate::butterfly::ButterflyField;
use crate::encoder::padded_message_len;
use crate::lch::transform_cosets;

/// Reed–Solomon encode a message held one folding column at a time.
///
/// # Overview
///
/// A commitment that folds `f` variables at a time opens whole folding blocks at once.
/// Its codeword is therefore a matrix of `2^f` columns, one row per domain point.
///
/// A prover holds the opposite layout, with each column's coefficients contiguous.
/// That is what binding a prefix of the variables leaves behind.
///
/// Bringing the two layouts together as a pass of its own writes the message once more.
///
/// The transform this hands the result to skips the layers that cross the padding.
/// So the interleaving rides along with a copy that has to happen anyway.
///
/// # Arguments
///
/// - `columns`: the message's columns back to back, each `2^log_message` values long.
/// - `log_message`: base-two logarithm of the values one column holds.
/// - `log_inv_rate`: dimensions added to the message's own domain.
///
/// # Returns
///
/// One row per domain point, holding every column's value at that point.
///
/// # Panics
///
/// Panics if there is no column, or if the values do not divide into whole columns.
/// Panics if the extended domain exceeds the bit width of the level.
///
/// Panics if the codeword length overflows the address space.
#[must_use]
pub fn interleaved_encode_batch<F: ButterflyField>(
    columns: &[F],
    log_message: usize,
    log_inv_rate: usize,
) -> RowMajorMatrix<F> {
    // A codeword of no columns has neither a width to interleave into nor a height to read.
    assert!(!columns.is_empty(), "at least one message column");
    assert!(
        log_message + log_inv_rate <= 1 << F::LOG_BITS,
        "domain exceeds field dimension"
    );

    // The widest level admits 128 dimensions, which is past what a length can address.
    // So the row count is what bounds the dimension, not the level.
    let rows = u32::try_from(log_message)
        .ok()
        .and_then(|bits| 1usize.checked_shl(bits))
        .expect("message dimension overflows usize");
    assert_eq!(columns.len() % rows, 0, "whole message columns");

    let width = columns.len() / rows;
    let len = columns.len();
    let mut values = F::zero_vec(padded_message_len(len, log_inv_rate));

    // The first coset is the message with its columns interleaved, blocked by the transpose.
    let source = RowMajorMatrixView::new(columns, rows);
    let mut target = RowMajorMatrixViewMut::new(&mut values[..len], width);
    source.transpose_into(&mut target);

    // Every coset above it is a copy of that one, then its own shifted transform.
    transform_cosets::<F>(&mut values, width, log_message);

    RowMajorMatrix::new(values, width)
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField16, BinaryField32, BinaryField128, Ghash128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
    use proptest::prelude::*;

    use super::interleaved_encode_batch;
    use crate::butterfly::ButterflyField;
    use crate::lch::LchNtt;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// Widths covering one column, an odd count, a register-sized row and a folding block.
    const WIDTHS: [usize; 4] = [1, 3, 8, 16];

    /// A column-major message whose entries depend on both the position and the seed.
    fn columns<F: TowerLevel>(log_message: usize, width: usize, seed: u64) -> Vec<F> {
        (0..(width << log_message))
            .map(|i| {
                let bits = seed
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(i as u64 + 1)
                    .wrapping_mul(0xbf58_476d_1ce4_e5b9);
                F::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
            })
            .collect()
    }

    /// The reference: interleave in a pass of its own, pad with zeros, transform the whole lot.
    fn transpose_then_encode<F: ButterflyField, N: AdditiveNtt<F>>(
        ntt: &N,
        columns: &[F],
        log_message: usize,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<F> {
        let rows = 1usize << log_message;
        let width = columns.len() / rows;

        let mut values = F::zero_vec(columns.len() << log_inv_rate);
        if width != 0 {
            let source = RowMajorMatrixView::new(columns, rows);
            let mut target = RowMajorMatrixViewMut::new(&mut values[..columns.len()], width);
            source.transpose_into(&mut target);
        }
        ntt.ntt_batch(RowMajorMatrix::new(values, width))
    }

    /// The fused encoder against the two-pass reference, at one shape.
    fn check_against_the_two_pass<F: ButterflyField>(
        log_message: usize,
        width: usize,
        log_inv_rate: usize,
        seed: u64,
    ) {
        let columns = columns::<F>(log_message, width, seed);
        let expected =
            transpose_then_encode(&LchNtt::<F>::default(), &columns, log_message, log_inv_rate);
        let actual = interleaved_encode_batch::<F>(&columns, log_message, log_inv_rate);
        assert_eq!(
            actual, expected,
            "log_message={log_message} width={width} rate={log_inv_rate}"
        );
    }

    #[test]
    fn the_fused_encoder_matches_the_two_pass_one() {
        // Rates from none at all, where no coset is copied, up to three added dimensions.
        for width in WIDTHS {
            for log_message in 0..=6 {
                for log_inv_rate in 0..=3 {
                    check_against_the_two_pass::<BinaryField32>(
                        log_message,
                        width,
                        log_inv_rate,
                        7,
                    );
                    check_against_the_two_pass::<BinaryField128>(
                        log_message,
                        width,
                        log_inv_rate,
                        11,
                    );
                    check_against_the_two_pass::<Ghash128>(log_message, width, log_inv_rate, 13);
                }
            }
        }
    }

    #[test]
    fn the_fused_encoder_matches_the_reference_oracle() {
        // The oracle evaluates the novel basis from its product definition.
        // So it pins the coset split to that basis, not to another split of the network.
        for width in [1usize, 3] {
            for log_message in 0..=4 {
                for log_inv_rate in 0..=2 {
                    let columns = columns::<BinaryField16>(log_message, width, 17);
                    let expected = transpose_then_encode(
                        &NaiveAdditiveNtt::<BinaryField16>::default(),
                        &columns,
                        log_message,
                        log_inv_rate,
                    );
                    let actual = interleaved_encode_batch::<BinaryField16>(
                        &columns,
                        log_message,
                        log_inv_rate,
                    );
                    let label = format!("log_message={log_message} width={width}");
                    assert_eq!(actual, expected, "{label} rate={log_inv_rate}");
                }
            }
        }
    }

    #[test]
    fn a_column_reappears_as_the_codeword_column() {
        // Invariant: column `j` of the codeword is that column's own encoding, on its own.
        //
        // Fixture state: four columns of 2^5 values at rate 1/2.
        //
        //     columns:  [ c0 | c1 | c2 | c3 ]      each 32 values, contiguous
        //     codeword:  row i = [ C0[i], C1[i], C2[i], C3[i] ]
        const LOG_MESSAGE: usize = 5;
        const WIDTH: usize = 4;
        let columns = columns::<BinaryField128>(LOG_MESSAGE, WIDTH, 23);
        let codeword = interleaved_encode_batch::<BinaryField128>(&columns, LOG_MESSAGE, 1);

        for (j, column) in columns
            .as_chunks::<{ 1 << LOG_MESSAGE }>()
            .0
            .iter()
            .enumerate()
        {
            // That column alone, padded and transformed as a single-column message.
            let mut padded = column.to_vec();
            padded.resize(padded.len() * 2, BinaryField128::ZERO);
            let alone =
                LchNtt::<BinaryField128>::default().ntt_batch(RowMajorMatrix::new(padded, 1));

            for (i, value) in alone.values.iter().enumerate() {
                assert_eq!(codeword.values[i * WIDTH + j], *value, "column={j} row={i}");
            }
        }
    }

    #[test]
    #[should_panic = "at least one message column"]
    fn a_message_with_no_column_is_refused() {
        // A codeword of no columns has no width to interleave into and no height to read off.
        // Every later step divides by one of the two, so the shape has to be rejected here.
        let _ = interleaved_encode_batch::<BinaryField32>(&[], 3, 1);
    }

    #[test]
    #[should_panic = "message dimension overflows usize"]
    fn a_message_dimension_past_the_address_space_is_refused() {
        // The widest level admits 128 domain dimensions, and a length addresses at most 64.
        // So a dimension the level allows can still have no row count to describe it.
        let values = columns::<BinaryField128>(1, 1, 0);
        let _ = interleaved_encode_batch::<BinaryField128>(&values, 100, 1);
    }

    #[test]
    #[should_panic = "whole message columns"]
    fn a_partial_column_is_refused() {
        // Nine values do not divide into columns of four, so the interleaving is ill-posed.
        let values = columns::<BinaryField32>(2, 3, 0);
        let _ = interleaved_encode_batch::<BinaryField32>(&values[..9], 2, 0);
    }

    #[test]
    #[should_panic = "domain exceeds field dimension"]
    fn a_domain_past_the_level_dimension_is_refused() {
        // A byte level has sixteen Cantor basis vectors, and this asks for seventeen.
        let values = columns::<BinaryField16>(15, 1, 0);
        let _ = interleaved_encode_batch::<BinaryField16>(&values, 15, 2);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Random shapes against the two-pass reference, over two representations.
        #[test]
        fn random_shapes_match_the_two_pass_encoder(
            log_message in 0usize..=7,
            width in 1usize..=5,
            log_inv_rate in 0usize..=2,
            seed in any::<u64>(),
        ) {
            check_against_the_two_pass::<BinaryField128>(log_message, width, log_inv_rate, seed);
            check_against_the_two_pass::<Ghash128>(log_message, width, log_inv_rate, seed);
        }
    }
}
