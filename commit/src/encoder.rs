//! Linear codes applied column-wise to a matrix.

use alloc::vec::Vec;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;

/// Receives finished rows of a codeword, as `(first_row, rows)`, from any worker.
///
/// Every row is handed over exactly once, in blocks of whole rows.
pub type RowSink<'a, T> = dyn Fn(usize, &[T]) + Sync + 'a;

/// Bytes of one block of rows handed to a sink.
const ROW_BLOCK_BYTES: usize = 256 * 1024;

/// Hand rows starting at `first_row` to a sink once each, in blocks of whole rows, from the pool.
pub fn hand_over_rows<T: Sync>(
    values: &[T],
    width: usize,
    first_row: usize,
    sink: &RowSink<'_, T>,
) {
    let rows = (ROW_BLOCK_BYTES / (size_of::<T>() * width).max(1)).max(1);
    values
        .par_chunks(rows * width.max(1))
        .enumerate()
        .for_each(|(index, block)| sink(first_row + index * rows, block));
}

/// A borrowed message copied into a zeroed matrix `2^log_inv_rate` times its height.
///
/// The message fills the leading rows and every row after it is zero.
///
/// That is the matrix [`Encoder::encode_batch_padded`] takes.
///
/// The copy splits across workers only where the cost model says the message is worth it.
///
/// # Panics
///
/// Panics if the padded length overflows `usize`.
#[must_use]
pub fn zero_padded<F: Field>(
    message: RowMajorMatrixView<'_, F>,
    log_inv_rate: usize,
) -> RowMajorMatrix<F> {
    let len = message.values.len();
    let padded_len = u32::try_from(log_inv_rate)
        .ok()
        .and_then(|rate| len.checked_shl(rate))
        // `checked_shl` only rejects a shift amount that is too wide.
        //
        // Recovering the original length from the shifted one is what proves no bits were lost.
        .filter(|&padded| padded >> log_inv_rate == len)
        .expect("codeword length overflows usize");
    let mut values = Vec::with_capacity(padded_len);
    let (head, tail) = values.spare_capacity_mut()[..padded_len].split_at_mut(len);

    // Each element is read once and written once.
    let chunk = min_task_len(len, 2 * size_of::<F>());
    head.par_chunks_mut(chunk)
        .zip(message.values.par_chunks(chunk))
        .for_each(|(destination, source)| {
            for (slot, &value) in destination.iter_mut().zip(source) {
                slot.write(value);
            }
        });

    // The zero tail is written in parallel too, since a recycled allocation is not zero already.
    let chunk = min_task_len(tail.len(), size_of::<F>());
    tail.par_chunks_mut(chunk).for_each(|destination| {
        for slot in destination {
            slot.write(F::ZERO);
        }
    });

    // SAFETY: the two sweeps above initialised every one of the first `padded_len` slots.
    unsafe { values.set_len(padded_len) };
    RowMajorMatrix::new(values, message.width)
}

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
    /// The original message occupies the first `height / 2^log_inv_rate` rows; the caller must set the remaining entries to zero.
    ///
    /// Implementations may ignore that tail and use the rate to avoid work on zero coefficients.
    ///
    /// The default preserves the ordinary transform of the full padded matrix.
    ///
    /// # Panics
    ///
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

    /// Encodes each column of a borrowed message into a codeword, leaving the message as it is.
    ///
    /// The codeword is what [`Self::encode_batch_padded`] makes of the message zero-padded to
    /// `2^(k + log_inv_rate)` rows.
    ///
    /// The default builds that padded matrix, copying the message into it.
    ///
    /// An encoder that reads the message where it lies skips the copy.
    ///
    /// # Panics
    ///
    /// Panics if the message height is not a power of two.
    ///
    /// Panics if the codeword height `2^(k + log_inv_rate)` overflows `usize`.
    fn encode_batch_borrowed(
        &self,
        message: RowMajorMatrixView<'_, F>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<F> {
        self.encode_batch_padded(zero_padded(message, log_inv_rate), log_inv_rate)
    }

    /// As [`Self::encode_batch_borrowed`], handing every finished codeword row to `sink` once.
    ///
    /// An encoder may hand rows over while it still holds them in cache.
    ///
    /// The default hands them over once the whole codeword is done.
    fn encode_batch_borrowed_with(
        &self,
        message: RowMajorMatrixView<'_, F>,
        log_inv_rate: usize,
        sink: &RowSink<'_, F>,
    ) -> RowMajorMatrix<F> {
        let codeword = self.encode_batch_borrowed(message, log_inv_rate);
        hand_over_rows(&codeword.values, codeword.width, 0, sink);
        codeword
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
        self.encode_batch_padded(message, log_inv_rate)
    }

    fn encode_batch_padded(
        &self,
        message: RowMajorMatrix<F>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<F> {
        self.dft_batch_padded(message, log_inv_rate)
            .to_row_major_matrix()
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
    fn a_long_borrowed_message_encodes_as_its_padding() {
        // Widths of three and eight rows of 2^14 give lengths a power of two does and does not
        // divide, both long enough for the copy to split across workers.
        let dft = Radix2DitParallel::<BabyBear>::default();
        for width in [3, 8] {
            let mut rng = SmallRng::seed_from_u64(2);
            let message = RowMajorMatrix::<BabyBear>::rand(&mut rng, 1 << 14, width);
            let expected = dft.encode_batch(message.clone(), 1);
            assert_eq!(
                dft.encode_batch_borrowed(message.as_view(), 1),
                expected,
                "width={width}"
            );
        }
    }

    #[test]
    fn a_zero_padded_copy_holds_the_message_over_a_zero_tail() {
        // Lengths the copy keeps whole, splits, and splits with a short last chunk.
        for (height, width) in [(1, 1), (4, 3), (1 << 12, 5), (1 << 14, 7)] {
            let mut rng = SmallRng::seed_from_u64(height as u64 ^ width as u64);
            let message = RowMajorMatrix::<BabyBear>::rand(&mut rng, height, width);
            for log_inv_rate in 0..=2 {
                let padded = super::zero_padded(message.as_view(), log_inv_rate);
                let len = message.values.len();
                assert_eq!(padded.width, width);
                assert_eq!(padded.values.len(), len << log_inv_rate);
                assert_eq!(&padded.values[..len], message.values.as_slice());
                assert!(padded.values[len..].iter().all(|v| *v == BabyBear::ZERO));
            }
        }
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
