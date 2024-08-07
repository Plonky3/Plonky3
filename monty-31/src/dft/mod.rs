//! An implementation of the FFT for `MontyField31`
extern crate alloc;

use alloc::vec::Vec;
use core::cell::RefCell;

use p3_dft::TwoAdicSubgroupDft;
use p3_matrix::bitrev::{BitReversableMatrix, BitReversedMatrixView};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::split_at_mut_unchecked;
use tracing::debug_span;

mod forward;

use crate::{FieldParameters, MontyField31, MontyParameters, TwoAdicData};

/// Radix-2 decimation-in-frequency FFT
#[derive(Clone, Debug, Default)]
pub struct Radix2Dif<F> {
    /// Memoized twiddle factors for each length log_n.
    ///
    /// TODO: The use of RefCell means this can't be shared across
    /// threads; consider using RwLock or finding a better design
    /// instead.
    twiddles: RefCell<Vec<Vec<F>>>,
}

impl<MP: FieldParameters + TwoAdicData> Radix2Dif<MontyField31<MP>> {
    pub fn new(n: usize) -> Self {
        Self {
            twiddles: RefCell::new(MontyField31::roots_of_unity_table(n)),
        }
    }

    pub fn dft_batch_transposed_bitrevd_with_scratch(
        &self,
        mat: &mut RowMajorMatrix<MontyField31<MP>>,
        scratch: &mut RowMajorMatrix<MontyField31<MP>>,
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        if mat.height() <= 1 {
            scratch.values.copy_from_slice(&mat.values);
            return;
        }

        // transpose input
        debug_span!(
            "initial transpose",
            nrows = mat.height(),
            ncols = mat.width()
        )
        .in_scope(|| mat.transpose_into(scratch));

        // Compute twiddle factors, or take memoized ones if already available.
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.
        debug_span!("maybe calculate twiddles").in_scope(|| {
            let curr_max_fft_len = 1 << self.twiddles.borrow().len();
            if mat.height() > curr_max_fft_len {
                let new_twiddles = MontyField31::roots_of_unity_table(mat.height());
                self.twiddles.replace(new_twiddles);
            }
        });

        let lg_fft_len = p3_util::log2_ceil_usize(mat.height());

        // TODO: We're only cloning because of the RefCell; it
        // shouldn't be necessary, though it only costs ~20μs.
        let twiddles = debug_span!("clone twiddles").in_scope(|| self.twiddles.borrow().clone());
        let roots_idx = (twiddles.len() + 1) - lg_fft_len;
        let twiddles = &twiddles[roots_idx..];

        debug_span!(
            "parallel forward dft",
            n_dfts = scratch.height(),
            lengths = scratch.width()
        )
        .in_scope(|| {
            scratch
                .par_rows_mut()
                .for_each(|v| MontyField31::forward_fft(v, twiddles))
        });
    }

    pub fn dft_batch_bitrevd_with_scratch(
        &self,
        mat: &mut RowMajorMatrix<MontyField31<MP>>,
        scratch: &mut RowMajorMatrix<MontyField31<MP>>,
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        self.dft_batch_transposed_bitrevd_with_scratch(mat, scratch);

        // transpose output
        debug_span!(
            "final transpose",
            nrows = scratch.height(),
            ncols = scratch.width()
        )
        .in_scope(|| scratch.transpose_into(mat));
    }
}

impl<MP: MontyParameters + FieldParameters + TwoAdicData> TwoAdicSubgroupDft<MontyField31<MP>>
    for Radix2Dif<MontyField31<MP>>
{
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<MontyField31<MP>>>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<MontyField31<MP>>) -> Self::Evaluations
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        let mut scratch = debug_span!("allocate scratch space")
            .in_scope(|| RowMajorMatrix::default(mat.height(), mat.width()));

        debug_span!("dft batch")
            .in_scope(|| self.dft_batch_bitrevd_with_scratch(&mut mat, &mut scratch));

        // TODO: In principle bit reversal shouldn't be necessary when
        // doing the transform inplace, though it might still be
        // beneficial for memory coherence.
        debug_span!("final bitrev").in_scope(|| mat.bit_reverse_rows())
    }

    // FIXME: Implement coset_lde_batch in terms of dft_batch_bitrevd_with_scratch()
    // and without the two transposes in the middle

    /*
    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        let result_height = mat.height().checked_shl(added_bits).unwrap();
        let mut scratch = debug_span!("allocate scratch space")
            .in_scope(|| RowMajorMatrix::default(result_height, mat.width()));

        let mut coeffs = self.idft_batch(mat);
        // PANICS: possible panic if the new resized length overflows
        coeffs.values.resize(
            coeffs
                .values
                .len()
                .checked_shl(added_bits.try_into().unwrap())
                .unwrap(),
            F::zero(),
        );
        self.coset_dft_batch(coeffs, shift)
    }
     */
}
