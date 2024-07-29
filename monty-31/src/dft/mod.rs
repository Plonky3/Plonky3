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
        // transpose input
        mat.transpose_into(scratch);

        // Compute twiddle factors, or take memoized ones if already available.
        let curr_max_fft_len = 1 << self.twiddles.borrow().len();
        if mat.height() > curr_max_fft_len {
            let new_twiddles = MontyField31::roots_of_unity_table(mat.height());
            self.twiddles.replace(new_twiddles);
        }

        // TODO: We're only cloning because of the RefCell; it shouldn't be necessary
        let twiddles = self.twiddles.borrow().clone();
        scratch
            .par_rows_mut()
            .for_each(|v| MontyField31::forward_fft(v, &twiddles));
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
        scratch.transpose_into(mat);
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
        let mut scratch = RowMajorMatrix::default(mat.height(), mat.width());
        self.dft_batch_bitrevd_with_scratch(&mut mat, &mut scratch);

        // TODO: In principle bit reversal shouldn't be necessary when
        // doing the transform inplace, though it might still be
        // beneficial for memory coherence.
        mat.bit_reverse_rows()
    }
}
