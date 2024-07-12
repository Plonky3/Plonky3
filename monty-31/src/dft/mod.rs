//! An implementation of the FFT for `MontyField31`
extern crate alloc;

use alloc::vec::Vec;
use core::cell::RefCell;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

mod backward;
mod forward;

use crate::{FieldParameters, MontyField31, MontyParameters, TwoAdicData};

// TODO: These are only pub for benches at the moment...
//pub mod backward;

// TODO: These two functions belong somewhere else

/// Copied from Rust nightly sources
#[inline(always)]
unsafe fn from_raw_parts_mut<'a, T>(data: *mut T, len: usize) -> &'a mut [T] {
    unsafe { &mut *core::ptr::slice_from_raw_parts_mut(data, len) }
}

/// Copied from Rust nightly sources
#[inline(always)]
pub(crate) unsafe fn split_at_mut_unchecked<T>(v: &mut [T], mid: usize) -> (&mut [T], &mut [T]) {
    let len = v.len();
    let ptr = v.as_mut_ptr();

    // SAFETY: Caller has to check that `0 <= mid <= self.len()`.
    //
    // `[ptr; mid]` and `[mid; len]` are not overlapping, so returning
    // a mutable reference is fine.
    unsafe {
        (
            from_raw_parts_mut(ptr, mid),
            from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

/// The DIT FFT algorithm.
#[derive(Default, Clone, Debug)]
pub struct Radix2Dit<F> {
    /// Memoized twiddle factors for each length log_n.
    ///
    /// TODO: The use of RefCell means this can't be shared across
    /// threads; consider using RwLock or finding a better design
    /// instead.
    twiddles: RefCell<Vec<Vec<F>>>,
}

impl<MP: FieldParameters + TwoAdicData> Radix2Dit<MontyField31<MP>> {
    pub fn new(n: usize) -> Self {
        Self {
            twiddles: RefCell::new(MontyField31::roots_of_unity_table(n)),
        }
    }
}

impl<MP: MontyParameters + FieldParameters + TwoAdicData> TwoAdicSubgroupDft<MontyField31<MP>>
    for Radix2Dit<MontyField31<MP>>
{
    type Evaluations = RowMajorMatrix<MontyField31<MP>>;

    fn dft_batch(
        &self,
        mut mat: RowMajorMatrix<MontyField31<MP>>,
    ) -> RowMajorMatrix<MontyField31<MP>>
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        let mut scratch = RowMajorMatrix::default(mat.height(), mat.width());

        // transpose input
        mat.transpose_into(&mut scratch);

        // Compute twiddle factors, or take memoized ones if already available.
        let curr_max_fft_len = 1 << self.twiddles.borrow().len();
        if mat.height() > curr_max_fft_len {
            let new_twiddles = MontyField31::roots_of_unity_table(mat.height());
            self.twiddles.replace(new_twiddles);
        }

        scratch
            .par_rows_mut()
            .for_each(|v| MontyField31::forward_fft(v, &self.twiddles.borrow()));

        // FIXME: depending on what the result is being used for, we
        // can potentially avoid one or both of the final transpose
        // and bit reversal.

        // transpose output
        scratch.transpose_into(&mut mat);

        // FIXME: Either do bit reversal or don't do inplace
        //mat.bit_reverse_rows();

        mat
    }
}
