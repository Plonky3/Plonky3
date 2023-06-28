use crate::dense::RowMajorMatrix;
use crate::sparse::CsrMatrix;
use crate::{Matrix, MatrixRows};
use alloc::vec;
use p3_field::Field;
use p3_maybe_rayon::{MaybeIntoParIter, ParallelIterator};

/// Compute `C = A * B`, where `A` in a CSR matrix and `B` is a dense matrix.
///
/// # Panics
/// Panics if dimensions of input matrices don't match.
pub fn mul_csr_dense<'a, F, B>(a: &CsrMatrix<F>, b: &'a B) -> RowMajorMatrix<F>
where
    F: Field,
    B: MatrixRows<'a, F> + Sync,
{
    assert_eq!(a.width(), b.height(), "A, B dimensions don't match");
    let c_width = b.width();

    let c_values = (0..a.height())
        .into_par_iter()
        .flat_map(|a_row_idx| {
            let mut c_row = vec![F::ZERO; c_width];
            for &(a_col_idx, a_val) in a.row(a_row_idx) {
                add_scaled_slice_in_place(&mut c_row, b.row(a_col_idx).into_iter(), a_val);
            }
            c_row
        })
        .collect();

    RowMajorMatrix::new(c_values, c_width)
}

/// `x += y * s`, where `s` is a scalar.
fn add_scaled_slice_in_place<'a, F, Y>(x: &mut [F], y: Y, s: F)
where
    F: Field,
    Y: Iterator<Item = &'a F>,
{
    // TODO: Use PackedField
    x.iter_mut().zip(y).for_each(|(x_i, y_i)| *x_i += *y_i * s);
}
