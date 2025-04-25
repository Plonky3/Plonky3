use alloc::vec::Vec;
use core::iter;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_field::{ExtensionField, Field};

use crate::Matrix;

/// A view that flattens a matrix of extension field elements into a matrix of base field elements.
///
/// Each element of the original matrix is an extension field element `EF`, composed of several
/// base field elements `F`. This view expands each `EF` element into its base field components,
/// effectively increasing the number of columns (width) while keeping the number of rows unchanged.
#[derive(Debug)]
pub struct FlatMatrixView<F, EF, Inner>(Inner, PhantomData<(F, EF)>);

impl<F, EF, Inner> FlatMatrixView<F, EF, Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self(inner, PhantomData)
    }
}

impl<F, EF, Inner> Deref for FlatMatrixView<F, EF, Inner> {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, EF, Inner> Matrix<F> for FlatMatrixView<F, EF, Inner>
where
    F: Field,
    EF: ExtensionField<F>,
    Inner: Matrix<EF>,
{
    fn width(&self) -> usize {
        self.0.width() * EF::DIMENSION
    }

    fn height(&self) -> usize {
        self.0.height()
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> F {
        // The c'th base field element in a row of extension field elements is
        // at index c % EF::DIMENSION in the c / EF::DIMENSION'th extension element.
        let c_inner = c / EF::DIMENSION;
        let inner = unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width().
            // Assuming this, c / EF::DIMENSION < self.0.width().
            self.0.get_unchecked(r, c_inner)
        };
        inner.as_basis_coefficients_slice()[c % EF::DIMENSION]
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = F, IntoIter = impl Iterator<Item = F> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height().
            FlatIter {
                inner: self.0.row_unchecked(r).into_iter().peekable(),
                idx: 0,
                _phantom: PhantomData,
            }
        }
    }

    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = F, IntoIter = impl Iterator<Item = F> + Send + Sync> {
        // We can skip the first start / EF::DIMENSION elements in the row.
        let len = end - start;
        let inner_start = start / EF::DIMENSION;
        unsafe {
            // Safety: The caller must ensure that r < self.height(), start <= end and end < self.width().
            FlatIter {
                inner: self
                    .0
                    // We set end to be the width of the inner matrix and use take to ensure we get the right
                    // number of elements.
                    .row_subseq_unchecked(r, inner_start, self.0.width())
                    .into_iter()
                    .peekable(),
                idx: start,
                _phantom: PhantomData,
            }
            .take(len)
        }
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [F]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height().
            self.0
                .row_slice_unchecked(r)
                .iter()
                .flat_map(|val| val.as_basis_coefficients_slice())
                .copied()
                .collect::<Vec<_>>()
        }
    }
}

pub struct FlatIter<F, I: Iterator> {
    inner: iter::Peekable<I>,
    idx: usize,
    _phantom: PhantomData<F>,
}

impl<F, EF, I> Iterator for FlatIter<F, I>
where
    F: Field,
    EF: ExtensionField<F>,
    I: Iterator<Item = EF>,
{
    type Item = F;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == EF::DIMENSION {
            self.idx = 0;
            self.inner.next();
        }
        let value = self.inner.peek()?.as_basis_coefficients_slice()[self.idx];
        self.idx += 1;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_field::extension::Complex;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::dense::RowMajorMatrix;
    type F = Mersenne31;
    type EF = Complex<Mersenne31>;

    #[test]
    fn flat_matrix() {
        let values = vec![
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 10)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 20)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 30)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 40)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 50)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 60)),
        ];
        let ext = RowMajorMatrix::<EF>::new(values, 2);
        let flat = FlatMatrixView::<F, EF, _>::new(ext);

        assert_eq!(flat.width(), 4);
        assert_eq!(flat.height(), 3);

        assert_eq!(flat.get(0, 2), Some(F::from_u8(20)));
        assert_eq!(flat.get(1, 3), Some(F::from_u8(41)));
        assert_eq!(flat.get(2, 0), Some(F::from_u8(50)));

        unsafe {
            assert_eq!(flat.get_unchecked(0, 1), F::from_u8(11));
            assert_eq!(flat.get_unchecked(1, 0), F::from_u8(30));
            assert_eq!(flat.get_unchecked(2, 2), F::from_u8(60));
        }

        assert_eq!(
            &*flat.row_slice(0).unwrap(),
            &[10, 11, 20, 21].map(F::from_u8)
        );
        unsafe {
            assert_eq!(
                &*flat.row_slice_unchecked(1),
                &[30, 31, 40, 41].map(F::from_u8)
            );
            assert_eq!(
                &*flat.row_subslice_unchecked(2, 0, 3),
                &[50, 51, 60].map(F::from_u8)
            );
        }

        assert_eq!(
            flat.row(2).unwrap().into_iter().collect_vec(),
            [50, 51, 60, 61].map(F::from_u8)
        );
        unsafe {
            assert_eq!(
                flat.row_unchecked(1).into_iter().collect_vec(),
                [30, 31, 40, 41].map(F::from_u8)
            );
            assert_eq!(
                flat.row_subseq_unchecked(0, 1, 4).into_iter().collect_vec(),
                [11, 20, 21].map(F::from_u8)
            );
        }

        assert!(flat.get(0, 4).is_none()); // Width out of bounds
        assert!(flat.get(3, 0).is_none()); // Height out of bounds
        assert!(flat.row(3).is_none()); // Height out of bounds
        assert!(flat.row_slice(3).is_none()); // Height out of bounds
    }

    #[test]
    fn test_flat_matrix_width() {
        // Create a 2-column, 2-row matrix of EF elements.
        // Each EF element expands to EF::DIMENSION base field elements when flattened.
        // Therefore, the flattened width should be 2 * EF::DIMENSION.
        let matrix = RowMajorMatrix::<EF>::new(vec![EF::default(); 4], 2);
        let flat = FlatMatrixView::<F, EF, _>::new(matrix);
        assert_eq!(flat.width(), 2 * <EF as BasedVectorSpace<F>>::DIMENSION);
    }

    #[test]
    fn test_flat_matrix_height() {
        // Construct a 3-column matrix with 6 EF elements (2 rows).
        // The flattened view should preserve the original number of rows.
        let matrix = RowMajorMatrix::<EF>::new(vec![EF::default(); 6], 3);
        let flat = FlatMatrixView::<F, EF, _>::new(matrix);
        assert_eq!(flat.height(), 2);
    }

    #[test]
    fn test_flat_matrix_row_iterator() {
        // Create a single row of two EF elements:
        // First EF = [1, 2], second EF = [10, 11] (in base field representation).
        let values = vec![
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 1)),
            EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + 10)),
        ];
        let matrix = RowMajorMatrix::new(values, 2);
        let flat = FlatMatrixView::<F, EF, _>::new(matrix);

        // Flattened row should concatenate basis coefficients of both EF elements.
        let row: Vec<_> = flat.first_row().unwrap().into_iter().collect();
        let expected = [1, 2, 10, 11].map(F::from_u8).to_vec();

        assert_eq!(row, expected);
    }

    #[test]
    fn test_flat_matrix_row_slice_correctness() {
        // Construct a row with two EF values: [1, 2] and [10, 11].
        // Verify that row_slice() correctly returns a flat &[F] of base field values.
        let ef = |offset| EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + offset));
        let matrix = RowMajorMatrix::new(vec![ef(1), ef(10)], 2);
        let flat = FlatMatrixView::<F, EF, _>::new(matrix);

        assert_eq!(
            &*flat.row_slice(0).unwrap(),
            &[1, 2, 10, 11].map(F::from_u8)
        );
    }

    #[test]
    fn test_flat_matrix_empty() {
        // Edge case: test behavior on empty matrix.
        // Expect zero width and height in the flattened view.
        let matrix = RowMajorMatrix::<EF>::new(vec![], 0);
        let flat = FlatMatrixView::<F, EF, _>::new(matrix);

        assert_eq!(flat.height(), 0);
        assert_eq!(flat.width(), 0);
    }

    #[test]
    fn test_flat_iter_length_and_values() {
        // Create a row with three EF values, each with offset base coefficients:
        // [0,1], [10,11], [20,21] -> flattened row should be [0,1,10,11,20,21].
        let ef = |offset| EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + offset));
        let values = vec![ef(0), ef(10), ef(20)];
        let matrix = RowMajorMatrix::new(values, 3); // 1 row
        let flat = FlatMatrixView::<F, EF, _>::new(matrix);

        let row: Vec<_> = flat.first_row().unwrap().into_iter().collect();
        let expected = [0, 1, 10, 11, 20, 21].map(F::from_u8).to_vec();
        assert_eq!(row, expected);
    }

    #[test]
    fn test_flat_matrix_multiple_rows() {
        // Construct a 2-column, 2-row matrix of EF values, with varying offsets per row.
        // Row 0: [0,1], [10,11]; Row 1: [20,21], [30,31].
        // Verify that the flattening preserves row structure and ordering.
        let ef = |base| EF::from_basis_coefficients_fn(|i| F::from_u8(base + i as u8));
        let matrix = RowMajorMatrix::new(vec![ef(0), ef(10), ef(20), ef(30)], 2);
        let flat = FlatMatrixView::<F, EF, _>::new(matrix);

        let row0: Vec<_> = flat.first_row().unwrap().into_iter().collect();
        let row1: Vec<_> = flat.row(1).unwrap().into_iter().collect();

        assert_eq!(row0, [0, 1, 10, 11].map(F::from_u8).to_vec());
        assert_eq!(row1, [20, 21, 30, 31].map(F::from_u8).to_vec());
    }

    #[test]
    fn test_flat_iter_yields_across_multiple_efs() {
        // Build 1 row with 3 EF elements:
        // - ef(0)   = [0, 1]
        // - ef(10)  = [10, 11]
        // - ef(20)  = [20, 21]
        //
        // The flattened row should yield:
        // [0, 1, 10, 11, 20, 21] as base field elements (F)
        let ef = |offset| EF::from_basis_coefficients_fn(|i| F::from_u8(i as u8 + offset));
        let matrix = RowMajorMatrix::new(vec![ef(0), ef(10), ef(20)], 3); // 1 row, 3 EF elements
        let flat = FlatMatrixView::<F, EF, _>::new(matrix);

        let mut row_iter = flat.row(0).unwrap().into_iter();

        // Expected flattened result
        let expected = [0, 1, 10, 11, 20, 21].map(F::from_u8);

        for expected_val in expected {
            assert_eq!(row_iter.next(), Some(expected_val));
        }

        // Iterator should now be exhausted
        assert_eq!(row_iter.next(), None);
    }
}
