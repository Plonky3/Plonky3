use std::ops::Deref;

use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{PackedValue, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::EitherRow;
use p3_monty_31::dft::RecursiveDft;

/// DFT evaluations in natural or bit-reversed row order.
///
/// Row access preserves the underlying storage's specialized implementation.
#[derive(Debug)]
pub enum MaybeBitreversedMatrix<T> {
    /// Rows viewed through a bit-reversal permutation.
    Yes(BitReversedMatrixView<RowMajorMatrix<T>>),
    /// Rows in natural order.
    No(RowMajorMatrix<T>),
}

impl<T> From<RowMajorMatrix<T>> for MaybeBitreversedMatrix<T> {
    #[inline(always)]
    fn from(mat: RowMajorMatrix<T>) -> Self {
        Self::No(mat)
    }
}

impl<T> From<BitReversedMatrixView<RowMajorMatrix<T>>> for MaybeBitreversedMatrix<T> {
    #[inline(always)]
    fn from(mat: BitReversedMatrixView<RowMajorMatrix<T>>) -> Self {
        Self::Yes(mat)
    }
}

impl<T> Matrix<T> for MaybeBitreversedMatrix<T>
where
    T: Send + Sync + Clone,
{
    fn width(&self) -> usize {
        match self {
            Self::Yes(inner) => inner.width(),
            Self::No(inner) => inner.width(),
        }
    }

    fn height(&self) -> usize {
        match self {
            Self::Yes(inner) => inner.height(),
            Self::No(inner) => inner.height(),
        }
    }

    #[inline(always)]
    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
    {
        match self {
            Self::Yes(inner) => inner.to_row_major_matrix(),
            Self::No(inner) => inner.to_row_major_matrix(),
        }
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            match self {
                Self::Yes(inner) => inner.get_unchecked(r, c),
                Self::No(inner) => inner.get_unchecked(r, c),
            }
        }
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // SAFETY: The caller supplies a valid row in the unchanged dimensions.
        unsafe {
            match self {
                Self::Yes(inner) => EitherRow::Left(inner.row_unchecked(r).into_iter()),
                Self::No(inner) => EitherRow::Right(inner.row_unchecked(r).into_iter()),
            }
        }
    }

    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        // SAFETY: The caller supplies a valid row and an in-bounds column range.
        // Both dimensions match the underlying storage.
        unsafe {
            match self {
                Self::Yes(inner) => {
                    EitherRow::Left(inner.row_subseq_unchecked(r, start, end).into_iter())
                }
                Self::No(inner) => {
                    EitherRow::Right(inner.row_subseq_unchecked(r, start, end).into_iter())
                }
            }
        }
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        // SAFETY: The caller supplies a valid row in the unchanged dimensions.
        unsafe {
            match self {
                Self::Yes(inner) => EitherRow::Left(inner.row_slice_unchecked(r)),
                Self::No(inner) => EitherRow::Right(inner.row_slice_unchecked(r)),
            }
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        // SAFETY: The caller supplies a valid row and an in-bounds column range.
        // Both dimensions match the underlying storage.
        unsafe {
            match self {
                Self::Yes(inner) => EitherRow::Left(inner.row_subslice_unchecked(r, start, end)),
                Self::No(inner) => EitherRow::Right(inner.row_subslice_unchecked(r, start, end)),
            }
        }
    }

    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (
        impl Iterator<Item = P> + Send + Sync,
        impl Iterator<Item = T> + Send + Sync,
    )
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        // Preserve both the full packs and the unpacked tail.
        match self {
            Self::Yes(inner) => {
                let (packed, suffix) = inner.horizontally_packed_row(r);
                (EitherRow::Left(packed), EitherRow::Left(suffix))
            }
            Self::No(inner) => {
                let (packed, suffix) = inner.horizontally_packed_row(r);
                (EitherRow::Right(packed), EitherRow::Right(suffix))
            }
        }
    }

    fn padded_horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        // Delegate padding to the underlying storage.
        match self {
            Self::Yes(inner) => EitherRow::Left(inner.padded_horizontally_packed_row(r)),
            Self::No(inner) => EitherRow::Right(inner.padded_horizontally_packed_row(r)),
        }
    }
}

impl<T> BitReversibleMatrix<T> for MaybeBitreversedMatrix<T>
where
    T: Send + Sync + Clone,
{
    type BitRev = Self;

    #[inline(always)]
    fn bit_reverse_rows(self) -> Self::BitRev {
        match self {
            Self::Yes(inner) => inner.inner.into(),
            Self::No(inner) => BitReversalPerm::new_view(inner).into(),
        }
    }
}

/// An enum containing several different options for discrete Fourier Transform.
///
/// This implements `TwoAdicSubgroupDft` by passing to whatever the contained struct is.
#[derive(Clone, Debug)]
pub enum DftChoice<F> {
    Recursive(RecursiveDft<F>),
    Parallel(Radix2DitParallel<F>),
    SmallBatch(Radix2DFTSmallBatch<F>),
}

impl<F: Default> Default for DftChoice<F> {
    // We have to fix a default for the `TwoAdicSubgroupDft` trait. We choose `Radix2DitParallel` as one of the features
    // of `RecursiveDft` is that it works better when initialized with knowledge of the expected size.
    fn default() -> Self {
        Self::Parallel(Radix2DitParallel::<F>::default())
    }
}

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for DftChoice<F>
where
    RecursiveDft<F>: TwoAdicSubgroupDft<F, Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>>,
    Radix2DitParallel<F>:
        TwoAdicSubgroupDft<F, Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>>,
{
    type Evaluations = MaybeBitreversedMatrix<F>;

    #[inline]
    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations {
        match self {
            Self::Recursive(inner_dft) => inner_dft.dft_batch(mat).into(),
            Self::Parallel(inner_dft) => inner_dft.dft_batch(mat).into(),
            Self::SmallBatch(inner_dft) => inner_dft.dft_batch(mat).into(),
        }
    }

    #[inline]
    fn coset_dft_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> Self::Evaluations {
        match self {
            Self::Recursive(inner_dft) => inner_dft.coset_dft_batch(mat, shift).into(),
            Self::Parallel(inner_dft) => inner_dft.coset_dft_batch(mat, shift).into(),
            Self::SmallBatch(inner_dft) => inner_dft.coset_dft_batch(mat, shift).into(),
        }
    }

    #[inline]
    fn idft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        match self {
            Self::Recursive(inner_dft) => inner_dft.idft_batch(mat),
            Self::Parallel(inner_dft) => inner_dft.idft_batch(mat),
            Self::SmallBatch(inner_dft) => inner_dft.idft_batch(mat),
        }
    }

    #[inline]
    fn coset_idft_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        match self {
            Self::Recursive(inner_dft) => inner_dft.coset_idft_batch(mat, shift),
            Self::Parallel(inner_dft) => inner_dft.coset_idft_batch(mat, shift),
            Self::SmallBatch(inner_dft) => inner_dft.coset_idft_batch(mat, shift),
        }
    }

    #[inline]
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        match self {
            Self::Recursive(inner_dft) => inner_dft.coset_lde_batch(mat, added_bits, shift).into(),
            Self::Parallel(inner_dft) => inner_dft.coset_lde_batch(mat, added_bits, shift).into(),
            Self::SmallBatch(inner_dft) => inner_dft.coset_lde_batch(mat, added_bits, shift).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::FieldArray;
    use p3_matrix::Matrix;
    use p3_matrix::bitrev::BitReversalPerm;
    use p3_matrix::dense::RowMajorMatrix;

    use super::MaybeBitreversedMatrix;

    type F = BabyBear;

    fn assert_accessors_match<M: Matrix<F>>(wrapper: &MaybeBitreversedMatrix<F>, inner: &M) {
        // Invariant: wrapping preserves dimensions and logical row contents.
        assert_eq!(wrapper.width(), inner.width());
        assert_eq!(wrapper.height(), inner.height());

        // Safe access rejects the first row beyond the matrix.
        assert!(wrapper.row(inner.height()).is_none());
        assert!(wrapper.row_slice(inner.height()).is_none());

        for r in 0..inner.height() {
            // Iteration and borrowed slices must expose the same logical row.
            let actual: Vec<F> = wrapper.row(r).unwrap().into_iter().collect();
            let expected: Vec<F> = inner.row(r).unwrap().into_iter().collect();
            assert_eq!(actual, expected, "row({r})");

            // Compare slices without allocating copies.
            assert_eq!(
                &*wrapper.row_slice(r).unwrap(),
                &*inner.row_slice(r).unwrap(),
                "row_slice({r})"
            );

            // Cover every column range, including empty ranges at both ends.
            for start in 0..=inner.width() {
                for end in start..=inner.width() {
                    // SAFETY: The row exists and the loop bounds keep the range within its width.
                    let actual: Vec<F> = unsafe { wrapper.row_subseq_unchecked(r, start, end) }
                        .into_iter()
                        .collect();

                    // SAFETY: The same bounds apply to the underlying matrix.
                    let expected: Vec<F> = unsafe { inner.row_subseq_unchecked(r, start, end) }
                        .into_iter()
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "row_subseq_unchecked({r}, {start}, {end})"
                    );

                    // SAFETY: Both matrices have the checked row and column bounds.
                    assert_eq!(
                        &*unsafe { wrapper.row_subslice_unchecked(r, start, end) },
                        &*unsafe { inner.row_subslice_unchecked(r, start, end) },
                        "row_subslice_unchecked({r}, {start}, {end})"
                    );
                }
            }

            // Two lanes exercise short rows, exact packs, and a trailing element.
            let (actual_packed, actual_suffix) =
                wrapper.horizontally_packed_row::<FieldArray<F, 2>>(r);
            let (expected_packed, expected_suffix) =
                inner.horizontally_packed_row::<FieldArray<F, 2>>(r);
            assert_eq!(
                actual_packed.collect::<Vec<_>>(),
                expected_packed.collect::<Vec<_>>(),
                "horizontally_packed_row({r}), packed half"
            );
            assert_eq!(
                actual_suffix.collect::<Vec<_>>(),
                expected_suffix.collect::<Vec<_>>(),
                "horizontally_packed_row({r}), suffix half"
            );

            // The final pack must retain the tail and pad unused lanes with zero.
            assert_eq!(
                wrapper
                    .padded_horizontally_packed_row::<FieldArray<F, 2>>(r)
                    .collect::<Vec<_>>(),
                inner
                    .padded_horizontally_packed_row::<FieldArray<F, 2>>(r)
                    .collect::<Vec<_>>(),
                "padded_horizontally_packed_row({r})"
            );
        }
    }

    #[test]
    fn no_variant_matches_the_row_major_matrix() {
        // Include a single row and an empty matrix in natural order.
        for height in [0, 1, 4, 8] {
            for width in 1..=4 {
                // Distinct entries expose row-order and column-range errors.
                let inner =
                    RowMajorMatrix::new((1..=(height * width) as u32).map(F::new).collect(), width);
                let wrapper = MaybeBitreversedMatrix::No(inner.clone());

                // Every access path must agree with the underlying storage.
                assert_accessors_match(&wrapper, &inner);

                // Materialization preserves the natural row order.
                assert_eq!(wrapper.to_row_major_matrix().values, inner.values);
            }
        }
    }

    #[test]
    fn yes_variant_matches_the_bit_reversed_view() {
        // Bit reversal needs a positive power-of-two height.
        for height in [1, 4, 8] {
            for width in 1..=4 {
                // Distinct rows expose the permutation: [0, 1, 2, 3] -> [0, 2, 1, 3].
                let matrix =
                    RowMajorMatrix::new((1..=(height * width) as u32).map(F::new).collect(), width);
                let inner = BitReversalPerm::new_view(matrix);
                let wrapper =
                    MaybeBitreversedMatrix::Yes(BitReversalPerm::new_view(inner.inner.clone()));

                // Every access path must apply the same row permutation.
                assert_accessors_match(&wrapper, &inner);

                // Materialization applies the permutation to the stored rows.
                assert_eq!(
                    wrapper.to_row_major_matrix().values,
                    inner.to_row_major_matrix().values
                );
            }
        }
    }
}
