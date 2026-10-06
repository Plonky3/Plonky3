use std::ops::Deref;

use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{PackedValue, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::EitherRow;
use p3_monty_31::dft::RecursiveDft;

/// A matrix whose rows may or may not have been bit-reversed.
///
/// The two arms come from DFT backends that disagree on whether their evaluations come back in
/// bit-reversed order, and the wrapper is what lets [`DftChoice`] name a single `Evaluations`
/// type.
///
/// Every [`Matrix`] accessor delegates to whichever arm is present, wrapping the two possible
/// return types in [`EitherRow`]. Delegating, rather than leaning on the trait's defaults, keeps
/// each inner matrix's specialized implementation: the defaults for the row-slice family fall
/// back to collecting a fresh `Vec` per row.
#[derive(Debug)]
pub enum MaybeBitreversedMatrix<T> {
    Yes(BitReversedMatrixView<RowMajorMatrix<T>>),
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
        match self {
            // Safety: The caller must ensure that `r < self.height()`.
            Self::Yes(inner) => EitherRow::Left(unsafe { inner.row_unchecked(r) }.into_iter()),
            // Safety: The caller must ensure that `r < self.height()`.
            Self::No(inner) => EitherRow::Right(unsafe { inner.row_unchecked(r) }.into_iter()),
        }
    }

    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        match self {
            // Safety: The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
            Self::Yes(inner) => {
                EitherRow::Left(unsafe { inner.row_subseq_unchecked(r, start, end) }.into_iter())
            }
            // Safety: The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
            Self::No(inner) => {
                EitherRow::Right(unsafe { inner.row_subseq_unchecked(r, start, end) }.into_iter())
            }
        }
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        match self {
            // Safety: The caller must ensure that `r < self.height()`.
            Self::Yes(inner) => EitherRow::Left(unsafe { inner.row_slice_unchecked(r) }),
            // Safety: The caller must ensure that `r < self.height()`.
            Self::No(inner) => EitherRow::Right(unsafe { inner.row_slice_unchecked(r) }),
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        match self {
            // Safety: The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
            Self::Yes(inner) => {
                EitherRow::Left(unsafe { inner.row_subslice_unchecked(r, start, end) })
            }
            // Safety: The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
            Self::No(inner) => {
                EitherRow::Right(unsafe { inner.row_subslice_unchecked(r, start, end) })
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

    /// Three columns, against a packing width of two, so `horizontally_packed_row` returns one
    /// packed element and one trailing element rather than an empty tail.
    const WIDTH: usize = 3;

    /// Four rows, so the bit-reversal permutation actually moves a row. Row `r` holds
    /// `[3r + 1, 3r + 2, 3r + 3]`.
    fn matrix() -> RowMajorMatrix<F> {
        RowMajorMatrix::new((1u32..=12).map(F::new).collect(), WIDTH)
    }

    /// Every `Matrix` accessor of `wrapper` must answer exactly what the matrix it wraps answers.
    ///
    /// Before the fix each variant panicked in three of these, in complementary places: `Yes` on
    /// `row_unchecked`, `row_subslice_unchecked` and `padded_horizontally_packed_row`, and `No`
    /// on `row_subseq_unchecked`, `row_slice_unchecked` and `horizontally_packed_row`.
    fn assert_accessors_match<M: Matrix<F>>(wrapper: &MaybeBitreversedMatrix<F>, inner: &M) {
        assert_eq!(wrapper.width(), inner.width());
        assert_eq!(wrapper.height(), inner.height());

        for r in 0..inner.height() {
            let actual: Vec<F> = wrapper.row(r).unwrap().into_iter().collect();
            let expected: Vec<F> = inner.row(r).unwrap().into_iter().collect();
            assert_eq!(actual, expected, "row({r})");

            assert_eq!(
                wrapper.row_slice(r).unwrap().to_vec(),
                inner.row_slice(r).unwrap().to_vec(),
                "row_slice({r})"
            );

            for (start, end) in [(0, WIDTH), (1, WIDTH), (2, 3)] {
                let actual: Vec<F> = unsafe { wrapper.row_subseq_unchecked(r, start, end) }
                    .into_iter()
                    .collect();
                let expected: Vec<F> = unsafe { inner.row_subseq_unchecked(r, start, end) }
                    .into_iter()
                    .collect();
                assert_eq!(
                    actual, expected,
                    "row_subseq_unchecked({r}, {start}, {end})"
                );

                assert_eq!(
                    unsafe { wrapper.row_subslice_unchecked(r, start, end) }.to_vec(),
                    unsafe { inner.row_subslice_unchecked(r, start, end) }.to_vec(),
                    "row_subslice_unchecked({r}, {start}, {end})"
                );
            }

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
        let wrapper = MaybeBitreversedMatrix::No(matrix());
        assert_accessors_match(&wrapper, &matrix());
        assert_eq!(wrapper.to_row_major_matrix().values, matrix().values);
    }

    #[test]
    fn yes_variant_matches_the_bit_reversed_view() {
        let wrapper = MaybeBitreversedMatrix::Yes(BitReversalPerm::new_view(matrix()));
        assert_accessors_match(&wrapper, &BitReversalPerm::new_view(matrix()));
        assert_eq!(
            wrapper.to_row_major_matrix().values,
            BitReversalPerm::new_view(matrix())
                .to_row_major_matrix()
                .values
        );
    }
}
