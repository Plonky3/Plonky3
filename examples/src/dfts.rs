use std::ops::Deref;

use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{PackedValue, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::RowMajorMatrix;
use p3_monty_31::dft::RecursiveDft;

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

// TODO: This is pretty nasty. Anyone actually using the matrix trait for
// MaybeBitreversedMatrix outside of the to_row_major_matrix trait is going to have
// a bad time. Might want to refactor this somehow. We only use to_row_major_matrix
// from this trait in the proving loop so it's not a huge issue for now.
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
            Self::Yes(_) => {
                unimplemented!("Not implemented for MaybeBitreversedMatrix")
            }
            Self::No(inner) => unsafe { inner.row_unchecked(r) },
        }
    }

    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        match self {
            Self::Yes(inner) => unsafe { inner.row_subseq_unchecked(r, start, end) },
            Self::No(_) => {
                unimplemented!("Not implemented for MaybeBitreversedMatrix")
            }
        }
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        match self {
            Self::Yes(inner) => unsafe { inner.row_slice_unchecked(r) },
            Self::No(_) => {
                unimplemented!("Not implemented for MaybeBitreversedMatrix")
            }
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        match self {
            Self::Yes(_) => {
                unimplemented!("Not implemented for MaybeBitreversedMatrix")
            }
            Self::No(inner) => unsafe { inner.row_subslice_unchecked(r, start, end) },
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
            Self::Yes(inner) => inner.horizontally_packed_row(r),
            Self::No(_) => {
                unimplemented!("Not implemented for MaybeBitreversedMatrix")
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
            Self::Yes(_) => {
                unimplemented!("Not implemented for MaybeBitreversedMatrix")
            }
            Self::No(inner) => inner.padded_horizontally_packed_row(r),
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
