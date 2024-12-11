use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::TwoAdicField;
use p3_matrix::{bitrev::BitReversedMatrixView, dense::RowMajorMatrix};
use p3_monty_31::dft::RecursiveDft;

/// An enum containing several different options for discrete Fourier Transform.
///
/// This implements `TwoAdicSubgroupDft` by passing to whatever the contained struct is.
#[derive(Clone, Debug)]
pub enum DFTOptions<F> {
    Recursive(RecursiveDft<F>),
    Parallel(Radix2DitParallel<F>),
}

impl<F: Default> Default for DFTOptions<F> {
    fn default() -> Self {
        DFTOptions::<F>::Parallel(Radix2DitParallel::<F>::default())
    }
}

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for DFTOptions<F>
where
    RecursiveDft<F>: TwoAdicSubgroupDft<F, Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>>,
    Radix2DitParallel<F>:
        TwoAdicSubgroupDft<F, Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>>,
{
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>;

    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations {
        match self {
            DFTOptions::<F>::Recursive(inner_dft) => inner_dft.dft_batch(mat),
            DFTOptions::<F>::Parallel(inner_dft) => inner_dft.dft_batch(mat),
        }
    }

    fn coset_dft_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> Self::Evaluations {
        match self {
            DFTOptions::<F>::Recursive(inner_dft) => inner_dft.coset_dft_batch(mat, shift),
            DFTOptions::<F>::Parallel(inner_dft) => inner_dft.coset_dft_batch(mat, shift),
        }
    }

    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        match self {
            DFTOptions::<F>::Recursive(inner_dft) => {
                inner_dft.coset_lde_batch(mat, added_bits, shift)
            }
            DFTOptions::<F>::Parallel(inner_dft) => {
                inner_dft.coset_lde_batch(mat, added_bits, shift)
            }
        }
    }
}
