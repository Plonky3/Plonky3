use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_field::{ExtensionField, Field};
use p3_matrix::extension::FlatMatrixView;
use p3_matrix::{Dimensions, Matrix};

use crate::{BatchOpening, BatchOpeningRef, Mmcs};

/// Errors encountered while verifying an extension-field MMCS opening.
#[derive(Debug, thiserror::Error)]
pub enum ExtensionMmcsError<E> {
    /// A claimed extension width cannot be represented in base-field coordinates.
    #[error("extension width {width} for matrix {matrix} overflows in base-field coordinates")]
    WidthOverflow {
        /// Index of the matrix in the batch.
        matrix: usize,
        /// Claimed width in extension-field elements.
        width: usize,
    },
    /// A query count or matrix count does not match the verifier's dimensions.
    #[error("batch size mismatch: expected {expected} entries, got {got}")]
    WrongBatchSize {
        /// Number of queries or matrices expected by the verifier.
        expected: usize,
        /// Number supplied in the opening.
        got: usize,
    },
    /// An opened row does not have its claimed extension-field width.
    #[error("wrong extension width: matrix {matrix} expected {expected} values, got {got}")]
    WrongWidth {
        /// Index of the matrix in the batch.
        matrix: usize,
        /// Claimed width in extension-field elements.
        expected: usize,
        /// Number of extension-field elements in the opened row.
        got: usize,
    },
    /// Verification by the underlying base-field MMCS failed.
    #[error("inner MMCS verification failed: {0:?}")]
    Inner(E),
}

/// A wrapper to lift an MMCS from a base field `F` to an extension field `EF`.
///
/// `ExtensionMmcs` allows committing to and opening matrices over an extension field by internally
/// using an MMCS defined on the base field. It works by flattening each extension field element
/// into its base field coordinates for commitment, and then reconstructing them on opening.
#[derive(Clone, Debug)]
pub struct ExtensionMmcs<F, EF, InnerMmcs> {
    /// The inner MMCS instance used to handle commitments at the base field level.
    pub(crate) inner: InnerMmcs,

    pub(crate) _phantom: PhantomData<(F, EF)>,
}

impl<F, EF, InnerMmcs> ExtensionMmcs<F, EF, InnerMmcs> {
    pub const fn new(inner: InnerMmcs) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<F, EF, InnerMmcs> ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: Mmcs<F>,
{
    fn base_dimensions(
        dimensions: &[Dimensions],
    ) -> Result<Vec<Dimensions>, ExtensionMmcsError<InnerMmcs::Error>> {
        dimensions
            .iter()
            .enumerate()
            .map(|(matrix, dim)| {
                let width = dim.width.checked_mul(EF::DIMENSION).ok_or(
                    ExtensionMmcsError::WidthOverflow {
                        matrix,
                        width: dim.width,
                    },
                )?;
                Ok(Dimensions {
                    width,
                    height: dim.height,
                })
            })
            .collect()
    }

    fn check_widths<R: AsRef<[EF]>>(
        dimensions: &[Dimensions],
        rows: &[R],
    ) -> Result<(), ExtensionMmcsError<InnerMmcs::Error>> {
        for (matrix, (dim, row)) in dimensions.iter().zip(rows).enumerate() {
            let got = row.as_ref().len();
            if dim.width != got {
                return Err(ExtensionMmcsError::WrongWidth {
                    matrix,
                    expected: dim.width,
                    got,
                });
            }
        }
        Ok(())
    }
}

impl<F, EF, InnerMmcs> Mmcs<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: Mmcs<F>,
{
    type ProverData<M> = InnerMmcs::ProverData<FlatMatrixView<F, EF, M>>;
    type Commitment = InnerMmcs::Commitment;
    type Proof = InnerMmcs::Proof;
    type MultiProof = InnerMmcs::MultiProof;
    type Error = ExtensionMmcsError<InnerMmcs::Error>;

    fn commit<M: Matrix<EF>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        self.inner
            .commit(inputs.into_iter().map(FlatMatrixView::new).collect())
    }

    fn open_batch<M: Matrix<EF>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<EF, Self> {
        let (inner_opened_values, inner_proof) = self.inner.open_batch(index, prover_data).unpack();
        let opened_ext_values = inner_opened_values
            .into_iter()
            .map(EF::reconstitute_from_base)
            .collect();
        BatchOpening::new(opened_ext_values, inner_proof)
    }

    fn get_matrices<'a, M: Matrix<EF>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .map(|mat| mat.deref())
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, EF, Self>,
    ) -> Result<(), Self::Error> {
        if batch_opening.opened_values.len() != dimensions.len() {
            return Err(ExtensionMmcsError::WrongBatchSize {
                expected: dimensions.len(),
                got: batch_opening.opened_values.len(),
            });
        }
        let base_dimensions = Self::base_dimensions(dimensions)?;
        Self::check_widths(dimensions, batch_opening.opened_values)?;
        let opened_base_values: Vec<Vec<F>> = batch_opening
            .opened_values
            .iter()
            .cloned()
            .map(EF::flatten_to_base)
            .collect();
        self.inner
            .verify_batch(
                commit,
                &base_dimensions,
                index,
                BatchOpeningRef::new(&opened_base_values, batch_opening.opening_proof),
            )
            .map_err(ExtensionMmcsError::Inner)
    }

    fn open_multi_batch<M: Matrix<EF>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<EF>>>, Self::MultiProof) {
        let (base_values, proof) = self.inner.open_multi_batch(indices, prover_data);
        let ext_values = base_values
            .into_iter()
            .map(|rows| rows.into_iter().map(EF::reconstitute_from_base).collect())
            .collect();
        (ext_values, proof)
    }

    fn verify_multi_batch<R: AsRef<[EF]> + PartialEq>(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        if opened_values.len() != indices.len() {
            return Err(ExtensionMmcsError::WrongBatchSize {
                expected: indices.len(),
                got: opened_values.len(),
            });
        }
        for rows in opened_values {
            if rows.len() != dimensions.len() {
                return Err(ExtensionMmcsError::WrongBatchSize {
                    expected: dimensions.len(),
                    got: rows.len(),
                });
            }
        }
        let base_dimensions = Self::base_dimensions(dimensions)?;
        for rows in opened_values {
            Self::check_widths(dimensions, rows)?;
        }
        // Each extension row reinterprets as `EF::DIMENSION` base elements.
        // This layer always materializes owned base rows.
        let opened_base_values: Vec<Vec<Vec<F>>> = opened_values
            .iter()
            .map(|rows| {
                rows.iter()
                    .map(|row| EF::flatten_to_base(row.as_ref().to_vec()))
                    .collect()
            })
            .collect();
        self.inner
            .verify_multi_batch(
                commit,
                &base_dimensions,
                indices,
                &opened_base_values,
                proof,
            )
            .map_err(ExtensionMmcsError::Inner)
    }
}
