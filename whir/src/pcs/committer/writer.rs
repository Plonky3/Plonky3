use p3_challenger::CanObserve;
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::extension::FlatMatrixView;
use p3_multilinear_util::poly::Poly;
use tracing::info_span;

use crate::sumcheck::strategy::VariableOrder;

/// Encodes and commits the initial base-field polynomial.
///
/// This is the first WHIR commitment. It lays out the polynomial according to
/// the residual variable order, applies the Reed-Solomon expansion with `dft`,
/// commits the resulting codeword matrix with `mmcs`, and observes the Merkle
/// root in the transcript.
///
/// Prefix order transposes the local folding block before padding so the first
/// folded variables become columns. Suffix order keeps the folding block as the
/// row width and only zero-pads the row count.
pub(crate) fn commit_base<F, Dft, MT, Challenger>(
    order: VariableOrder,
    dft: &Dft,
    mmcs: &MT,
    challenger: &mut Challenger,
    poly: &Poly<F>,
    folding: usize,
    starting_log_inv_rate: usize,
) -> (MT::Commitment, MT::ProverData<DenseMatrix<F>>)
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: CanObserve<MT::Commitment>,
{
    let num_variables = poly.num_variables();
    let height = 1 << (num_variables + starting_log_inv_rate - folding);

    let encoded = match order {
        VariableOrder::Prefix => {
            let padded = info_span!("transpose & pad").in_scope(|| {
                let mut mat =
                    RowMajorMatrixView::new(poly.as_slice(), 1 << (num_variables - folding))
                        .transpose();
                mat.pad_to_height(height, F::ZERO);
                mat
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_batch(padded).to_row_major_matrix())
        }
        VariableOrder::Suffix => {
            let padded = info_span!("pad").in_scope(|| {
                let mut mat = RowMajorMatrix::new(poly.as_slice().to_vec(), 1 << folding);
                mat.pad_to_height(height, F::ZERO);
                mat
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_batch(padded).to_row_major_matrix())
        }
    };

    let (root, prover_data) = info_span!("commit_matrix").in_scope(|| mmcs.commit_matrix(encoded));
    challenger.observe(root.clone());
    (root, prover_data)
}

/// Encodes and commits a folded extension-field polynomial.
///
/// This is used after each non-final WHIR folding round. The layout mirrors
/// the base-field path, but the DFT runs over extension-field values and the
/// commitment is made through an extension MMCS that views extension rows as
/// base-field data for the underlying Merkle tree.
#[allow(clippy::type_complexity)]
pub(crate) fn commit_extension<F, EF, Dft, MT>(
    order: VariableOrder,
    dft: &Dft,
    extension_mmcs: &ExtensionMmcs<F, EF, MT>,
    poly: &Poly<EF>,
    folding: usize,
    inv_rate: usize,
) -> (
    MT::Commitment,
    <MT as Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
{
    let num_variables = poly.num_variables();
    let height = inv_rate * (1 << (num_variables - folding));

    let encoded = match order {
        VariableOrder::Prefix => {
            let padded = info_span!("transpose & pad").in_scope(|| {
                let mut mat =
                    RowMajorMatrixView::new(poly.as_slice(), 1 << (num_variables - folding))
                        .transpose();
                mat.pad_to_height(height, EF::ZERO);
                mat
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_algebra_batch(padded).to_row_major_matrix())
        }
        VariableOrder::Suffix => {
            let padded = info_span!("pad").in_scope(|| {
                let mut mat = RowMajorMatrix::new(poly.as_slice().to_vec(), 1 << folding);
                mat.pad_to_height(height, EF::ZERO);
                mat
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_algebra_batch(padded).to_row_major_matrix())
        }
    };

    info_span!("commit_matrix").in_scope(|| extension_mmcs.commit_matrix(encoded))
}
