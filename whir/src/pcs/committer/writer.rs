use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PackedFieldExtension, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
use p3_matrix::extension::FlatMatrixView;
use p3_sumcheck::product_polynomial::PolyView;
use p3_sumcheck::strategy::VariableOrder;
use tracing::info_span;

/// Encodes and commits a folded extension-field polynomial.
///
/// Runs after each non-final WHIR folding round.
/// The layout mirrors the base-field path, with two differences:
/// - the DFT runs over extension-field values;
/// - the extension MMCS views each extension row as base-field data for the Merkle tree.
///
/// `poly` is borrowed as a [`PolyView`] over the live sumcheck buffer.
/// No intermediate scalar copy is materialized.
#[allow(clippy::type_complexity)]
pub(crate) fn commit_extension<F, EF, Dft, MT>(
    order: VariableOrder,
    dft: &Dft,
    extension_mmcs: &ExtensionMmcs<F, EF, MT>,
    poly: PolyView<'_, F, EF>,
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
    let width = 1 << folding;

    let encoded = match order {
        VariableOrder::Prefix => {
            let padded = info_span!("transpose & pad").in_scope(|| {
                // Allocate the zero-padded codeword buffer once.
                // Trailing rows stay zero and become the Reed-Solomon expansion.
                let mut values = EF::zero_vec(height * width);
                let leading = &mut values[..1 << num_variables];
                let src_width = 1 << (num_variables - folding);

                // Transpose the folding blocks straight into the leading rows.
                match poly {
                    // Packed source: fuse the lane unpacking with the transpose; no scalar staging.
                    PolyView::Packed(packed) => {
                        EF::ExtensionPacking::unpack_transpose_into(
                            packed.as_slice(),
                            leading,
                            src_width,
                        );
                    }
                    // Scalar source: reuse the cache-blocked transpose directly.
                    PolyView::Scalar(scalar) => {
                        RowMajorMatrixView::new(scalar.as_slice(), src_width)
                            .transpose_into(&mut RowMajorMatrixViewMut::new(leading, width));
                    }
                }

                RowMajorMatrix::new(values, width)
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_algebra_batch(padded))
        }
        VariableOrder::Suffix => {
            let padded = info_span!("pad").in_scope(|| {
                let mut values = EF::zero_vec(height * width);
                // Unpack the live evaluations straight into the leading rows; no scalar copy.
                poly.unpack_into(&mut values[..poly.num_evals()]);
                RowMajorMatrix::new(values, width)
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.dft_algebra_batch(padded))
        }
    };

    info_span!("commit_matrix").in_scope(|| extension_mmcs.commit_matrix(encoded))
}
