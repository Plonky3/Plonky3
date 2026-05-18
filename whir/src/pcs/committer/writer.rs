use alloc::vec::Vec;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::extension::FlatMatrixView;
use p3_multilinear_util::poly::Poly;
use tracing::info_span;

use crate::sumcheck::strategy::VariableOrder;

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

/// Encodes and commits a folded extension-field polynomial with ZK randomness.
///
/// Same layout as [`commit_extension`], but places `zk_randomness` in the
/// coefficient positions after the message instead of zeros
/// (Construction 9.7, step 1 of eprint 2026/391).
///
/// Only supports [`VariableOrder::Suffix`]. Prefix ordering interleaves
/// high-degree coefficients across columns, requiring a different layout
/// for the ZK mask; this is left to a follow-up.
#[allow(clippy::type_complexity)]
pub(crate) fn commit_extension_zk<F, EF, Dft, MT>(
    order: VariableOrder,
    dft: &Dft,
    extension_mmcs: &ExtensionMmcs<F, EF, MT>,
    poly: &Poly<EF>,
    zk_randomness: &[EF],
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
    assert!(
        matches!(order, VariableOrder::Suffix),
        "ZK commitment requires Suffix variable ordering"
    );

    let num_variables = poly.num_variables();
    let width = 1 << folding;
    let height = inv_rate * (1 << (num_variables - folding));
    let total = height * width;

    // Build coefficient vector: (message ∥ randomness ∥ zeros).
    // The randomness occupies coefficients [msg_len .. msg_len + t) before
    // the zero tail, matching the RS ZK encoding (Proposition 3.19).
    let msg_len = poly.as_slice().len();
    assert!(
        msg_len + zk_randomness.len() <= total,
        "message ({msg_len}) + ZK randomness ({}) exceeds codeword capacity ({total})",
        zk_randomness.len(),
    );
    let mut coeffs = Vec::with_capacity(total);
    coeffs.extend_from_slice(poly.as_slice());
    coeffs.extend_from_slice(zk_randomness);
    coeffs.resize(total, EF::ZERO);

    let encoded = {
        let mat = info_span!("build (zk)").in_scope(|| RowMajorMatrix::new(coeffs, width));
        info_span!("dft", height = mat.height(), width = mat.width())
            .in_scope(|| dft.dft_algebra_batch(mat).to_row_major_matrix())
    };

    info_span!("commit_matrix").in_scope(|| extension_mmcs.commit_matrix(encoded))
}
