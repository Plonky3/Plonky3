use alloc::vec::Vec;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
use p3_matrix::extension::FlatMatrixView;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::{PolyMaybePacked, PolyMaybePackedView};
use p3_sumcheck::strategy::VariableOrder;
use tracing::info_span;

/// A vector of `len` zeros, written by every worker.
///
/// A recycled allocation is not zero already, and one thread clearing it would serialise the commit.
fn zeroed<EF: Field>(len: usize) -> Vec<EF> {
    let mut values = Vec::with_capacity(len);
    let chunk = min_task_len(len, size_of::<EF>());
    values.spare_capacity_mut()[..len]
        .par_chunks_mut(chunk)
        .for_each(|slots| {
            for slot in slots {
                slot.write(EF::ZERO);
            }
        });
    // SAFETY: the sweep above initialised every one of the first `len` slots.
    unsafe { values.set_len(len) };
    values
}

use crate::domain::WhirDomain;

/// Encodes and commits a folded extension-field polynomial.
///
/// Runs after each non-final WHIR folding round.
/// The layout mirrors the base-field path, with two differences:
/// - the DFT runs over extension-field values;
/// - the extension MMCS views each extension row as base-field data for the Merkle tree.
///
/// `poly` is borrowed as a [`PolyMaybePackedView`] over the live sumcheck buffer.
/// No intermediate scalar copy is materialized.
#[allow(clippy::type_complexity)]
pub(crate) fn commit_extension<F, EF, Dft, MT>(
    order: VariableOrder,
    dft: &Dft,
    extension_mmcs: &ExtensionMmcs<F, EF, MT>,
    poly: PolyMaybePackedView<'_, F, EF>,
    folding: usize,
    log_inv_rate: usize,
) -> (
    MT::Commitment,
    <MT as Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
)
where
    F: Field,
    EF: ExtensionField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
{
    let num_variables = poly.num_variables();
    let height = (1 << log_inv_rate) * (1 << (num_variables - folding));
    let width = 1 << folding;

    let encoded = match order {
        VariableOrder::Prefix => {
            let padded = info_span!("transpose & pad").in_scope(|| {
                // Allocate the zero-padded codeword buffer once.
                // Trailing rows stay zero and become the Reed-Solomon expansion.
                let mut values = zeroed(height * width);
                let leading = &mut values[..1 << num_variables];
                let src_width = 1 << (num_variables - folding);

                // Transpose the folding blocks straight into the leading rows.
                match poly {
                    // Packed source: fuse the lane unpacking with the transpose; no scalar staging.
                    PolyMaybePacked::Packed(packed) => {
                        EF::ExtensionPacking::unpack_transpose_into(
                            packed.as_slice(),
                            leading,
                            src_width,
                        );
                    }
                    // Scalar source: reuse the cache-blocked transpose directly.
                    PolyMaybePacked::Scalar(scalar) => {
                        RowMajorMatrixView::new(scalar.as_slice(), src_width)
                            .transpose_packable_into(&mut RowMajorMatrixViewMut::new(
                                leading, width,
                            ));
                    }
                }

                RowMajorMatrix::new(values, width)
            });
            info_span!("dft", height = padded.height(), width = padded.width())
                .in_scope(|| dft.encode_extension_batch_padded(padded, log_inv_rate))
        }
        VariableOrder::Suffix => {
            // The folding blocks are already rows, so the message is the live evaluations.
            let unpacked;
            let message = match poly {
                PolyMaybePacked::Scalar(scalar) => scalar.into_slice(),
                PolyMaybePacked::Packed(packed) => {
                    unpacked = info_span!("unpack").in_scope(|| {
                        // Each worker unpacks its own run of packed elements.
                        let lanes = F::Packing::WIDTH;
                        let chunk = min_task_len(packed.num_evals(), size_of::<EF>() * lanes);
                        let mut values = EF::zero_vec(packed.num_evals() * lanes);
                        values
                            .par_chunks_mut(chunk * lanes)
                            .zip(packed.as_slice().par_chunks(chunk))
                            .for_each(|(out, packed)| {
                                for (out, value) in out
                                    .iter_mut()
                                    .zip(EF::ExtensionPacking::to_ext_iter(packed.iter().copied()))
                                {
                                    *out = value;
                                }
                            });
                        values
                    });
                    &unpacked
                }
            };
            info_span!("dft", height, width).in_scope(|| {
                dft.encode_extension_batch_borrowed(
                    RowMajorMatrixView::new(message, width),
                    log_inv_rate,
                )
            })
        }
    };

    info_span!("commit_matrix").in_scope(|| extension_mmcs.commit_matrix(encoded))
}
