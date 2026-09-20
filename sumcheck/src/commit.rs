//! Base-field commitment used by the sumcheck opening protocol.

use p3_commit::{Encoder, Mmcs};
use p3_field::Field;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use tracing::info_span;

use crate::strategy::VariableOrder;

/// Chunk size for the parallel copy of the suffix-order message, chosen so each
/// rayon task copies enough elements to outweigh the fork-join overhead.
const COPY_CHUNK: usize = 1 << 16;

/// Writes a stacked polynomial into the committed message, in the residual variable order.
///
/// # Layout
///
/// Prefix order transposes the local folding block.
///
/// The first folded variables then become columns.
///
/// Suffix order keeps the folding block as the row width, so the blocks are already
/// contiguous and the row width alone selects them.
///
/// # Panics
///
/// - `message` must hold one cell per evaluation of `poly`.
pub fn write_stacked_message<F: Field>(
    order: VariableOrder,
    poly: &Poly<F>,
    folding: usize,
    message: &mut [F],
) {
    let values = poly.as_slice();
    assert_eq!(
        message.len(),
        values.len(),
        "message must cover the whole stacked hypercube"
    );
    let width = 1 << folding;
    match order {
        VariableOrder::Prefix => info_span!("transpose").in_scope(|| {
            let view = RowMajorMatrixView::new(values, values.len() / width);
            let mut prefix = RowMajorMatrixViewMut::new(message, width);
            view.transpose_into(&mut prefix);
        }),
        VariableOrder::Suffix => message
            .par_chunks_mut(COPY_CHUNK)
            .zip(values.par_chunks(COPY_CHUNK))
            .for_each(|(destination, source)| destination.copy_from_slice(source)),
    }
}

/// Encodes and Merkle-commits the initial base-field message.
///
/// # Overview
///
/// `write_message` receives the leading `2^num_variables` cells of the codeword buffer,
/// zeroed, and lays the committed polynomial out in the residual variable order.
///
/// Every cell it leaves untouched is committed as zero, so a callback that means to commit
/// a non-zero value at a cell must write it: there is no guard against a partial write, and
/// a callback that writes nothing at all commits the zero polynomial without complaint.
///
/// The message is therefore built directly at codeword height, with a zero tail.
///
/// The encoder can then skip the zero coefficients, and reuse this one allocation.
///
/// It is then expanded by the Reed-Solomon encoder, and committed.
///
/// Nothing is absorbed here.
///
/// The caller owns the transcript and absorbs the returned root itself.
pub fn commit_base<F, E, MT>(
    encoder: &E,
    mmcs: &MT,
    num_variables: usize,
    folding: usize,
    starting_log_inv_rate: usize,
    write_message: impl FnOnce(&mut [F]),
) -> (MT::Commitment, MT::ProverData<DenseMatrix<F>>)
where
    F: Field,
    E: Encoder<F>,
    MT: Mmcs<F>,
{
    let width = 1 << folding;
    let message_height = 1 << (num_variables - folding);
    let codeword_height = message_height << starting_log_inv_rate;

    let mut values = F::zero_vec(codeword_height * width);
    write_message(&mut values[..message_height * width]);
    let message = RowMajorMatrix::new(values, width);

    let encoded = info_span!("encode", height = codeword_height, width)
        .in_scope(|| encoder.encode_batch_padded(message, starting_log_inv_rate));

    info_span!("commit_matrix").in_scope(|| mmcs.commit_matrix(encoded))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::{Encoder, Mmcs};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{commit_base, write_stacked_message};
    use crate::strategy::VariableOrder;

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

    /// Doubles every message entry and appends zero rows. Not a DFT, and not linear-code
    /// shaped: it exists only to show that `commit_base` commits exactly what the encoder
    /// returns, applied to the message layout the variable order prescribes.
    #[derive(Clone, Debug)]
    struct DoublingEncoder;

    impl Encoder<F> for DoublingEncoder {
        fn encode_batch(
            &self,
            mut message: RowMajorMatrix<F>,
            log_inv_rate: usize,
        ) -> RowMajorMatrix<F> {
            message.values.iter_mut().for_each(|v| *v = v.double());
            message
                .values
                .resize(message.values.len() << log_inv_rate, F::ZERO);
            message
        }
    }

    fn mmcs() -> MyMmcs {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0)
    }

    /// Commits the encoder's output over the message layout the variable order prescribes.
    ///
    /// Prefix order transposes the folding blocks.
    ///
    /// Suffix order leaves them contiguous.
    fn check_commits_encoder_output(order: VariableOrder, expected_message: RowMajorMatrix<F>) {
        const NUM_VARIABLES: usize = 5;
        const FOLDING: usize = 2;
        const LOG_INV_RATE: usize = 1;

        let poly = Poly::new(
            (0..1 << NUM_VARIABLES)
                .map(F::from_usize)
                .collect::<Vec<_>>(),
        );
        let mmcs = mmcs();

        let (root, _data) = commit_base(
            &DoublingEncoder,
            &mmcs,
            NUM_VARIABLES,
            FOLDING,
            LOG_INV_RATE,
            |message| write_stacked_message(order, &poly, FOLDING, message),
        );

        let expected_codeword = DoublingEncoder.encode_batch(expected_message, LOG_INV_RATE);
        let (expected_root, _) = mmcs.commit_matrix(expected_codeword);
        assert_eq!(root, expected_root);
    }

    #[test]
    fn prefix_commits_the_transposed_message() {
        // The polynomial is viewed as 4 rows of 2^(5-2) = 8 columns, then transposed.
        let expected =
            RowMajorMatrix::new((0..32).map(F::from_usize).collect::<Vec<_>>(), 1 << 3).transpose();
        check_commits_encoder_output(VariableOrder::Prefix, expected);
    }

    #[test]
    fn suffix_commits_the_contiguous_message() {
        // The folding blocks are already contiguous: 8 rows of 2^FOLDING = 4 columns.
        let expected = RowMajorMatrix::new((0..32).map(F::from_usize).collect::<Vec<_>>(), 1 << 2);
        check_commits_encoder_output(VariableOrder::Suffix, expected);
    }
}
