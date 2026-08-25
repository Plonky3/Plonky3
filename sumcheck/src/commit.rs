//! Base-field commitment used by the sumcheck opening protocol.

use p3_challenger::CanObserve;
use p3_commit::{Encoder, Mmcs};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix, RowMajorMatrixView};
use p3_multilinear_util::poly::Poly;
use tracing::info_span;

use crate::strategy::VariableOrder;

/// Encodes and commits the initial base-field polynomial.
///
/// This is the first WHIR commitment. It lays out the polynomial according to
/// the residual variable order, applies the Reed-Solomon expansion with
/// `encoder`, commits the resulting codeword matrix with `mmcs`, and observes
/// the Merkle root in the transcript.
///
/// Prefix order transposes the local folding block so the first folded
/// variables become columns. Suffix order keeps the folding block as the row
/// width. The encoder owns the expansion, so neither branch pads.
pub fn commit_base<F, E, MT, Challenger>(
    order: VariableOrder,
    encoder: &E,
    mmcs: &MT,
    challenger: &mut Challenger,
    poly: &Poly<F>,
    folding: usize,
    starting_log_inv_rate: usize,
) -> (MT::Commitment, MT::ProverData<DenseMatrix<F>>)
where
    F: Field,
    E: Encoder<F>,
    MT: Mmcs<F>,
    Challenger: CanObserve<MT::Commitment>,
{
    let num_variables = poly.num_variables();
    let width = 1 << folding;

    let message = match order {
        VariableOrder::Prefix => info_span!("transpose").in_scope(|| {
            // Transposing the folding blocks turns the first folded variables into columns.
            RowMajorMatrixView::new(poly.as_slice(), 1 << (num_variables - folding)).transpose()
        }),
        // Folding blocks are already contiguous, so the row width alone selects them.
        VariableOrder::Suffix => RowMajorMatrix::new(poly.as_slice().to_vec(), width),
    };

    let encoded = info_span!("encode", height = message.height(), width = message.width())
        .in_scope(|| encoder.encode_batch(message, starting_log_inv_rate));

    let (root, prover_data) = info_span!("commit_matrix").in_scope(|| mmcs.commit_matrix(encoded));
    challenger.observe(root.clone());
    (root, prover_data)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::CanObserve;
    use p3_commit::{Encoder, Mmcs};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::commit_base;
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

    /// A challenger that only has to absorb the root.
    struct RootObserver;

    impl<T> CanObserve<T> for RootObserver {
        fn observe(&mut self, _value: T) {}
    }

    fn mmcs() -> MyMmcs {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0)
    }

    /// `commit_base` must commit `encoder.encode_batch(message)` for the message layout of
    /// the given variable order: the transposed folding blocks in prefix order, the
    /// contiguous folding blocks in suffix order.
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
            order,
            &DoublingEncoder,
            &mmcs,
            &mut RootObserver,
            &poly,
            FOLDING,
            LOG_INV_RATE,
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
