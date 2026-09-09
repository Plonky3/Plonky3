//! Phase 2 exit criterion: the multilinear commit path runs over a binary tower field.

use p3_binary_dft::{AdditiveRsEncoder, LchNtt, NaiveAdditiveNtt};
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_challenger::HashChallenger;
use p3_commit::{Encoder, Mmcs};
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::commit::commit_base;
use p3_sumcheck::layout::{Layout, PrefixProver, Table};
use p3_sumcheck::strategy::VariableOrder;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField128;

type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<F, u8, MyHash, MyCompress, 2, 32>;
type MyChallenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

const NUM_VARIABLES: usize = 8;
const FOLDING: usize = 2;
const LOG_INV_RATE: usize = 2;

const fn mmcs() -> MyMmcs {
    MyMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        0,
    )
}

const fn challenger() -> MyChallenger {
    MyChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// One fixed random table, rebuilt from the seed so the two commits below see the same data.
fn table() -> Table<F> {
    let mut rng = SmallRng::seed_from_u64(2);
    Table::rand(&mut rng, 1, NUM_VARIABLES)
}

/// `commit_base` over `BinaryField128` reproduces the Merkle root of a matrix encoded by hand
/// with the reference transform. This pins the message layout as well as the transform.
#[test]
fn commit_base_matches_hand_encoding() {
    let mut rng = SmallRng::seed_from_u64(1);
    let values: Vec<F> = (0..1 << NUM_VARIABLES).map(|_| rng.random()).collect();
    let mmcs = mmcs();

    let (root, _data) = commit_base(
        VariableOrder::Prefix,
        &AdditiveRsEncoder::<F, LchNtt<F>>::default(),
        &mmcs,
        &mut challenger(),
        &Poly::new(values.clone()),
        FOLDING,
        LOG_INV_RATE,
    );

    // Prefix order transposes the folding blocks; the reference transform does the encoding.
    let message = RowMajorMatrixView::new(&values, 1 << (NUM_VARIABLES - FOLDING)).transpose();
    let codeword =
        AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default().encode_batch(message, LOG_INV_RATE);
    assert_eq!(
        codeword.height(),
        1 << (NUM_VARIABLES - FOLDING + LOG_INV_RATE)
    );
    let (expected_root, _) = mmcs.commit_matrix(codeword);

    assert_eq!(root, expected_root);
}

/// `PrefixProver::commit` runs end to end over a binary field: the same witness committed
/// through `LchNtt` and through the reference transform yields the same root.
#[test]
fn prefix_prover_commits_over_a_binary_field() {
    let mmcs = mmcs();

    let (_layout, root_fast, _data) = PrefixProver::<F, F>::commit(
        &AdditiveRsEncoder::<F, LchNtt<F>>::default(),
        &mmcs,
        &mut challenger(),
        PrefixProver::<F, F>::new_witness(vec![table()], FOLDING),
        FOLDING,
        LOG_INV_RATE,
    );

    let (_layout_ref, root_ref, _data_ref) = PrefixProver::<F, F>::commit(
        &AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default(),
        &mmcs,
        &mut challenger(),
        PrefixProver::<F, F>::new_witness(vec![table()], FOLDING),
        FOLDING,
        LOG_INV_RATE,
    );

    assert_eq!(root_fast, root_ref);
}

/// The padded production path preserves both layouts and their independently encoded roots.
#[test]
fn polynomial_commit_matches_naive_for_both_orders() {
    use p3_matrix::dense::RowMajorMatrix;

    let mut rng = SmallRng::seed_from_u64(19);
    let values: Vec<F> = (0..1 << 8).map(|_| rng.random()).collect();
    let mmcs = mmcs();
    for folding in [0, 2, 4, 6] {
        for rate in [1, 2, 3] {
            let mut unfolded_root = None;
            for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
                let reference_root = || {
                    let message = match order {
                        VariableOrder::Prefix => {
                            RowMajorMatrixView::new(&values, 1 << (8 - folding)).transpose()
                        }
                        VariableOrder::Suffix => RowMajorMatrix::new(values.clone(), 1 << folding),
                    };
                    let expected = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default()
                        .encode_batch(message, rate);
                    mmcs.commit_matrix(expected).0
                };
                // Without folding, both orders have the same single-column message.
                // Share its expensive reference encoding, but check both production paths.
                let expected_root = if folding == 0 {
                    unfolded_root.get_or_insert_with(reference_root).clone()
                } else {
                    reference_root()
                };
                let (root, _) = commit_base(
                    order,
                    &AdditiveRsEncoder::<F>::default(),
                    &mmcs,
                    &mut challenger(),
                    &Poly::new(values.clone()),
                    folding,
                    rate,
                );
                assert_eq!(root, expected_root);
            }
        }
    }
}
