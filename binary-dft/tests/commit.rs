//! Phase 2 exit criterion: the multilinear commit path runs over a binary tower field.

use p3_binary_dft::{AdditiveRsEncoder, LchNtt, NaiveAdditiveNtt};
use p3_binary_field::BinaryField128;
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

    // The default folding depth, and a depth of zero, which leaves the width-1 message
    // `p3-binary-pcs` commits. The reference encoding costs `O(n^2)` per column, so the
    // width-1 shape is pinned at a height of its own rather than at `NUM_VARIABLES`.
    //
    // Neither height reaches a staging tile: at width 1 a contiguous tile already holds `2^11`
    // rows, so the whole transform runs inside one. What a gathered run does to the transform
    // is pinned against the reference oracle in `p3-binary-dft`'s own tests, where the cut
    // points are set directly rather than bought with a taller matrix.
    for (num_variables, folding) in [(NUM_VARIABLES, FOLDING), (5, 0)] {
        let values = &values[..1 << num_variables];

        // Prefix order transposes the folding blocks, and the reference transform does the
        // encoding.
        let message = RowMajorMatrixView::new(values, 1 << (num_variables - folding)).transpose();
        let codeword = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default()
            .encode_batch(message, LOG_INV_RATE);
        assert_eq!(
            codeword.height(),
            1 << (num_variables - folding + LOG_INV_RATE)
        );
        let (expected_root, _) = mmcs.commit_matrix(codeword);

        // The portable tower transform, and the one the PCS commits through by default, which
        // routes to `PolyBasisNtt` only on a target that has a carryless multiply.
        let poly = Poly::new(values.to_vec());
        let tower = commit_base(
            VariableOrder::Prefix,
            &AdditiveRsEncoder::<F, LchNtt<F>>::default(),
            &mmcs,
            &poly,
            folding,
            LOG_INV_RATE,
        );
        let default = commit_base(
            VariableOrder::Prefix,
            &AdditiveRsEncoder::<F>::default(),
            &mmcs,
            &poly,
            folding,
            LOG_INV_RATE,
        );
        for (name, root) in [("tower", tower.0), ("default", default.0)] {
            assert_eq!(root, expected_root, "{name} folding={folding}");
        }
    }
}

/// `PrefixProver::commit` runs end to end over a binary field: the same witness committed
/// through `LchNtt` and through the reference transform yields the same root.
#[test]
fn prefix_prover_commits_over_a_binary_field() {
    let mmcs = mmcs();

    let (_layout, root_fast, _data) = PrefixProver::<F, F>::commit(
        &AdditiveRsEncoder::<F, LchNtt<F>>::default(),
        &mmcs,
        PrefixProver::<F, F>::new_witness(vec![table()], FOLDING),
        FOLDING,
        LOG_INV_RATE,
    );

    let (_layout_ref, root_ref, _data_ref) = PrefixProver::<F, F>::commit(
        &AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default(),
        &mmcs,
        PrefixProver::<F, F>::new_witness(vec![table()], FOLDING),
        FOLDING,
        LOG_INV_RATE,
    );

    assert_eq!(root_fast, root_ref);
}

/// The padded production path preserves both layouts and their independently encoded roots.
#[test]
#[ignore = "20-way naive-vs-fast binary NTT sweep; run from heavy CI"]
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
                    &Poly::new(values.clone()),
                    folding,
                    rate,
                );
                assert_eq!(root, expected_root);
            }
        }
    }
}
