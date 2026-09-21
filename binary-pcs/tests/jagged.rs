//! Jagged stacking over the ring-switching bit commitment.
//!
//! The columns are bits of unequal height, and the commitment binds exactly as many as are live.

use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_binary_pcs::{BinaryPcsConfig, BinaryPcsParams, BooleanTracePcs};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::jagged::{
    BoundJaggedLayout, CellBudget, ColumnSource, JaggedLayout, JaggedOpeningError, JaggedPoint,
    JaggedWitness, TraceSource,
};
use p3_sumcheck::layout::Table;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

type EF = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
type MyChallenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;
type MyPcs = BooleanTracePcs<EF, MyMmcs, MyMmcs>;
type Commitment = <MyPcs as MultilinearPcs<EF, MyChallenger>>::Commitment;
type ProverData = <MyPcs as MultilinearPcs<EF, MyChallenger>>::ProverData;

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

// Equality weight of one Boolean index against a point, written straight from the definition.
fn equality_weight(point: &Point<EF>, index: usize) -> EF {
    let width = point.num_variables();
    (0..width)
        .map(|position| {
            let bit = (index >> (width - 1 - position)) & 1 == 1;
            if bit {
                point[position]
            } else {
                EF::ONE - point[position]
            }
        })
        .product()
}

// Value of the virtual jagged table, built from the heights alone.
fn jagged_evaluation(heights: &[usize], cells: &[EF], point: &JaggedPoint<EF>) -> EF {
    let mut total = EF::ZERO;
    let mut start = 0;
    for (column, &height) in heights.iter().enumerate() {
        let column_weight = equality_weight(point.column(), column);
        for row in 0..height {
            total += column_weight * equality_weight(point.row(), row) * cells[start + row];
        }
        start += height;
    }
    total
}

// Eight bit columns of unequal height, delivered as packed words.
//
//     heights   2048, 1536, 1200, 1024, 900, 800, 500, 184
//     live      8192 bits, which is already a power of two
fn trace() -> (Vec<usize>, Vec<Vec<u64>>) {
    let heights = vec![2048usize, 1536, 1200, 1024, 900, 800, 500, 184];
    let words = heights
        .iter()
        .enumerate()
        .map(|(column, &height)| {
            (0..height.div_ceil(64))
                .map(|word| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((column + word + 1) as u64))
                .collect()
        })
        .collect();
    (heights, words)
}

fn commit(cells: &[EF], challenger: &mut MyChallenger) -> (MyPcs, Commitment, ProverData) {
    // One element of the committed alphabet absorbs one hundred and twenty-eight bits.
    let num_variables = p3_util::log2_strict_usize(cells.len());
    let config = BinaryPcsConfig::try_new::<EF, EF>(
        num_variables - 7,
        BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        },
    )
    .unwrap();
    let pcs = MyPcs::new(config, mmcs(), mmcs(), num_variables).unwrap();

    let table = Table::new(RowMajorMatrix::new(cells.to_vec(), cells.len()));
    let (commitment, data) = pcs.commit(vec![table], challenger).unwrap();
    (pcs, commitment, data)
}

#[test]
fn bit_columns_of_unequal_height_pay_for_no_dead_cell() {
    let (heights, words) = trace();
    let layout = JaggedLayout::new(11, &heights).unwrap();

    // The measurement criterion four asks for, on the same heights.
    //
    //     stacked   2048 + 2048 + 2048 + 1024 + 1024 + 1024 + 512 + 256 = 9984, rounded to 16384
    //     jagged    8192, which is the live area itself
    let stacked = CellBudget::stacked(&heights);
    assert_eq!(stacked.live(), 8192);
    assert_eq!(stacked.provisioned(), 16384);
    assert_eq!(CellBudget::of(&layout).provisioned(), 8192);
    assert_eq!(CellBudget::of(&layout).dead(), 0);

    // The producer is bit-sliced, so the widening pass is named in the type and charged.
    let sources = heights
        .iter()
        .enumerate()
        .map(|(column, &height)| ColumnSource::Bits {
            words: &words[column],
            height,
        })
        .collect::<Vec<_>>();
    let (witness, report) = JaggedWitness::read(&layout, TraceSource::Columns(&sources)).unwrap();
    assert_eq!(report.live(), 8192);
    assert_eq!(report.converted(), 8192);
    assert_eq!(report.envelope(), 0);

    let mut prover = challenger();
    let (pcs, commitment, data) = commit(&witness, &mut prover);
    let bound = BoundJaggedLayout::new::<EF, _>(&layout, &mut prover);
    let point = bound.sample_point::<EF, EF, _>(&mut prover);
    let value = jagged_evaluation(&heights, &witness, &point);
    let opening = bound
        .open(&pcs, data, &witness, &point, value, &mut prover)
        .expect("an honest bit trace opens");

    let mut verifier = challenger();
    pcs.observe_commitment(&commitment, &mut verifier);
    let bound = BoundJaggedLayout::new::<EF, _>(&layout, &mut verifier);
    let replayed = bound.sample_point::<EF, EF, _>(&mut verifier);
    assert_eq!(replayed, point);
    bound
        .verify(&pcs, &commitment, &opening, &replayed, value, &mut verifier)
        .expect("the ring-switched bit commitment authenticates the sparse claim");
}

#[test]
fn a_bit_trace_that_was_not_committed_is_refused() {
    let (heights, words) = trace();
    let layout = JaggedLayout::new(11, &heights).unwrap();
    let sources = heights
        .iter()
        .enumerate()
        .map(|(column, &height)| ColumnSource::Bits {
            words: &words[column],
            height,
        })
        .collect::<Vec<_>>();
    let (committed, _) = JaggedWitness::read(&layout, TraceSource::Columns(&sources)).unwrap();

    // Mutation: one live bit of the vector the reduction speaks about, which was never committed.
    let mut forged = committed.to_vec();
    forged[4000] += EF::ONE;

    let mut prover = challenger();
    let (pcs, commitment, data) = commit(&committed, &mut prover);
    let bound = BoundJaggedLayout::new::<EF, _>(&layout, &mut prover);
    let point = bound.sample_point::<EF, EF, _>(&mut prover);
    let value = jagged_evaluation(&heights, &forged, &point);
    assert_ne!(value, jagged_evaluation(&heights, &committed, &point));
    let opening = bound
        .open(&pcs, data, &forged, &point, value, &mut prover)
        .expect("the reduction proves the forged statement on its own");

    let mut verifier = challenger();
    pcs.observe_commitment(&commitment, &mut verifier);
    let bound = BoundJaggedLayout::new::<EF, _>(&layout, &mut verifier);
    let replayed = bound.sample_point::<EF, EF, _>(&mut verifier);
    // A rejection anywhere else would mean the opening never reached the comparison under test.
    let error = bound
        .verify(&pcs, &commitment, &opening, &replayed, value, &mut verifier)
        .unwrap_err();
    assert!(
        matches!(error, JaggedOpeningError::DenseMismatch),
        "the committed bits must be what refuse the claim, not {error:?}"
    );
}
