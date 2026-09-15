//! A binary zerocheck whose closing claim a real commitment discharges.
//!
//! Every step goes through a public surface a caller would use.
//!
//! The scheme's prescribed-point opening, and the zerocheck's own prove and verify.
//!
//! The prover and the verifier run on two independently constructed challengers.
//!
//! One cloned after proving already carries every observation the prover made.
//!
//! Such a challenger can never disagree with a proof that desyncs the transcript.

use p3_binary_field::{BinaryChallenger, BinaryField8, BinaryField128};
use p3_binary_pcs::{BinaryPcs, BinaryPcsConfig, BinaryPcsParams};
use p3_challenger::{CanObserve, HashChallenger};
use p3_commit::MultilinearPcs;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout, SuffixProver, Table};
use p3_sumcheck::univariate_skip::{
    BinaryZerocheck, Conjunction, ZerocheckClaim, ZerocheckError, embed_bits,
};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// The subspace the skip round runs over lives in a byte field.
type A = BinaryField8;

/// The commitment, the challenges and every value the rounds carry live in the field above it.
type F = BinaryField128;

type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<F, u8, MyHash, MyCompress, 2, 32>;
type MyChallenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
type MyPcs = BinaryPcs<MyMmcs>;

/// Variables the skip round binds in one go.
const LOG_SKIP: usize = 6;

/// Total variables, rows and skipped together.
const LOG_HEIGHT: usize = 10;

/// Operands the conjunction constraint reads.
const ARITY: usize = 3;

/// Variables the stacked commitment covers.
///
/// The three columns are stacked into one polynomial, and three pads to four.
///
///     10 variables per column + 2 to index four columns = 12
const LOG_STACKED: usize = LOG_HEIGHT + 2;

/// The three packed operands, in the order the constraint reads them.
type Operands = [Vec<u8>; ARITY];

/// What the chain ends on: the opened operand values, and the claim they must answer.
#[derive(Debug)]
struct Closing {
    /// One opened value per operand, in column order.
    opened: Vec<F>,
    /// The claim the zerocheck left for the commitment to discharge.
    claim: ZerocheckClaim<F>,
}

impl Closing {
    /// Whether the commitment's answer is the one the zerocheck asked for.
    ///
    /// The claim owns the recombination.
    ///
    /// This test therefore cannot batch the openings in an order the proof did not use.
    fn is_answered(&self) -> bool {
        self.claim.is_answered_by(&self.opened)
    }
}

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

/// The commitment scheme the operands are held in.
///
/// A modest security level keeps the round trip quick, and the shape is what is under test.
fn pcs() -> MyPcs {
    BinaryPcs::new(
        BinaryPcsConfig::try_new(
            LOG_STACKED,
            BinaryPcsParams {
                log_inv_rate: 2,
                pow_bits: 0,
                security_level: 40,
            },
        )
        .unwrap(),
        mmcs(),
    )
}

/// One batch naming every operand column, opened at one prescribed point.
fn protocol() -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(LOG_HEIGHT, ARITY),
        vec![OpeningBatch::new((0..ARITY).collect(), vec![])],
    )])
}

/// A bit witness satisfying the conjunction, packed row by row.
fn witness(seed: u64, num_rows: usize, row_bytes: usize) -> Operands {
    let mut rng = SmallRng::seed_from_u64(seed);

    // Two free operands, and the third pinned to their bitwise conjunction.
    let a = (0..num_rows * row_bytes)
        .map(|_| rng.random::<u8>())
        .collect::<Vec<_>>();
    let b = (0..num_rows * row_bytes)
        .map(|_| rng.random::<u8>())
        .collect::<Vec<_>>();
    let c = a.iter().zip(&b).map(|(&x, &y)| x & y).collect::<Vec<_>>();
    [a, b, c]
}

/// The committed table, one column per operand.
///
/// Each column is that operand's bit multilinear, one field element per cell.
fn table(operands: &Operands) -> Table<F> {
    let evals = operands
        .iter()
        .flat_map(|packed| embed_bits::<F>(packed).into_evals())
        .collect::<Vec<_>>();

    // One matrix row per column of the logical table, each holding that column's hypercube.
    Table::new(RowMajorMatrix::new(evals, 1 << LOG_HEIGHT))
}

/// Run the whole chain, committing one witness and proving over another.
///
///     commit(committed) -> zerocheck(proven) -> claim at a point -> open there
///
/// The two coincide in honest use.
///
/// Passing them apart checks that the commitment, not the witness, discharges the claim.
fn run(
    committed: &Operands,
    proven: &Operands,
    check: &BinaryZerocheck<A, Conjunction>,
) -> Result<Closing, ZerocheckError> {
    let scheme = pcs();
    let protocol = protocol();

    // Prover: commit, prove against the transcript the commitment is now in, then open.
    let mut prover_challenger = challenger();
    let (commitment, prover_data) = scheme
        .commit(
            SuffixProver::<F, F>::new_witness(vec![table(committed)], 0),
            &mut prover_challenger,
        )
        .unwrap();

    let packed = [
        proven[0].as_slice(),
        proven[1].as_slice(),
        proven[2].as_slice(),
    ];
    let (proof, prover_claim) = check.prove::<F, _>(&packed, LOG_HEIGHT, &mut prover_challenger);
    let opening = scheme
        .open_at(
            prover_data,
            &protocol,
            core::slice::from_ref(&prover_claim.point),
            &mut prover_challenger,
        )
        .unwrap();

    // Verifier: replay from a fresh transcript, holding only the commitment and the proofs.
    let mut verifier_challenger = challenger();
    verifier_challenger.observe(commitment.clone());

    let claim = check.verify::<F, _>(&proof, LOG_HEIGHT, &mut verifier_challenger)?;
    assert_eq!(claim.point, prover_claim.point);

    let opened = scheme
        .verify_at(
            &commitment,
            &opening,
            &protocol,
            core::slice::from_ref(&claim.point),
            &mut verifier_challenger,
        )
        .unwrap();

    Ok(Closing {
        opened: opened[0].current().to_vec(),
        claim,
    })
}

#[test]
fn the_commitment_discharges_the_claim_the_zerocheck_leaves() {
    // Fixture state: 10 variables, 6 skipped, 3 operands committed as one 3-column table.
    //
    // Invariant: the opened values, batched under the challenge, are the claimed value.
    //
    // Nothing on the verifying side is read from the witness.
    let check = BinaryZerocheck::<A, _>::new(LOG_SKIP, Conjunction, 0).unwrap();
    let num_rows = 1 << (LOG_HEIGHT - LOG_SKIP);
    let operands = witness(0xC0117, num_rows, check.round().row_bytes());

    let closing = run(&operands, &operands, &check).unwrap();

    assert_eq!(closing.opened.len(), ARITY);
    assert!(closing.is_answered());
}

#[test]
fn grinding_carries_through_the_committed_chain() {
    // Fixture state: 4 bits of difficulty, cheap enough for a test.
    //
    // The grinding witness travels in the proof, so the whole chain has to survive it.
    let check = BinaryZerocheck::<A, _>::new(LOG_SKIP, Conjunction, 4).unwrap();
    let num_rows = 1 << (LOG_HEIGHT - LOG_SKIP);
    let operands = witness(0x6D1, num_rows, check.round().row_bytes());

    let closing = run(&operands, &operands, &check).unwrap();

    assert!(closing.is_answered());
}

#[test]
fn a_broken_constraint_is_refused_before_any_opening() {
    // Mutation: flip one bit of the claimed conjunction.
    //
    //     a & b  ->  (a & b) ^ 1  in cell 0
    //
    // The residual rounds then no longer end on the constraint of the blends they carry.
    let check = BinaryZerocheck::<A, _>::new(LOG_SKIP, Conjunction, 0).unwrap();
    let num_rows = 1 << (LOG_HEIGHT - LOG_SKIP);
    let mut operands = witness(0xBAD, num_rows, check.round().row_bytes());
    operands[2][0] ^= 1;

    assert_eq!(
        run(&operands, &operands, &check).unwrap_err(),
        ZerocheckError::ResidualClaimMismatch
    );
}

#[test]
fn a_claim_about_another_witness_does_not_open_against_this_commitment() {
    // Mutation: commit one satisfying witness and prove over a different satisfying one.
    //
    //     commit(seed 1)  ->  opens the first witness at the point
    //     prove (seed 2)  ->  claims the second witness's value there
    //
    // Both witnesses satisfy the constraint, so the zerocheck itself has nothing to catch.
    //
    // Only the commitment separates them, which is what closing the chain buys.
    let check = BinaryZerocheck::<A, _>::new(LOG_SKIP, Conjunction, 0).unwrap();
    let num_rows = 1 << (LOG_HEIGHT - LOG_SKIP);
    let row_bytes = check.round().row_bytes();
    let committed = witness(0x1, num_rows, row_bytes);
    let proven = witness(0x2, num_rows, row_bytes);

    let closing = run(&committed, &proven, &check).unwrap();

    assert!(!closing.is_answered());
}
