//! Commit/open/verify round trips across the configuration sweep, and the negative tests that
//! check the verifier rejects a malformed or mismatched proof for a specific typed reason
//! rather than only failing. Everything here goes through `p3-binary-pcs`'s public
//! `MultilinearPcs`/`PrescribedPointPcs` surface, exactly as an external caller would.
//!
//! Every genuine proof is produced with `SuffixProver` and `folding = 0`, the only
//! configuration this crate's commit phase accepts, and every prover/verifier pair uses two
//! independently constructed challengers seeded identically rather than one challenger cloned
//! after proving: a challenger cloned from the prover's own transcript already carries every
//! observation the prover made, correct or not, so it can never disagree with a proof that
//! desyncs the transcript from what the prover actually produced.

use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_binary_pcs::{BinaryPcs, BinaryPcsConfig, BinaryPcsError, BinaryPcsParams, BinaryPcsProof};
use p3_challenger::{FieldChallenger, HashChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Layout, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<F, u8, MyHash, MyCompress, 2, 32>;
type MyChallenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
type MyPcs = BinaryPcs<MyMmcs, SuffixProver<F, F>>;

/// Default shape shared by every negative test: enough intermediate rounds
/// (`num_fold_rounds - 1 == 7`) to truncate or permute, and `pow_bits > 0` so the grinding
/// rejection has something to catch.
const NUM_VARIABLES: usize = 8;
const LOG_INV_RATE: usize = 2;
const POW_BITS: usize = 4;
const SECURITY_LEVEL: usize = 40;

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

const fn params(log_inv_rate: usize, pow_bits: usize, security_level: usize) -> BinaryPcsParams {
    BinaryPcsParams {
        log_inv_rate,
        pow_bits,
        security_level,
    }
}

/// Commits a random single-column table, opens column 0 at a transcript-sampled point with a
/// fresh prover challenger, and returns everything a negative test needs: the PCS instance
/// (reusable for a fresh `verify` call), the commitment, the genuine proof, and the opening
/// protocol that produced it.
fn run_lifecycle(
    num_variables: usize,
    log_inv_rate: usize,
    pow_bits: usize,
    security_level: usize,
    seed: u64,
) -> (
    MyPcs,
    <MyMmcs as Mmcs<F>>::Commitment,
    BinaryPcsProof<MyMmcs>,
    OpeningProtocol,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let table = Table::rand(&mut rng, 1, num_variables);
    let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )]);

    let config = BinaryPcsConfig::try_new(
        num_variables,
        0,
        params(log_inv_rate, pow_bits, security_level),
    )
    .unwrap();
    let pcs = BinaryPcs::new(config, mmcs());

    let mut prover_challenger = challenger();
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
    let proof = pcs.open(prover_data, protocol.clone(), &mut prover_challenger);

    (pcs, commitment, proof, protocol)
}

/// Runs one commit/open/verify round trip at `num_variables` and `log_inv_rate`, with the
/// prover and the verifier on two independently constructed challengers seeded identically.
fn assert_round_trips(num_variables: usize, log_inv_rate: usize, seed: u64) {
    let (pcs, commitment, proof, protocol) =
        run_lifecycle(num_variables, log_inv_rate, POW_BITS, SECURITY_LEVEL, seed);

    let mut verifier_challenger = challenger();
    pcs.verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_or_else(|err| {
            panic!("num_variables={num_variables} log_inv_rate={log_inv_rate}: {err:?}")
        });
}

/// The configuration sweep, `SuffixProver` / `folding = 0` only: `PrefixProver` does not stay
/// in lockstep with the codeword fold (covered in `prover.rs`'s unit tests), and `folding > 0`
/// is rejected by `BinaryPcsConfig::try_new`.
#[test]
fn small_configurations_round_trip() {
    for &num_variables in &[6usize, 8, 10, 12] {
        for &log_inv_rate in &[1usize, 2, 3] {
            let seed = (num_variables as u64) << 8 | log_inv_rate as u64;
            assert_round_trips(num_variables, log_inv_rate, seed);
        }
    }
}

/// The degenerate edge below `small_configurations_round_trip`'s sweep: `num_variables = 1`
/// gives `num_fold_rounds() == 1`, so `fold_rounds`'s `rounds` vector is empty and every
/// `num_fold_rounds - 1` derivation in the verifier bottoms out at zero rather than
/// underflowing.
#[test]
fn nv_1_round_trip() {
    for &log_inv_rate in &[1usize, 2] {
        let seed = 1u64 << 8 | log_inv_rate as u64;
        assert_round_trips(1, log_inv_rate, seed);
    }
}

/// This phase's exit criterion: a commit/open/verify round trip at `num_variables = 16`. The
/// unoptimized fold and Merkle paths make a `2^16`-row codeword too slow for `cargo test`'s
/// default debug profile, so this runs explicitly rather than in every default invocation:
/// `cargo test --release -- --ignored` locally, or the `p3-binary-pcs` job in `ci-heavy.yml`.
#[test]
#[ignore = "commit/open/verify at num_variables = 16; run via `cargo test --release -- --ignored`, or through ci-heavy.yml's p3-binary-pcs job"]
fn large_configuration_2_16_round_trips() {
    for &log_inv_rate in &[1usize, 2, 3] {
        let seed = 16u64 << 8 | log_inv_rate as u64;
        assert_round_trips(16, log_inv_rate, seed);
    }
}

/// A tampered `final_codeword` symbol is rejected: with `log_inv_rate = 2` the genuine final
/// codeword has 4 symbols, all equal to the sumcheck's final value, so flipping one breaks the
/// uniformity `verify_opening` checks before it ever reaches the query phase.
///
/// The tampered index is 1, not 0: index 0 also serves as `final_value` in the final check's
/// product clause, so tampering it would be caught by that clause alone and this test would
/// still pass with the uniformity check deleted. Tampering index 1 leaves `final_value`
/// correct, so only the uniformity check can catch it.
#[test]
fn tampered_final_codeword_symbol_is_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 1);

    let symbol = &mut proof.final_codeword.as_mut_slice()[1];
    *symbol += F::ONE;

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::FinalCheck),
        "expected FinalCheck, got {err:?}"
    );
}

/// A tampered `opened_values` entry in an intermediate round breaks that round's Merkle
/// multiproof, which is checked before the fold-consistency chain that reads the row's content.
#[test]
fn tampered_round_opened_value_is_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 2);

    assert!(!proof.rounds.is_empty());
    proof.rounds[0].opened_values[0][0] += F::ONE;

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::MerkleFailed { round: 1, .. }),
        "expected MerkleFailed at round 1, got {err:?}"
    );
}

/// A truncated `rounds` vector is rejected by a structural check — `proof.rounds.len()` against
/// `config.num_fold_rounds() - 1` — that runs before any transcript operation. `verify_at` is
/// used rather than `verify` because `verify` unconditionally observes the commitment as its
/// first step regardless of the proof's validity; `verify_at` leaves that to the caller, so a
/// challenger that never called it is the correct "untouched" baseline to compare against.
#[test]
fn truncated_rounds_vector_is_rejected_without_touching_the_challenger() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 3);

    let expected_rounds = proof.rounds.len();
    assert!(expected_rounds > 0);
    proof.rounds.truncate(expected_rounds - 1);

    let points = [Point::<F>::rand(
        &mut SmallRng::seed_from_u64(4),
        NUM_VARIABLES,
    )];

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify_at(
            &commitment,
            &proof,
            &protocol,
            &points,
            &mut verifier_challenger,
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            BinaryPcsError::RoundCountMismatch { expected, actual }
                if expected == expected_rounds && actual == expected_rounds - 1
        ),
        "expected RoundCountMismatch, got {err:?}"
    );

    // The rejection above must not have touched `verifier_challenger` at all: sampling from it
    // now must agree with sampling from a challenger that never saw `verify_at` in the first
    // place, and disagree only if some transcript operation ran before the structural check.
    let actual: F = verifier_challenger.sample_algebra_element();
    let expected: F = challenger().sample_algebra_element();
    assert_eq!(
        actual, expected,
        "a malformed proof must not become a transcript oracle"
    );
}

/// A proof verified against a different, equally valid `OpeningProtocol` of the same table
/// shape — opening column 1 of a two-column table instead of column 0 — must not verify: the
/// evaluations the proof actually carries are claims about the wrong column.
#[test]
fn a_proof_checked_against_a_different_protocol_is_rejected() {
    // A two-column table costs one selector bit for the extra column, so its per-column arity
    // must be one less than the committed witness's `num_variables` for the two to match.
    let column_arity = NUM_VARIABLES - 1;
    let mut rng = SmallRng::seed_from_u64(5);
    let table = Table::rand(&mut rng, 2, column_arity);
    let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

    let opens_column = |col: usize| {
        OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(column_arity, 2),
            vec![OpeningBatch::new(vec![col], Vec::new())],
        )])
    };
    let protocol_a = opens_column(0);
    let protocol_b = opens_column(1);

    let config = BinaryPcsConfig::try_new(
        NUM_VARIABLES,
        0,
        params(LOG_INV_RATE, POW_BITS, SECURITY_LEVEL),
    )
    .unwrap();
    let pcs: MyPcs = BinaryPcs::new(config, mmcs());

    let mut prover_challenger = challenger();
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
    let proof = pcs.open(prover_data, protocol_a, &mut prover_challenger);

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol_b)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::FinalCheck),
        "expected FinalCheck, got {err:?}"
    );
}

/// A proof verified against the commitment of a different polynomial must not verify: the
/// commitment the verifier observes drives the batching challenge and the query positions, so a
/// swapped commitment desyncs both from what the proof actually carries.
#[test]
fn a_proof_checked_against_a_different_commitment_is_rejected() {
    let (pcs, _commitment, proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 6);

    let mut rng = SmallRng::seed_from_u64(7);
    let other_table = Table::rand(&mut rng, 1, NUM_VARIABLES);
    let other_witness = SuffixProver::<F, F>::new_witness(vec![other_table], 0);
    let mut other_challenger = challenger();
    let (other_commitment, _other_prover_data) = pcs.commit(other_witness, &mut other_challenger);

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(
            &other_commitment,
            &proof,
            &mut verifier_challenger,
            protocol,
        )
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::FinalCheck),
        "expected FinalCheck, got {err:?}"
    );
}

/// `pow_bits > 0` with a corrupted `pow_witness` is rejected by the grinding check. The
/// witness is corrupted by perturbing the genuine one, never by grinding a second time: under
/// `--features parallel`, `GrindingChallenger::grind`'s search can legitimately return a
/// different valid witness on a second call, which would make a re-grinding test
/// non-deterministic for a reason unrelated to what it is checking.
#[test]
fn corrupted_pow_witness_is_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 8);

    proof.pow_witness += F::ONE;

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::InvalidPowWitness),
        "expected InvalidPowWitness, got {err:?}"
    );
}

/// A proof carrying PoW witnesses is rejected outright: every fold round replays with a
/// freshly built, always-empty `pow_witnesses` vector (see `BinaryPcs::verify_opening`), so
/// nothing in the proof's own transcript reads or binds the ones this test appends. Without
/// this guard such a proof would still verify, so a third party could mutate those bytes and
/// keep a valid proof.
#[test]
fn nonempty_sumcheck_pow_witnesses_are_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 10);

    proof.sumcheck.pow_witnesses.push(F::ONE);

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::NonEmptyPowWitnesses { actual: 1 }),
        "expected NonEmptyPowWitnesses, got {err:?}"
    );
}

/// An honest proof whose intermediate round entries are permuted must not verify.
///
/// Every intermediate round's commitment is observed into the transcript between two
/// sumcheck-round replays (`verify_opening` interleaves them), so permuting `proof.rounds`
/// changes which commitment is observed at which point in the sequence and desyncs every fold
/// challenge sampled afterwards. That surfaces at the final check — the sumcheck's claimed sum
/// no longer matches the alpha-batched weight polynomial evaluated at the (now wrong) fold
/// point — before the query phase, where `FoldMismatch` lives, is ever reached.
///
/// `FoldMismatch` itself guards a different failure mode: a prover who commits genuinely
/// inconsistent codewords from round to round while still producing sumcheck messages and a
/// final codeword that satisfy the final check on their own. That is a property of what gets
/// committed, not of the order proof fields are read back in, so no post-hoc permutation of an
/// otherwise honest proof reaches it: swapping whole `RoundProof` entries (this test) desyncs
/// the transcript and lands on `FinalCheck`; swapping only the opened values and multiproof
/// between two rounds while leaving commitments in place instead breaks that round's own Merkle
/// check, landing on `MerkleFailed`.
#[test]
fn permuted_round_commitments_are_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 9);

    assert!(proof.rounds.len() >= 2);
    proof.rounds.swap(0, 1);

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::FinalCheck),
        "expected FinalCheck, got {err:?}"
    );
}

/// Commits a random single-column table against a **zero-claim** `OpeningProtocol` —
/// `TableSpec::new(shape, Vec::new())`, legal on the public API — and returns everything a test
/// needs to replay or mutate the proof.
///
/// With no claim recorded, `Verifier::constraint`'s alpha-batched weight has no terms, so
/// `claimed_sum` and its evaluation at the fold point both collapse to zero; the final check's
/// product clause, `claimed_sum == w(r) * final_value`, then reads `0 == 0` regardless of what
/// `final_value` is. The only thing standing between a proximity-only commitment and an
/// arbitrary uniform final codeword in that branch is the fold-consistency chain
/// `verify_query_paths` walks.
fn zero_claim_lifecycle(
    num_variables: usize,
    seed: u64,
) -> (
    MyPcs,
    <MyMmcs as Mmcs<F>>::Commitment,
    BinaryPcsProof<MyMmcs>,
    OpeningProtocol,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let table = Table::rand(&mut rng, 1, num_variables);
    let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        Vec::new(),
    )]);

    let config = BinaryPcsConfig::try_new(
        num_variables,
        0,
        params(LOG_INV_RATE, POW_BITS, SECURITY_LEVEL),
    )
    .unwrap();
    let pcs = BinaryPcs::new(config, mmcs());

    let mut prover_challenger = challenger();
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
    let proof = pcs.open(prover_data, protocol.clone(), &mut prover_challenger);

    (pcs, commitment, proof, protocol)
}

/// A single tampered final-codeword symbol is caught by the final check's uniformity clause
/// even with a zero-claim protocol, independent of the (here vacuous) product clause — the
/// zero-claim configuration alone is not what is under test.
///
/// Shifting every symbol by the same constant instead keeps the codeword uniform, so both the
/// product clause and the uniformity check pass; `FoldMismatch` is the only remaining check
/// that ties the final codeword to the rounds committed before it, and it is what catches this.
#[test]
fn a_zero_claim_proof_with_a_uniformly_shifted_final_codeword_is_rejected_by_the_fold_chain() {
    let (pcs, commitment, proof, protocol) = zero_claim_lifecycle(NUM_VARIABLES, 11);

    let mut honest_challenger = challenger();
    pcs.verify(
        &commitment,
        &proof,
        &mut honest_challenger,
        protocol.clone(),
    )
    .unwrap();

    let mut single_symbol = proof.clone();
    single_symbol.final_codeword.as_mut_slice()[1] += F::ONE;
    let mut single_symbol_challenger = challenger();
    let err = pcs
        .verify(
            &commitment,
            &single_symbol,
            &mut single_symbol_challenger,
            protocol.clone(),
        )
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::FinalCheck),
        "expected FinalCheck, got {err:?}"
    );

    let mut uniform_shift = proof;
    for v in uniform_shift.final_codeword.as_mut_slice() {
        *v += F::ONE;
    }
    let mut uniform_shift_challenger = challenger();
    let err = pcs
        .verify(
            &commitment,
            &uniform_shift,
            &mut uniform_shift_challenger,
            protocol,
        )
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::FoldMismatch { round, query: 0 } if round == NUM_VARIABLES),
        "expected FoldMismatch at round {NUM_VARIABLES} query 0, got {err:?}"
    );
}

/// The different-commitment test desyncs the whole transcript, so `FinalCheck` fires long
/// before the base Merkle check ever runs — it proves transcript binding, not commitment
/// binding. Tampering an opened value directly is what exercises `verify_multi_batch` against
/// the base commitment itself.
#[test]
fn tampered_base_opened_value_is_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 12);

    proof.base_opened_values[0][0] += F::ONE;

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::MerkleFailed { round: 0, .. }),
        "expected MerkleFailed at round 0, got {err:?}"
    );
}

/// A `base_opened_values` entry short of what the sampled query count demands is rejected by
/// the structural row-count check.
#[test]
fn a_short_base_opened_values_is_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 13);

    proof.base_opened_values.pop();

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::OpeningCountMismatch { round: 0, .. }),
        "expected OpeningCountMismatch at round 0, got {err:?}"
    );
}

/// An opened base row wider than the width-1 codeword every round commits is rejected before
/// the fold chain ever reads it.
#[test]
fn a_wide_base_opened_row_is_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 14);

    proof.base_opened_values[0].push(F::ONE);

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(err, BinaryPcsError::RowWidthMismatch { round: 0, .. }),
        "expected RowWidthMismatch at round 0, got {err:?}"
    );
}

/// A final codeword of the wrong length is rejected before any transcript operation the proof
/// could otherwise ride along with.
#[test]
fn a_wrong_length_final_codeword_is_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 15);

    proof.final_codeword = Poly::new(vec![F::ZERO; 8]);

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(
            err,
            BinaryPcsError::FinalCodewordLengthMismatch {
                expected: 4,
                actual: 8
            }
        ),
        "expected FinalCodewordLengthMismatch, got {err:?}"
    );
}

/// Fewer evaluation batches than the protocol schedules is rejected before any claim is
/// recorded against the transcript.
#[test]
fn a_short_evals_vector_is_rejected() {
    let (pcs, commitment, mut proof, protocol) =
        run_lifecycle(NUM_VARIABLES, LOG_INV_RATE, POW_BITS, SECURITY_LEVEL, 16);

    proof.evals.pop();

    let mut verifier_challenger = challenger();
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, protocol)
        .unwrap_err();
    assert!(
        matches!(
            err,
            BinaryPcsError::OpeningBatchCountMismatch {
                expected: 1,
                actual: 0
            }
        ),
        "expected OpeningBatchCountMismatch, got {err:?}"
    );
}
