//! End-to-end multilinear AIR SNARK over WHIR: commit, zerocheck, open, verify.

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::zerocheck::ZerocheckError;
use p3_multi_stark::{VerificationError, prove, verify};
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::OpeningBatch;
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use p3_whir::{
    DomainSeparator, FoldingFactor, ProtocolParameters, SecurityAssumption,
    VerifierError as WhirVerifierError, WhirConfig, WhirProver,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

type MyDft = Radix2DFTSmallBatch<F>;
type L = PrefixProver<F, EF>;
type TestPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

/// First-round folding factor; also the per-table padding floor.
const FOLDING: usize = 2;

/// A WHIR-backed multilinear AIR configuration over BabyBear.
struct WhirConfigForTest {
    /// The WHIR commitment scheme, fixed to one stacked-table arity.
    pcs: TestPcs,
}

impl MultiStarkConfig for WhirConfigForTest {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn min_num_variables(&self) -> usize {
        // The witness pads each table to the first-round folding factor.
        FOLDING
    }

    fn build_witness(&self, columns: Vec<Poly<F>>) -> Witness<F> {
        // Version one commits a single table: the whole trace.
        L::new_witness(vec![Table::new(columns)], FOLDING)
    }
}

/// Fixed permutation so prover and verifier transcripts match exactly.
fn perm() -> Perm {
    let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
    Perm::new_from_rng_128(&mut rng)
}

/// Per-round log-inverse rates for a stacked polynomial.
fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let folding_schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .expect("valid folding schedule");
    let num_rounds = folding_schedule.len() - 1;
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = 1;
    for &folding in folding_schedule.iter().take(num_rounds) {
        rate += folding - 1;
        rates.push(rate);
    }
    rates
}

/// Build a configuration sized for a trace shape.
///
/// The committed polynomial stacks every trace column.
/// The WHIR configuration must match that stacked arity.
fn config_for(log_height: usize, width: usize) -> WhirConfigForTest {
    let stacked_num_variables = log_height + log2_ceil_usize(width);
    let folding_factor = FoldingFactor::Constant(FOLDING);

    let mmcs = MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0);
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        round_log_inv_rates: default_round_log_inv_rates(stacked_num_variables, &folding_factor),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let whir_config = WhirConfig::new(stacked_num_variables, params).unwrap();
    WhirConfigForTest {
        pcs: TestPcs::new(whir_config, MyDft::default(), mmcs),
    }
}

/// A challenger seeded with the same domain separator on both proof and verify sides.
fn challenger(config: &WhirConfigForTest) -> MyChallenger {
    let mut challenger = MyChallenger::new(perm());
    let mut ds = DomainSeparator::new(vec![]);
    config.pcs.add_domain_separator::<8>(&mut ds);
    ds.observe_domain_separator(&mut challenger);
    challenger
}

const NUM_COLS: usize = 2;

/// Fibonacci AIR.
///
/// - The first row equals the first two public values.
/// - Each transition advances the Fibonacci recurrence.
/// - The final row equals the output public value.
struct FibAir;

struct FibRow<T> {
    left: T,
    right: T,
}

impl<T> Borrow<FibRow<T>> for [T] {
    fn borrow(&self) -> &FibRow<T> {
        // Safety: two fields of type T in declaration order match the layout of [T; 2].
        debug_assert_eq!(self.len(), NUM_COLS);
        let ptr = self.as_ptr() as *const FibRow<T>;
        unsafe { &*ptr }
    }
}

impl<X> BaseAir<X> for FibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for FibAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a, b, x) = (pis[0], pis[1], pis[2]);

        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();

        let mut first = builder.when_first_row();
        first.assert_eq(local.left, a);
        first.assert_eq(local.right, b);

        let mut trans = builder.when_transition();
        trans.assert_eq(local.right, next.left);
        trans.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}

/// Build a Fibonacci trace seeded with zero and one.
fn fib_trace(n: usize) -> RowMajorMatrix<F> {
    let mut left = F::ZERO;
    let mut right = F::ONE;
    let mut values = Vec::with_capacity(NUM_COLS * n);
    for _ in 0..n {
        values.push(left);
        values.push(right);
        let next_left = right;
        let next_right = left + right;
        left = next_left;
        right = next_right;
    }
    RowMajorMatrix::new(values, NUM_COLS)
}

/// Public inputs for the first row and final output.
fn fib_public_values(n: usize) -> [F; 3] {
    let trace = fib_trace(n);
    let last = trace.values[(n - 1) * NUM_COLS + 1];
    [F::ZERO, F::ONE, last]
}

#[test]
fn prove_verify_fibonacci_roundtrips() {
    // A satisfying trace must prove and verify end to end through WHIR.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let config = config_for(log2_strict_usize(n), NUM_COLS);

    let proof = prove(&config, &FibAir, &trace, &pis, 0, &mut challenger(&config));

    verify(&config, &FibAir, &proof, &pis, 0, &mut challenger(&config))
        .expect("honest Fibonacci proof must verify");
}

#[test]
fn verify_rejects_tampered_opening() {
    // Fixture state: the proof carries commitment-bound trace openings.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let config = config_for(log2_strict_usize(n), NUM_COLS);

    let mut proof = prove(&config, &FibAir, &trace, &pis, 0, &mut challenger(&config));

    // Mutation: shift the first claimed current-row value by one field element.
    let batch = &proof.opening.evals[0];
    let mut current = batch.current().to_vec();
    current[0] += EF::ONE;
    proof.opening.evals[0] = OpeningBatch::new(current, batch.next().to_vec());

    // Expected rejection: the tampered value is no longer bound to the commitment.
    // Why: the verifier samples query positions from the absorbed value.
    //   the round verifies as one pruned multiproof
    //   -> failure reports a batched placeholder position, not a per-query index.
    let err = verify(&config, &FibAir, &proof, &pis, 0, &mut challenger(&config)).unwrap_err();
    match err {
        VerificationError::Opening(WhirVerifierError::MerkleProofInvalid { position, reason }) => {
            assert_eq!(position, 0);
            assert_eq!(reason, "Base field Merkle multiproof verification failed");
        }
        other => panic!("expected a Merkle opening rejection, got {other:?}"),
    }
}

#[test]
fn verify_rejects_violated_constraint() {
    // Fixture state: a satisfying trace obeys every Fibonacci transition.
    let n = 256;
    let mut trace = fib_trace(n);
    // Mutation: shift one row value used by the transition constraints.
    trace.values[2 * NUM_COLS] += F::ONE;
    let pis = fib_public_values(n);
    let config = config_for(log2_strict_usize(n), NUM_COLS);

    let proof = prove(&config, &FibAir, &trace, &pis, 0, &mut challenger(&config));

    // Expected rejection: the zerocheck closes on a nonzero constraint value.
    let err = verify(&config, &FibAir, &proof, &pis, 0, &mut challenger(&config)).unwrap_err();
    assert!(
        matches!(
            err,
            VerificationError::Zerocheck(ZerocheckError::FinalSumMismatch)
        ),
        "expected zerocheck final-sum mismatch, got {err:?}"
    );
}

#[test]
fn verify_rejects_tampered_public_values() {
    // Fixture state: the public output equals the final trace row.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let config = config_for(log2_strict_usize(n), NUM_COLS);

    let proof = prove(&config, &FibAir, &trace, &pis, 0, &mut challenger(&config));

    // Mutation: shift the claimed output by one field element.
    let mut wrong = pis;
    wrong[2] += F::ONE;
    // Expected rejection: the wrong public value desyncs the transcript.
    // Why: the verifier derives a different opening point than the prover used.
    //   the round verifies as one pruned multiproof
    //   -> failure reports a batched placeholder position, not a per-query index.
    let err = verify(
        &config,
        &FibAir,
        &proof,
        &wrong,
        0,
        &mut challenger(&config),
    )
    .unwrap_err();
    match err {
        VerificationError::Opening(WhirVerifierError::MerkleProofInvalid { position, reason }) => {
            assert_eq!(position, 0);
            assert_eq!(reason, "Base field Merkle multiproof verification failed");
        }
        other => panic!("expected a Merkle opening rejection, got {other:?}"),
    }
}
