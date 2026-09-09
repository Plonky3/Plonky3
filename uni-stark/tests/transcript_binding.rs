//! End-to-end coverage of what the uni-STARK transcript binds.
//!
//! Each test perturbs one absorbed value, or one number the seed carries.
//! Verification then has to reject the result.
//!
//! Three absorbed values are covered by sibling files instead.
//!
//! - The public values, in `fib_air`.
//! - The preprocessed commitment, in `mul_fib_pair`.
//! - The out-of-domain grinding witness, in `grinding`.

use std::borrow::Cow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::{Proof, StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::{SmallRng, StdRng};

/// A single-row AIR asserting `a * a == b`, declaring `P` periodic columns.
///
/// The constraint never reads a periodic value.
/// So `P` moves the AIR's shape and leaves its satisfying witnesses where they were.
struct SquareAir<const P: usize>;

impl<F: PrimeCharacteristicRing + Sync, const P: usize> BaseAir<F> for SquareAir<P> {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn num_periodic_columns(&self) -> usize {
        P
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]>
    where
        F: Clone,
    {
        // Period 1 is the shortest legal column: a subdomain of size 2^0.
        Cow::Owned(vec![vec![F::ONE]; P])
    }
}

impl<AB: AirBuilder, const P: usize> Air<AB> for SquareAir<P> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let a = main.current(0).unwrap();
        let b = main.current(1).unwrap();
        builder.assert_eq(a * a, b);
    }
}

/// A trace of `n` rows whose second column squares the first.
fn square_trace<F: PrimeField64>(n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());
    let mut values = F::zero_vec(n * 2);
    for i in 0..n {
        let a = F::from_u64((i + 1) as u64);
        values[i * 2] = a;
        values[i * 2 + 1] = a * a;
    }
    RowMajorMatrix::new(values, 2)
}

type Val = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

type ZkByteHash = Keccak256Hash;
type ZkU64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type ZkFieldHash = SerializingHasher<ZkU64Hash>;
type ZkCompress = CompressionFunctionFromHasher<ZkU64Hash, 2, 4>;
type ZkValHidingMmcs = MerkleTreeHidingMmcs<
    [Val; p3_keccak::VECTOR_LEN],
    [u64; p3_keccak::VECTOR_LEN],
    ZkFieldHash,
    ZkCompress,
    StdRng,
    2,
    4,
    4,
>;
type ZkChallenger = SerializingChallenger32<Val, HashChallenger<u8, ZkByteHash, 32>>;
type ZkChallengeHidingMmcs = ExtensionMmcs<Val, Challenge, ZkValHidingMmcs>;
type ZkHidingPcs = HidingFriPcs<Val, Dft, ZkValHidingMmcs, ZkChallengeHidingMmcs, StdRng>;
type ZkConfig = StarkConfig<ZkHidingPcs, Challenge, ZkChallenger>;

/// Log-height of every trace proven here.
const LOG_DEGREE: usize = 3;

fn make_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    MyConfig::new(
        Pcs::new(Dft::default(), val_mmcs, fri_params),
        Challenger::new(perm),
    )
}

fn make_zk_config() -> ZkConfig {
    let byte_hash = ZkByteHash {};
    let u64_hash = ZkU64Hash::new(KeccakF {});
    let field_hash = ZkFieldHash::new(u64_hash);
    let compress = ZkCompress::new(u64_hash);
    let val_mmcs = ZkValHidingMmcs::new(field_hash, compress, 0, StdRng::seed_from_u64(1));
    let challenge_mmcs = ZkChallengeHidingMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = ZkHidingPcs::new(
        Dft::default(),
        val_mmcs,
        fri_params,
        4,
        StdRng::seed_from_u64(2),
    );
    ZkConfig::new(pcs, ZkChallenger::from_hasher(vec![], byte_hash))
}

/// One honest proof of the square AIR, with no periodic columns.
fn honest_proof(config: &MyConfig) -> Proof<MyConfig> {
    prove(
        config,
        &SquareAir::<0>,
        square_trace::<Val>(1 << LOG_DEGREE),
        &[],
    )
}

#[test]
fn an_honest_proof_verifies() {
    // Baseline: every mutation below starts from this proof.
    let config = make_config();
    let proof = honest_proof(&config);

    verify(&config, &SquareAir::<0>, &proof, &[]).expect("an untouched proof verifies");
}

#[test]
fn a_tampered_trace_commitment_is_rejected() {
    // Fixture state: an honest proof, whose trace commitment is then replaced.
    let config = make_config();
    let mut proof = honest_proof(&config);

    // Mutation: the quotient commitment stands in for the trace one.
    //
    // Both are commitments of the same type, so nothing rejects this on shape alone.
    proof.commitments.trace = proof.commitments.quotient_chunks.clone();

    verify(&config, &SquareAir::<0>, &proof, &[])
        .expect_err("a swapped trace commitment must be rejected");
}

#[test]
fn a_tampered_quotient_commitment_is_rejected() {
    // Fixture state: an honest proof, whose quotient commitment is then replaced.
    let config = make_config();
    let mut proof = honest_proof(&config);

    // Mutation: the trace commitment stands in for the quotient one.
    proof.commitments.quotient_chunks = proof.commitments.trace.clone();

    verify(&config, &SquareAir::<0>, &proof, &[])
        .expect_err("a swapped quotient commitment must be rejected");
}

#[test]
fn a_tampered_randomization_commitment_is_rejected() {
    // Fixture state: a zero-knowledge proof, which carries a randomization commitment.
    let config = make_zk_config();
    let trace = square_trace::<Val>(1 << 5);
    let mut proof = prove(&config, &SquareAir::<0>, trace, &[]);

    verify(&config, &SquareAir::<0>, &proof, &[]).expect("an untouched proof verifies");

    // Mutation: the trace commitment stands in for the randomization one.
    proof.commitments.random = Some(proof.commitments.trace.clone());

    verify(&config, &SquareAir::<0>, &proof, &[])
        .expect_err("a swapped randomization commitment must be rejected");
}

#[test]
fn a_tampered_degree_bits_is_rejected() {
    // The trace height is not absorbed as a message; it seeds the transcript.
    //
    //     honest:   log-height 3
    //     tampered: log-height 4  -> a different seed, and a different domain
    let config = make_config();
    let mut proof = honest_proof(&config);

    proof.degree_bits = LOG_DEGREE + 1;

    verify(&config, &SquareAir::<0>, &proof, &[])
        .expect_err("a tampered trace height must be rejected");
}

#[test]
fn an_air_of_a_different_shape_does_not_verify() {
    // The two AIRs assert the same constraint over the same trace.
    //
    //     prover:   SquareAir<0>   no periodic columns
    //     verifier: SquareAir<1>   one periodic column, never read
    //
    // The proof is structurally valid for both, so nothing rejects it on shape.
    // Only the seed separates them, which is what makes the rejection happen.
    let config = make_config();
    let proof = honest_proof(&config);

    verify(&config, &SquareAir::<1>, &proof, &[])
        .expect_err("an AIR of another shape must not verify this proof");
}

#[test]
fn an_air_of_a_different_shape_proves_and_verifies_on_its_own() {
    // Both AIRs are honest AIRs; the shape alone is what separates their transcripts.
    let config = make_config();
    let trace = square_trace::<Val>(1 << LOG_DEGREE);
    let proof = prove(&config, &SquareAir::<1>, trace, &[]);

    verify(&config, &SquareAir::<1>, &proof, &[]).expect("the periodic AIR verifies its own proof");
}
