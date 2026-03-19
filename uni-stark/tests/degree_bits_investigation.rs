use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeField64};
use p3_fri::{
    FriParameters, HidingFriPcs, TwoAdicFriPcs, create_test_fri_params_zk,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Minimal AIR that never reads next row.
struct SquareAir;

impl<F> BaseAir<F> for SquareAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB: AirBuilder> Air<AB> for SquareAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let a = main.current(0).unwrap();
        let b = main.current(1).unwrap();
        builder.assert_eq(a * a, b);
    }
}

fn generate_square_trace<F: PrimeField64>(n: usize) -> RowMajorMatrix<F> {
    let mut values = F::zero_vec(n * 2);
    for i in 0..n {
        let a = F::from_u64((i + 1) as u64);
        values[i * 2] = a;
        values[i * 2 + 1] = a * a;
    }
    RowMajorMatrix::new(values, 2)
}

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;

type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type Config = StarkConfig<Pcs, Challenge, Challenger>;

const SALT_ELEMS: usize = 4;
type ValHidingMmcs = MerkleTreeHidingMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    Hash,
    Compress,
    SmallRng,
    2,
    8,
    SALT_ELEMS,
>;
type ChallengeHidingMmcs = ExtensionMmcs<Val, Challenge, ValHidingMmcs>;
type HidingPcs = HidingFriPcs<Val, Dft, ValHidingMmcs, ChallengeHidingMmcs, SmallRng>;
type HidingConfig = StarkConfig<HidingPcs, Challenge, Challenger>;

fn make_config() -> Config {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    Config::new(pcs, challenger)
}

fn make_hiding_config() -> HidingConfig {
    let mut rng = SmallRng::seed_from_u64(1337);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());

    let val_mmcs = ValHidingMmcs::new(hash, compress, 0, rng);
    let challenge_mmcs = ChallengeHidingMmcs::new(val_mmcs.clone());

    let fri_params = create_test_fri_params_zk(challenge_mmcs);
    let pcs = HidingPcs::new(Dft::default(), val_mmcs, fri_params, 4, SmallRng::seed_from_u64(7));
    let challenger = Challenger::new(perm);
    HidingConfig::new(pcs, challenger)
}

#[test]
fn uni_stark_rejects_malformed_degree_bits_non_zk() {
    let config = make_config();
    let trace = generate_square_trace::<Val>(1 << 3);
    let proof = prove(&config, &SquareAir, trace.clone(), &[]);

    verify(&config, &SquareAir, &proof, &[]).expect("baseline verification should succeed");

    for mutated in [63usize, 64, 255] {
        // Proof<SC> is not Clone; regenerate a valid proof and mutate only degree_bits.
        let mut mutated_proof = prove(&config, &SquareAir, trace.clone(), &[]);
        mutated_proof.degree_bits = mutated;
        let result = verify(&config, &SquareAir, &mutated_proof, &[]);
        assert!(
            result.is_err(),
            "expected clean verifier rejection for degree_bits={mutated}"
        );
    }
}

#[test]
fn uni_stark_rejects_malformed_degree_bits_zk() {
    let config = make_hiding_config();
    let trace = generate_square_trace::<Val>(1 << 3);
    let mut proof = prove(&config, &SquareAir, trace, &[]);

    verify(&config, &SquareAir, &proof, &[]).expect("baseline verification should succeed");

    proof.degree_bits = 0;
    let result = verify(&config, &SquareAir, &proof, &[]);
    assert!(
        result.is_err(),
        "expected clean verifier rejection for degree_bits=0 in ZK mode"
    );
}
