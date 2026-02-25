use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::TwoAdicFriPcs;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// A minimal single-row AIR: enforces `a * a == b` per row, never reads the next row.
struct SquareAir;

impl<F> BaseAir<F> for SquareAir {
    fn width(&self) -> usize {
        2
    }

    fn main_uses_next_row(&self) -> bool {
        false
    }
}

impl<AB: AirBuilder> Air<AB> for SquareAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("Matrix is empty?");
        let a = local[0].clone();
        let b = local[1].clone();
        builder.assert_eq(a.clone() * a, b);
    }
}

fn generate_square_trace<F: PrimeField64>(n: usize) -> RowMajorMatrix<F> {
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
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

fn make_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = p3_fri::FriParameters {
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
    MyConfig::new(pcs, challenger)
}

#[test]
fn test_no_next_row_air() {
    let config = make_config();
    let trace = generate_square_trace::<Val>(1 << 3);
    let proof = prove(&config, &SquareAir, trace, &[]);

    // trace_next should be None
    assert!(
        proof.opened_values.trace_next.is_none(),
        "trace_next should be None for an AIR that does not use the next row"
    );

    verify(&config, &SquareAir, &proof, &[]).expect("verification should succeed");
}

#[test]
fn test_no_next_row_rejects_present_trace_next() {
    let config = make_config();
    let trace = generate_square_trace::<Val>(1 << 3);
    let proof = prove(&config, &SquareAir, trace, &[]);

    // Tamper: set trace_next to Some(zeros) — verifier should reject
    let mut tampered = proof;
    let air_width = <SquareAir as BaseAir<Val>>::width(&SquareAir);
    tampered.opened_values.trace_next = Some(vec![Challenge::ZERO; air_width]);

    let result = verify(&config, &SquareAir, &tampered, &[]);
    assert!(
        result.is_err(),
        "verifier should reject proof with trace_next present when AIR does not use next row"
    );
}
