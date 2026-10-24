use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

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

fn config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let val_mmcs = ValMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm.clone()), 0);
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: ChallengeMmcs::new(val_mmcs.clone()),
    };
    MyConfig::new(
        Pcs::new(Dft::default(), val_mmcs, fri_params),
        Challenger::new(perm),
    )
}

// Asserts a^2 = b on transition rows, under one or two transition selectors.
struct SquareAir {
    nested: bool,
}

impl<F> BaseAir<F> for SquareAir {
    fn width(&self) -> usize {
        2
    }
}

impl<AB: AirBuilder> Air<AB> for SquareAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let a = main.current(0).unwrap();
        let b = main.current(1).unwrap();
        let mut outer = builder.when_transition();
        if self.nested {
            outer.when_transition().assert_eq(a * a, b);
        } else {
            outer.assert_eq(a * a, b);
        }
    }
}

fn trace(rows: usize) -> RowMajorMatrix<Val> {
    let mut rng = SmallRng::seed_from_u64(7);
    let values = (0..rows)
        .flat_map(|_| {
            let a: Val = rng.random();
            [a, a.square()]
        })
        .collect();
    RowMajorMatrix::new(values, 2)
}

#[test]
fn one_transition_factor_verifies() {
    // Fixture state: 8 rows, constraint T(X) * (a(X)^2 - b(X)), degree 1 + 14 = 15.
    let config = config();
    let air = SquareAir { nested: false };
    let proof = prove(&config, &air, trace(8), &[]).unwrap();
    verify(&config, &air, &proof, &[]).unwrap();
}

#[test]
fn two_transition_factors_verify() {
    // Fixture state: 8 rows, constraint T(X)^2 * (a(X)^2 - b(X)).
    //
    //     deg C = 2 + 2 * 7 = 16,   deg Q = 16 - 8 = 8
    //     degree multiple 2   ->  1 chunk of 8 coefficients, too small for deg Q = 8
    //     constraint degree 3 ->  2 chunks of 8 coefficients
    //
    // Invariant: an honest proof of a satisfied trace verifies.
    // Mutation: size the quotient by the degree multiple alone.
    //
    // Verification then fails with an out-of-domain evaluation mismatch.
    let config = config();
    let air = SquareAir { nested: true };
    let proof = prove(&config, &air, trace(8), &[]).unwrap();
    verify(&config, &air, &proof, &[]).unwrap();
}
