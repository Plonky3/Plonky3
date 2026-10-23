use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_fri::TwoAdicFriPcs;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, VerificationError, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// One constraint that adds two selector-filtered terms: `is_first_row * a + is_transition * b`.
struct FirstPlusTransitionAir;

impl<F> BaseAir<F> for FirstPlusTransitionAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB: AirBuilder> Air<AB> for FirstPlusTransitionAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let a = main.current(0).unwrap();
        let b = main.current(1).unwrap();
        builder.assert_zero(builder.is_first_row() * a + builder.is_transition() * b);
    }
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

fn make_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = p3_fri::FriParameters {
        log_blowup: 2,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(Dft::default(), val_mmcs, fri_params);
    MyConfig::new(pcs, Challenger::new(perm))
}

/// Columns `(a, b)` on row 0 of an 8-row trace, zero elsewhere.
fn row_zero_trace(a: Val, b: Val) -> RowMajorMatrix<Val> {
    let mut values = Val::zero_vec(8 * 2);
    values[0] = a;
    values[1] = b;
    RowMajorMatrix::new(values, 2)
}

#[test]
fn debug_check_accepts_a_trace_the_stark_rejects() {
    // Fixture state: 8 rows, (a, b) = (1, -1) on row 0, zero elsewhere.
    //
    //     0/1 selectors on row 0:     1 * 1 + 1 * (-1)               = 0
    //     STARK selectors on row 0:   8 * 1 + (1 - omega^7) * (-1)   = 7 + omega^7 != 0
    //
    // The debug check inside `prove` passes.
    //
    // Yet the constraint polynomial does not vanish on row 0.
    let config = make_config();
    let trace = row_zero_trace(Val::ONE, Val::NEG_ONE);
    let proof = prove(&config, &FirstPlusTransitionAir, trace, &[]).unwrap();

    // Invariant: the verifier rejects at the out-of-domain identity.
    let err = verify(&config, &FirstPlusTransitionAir, &proof, &[]).unwrap_err();
    assert!(
        matches!(err, VerificationError::OodEvaluationMismatch { index: None }),
        "unexpected error: {err:?}"
    );
}

/// Columns `(a, b)` on row 0 that satisfy the STARK's selector values but not the 0/1 ones.
fn stark_only_trace() -> RowMajorMatrix<Val> {
    // Fixture state: 8 rows, (a, b) = ((1 - omega^7) / 8, -1) on row 0, zero elsewhere.
    //
    //     STARK selectors on row 0:   8 a + (1 - omega^7) * (-1)     = 0
    //     0/1 selectors on row 0:     a - 1                          != 0
    let omega = Val::two_adic_generator(3);
    let a = (Val::ONE - omega.exp_u64(7)) * Val::from_u64(8).inverse();
    row_zero_trace(a, Val::NEG_ONE)
}

#[test]
#[should_panic(expected = "constraints not satisfied on row 0: failed constraints = [#0]")]
fn debug_check_rejects_a_trace_the_stark_accepts() {
    p3_air::check_constraints(&FirstPlusTransitionAir, &stark_only_trace(), &[]);
}

// Release builds skip the debug check inside `prove`, so the proof is produced and verifies.
#[cfg(not(debug_assertions))]
#[test]
fn stark_accepts_a_trace_the_debug_check_rejects() {
    let config = make_config();
    let proof = prove(&config, &FirstPlusTransitionAir, stark_only_trace(), &[]).unwrap();
    verify(&config, &FirstPlusTransitionAir, &proof, &[]).unwrap();
}
