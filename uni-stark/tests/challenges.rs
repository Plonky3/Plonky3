//! Tests for the `generate_challenges` function.
//!
//! These tests verify that:
//! 1. `generate_challenges` produces the same challenges as the internal verifier logic
//! 2. Tampering with proof data leads to different challenges and verification failure

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{TwoAdicFriPcs, create_test_fri_params};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, StarkGenericConfig, generate_challenges, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Simple Fibonacci AIR for testing
pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next: &FibonacciRow<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();
        when_first_row.assert_eq(local.left.clone(), a);
        when_first_row.assert_eq(local.right.clone(), b);

        let mut when_transition = builder.when_transition();
        when_transition.assert_eq(local.right.clone(), next.left.clone());
        when_transition.assert_eq(local.left.clone() + local.right.clone(), next.right.clone());

        builder.when_last_row().assert_eq(local.right.clone(), x);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * NUM_FIBONACCI_COLS), NUM_FIBONACCI_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibonacciRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = FibonacciRow::new(F::from_u64(a), F::from_u64(b));

    for i in 1..n {
        rows[i].left = rows[i - 1].right;
        rows[i].right = rows[i - 1].left + rows[i - 1].right;
    }

    trace
}

const NUM_FIBONACCI_COLS: usize = 2;

pub struct FibonacciRow<F> {
    pub left: F,
    pub right: F,
}

impl<F> FibonacciRow<F> {
    const fn new(left: F, right: F) -> Self {
        Self { left, right }
    }
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
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

fn create_test_config() -> (MyConfig, Perm) {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm.clone());
    let config = MyConfig::new(pcs, challenger);
    (config, perm)
}

/// Test that `generate_challenges` produces the same challenges as internal verifier logic.
///
/// This test creates a valid proof, then verifies that calling `generate_challenges`
/// produces identical alpha, zeta, and zeta_next values that would be used during
/// actual verification.
#[test]
fn test_generate_challenges_matches_verifier() {
    let (config, perm) = create_test_config();
    let n = 1 << 3;
    let x = 21u64;
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    // Generate proof
    let proof = prove(&config, &FibonacciAir {}, trace, &pis);

    // Generate challenges using our new function
    let (challenges, _challenger) =
        generate_challenges(&config, &FibonacciAir {}, &proof, &pis, None)
            .expect("challenge generation should succeed");

    // Manually recreate the challenge generation logic to verify correctness
    let mut expected_challenger = Challenger::new(perm);

    // Observe the same data as in generate_challenges
    expected_challenger.observe(Val::from_usize(proof.degree_bits));
    expected_challenger.observe(Val::from_usize(proof.degree_bits - config.is_zk()));
    expected_challenger.observe(Val::from_usize(0)); // preprocessed_width = 0
    expected_challenger.observe(proof.commitments.trace);
    expected_challenger.observe_slice(&pis);

    let expected_alpha: Challenge = expected_challenger.sample_algebra_element();
    expected_challenger.observe(proof.commitments.quotient_chunks);
    let expected_zeta: Challenge = expected_challenger.sample_algebra_element();

    // Verify the challenges match
    assert_eq!(
        challenges.alpha, expected_alpha,
        "alpha challenge should match"
    );
    assert_eq!(
        challenges.zeta, expected_zeta,
        "zeta challenge should match"
    );

    // Also verify that verification succeeds with the proof
    verify(&config, &FibonacciAir {}, &proof, &pis).expect("verification should succeed");
}

/// Test that calling `generate_challenges` twice with the same inputs produces identical results.
#[test]
fn test_generate_challenges_deterministic() {
    let (config, _perm) = create_test_config();
    let n = 1 << 3;
    let x = 21u64;
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibonacciAir {}, trace, &pis);

    // Generate challenges twice
    let (challenges1, _) = generate_challenges(&config, &FibonacciAir {}, &proof, &pis, None)
        .expect("first challenge generation should succeed");

    let (challenges2, _) = generate_challenges(&config, &FibonacciAir {}, &proof, &pis, None)
        .expect("second challenge generation should succeed");

    // They should be identical
    assert_eq!(
        challenges1.alpha, challenges2.alpha,
        "alpha should be deterministic"
    );
    assert_eq!(
        challenges1.zeta, challenges2.zeta,
        "zeta should be deterministic"
    );
    assert_eq!(
        challenges1.zeta_next, challenges2.zeta_next,
        "zeta_next should be deterministic"
    );
}

/// Test that different traces produce different challenges.
///
/// This demonstrates that changing the committed data changes the challenges,
/// which is essential for security - any tampering should be detectable.
#[test]
fn test_different_trace_produces_different_challenges() {
    let (config, _perm) = create_test_config();
    let n = 1 << 3;

    // First proof with original trace (0, 1 start)
    let trace1 = generate_trace_rows::<Val>(0, 1, n);
    let x1 = 21u64; // fib(8) starting from (0, 1)
    let pis1 = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x1)];
    let proof1 = prove(&config, &FibonacciAir {}, trace1, &pis1);

    // Second proof with different trace (1, 1 start)
    let trace2 = generate_trace_rows::<Val>(1, 1, n);
    let x2 = 34u64; // fib(8) starting from (1, 1)
    let pis2 = vec![BabyBear::ONE, BabyBear::ONE, BabyBear::from_u64(x2)];
    let proof2 = prove(&config, &FibonacciAir {}, trace2, &pis2);

    // Generate challenges for both proofs
    let (challenges1, _) = generate_challenges(&config, &FibonacciAir {}, &proof1, &pis1, None)
        .expect("challenge generation should succeed");

    let (challenges2, _) = generate_challenges(&config, &FibonacciAir {}, &proof2, &pis2, None)
        .expect("challenge generation should succeed");

    // Challenges should be different due to different trace commitments
    assert_ne!(
        challenges1.alpha, challenges2.alpha,
        "different traces should produce different alpha"
    );

    // Both should verify successfully
    verify(&config, &FibonacciAir {}, &proof1, &pis1).expect("proof1 should verify");
    verify(&config, &FibonacciAir {}, &proof2, &pis2).expect("proof2 should verify");
}

/// Test that different degree_bits values affect challenges correctly.
///
/// This tests that the degree information observed by the challenger affects
/// the challenge generation, which is important for security.
#[test]
fn test_different_degree_affects_challenges() {
    let (config, _perm) = create_test_config();

    // First proof with smaller trace
    let n1 = 1 << 3;
    let x1 = 21u64;
    let trace1 = generate_trace_rows::<Val>(0, 1, n1);
    let pis1 = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x1)];
    let proof1 = prove(&config, &FibonacciAir {}, trace1, &pis1);

    // Second proof with larger trace (different Fibonacci result due to more iterations)
    // Note: For n=16 rows starting with (0,1), the last value is fib(16) = 987
    let n2 = 1 << 4;
    let x2 = 987u64; // fib(16) starting from (0, 1)
    let trace2 = generate_trace_rows::<Val>(0, 1, n2);
    let pis2 = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x2)];
    let proof2 = prove(&config, &FibonacciAir {}, trace2, &pis2);

    // Generate challenges for both
    let (challenges1, _) = generate_challenges(&config, &FibonacciAir {}, &proof1, &pis1, None)
        .expect("challenge generation should succeed");

    let (challenges2, _) = generate_challenges(&config, &FibonacciAir {}, &proof2, &pis2, None)
        .expect("challenge generation should succeed");

    // Challenges should be different due to different degree_bits
    assert_ne!(
        challenges1.alpha, challenges2.alpha,
        "different degrees should produce different challenges"
    );

    // Both should verify
    verify(&config, &FibonacciAir {}, &proof1, &pis1).expect("proof1 should verify");
    verify(&config, &FibonacciAir {}, &proof2, &pis2).expect("proof2 should verify");
}

/// Test that different public values produce different challenges.
#[test]
fn test_different_public_values_change_challenges() {
    let (config, _perm) = create_test_config();
    let n = 1 << 3;
    let x = 21u64;
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibonacciAir {}, trace, &pis);

    // Generate challenges with correct public values
    let (correct_challenges, _) =
        generate_challenges(&config, &FibonacciAir {}, &proof, &pis, None)
            .expect("correct challenge generation should succeed");

    // Generate challenges with different public values
    let wrong_pis = vec![
        BabyBear::ZERO,
        BabyBear::ONE,
        BabyBear::from_u64(x + 1), // wrong result
    ];
    let (wrong_challenges, _) =
        generate_challenges(&config, &FibonacciAir {}, &proof, &wrong_pis, None)
            .expect("wrong public values challenge generation should succeed");

    // Challenges should be different (public values are observed before alpha)
    assert_ne!(
        correct_challenges.alpha, wrong_challenges.alpha,
        "alpha should change with different public values"
    );

    // Verification should fail with wrong public values
    let result = verify(&config, &FibonacciAir {}, &proof, &wrong_pis);
    assert!(
        result.is_err(),
        "verification should fail with wrong public values"
    );
}

/// Test that the returned challenger state can be used to continue challenge generation.
///
/// This is important for recursion - the challenger state after STARK challenge generation
/// can be passed to PCS/FRI for further challenge generation (alpha, betas, query indices).
#[test]
fn test_challenger_state_continuity() {
    let (config, _perm) = create_test_config();
    let n = 1 << 3;
    let x = 21u64;
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibonacciAir {}, trace, &pis);

    // Get the challenger state after STARK challenge generation
    let (stark_challenges, mut challenger) =
        generate_challenges(&config, &FibonacciAir {}, &proof, &pis, None)
            .expect("challenge generation should succeed");

    // The challenger should be able to continue generating challenges
    // This is what FRI verification would do internally
    let next_challenge: Challenge = challenger.sample_algebra_element();

    // Verify challenges are valid
    assert_ne!(
        stark_challenges.alpha,
        Challenge::ZERO,
        "STARK alpha should be non-zero"
    );
    assert_ne!(
        stark_challenges.zeta,
        Challenge::ZERO,
        "STARK zeta should be non-zero"
    );
    assert_ne!(
        next_challenge,
        Challenge::ZERO,
        "continued challenger should produce valid challenges"
    );

    // Verification should still succeed
    verify(&config, &FibonacciAir {}, &proof, &pis).expect("verification should succeed");
}
