use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{
    InvalidProofShapeError, StarkConfig, VerificationError, prove_with_preprocessed,
    setup_preprocessed, verify_with_preprocessed,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// One main column that must equal one preprocessed column on the current row.
///
/// Neither trace is read at the next row.
struct CopyAir {
    num_rows: usize,
}

impl<F: PrimeField64> BaseAir<F> for CopyAir {
    fn width(&self) -> usize {
        1
    }

    fn preprocessed_width(&self) -> usize {
        1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(RowMajorMatrix::new(
            (0..self.num_rows).map(F::from_usize).collect(),
            1,
        ))
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB: AirBuilder> Air<AB> for CopyAir
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current(0).unwrap();
        let prep = builder.preprocessed().current(0).unwrap();
        builder.assert_eq(local, prep);
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
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = Pcs::new(Dft::default(), val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    MyConfig::new(pcs, challenger)
}

fn prove_copy_air() -> (
    MyConfig,
    CopyAir,
    p3_uni_stark::Proof<MyConfig>,
    p3_uni_stark::PreprocessedVerifierKey<MyConfig>,
) {
    let num_rows = 1 << 3;
    let config = make_config();
    let air = CopyAir { num_rows };
    let trace = RowMajorMatrix::new((0..num_rows).map(Val::from_usize).collect(), 1);
    let (prover_data, vk) = setup_preprocessed::<MyConfig, _>(&config, &air, 3)
        .unwrap()
        .unwrap();
    let proof = prove_with_preprocessed(&config, &air, trace, &[], Some(&prover_data)).unwrap();
    (config, air, proof, vk)
}

#[test]
fn test_preprocessed_no_next_row_air() {
    let (config, air, proof, vk) = prove_copy_air();

    assert!(
        proof.opened_values.preprocessed_next.is_none(),
        "preprocessed_next should be None for an AIR that does not read the next preprocessed row"
    );

    verify_with_preprocessed(&config, &air, &proof, &[], Some(&vk))
        .expect("verification should succeed");
}

#[test]
fn test_preprocessed_no_next_row_rejects_present_preprocessed_next() {
    let (config, air, proof, vk) = prove_copy_air();

    // Invariant: a proof carries exactly the openings its AIR reads, and no others.
    //
    // Fixture state: this AIR reads the preprocessed trace on the current row only.
    // An honest proof therefore carries no next-row preprocessed opening.
    //
    // Mutation: add an empty next-row preprocessed opening.
    //
    //     honest:   preprocessed next = absent
    //     tampered: preprocessed next = [] (present, zero columns)
    //
    // Zero columns is the width expected of an absent opening, so presence is what rejects it.
    //
    // No other field has to change, because the opening argument does not cover this one.
    let mut tampered = proof;
    tampered.opened_values.preprocessed_next = Some(vec![]);

    let err = verify_with_preprocessed(&config, &air, &tampered, &[], Some(&vk))
        .expect_err("verifier should reject a present preprocessed_next when the AIR does not read the next row");
    assert!(
        matches!(
            err,
            VerificationError::InvalidProofShape(
                InvalidProofShapeError::UnexpectedPreprocessedNext { air: None }
            )
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn test_no_next_row_rejects_present_trace_next() {
    let (config, air, proof, vk) = prove_copy_air();

    // Invariant: the same rule applies to the main trace as to the preprocessed trace.
    //
    // Fixture state: this AIR reads the main trace on the current row only.
    // An honest proof therefore carries no next-row main opening.
    //
    // Mutation: add an empty next-row main opening.
    //
    //     honest:   trace next = absent
    //     tampered: trace next = [] (present, zero columns)
    //
    // The rejection names the next main row rather than reporting a generic dimension mismatch.
    let mut tampered = proof;
    tampered.opened_values.trace_next = Some(vec![]);

    let err = verify_with_preprocessed(&config, &air, &tampered, &[], Some(&vk)).expect_err(
        "verifier should reject a present trace_next when the AIR does not read the next row",
    );
    assert!(
        matches!(
            err,
            VerificationError::InvalidProofShape(InvalidProofShapeError::UnexpectedTraceNext {
                air: None
            })
        ),
        "unexpected error: {err:?}"
    );
}
