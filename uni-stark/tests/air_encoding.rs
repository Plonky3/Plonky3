use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::verifier::FriError;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, VerificationError, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Config = StarkConfig<Pcs, Challenge, Challenger>;

const LOG_TRACE_ROWS: usize = 6;

/// One-column AIR carrying an instance tag in its encoding but not in its constraints.
///
/// The tag is absorbed through the instance encoding, so two tags yield two transcripts.
/// The single constraint pins the column to zero and never reads the tag.
/// A proof is therefore valid for at most one tag.
#[derive(Clone)]
struct TaggedAir {
    tag: u64,
}

impl<F: Field> BaseAir<F> for TaggedAir {
    fn width(&self) -> usize {
        1
    }

    fn instance_encoding(&self) -> Vec<F>
    where
        F: PrimeCharacteristicRing,
    {
        vec![F::from_u64(self.tag)]
    }
}

impl<AB: AirBuilder> Air<AB> for TaggedAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        // The single column must be zero on every row, independent of the tag.
        let main = builder.main();
        let local = main.current_slice();
        builder.assert_zero(local[0]);
    }
}

fn config() -> Config {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len: 3,
        max_log_arity: 2,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(Dft::default(), val_mmcs, fri_params);
    StarkConfig::new(pcs, Challenger::new(perm))
}

#[test]
fn air_owned_data_is_bound_into_the_transcript() {
    let config = config();

    // An all-zero single-column trace satisfies the constraint for any tag.
    let trace = RowMajorMatrix::new(Val::zero_vec(1 << LOG_TRACE_ROWS), 1);

    // Prove against tag 1.
    let air_1 = TaggedAir { tag: 1 };
    let proof = prove(&config, &air_1, trace, &[]);

    // The same instance verifies: the transcript replays identically.
    verify(&config, &air_1, &proof, &[]).expect("matching instance must verify");

    // Tag 2 diverges the transcript at the instance encoding.
    // The sampled challenges differ, so the query grinding witness no longer holds.
    let air_2 = TaggedAir { tag: 2 };
    let result = verify(&config, &air_2, &proof, &[]);
    assert!(
        matches!(
            result,
            Err(VerificationError::InvalidOpeningArgument(
                FriError::InvalidPowWitness
            ))
        ),
        "a proof must not verify against a different instance, got {result:?}"
    );
}
