use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{CanSampleBits, DuplexChallenger, FieldChallenger};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_stir::{SoundnessType, StirConfig};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

fn mmcs_config() -> ChallengeMmcs {
    let mut rng = ChaCha20Rng::from_entropy();
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut rng,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    ChallengeMmcs::new(ValMmcs::new(hash, compress))
}

#[test]
fn test_config() {
    let security_level = 128;
    let protocol_security_level = 106;
    let starting_degree = 1 << 18;
    let stopping_degree = 1 << 6;
    let folding_factor = 1 << 4;
    let starting_log_inv_rate = 1;
    let soundness_type = SoundnessType::Conjecture;
    let mmcs_config = mmcs_config();

    let config = StirConfig::new(
        security_level,
        protocol_security_level,
        mmcs_config,
        soundness_type,
        starting_degree,
        stopping_degree,
        folding_factor,
        starting_log_inv_rate,
    );

    assert_eq!(config.num_rounds, 2);
    assert_eq!(config.degrees, vec![1 << 18, 1 << 14, 1 << 10]);
    assert_eq!(config.log_inv_rates, vec![1, 4, 7]);
    assert_eq!(config.repetitions, vec![106, 27, 16]);
    assert_eq!(config.proof_of_work_bits, vec![22, 20, 16]);
}
