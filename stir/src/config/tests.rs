use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{SecurityAssumption, StirConfig, StirParameters};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

pub fn test_mmcs_config() -> ChallengeMmcs {
    let mut rng = ChaCha20Rng::from_entropy();
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    ChallengeMmcs::new(ValMmcs::new(hash, compress))
}

#[test]
fn test_config() {
    let security_level = 128;
    let log_starting_degree = 18;
    let log_starting_folding_factor = 4;
    let log_starting_inv_rate = 1;
    let security_assumption = SecurityAssumption::CapacityBound;
    let num_rounds = 4;
    let pow_bits = 20;

    let parameters = StirParameters::fixed_domain_shift(
        log_starting_degree,
        log_starting_inv_rate,
        log_starting_folding_factor,
        num_rounds,
        security_assumption,
        security_level,
        pow_bits,
        test_mmcs_config(),
    );

    let config: StirConfig<Challenge, ChallengeMmcs> = StirConfig::new(parameters);

    assert_eq!(config.starting_domain_log_size(), 19);
    assert_eq!(config.starting_folding_pow_bits(), 30);
    assert_eq!(config.log_stopping_degree(), 2);
    assert_eq!(config.final_log_inv_rate(), 10);
    assert_eq!(config.final_num_queries(), 11);
    assert_eq!(config.final_pow_bits(), 19);

    // (folding_factor, log_domain_size, queries, log_inv_rate, pow_bits, ood_samples)
    let expected_round_configs = vec![(4, 117, 1, 42, 2), (4, 28, 4, 45, 2), (4, 16, 7, 49, 2)];

    for (
        round_config,
        (log_folding_factor, num_queries, log_inv_rate, pow_bits, num_ood_samples),
    ) in config.round_configs().iter().zip(expected_round_configs)
    {
        assert_eq!(round_config.log_folding_factor, log_folding_factor);
        assert_eq!(round_config.num_queries, num_queries);
        assert_eq!(round_config.log_inv_rate, log_inv_rate);
        assert_eq!(round_config.pow_bits, pow_bits);
        assert_eq!(round_config.num_ood_samples, num_ood_samples);
    }
}
