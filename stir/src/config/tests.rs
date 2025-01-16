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
type BabyBear4 = BinomialExtensionField<BabyBear, 4>;

fn mmcs_config() -> ChallengeMmcs {
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
    let num_rounds = 3;
    let pow_bits = 20;

    let parameters = StirParameters::fixed_domain_shift(
        log_starting_degree,
        log_starting_inv_rate,
        log_starting_folding_factor,
        num_rounds,
        security_assumption,
        security_level,
        pow_bits,
        mmcs_config(),
    );

    let config = StirConfig::new::<BabyBear4>(parameters);

    assert_eq!(config.starting_domain_log_size(), 19);
    assert_eq!(
        format!("{}", config.starting_folding_pow_bits()),
        "29.228818690495885"
    );
    assert_eq!(config.log_stopping_degree(), 2);
    assert_eq!(config.final_log_inv_rate(), 10);
    assert_eq!(config.final_num_queries(), 11);
    assert_eq!(format!("{}", config.final_pow_bits()), "18.77428260680469");

    // (folding_factor, log_domain_size, queries, log_inv_rate, pow_bits, ood_samples)
    let expected_round_configs = vec![
        (4, 18, 117, 1, "41.2045711442492", 2),
        (4, 17, 28, 4, "44.17990909001493", 2),
        (4, 16, 16, 7, "48.4093909361377", 2),
    ];

    for (
        round_config,
        (log_folding_factor, log_domain_size, num_queries, log_inv_rate, pow_bits, ood_samples),
    ) in config.round_configs().iter().zip(expected_round_configs)
    {
        assert_eq!(round_config.log_folding_factor, log_folding_factor);
        assert_eq!(round_config.log_evaluation_domain_size, log_domain_size);
        assert_eq!(round_config.num_queries, num_queries);
        assert_eq!(round_config.log_inv_rate, log_inv_rate);
        assert_eq!(format!("{}", round_config.pow_bits), pow_bits);
        assert_eq!(round_config.ood_samples, ood_samples);
    }
}
