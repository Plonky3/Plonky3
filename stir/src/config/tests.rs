use alloc::vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{SecurityAssumption, StirConfig, StirParameters};

type Bb = BabyBear;
type BbExt = BinomialExtensionField<Bb, 4>;

type BbPerm = Poseidon2BabyBear<16>;
type BbHash = PaddingFreeSponge<BbPerm, 16, 8, 8>;
type BbCompress = TruncatedPermutation<BbPerm, 2, 8, 16>;

type BbMmcs = MerkleTreeMmcs<<Bb as Field>::Packing, <Bb as Field>::Packing, BbHash, BbCompress, 8>;
type BbExtMmcs = ExtensionMmcs<Bb, BbExt, BbMmcs>;

pub fn test_mmcs_config() -> BbExtMmcs {
    let mut rng = ChaCha20Rng::seed_from_u64(467);
    let perm = BbPerm::new_from_rng_128(&mut rng);
    let hash = BbHash::new(perm.clone());
    let compress = BbCompress::new(perm.clone());
    BbExtMmcs::new(BbMmcs::new(hash, compress))
}

#[test]
// Checks the output configuration against one obtained from the co-author's
// repository:
// https://github.com/WizardOfMenlo/stir-whir-scripts/blob/main/src/stir.rs
fn test_config() {
    let security_level = 128;
    let log_starting_degree = 18;
    let log_starting_folding_factor = 4;
    let log_starting_inv_rate = 1;
    let security_assumption = SecurityAssumption::CapacityBound;
    let num_rounds = 4;
    let pow_bits = 20;

    let parameters = StirParameters::constant_folding_factor(
        (security_level, security_assumption),
        log_starting_degree,
        log_starting_inv_rate,
        log_starting_folding_factor,
        num_rounds,
        pow_bits,
        test_mmcs_config(),
    );

    let config: StirConfig<BbExt, BbExtMmcs> = StirConfig::new(parameters);

    assert_eq!(config.starting_domain_log_size(), 19);
    assert_eq!(config.starting_folding_pow_bits(), 30);
    assert_eq!(config.log_starting_inv_rate(), 1);
    assert_eq!(config.log_stopping_degree(), 2);
    assert_eq!(config.final_log_inv_rate(), 10);
    assert_eq!(config.final_num_queries(), 11);
    assert_eq!(config.final_pow_bits(), 19);

    // (folding_factor, log_domain_size, queries, log_inv_rate, pow_bits, ood_samples)
    let expected_round_configs = vec![(4, 117, 4, 42, 2), (4, 28, 7, 45, 2), (4, 16, 10, 49, 2)];

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
