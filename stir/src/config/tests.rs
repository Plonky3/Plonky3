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

type BB = BabyBear;
type BBExt = BinomialExtensionField<BB, 4>;

type BBPerm = Poseidon2BabyBear<16>;
type BBHash = PaddingFreeSponge<BBPerm, 16, 8, 8>;
type BBCompress = TruncatedPermutation<BBPerm, 2, 8, 16>;

type BBMMCS = MerkleTreeMmcs<<BB as Field>::Packing, <BB as Field>::Packing, BBHash, BBCompress, 8>;
type BBExtMMCS = ExtensionMmcs<BB, BBExt, BBMMCS>;

pub fn test_mmcs_config() -> BBExtMMCS {
    let mut rng = ChaCha20Rng::from_os_rng();
    let perm = BBPerm::new_from_rng_128(&mut rng);
    let hash = BBHash::new(perm.clone());
    let compress = BBCompress::new(perm.clone());
    BBExtMMCS::new(BBMMCS::new(hash, compress))
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
        log_starting_degree,
        log_starting_inv_rate,
        log_starting_folding_factor,
        num_rounds,
        security_assumption,
        security_level,
        pow_bits,
        test_mmcs_config(),
    );

    let config: StirConfig<BBExtMMCS> = StirConfig::new::<BBExt>(parameters);

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
