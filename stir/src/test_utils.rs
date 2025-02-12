use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::{extension::BinomialExtensionField, Field};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{SecurityAssumption, StirConfig, StirParameters};

// This configuration is insecure (the field is too small). Use for testing
// purposes only!
pub type BB = BabyBear;
pub type BBExt = BinomialExtensionField<BB, 5>;

type BBPerm = Poseidon2BabyBear<16>;
type BBHash = PaddingFreeSponge<BBPerm, 16, 8, 8>;
type BBCompress = TruncatedPermutation<BBPerm, 2, 8, 16>;
type BBPacking = <BB as Field>::Packing;

type BBMMCS = MerkleTreeMmcs<BBPacking, BBPacking, BBHash, BBCompress, 8>;
pub type BBExtMMCS = ExtensionMmcs<BB, BBExt, BBMMCS>;

pub type BBChallenger = DuplexChallenger<BB, BBPerm, 16, 8>;

pub fn test_bb_mmcs_config() -> BBExtMMCS {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let perm = BBPerm::new_from_rng_128(&mut rng);
    let hash = BBHash::new(perm.clone());
    let compress = BBCompress::new(perm.clone());
    BBExtMMCS::new(BBMMCS::new(hash, compress))
}

pub fn test_bb_challenger() -> BBChallenger {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let perm = BBPerm::new_from_rng_128(&mut rng);
    BBChallenger::new(perm)
}

pub fn test_bb_stir_config(
    log_starting_degree: usize,
    log_starting_inv_rate: usize,
    log_folding_factor: usize,
    num_rounds: usize,
) -> StirConfig<BBExt, BBExtMMCS> {
    let security_level = 128;
    let security_assumption = SecurityAssumption::CapacityBound;
    let pow_bits = 20;

    let parameters = StirParameters::fixed_domain_shift(
        log_starting_degree,
        log_starting_inv_rate,
        log_folding_factor,
        num_rounds,
        security_assumption,
        security_level,
        pow_bits,
        test_bb_mmcs_config(),
    );

    StirConfig::new(parameters)
}

pub fn test_stir_config_folding_factors(
    log_starting_degree: usize,
    log_starting_inv_rate: usize,
    log_folding_factors: Vec<usize>,
) -> StirConfig<BBExt, BBExtMMCS> {
    let security_level = 128;
    let security_assumption = SecurityAssumption::CapacityBound;
    let pow_bits = 20;

    // With each subsequent round, the size of the evaluation domain is
    // decreased by a factor of 2 whereas the degree bound (plus 1) of the
    // polynomial is decreased by a factor of 2^log_folding_factor. Thus,
    // the logarithm of the inverse of the rate increases by log_k - 1.
    let mut i_th_log_rate = log_starting_inv_rate;

    let log_inv_rates = log_folding_factors
        .iter()
        .map(|log_k| {
            i_th_log_rate = i_th_log_rate + log_k - 1;
            i_th_log_rate
        })
        .collect();

    let parameters = StirParameters {
        log_starting_degree,
        log_starting_inv_rate,
        log_folding_factors,
        log_inv_rates,
        security_assumption,
        security_level,
        pow_bits,
        mmcs_config: test_bb_mmcs_config(),
    };

    StirConfig::new(parameters)
}
