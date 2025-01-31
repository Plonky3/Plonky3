use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::Field;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{SecurityAssumption, StirConfig, StirParameters};

// This configuration is insecure (the field is too small). Use for testing
// purposes only!
type BB = BabyBear;
type BBPerm = Poseidon2BabyBear<16>;
type BBHash = PaddingFreeSponge<BBPerm, 16, 8, 8>;
type BBCompress = TruncatedPermutation<BBPerm, 2, 8, 16>;
type BBPacking = <BB as Field>::Packing;
type BBMMCS = MerkleTreeMmcs<BBPacking, BBPacking, BBHash, BBCompress, 8>;
type BBChallenger = DuplexChallenger<BB, BBPerm, 16, 8>;

pub fn test_mmcs_config() -> BBMMCS {
    let mut rng = ChaCha20Rng::from_entropy();
    let perm = BBPerm::new_from_rng_128(&mut rng);
    let hash = BBHash::new(perm.clone());
    let compress = BBCompress::new(perm.clone());
    BBMMCS::new(hash, compress)
}

pub fn test_challenger() -> BBChallenger {
    let mut rng = ChaCha20Rng::from_entropy();
    let perm = BBPerm::new_from_rng_128(&mut rng);
    BBChallenger::new(perm)
}

pub fn test_stir_config(
    log_starting_degree: usize,
    log_starting_inv_rate: usize,
    log_folding_factor: usize,
    num_rounds: usize,
) -> StirConfig<BB, BBMMCS> {
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
        test_mmcs_config(),
    );

    StirConfig::new(parameters)
}

// NP TODO ask Giacomo if the computation is okay
pub fn test_stir_config_folding_factors(
    log_starting_degree: usize,
    log_starting_inv_rate: usize,
    log_folding_factors: Vec<usize>,
) -> StirConfig<BB, BBMMCS> {
    let security_level = 128;
    let security_assumption = SecurityAssumption::CapacityBound;
    let pow_bits = 20;

    // There is one folding per round, but the last folded polynomial is not
    // encoded but rather sent in plain
    let num_folded_codewords = log_folding_factors.len() - 1;

    // On the other hand, the first codeword doesn't come from a folded
    // polynomial, i. e. it corresponds to a folding factor of 2^0
    let codeword_folding_factors = (0_usize..1).chain(
        log_folding_factors
            .clone()
            .into_iter()
            .take(num_folded_codewords),
    );

    // With each subsequent round, the size of the evaluation domain is
    // decreased by a factor of 2 whereas the degree bound (plus 1) of the
    // polynomial is decreased by a factor of 2^log_folding_factor. Thus,
    // the logarithm of the inverse of the rate increases by log_k - 1.
    let mut i_th_log_rate = log_starting_inv_rate;

    let log_inv_rates = codeword_folding_factors
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
        mmcs_config: test_mmcs_config(),
    };

    StirConfig::new(parameters)
}
