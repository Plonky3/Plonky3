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
