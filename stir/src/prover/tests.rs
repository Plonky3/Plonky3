use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::{extension::BinomialExtensionField, Field};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{SecurityAssumption, StirConfig, StirParameters};

use super::prove_round;

// This configuration is insecure!
type BB = BabyBear;
type BBExt = BinomialExtensionField<BB, 4>;
type BBExtPerm = Poseidon2BabyBear<16>;
type BBExtHash = PaddingFreeSponge<BBExtPerm, 16, 8, 8>;
type BBExtCompress = TruncatedPermutation<BBExtPerm, 2, 8, 16>;
type BBExtPacking = <BBExt as Field>::Packing;

type BBExtMMCS = MerkleTreeMmcs<BBExtPacking, BBExtPacking, BBExtHash, BBExtCompress, 8>;

pub fn test_mmcs_config() -> BBExtMMCS {
    let mut rng = ChaCha20Rng::from_entropy();
    let perm = BBExtPerm::new_from_rng_128(&mut rng);
    let hash = BBExtHash::new(perm.clone());
    let compress = BBExtCompress::new(perm.clone());
    BBExtMMCS::new(hash, compress)
}

fn test_stir_config() -> StirConfig<BBExtMMCS> {
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
        test_mmcs_config(),
    );

    StirConfig::new::<BB>(parameters)
}

#[test]
fn test_prove_round() {
    let config = test_stir_config();

    let challenger = MockChallenger::new();

    prove_round(&config, witness, &mut challenger);
}
