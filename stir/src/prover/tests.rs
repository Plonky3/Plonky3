use itertools::Itertools;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::MockChallenger;
use p3_commit::Mmcs;
use p3_field::{extension::BinomialExtensionField, Field, FieldAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{
    coset::Radix2Coset, polynomial::Polynomial, prover::commit, utils::field_element_from_isize,
    SecurityAssumption, StirConfig, StirParameters,
};

use super::{prove_round, RoundConfig, StirWitness};

// This configuration is insecure (the field is too small). Use for testing
// purposes only!
type BB = BabyBear;
type BBPerm = Poseidon2BabyBear<16>;
type BBHash = PaddingFreeSponge<BBPerm, 16, 8, 8>;
type BBCompress = TruncatedPermutation<BBPerm, 2, 8, 16>;
type BBPacking = <BB as Field>::Packing;
type BBMMCS = MerkleTreeMmcs<BBPacking, BBPacking, BBHash, BBCompress, 8>;

pub fn test_mmcs_config() -> BBMMCS {
    let mut rng = ChaCha20Rng::from_entropy();
    let perm = BBPerm::new_from_rng_128(&mut rng);
    let hash = BBHash::new(perm.clone());
    let compress = BBCompress::new(perm.clone());
    BBMMCS::new(hash, compress)
}

fn test_stir_config() -> StirConfig<BBMMCS> {
    let security_level = 128;
    let log_starting_degree = 4;
    let log_folding_factor = 2;
    let log_starting_inv_rate = 1;
    let security_assumption = SecurityAssumption::CapacityBound;
    let num_rounds = 2;
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

    StirConfig::new::<BB>(parameters)
}

#[test]
fn test_prove_round() {
    let config = test_stir_config();

    // TODO remove
    println!("REACHES 0");

    let round = 0;

    let round_config = config.round_config(round);

    // TODO remove
    println!("REACHES 1");

    let RoundConfig {
        log_evaluation_domain_size,
        ood_samples,
        num_queries,
        ..
    } = round_config.clone();

    // TODO remove
    println!("REACHES 2");

    let field_replies = [
        // ood_samples
        (0..ood_samples).map(|_| BB::ZERO).collect_vec(),
        vec![
            // comb_randomness
            BB::ZERO,
            // folding_randomness
            BB::ZERO,
            // shake_randomness (unused)
            BB::ZERO,
        ],
    ]
    .concat();

    // TODO remove
    println!("REACHES 3");

    // indices
    let bit_replies = (0..num_queries).map(|_| 0).collect::<Vec<_>>();

    // TODO remove
    println!("REACHES 4");

    let mut challenger = MockChallenger::new(field_replies, bit_replies);

    // TODO remove
    println!("REACHES 5");

    // Starting domain: 10 <w> with w of size

    // Starting polynomial: -2 + 17x + 42x^2 + 3x^3 - x^4 - x^5 + 4x^6 + 5x^7
    let coeffs: Vec<BB> = vec![-2, 17, 42, 3, -1, -1, 4, 5]
        .into_iter()
        .map(field_element_from_isize)
        .collect_vec();

    // TODO remove
    println!("REACHES 6");

    let f = Polynomial::from_coeffs(coeffs);

    // TODO remove
    println!("REACHES 7");

    let original_domain =
        Radix2Coset::new(BB::from_canonical_usize(10), log_evaluation_domain_size);

    // TODO remove
    println!("REACHES 8");

    let original_evals = original_domain.evaluate_polynomial(&f);

    // TODO remove
    println!("REACHES 9");

    let stacked_original_evals =
        RowMajorMatrix::new(original_evals, 1 << config.log_starting_folding_factor());

    // TODO remove
    println!("REACHES 10");

    let (_, merkle_tree) = config
        .mmcs_config()
        .commit_matrix(stacked_original_evals.clone());

    // TODO remove
    println!("REACHES 11");

    let witness = StirWitness {
        domain: original_domain,
        polynomial: f,
        merkle_tree,
        stacked_evals: stacked_original_evals,
        round,
        folding_randomness: BB::ZERO,
    };

    // TODO remove
    println!("REACHES 12");

    let (round_proof, witness) = prove_round(&config, witness, &mut challenger);
}

// NP TODO discuss with Giacomo Every round needs two: this round's, to know how
// to fold; and next round's, to know how to stack the evaluations of the final
// polynomial f' produced by this round

// Original word: 2^10

// Starting folding factor: 2^1
// Next folding factors: [2^2, 2^3, ...]

// Prepare witness for the first round:
// Evaluations of the original word
// Have to stack the evaluations in rows of 2^1 elements

// Round 1:
//   Fold the polynomial with arity 2^1
//   Open and prove folding

// NP TODO polynomial with degree strictly lower than the bound

// NP TODO failed config creation where the num of rounds is too large for the starting degree and folding factor (maybe in /config)
