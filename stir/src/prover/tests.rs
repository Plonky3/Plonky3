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
    coset::Radix2Coset, polynomial::Polynomial, utils::field_element_from_isize,
    SecurityAssumption, StirConfig, StirParameters,
};

use super::{prove_round, RoundConfig, StirWitness};

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
    let log_starting_degree = 4;
    let log_folding_factor = 2;
    let log_starting_inv_rate = 1;
    let security_assumption = SecurityAssumption::CapacityBound;
    let num_rounds = 1;
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

    let round = 0;

    let round_config = config.round_config(round);

    let RoundConfig {
        log_evaluation_domain_size,
        ood_samples,
        num_queries,
        ..
    } = round_config.clone();

    let field_replies = [
        // ood_samples
        (0..ood_samples).map(|_| BBExt::ZERO).collect_vec(),
        vec![
            // comb_randomness
            BBExt::ZERO,
            // folding_randomness
            BBExt::ZERO,
            // shake_randomness (unused)
            BBExt::ZERO,
        ],
    ]
    .concat();

    // indices
    let bit_replies = (0..num_queries).map(|_| 0).collect::<Vec<_>>();

    let challenger = MockChallenger::new(field_replies, bit_replies);

    // Starting domain: 10 <w> with w of size

    // Starting polynomial: -2 + 17x + 42x^2 + 3x^3 - x^4 - x^5 + 4x^6 + 5x^7
    let coeffs: Vec<BBExt> = vec![-2, 17, 42, 3, -1, -1, 4, 5]
        .into_iter()
        .map(field_element_from_isize)
        .collect_vec();

    let f = Polynomial::from_coeffs(coeffs);

    let original_domain =
        Radix2Coset::new(BBExt::from_canonical_usize(10), log_evaluation_domain_size);

    let original_evals = original_domain.evaluate_polynomial(&f);
    let stacked_original_evals =
        RowMajorMatrix::new(original_evals, 1 << config.log_starting_folding_factor());

    // let _ = config.mmcs_config().commit_matrix(stacked_original_evals);

    // let witness = StirWitness {
    //     domain: original_domain,
    //     polynomial: f,
    //     merkle_tree,
    //     stacked_evals: ,
    //     round,
    //     folding_randomness: F::ZERO,
    // };

    // prove_round(&config, witness, &mut challenger);
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
