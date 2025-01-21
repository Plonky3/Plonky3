use itertools::Itertools;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::MockChallenger;
use p3_commit::Mmcs;
use p3_field::{Field, FieldAlgebra};
use p3_matrix::{dense::RowMajorMatrix, Dimensions};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    coset::Radix2Coset,
    polynomial::{rand_poly, Polynomial},
    proof::RoundProof,
    utils::field_element_from_isize,
    utils::fold_polynomial,
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

fn test_stir_config(
    log_starting_degree: usize,
    log_starting_inv_rate: usize,
    log_folding_factor: usize,
) -> StirConfig<BB, BBMMCS> {
    let security_level = 128;
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

    StirConfig::new(parameters)
}

// NP TODO either remove the manual values or use them by reducing the soundness
#[test]
fn test_prove_round_zero() {
    let config = test_stir_config(3, 1, 1);

    let round = 0;

    let round_config = config.round_config(round);

    let RoundConfig {
        log_evaluation_domain_size,
        num_ood_samples,
        num_queries,
        log_folding_factor,
        ..
    } = round_config.clone();

    let field_replies = [
        // ood_samples
        (0..num_ood_samples)
            .map(|x| BB::from_canonical_usize(3) * BB::from_canonical_usize(x))
            .collect_vec(),
        vec![
            // comb_randomness
            BB::ONE,
            // folding_randomness
            BB::ONE,
            // shake_randomness (unused)
            BB::ONE,
        ],
    ]
    .concat();

    // indices

    let log_size_second_codeword = config.log_starting_degree() + config.log_starting_inv_rate()
        - config.log_starting_folding_factor();

    let bit_replies = (0..num_queries)
        .map(|i| i % (1 << log_size_second_codeword))
        .collect::<Vec<_>>();

    let mut challenger = MockChallenger::new(field_replies, bit_replies.clone());

    // Starting polynomial: -2 + 17x + 42x^2 + 3x^3 - x^4 - x^5 + 4x^6 + 5x^7
    let coeffs: Vec<BB> = vec![-2, 17, 42, 3, -1, -1, 4, 5]
        .into_iter()
        .map(field_element_from_isize)
        .collect_vec();

    let f = Polynomial::from_coeffs(coeffs);

    let original_domain = Radix2Coset::new(
        BB::ONE,
        config.log_starting_degree() + config.log_starting_inv_rate(),
    );

    let original_evals = original_domain.evaluate_polynomial(&f);

    let stacked_original_evals =
        RowMajorMatrix::new(original_evals, 1 << config.log_starting_folding_factor());

    let (_, merkle_tree) = config
        .mmcs_config()
        .commit_matrix(stacked_original_evals.clone());

    let witness = StirWitness {
        domain: original_domain.clone(),
        polynomial: f,
        merkle_tree,
        stacked_evals: stacked_original_evals,
        round,
        folding_randomness: BB::from_canonical_usize(2),
    };

    let (witness, round_proof) = prove_round(&config, witness, &mut challenger);

    // =============== Witness Checks ===============

    // expected_shift = omega * original_shift^2 (with original_shift = 1)
    let expected_shift = original_domain.generator() * original_domain.shift().square();
    let expected_domain = original_domain.shrink_subgroup(1).set_shift(expected_shift);

    let expected_round = 1;
    let expected_folding_randomness = BB::ONE;

    let StirWitness {
        domain,
        polynomial,
        merkle_tree,
        stacked_evals,
        folding_randomness,
        round,
    } = witness;

    // Domain testing
    assert_eq!(domain, expected_domain);
    assert_eq!(folding_randomness, expected_folding_randomness);

    // Round-number testing
    assert_eq!(round, expected_round);

    // Polynomial testing In this case, the security level means the
    // interpolator has the same degree as the folded polynomial
    assert!(polynomial.is_zero());

    // Polynomial-evaluation testing
    assert!(domain.iter().all(|x| polynomial.evaluate(&x) == BB::ZERO));

    // ============== Round Proof Checks ===============

    let RoundProof {
        g_root,
        betas,
        ans_polynomial,
        query_proofs,
        shake_polynomial,
        pow_witness,
    } = round_proof;

    for (&i, (leaf, proof)) in bit_replies.iter().zip(query_proofs) {
        config
            .mmcs_config()
            .verify_batch(
                &g_root,
                &[Dimensions {
                    width: 1 << log_folding_factor,
                    height: 1 << (original_domain.log_size() - log_folding_factor),
                }],
                i,
                &leaf,
                &proof,
            )
            .unwrap();
    }
}

#[test]
fn test_prove_round_large() {
    let mut rng = rand::thread_rng();

    let config = test_stir_config(10, 3, 2);

    let round = 0;

    let round_config = config.round_config(round);

    let RoundConfig {
        log_folding_factor,
        log_inv_rate,
        num_ood_samples,
        num_queries,
        ..
    } = round_config.clone();

    let r_0: BB = rng.gen(); // Initial folding randomness

    // Field randomness produced by the sponge
    let r_1: BB = rng.gen(); // Folding randomness for round 2 (which never happens)
    let ood_randomness: Vec<BB> = (0..num_ood_samples).map(|_| rng.gen()).collect();
    let comb_randomness = rng.gen();
    let _shake_randomness = rng.gen();

    let mut field_replies = ood_randomness.clone();
    field_replies.push(comb_randomness);
    field_replies.push(r_1);
    field_replies.push(_shake_randomness);

    // Index randomness
    let log_size_second_codeword = config.log_starting_degree() - log_folding_factor + log_inv_rate;

    let bit_replies = (0..num_queries)
        .map(|_| rng.gen())
        .map(|i: usize| i % (1 << log_size_second_codeword))
        .collect::<Vec<_>>();

    // Preloading fake randomness
    let mut challenger = MockChallenger::new(field_replies, bit_replies.clone());

    // Starting polynomial
    let f_0 = rand_poly((1 << config.log_starting_degree()) - 1);

    let original_domain = Radix2Coset::new(
        BB::ONE,
        config.log_starting_degree() + config.log_starting_inv_rate(),
    );

    let original_evals = original_domain.evaluate_polynomial(&f_0);

    let stacked_original_evals =
        RowMajorMatrix::new(original_evals, 1 << config.log_starting_folding_factor());

    let (_, merkle_tree) = config
        .mmcs_config()
        .commit_matrix(stacked_original_evals.clone());

    let witness = StirWitness {
        domain: original_domain.clone(),
        polynomial: f_0.clone(),
        merkle_tree,
        stacked_evals: stacked_original_evals,
        round,
        folding_randomness: r_0,
    };

    let (witness, _) = prove_round(&config, witness, &mut challenger);

    // =============== Witness Checks ===============

    let StirWitness {
        domain,
        polynomial: f_1,
        folding_randomness,
        round,
        ..
    } = witness;

    // expected_shift = omega * original_shift^2 (with original_shift = 1)
    let expected_shift = original_domain.generator() * original_domain.shift().square();
    let expected_domain = original_domain.shrink_subgroup(1).set_shift(expected_shift);

    let expected_round = 1;

    // Domain testing
    assert_eq!(domain, expected_domain);
    assert_eq!(r_1, folding_randomness);

    // Round-number testing
    assert_eq!(round, expected_round);

    // Polynomial testing
    let g_1 = fold_polynomial(&f_0, r_0, log_folding_factor);

    // NP TODO repeat points
    let original_domain_pow_k = original_domain.shrink_coset(log_folding_factor);
    let stir_randomness = bit_replies
        .iter()
        .map(|&i| original_domain_pow_k.element(i as u64));
    let quotient_set = stir_randomness
        .chain(ood_randomness.into_iter())
        .unique()
        .collect_vec();
    let quotient_set_points = quotient_set
        .iter()
        .map(|x| (*x, g_1.evaluate(x)))
        .collect_vec();
    let ans_polynomial = Polynomial::lagrange_interpolation(quotient_set_points);
    let quotient_polynomial =
        &(&g_1 - &ans_polynomial) / &Polynomial::vanishing_polynomial(quotient_set.clone());
    let expected_f_1 =
        &Polynomial::power_polynomial(comb_randomness, quotient_set.len()) * &quotient_polynomial;

    assert_eq!(f_1, expected_f_1);
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

// L0 = shift <w>
// L1 = shift² <w²>
// L0^(2^logk) \cap L1 = empty

// L2 = w^2 * L1^2 = w^2 * w^2 * <w^2^2> = w^4 * <w^4> = <w^4>

// L0 = 1 * <w>
//
// L_i.gen = L_(i - 1).gen^2
// L_i.shift = if L_(i - 1).shift == 1 { w } else { 1 }

// L_2j = 1 * <w^{2^{2j}}>
// L_{2j + 1} = w * <w^{2^{2j + 1}}>
// L_2j^k

// o^2 * <w^2> \cap o * <w^2>
// LHS: {o^2 * w^(2i): i...}
// RHS: {o * w^(2i): i...}

// Take any i, j. The shift o = w^(2i - 2j) = w^(2i) / w^(2j) gives you an intersection point

// j-th element of the LHS: o^2 * w^(2j) = o * w^(2i)
// i-th element of the RHS: o * w^(2i)

// L0 = w * <w>
// L1 = w * <w^2>
// L2 = w * <w^4>
// L3 = w * <w^8>

// L0 = w * <w>
// L1 = w * <w^2>
// L2 = w * <w^4>
// L3 = w * <w^8>
