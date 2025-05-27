use alloc::{vec, vec::Vec};

use itertools::Itertools;
use p3_challenger::MockChallenger;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{eval_poly, BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::RoundConfig;
use crate::prover::{commit_polynomial, prove};
use crate::test_utils::*;
use crate::utils::{
    divide_poly_with_remainder, fold_polynomial, lagrange_interpolation, mul_polys,
    power_polynomial, subtract_polys, vanishing_polynomial,
};
use crate::SecurityAssumption;

// Auxiliary function to read an algebra element from a slice of base-field
// elements
#[inline]
fn read_algebra_element<F: Field, EF: ExtensionField<F>>(coefficients: &[F], index: usize) -> EF {
    EF::from_basis_coefficients_slice(
        &coefficients[index * EF::DIMENSION..(index + 1) * EF::DIMENSION],
    )
    .unwrap()
}

#[test]
// Checks that prove runs from beginning to end and performs a degree check on
// the final polynomial p = g_{num_rounds} (where num_rounds = M + 1 in the
// notation of the article)
fn test_prove() {
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        14,
        1,
        4,
        3,
    );

    let polynomial = rand_poly_coeffs_seeded((1 << config.log_starting_degree()) - 1, Some(103));

    let (witness, commitment) = commit_polynomial(&config, polynomial);

    let mut challenger = test_bb_challenger();

    let proof = prove(&config, witness, commitment, &mut challenger);

    // Final-degree testing
    assert_eq!(config.log_stopping_degree(), 2);
    assert!(proof.final_polynomial.len() <= 1 << 2);
}

#[test]
// Checks that the final polynomial p = g_3 is the expected one in three-round
// STIR
fn test_prove_final_polynomial() {
    let mut rng = SmallRng::seed_from_u64(101);

    let log_starting_degree = 15;
    let log_inv_rate = 2;
    let log_folding_factor = 3;

    // We use a config that will result in the following polynomials and degree
    // bounds:
    //  - f_0:      2^15 - 1
    //  - f_1, g_1: 2^12 - 1
    //  - f_2, g_2: 2^9 - 1
    //  - g_3:      2^6 - 1
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        log_starting_degree,
        log_inv_rate,
        log_folding_factor,
        3,
    );

    let ext_degree = <BbExt as BasedVectorSpace<Bb>>::DIMENSION;

    let log_initial_codeword_size = log_starting_degree + config.log_starting_inv_rate();

    // ======================= Preparing fake randomness =======================
    let mut field_replies = Vec::new();
    let mut bit_replies = Vec::new();

    let mut round_ood_replies = Vec::new();
    let mut round_comb_replies = Vec::new();
    let mut round_r_replies = Vec::new();
    let mut round_shake_replies = Vec::new();
    let mut round_bit_replies = Vec::new();

    let r_0 = vec![rng.random(); ext_degree];
    field_replies.extend_from_slice(&r_0);
    round_r_replies.extend(r_0);

    for round in 1..=2 {
        let round_config = config.round_config(round);

        let RoundConfig {
            log_folding_factor,
            num_ood_samples,
            num_queries,
            ..
        } = round_config.clone();

        // Out of domain randomness
        // We need to construct this and the other MockChallenger values as
        // sponges over Bb instead of BbExt and then convert slices to sampled
        // values into extension-field elements.
        let ood_randomness: Vec<Bb> = (0..num_ood_samples * ext_degree)
            .map(|_| rng.random())
            .collect();

        // Comb randomness
        let comb_randomness: Vec<Bb> = vec![rng.random(); ext_degree];

        // Folding randomness
        let r: Vec<Bb> = vec![rng.random(); ext_degree];

        // Shake randomness (which is squeezed but not used by the prover)
        let shake_randomness: Vec<Bb> = vec![rng.random(); ext_degree];

        field_replies.extend_from_slice(&ood_randomness);
        field_replies.extend_from_slice(&comb_randomness);
        field_replies.extend_from_slice(&r);
        field_replies.extend_from_slice(&shake_randomness);

        round_ood_replies.push(ood_randomness);
        round_comb_replies.extend(comb_randomness);
        round_r_replies.extend(r);
        round_shake_replies.extend(shake_randomness);

        // This is the log2 of |L_{i - 1}^{k_{i - 1}}|
        let log_prev_domain_size = log_initial_codeword_size - (round - 1) - log_folding_factor;

        // Random queried indices (in the form of bits, not field elements)
        let new_bit_replies = (0..num_queries)
            .map(|_| rng.random_range(0..usize::MAX))
            .map(|i: usize| i % (1 << log_prev_domain_size))
            .collect::<Vec<_>>();

        bit_replies.extend(new_bit_replies.clone());
        round_bit_replies.push(new_bit_replies);
    }

    // log2 of |L_2^{k_2}|
    let log_final_domain_size = log_initial_codeword_size - 2 - log_folding_factor;

    // Prepare the final-round fake randomness (irrelevant to the final
    // polynomial p = g_3, but sampled from the challenger by the prover)
    let final_bit_replies = (0..config.final_num_queries())
        .map(|_| rng.random_range(0..usize::MAX))
        .map(|i: usize| i % (1 << log_final_domain_size))
        .collect::<Vec<_>>();

    bit_replies.extend(final_bit_replies.clone());

    // Preloading fake randomness
    let mut challenger = MockChallenger::new(field_replies, bit_replies);

    // ================================ Proving ================================

    let mut polynomial = rand_poly_coeffs((1 << config.log_starting_degree()) - 1, &mut rng);

    let (witness, commitment) = commit_polynomial(&config, polynomial.clone());

    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();
    let generator = BbExt::two_adic_generator(log_size);

    let proof = prove(&config, witness, commitment, &mut challenger);

    // ================ Computing expected final polynomial g_3 ===============

    // Computing f_1 and f_2 manually
    for round in 1..=2 {
        let r_i_minus_1 = read_algebra_element(&round_r_replies, round - 1);
        let g_i = fold_polynomial(&polynomial, r_i_minus_1, log_folding_factor);

        // Computing the domain L_{i - 1}^{k_{i - 1}}
        let domain_pow_k =
            TwoAdicMultiplicativeCoset::new(generator, log_initial_codeword_size - (round - 1))
                .unwrap()
                .exp_power_of_2(log_folding_factor)
                .unwrap();

        let stir_randomness = round_bit_replies[round - 1]
            .iter()
            .map(|&i| domain_pow_k.element(i));

        let num_ood = config.round_config(round).num_ood_samples;
        let ood_randomness: Vec<BbExt> = (0..num_ood)
            .map(|i| read_algebra_element(&round_ood_replies[round - 1], i))
            .collect();

        let quotient_set = stir_randomness.chain(ood_randomness).unique().collect_vec();

        let quotient_set_points = quotient_set
            .iter()
            .map(|x| (*x, eval_poly(&g_i, *x)))
            .collect_vec();

        let ans_polynomial = lagrange_interpolation(quotient_set_points.clone());

        let (quotient_polynomial, remainder) = divide_poly_with_remainder(
            subtract_polys(&g_i, &ans_polynomial),
            vanishing_polynomial(quotient_set.clone()),
        );

        assert!(remainder.is_empty());

        let comb_randomness = read_algebra_element(&round_comb_replies, round - 1);

        // New round polynomial f_i
        polynomial = mul_polys(
            &power_polynomial(comb_randomness, quotient_set.len()),
            &quotient_polynomial,
        );
    }

    let f_2 = polynomial.clone();

    // Computing the expected final polynomial p = g_3
    let r_2 = read_algebra_element(&round_r_replies, 2);
    let expected_final_polynomial = fold_polynomial(&f_2, r_2, log_folding_factor);

    assert_eq!(proof.final_polynomial, expected_final_polynomial);
}

#[test]
#[should_panic(expected = "The degree of the polynomial (16384) is too large")]
// Checks that the commit method panics if the polynomial is larger than allowed
// by the configuration
fn test_incorrect_polynomial() {
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        14,
        1,
        4,
        3,
    );

    let polynomial = rand_poly_coeffs_seeded(1 << config.log_starting_degree(), Some(107));

    commit_polynomial(&config, polynomial);
}
