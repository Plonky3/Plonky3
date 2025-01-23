// The virtual function
//     `DegCor(Quotient(f, interpolating_polynomial), quotient_set)`
// in the notation of the paper, where `f` is the underlying function.
struct VirtualFunction<F: TwoAdicField> {
    // In in the case of interest,
    // - f is g_i
    // - interpolating_polynomial is Ans_i
    // - quotient_set is G_i
    comb_randomness: F,
    interpolating_polynomial: Polynomial<F>,
    quotient_set: Vec<F>,
}

// NP TODO rethink generality of description

// Oracle allowing the verifier to compute values of f_i, either directly for
// the original codeword, or using the combination (i. e. degree-correction)
// and folding randomness and the values of g_i.
enum Oracle<F: TwoAdicField> {
    // Transparent oracle: it takes the same value as the underlying function
    Transparent,
    // Virtual oracle: it takes the value of the virtual function over the underlying function
    Virtual(VirtualFunction<F>),
}

impl<F: TwoAdicField> Oracle<F> {
    // Compute the value v(x) of the oracle v at the point x given the value f(x) of the underlying function
    fn evaluate(
        &self,
        x: F,
        f_x: F,
        // NP TODO optimise
        // common_factors_inverse: F,
        // denom_hint: F,
        // ans_eval: F,
    ) -> F {
        match self {
            // In this case, the oracle contains the values of f_0 = g_0
            Oracle::Initial => f_x,

            // In this case, we need to apply degree correction and the quotient
            Oracle::Virtual(virtual_function) => {
                assert!(
                    virtual_function.quotient_set.iter().all(|&q| q != x),
                    "The virtual function is undefined at points in its quotient set"
                );

                let quotient_num = f_x - virtual_function.interpolating_polynomial.evaluate(x);
                let quotient_denom = virtual_function
                    .quotient_set
                    .iter()
                    .map(|q| x - q)
                    .product();
                let quotient_evalution = quotient_num * quotient_denom.inverse();

                let num_terms = virtual_function.quotient_set.len();
                let common_factor = evaluation_point * virtual_function.comb_randomness;

                let scale_factor = if common_factor != F::ONE {
                    (F::ONE - common_factor.exp_u64((num_terms + 1) as u64))
                        * (F::ONE - common_factor).inverse()
                } else {
                    F::from_canonical_usize(num_terms + 1)
                };

                quotient_evaluation * scale_factor
            }
        }
    }
}

/// Input to verify_round
pub struct VerificationState<F: TwoAdicField> {
    // The indices are given in the following frame of reference: Self is
    // produced inside verify_round for round i (in {1, ..., num_rounds}). The
    // final round ("consistency with the final polynomial"), with index
    // num_rounds + 1, does not produce a StirWitness.

    // Oracle used to compute the value of the virtual function f_i
    oracle: Oracle<F>,

    // NP TODO maybe move to the config or somewehre else (this is proof-independent)

    // Domain L_i
    domain: Radix2Coset<F>,

    // Folding randomness r_i to be used in the next round
    folding_randomness: F,

    // Index i in the main loop - starts at 0 with the verification state computed before
    // the first round (round 1)
    round: usize,
}

pub fn verify<F, M, C>(
    config: &StirConfig<F, M>,
    commitment: M::Commitment,
    proof: &StirProof<F, M, C::Witness>,
    challenger: &mut C,
) -> bool
where
    F: TwoAdicField,
    M: Mmcs<F>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    let Proof {
        round_proofs,
        final_polynomial,
        pow_witness,
        final_round_queries,
    } = proof;

    // NP TODO return meaningful verification error

    if final_polynomial.degree() >= 1 << config.log_stopping_degree() {
        return false;
    }

    // NP TODO verify merkle paths (inside main loop instead of separately PLUS final round)
    challenger.observe(&commitment);
    let folding_randomness = challenger.sample_ext_element();

    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    // Cf. prover/mod.rs for an explanation on the chosen sequence of domain
    // sizes
    let domain = Radix2Coset::new(F::two_adic_generator(log_size), log_size);

    let mut verification_state = VerificationState {
        oracle: Oracle::Initial,
        domain,
        folding_randomness,
        round: 0,
    };

    for round_proof in round_proofs {
        verification_state = if let Some(vs) =
            self.verify_round(config, verification_state, round_proof, &mut challenger)
        {
            vs
        } else {
            return false;
        };
    }

    // Step 2: Consistency with final polynomial

    // Logarithm of |(L_M)^k_M|
    let final_log_size = verification_state.domain.log_size() - config.log_last_folding_factor();

    let final_queried_indices: Vec<u64> = (0..config.final_num_queries())
        .map(|_| challenger.sample_bits(final_log_size) as u64)
        .unique()
        .collect();

    // Recover the evaluations of g_M needed to compute the values of f_M at
    // points which are relevant to evaluate p(r_i) = Fold(f_M, ...)(r_i), where
    // r_i runs over the final queried indices
    let final_oracle_answers = proof.queries_to_final.0.clone();

    let folded_answers = self.compute_folded_evaluations(
        &verification_state,
        final_randomness_indexes,
        final_oracle_answers,
    );

    if !folded_answers
        .into_iter()
        .all(|(point, value)| proof.final_polynomial.evaluate(&point) == value)
    {
        return false;
    }

    if !challenger.check_witness(config.final_pow_bits(), pow_witness) {
        return false;
    }

    return true;
}

pub fn verify_round(
    config: &StirConfig<F, M>,
    verification_state: VerificationState<F>,
    round_proof: &RoundProof<F, M, C::Witness>,
    challenger: &mut C,
) -> Option<VerificationState<F>>
where
    F: TwoAdicField,
    M: Mmcs<F>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    // De-structure the round-specific configuration and the verification state
    let RoundConfig {
        log_folding_factor,
        log_next_folding_factor,
        log_evaluation_domain_size,
        pow_bits,
        num_queries,
        num_ood_samples,
        log_inv_rate,
    } = config.round_config(witness.round).clone();

    let VerificationState {
        oracle,
        domain,
        folding_randomness,
        round,
    } = verification_state;

    let RoundProof {
        g_root,
        betas,
        ans_polynomial,
        query_proofs,
        shake_polynomial,
        pow_witness,
    } = round_proof;

    // Update the transcript with the root of the Merkle tree
    challenger.observe(&g_root);

    // Rejection sampling on the out of domain samples
    let mut ood_samples = Vec::new();

    while ood_samples.len() < num_ood_samples {
        let el: F = challenger.sample_ext_element();
        if !new_domain.contains(el) {
            ood_samples.push(el);
        }
    }

    // Observe the betas
    challenger.observe_slice(&betas);

    // Sample ramdomness used for degree correction
    let comb_randomness = challenger.sample_ext_element();

    // Sample folding randomness for the next round
    let new_folding_randomness = challenger.sample_ext_element();

    // Sample queried indices of elements in L_{i - 1}^k_{i-1}
    let log_query_domain_size = domain.log_size() - log_folding_factor;

    let queried_indices: Vec<u64> = (0..num_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size) as u64)
        .unique()
        .collect();

    // Verify proof of work
    if !challenger.check_witness(pow_bits.ceil(), pow_witness) {
        return false;
    }

    let shake_randomness: F = challenger.sample_ext_element();

    // Verify Merkle paths
    for (&i, (leaf, proof)) in queried_indices.iter().unique().zip(query_proofs) {
        if config
            .mmcs_config()
            .verify_batch(
                &g_root,
                &[Dimensions {
                    width: 1 << log_folding_factor,
                    height: 1 << (domain.log_size() - log_folding_factor),
                }],
                i,
                &leaf,
                &proof,
            )
            .is_err()
        {
            return false;
        }
    }

    // The j-th element of this vector is the list of values of g_{i - 1} which
    // result in the list of values of f_{i - 1} (by virtue of f_{i - 1} being
    // a virtual function reliant on g_{i - 1}) which get folded into
    // g_i(r_{i, j}^shift)
    let previous_g_values = queried_indices.into_iter().map(|(leaf, _)| leaf[0]);

    // Compute the values of f_{i - 1} from those of g_{i - 1}
    let previous_f_values =
        previous_g_values.map(|g_list| g_list.map(|g_value| oracle.evaluate(g_value)).collect_vec);

    todo!()
}
