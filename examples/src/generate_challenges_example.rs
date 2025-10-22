//! Example demonstrating the usage of the `generate_challenges` function
//! for recursion in Plonky3.

use p3_air::Air;
use p3_uni_stark::{StarkGenericConfig, VerificationChallenges, generate_challenges};

/// Example function showing how to use `generate_challenges` for recursion.
///
/// This function demonstrates how to generate all challenges that would be
/// produced during verification without actually executing the verifier circuit.
/// This is useful for recursion where we need to know all challenge values
/// before starting the actual verification process.
pub fn example_generate_challenges<SC, A>(
    config: &SC,
    air: &A,
    proof: &p3_uni_stark::Proof<SC>,
    public_values: &[p3_uni_stark::Val<SC>],
    degree_bits: usize,
    log_quotient_degree: usize,
    trace_domain: <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::Domain,
) -> VerificationChallenges<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<p3_uni_stark::Val<SC>>>,
{
    // Generate all challenges that would be produced during verification
    generate_challenges(
        config,
        air,
        proof,
        public_values,
        degree_bits,
        log_quotient_degree,
        trace_domain,
    )
}

/// Example showing how the challenges can be used in a recursive context.
pub fn example_recursive_usage<SC, A>(
    config: &SC,
    air: &A,
    proof: &p3_uni_stark::Proof<SC>,
    public_values: &[p3_uni_stark::Val<SC>],
    degree_bits: usize,
    log_quotient_degree: usize,
    trace_domain: <SC::Pcs as p3_commit::Pcs<SC::Challenge, SC::Challenger>>::Domain,
) where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<p3_uni_stark::Val<SC>>>,
{
    // Step 1: Generate all challenges first
    let challenges = generate_challenges(
        config,
        air,
        proof,
        public_values,
        degree_bits,
        log_quotient_degree,
        trace_domain,
    );

    // Step 2: Use the challenges in your recursive circuit
    // For example, you might want to:
    // - Use challenges.alpha in constraint evaluation
    // - Use challenges.zeta for opening points
    // - Use challenges.zeta_next for next row evaluations
    // - Use challenges.fri_challenges for FRI verification

    // This allows you to build a recursive verifier circuit that knows
    // all the challenge values upfront, which is essential for recursion.

    println!("Generated challenges:");
    println!("  Alpha: {:?}", challenges.alpha);
    println!("  Zeta: {:?}", challenges.zeta);
    println!("  Zeta next: {:?}", challenges.zeta_next);

    if let Some(fri_challenges) = &challenges.fri_challenges {
        println!("  FRI Alpha: {:?}", fri_challenges.alpha);
        println!("  FRI Betas: {:?}", fri_challenges.betas);
        println!("  Query indices: {:?}", fri_challenges.query_indices);
    }
}
