//! Example demonstrating the usage of the `generate_challenges` function
//! for recursion in Plonky3.

use p3_air::Air;
use p3_uni_stark::{generate_challenges, StarkGenericConfig, VerificationChallenges};
use p3_field::Field;

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
    public_values: &Vec<p3_uni_stark::Val<SC>>,
) -> VerificationChallenges<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<p3_uni_stark::Val<SC>>>,
{
    // Generate all challenges that would be produced during verification
    let challenges = generate_challenges(config, air, proof, public_values);
    
    // Now we have access to all the challenges:
    // - challenges.alpha: The first Fiat-Shamir challenge used to combine constraint polynomials
    // - challenges.zeta: The out-of-domain point to open values at
    // - challenges.zeta_next: The next point after zeta in the trace domain
    
    // These challenges can be used in recursion scenarios where we need to
    // know the challenge values before executing the verifier circuit.
    
    challenges
}

/// Example showing how the challenges can be used in a recursive context.
pub fn example_recursive_usage<SC, A>(
    config: &SC,
    air: &A,
    proof: &p3_uni_stark::Proof<SC>,
    public_values: &Vec<p3_uni_stark::Val<SC>>,
) where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<p3_uni_stark::Val<SC>>>,
{
    // Step 1: Generate all challenges first
    let challenges = generate_challenges(config, air, proof, public_values);
    
    // Step 2: Use the challenges in your recursive circuit
    // For example, you might want to:
    // - Use challenges.alpha in constraint evaluation
    // - Use challenges.zeta for opening points
    // - Use challenges.zeta_next for next row evaluations
    
    // This allows you to build a recursive verifier circuit that knows
    // all the challenge values upfront, which is essential for recursion.
    
    println!("Generated challenges:");
    println!("  Alpha: {:?}", challenges.alpha);
    println!("  Zeta: {:?}", challenges.zeta);
    println!("  Zeta next: {:?}", challenges.zeta_next);
}
