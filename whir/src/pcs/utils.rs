use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_util::log2_strict_usize;

use crate::fiat_shamir::errors::FiatShamirError;

/// Sample cryptographically secure STIR query indices from the transcript.
///
/// - Draws `num_queries` random indices in `[0, folded_domain_size)`,
/// - Then sorts and deduplicates them.
///
/// The returned vector may therefore contain **fewer** than `num_queries` entries when collisions occur.
///
/// # Soundness note
///
/// The WHIR shift-query soundness bound is `(1 - δ)^t` where `t` is the number
/// of *distinct* query positions (Theorem 5.2, ε^shift). Duplicate queries test
/// the same codeword position twice and contribute no additional soundness.
/// Because `folded_domain_size` is typically ≥ 2^18 and `num_queries` = O(λ)
/// with λ ≤ 128, the birthday-bound collision probability is negligible
/// (~t² / 2n ≈ 2^{-4}), so dedup almost never reduces the effective count.
///
/// TODO: consider switching to rejection sampling (sample-without-replacement)
/// to guarantee exactly `num_queries` distinct indices and tighten the
/// soundness accounting. The current approach relies on the collision
/// probability being negligible, which should be validated for small domains
/// or high query counts.
///
/// # Batching strategy
///
/// When possible, multiple query indices are extracted from a single
/// `challenger.sample_bits` call to reduce Fiat-Shamir overhead:
///
/// - If all bits fit in one call (≤ `max_bits_per_call`), a single sample
///   suffices.
/// - Otherwise, indices are batched into groups that fit within the per-call
///   bit budget.
/// - As a fallback, one transcript call is made per query.
///
/// # Panics
///
/// Panics if `domain_size` is not a power of two.
pub fn get_challenge_stir_queries<Challenger, F, EF>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    challenger: &mut Challenger,
) -> Result<Vec<usize>, FiatShamirError>
where
    Challenger: FieldChallenger<F>,
    F: Field,
    EF: ExtensionField<F>,
{
    // Apply folding to get the reduced domain size.
    let folded_domain_size = domain_size >> folding_factor;
    // Bits needed to index the folded domain.
    let domain_size_bits = log2_strict_usize(folded_domain_size);

    // Conservative limit to avoid statistical bias.
    let max_bits_per_call = (F::bits() - 1).min(20);

    let total_bits_needed = num_queries * domain_size_bits;
    let mut queries = Vec::with_capacity(num_queries);

    if total_bits_needed <= max_bits_per_call {
        // All bits fit in a single transcript call.
        let mut all_bits = challenger.sample_bits(total_bits_needed);
        let mask = (1 << domain_size_bits) - 1;

        for _ in 0..num_queries {
            let query_bits = all_bits & mask;
            queries.push(query_bits % folded_domain_size);
            all_bits >>= domain_size_bits;
        }
    } else {
        let queries_per_batch = max_bits_per_call / domain_size_bits;

        if queries_per_batch >= 2 {
            // Batch multiple queries per transcript call.
            let mut remaining = num_queries;
            let mask = (1 << domain_size_bits) - 1;

            while remaining > 0 {
                let batch_size = remaining.min(queries_per_batch);
                let batch_bits = batch_size * domain_size_bits;

                let mut all_bits = challenger.sample_bits(batch_bits);

                for _ in 0..batch_size {
                    let query_index = (all_bits & mask) % folded_domain_size;
                    queries.push(query_index);
                    all_bits >>= domain_size_bits;
                }

                remaining -= batch_size;
            }
        } else {
            // Fallback: one transcript call per query.
            for _ in 0..num_queries {
                let value = challenger.sample_bits(domain_size_bits);
                queries.push(value);
            }
        }
    }

    queries.sort_unstable();
    queries.dedup();

    Ok(queries)
}
