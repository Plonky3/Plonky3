//! Security-parameter helpers for a batch-STARK proof.
//!
//! [`p3_uni_stark::num_batched_openings`] counts the `(column, opening point)` pairs a single
//! uni-STARK instance contributes to the polynomial commitment scheme's opening-batching
//! challenge. A batch-STARK proof commits everything that formula already covers per instance,
//! plus a permutation (LogUp) trace on instances that declare lookups, and random-linear-combines
//! all of it — across every instance in the batch — into the same single low-degree-test
//! instance. [`num_batched_openings`] extends the per-instance formula with that permutation term;
//! the caller sums it over the batch to get the count
//! [`p3_uni_stark::StarkSecurityParams::num_batched_functions`] should be set to.

use p3_uni_stark::num_batched_openings as num_batched_openings_uni_stark;

/// Number of `(column, opening point)` pairs one batch-STARK instance contributes to the
/// opening-batching challenge.
///
/// Extends [`p3_uni_stark::num_batched_openings`] with the permutation trace: on an instance with
/// `num_lookups > 0`, `p3_lookup::logup::LogUpGadget::generate_permutation` commits
/// `num_lookups + 1` extension-field columns (one running-sum accumulator plus one fraction per
/// lookup), and `batch-stark`'s prover opens that commitment at both `zeta` and `zeta·g`
/// unconditionally — unlike the main and preprocessed traces, whose second point depends on the
/// AIR's own next-row access (`batch-stark/src/prover.rs`'s `lookup_points` vs. `round1_points`).
/// An instance with `num_lookups == 0` commits no permutation trace and contributes nothing here.
// The list mirrors `p3_uni_stark::num_batched_openings`'s own arguments plus `num_lookups`;
// grouping any of them behind another name would only move the same values one level down.
#[allow(clippy::too_many_arguments)]
pub const fn num_batched_openings(
    main_width: usize,
    main_next: bool,
    preprocessed_width: usize,
    preprocessed_next: bool,
    num_quotient_chunks: usize,
    num_lookups: usize,
    challenge_dimension: usize,
    is_zk: bool,
) -> usize {
    let base = num_batched_openings_uni_stark(
        main_width,
        main_next,
        preprocessed_width,
        preprocessed_next,
        num_quotient_chunks,
        challenge_dimension,
        is_zk,
    );
    let permutation = if num_lookups > 0 {
        2 * (num_lookups + 1) * challenge_dimension
    } else {
        0
    };
    base + permutation
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matches `p3_uni_stark`'s own count when no instance has lookups.
    #[test]
    fn num_batched_openings_matches_uni_stark_without_lookups() {
        assert_eq!(
            num_batched_openings(100, true, 0, false, 2, 0, 4, false),
            num_batched_openings_uni_stark(100, true, 0, false, 2, 4, false),
        );
    }

    /// Lookups add `2 * (num_lookups + 1) * challenge_dimension`: eight lookups over a degree-4
    /// extension is one accumulator plus eight fractions, nine columns, opened at two points.
    #[test]
    fn num_batched_openings_counts_the_permutation_trace() {
        let base = num_batched_openings_uni_stark(100, true, 0, false, 2, 4, false);
        assert_eq!(
            num_batched_openings(100, true, 0, false, 2, 8, 4, false),
            base + 2 * 9 * 4,
        );
    }

    /// An instance with no lookups contributes no permutation term regardless of the other shape
    /// parameters.
    #[test]
    fn num_batched_openings_zero_lookups_contributes_nothing() {
        assert_eq!(
            num_batched_openings(100, true, 6, true, 2, 0, 4, true),
            num_batched_openings_uni_stark(100, true, 6, true, 2, 4, true),
        );
    }
}
