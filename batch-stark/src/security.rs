//! Security-parameter helpers for a batch-STARK proof.
//!
//! [`p3_uni_stark::num_batched_openings`] counts the powers a single uni-STARK instance
//! contributes to the polynomial commitment scheme's opening-batching
//! challenge. A batch-STARK proof commits everything that formula already covers per instance,
//! plus a permutation (LogUp) trace on instances that declare lookups, and random-linear-combines
//! all of it — across every instance in the batch — into the same single low-degree-test
//! instance. [`num_batched_openings`] extends the per-instance formula with that permutation term;
//! the caller sums it over the batch to get the count
//! [`p3_uni_stark::StarkSecurityParams::num_batched_functions`] should be set to.

use p3_uni_stark::{OpeningShape, num_batched_openings as num_batched_openings_uni_stark};

/// Number of batching powers one batch-STARK instance contributes to the opening-batching
/// challenge. Circle consumes two powers per `(column, opening point)` pair.
///
/// Extends [`p3_uni_stark::num_batched_openings`] with the permutation trace: on an instance with
/// `num_lookups > 0`, `p3_lookup::logup::LogUpGadget::generate_permutation` commits
/// `num_lookups + 1` extension-field columns (one running-sum accumulator plus one fraction per
/// lookup), and `batch-stark`'s prover opens that commitment at both `zeta` and `zeta·g`
/// unconditionally — unlike the main and preprocessed traces, whose second point depends on the
/// AIR's own next-row access (`batch-stark/src/prover.rs`'s `lookup_points` vs. `round1_points`).
/// An instance with `num_lookups == 0` commits no permutation trace and contributes nothing here.
/// Hiding FRI appends `openings.num_random_codewords()` base-field columns to each permutation
/// matrix, which are opened at both points as well.
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
    openings: OpeningShape,
) -> usize {
    let base = num_batched_openings_uni_stark(
        main_width,
        main_next,
        preprocessed_width,
        preprocessed_next,
        num_quotient_chunks,
        challenge_dimension,
        openings,
    );
    let permutation = if num_lookups > 0 {
        2 * ((num_lookups + 1) * challenge_dimension + openings.num_random_codewords())
            * openings.powers_per_opening()
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
            num_batched_openings(100, true, 0, false, 2, 0, 4, OpeningShape::new()),
            num_batched_openings_uni_stark(100, true, 0, false, 2, 4, OpeningShape::new()),
        );
    }

    /// Lookups add `2 * (num_lookups + 1) * challenge_dimension`: eight lookups over a degree-4
    /// extension is one accumulator plus eight fractions, nine columns, opened at two points.
    #[test]
    fn num_batched_openings_counts_the_permutation_trace() {
        let base = num_batched_openings_uni_stark(100, true, 0, false, 2, 4, OpeningShape::new());
        assert_eq!(
            num_batched_openings(100, true, 0, false, 2, 8, 4, OpeningShape::new()),
            base + 2 * 9 * 4,
        );
    }

    /// An instance with no lookups contributes no permutation term regardless of the other shape
    /// parameters.
    #[test]
    fn num_batched_openings_zero_lookups_contributes_nothing() {
        assert_eq!(
            num_batched_openings(100, true, 6, true, 2, 0, 4, OpeningShape::hiding(0)),
            num_batched_openings_uni_stark(100, true, 6, true, 2, 4, OpeningShape::hiding(0)),
        );
    }

    #[test]
    fn num_batched_openings_counts_circle_permutation_powers() {
        // 200 trace + 8 quotient + 72 permutation openings, each consuming two powers.
        assert_eq!(
            num_batched_openings(100, true, 0, false, 2, 8, 4, OpeningShape::Circle),
            560,
        );
    }
}
