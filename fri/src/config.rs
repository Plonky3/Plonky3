use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_security::fri::FriRegime;

/// A set of parameters defining a specific instance of the FRI protocol.
#[derive(Clone, Debug)]
pub struct FriParameters<M> {
    pub log_blowup: usize,
    // TODO: This parameter and FRI early stopping are not yet implemented in `CirclePcs`.
    pub log_final_poly_len: usize,
    /// Maximum folding arity (log2). 1 = binary folding.
    /// The actual arity per round may be smaller to ensure commitments exist at each input height.
    pub max_log_arity: usize,
    pub num_queries: usize,
    /// Number of bits for the PoW phase before sampling _each_ batching challenge.
    pub commit_proof_of_work_bits: usize,
    /// Number of bits for the PoW phase before sampling the queries.
    pub query_proof_of_work_bits: usize,
    pub mmcs: M,
}

impl<M> FriParameters<M> {
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }

    pub const fn final_poly_len(&self) -> usize {
        1 << self.log_final_poly_len
    }

    pub const fn max_arity(&self) -> usize {
        1 << self.max_log_arity
    }

    /// Returns the soundness bits of this FRI instance based on the
    /// [ethSTARK](https://eprint.iacr.org/2021/582) conjecture.
    ///
    /// Certain users may instead want to look at proven soundness, a more complex calculation which
    /// isn't currently supported by this crate.
    pub const fn conjectured_soundness_bits(&self) -> usize {
        self.log_blowup * self.num_queries + self.query_proof_of_work_bits
    }

    /// Assemble the [`FriRegime`] mirror consumed by `p3-security` for
    /// soundness analysis.
    ///
    /// The exhaustive destructuring is deliberate: adding a field to
    /// `FriParameters` breaks this method until the new field is either
    /// mapped into [`FriRegime`] or explicitly ignored, so the runtime config
    /// and the soundness model cannot drift apart silently.
    pub const fn security_regime(&self) -> FriRegime {
        let Self {
            log_blowup,
            log_final_poly_len,
            max_log_arity,
            num_queries,
            commit_proof_of_work_bits,
            query_proof_of_work_bits,
            mmcs: _,
        } = self;
        FriRegime {
            log_blowup: *log_blowup,
            num_queries: *num_queries,
            log_final_poly_len: *log_final_poly_len,
            max_log_arity: *max_log_arity,
            commit_pow_bits: *commit_proof_of_work_bits,
            query_pow_bits: *query_proof_of_work_bits,
        }
    }

    /// Creates a minimal set of `FriParameters` for testing purposes.
    /// These parameters are designed to reduce computational cost during tests.
    pub const fn new_testing(mmcs: M, log_final_poly_len: usize) -> Self {
        Self {
            log_blowup: 2,
            log_final_poly_len,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 1,
            query_proof_of_work_bits: 1,
            mmcs,
        }
    }

    /// Creates a minimal set of `FriParameters` for testing purposes, with zk enabled.
    /// These parameters are designed to reduce computational cost during tests.
    pub const fn new_testing_zk(mmcs: M) -> Self {
        Self {
            log_blowup: 2,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 1,
            query_proof_of_work_bits: 1,
            mmcs,
        }
    }

    /// Creates a set of `FriParameters` suitable for benchmarking.
    /// These parameters represent typical settings used in production-like scenarios.
    pub const fn new_benchmark(mmcs: M) -> Self {
        Self {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 100,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 16,
            mmcs,
        }
    }

    /// Creates a set of `FriParameters` suitable for benchmarking with high arity.
    /// These parameters represent typical settings used in production-like scenarios.
    pub const fn new_benchmark_high_arity(mmcs: M) -> Self {
        Self {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 3,
            num_queries: 100,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 16,
            mmcs,
        }
    }

    /// Creates a set of `FriParameters` suitable for benchmarking with zk enabled.
    /// These parameters represent typical settings used in production-like scenarios.
    pub const fn new_benchmark_zk(mmcs: M) -> Self {
        Self {
            log_blowup: 2,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 100,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 16,
            mmcs,
        }
    }
}

/// Whereas `FriParameters` encompasses parameters the end user can set, `FriFoldingStrategy` is
/// set by the PCS calling FRI, and abstracts over implementation details of the PCS.
pub trait FriFoldingStrategy<F: Field, EF: ExtensionField<F>> {
    type InputProof;
    type InputError: Debug;

    /// We can ask FRI to sample extra query bits (LSB) for our own purposes.
    /// They will be passed to our callbacks, but ignored (shifted off) by FRI.
    fn extra_query_index_bits(&self) -> usize;

    /// Fold a row with the specified arity, returning a single value.
    /// The input row has `2^log_arity` elements.
    fn fold_row(
        &self,
        index: usize,
        log_height: usize,
        log_arity: usize,
        beta: EF,
        evals: impl Iterator<Item = EF>,
    ) -> EF;

    /// Fold an entire matrix with the specified arity.
    /// The matrix has width `2^log_arity` and the result has length `matrix.height()`.
    fn fold_matrix<M: Matrix<EF>>(&self, beta: EF, log_arity: usize, m: M) -> Vec<EF>;
}

/// Computes the log_arity for the current round.
///
/// Given the current log_height, the next input's log_height (if any), the log of the
/// final target height, and the maximum allowed log_arity, returns the actual log_arity
/// to use for this round.
///
/// This ensures we always commit at each input height level and don't go past the final
/// target height.
#[inline]
pub fn compute_log_arity_for_round(
    log_current_height: usize,
    next_input_log_height: Option<usize>,
    log_final_height: usize,
    max_log_arity: usize,
) -> usize {
    assert!(
        max_log_arity > 0,
        "max_log_arity must be at least 1 to guarantee folding progress"
    );
    debug_assert!(
        log_current_height > log_final_height,
        "should only be called when above final height"
    );

    let max_fold_to_target = log_current_height - log_final_height;

    let max_fold = next_input_log_height.map_or(max_fold_to_target, |next_log_height| {
        debug_assert!(
            log_current_height > next_log_height,
            "next input height should be strictly smaller"
        );
        let max_fold_to_next = log_current_height - next_log_height;
        max_fold_to_next.min(max_fold_to_target)
    });

    max_fold.min(max_log_arity)
}

/// Derive the whole folding schedule before any folding happens.
///
/// # Overview
///
/// A round's arity is capped by three things:
///
/// - distance left to the final height,
/// - distance down to where the next input joins,
/// - the configured maximum.
///
/// All three are known before folding starts.
/// So the schedule is known too, and neither side reads it from a proof.
///
/// # Arguments
///
/// - `input_log_heights`: log-heights of the folding inputs, strictly decreasing.
/// - `log_final_height`: log-height at which folding stops.
/// - `max_log_arity`: largest arity any single round may use.
///
/// # Returns
///
/// One log-arity per commit round, in round order.
///
/// Empty when nothing sits above the final height.
/// A verifier derives this from untrusted heights, so it returns rather than panics.
///
/// # Panics
///
/// When the input heights are not strictly decreasing.
#[must_use]
pub fn fold_schedule(
    input_log_heights: &[usize],
    log_final_height: usize,
    max_log_arity: usize,
) -> Vec<usize> {
    assert!(
        input_log_heights.windows(2).all(|pair| pair[0] > pair[1]),
        "input log-heights must be strictly decreasing",
    );

    // Folding starts at the tallest input and stops at the final height.
    let Some(&tallest) = input_log_heights.first() else {
        return Vec::new();
    };
    if tallest <= log_final_height {
        return Vec::new();
    }
    let mut log_height = tallest;

    // Index of the next input still waiting to be rolled in.
    let mut next_input = 1;
    let mut schedule = Vec::new();

    while log_height > log_final_height {
        // Cap by the final height, the next input, and the configured maximum.
        let log_arity = compute_log_arity_for_round(
            log_height,
            input_log_heights.get(next_input).copied(),
            log_final_height,
            max_log_arity,
        );
        schedule.push(log_arity);
        log_height -= log_arity;

        // An input whose height the fold just reached is rolled in here.
        // The next round then folds the combined codeword.
        if input_log_heights.get(next_input) == Some(&log_height) {
            next_input += 1;
        }
    }

    schedule
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn schedule_folds_the_whole_way_down() {
        // Invariant: the arities sum to the distance the codeword has to travel.
        //
        // Fixture state: one input at height 2^10, folding down to 2^2.
        //
        //     10 -> 2 is 8 levels, in steps of at most 3
        //     8 = 3 + 3 + 2
        assert_eq!(fold_schedule(&[10], 2, 3), vec![3, 3, 2]);

        // Binary folding takes one level per round.
        assert_eq!(fold_schedule(&[10], 2, 1), vec![1; 8]);
    }

    #[test]
    fn schedule_pauses_where_an_input_joins() {
        // Invariant: a round never folds past the height of the next input.
        //
        // An input joins only once the codeword reaches its height.
        // A round that overshot would leave it nowhere to join.
        //
        // Fixture state: inputs at 2^10 and 2^6, final height 2^2, max arity 3.
        //
        //     10 --3--> 7 --1--> 6   <- input joins here
        //      6 --3--> 3 --1--> 2   <- final height
        assert_eq!(fold_schedule(&[10, 6], 2, 3), vec![3, 1, 3, 1]);
    }

    #[test]
    fn schedule_is_empty_without_inputs() {
        // Boundary: nothing to fold means no commit rounds.
        assert_eq!(fold_schedule(&[], 2, 3), Vec::<usize>::new());
    }

    #[test]
    fn every_schedule_lands_exactly_on_the_final_height() {
        // Invariant: folding never overshoots or stops short.
        //
        // Overshooting loses the evaluations the query phase needs.
        for max_log_arity in 1..=4 {
            for log_final_height in 0..4 {
                for tall in (log_final_height + 1)..12 {
                    // A second input somewhere strictly between the two ends.
                    for short in (log_final_height + 1)..tall {
                        let schedule =
                            fold_schedule(&[tall, short], log_final_height, max_log_arity);
                        assert_eq!(
                            schedule.iter().sum::<usize>(),
                            tall - log_final_height,
                            "max_log_arity={max_log_arity} final={log_final_height} \
                             inputs=[{tall}, {short}]",
                        );
                        // No round may exceed the configured maximum.
                        assert!(schedule.iter().all(|&a| a <= max_log_arity && a > 0));
                    }
                }
            }
        }
    }

    /// Pins the field-by-field mapping in [`FriParameters::security_regime`].
    /// Distinct values per field catch a mis-wired mapping (e.g. swapping the
    /// commit and query PoW bits); the method's exhaustive destructuring is
    /// what catches a newly-added `FriParameters` field.
    #[test]
    fn security_regime_mirrors_parameters() {
        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 2,
            max_log_arity: 3,
            num_queries: 4,
            commit_proof_of_work_bits: 5,
            query_proof_of_work_bits: 6,
            mmcs: (),
        };

        let regime = params.security_regime();

        assert_eq!(regime.log_blowup, 1);
        assert_eq!(regime.log_final_poly_len, 2);
        assert_eq!(regime.max_log_arity, 3);
        assert_eq!(regime.num_queries, 4);
        assert_eq!(regime.commit_pow_bits, 5);
        assert_eq!(regime.query_pow_bits, 6);
    }
}
