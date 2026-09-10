use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_security::fri::FriRegime;
use p3_security::grinding::GrindingSites;

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
    /// Number of bits for the PoW phase before sampling the challenge that
    /// batches the openings into a single FRI instance.
    ///
    /// This is the `alpha` sampled by [`crate::TwoAdicFriPcs`]'s `Pcs::open`,
    /// which random-linear-combines every `(f(zeta) - f(x))/(zeta - x)` quotient
    /// into the one function FRI is run on. Its round's error grows with the
    /// number of batched `(function, opening point)` pairs, so on a wide
    /// instance over a small field it — not the query phase — is what limits
    /// *proven* soundness; grinding here is what lets a query count be traded
    /// down against a proven target. Credited to
    /// [`p3_security::GrindingSites::batch_combination`].
    pub batch_proof_of_work_bits: usize,
    /// Number of bits for the PoW phase before sampling _each_ commit-phase
    /// folding challenge.
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
            // The opening-batching round belongs to the PCS wrapped around
            // FRI, not to the low-degree test, so it is reported through
            // `grinding_sites` instead of this regime. `FriRegime` models only
            // rounds the LDT itself owns.
            batch_proof_of_work_bits: _,
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

    /// The grinding sites these parameters own that [`FriRegime`] does not
    /// model, in the form `p3-security`'s composite consumes.
    ///
    /// Companion to [`Self::security_regime`]: together the two cover every
    /// proof-of-work phase in this config. Sites belonging to the surrounding
    /// STARK rather than to the PCS — the DEEP out-of-domain point, the lookup
    /// challenges — are not visible here and are merged in by the caller (see
    /// `p3_uni_stark::StarkSecurityParams`).
    pub const fn grinding_sites(&self) -> GrindingSites {
        GrindingSites {
            batch_combination: self.batch_proof_of_work_bits,
            ..GrindingSites::NONE
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
            // Keep the minimal test configuration cheap. Tests exercising the batch
            // grinding site set its difficulty explicitly.
            batch_proof_of_work_bits: 0,
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
            // See `new_testing`.
            batch_proof_of_work_bits: 0,
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
            // Callers with wide opening batches can enable this site to raise the
            // batching bound without changing the query-phase difficulty.
            batch_proof_of_work_bits: 0,
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
            batch_proof_of_work_bits: 10,
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
            batch_proof_of_work_bits: 10,
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
    fn every_derived_schedule_satisfies_the_two_invariants_the_verifier_leans_on() {
        // Invariant: two properties of the derivation stand in for two runtime checks.
        //
        //     sum(schedule) == tallest - final   no height cross-check needed
        //     1 <= entry <= cap                  no arity bound needed
        //
        // A verifier that derives its schedule has nothing left to compare against.
        //
        // So both properties are swept exhaustively rather than spot-checked.
        //
        // Fixture state: every strictly-decreasing set of heights drawn from 0..10.
        //
        //     bitmask 0b0000000101  ->  heights [2, 0]
        //     bitmask 0b1000000000  ->  heights [9]
        //
        // Enumerating subsets by bitmask covers every input count at once.
        //
        // It also covers heights below the final height, which never join the fold.
        for max_log_arity in 1..=4 {
            for log_final_height in 0..4 {
                for mask in 1u32..(1 << 10) {
                    // Set bits become heights, tallest first.
                    //
                    // The list is therefore strictly decreasing by construction.
                    let input_log_heights: Vec<usize> =
                        (0..10).rev().filter(|bit| mask >> bit & 1 == 1).collect();

                    let schedule =
                        fold_schedule(&input_log_heights, log_final_height, max_log_arity);

                    // A tallest input at or below the final height folds nothing.
                    //
                    // There is then no round to make a claim about.
                    let tallest = input_log_heights[0];
                    if tallest <= log_final_height {
                        assert!(
                            schedule.is_empty(),
                            "nothing to fold, yet a round was scheduled: \
                             heights={input_log_heights:?} final={log_final_height}",
                        );
                        continue;
                    }

                    // Property 1: the arities account for the distance travelled.
                    //
                    // So the global height needs only one derivation.
                    //
                    // A second one, plus a check that the two agree, would add nothing.
                    assert_eq!(
                        schedule.iter().sum::<usize>(),
                        tallest - log_final_height,
                        "schedule does not land on the final height: \
                         heights={input_log_heights:?} final={log_final_height} \
                         cap={max_log_arity}",
                    );

                    // Property 2: every round folds by one bit or more, up to the cap.
                    //
                    // This is what makes a per-round arity bound unnecessary.
                    assert!(
                        schedule
                            .iter()
                            .all(|&arity| (1..=max_log_arity).contains(&arity)),
                        "schedule leaves the arity bounds: {schedule:?} cap={max_log_arity}",
                    );
                }
            }
        }
    }

    /// Pins the field-by-field mapping in [`FriParameters::security_regime`].
    /// Distinct values per field catch a mis-wired mapping (e.g. swapping the
    /// commit and query PoW bits); the method's exhaustive destructuring is
    /// what catches a newly-added `FriParameters` field.
    ///
    /// The batch PoW bits are deliberately absent from the regime and are
    /// pinned by [`grinding_sites_carries_the_batch_pow_bits`] instead.
    #[test]
    fn security_regime_mirrors_parameters() {
        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 2,
            max_log_arity: 3,
            num_queries: 4,
            batch_proof_of_work_bits: 7,
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

    /// The batch PoW bits reach `p3-security` through `grinding_sites`, and
    /// through nothing else: the two mappings must together account for every
    /// PoW field, without either one double-counting a site the other already
    /// carries.
    #[test]
    fn grinding_sites_carries_the_batch_pow_bits() {
        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 2,
            max_log_arity: 3,
            num_queries: 4,
            batch_proof_of_work_bits: 7,
            commit_proof_of_work_bits: 5,
            query_proof_of_work_bits: 6,
            mmcs: (),
        };

        assert_eq!(
            params.grinding_sites(),
            GrindingSites {
                batch_combination: 7,
                ..GrindingSites::NONE
            },
            "only the batch site belongs to these parameters"
        );

        // A config that grinds nowhere maps to the neutral element, so the
        // sites can be merged into a protocol's own without altering it.
        let ungrounded = FriParameters {
            batch_proof_of_work_bits: 0,
            ..params
        };
        assert_eq!(ungrounded.grinding_sites(), GrindingSites::NONE);
    }
}
