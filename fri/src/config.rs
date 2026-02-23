use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;

/// A set of parameters defining a specific instance of the FRI protocol.
#[derive(Clone, Debug)]
pub struct FriParameters<M> {
    pub log_blowup: usize,
    // TODO: This parameter and FRI early stopping are not yet implemented in `CirclePcs`.
    pub log_final_poly_len: usize,
    /// Maximum folding arity (log2). Defaults to 1 (binary folding).
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

/// Creates a minimal set of `FriParameters` for testing purposes.
/// These parameters are designed to reduce computational cost during tests.
pub const fn create_test_fri_params<Mmcs>(
    mmcs: Mmcs,
    log_final_poly_len: usize,
) -> FriParameters<Mmcs> {
    FriParameters {
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
pub const fn create_test_fri_params_zk<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
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
pub const fn create_benchmark_fri_params<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 100,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 16,
        mmcs,
    }
}

/// Creates a set of `FriParameters` suitable for benchmarking.
/// These parameters represent typical settings used in production-like scenarios.
pub const fn create_benchmark_fri_params_high_arity<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
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
pub const fn create_benchmark_fri_params_zk<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
        log_blowup: 2,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 100,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 16,
        mmcs,
    }
}
