//! Shared data between batch-STARK prover and verifier.
//!
//! This module is intended to store per-instance data that is common to both
//! proving and verification, such as lookup tables and preprocessed traces.
//!
//! The preprocessed support integrates with `p3-uni-stark`'s transparent
//! preprocessed columns API.

use alloc::vec::Vec;

use p3_air::Air;
use p3_field::BasedVectorSpace;
use p3_uni_stark::{PreprocessedProverData, ProverConstraintFolder, SymbolicAirBuilder, Val};

use crate::config::{Challenge, StarkGenericConfig as SGC};

/// Struct storing data common to both the prover and verifier.
///
/// TODO: Add lookup metadata (e.g. `Vec<Vec<Lookup<Val<SC>>>>`).
pub struct CommonData<SC: SGC> {
    /// Optional preprocessed prover data for each STARK instance.
    ///
    /// There is one entry per STARK instance, in the same order as the
    /// `StarkInstance`s provided to batch proving / verifying.
    pub preprocessed: Vec<Option<PreprocessedProverData<SC>>>,
}

impl<SC: SGC> CommonData<SC> {
    /// Construct a `CommonData` with only preprocessed information.
    pub const fn new(preprocessed: Vec<Option<PreprocessedProverData<SC>>>) -> Self {
        Self { preprocessed }
    }
}

impl<SC> CommonData<SC>
where
    SC: SGC,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    /// Build `CommonData` from a list of AIRs and their extended degrees.
    ///
    /// This will:
    /// - For each instance, call `setup_preprocessed` with the base (non-ZK) trace degree
    ///   derived from `ext_degree_bits` and `config.is_zk()`.
    /// - Store the resulting `PreprocessedProverData` (if any) in `preprocessed`.
    ///
    /// This is a convenience helper; callers that want to cache preprocessed data across
    /// proofs can precompute it once and reuse the resulting `CommonData`.
    pub fn from_airs_and_degrees<A>(config: &SC, airs: &[A], ext_degree_bits: &[usize]) -> Self
    where
        A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
    {
        assert_eq!(
            airs.len(),
            ext_degree_bits.len(),
            "airs and ext_degree_bits must have the same length"
        );

        let preprocessed = airs
            .iter()
            .zip(ext_degree_bits.iter())
            .map(|(air, &ext_db)| {
                let base_db = ext_db
                    .checked_sub(config.is_zk())
                    .expect("ext_degree_bits must be >= is_zk()");
                p3_uni_stark::setup_preprocessed::<SC, _>(config, air, base_db)
            })
            .collect();

        Self { preprocessed }
    }
}
