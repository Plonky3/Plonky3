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
use p3_matrix::Matrix;
use p3_uni_stark::{PreprocessedProverData, ProverConstraintFolder, SymbolicAirBuilder, Val};
use p3_util::log2_strict_usize;

use crate::config::{Challenge, StarkGenericConfig as SGC};
use crate::prover::StarkInstance;

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
    /// Create `CommonData` with no preprocessed columns.
    ///
    /// Use this when none of your AIRs have preprocessed columns.
    pub fn empty(num_instances: usize) -> Self {
        Self {
            preprocessed: (0..num_instances).map(|_| None).collect(),
        }
    }
}

impl<SC> CommonData<SC>
where
    SC: SGC,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    /// Build `CommonData` directly from STARK instances.
    ///
    /// This automatically:
    /// - Derives trace degrees from trace heights
    /// - Computes extended degrees (base + ZK padding)
    /// - Sets up preprocessed columns for AIRs that define them
    pub fn from_instances<A>(config: &SC, instances: &[StarkInstance<'_, SC, A>]) -> Self
    where
        A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>> + Copy,
    {
        let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
        let log_ext_degrees: Vec<usize> = degrees
            .iter()
            .map(|&d| log2_strict_usize(d) + config.is_zk())
            .collect();
        let airs: Vec<A> = instances.iter().map(|i| *i.air).collect();
        Self::from_airs_and_degrees(config, &airs, &log_ext_degrees)
    }

    /// Build `CommonData` from AIRs and their extended trace degree bits.
    ///
    /// # Arguments
    ///
    /// * `trace_ext_degree_bits` - Log2 of extended trace degrees (including ZK padding)
    ///
    /// # Returns
    ///
    /// Preprocessed data for each AIR. Entries are `None` for AIRs without preprocessed columns.
    pub fn from_airs_and_degrees<A>(
        config: &SC,
        airs: &[A],
        trace_ext_degree_bits: &[usize],
    ) -> Self
    where
        A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
    {
        assert_eq!(
            airs.len(),
            trace_ext_degree_bits.len(),
            "airs and trace_ext_degree_bits must have the same length"
        );

        let preprocessed = airs
            .iter()
            .zip(trace_ext_degree_bits.iter())
            .map(|(air, &ext_db)| {
                let base_db = ext_db - config.is_zk();
                p3_uni_stark::setup_preprocessed::<SC, _>(config, air, base_db)
            })
            .collect();

        Self { preprocessed }
    }
}
