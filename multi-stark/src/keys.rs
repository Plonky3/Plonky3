//! Proving and verifying keys carrying the reusable preprocessed commitment.
//!
//! The preprocessed trace is fixed by the AIR, not the witness.
//! It is committed once here and reused across every proof for that AIR and trace height.
//!
//! The prover key keeps the committed data so each proof opens it without re-encoding.
//! The verifier key keeps only the commitment.
//! Both sides absorb that commitment before sampling any challenge.

use alloc::vec::Vec;

use p3_air::BaseAir;
use p3_commit::MultilinearPcs;
use p3_sumcheck::layout::Table;

use crate::config::{Commitment, MultiStarkConfig, ProverData};

/// Batched preprocessed data the prover reuses across proofs.
///
/// All preprocessed tables are stacked into one commitment in AIR-instance
/// order, skipping AIRs that have no preprocessed columns.
pub(crate) struct PreprocessedProverData<C: MultiStarkConfig> {
    /// Commitment to the stacked preprocessed tables.
    pub(crate) commitment: Commitment<C>,
    /// Committed prover data behind the batched commitment, cloned per proof to open.
    pub(crate) prover_data: ProverData<C>,
}

/// The prover's key for a fixed AIR and trace height.
pub struct ProvingKey<C: MultiStarkConfig> {
    /// Batched preprocessed data, present only when at least one AIR declares it.
    pub(crate) preprocessed: Option<PreprocessedProverData<C>>,
}

/// The verifier's key for a fixed AIR and trace height.
pub struct VerifyingKey<C: MultiStarkConfig> {
    /// Batched preprocessed commitment, present only when at least one AIR declares it.
    pub(crate) preprocessed: Option<Commitment<C>>,
}

/// Commit all AIR preprocessed traces once, returning matched prover and verifier keys.
///
/// When the AIR declares no preprocessed trace, both keys carry no preprocessed data.
/// The proof then runs exactly as the main-only flow does.
///
/// The AIR order fixed here is the batch order.
/// The prover-side and verifier-side batches must list their instances in this same order.
/// The preprocessed tables are stacked in this order, so a mismatch pairs each instance with the wrong table.
///
/// The commitment is deterministic in the trace.
/// The throwaway challenger therefore only satisfies the commit signature.
/// Its post-state is discarded.
/// Prover and verifier both re-absorb the stored commitment into the real transcript before sampling.
///
/// # Arguments
///
/// - `config`: the proof configuration selecting the preprocessed commitment scheme.
/// - `airs`: AIRs in proof order.
/// - `challenger`: a throwaway transcript used only for its commit side effect.
///
/// # Panics
///
/// Panics if an AIR declares preprocessed columns but does not return a preprocessed trace.
pub fn setup<C, A>(
    config: &C,
    airs: &[&A],
    challenger: &mut C::Challenger,
) -> (ProvingKey<C>, VerifyingKey<C>)
where
    C: MultiStarkConfig,
    A: BaseAir<C::Val>,
    Commitment<C>: Clone,
{
    let mut tables = Vec::new();

    for air in airs.iter().filter(|air| air.preprocessed_width() != 0) {
        let trace = air
            .preprocessed_trace()
            .expect("AIR with preprocessed columns must return a preprocessed trace");

        tables.push(Table::new(trace.transpose()));
    }

    if tables.is_empty() {
        return (
            ProvingKey { preprocessed: None },
            VerifyingKey { preprocessed: None },
        );
    }

    // Turn traces into one multilinear per column, then commit the stacked tables once.
    let witness = config.build_witness(tables);
    let (commitment, prover_data) = config.preprocessed_pcs().commit(witness, challenger);

    // The prover key keeps committed data to open each proof.
    let proving = ProvingKey {
        preprocessed: Some(PreprocessedProverData {
            commitment: commitment.clone(),
            prover_data,
        }),
    };
    // The verifier key keeps only the commitment; shape facts come from AIR metadata.
    let verifying = VerifyingKey {
        preprocessed: Some(commitment),
    };
    (proving, verifying)
}
