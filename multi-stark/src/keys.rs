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
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use crate::commit::trace_to_columns;
use crate::config::{Commitment, MultiStarkConfig, ProverData};

/// Preprocessed data the prover reuses across proofs.
///
/// The data is committed once at setup.
/// Each proof clones it to open the columns at that proof's bound point.
pub struct PreprocessedProverData<C: MultiStarkConfig> {
    /// The preprocessed trace, one column per preprocessed AIR column.
    ///
    /// Retained so the zerocheck round state can fold it alongside the main trace.
    pub(crate) trace: RowMajorMatrix<C::Val>,
    /// Commitment to the stacked preprocessed columns.
    pub(crate) commitment: Commitment<C>,
    /// Committed prover data behind the commitment, cloned per proof to open.
    pub(crate) prover_data: ProverData<C>,
    /// Number of preprocessed columns.
    pub(crate) width: usize,
    /// Preprocessed columns whose next row the AIR reads.
    pub(crate) next_columns: Vec<usize>,
    /// Base-two logarithm of the preprocessed trace height.
    pub(crate) log_height: usize,
}

/// The prover's key for a fixed AIR and trace height.
pub struct ProvingKey<C: MultiStarkConfig> {
    /// Preprocessed data, present only when the AIR declares preprocessed columns.
    pub(crate) preprocessed: Option<PreprocessedProverData<C>>,
}

/// Preprocessed data the verifier reuses across proofs.
pub struct PreprocessedVerifierData<C: MultiStarkConfig> {
    /// Commitment to the stacked preprocessed columns.
    pub(crate) commitment: Commitment<C>,
    /// Number of preprocessed columns.
    pub(crate) width: usize,
    /// Preprocessed columns whose next row the AIR reads.
    pub(crate) next_columns: Vec<usize>,
    /// Base-two logarithm of the preprocessed trace height.
    pub(crate) log_height: usize,
}

/// The verifier's key for a fixed AIR and trace height.
pub struct VerifyingKey<C: MultiStarkConfig> {
    /// Preprocessed data, present only when the AIR declares preprocessed columns.
    pub(crate) preprocessed: Option<PreprocessedVerifierData<C>>,
}

/// Commit the AIR's preprocessed trace once, returning the matched prover and verifier keys.
///
/// When the AIR declares no preprocessed trace, both keys carry no preprocessed data.
/// The proof then runs exactly as the main-only flow does.
///
/// The commitment is deterministic in the trace.
/// The throwaway challenger therefore only satisfies the commit signature.
/// Its post-state is discarded.
/// Prover and verifier both re-absorb the stored commitment into the real transcript before sampling.
///
/// # Arguments
///
/// - `config`: the proof configuration selecting the preprocessed commitment scheme.
/// - `air`: the AIR whose preprocessed trace is committed.
/// - `challenger`: a throwaway transcript used only for its commit side effect.
///
/// # Panics
///
/// Panics if the preprocessed trace height is not a power of two.
pub fn setup<C, A>(
    config: &C,
    air: &A,
    challenger: &mut C::Challenger,
) -> (ProvingKey<C>, VerifyingKey<C>)
where
    C: MultiStarkConfig,
    A: BaseAir<C::Val>,
    Commitment<C>: Clone,
{
    // An AIR without a preprocessed trace yields empty keys, so the proof stays main-only.
    let Some(trace) = air.preprocessed_trace() else {
        return (
            ProvingKey { preprocessed: None },
            VerifyingKey { preprocessed: None },
        );
    };

    // Shape facts both keys record so prover and verifier build matching opening protocols.
    let width = trace.width;
    let log_height = log2_strict_usize(trace.height());
    let next_columns = air.preprocessed_next_row_columns();

    // Turn the trace into one multilinear per column, then commit the stacked columns once.
    let columns = trace_to_columns(&trace);
    let witness = config.build_witness(columns);
    let (commitment, prover_data) = config.preprocessed_pcs().commit(witness, challenger);

    // The prover key keeps the trace to fold and the committed data to open each proof.
    let proving = ProvingKey {
        preprocessed: Some(PreprocessedProverData {
            trace,
            commitment: commitment.clone(),
            prover_data,
            width,
            next_columns: next_columns.clone(),
            log_height,
        }),
    };
    // The verifier key keeps only the commitment and the shape facts.
    let verifying = VerifyingKey {
        preprocessed: Some(PreprocessedVerifierData {
            commitment,
            width,
            next_columns,
            log_height,
        }),
    };
    (proving, verifying)
}
