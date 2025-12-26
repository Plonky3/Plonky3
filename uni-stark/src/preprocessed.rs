use p3_air::Air;
use p3_commit::Pcs;
use p3_field::Field;
use p3_matrix::Matrix;
use tracing::debug_span;

use crate::{ProverConstraintFolder, StarkGenericConfig, SymbolicAirBuilder, Val};

/// Prover-side reusable data for preprocessed columns.
///
/// This allows committing to the preprocessed trace once per [`Air`]/degree and reusing
/// the commitment and [`Pcs`] prover data across many proofs.
pub struct PreprocessedProverData<SC: StarkGenericConfig> {
    /// The width (number of columns) of the preprocessed trace.
    pub width: usize,
    /// The log2 of the degree of the domain over which the preprocessed trace is committed.
    ///
    /// In the current uni-stark implementation this matches `degree_bits` in [`Proof`](crate::Proof),
    /// i.e. the (extended) trace degree.
    pub degree_bits: usize,
    /// [`Pcs`] commitment to the preprocessed trace.
    pub commitment: <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
    /// [`Pcs`] prover data for the preprocessed trace.
    pub prover_data: <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::ProverData,
}

/// Verifier-side reusable data for preprocessed columns.
///
/// This allows committing to the preprocessed trace once per [`Air`]/degree and reusing
/// the commitment across many verifications.
#[derive(Clone)]
pub struct PreprocessedVerifierKey<SC: StarkGenericConfig> {
    /// The width (number of columns) of the preprocessed trace.
    pub width: usize,
    /// The log2 of the degree of the domain over which the preprocessed trace is committed.
    ///
    /// This should match `degree_bits` in [`Proof`](crate::Proof), i.e. the (extended) trace degree.
    pub degree_bits: usize,
    /// [`Pcs`] commitment to the preprocessed trace.
    pub commitment: <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
}

/// Set up and commit the preprocessed trace for a given [`Air`] and degree.
///
/// This can be called once per [`Air`]/degree configuration to obtain reusable
/// prover data for preprocessed columns. Returns `None` if the [`Air`] does not
/// define any preprocessed columns.
pub fn setup_preprocessed<SC, A>(
    config: &SC,
    air: &A,
    degree_bits: usize,
) -> Option<(PreprocessedProverData<SC>, PreprocessedVerifierKey<SC>)>
where
    SC: StarkGenericConfig,
    Val<SC>: Field,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    let pcs = config.pcs();
    let is_zk = config.is_zk();

    let init_degree = 1 << degree_bits;
    let degree = 1 << (degree_bits + is_zk);

    let preprocessed = air.preprocessed_trace()?;

    let width = preprocessed.width();
    if width == 0 {
        return None;
    }

    assert_eq!(
        preprocessed.height(),
        init_degree,
        "preprocessed trace height must equal trace degree"
    );

    let trace_domain = pcs.natural_domain_for_degree(degree);
    let (commitment, prover_data) = debug_span!("commit to preprocessed trace")
        .in_scope(|| pcs.commit_preprocessing([(trace_domain, preprocessed)]));

    let degree_bits = degree_bits + is_zk;
    let prover_data = PreprocessedProverData {
        width,
        degree_bits,
        commitment: commitment.clone(),
        prover_data,
    };
    let vk = PreprocessedVerifierKey {
        width,
        degree_bits,
        commitment,
    };
    Some((prover_data, vk))
}
