use p3_air::Air;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use tracing::debug_span;

use crate::{
    ProverConstraintFolder, StarkGenericConfig, SymbolicAirBuilder, Val,
    get_log_num_quotient_chunks,
};

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
    /// Preprocessed evaluations on the quotient domain, materialized at setup time
    /// to avoid recomputing the PCS view on every proof.
    pub quotient_domain_evals: RowMajorMatrix<Val<SC>>,
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
///
/// `num_public_values` is needed to compute the constraint degree, which determines
/// the quotient domain on which preprocessed evaluations are precomputed and cached.
pub fn setup_preprocessed<SC, A>(
    config: &SC,
    air: &A,
    degree_bits: usize,
    num_public_values: usize,
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

    let log_num_quotient_chunks =
        get_log_num_quotient_chunks::<Val<SC>, A>(air, width, num_public_values, is_zk);
    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (degree_bits + log_num_quotient_chunks));

    let quotient_domain_evals = debug_span!("materialize preprocessed on quotient domain")
        .in_scope(|| {
            pcs.get_evaluations_on_domain_no_random(&prover_data, 0, quotient_domain)
                .to_row_major_matrix()
        });

    let prover_data = PreprocessedProverData {
        width,
        degree_bits,
        commitment: commitment.clone(),
        prover_data,
        quotient_domain_evals,
    };
    let vk = PreprocessedVerifierKey {
        width,
        degree_bits,
        commitment,
    };
    Some((prover_data, vk))
}
