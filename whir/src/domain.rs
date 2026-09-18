//! Evaluation-domain operations used by the non-hiding WHIR prover.

use p3_commit::Encoder;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;

/// Selector coordinates for one queried codeword position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhirQueryPoint<F> {
    /// A univariate point expanded through successive squares.
    Univariate(F),
    /// Direct coordinates in the code's multilinear basis.
    Multilinear(Point<F>),
}

/// A nested linear code and its evaluation-point map.
///
/// WHIR only needs nested power-of-two domains. Implementations are statically
/// dispatched, so choosing a domain adds no virtual calls to the prover.
pub trait WhirDomain<F, EF>: Encoder<F> + Sync
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Stable identifier bound into the Fiat--Shamir instance label.
    ///
    /// The identifier must cover the code and query-point map.
    ///
    /// The empty identifier is reserved for the legacy two-adic transcript.
    fn protocol_id(&self) -> &'static [u8];

    /// Whether queries use the canonical power-of-two stratified schedule.
    fn stratified_queries(&self) -> bool {
        false
    }

    /// Largest supported base-two logarithm of a domain size.
    fn max_log_domain_size(&self) -> usize;

    /// Encodes an already-padded extension-field coefficient matrix.
    fn encode_extension_batch_padded(
        &self,
        message: RowMajorMatrix<EF>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<EF>;

    /// Returns selector coordinates for a queried codeword position.
    fn query_point(
        &self,
        log_domain_size: usize,
        num_variables: usize,
        index: usize,
    ) -> WhirQueryPoint<F>;
}

impl<F, EF, Dft> WhirDomain<F, EF> for Dft
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F> + Sync,
{
    fn protocol_id(&self) -> &'static [u8] {
        b""
    }

    fn max_log_domain_size(&self) -> usize {
        F::TWO_ADICITY
    }

    fn encode_extension_batch_padded(
        &self,
        message: RowMajorMatrix<EF>,
        _log_inv_rate: usize,
    ) -> RowMajorMatrix<EF> {
        self.dft_algebra_batch(message)
    }

    fn query_point(
        &self,
        log_domain_size: usize,
        _num_variables: usize,
        index: usize,
    ) -> WhirQueryPoint<F> {
        WhirQueryPoint::Univariate(F::two_adic_generator(log_domain_size).exp_u64(index as u64))
    }
}
