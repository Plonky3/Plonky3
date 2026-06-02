//! Transcript schema and oracle handle for the HVZK sumcheck.

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use p3_zk_codes::ZkEncoding;

/// Per-round prover output of the HVZK sumcheck protocol.
///
/// - Prover writes;
/// - Verifier reads back during Fiat-Shamir replay.
///
/// One instance covers a full run of `k` rounds.
///
/// # Wire format
///
/// Per round, the polynomial has coefficient layout
///
/// ```text
///     [ c_0, c_1, c_2, ..., c_d ]    with  d = max(ell_zk - 1, 2)
/// ```
///
/// The linear coefficient `c_1` is dropped on the wire.
///
/// The verifier reconstructs `c_1` from the affine identity
///
/// ```text
///     h_j(0) + h_j(1) = 2 * c_0 + sum_{i >= 1} c_i = target
/// ```
///
/// applied to the previous round's target.
///
/// # Soundness link to Lemma 6.4
///
/// Valid transcripts form an affine subspace of dimension `1 + k * (ell_zk - 1)`.
/// The `k` dropped linear coefficients are exactly the redundant degrees of freedom of the rank-nullity argument.
#[derive(Debug, Clone)]
pub struct ZkSumcheckData<F, EF> {
    /// Sum of all mask polynomial evaluations across the boolean hypercube `{0,1}^k`.
    ///
    /// Observed on the transcript before the verifier samples the combining challenge.
    /// Lives in the extension field because the mask coefficients do.
    pub mu_tilde: EF,

    /// Message length of the zero-knowledge mask code.
    ///
    /// The verifier rejects up front if its own expected value disagrees with this.
    /// Pinning this in the transcript closes a non-injectivity gap in the wire-length check: lengths `2` and `3` share a wire layout.
    pub ell_zk: usize,

    /// Per-round wire payload with the linear coefficient dropped.
    ///
    /// One entry per sumcheck round.
    /// Layout per entry: `[c_0, c_2, c_3, ..., c_d]` with `d = max(ell_zk - 1, 2)`.
    pub round_coefficients: Vec<Vec<EF>>,

    /// Per-round proof-of-work witnesses.
    ///
    /// Length equals the number of rounds when grinding is enabled.
    /// Empty when `pow_bits == 0`.
    pub pow_witnesses: Vec<F>,
}

impl<F, EF: Field> Default for ZkSumcheckData<F, EF> {
    fn default() -> Self {
        Self {
            // Real runs overwrite this in step 2 once the prover has summed the masks.
            mu_tilde: EF::ZERO,
            // Sentinel: honest runs set this to the encoding's message length; the verifier rejects 0.
            ell_zk: 0,
            // Filled with one wire entry per sumcheck round.
            round_coefficients: Vec::new(),
            // Filled only when grinding is enabled.
            pow_witnesses: Vec::new(),
        }
    }
}

/// Handle to one encoded mask codeword.
///
/// Pairs the public Merkle root with the prover-side data needed to open the codeword at requested positions.
/// Downstream consumers use this during the codeswitch step to produce opening proofs against the mask oracles.
pub type MaskOracle<EF, Enc, M> = (
    <M as Mmcs<EF>>::Commitment,
    <M as Mmcs<EF>>::ProverData<<Enc as ZkEncoding<EF>>::Codeword>,
);
