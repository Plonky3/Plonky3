use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::OpeningBatch;
pub use p3_sumcheck::SumcheckData;
use serde::{Deserialize, Serialize};

/// Complete WHIR proof.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Initial OOD evaluations.
    pub initial_ood_answers: Vec<EF>,
    /// Initial sumcheck data.
    pub initial_sumcheck: SumcheckData<F, EF>,
    /// Per-round proofs.
    pub rounds: Vec<WhirRoundProof<F, EF, MT>>,
    /// Final polynomial coefficients (sent in the clear).
    pub final_poly: Option<Poly<EF>>,
    /// Final round PoW witness.
    pub final_pow_witness: F,
    /// Final round query openings.
    pub final_queries: Vec<QueryOpening<F, EF, MT::Proof>>,
    /// Final sumcheck data (if `final_sumcheck_rounds > 0`).
    pub final_sumcheck: Option<SumcheckData<F, EF>>,
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: Mmcs<F>> Default for WhirProof<F, EF, MT> {
    fn default() -> Self {
        Self {
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: Vec::new(),
            final_poly: None,
            final_pow_witness: F::default(),
            final_queries: Vec::new(),
            final_sumcheck: None,
        }
    }
}

/// Public opening proof produced by the WHIR PCS adapter.
///
/// # Layout
///
/// Two pieces travel together so the verifier can replay the protocol from
/// a single proof object:
///
/// - The proximity transcript: sumcheck rounds, intermediate commitments,
///   STIR query openings, and the final polynomial sent in the clear.
/// - The public opening evaluations indexed by batch.
///   Each batch keeps the evaluations at the current opening points separate
///   from the evaluations at the repeat-last successor points.
///
/// # Ordering invariant
///
/// The batches appear in the same order as the public opening schedule, so
/// the verifier can walk both side-by-side without re-sorting. A length
/// mismatch on either axis causes the adapter to reject before any Merkle
/// or sumcheck check runs.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct PcsProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Proximity transcript: initial commitment, sumcheck rounds, per-round
    /// commitments, STIR query openings, and the final polynomial.
    pub whir: WhirProof<F, EF, MT>,
    /// Opening evaluations in schedule order.
    /// Each batch stores the current-point evaluations separately from the
    /// repeat-last successor-point evaluations.
    pub evals: Vec<OpeningBatch<EF>>,
}

/// Per-round proof data.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirRoundProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Round commitment (Merkle root).
    pub commitment: Option<MT::Commitment>,
    /// OOD evaluations for this round.
    pub ood_answers: Vec<EF>,
    /// PoW witness after commitment.
    pub pow_witness: F,
    /// STIR query openings.
    pub queries: Vec<QueryOpening<F, EF, MT::Proof>>,
    /// Sumcheck data for this round.
    pub sumcheck: SumcheckData<F, EF>,
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: Mmcs<F>> Default
    for WhirRoundProof<F, EF, MT>
{
    fn default() -> Self {
        Self {
            commitment: None,
            ood_answers: Vec::new(),
            pow_witness: F::default(),
            queries: Vec::new(),
            sumcheck: SumcheckData::default(),
        }
    }
}

/// Merkle opening for a single query position.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(
    bound(
        serialize = "F: Serialize, EF: Serialize, Proof: Serialize",
        deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, Proof: Deserialize<'de>"
    ),
    tag = "type"
)]
pub enum QueryOpening<F, EF, Proof> {
    /// Base field opening (initial round).
    #[serde(rename = "base")]
    Base { values: Vec<F>, proof: Proof },
    /// Extension field opening (subsequent rounds).
    #[serde(rename = "extension")]
    Extension { values: Vec<EF>, proof: Proof },
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: Mmcs<F>> WhirProof<F, EF, MT> {
    /// Allocate an empty proof sized for the given intermediate-round and final-query counts.
    pub(crate) fn empty(num_rounds: usize, num_queries: usize) -> Self {
        Self {
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            // One default round-proof slot per intermediate WHIR round.
            rounds: (0..num_rounds).map(|_| WhirRoundProof::default()).collect(),
            final_poly: None,
            final_pow_witness: F::default(),
            // Reserve space for the final-round STIR query openings.
            final_queries: Vec::with_capacity(num_queries),
            final_sumcheck: None,
        }
    }
}

impl<F: Clone + Send + Sync + Default, EF, MT: Mmcs<F>> WhirProof<F, EF, MT> {
    /// Retrieve the PoW witness at a given round index.
    pub(crate) fn get_pow_after_commitment(&self, round_index: usize) -> Option<F> {
        self.rounds
            .get(round_index)
            .map(|round| round.pow_witness.clone())
    }

    /// Store sumcheck data for a specific round.
    pub(crate) fn set_sumcheck_data_at(&mut self, data: SumcheckData<F, EF>, round_index: usize) {
        self.rounds[round_index].sumcheck = data;
    }

    /// Store the final sumcheck data.
    pub(crate) fn set_final_sumcheck_data(&mut self, data: SumcheckData<F, EF>) {
        self.final_sumcheck = Some(data);
    }
}
