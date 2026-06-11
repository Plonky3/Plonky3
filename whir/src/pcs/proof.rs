use alloc::vec;
use alloc::vec::Vec;

use p3_commit::MultiOpeningMmcs;
use p3_matrix::{Dimensions, Matrix};
use p3_multilinear_util::poly::Poly;
pub use p3_sumcheck::SumcheckData;
use serde::{Deserialize, Serialize};

/// Complete WHIR proof.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct WhirProof<F: Send + Sync + Clone, EF, MT: MultiOpeningMmcs<F>> {
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
    /// Final round STIR query openings.
    pub final_openings: Option<QueryOpenings<F, EF, MT::MultiProof>>,
    /// Final sumcheck data (if `final_sumcheck_rounds > 0`).
    pub final_sumcheck: Option<SumcheckData<F, EF>>,
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: MultiOpeningMmcs<F>> Default
    for WhirProof<F, EF, MT>
{
    fn default() -> Self {
        Self {
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            rounds: Vec::new(),
            final_poly: None,
            final_pow_witness: F::default(),
            final_openings: None,
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
/// - The public opening evaluations indexed by batch then by column:
///
/// ```text
///     evals[i][j]  =  value of the j-th opened column in the i-th batch
/// ```
///
/// # Ordering invariant
///
/// The batches appear in the same order as the public opening schedule, so
/// the verifier can walk both side-by-side without re-sorting. A length
/// mismatch on either axis causes the adapter to reject before any Merkle
/// or sumcheck check runs.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct PcsProof<F: Send + Sync + Clone, EF, MT: MultiOpeningMmcs<F>> {
    /// Proximity transcript: initial commitment, sumcheck rounds, per-round
    /// commitments, STIR query openings, and the final polynomial.
    pub whir: WhirProof<F, EF, MT>,
    /// Outer index walks opening batches in schedule order; inner index walks
    /// the columns opened inside each batch in their requested order.
    pub evals: Vec<Vec<EF>>,
}

/// Per-round proof data.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct WhirRoundProof<F: Send + Sync + Clone, EF, MT: MultiOpeningMmcs<F>> {
    /// Round commitment (Merkle root).
    pub commitment: Option<MT::Commitment>,
    /// OOD evaluations for this round.
    pub ood_answers: Vec<EF>,
    /// PoW witness after commitment.
    pub pow_witness: F,
    /// STIR query openings against the previous round's commitment.
    pub openings: Option<QueryOpenings<F, EF, MT::MultiProof>>,
    /// Sumcheck data for this round.
    pub sumcheck: SumcheckData<F, EF>,
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: MultiOpeningMmcs<F>> Default
    for WhirRoundProof<F, EF, MT>
{
    fn default() -> Self {
        Self {
            commitment: None,
            ood_answers: Vec::new(),
            pow_witness: F::default(),
            openings: None,
            sumcheck: SumcheckData::default(),
        }
    }
}

/// Rows opened at many queried positions, plus one proof covering them all.
///
/// One multiproof authenticates every row together,
/// so sibling digests shared between queries travel once.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "T: Serialize, P: Serialize",
    deserialize = "T: Deserialize<'de>, P: Deserialize<'de>"
))]
pub struct MultiOpening<T, P> {
    /// `rows[q]` is the opened leaf row at the `q`-th queried position.
    pub rows: Vec<Vec<T>>,
    /// Compact multiproof authenticating every row at once.
    pub proof: P,
}

impl<T: Send + Sync + Clone, P> MultiOpening<T, P> {
    /// Verifies the rows against `commit` at the verifier-derived `indices`.
    ///
    /// The multiproof binds each row to its index,
    /// so a row/index count mismatch or a substituted leaf is rejected.
    pub(crate) fn verify<MT>(
        &self,
        mmcs: &MT,
        commit: &MT::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
    ) -> Result<(), MT::Error>
    where
        MT: MultiOpeningMmcs<T, MultiProof = P>,
    {
        let opened_values: Vec<Vec<Vec<T>>> =
            self.rows.iter().map(|row| vec![row.clone()]).collect();
        mmcs.verify_multi_batch(commit, dimensions, indices, &opened_values, &self.proof)
    }
}

/// Opens a single-matrix commitment and packages the rows as a [`MultiOpening`].
///
/// Lets callers write `mmcs.open_rows(indices, data)` instead of threading the
/// mmcs through a free constructor.
pub(crate) trait OpenMultiRows<T: Send + Sync + Clone>: MultiOpeningMmcs<T> {
    /// Opens `indices` on a commitment holding exactly one matrix.
    fn open_rows<M: Matrix<T>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> MultiOpening<T, Self::MultiProof>;
}

impl<T: Send + Sync + Clone, MT: MultiOpeningMmcs<T>> OpenMultiRows<T> for MT {
    fn open_rows<M: Matrix<T>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> MultiOpening<T, Self::MultiProof> {
        let (values, proof) = self.open_multi_batch(indices, prover_data);
        let rows = values
            .into_iter()
            .map(|mut per_matrix| {
                // WHIR commits a single matrix per round, so each query opens one row.
                assert_eq!(
                    per_matrix.len(),
                    1,
                    "WHIR opens commitments holding exactly one matrix"
                );
                per_matrix.swap_remove(0)
            })
            .collect();
        MultiOpening { rows, proof }
    }
}

/// Field-tagged multi-opening for one queried oracle.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum QueryOpenings<F, EF, P> {
    /// Base-field rows (the initial commitment).
    Base(MultiOpening<F, P>),
    /// Extension-field rows (every folded round commitment).
    Extension(MultiOpening<EF, P>),
}

impl<F: Default + Send + Sync + Clone, EF: Default, MT: MultiOpeningMmcs<F>> WhirProof<F, EF, MT> {
    /// Allocate an empty proof sized for the given intermediate-round count.
    pub(crate) fn empty(num_rounds: usize) -> Self {
        Self {
            initial_ood_answers: Vec::new(),
            initial_sumcheck: SumcheckData::default(),
            // One default round-proof slot per intermediate WHIR round.
            rounds: (0..num_rounds).map(|_| WhirRoundProof::default()).collect(),
            final_poly: None,
            final_pow_witness: F::default(),
            final_openings: None,
            final_sumcheck: None,
        }
    }
}

impl<F: Clone + Send + Sync + Default, EF, MT: MultiOpeningMmcs<F>> WhirProof<F, EF, MT> {
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
