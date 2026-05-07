//! Per-round prover state for the WHIR protocol.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_multilinear_util::point::Point;
use tracing::instrument;

use crate::constraints::statement::initial::InitialStatement;
use crate::fiat_shamir::errors::FiatShamirError;
use crate::sumcheck::SumcheckData;
use crate::sumcheck::single::SingleSumcheck;
use crate::sumcheck::strategy::SumcheckProver;

/// Per-round state during WHIR proof generation.
///
/// Tracks the sumcheck prover, folding randomness, and Merkle
/// commitments across base and extension field rounds.
#[derive(Debug)]
pub struct RoundState<EF, F, BaseData, ExtData>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Sumcheck prover managing constraint batching and polynomial folding.
    pub sumcheck_prover: SumcheckProver<F, EF>,
    /// Folding challenges (alpha_1, ..., alpha_k) for the current round.
    pub folding_randomness: Point<EF>,
    /// Merkle commitment for the base field polynomial (initial round).
    pub commitment_merkle_prover_data: BaseData,
    /// Merkle commitment for folded extension field polynomials (rounds > 0).
    pub merkle_prover_data: Option<ExtData>,
}

impl<EF, F, BaseData, ExtData> RoundState<EF, F, BaseData, ExtData>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Initialize the first round state from the committed polynomial.
    ///
    /// Runs the initial sumcheck (if constraints exist) or samples
    /// folding randomness directly from the transcript.
    #[instrument(skip_all)]
    pub fn initialize_first_round_state<Challenger>(
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        statement: &InitialStatement<F, EF>,
        commitment_merkle_prover_data: BaseData,
        folding_factor: usize,
        pow_bits: usize,
    ) -> Result<Self, FiatShamirError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (sumcheck_prover, folding_randomness) = SingleSumcheck::new(
            sumcheck_data,
            challenger,
            folding_factor,
            pow_bits,
            statement,
        );

        Ok(Self {
            sumcheck_prover,
            folding_randomness,
            commitment_merkle_prover_data,
            merkle_prover_data: None,
        })
    }
}
