use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSample, CanSampleUniformBits, GrindingChallenger};
use p3_commit::{Mmcs, MmcsArith};
use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::point::Point;
use p3_sumcheck::constraints::statement::EqStatement;
use p3_verifier_arith::Lift;

use crate::pcs::proof::WhirProof;
use crate::pcs::verifier::errors::VerifierError;
use crate::transcript::WhirVerifierTranscript;

/// Parsed commitment extracted from the verifier's transcript.
///
/// Contains the Merkle root and the OOD equality constraints
/// needed for verification.
#[derive(Debug, Clone)]
pub struct ParsedCommitment<F, D> {
    /// Merkle root of the committed evaluation table.
    pub root: D,
    /// OOD challenge points and their claimed evaluations.
    pub ood_statement: EqStatement<F>,
}

impl<F: PrimeCharacteristicRing, D> ParsedCommitment<F, D> {
    /// Parse a commitment for a specific round, within an arithmetic.
    ///
    /// # Errors
    ///
    /// - Round index is past the last round carried by the proof.
    /// - Round entry is present but its Merkle-root slot is empty.
    pub fn parse_with_round<A, MT, Challenger>(
        arith: &mut A,
        mmcs: &MT,
        proof: &WhirProof<A::Val, A::Challenge, MT>,
        transcript: &mut WhirVerifierTranscript<'_, Challenger, A::Val, A::Challenge>,
        num_variables: usize,
        ood_samples: usize,
        round_index: usize,
    ) -> Result<Self, VerifierError>
    where
        A: Lift<Val: TranscriptField, EF = F>,
        MT: Mmcs<A::Val>
            + MmcsArith<A, A::F, Commitment = <MT as Mmcs<A::Val>>::Commitment, Target = D>,
        Challenger: CanObserve<A::Val>
            + CanSample<A::Val>
            + CanSampleUniformBits<A::Val>
            + GrindingChallenger<Witness = A::Val>
            + CanObserve<<MT as Mmcs<A::Val>>::Commitment>,
    {
        let round_proof = proof
            .rounds
            .get(round_index)
            .ok_or(VerifierError::InvalidRoundIndex { index: round_index })?;
        let root = round_proof
            .commitment
            .clone()
            .ok_or(VerifierError::MissingRoundCommitment { round: round_index })?;
        let ood_answers = &round_proof.ood_answers;
        if ood_answers.len() != ood_samples {
            return Err(VerifierError::RoundOodAnswerCountMismatch {
                round: round_index,
                expected: ood_samples,
                actual: ood_answers.len(),
            });
        }

        // Observe the Merkle root in the transcript.
        let target = mmcs.load_commitment(arith, &root);
        mmcs.will_observe(arith, &target);
        transcript.commitment(root);

        // Reconstruct equality constraints from OOD challenge points and answers.
        //
        // The count was pinned against the configuration above.
        // Every index below is therefore in bounds, and every draw is described.
        let mut ood_statement = EqStatement::initialize(num_variables);
        for &answer in ood_answers {
            let challenge = arith.sampled_ext(transcript.ood_point());
            let point = Point::expand_from_univariate(challenge, num_variables);
            let eval = arith.load_ext(answer);
            arith.will_observe_ext(&eval);
            transcript.ood_answer(answer);
            ood_statement.add_evaluated_constraint(point, eval);
        }

        Ok(Self {
            root: target,
            ood_statement,
        })
    }
}
