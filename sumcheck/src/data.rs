use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use p3_field::ExtensionField;
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use crate::SumcheckError;
use crate::strategy::Basis;
use crate::transcript::{ProverTranscript, SumcheckShape, VerifierTranscript};

/// Sumcheck polynomial data
///
/// Stores the polynomial evaluations for sumcheck rounds in a compact format.
/// Each round stores two evaluations: `[h(0), h(inf)]` in the evaluation
/// basis (`h(1)` derived as `claimed_sum - h(0)`), or `[s(1), s(inf)]` in the
/// projective basis (`s(0)` derived as `claimed_sum - s(inf)`).
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct SumcheckData<F, EF> {
    /// Polynomial evaluations for each sumcheck round.
    ///
    /// Each entry is `[h(0), h(inf)]`:
    /// - `h(0)` is the constant term.
    /// - `h(inf)` is the leading coefficient (evaluation at infinity).
    ///
    /// `h(1)` is derived as `claimed_sum - h(0)` by the verifier.
    ///
    /// Length: folding_factor
    pub polynomial_evaluations: Vec<[EF; 2]>,

    /// PoW witnesses for each sumcheck round
    /// Length: folding_factor
    pub pow_witnesses: Vec<F>,
}

impl<F, EF> SumcheckData<F, EF> {
    /// Returns the polynomial evaluations `[h(0), h(inf)]` for each round.
    #[must_use]
    pub fn polynomial_evaluations(&self) -> &[[EF; 2]] {
        &self.polynomial_evaluations
    }

    /// Returns the number of rounds stored in this proof data.
    #[must_use]
    pub const fn num_rounds(&self) -> usize {
        self.polynomial_evaluations.len()
    }

    /// Records one round in this proof and plays the matching transcript step.
    ///
    /// This is the only place a round reaches both the proof and the sponge.
    /// Recording and absorbing therefore cannot drift apart.
    ///
    /// # Arguments
    ///
    /// * `transcript` - driver of the batch of rounds this one belongs to.
    /// * `c_a` - finite-point value: `h(0)` (evaluation) or `s(1)` (projective).
    /// * `c_inf` - leading coefficient `h(inf)` / `s(inf)`.
    ///
    /// # Returns
    ///
    /// The sampled challenge `r`.
    ///
    /// The two values are recorded and absorbed verbatim.
    /// Which basis defines them is fixed by the shape the driver was built from.
    ///
    /// # Panics
    ///
    /// When the batch has already played every round it was described with.
    pub fn observe_and_sample<Challenger>(
        &mut self,
        transcript: &mut ProverTranscript<'_, Challenger, F, EF>,
        c_a: EF,
        c_inf: EF,
    ) -> EF
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
    {
        // Absorb the pair, do the optional grinding, and take this round's challenge.
        let (challenge, witness) = transcript.round(c_a, c_inf);

        // Record what the round produced alongside what it bound.
        //
        // Two of the quadratic's three values cross the wire; the verifier
        // derives the third from the basis-dependent round identity.
        self.polynomial_evaluations.push([c_a, c_inf]);
        self.pow_witnesses.extend(witness);

        challenge
    }

    /// Verifies standard sumcheck rounds and extracts folding randomness from the transcript.
    ///
    /// # Arguments
    ///
    /// * `challenger` - sponge of the surrounding protocol, borrowed for the batch.
    /// * `claimed_sum` - Running claim, folded in place to `h(r)` after each round.
    /// * `expected_rounds` - Protocol-fixed number of rounds this proof must carry.
    /// * `pow_bits` - PoW difficulty (0 to skip grinding).
    /// * `basis` - how the two transmitted values are read.
    ///
    /// # Returns
    ///
    /// The folding randomness, one challenge per round.
    ///
    /// # Shape checks
    ///
    /// Both counts this proof carries are attacker-controlled.
    /// Both are checked before any transcript work.
    ///
    /// The round count decides how many steps the batch is described with.
    /// The witness count is what the round loop indexes into.
    ///
    /// Taking either from the proof would let a wrong one desynchronise Fiat-Shamir.
    /// Both are therefore compared against the caller's own configuration instead.
    ///
    /// # Errors
    ///
    /// - The proof does not carry exactly `expected_rounds` rounds.
    /// - The witness count is not the one the difficulty implies.
    /// - A round carries a witness that misses the required difficulty.
    pub fn verify_rounds<Challenger>(
        &self,
        challenger: &mut Challenger,
        claimed_sum: &mut EF,
        expected_rounds: usize,
        pow_bits: usize,
        basis: Basis,
    ) -> Result<Point<EF>, SumcheckError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Bind the round count to the protocol-fixed value before folding anything.
        //
        // Centralizing the check here makes the safe path the only path:
        // every caller is forced to declare how many rounds it expects.
        if self.polynomial_evaluations.len() != expected_rounds {
            return Err(SumcheckError::RoundCountMismatch {
                expected: expected_rounds,
                actual: self.polynomial_evaluations.len(),
            });
        }

        // Canonical proof shape — every accepting proof has a unique form:
        // - zero difficulty requires an empty witness vector,
        // - positive difficulty requires exactly one witness per round.
        //
        // The loop below indexes the witness vector, so this also keeps it in bounds.
        let expected_witnesses = if pow_bits > 0 { expected_rounds } else { 0 };
        if self.pow_witnesses.len() != expected_witnesses {
            return Err(SumcheckError::PowWitnessCountMismatch {
                expected: expected_witnesses,
                actual: self.pow_witnesses.len(),
            });
        }

        // Seeded from the same numbers the prover seeded with.
        let shape = SumcheckShape::new(expected_rounds, pow_bits, basis);
        let mut transcript = VerifierTranscript::<Challenger, F, EF>::new(challenger, shape);

        let mut randomness = Vec::with_capacity(expected_rounds);

        // Driven by the same number the description was built from, not by the proof's length.
        //
        // The two agree only because of the round-count check above.
        // Reading the count once keeps the loop and the description from ever disagreeing:
        //
        //     too few iterations  -> steps left unplayed, and closing the transcript panics
        //     too many            -> a step past the end of the description, which panics
        //
        // Both indices below are in bounds by the two checks above.
        for round in 0..expected_rounds {
            let [c_a, c_inf] = self.polynomial_evaluations[round];

            // One call binds both values, re-checks the grind, and draws the challenge.
            //
            // A rejection here releases the driver's completeness check on its way out.
            let witness = (pow_bits > 0).then(|| self.pow_witnesses[round]);
            let r = transcript.round(c_a, c_inf, witness)?;

            // Reconstruct h(r); the basis-dependent round identity supplies the
            // third quadratic value (shared with the prover via `Basis::reduce_claim`).
            *claimed_sum = basis.reduce_claim(c_a, c_inf, r, *claimed_sum);
            randomness.push(r);
        }

        // Require that every described step was played.
        transcript.finish();

        Ok(Point::new(randomness))
    }
}

/// Verify the final sumcheck rounds.
///
/// This is a free function because a run of no rounds may carry no sumcheck data at all.
///
/// # Returns
///
/// The folding randomness, one challenge per round.
///
/// # Errors
///
/// - The run is described with rounds but carries no sumcheck data.
/// - Any rejection the round replay itself raises.
pub fn verify_final_sumcheck_rounds<F, EF, Challenger>(
    final_sumcheck: Option<&SumcheckData<F, EF>>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
    basis: Basis,
) -> Result<Point<EF>, SumcheckError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if rounds == 0 {
        return Ok(Point::new(Vec::new()));
    }

    let sumcheck = final_sumcheck.ok_or(SumcheckError::MissingSumcheckData {
        expected_rounds: rounds,
    })?;

    // `verify_rounds` binds the round count to `rounds`.
    sumcheck.verify_rounds(challenger, claimed_sum, rounds, pow_bits, basis)
}
