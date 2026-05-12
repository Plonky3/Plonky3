use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use crate::sumcheck::{SumcheckError, extrapolate_01inf};

/// Sumcheck polynomial data
///
/// Stores the polynomial evaluations for sumcheck rounds in a compact format.
/// Each round stores `[h(0), h(inf)]` where `h(1)` is derived as `claimed_sum - h(0)`.
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

    /// Commits polynomial coefficients to the transcript and returns a challenge.
    ///
    /// This helper function handles the Fiat-Shamir interaction for a sumcheck round.
    ///
    /// # Arguments
    ///
    /// * `challenger` - Fiat-Shamir transcript.
    /// * `c0` - Constant coefficient `h(0)`.
    /// * `c_inf` - Leading coefficient `h(inf)`.
    /// * `pow_bits` - PoW difficulty (0 to skip grinding).
    ///
    /// # Returns
    ///
    /// The sampled challenge `r`.
    pub fn observe_and_sample<Challenger, BF>(
        &mut self,
        challenger: &mut Challenger,
        c0: EF,
        c_inf: EF,
        pow_bits: usize,
    ) -> EF
    where
        BF: Field,
        EF: ExtensionField<BF>,
        F: Clone,
        Challenger: FieldChallenger<BF> + GrindingChallenger<Witness = F>,
    {
        // Record the polynomial coefficients in the proof.
        self.polynomial_evaluations.push([c0, c_inf]);

        // Absorb coefficients into the transcript.
        //
        // We send (h(0), h(inf)). The verifier derives h(1) from the sum constraint.
        challenger.observe_algebra_slice(&[c0, c_inf]);

        // Optional proof-of-work to increase prover cost.
        //
        // This makes it expensive for a malicious prover to "mine" favorable challenges.
        if pow_bits > 0 {
            self.pow_witnesses.push(challenger.grind(pow_bits));
        }

        // Sample the verifier's challenge for this round.
        challenger.sample_algebra_element()
    }

    /// Verifies standard sumcheck rounds and extracts folding randomness from the transcript.
    ///
    /// # Returns
    ///
    /// A `Point` of folding randomness values.
    pub fn verify_rounds<Challenger>(
        &self,
        challenger: &mut Challenger,
        claimed_sum: &mut EF,
        pow_bits: usize,
    ) -> Result<Point<EF>, SumcheckError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let mut randomness = Vec::with_capacity(self.polynomial_evaluations.len());

        // Grinding pushes one witness per round;
        //
        // Reject upfront if the proof is short so the loop below cannot panic on out-of-bounds indexing.
        if pow_bits > 0 && self.pow_witnesses.len() != self.polynomial_evaluations.len() {
            return Err(SumcheckError::PowWitnessCountMismatch {
                expected: self.polynomial_evaluations.len(),
                actual: self.pow_witnesses.len(),
            });
        }

        for (i, &[c0, c_inf]) in self.polynomial_evaluations.iter().enumerate() {
            // Observe only the sent polynomial evaluations (h(0) and h(inf)).
            challenger.observe_algebra_slice(&[c0, c_inf]);

            // Verify PoW (only if pow_bits > 0)
            if pow_bits > 0 && !challenger.check_witness(pow_bits, self.pow_witnesses[i]) {
                return Err(SumcheckError::InvalidPowWitness);
            }

            // Sample challenge and reconstruct h(r) from (h(0), h(1), h(inf)).
            let r: EF = challenger.sample_algebra_element();
            *claimed_sum = extrapolate_01inf(c0, *claimed_sum - c0, c_inf, r);
            randomness.push(r);
        }

        Ok(Point::new(randomness))
    }
}

/// Verify the final sumcheck rounds.
///
/// This is a free function because the caller may not have a `SumcheckData` at all when `rounds == 0`.
///
/// # Returns
///
/// A `Point` of folding randomness values.
pub fn verify_final_sumcheck_rounds<F, EF, Challenger>(
    final_sumcheck: Option<&SumcheckData<F, EF>>,
    challenger: &mut Challenger,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
) -> Result<Point<EF>, SumcheckError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if rounds == 0 {
        return Ok(Point::new(Vec::new()));
    }

    let sumcheck = final_sumcheck.ok_or(SumcheckError::MissingSumcheckData {
        expected_rounds: rounds,
    })?;

    if sumcheck.polynomial_evaluations.len() != rounds {
        return Err(SumcheckError::RoundCountMismatch {
            expected: rounds,
            actual: sumcheck.polynomial_evaluations.len(),
        });
    }
    sumcheck.verify_rounds(challenger, claimed_sum, pow_bits)
}
