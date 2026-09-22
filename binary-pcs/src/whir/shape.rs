//! What a derived schedule fixes about the proofs it can produce.

use p3_binary_field::TowerLevel;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_sumcheck::ring_switch::bits::BitTensor;
use p3_whir::WhirConfig;
use p3_whir::transcript::query_draws;

/// Everything a schedule fixes about the size of its proofs and the work of checking one.
///
/// Each figure is read off the derived schedule and the claim count, never off a witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofShape {
    /// Codeword positions opened across every round, one Merkle check each.
    pub stir_queries: usize,

    /// Alphabet elements opened from the first codeword.
    pub opened_base_elements: usize,

    /// Challenge-field elements opened from the folded codewords.
    pub opened_extension_elements: usize,

    /// Digests the proof carries, which are the folded codewords' roots and their opened paths.
    ///
    /// No prefix a path shares with another is discounted.
    pub merkle_digests: usize,

    /// Alphabet elements the proof carries outside the openings, which are its grinding witnesses.
    pub sent_base_elements: usize,

    /// Challenge-field elements the proof carries outside the openings.
    pub sent_extension_elements: usize,

    /// Hardest grinding any phase of the schedule demands.
    pub grinding_bits: usize,
}

impl ProofShape {
    /// Read the shape of every proximity opening a schedule can produce.
    ///
    /// The claim count fixes how many opened values travel beside the transcript.
    #[must_use]
    pub fn of<EF, F, Challenger>(
        config: &WhirConfig<EF, F, Challenger>,
        opened_values: usize,
    ) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let num_rounds = config.round_parameters().len();
        let mut shape = Self {
            stir_queries: 0,
            opened_base_elements: 0,
            opened_extension_elements: 0,
            // One root per folded codeword, before any authentication node is counted.
            merkle_digests: num_rounds,
            // One witness per folding round, one per folded codeword, and one closing the run.
            sent_base_elements: config.num_variables() + num_rounds + 1,
            // Two coefficients bound one degree-two round, and one value closes the run.
            sent_extension_elements: 2 * config.num_variables()
                + (1 << config.final_sumcheck_rounds())
                + config.commitment_ood_samples()
                + opened_values,
            grinding_bits: config.max_pow_bits(),
        };

        let rounds = config
            .round_parameters()
            .iter()
            .map(|round| {
                (
                    round.log_folded_domain_size,
                    round.folding_factor,
                    round.num_queries,
                    round.ood_samples,
                )
            })
            .chain(core::iter::once({
                let final_round = config.final_round_config();
                (
                    final_round.log_folded_domain_size,
                    final_round.folding_factor,
                    config.terminal().num_queries,
                    0,
                )
            }));

        for (index, (log_folded, folding, queries, ood)) in rounds.enumerate() {
            // A round drawing nothing opens the whole folded domain instead.
            let positions = if query_draws(1usize << log_folded, queries) == 0 {
                1usize << log_folded
            } else {
                queries
            };
            let elements = positions << folding;

            shape.stir_queries += positions;
            if index == 0 {
                shape.opened_base_elements += elements;
            } else {
                shape.opened_extension_elements += elements;
            }
            shape.merkle_digests += positions * log_folded;
            shape.sent_extension_elements += ood;
        }

        shape
    }

    /// Read the shape of every bit-witness opening a schedule can produce for this many claims.
    ///
    /// One reduction per claim travels beside the proximity opening.
    ///
    /// Each sends its element by rows, its own degree-two rounds, and the value it survives with.
    #[must_use]
    pub fn of_bit_readings<EF, Challenger>(
        config: &WhirConfig<EF, EF, Challenger>,
        num_claims: usize,
        successor_tensors: bool,
    ) -> Self
    where
        EF: Field + TowerLevel,
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
    {
        let mut shape = Self::of(config, num_claims);
        // The element alone, or the element with carry and last.
        let num_tensors = if successor_tensors { 3 } else { 1 };
        let rows = num_tensors * BitTensor::<EF>::DIMENSION;
        // A Boolean prefix only removes rounds, so the packing's arity bounds them.
        shape.sent_extension_elements += num_claims * (rows + 2 * config.num_variables() + 1);
        shape
    }

    /// Bytes a proof of this shape cannot exceed, given the width of each thing it carries.
    ///
    /// A serializer that pads a value beyond its own width invalidates the figure.
    ///
    /// Framing the serializer adds around a list is the caller's to allow for.
    #[must_use]
    pub const fn max_bytes(
        &self,
        base_element_bytes: usize,
        extension_element_bytes: usize,
        digest_bytes: usize,
    ) -> usize {
        (self.opened_base_elements + self.sent_base_elements) * base_element_bytes
            + (self.opened_extension_elements + self.sent_extension_elements)
                * extension_element_bytes
            + self.merkle_digests * digest_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::ProofShape;

    // A hand-built shape whose byte total is computable without the implementation.
    const SHAPE: ProofShape = ProofShape {
        stir_queries: 40,
        opened_base_elements: 100,
        opened_extension_elements: 200,
        merkle_digests: 300,
        sent_base_elements: 10,
        sent_extension_elements: 50,
        grinding_bits: 16,
    };

    #[test]
    fn the_estimate_prices_each_part_at_its_own_width() {
        // (100 + 10)*4 + (200 + 50)*16 + 300*32 = 440 + 4000 + 9600.
        assert_eq!(SHAPE.max_bytes(4, 16, 32), 14_040);
    }

    #[test]
    fn a_wider_alphabet_raises_the_estimate_by_exactly_the_alphabet_run() {
        // Only the first codeword and the grinding witnesses use the alphabet's width.
        assert_eq!(
            SHAPE.max_bytes(16, 16, 32) - SHAPE.max_bytes(4, 16, 32),
            110 * (16 - 4)
        );
    }
}
